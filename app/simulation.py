"""Text-mode call simulation. This is NOT a phone call.

The simulation runs the real HealthcareOutreachAgent, its real tools and the
real scheduling backend inside a LiveKit AgentSession, but with text in place
of audio: there is no SIP, STT, TTS or recording. The patient's side is a fixed
list of lines, or a `PatientResponder` that writes each line from the
transcript so far (scripts/verify_openai_conversation.py uses a second model
playing the patient).

The agent's side can come from:
* `ScriptedLLM`: a deterministic stand-in for the language model that emits
  pre-written replies and tool calls. Tool calls still execute for real, and
  scripted replies can read the real tool results. Used by tests and the
  offline demo.
* the real OpenAI model (`--llm openai` in scripts/run_demo.py), in which
  case the model decides what to say and which tools to call.

Every record produced here has mode="simulation" and that tag is carried into
the Opik trace, so simulated calls can never be mistaken for real ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import AgentSession, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from app.agent import HealthcareOutreachAgent
from app.call_state import CallState, new_call_id, utc_now
from app.config import SchedulerSimulation, Settings
from app.observability.opik_integration import OpikCallObserver
from app.patient import Patient
from app.pipeline import CallReport, CallReportStore, finalize_call
from app.post_call import PostCallAnalyzer
from app.prompts import build_instructions
from app.runtime import agent_info
from app.scheduling import SchedulingService
from app.schemas import (
    CallMode,
    DialStatus,
    RecordingReference,
    TerminationReason,
    TranscriptTurn,
)
from app.tools import CallTools
from app.transcript import transcript_from_chat_context

logger = logging.getLogger("healthcare_agent.simulation")

SCRIPTED_MODEL_NAME = "scripted-llm (simulation)"


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Say:
    text: str


@dataclass(frozen=True)
class CallTool:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SayAndCall:
    """Speak and call a tool in the same response (e.g. goodbye + end_call)."""

    text: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Silence:
    """Produce no reply, e.g. because the line dropped before the agent spoke."""


Action = Say | CallTool | SayAndCall | Silence
Step = Action | Callable[[llm.ChatContext], Action]
# Produces the patient's next line from the transcript so far; None hangs up.
PatientResponder = Callable[[list[TranscriptTurn]], Awaitable[str | None]]


def tool_result(chat_ctx: llm.ChatContext, tool_name: str) -> dict[str, Any]:
    """The most recent result of `tool_name` in the conversation, parsed from JSON."""
    for item in reversed(chat_ctx.items):
        if item.type == "function_call_output" and item.name == tool_name:
            return json.loads(item.output)
    raise LookupError(f"no result for tool {tool_name!r} in the conversation yet")


class _ScriptedStream(llm.LLMStream):
    def __init__(self, owner: ScriptedLLM, action: Action, **kwargs: Any) -> None:
        super().__init__(owner, **kwargs)
        self._action = action

    async def _run(self) -> None:
        action = self._action
        text = getattr(action, "text", None)
        tool_calls = []
        if isinstance(action, CallTool | SayAndCall):
            tool_calls.append(
                llm.FunctionToolCall(
                    name=action.name,
                    arguments=json.dumps(action.arguments),
                    call_id=f"call_{uuid.uuid4().hex[:12]}",
                )
            )
        if text or tool_calls:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=uuid.uuid4().hex,
                    delta=llm.ChoiceDelta(
                        role="assistant", content=text or None, tool_calls=tool_calls
                    ),
                )
            )


class ScriptedLLM(llm.LLM):
    """Returns one scripted step per model request, in order."""

    def __init__(self, steps: list[Step]) -> None:
        super().__init__()
        self._steps = list(steps)
        self.requests = 0

    @property
    def model(self) -> str:
        return SCRIPTED_MODEL_NAME

    @property
    def provider(self) -> str:
        return "simulation"

    @property
    def remaining_steps(self) -> int:
        return len(self._steps)

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_: Any,
    ) -> llm.LLMStream:
        self.requests += 1
        if self._steps:
            step = self._steps.pop(0)
            action = step(chat_ctx) if callable(step) else step
        else:
            logger.warning("scripted LLM ran out of steps; replying with silence")
            action = Silence()
        return _ScriptedStream(
            self, action, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


# ---------------------------------------------------------------------------
# Running a simulated call
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    patient_lines: list[str]
    agent_script: list[Step]
    scheduler_simulation: SchedulerSimulation = SchedulerSimulation.NONE
    # Simulates the patient's line dropping after their last line.
    patient_hangs_up: bool = False
    expected_outcome: str | None = None


async def run_simulated_call(
    scenario: Scenario,
    *,
    patient: Patient,
    settings: Settings,
    scheduler: SchedulingService,
    analyzer: PostCallAnalyzer,
    observer: OpikCallObserver | None,
    store: CallReportStore,
    model: llm.LLM | None = None,
    call_id: str | None = None,
    patient_responder: PatientResponder | None = None,
    max_patient_turns: int = 14,
) -> CallReport:
    """Run one simulated call. With `patient_responder`, the patient's lines are
    produced turn by turn from the transcript so far (returning None ends the
    call) instead of being taken from `scenario.patient_lines`."""
    model = model or ScriptedLLM(scenario.agent_script)
    state = CallState(
        call_id=call_id or new_call_id(),
        patient=patient,
        mode=CallMode.SIMULATION,
        agent_info=agent_info(settings, llm_model=model.model, voice=False),
    )
    state.dial_status = DialStatus.SIMULATED
    state.answered_at = utc_now()

    hung_up = asyncio.Event()

    async def hangup() -> None:
        hung_up.set()

    agent = HealthcareOutreachAgent(
        call_tools=CallTools(state, scheduler),
        instructions=build_instructions(
            patient_first_name=patient.first_name,
            organization_name=settings.organization_name,
            emergency_number=settings.emergency_number,
        ),
        hangup=hangup,
    )

    async with AgentSession(llm=model) as session:
        await session.start(agent)
        if patient_responder is None:
            for line in scenario.patient_lines:
                if hung_up.is_set():
                    break
                await session.run(user_input=line)
        else:
            for _ in range(max_patient_turns):
                if hung_up.is_set():
                    break
                line = await patient_responder(transcript_from_chat_context(session.history))
                if line is None:
                    break
                await session.run(user_input=line)
        transcript = transcript_from_chat_context(session.history)

    if scenario.patient_hangs_up:
        state.mark_ended(TerminationReason.PATIENT_HUNG_UP, "simulated disconnect")
    state.mark_ended(TerminationReason.SIMULATION_COMPLETED)

    record = state.to_record(
        transcript,
        RecordingReference(status="not_available", reason="Simulation mode has no audio."),
    )
    report = await finalize_call(record, analyzer=analyzer, observer=observer, store=store)
    if report is None:  # only possible if another process is finalizing this call ID
        raise RuntimeError(f"call {record.call_id} is already being finalized")
    return report
