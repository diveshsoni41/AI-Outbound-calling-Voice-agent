"""LiveKit Agents worker for the outbound healthcare call.

Run it with:

    python -m app.agent dev      # development: hot reload, verbose logs
    python -m app.agent start    # production mode

The worker registers under AGENT_NAME and only runs when explicitly dispatched
(see app/call_manager.py). For each dispatched job it:

1. reads the call ID and patient from the job metadata,
2. dials the patient through the outbound SIP trunk and waits for an answer,
3. runs the voice conversation (OpenAI STT -> LLM -> TTS, Silero VAD),
4. when the session ends, builds the CallRecord and runs the post-call
   pipeline (analysis, local report, Opik trace).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable

from google.protobuf.duration_pb2 import Duration
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    CloseEvent,
    CloseReason,
    ErrorEvent,
    JobContext,
    JobProcess,
    RunContext,
    StopResponse,
    cli,
    function_tool,
    room_io,
)
from livekit.plugins import openai, silero

from app.call_manager import CallJob, UnauthorizedDestinationError, ensure_authorized_destination
from app.call_state import CallState
from app.config import ConfigurationError, Requirement, get_settings
from app.pipeline import finalize_call
from app.prompts import build_instructions, greeting_instruction
from app.recording import store_recording
from app.runtime import agent_info, build_analyzer, build_observer, build_scheduler, build_store
from app.scheduling import PartOfDay
from app.schemas import CallMode, TerminationReason
from app.tools import CallTools, DecisionValue, IdentityResponse
from app.transcript import transcript_from_chat_context

logger = logging.getLogger("healthcare_agent")

# How long to wait for the callee to say "hello" before the agent opens the call.
GREETING_FALLBACK_SECONDS = 3.0

TTS_STYLE = "Speak calmly and warmly at a moderate pace, like a friendly clinic assistant."


class HealthcareOutreachAgent(Agent):
    """The conversational agent. Tool methods forward to CallTools, which owns
    the rules and records what really happened."""

    def __init__(
        self,
        *,
        call_tools: CallTools,
        instructions: str,
        hangup: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(instructions=instructions)
        self._call_tools = call_tools
        self._hangup = hangup

    @function_tool
    async def verify_identity(self, context: RunContext, response: IdentityResponse) -> str:
        """Record who answered the phone. Call this as soon as the person answers whether
        they are the patient. Health results are only returned when the response is
        "confirmed".

        Args:
            response: "confirmed" if they said they are the patient; "wrong_person" if someone
                else answered on the patient's phone or household; "wrong_number" if the patient
                cannot be reached at this number; "not_available" if the patient cannot come to
                the phone right now.
        """
        return self._call_tools.verify_identity(response)

    @function_tool
    async def check_availability(self, context: RunContext, part_of_day: PartOfDay = "any") -> str:
        """Look up open doctor consultation slots. Only call after identity is verified and
        the patient wants to book.

        Args:
            part_of_day: "morning", "afternoon", or "any" if the patient has no preference.
        """
        return self._call_tools.check_availability(part_of_day)

    @function_tool
    async def book_appointment(
        self, context: RunContext, slot_id: str, patient_confirmed: bool
    ) -> str:
        """Book one of the slots returned by check_availability. The appointment exists only
        if this returns status "booked".

        Args:
            slot_id: The slot_id exactly as returned by check_availability.
            patient_confirmed: True only if the patient explicitly said yes to this exact time
                after you read it back to them.
        """
        return self._call_tools.book_appointment(slot_id, patient_confirmed)

    @function_tool
    async def record_patient_decision(
        self,
        context: RunContext,
        decision: DecisionValue,
        callback_preference: str | None = None,
    ) -> str:
        """Record that the patient declined, asked for a call back, or asked not to be called.

        Args:
            decision: "declined", "callback_requested" or "do_not_call".
            callback_preference: For call backs, when the patient would like to be called, in
                their words (for example "Thursday after 5pm"). Omit otherwise.
        """
        return self._call_tools.record_patient_decision(decision, callback_preference)

    @function_tool
    async def end_call(self, context: RunContext) -> None:
        """Hang up the phone. Only call this after you have said goodbye."""
        # Let the goodbye finish playing before the line drops.
        await context.wait_for_playout()
        self._call_tools.state.mark_ended(TerminationReason.AGENT_ENDED_CALL)
        await self._hangup()
        raise StopResponse()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Call state for jobs running in this process, keyed by job ID, so the
# on_session_end hook can find the state its entrypoint created.
_ACTIVE_CALLS: dict[str, CallState] = {}

_DISCONNECT_REASONS = {
    rtc.DisconnectReason.CLIENT_INITIATED: TerminationReason.PATIENT_HUNG_UP,
    rtc.DisconnectReason.USER_REJECTED: TerminationReason.PATIENT_HUNG_UP,
    rtc.DisconnectReason.USER_UNAVAILABLE: TerminationReason.PATIENT_HUNG_UP,
    rtc.DisconnectReason.ROOM_DELETED: TerminationReason.AGENT_ENDED_CALL,
}


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def on_session_end(ctx: JobContext) -> None:
    """Runs after the AgentSession has closed (so the recording file is final)."""
    state = _ACTIVE_CALLS.pop(ctx.job.id, None)
    if state is None:
        return
    settings = get_settings()
    store = build_store(settings)

    transcript, audio_path, audio_started_at = [], None, None
    try:
        report = ctx.make_session_report()
        transcript = transcript_from_chat_context(report.chat_history)
        audio_path, audio_started_at = (
            report.audio_recording_path,
            report.audio_recording_started_at,
        )
    except RuntimeError as exc:
        # Either no AgentSession was started (the call was never answered), or the
        # recorder did not stop cleanly. In the second case keep the transcript.
        try:
            transcript = transcript_from_chat_context(ctx.primary_session.history)
            state.record_error(f"session report unavailable: {exc}")
        except RuntimeError:
            pass

    state.mark_ended(TerminationReason.JOB_SHUTDOWN)  # no-op if a reason was already recorded
    recording = store_recording(
        audio_path, store.call_dir(state.call_id), started_at=audio_started_at
    )
    record = state.to_record(transcript, recording)
    await finalize_call(
        record,
        analyzer=build_analyzer(settings),
        observer=build_observer(settings),
        store=store,
    )


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session(agent_name=get_settings().agent_name, on_session_end=on_session_end)
async def outbound_call(ctx: JobContext) -> None:
    settings = get_settings()
    settings.require(Requirement.LIVEKIT, Requirement.TELEPHONY, Requirement.OPENAI)

    job = CallJob.model_validate_json(ctx.job.metadata)
    ctx.log_context_fields = {"call_id": job.call_id}
    state = CallState(
        call_id=job.call_id,
        patient=job.patient,
        mode=CallMode.TELEPHONY,
        agent_info=agent_info(settings),
        room_name=ctx.room.name,
    )
    _ACTIVE_CALLS[ctx.job.id] = state

    await ctx.connect()

    sip_identity = f"patient-{job.call_id}"
    if not await _dial(ctx, state, sip_identity):
        ctx.shutdown(reason="call not connected")
        return

    participant = await ctx.wait_for_participant(identity=sip_identity)
    _watch_disconnect(ctx, state, participant.identity)

    session = AgentSession(
        stt=openai.STT(model=settings.stt_model),
        llm=openai.LLM(model=settings.llm_model),
        tts=openai.TTS(model=settings.tts_model, voice=settings.tts_voice, instructions=TTS_STYLE),
        vad=ctx.proc.userdata["vad"],
    )
    _watch_session(ctx, session, state)

    async def hangup() -> None:
        await ctx.delete_room()

    agent = HealthcareOutreachAgent(
        call_tools=CallTools(state, build_scheduler(settings)),
        instructions=build_instructions(
            patient_first_name=job.patient.first_name,
            organization_name=settings.organization_name,
            emergency_number=settings.emergency_number,
        ),
        hangup=hangup,
    )
    callee_spoke, stop_watching = _watch_for_callee_speech(session)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(participant_identity=sip_identity),
        # Audio only: the session records both sides to the job's session directory.
        record={
            "audio": settings.record_audio,
            "traces": False,
            "logs": False,
            "transcript": False,
        },
    )
    await _greet_if_silent(
        session, callee_spoke, stop_watching, greeting_instruction(settings.organization_name)
    )


async def _dial(ctx: JobContext, state: CallState, sip_identity: str) -> bool:
    settings = get_settings()
    try:
        # Checked again here, not only in the call manager, so a crafted dispatch
        # cannot make the worker dial an unapproved number.
        ensure_authorized_destination(settings, state.patient.phone_number)
    except UnauthorizedDestinationError as exc:
        state.record_error(str(exc))
        state.mark_dial_failed(None, "destination not authorized")
        return False

    logger.info("dialing patient", extra={"call_id": state.call_id})
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=settings.sip_outbound_trunk_id,
                sip_call_to=state.patient.phone_number,
                participant_identity=sip_identity,
                participant_name="Patient",
                wait_until_answered=True,
                ringing_timeout=Duration(seconds=settings.ring_timeout_seconds),
                max_call_duration=Duration(seconds=settings.max_call_duration_seconds),
            )
        )
    except api.SipCallError as exc:
        logger.warning(
            "call not connected",
            extra={"call_id": state.call_id, "sip_status_code": exc.sip_status_code},
        )
        state.mark_dial_failed(exc.sip_status_code, exc.sip_status)
        return False
    except api.TwirpError as exc:
        # Invalid trunk, permissions, malformed number and similar API-level errors.
        state.record_error(f"create_sip_participant failed: {exc.code}")
        state.mark_dial_failed(None, f"{exc.code}: {exc.message}")
        return False

    state.mark_answered()
    logger.info("call answered", extra={"call_id": state.call_id})
    return True


def _watch_disconnect(ctx: JobContext, state: CallState, identity: str) -> None:
    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        if participant.identity != identity:
            return
        reason = _DISCONNECT_REASONS.get(
            participant.disconnect_reason, TerminationReason.CONNECTION_LOST
        )
        detail = rtc.DisconnectReason.Name(participant.disconnect_reason)
        state.mark_ended(reason, detail)
        logger.info("patient left the call", extra={"call_id": state.call_id, "reason": detail})


def _watch_session(ctx: JobContext, session: AgentSession, state: CallState) -> None:
    @session.on("error")
    def _on_error(event: ErrorEvent) -> None:
        recoverable = getattr(event.error, "recoverable", True)
        state.record_error(f"{type(event.error).__name__} from {type(event.source).__name__}")
        if not recoverable:
            state.mark_ended(TerminationReason.AGENT_ERROR, type(event.error).__name__)

    @session.on("close")
    def _on_close(event: CloseEvent) -> None:
        if event.reason == CloseReason.ERROR:
            state.mark_ended(
                TerminationReason.AGENT_ERROR, str(event.error)[:200] if event.error else None
            )
        ctx.shutdown(reason=f"session closed: {event.reason}")


def _watch_for_callee_speech(session: AgentSession) -> tuple[asyncio.Event, Callable[[], None]]:
    """Set an event as soon as the callee starts speaking.

    Registered before session.start() so a "Hello?" during startup is not missed.
    VAD speech start fires immediately; the transcript event is a backstop. With a
    non-streaming STT the transcript only arrives after the utterance has been
    transcribed, which alone could exceed the greeting fallback window.
    """
    heard = asyncio.Event()

    def _on_user_state(event: object) -> None:
        if getattr(event, "new_state", None) == "speaking":
            heard.set()

    def _on_transcribed(_event: object) -> None:
        heard.set()

    session.on("user_state_changed", _on_user_state)
    session.on("user_input_transcribed", _on_transcribed)

    def stop() -> None:
        session.off("user_state_changed", _on_user_state)
        session.off("user_input_transcribed", _on_transcribed)

    return heard, stop


async def _greet_if_silent(
    session: AgentSession, heard: asyncio.Event, stop: Callable[[], None], instruction: str
) -> None:
    """On outbound calls the callee usually speaks first ("Hello?"), and the agent
    answers that turn. If they stay silent, the agent opens the call instead."""
    try:
        await asyncio.wait_for(heard.wait(), timeout=GREETING_FALLBACK_SECONDS)
    except asyncio.TimeoutError:
        session.generate_reply(instructions=instruction)
    finally:
        stop()


if __name__ == "__main__":
    # Fail fast with a clear message instead of a worker that cannot register.
    if len(sys.argv) > 1 and sys.argv[1] in ("dev", "start", "connect"):
        try:
            get_settings().require(Requirement.LIVEKIT, Requirement.TELEPHONY, Requirement.OPENAI)
        except ConfigurationError as exc:
            print(f"Cannot start the agent worker: {exc}", file=sys.stderr)
            sys.exit(2)
    cli.run_app(server)
