"""Opik integration for finished calls.

This is the only module that imports the Opik tracing SDK. Its whole public
surface is:

    observer = OpikCallObserver.from_settings(settings)   # None if Opik is off
    result = observer.log_call(call_record, analysis)     # -> TelemetryResult

It depends on nothing in the voice pipeline except the plain CallRecord and
PostCallAnalysis models, so plugging it into a different LiveKit project means
building those two objects and calling `log_call`.

Each call becomes one Opik trace:

    outbound_call                       trace: call variables in, outcome/transcript out,
    │                                   recording attached, tags for rule targeting
    ├── call_setup                      dial result, SIP status
    ├── conversation                    turn statistics
    │   ├── tool:verify_identity        type=tool, real arguments and results
    │   ├── tool:check_availability
    │   └── tool:book_appointment
    ├── recording                       audio reference (or why there is none)
    └── post_call_analysis              deterministic outcome + optional LLM summary

The trace is written once, after the call, with the real start/end times of
every step. Delivery is asynchronous inside the SDK; `log_call` then flushes
with a timeout and reports whether everything was actually delivered. Opik
failures are caught and returned, never raised into the call pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from app.call_state import utc_now
from app.config import Settings
from app.schemas import CallRecord, PostCallAnalysis

logger = logging.getLogger("healthcare_agent.opik")

TRACE_NAME = "outbound_call"
# Online evaluation rules filter on this tag so they only score completed call traces.
CALL_TRACE_TAG = "healthcare-outbound-call"
SENT_TO_OPIK_TAG = "post-call-analyzed"


class TelemetryResult(BaseModel):
    # "pending" is only ever stored on disk between analysis and the Opik upload.
    status: Literal["pending", "sent", "failed", "disabled"]
    trace_id: str | None = None
    project_name: str | None = None
    project_url: str | None = None
    error: str | None = None
    attempted_at: datetime


def build_trace_input(record: CallRecord) -> dict[str, Any]:
    return {
        "call_id": record.call_id,
        "mode": record.mode.value,
        "patient": record.patient,
        "agent": record.agent.model_dump(),
    }


def build_trace_output(record: CallRecord, analysis: PostCallAnalysis) -> dict[str, Any]:
    """The trace output. Online evaluation rules read `booking` and `transcript` from here."""
    return {
        "outcome": analysis.outcome.value,
        "outcome_reason": analysis.outcome_reason,
        "booking": {
            "status": analysis.booking_status.value,
            "appointment_id": analysis.appointment_id,
            "appointment_start": analysis.appointment_start.isoformat()
            if analysis.appointment_start
            else None,
            "timezone": analysis.appointment_timezone,
        },
        "transcript": [
            {
                "speaker": turn.speaker,
                "text": turn.text,
                "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                "interrupted": turn.interrupted,
            }
            for turn in record.transcript
        ],
        "recording": record.recording.model_dump(mode="json"),
        "post_call_analysis": analysis.model_dump(mode="json"),
    }


def build_trace_metadata(record: CallRecord, analysis: PostCallAnalysis) -> dict[str, Any]:
    return {
        "call_id": record.call_id,
        "patient_id": record.patient.get("patient_id"),
        "mode": record.mode.value,
        "room_name": record.room_name,
        "dial_status": record.dial_status.value,
        "sip_status_code": record.sip_status_code,
        "termination_reason": record.termination_reason.value,
        "termination_detail": record.termination_detail,
        "identity_status": record.identity_status.value,
        "duration_seconds": round(record.duration_seconds, 1),
        "outcome": analysis.outcome.value,
        "booking_status": analysis.booking_status.value,
        "appointment_id": analysis.appointment_id,
        "recording_status": record.recording.status,
        "recording_uri": record.recording.uri,
        "agent_name": record.agent.agent_name,
        "agent_version": record.agent.agent_version,
        "prompt_version": record.agent.prompt_version,
        "llm_model": record.agent.llm_model,
        "stt_model": record.agent.stt_model,
        "tts_model": record.agent.tts_model,
        "analyzer_version": analysis.analyzer_version,
    }


def build_tags(record: CallRecord, analysis: PostCallAnalysis) -> list[str]:
    return [
        CALL_TRACE_TAG,
        SENT_TO_OPIK_TAG,
        f"outcome:{analysis.outcome.value}",
        f"mode:{record.mode.value}",
    ]


class OpikCallObserver:
    def __init__(
        self,
        client: Any,
        *,
        project_name: str,
        flush_timeout_seconds: int = 30,
        analysis_model: str | None = None,
        init_error: str | None = None,
    ) -> None:
        self._client = client
        self._project_name = project_name
        self._flush_timeout = flush_timeout_seconds
        self._analysis_model = analysis_model
        self._init_error = init_error

    @classmethod
    def from_settings(cls, settings: Settings) -> OpikCallObserver | None:
        """Create an observer, or return None when Opik is not configured.

        If Opik is configured but the client cannot be created, an observer is
        still returned so every call reports a "failed" telemetry status instead
        of silently looking like Opik was switched off.
        """
        if not settings.opik_configured:
            return None
        common = {
            "project_name": settings.opik_project_name,
            "flush_timeout_seconds": settings.opik_flush_timeout_seconds,
            "analysis_model": settings.analysis_model if settings.post_call_llm_enabled else None,
        }
        try:
            import opik

            client = opik.Opik(
                project_name=settings.opik_project_name,
                workspace=settings.opik_workspace,
                api_key=settings.opik_api_key.get_secret_value() if settings.opik_api_key else None,
                host=settings.opik_url_override,
                _show_misconfiguration_message=False,
            )
        except Exception as exc:
            logger.error("could not create Opik client: %s", type(exc).__name__)
            return cls(None, init_error=f"{type(exc).__name__}: {exc}"[:500], **common)
        return cls(client, **common)

    def log_call(self, record: CallRecord, analysis: PostCallAnalysis) -> TelemetryResult:
        attempted_at = utc_now()
        if self._client is None:
            return TelemetryResult(
                status="failed",
                project_name=self._project_name,
                error=f"Opik client unavailable: {self._init_error}",
                attempted_at=attempted_at,
            )
        try:
            trace_id = self._write_trace(record, analysis)
        except Exception as exc:
            logger.error(
                "failed to build Opik trace: %s",
                type(exc).__name__,
                extra={"call_id": record.call_id},
            )
            return TelemetryResult(
                status="failed",
                project_name=self._project_name,
                error=f"{type(exc).__name__}: {exc}"[:500],
                attempted_at=attempted_at,
            )

        delivered = self._client.flush(timeout=self._flush_timeout)
        if not delivered:
            detail = getattr(self._client, "last_flush_result", None)
            logger.error("Opik flush incomplete", extra={"call_id": record.call_id})
            return TelemetryResult(
                status="failed",
                trace_id=trace_id,
                project_name=self._project_name,
                # Covers both a timeout and data the server rejected (e.g. HTTP 401),
                # which the SDK reports as dropped messages in the flush result.
                error=f"Opik flush incomplete (timeout {self._flush_timeout}s or messages "
                f"dropped by the server): {detail}"[:500],
                attempted_at=attempted_at,
            )
        return TelemetryResult(
            status="sent",
            trace_id=trace_id,
            project_name=self._project_name,
            project_url=self._project_url(),
            attempted_at=attempted_at,
        )

    # -- internals -------------------------------------------------------------

    def _project_url(self) -> str | None:
        try:
            return self._client.get_project_url(project_name=self._project_name)
        except Exception:
            return None

    def _write_trace(self, record: CallRecord, analysis: PostCallAnalysis) -> str:
        import opik

        attachments = []
        if record.recording.status == "available" and record.recording.uri:
            path = Path(record.recording.uri)
            if path.is_file():
                attachments.append(
                    opik.Attachment(
                        data=str(path),
                        file_name=f"{record.call_id}{path.suffix}",
                        content_type=record.recording.content_type or "audio/ogg",
                    )
                )

        trace = self._client.trace(
            name=TRACE_NAME,
            start_time=record.started_at,
            end_time=record.ended_at,
            input=build_trace_input(record),
            output=build_trace_output(record, analysis),
            metadata=build_trace_metadata(record, analysis),
            tags=build_tags(record, analysis),
            attachments=attachments or None,
            project_name=self._project_name,
        )

        answered_or_end = record.answered_at or record.ended_at
        trace.span(
            name="call_setup",
            type="general",
            start_time=record.started_at,
            end_time=answered_or_end,
            input={"mode": record.mode.value, "room_name": record.room_name},
            output={
                "dial_status": record.dial_status.value,
                "sip_status_code": record.sip_status_code,
                "sip_status": record.sip_status,
            },
        )

        conversation = trace.span(
            name="conversation",
            type="general",
            start_time=answered_or_end,
            end_time=record.ended_at,
            input={"identity_status_at_end": record.identity_status.value},
            output={
                "turns": len(record.transcript),
                "agent_turns": sum(t.speaker == "agent" for t in record.transcript),
                "patient_turns": sum(t.speaker == "patient" for t in record.transcript),
                "interrupted_turns": sum(t.interrupted for t in record.transcript),
                "termination_reason": record.termination_reason.value,
            },
        )
        for event in record.tool_events:
            conversation.span(
                name=f"tool:{event.tool_name}",
                type="tool",
                start_time=event.started_at,
                end_time=event.ended_at,
                input=event.arguments,
                output=event.result,
                metadata={"ok": event.ok, "call_id": record.call_id},
                error_info=(
                    {"exception_type": "ToolFailure", "message": str(event.error), "traceback": ""}
                    if not event.ok
                    else None
                ),
            )

        trace.span(
            name="recording",
            type="general",
            start_time=record.recording.started_at or answered_or_end,
            end_time=record.ended_at,
            output=record.recording.model_dump(mode="json"),
        )

        analysis_span: dict[str, Any] = {
            "name": "post_call_analysis",
            "type": "general",
            "start_time": record.ended_at,
            "end_time": analysis.analyzed_at,
            "input": {
                "booking_backend_appointment_id": record.appointment.appointment_id
                if record.appointment
                else None,
                "tool_event_count": len(record.tool_events),
                "transcript_turns": len(record.transcript),
                "dial_status": record.dial_status.value,
            },
            "output": analysis.model_dump(mode="json"),
            "metadata": {"qualitative_source": analysis.qualitative_source},
        }
        if analysis.qualitative_source == "llm" and self._analysis_model:
            analysis_span.update(type="llm", model=self._analysis_model, provider="openai")
        trace.span(**analysis_span)

        return trace.id
