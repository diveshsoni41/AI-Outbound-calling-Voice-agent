"""Authoritative per-call state.

The LLM's conversation history says what was *said*; this object records what
actually *happened* (identity outcome, slots offered, backend booking result,
how the call ended). Post-call analysis trusts this over the transcript.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.patient import Patient
from app.scheduling import Appointment, SlotView
from app.schemas import (
    AgentInfo,
    CallMode,
    CallRecord,
    DialStatus,
    IdentityStatus,
    PatientDecision,
    RecordingReference,
    TerminationReason,
    ToolEvent,
    TranscriptTurn,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


# SIP response codes from create_sip_participant failures, mapped to dial outcomes.
_SIP_STATUS_MAP = {
    408: DialStatus.NO_ANSWER,  # request timeout
    480: DialStatus.NO_ANSWER,  # temporarily unavailable
    487: DialStatus.NO_ANSWER,  # request terminated (ringing timeout)
    486: DialStatus.BUSY,
    600: DialStatus.BUSY,
    603: DialStatus.REJECTED,
}


def dial_status_for_sip_code(code: int | None) -> DialStatus:
    if code is None:
        return DialStatus.FAILED
    return _SIP_STATUS_MAP.get(code, DialStatus.FAILED)


@dataclass
class CallState:
    call_id: str
    patient: Patient
    mode: CallMode
    agent_info: AgentInfo
    room_name: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    answered_at: datetime | None = None
    ended_at: datetime | None = None
    dial_status: DialStatus = DialStatus.NOT_STARTED
    sip_status_code: int | None = None
    sip_status: str | None = None
    termination_reason: TerminationReason | None = None
    termination_detail: str | None = None
    identity_status: IdentityStatus = IdentityStatus.UNKNOWN
    health_info_disclosed: bool = False
    patient_decision: PatientDecision | None = None
    callback_preference: str | None = None
    offered_slots: dict[str, SlotView] = field(default_factory=dict)
    appointment: Appointment | None = None
    tool_events: list[ToolEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def identity_verified(self) -> bool:
        return self.identity_status is IdentityStatus.VERIFIED

    def mark_answered(self) -> None:
        self.dial_status = DialStatus.ANSWERED
        self.answered_at = utc_now()

    def mark_dial_failed(self, sip_code: int | None, sip_status: str | None) -> None:
        self.dial_status = dial_status_for_sip_code(sip_code)
        self.sip_status_code = sip_code
        self.sip_status = sip_status
        self.mark_ended(TerminationReason.DIAL_FAILED, sip_status)

    def mark_ended(self, reason: TerminationReason, detail: str | None = None) -> None:
        """Record why the call ended. The first reason wins: once the agent hangs up,
        the participant-disconnected event that follows must not overwrite it."""
        if self.termination_reason is None:
            self.termination_reason = reason
            self.termination_detail = detail
        if self.ended_at is None:
            self.ended_at = utc_now()

    def record_error(self, message: str) -> None:
        self.errors.append(message)

    def to_record(
        self, transcript: list[TranscriptTurn], recording: RecordingReference
    ) -> CallRecord:
        ended_at = self.ended_at or utc_now()
        return CallRecord(
            call_id=self.call_id,
            mode=self.mode,
            patient=self.patient.telemetry_view(),
            agent=self.agent_info,
            room_name=self.room_name,
            started_at=self.started_at,
            answered_at=self.answered_at,
            ended_at=ended_at,
            dial_status=self.dial_status,
            sip_status_code=self.sip_status_code,
            sip_status=self.sip_status,
            termination_reason=self.termination_reason or TerminationReason.UNKNOWN,
            termination_detail=self.termination_detail,
            identity_status=self.identity_status,
            health_info_disclosed=self.health_info_disclosed,
            patient_decision=self.patient_decision,
            callback_preference=self.callback_preference,
            offered_slot_ids=list(self.offered_slots),
            appointment=self.appointment,
            tool_events=list(self.tool_events),
            transcript=transcript,
            recording=recording,
            errors=list(self.errors),
        )
