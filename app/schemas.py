"""Validated records shared by the agent, post-call analysis and observability."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.scheduling import Appointment


class CallMode(str, Enum):
    TELEPHONY = "telephony"
    SIMULATION = "simulation"


class DialStatus(str, Enum):
    NOT_STARTED = "not_started"
    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    REJECTED = "rejected"
    FAILED = "failed"
    SIMULATED = "simulated"


class IdentityStatus(str, Enum):
    UNKNOWN = "unknown"
    VERIFIED = "verified"
    WRONG_PERSON = "wrong_person"
    WRONG_NUMBER = "wrong_number"
    NOT_AVAILABLE = "not_available"


class PatientDecision(str, Enum):
    DECLINED = "declined"
    CALLBACK_REQUESTED = "callback_requested"
    DO_NOT_CALL = "do_not_call"


class TerminationReason(str, Enum):
    AGENT_ENDED_CALL = "agent_ended_call"
    PATIENT_HUNG_UP = "patient_hung_up"
    CONNECTION_LOST = "connection_lost"
    DIAL_FAILED = "dial_failed"
    AGENT_ERROR = "agent_error"
    JOB_SHUTDOWN = "job_shutdown"
    SIMULATION_COMPLETED = "simulation_completed"
    UNKNOWN = "unknown"


class ToolEvent(BaseModel):
    """One tool invocation as executed by our code (not as reported by the LLM)."""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    ok: bool
    error: str | None = None
    started_at: datetime
    ended_at: datetime


class TranscriptTurn(BaseModel):
    speaker: Literal["agent", "patient"]
    text: str
    timestamp: datetime | None = None
    interrupted: bool = False


class RecordingReference(BaseModel):
    """Where the call audio lives. Never contains the audio bytes themselves."""

    status: Literal["available", "not_available", "failed"]
    uri: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    started_at: datetime | None = None
    reason: str | None = None


class AgentInfo(BaseModel):
    agent_name: str
    agent_version: str
    llm_model: str
    stt_model: str | None
    tts_model: str | None
    tts_voice: str | None
    prompt_version: str


class CallRecord(BaseModel):
    """Everything known about a finished call. Input to post-call analysis."""

    call_id: str
    mode: CallMode
    patient: dict[str, Any] = Field(description="Patient.telemetry_view(): no name, masked phone")
    agent: AgentInfo
    room_name: str | None = None
    started_at: datetime
    answered_at: datetime | None = None
    ended_at: datetime
    dial_status: DialStatus
    sip_status_code: int | None = None
    sip_status: str | None = None
    termination_reason: TerminationReason
    termination_detail: str | None = None
    identity_status: IdentityStatus
    health_info_disclosed: bool
    patient_decision: PatientDecision | None = None
    callback_preference: str | None = None
    offered_slot_ids: list[str] = Field(default_factory=list)
    appointment: Appointment | None = None
    tool_events: list[ToolEvent] = Field(default_factory=list)
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    recording: RecordingReference
    errors: list[str] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


# Post-call analysis


class CallOutcome(str, Enum):
    APPOINTMENT_BOOKED = "appointment_booked"
    PATIENT_DECLINED = "patient_declined"
    CALLBACK_REQUESTED = "callback_requested"
    DO_NOT_CALL = "do_not_call"
    NO_ANSWER = "no_answer"
    WRONG_NUMBER = "wrong_number"
    IDENTITY_NOT_VERIFIED = "identity_not_verified"
    BOOKING_FAILED = "booking_failed"
    NO_SLOTS_AVAILABLE = "no_slots_available"
    CALL_DISCONNECTED = "call_disconnected"
    TECHNICAL_FAILURE = "technical_failure"
    INCOMPLETE = "incomplete"


class BookingStatus(str, Enum):
    BOOKED = "booked"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class QualitativeAnalysis(BaseModel):
    """Fields an LLM may fill from the transcript. Never used for booking facts."""

    summary: str = Field(description="Two or three neutral sentences describing the call")
    patient_interest: Literal["interested", "not_interested", "undecided", "unknown"]
    patient_concerns: list[str] = Field(description="Concerns or preferences the patient voiced")
    unresolved_questions: list[str] = Field(
        description="Questions the patient asked that the agent could not answer"
    )
    safety_concerns: list[str] = Field(
        description="Agent statements that diagnose, advise on medication or treatment, or "
        "invent clinical facts. Comparing a result with its reference range is not a "
        "concern. Empty if none."
    )


class BookingClaimCheck(BaseModel):
    value: float
    reason: str
    claims: list[dict[str, str]]


class PostCallAnalysis(BaseModel):
    call_id: str
    analyzer_version: str
    analyzed_at: datetime
    outcome: CallOutcome
    outcome_reason: str
    call_completed: bool
    patient_reached: bool
    identity_verified: bool
    booking_status: BookingStatus
    appointment_id: str | None = None
    appointment_start: datetime | None = None
    appointment_timezone: str | None = None
    no_booking_reason: str | None = None
    callback_preference: str | None = None
    biomarkers_discussed: list[str]
    booking_claim_check: BookingClaimCheck
    safety_flags: list[str]
    tool_failures: list[str]
    contradictions: list[str]
    qualitative: QualitativeAnalysis | None = None
    qualitative_source: Literal["llm", "not_requested", "unavailable"]
    analysis_errors: list[str] = Field(default_factory=list)
