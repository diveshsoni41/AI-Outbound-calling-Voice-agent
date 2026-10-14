"""Business logic behind the agent's function tools.

Each method validates its input, applies the call's rules (identity must be
verified first, only offered slots can be booked, the patient must have
confirmed), calls the scheduling backend, and records a ToolEvent with the
real result. The LiveKit `@function_tool` wrappers in app/agent.py only
forward to these methods, so all of this is testable without LiveKit or an LLM.

Results are returned as JSON strings because that is what the LLM reads.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Literal

from app.call_state import CallState, utc_now
from app.patient import summarize_biomarkers
from app.scheduling import PartOfDay, SchedulingService
from app.schemas import IdentityStatus, PatientDecision, ToolEvent

logger = logging.getLogger("healthcare_agent.tools")

IdentityResponse = Literal["confirmed", "wrong_person", "wrong_number", "not_available"]
DecisionValue = Literal["declined", "callback_requested", "do_not_call"]

NOT_VERIFIED_MESSAGE = (
    "Identity has not been verified. Do not discuss health results or scheduling yet; "
    "confirm you are speaking with the patient and call verify_identity first."
)


class CallTools:
    def __init__(self, state: CallState, scheduler: SchedulingService) -> None:
        self.state = state
        self.scheduler = scheduler

    # -- identity -----------------------------------------------------------------

    def verify_identity(self, response: IdentityResponse) -> str:
        started = utc_now()
        state = self.state
        if response == "confirmed":
            state.identity_status = IdentityStatus.VERIFIED
            state.health_info_disclosed = True
            result = {
                "status": "verified",
                "health_results": [
                    summary.spoken_text for summary in summarize_biomarkers(state.patient)
                ],
                "guidance": (
                    "Share these results in plain language, one at a time, without adding "
                    "any diagnosis or interpretation beyond this text. Share every result, "
                    "even if the patient asks to book first. Then offer to book a "
                    "consultation with a doctor."
                ),
            }
        else:
            state.identity_status = IdentityStatus(response)
            guidance = {
                "wrong_person": (
                    f"Do not share any health information or details about why you are calling. "
                    f"You may ask if there is a better time to reach {state.patient.first_name}, "
                    "then thank them and end the call."
                ),
                "wrong_number": (
                    "Apologize for the mistake, share no details, thank them and end the call."
                ),
                "not_available": (
                    f"Share no health details. Offer to call back at a better time for "
                    f"{state.patient.first_name}; if they give one, record it with "
                    "record_patient_decision (callback_requested). Then end the call."
                ),
            }[response]
            result = {"status": response, "guidance": guidance}
        return self._finish(
            "verify_identity", {"response": response}, result, ok=True, started=started
        )

    # -- scheduling -------------------------------------------------------------

    def check_availability(self, part_of_day: PartOfDay = "any") -> str:
        started = utc_now()
        arguments = {"part_of_day": part_of_day}
        if not self.state.identity_verified:
            return self._finish(
                "check_availability",
                arguments,
                {"status": "error", "error": NOT_VERIFIED_MESSAGE},
                ok=False,
                started=started,
            )

        availability = self.scheduler.check_availability(part_of_day=part_of_day)
        if not availability.ok:
            return self._finish(
                "check_availability",
                arguments,
                {"status": "error", "error": "The scheduling system is unavailable right now."},
                ok=False,
                started=started,
                error=availability.error,
            )

        for slot in availability.slots:
            self.state.offered_slots[slot.slot_id] = slot
        result: dict[str, Any] = {
            "status": "ok",
            "timezone": availability.timezone,
            "slots": [
                {"slot_id": slot.slot_id, "time": slot.spoken, "clinician": slot.clinician}
                for slot in availability.slots
            ],
        }
        if not availability.slots:
            result["guidance"] = (
                "No appointments are available. Do not offer or invent any times. Offer to have "
                "the care team call back to schedule, and record the patient's choice."
            )
        return self._finish("check_availability", arguments, result, ok=True, started=started)

    def book_appointment(self, slot_id: str, patient_confirmed: bool) -> str:
        started = utc_now()
        state = self.state
        arguments = {"slot_id": slot_id, "patient_confirmed": patient_confirmed}

        if not state.identity_verified:
            return self._finish(
                "book_appointment",
                arguments,
                {"status": "error", "error": NOT_VERIFIED_MESSAGE},
                ok=False,
                started=started,
            )
        if state.appointment is not None and state.appointment.slot_id != slot_id:
            # One appointment per call. Rescheduling is out of scope for this agent.
            existing = state.appointment
            return self._finish(
                "book_appointment",
                arguments,
                {
                    "status": "already_booked",
                    "appointment_id": existing.appointment_id,
                    "time": state.offered_slots[existing.slot_id].spoken
                    if existing.slot_id in state.offered_slots
                    else existing.start.isoformat(),
                    "guidance": "An appointment is already booked on this call; do not book another.",
                },
                ok=False,
                started=started,
                error="already_booked",
            )
        if slot_id not in state.offered_slots:
            return self._finish(
                "book_appointment",
                arguments,
                {
                    "status": "failed",
                    "error_code": "slot_not_offered",
                    "error": "That slot was not offered on this call. Only book a slot_id returned "
                    "by check_availability; offer the available options again.",
                },
                ok=False,
                started=started,
                error="slot_not_offered",
            )

        booking = self.scheduler.book_appointment(
            patient_id=state.patient.patient_id,
            slot_id=slot_id,
            # Stable per call and slot: an LLM retry or a repeated tool call maps to
            # the same appointment instead of creating a second one.
            idempotency_key=f"{state.call_id}:{slot_id}",
            patient_confirmed=patient_confirmed,
        )
        if booking.ok and booking.appointment is not None:
            state.appointment = booking.appointment
            result = {
                "status": "booked",
                "appointment_id": booking.appointment.appointment_id,
                "time": state.offered_slots[slot_id].spoken,
                "timezone": booking.appointment.timezone,
                "clinician": booking.appointment.clinician,
            }
            return self._finish("book_appointment", arguments, result, ok=True, started=started)

        result = {
            "status": "failed",
            "error_code": booking.error_code.value if booking.error_code else "unknown",
            "error": booking.error_message,
            "guidance": "The appointment is NOT booked. Tell the patient plainly that the booking "
            "did not go through, then offer another available time or a call back from the "
            "care team.",
        }
        if booking.error_code and booking.error_code.value == "confirmation_required":
            result["guidance"] = (
                "Not booked. Read the slot back to the patient and get a clear yes before "
                "calling book_appointment with patient_confirmed=true."
            )
        return self._finish(
            "book_appointment",
            arguments,
            result,
            ok=False,
            started=started,
            error=result["error_code"],
        )

    # -- decisions -----------------------------------------------------------------

    def record_patient_decision(
        self, decision: DecisionValue, callback_preference: str | None = None
    ) -> str:
        started = utc_now()
        arguments = {"decision": decision, "callback_preference": callback_preference}
        if self.state.appointment is not None and decision != "do_not_call":
            # Saying goodbye after booking is not a decline; keep the record consistent.
            return self._finish(
                "record_patient_decision",
                arguments,
                {
                    "status": "already_booked",
                    "guidance": "An appointment is already booked on this call, so there is no "
                    "decline or call back to record. Close the call politely.",
                },
                ok=False,
                started=started,
                error="already_booked",
            )
        self.state.patient_decision = PatientDecision(decision)
        if callback_preference:
            # Free text from the patient ("Thursday after 5"). Kept short; not parsed.
            self.state.callback_preference = callback_preference.strip()[:200]
        result = {"status": "recorded", "decision": decision}
        return self._finish("record_patient_decision", arguments, result, ok=True, started=started)

    # -- helpers -------------------------------------------------------------------

    def _finish(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        *,
        ok: bool,
        started: datetime,
        error: str | None = None,
    ) -> str:
        event = ToolEvent(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            ok=ok,
            error=error or (result.get("error") if not ok else None),
            started_at=started,
            ended_at=utc_now(),
        )
        self.state.tool_events.append(event)
        logger.info(
            "tool executed",
            extra={
                "call_id": self.state.call_id,
                "tool": tool_name,
                "ok": ok,
                "error": event.error,
            },
        )
        return json.dumps(result)
