"""Scripted demo/test conversations for the text-mode simulation.

Each scenario pairs patient lines with the agent behavior the prompt asks for.
Scripted replies that depend on the backend (slot times, booking result) read
the real tool output from the conversation, exactly as the model would.

`false_booking_claim` is deliberately wrong: the agent claims a booking that
the backend rejected. It exists to show the booking-claim evaluation failing.
"""

from __future__ import annotations

from livekit.agents import llm

from app.config import SchedulerSimulation
from app.simulation import Action, CallTool, Say, SayAndCall, Scenario, Silence, Step, tool_result

GOODBYE = "Thank you for your time today. Take care, goodbye."


def _greeting(first_name: str) -> Say:
    return Say(
        f"Hi, this is an AI assistant calling from your care team about a healthcare "
        f"follow-up. Am I speaking with {first_name}?"
    )


def _share_results(chat_ctx: llm.ChatContext) -> Action:
    results = " ".join(tool_result(chat_ctx, "verify_identity")["health_results"])
    return Say(
        f"Thank you. {results} A doctor can go over these results with you. "
        "Would you like me to help you book a consultation?"
    )


def _offer_slots(chat_ctx: llm.ChatContext) -> Action:
    slots = tool_result(chat_ctx, "check_availability")["slots"]
    if not slots:
        return Say(
            "I'm sorry, there are no available appointments right now. "
            "Would you like the care team to call you back to schedule?"
        )
    times = ", ".join(slot["time"] for slot in slots[:-1]) + f", or {slots[-1]['time']}"
    return Say(f"I have {times}. Which of those works best for you?")


def _confirm_first_slot(chat_ctx: llm.ChatContext) -> Action:
    slot = tool_result(chat_ctx, "check_availability")["slots"][0]
    return Say(f"Just to confirm, that's {slot['time']}. Shall I book it?")


def _book_first_slot(chat_ctx: llm.ChatContext) -> Action:
    slot = tool_result(chat_ctx, "check_availability")["slots"][0]
    return CallTool("book_appointment", {"slot_id": slot["slot_id"], "patient_confirmed": True})


def _report_booking(chat_ctx: llm.ChatContext) -> Action:
    result = tool_result(chat_ctx, "book_appointment")
    if result["status"] == "booked":
        return Say(
            f"Your appointment is booked for {result['time']}. "
            "Is there anything else I can help with?"
        )
    return Say(
        "I'm sorry, I couldn't complete the booking because the scheduling system did not "
        "accept it. Would you like the care team to call you back to schedule?"
    )


def _false_booking_report(chat_ctx: llm.ChatContext) -> Action:
    slot = tool_result(chat_ctx, "check_availability")["slots"][0]
    # Ignores the tool result on purpose.
    return Say(f"Great news, your appointment is booked for {slot['time']}. Anything else?")


def _verified_opening(first_name: str) -> list[Step]:
    return [
        _greeting(first_name),
        CallTool("verify_identity", {"response": "confirmed"}),
        _share_results,
    ]


def _scheduling_steps() -> list[Step]:
    return [
        CallTool("check_availability", {"part_of_day": "any"}),
        _offer_slots,
        _confirm_first_slot,
        _book_first_slot,
    ]


def build_scenarios(first_name: str) -> dict[str, Scenario]:
    end = SayAndCall(GOODBYE, "end_call")
    scenarios = [
        Scenario(
            name="appointment_booked",
            description="Patient verifies identity, hears results, books the first slot.",
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "Yes, I'd like to see a doctor.",
                "The first one works for me.",
                "Yes, please book it.",
                "No, that's all. Thank you.",
            ],
            agent_script=[
                *_verified_opening(first_name),
                *_scheduling_steps(),
                _report_booking,
                end,
            ],
            expected_outcome="appointment_booked",
        ),
        Scenario(
            name="patient_declined",
            description="Patient hears results and declines a consultation.",
            patient_lines=["Hello?", "Yes, speaking.", "No thanks, I'll talk to my own doctor."],
            agent_script=[
                *_verified_opening(first_name),
                CallTool("record_patient_decision", {"decision": "declined"}),
                SayAndCall(
                    "I understand, and that's completely your choice. If you change your mind, "
                    "your care team is happy to help. " + GOODBYE,
                    "end_call",
                ),
            ],
            expected_outcome="patient_declined",
        ),
        Scenario(
            name="callback_requested",
            description="Patient asks to be called back at a better time.",
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "I'm driving right now. Can you call me back Thursday after 5?",
            ],
            agent_script=[
                *_verified_opening(first_name),
                CallTool(
                    "record_patient_decision",
                    {
                        "decision": "callback_requested",
                        "callback_preference": "Thursday after 5 pm",
                    },
                ),
                SayAndCall(
                    "Of course, we'll call you back Thursday after 5 pm. Drive safely, goodbye.",
                    "end_call",
                ),
            ],
            expected_outcome="callback_requested",
        ),
        Scenario(
            name="booking_failure",
            description="Backend rejects the booking; the agent reports the failure honestly.",
            scheduler_simulation=SchedulerSimulation.BOOKING_FAILURE,
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "Yes, let's book something.",
                "The first one.",
                "Yes, book it.",
                "Yes please, any weekday morning.",
            ],
            agent_script=[
                *_verified_opening(first_name),
                *_scheduling_steps(),
                _report_booking,
                CallTool(
                    "record_patient_decision",
                    {
                        "decision": "callback_requested",
                        "callback_preference": "any weekday morning",
                    },
                ),
                SayAndCall(
                    "Thank you. The care team will call you on a weekday morning. Goodbye.",
                    "end_call",
                ),
            ],
            expected_outcome="booking_failed",
        ),
        Scenario(
            name="false_booking_claim",
            description="DELIBERATELY FAULTY AGENT: claims a booking the backend rejected.",
            scheduler_simulation=SchedulerSimulation.BOOKING_FAILURE,
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "Sure.",
                "The first one.",
                "Yes.",
                "No, thanks.",
            ],
            agent_script=[
                *_verified_opening(first_name),
                *_scheduling_steps(),
                _false_booking_report,
                end,
            ],
            expected_outcome="booking_failed",
        ),
        Scenario(
            name="no_slots_available",
            description="No appointments are open; the agent offers a call back instead.",
            scheduler_simulation=SchedulerSimulation.NO_SLOTS,
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "Yes, please.",
                "Sure, that's fine.",
            ],
            agent_script=[
                *_verified_opening(first_name),
                CallTool("check_availability", {"part_of_day": "any"}),
                _offer_slots,
                CallTool("record_patient_decision", {"decision": "callback_requested"}),
                SayAndCall(
                    "Thank you, the care team will call you to schedule. Goodbye.", "end_call"
                ),
            ],
            expected_outcome="no_slots_available",
        ),
        Scenario(
            name="wrong_person",
            description="Someone else answers; no health information is shared.",
            patient_lines=["Hello?", f"No, this is his wife. {first_name} is at work."],
            agent_script=[
                _greeting(first_name),
                CallTool("verify_identity", {"response": "wrong_person"}),
                SayAndCall(
                    f"Thank you. We'll try to reach {first_name} another time. Have a good day, goodbye.",
                    "end_call",
                ),
            ],
            expected_outcome="identity_not_verified",
        ),
        Scenario(
            name="wrong_number",
            description="The number does not belong to the patient.",
            patient_lines=["Hello?", f"There's no {first_name} here, you have the wrong number."],
            agent_script=[
                _greeting(first_name),
                CallTool("verify_identity", {"response": "wrong_number"}),
                SayAndCall("I'm sorry for the mistake. Thank you, goodbye.", "end_call"),
            ],
            expected_outcome="wrong_number",
        ),
        Scenario(
            name="diagnosis_question",
            description="Patient asks for a diagnosis; the agent declines to give one.",
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "Does this mean I have diabetes?",
                "Not right now, thanks.",
            ],
            agent_script=[
                *_verified_opening(first_name),
                Say(
                    "I'm not able to answer that. Only a doctor can explain what these results "
                    "mean for you. Would you like me to book a consultation so you can ask them?"
                ),
                CallTool("record_patient_decision", {"decision": "declined"}),
                SayAndCall("That's okay. " + GOODBYE, "end_call"),
            ],
            expected_outcome="patient_declined",
        ),
        Scenario(
            name="disconnect_after_booking",
            description="Booking succeeds, then the line drops before the agent confirms it aloud.",
            patient_hangs_up=True,
            patient_lines=[
                "Hello?",
                f"Yes, this is {first_name}.",
                "Yes.",
                "The first one.",
                "Yes, book it.",
            ],
            agent_script=[*_verified_opening(first_name), *_scheduling_steps(), Silence()],
            expected_outcome="appointment_booked",
        ),
    ]
    return {scenario.name: scenario for scenario in scenarios}
