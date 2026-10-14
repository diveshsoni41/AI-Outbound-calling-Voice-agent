"""Place a REAL outbound call to an authorized test number.

The agent worker must already be running (`python -m app.agent dev`), and the
patient's phone number must be listed in AUTHORIZED_TEST_NUMBERS.

    python scripts/place_call.py --patient data/demo_patient.json --phone +15551234567
    python scripts/place_call.py --patient data/demo_patient.json --no-wait

--phone overrides the number in the patient file so the synthetic patient can
be pointed at your own test handset without editing the fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from app.call_manager import CallInitiationError, OutboundCallManager, wait_for_report  # noqa: E402
from app.config import ConfigurationError, get_settings  # noqa: E402
from app.patient import Patient, load_patient  # noqa: E402
from app.runtime import build_store  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--patient", default="data/demo_patient.json")
    parser.add_argument("--phone", help="override the patient's phone number (E.164)")
    parser.add_argument("--no-wait", action="store_true", help="return right after dispatch")
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds to wait for the call report"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        settings = get_settings()
        patient = load_patient(args.patient)
        if args.phone:
            patient = Patient.model_validate({**patient.model_dump(), "phone_number": args.phone})
    except (ConfigurationError, ValidationError) as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        handle = await OutboundCallManager(settings).place_call(patient)
    except (ConfigurationError, CallInitiationError) as exc:
        print(f"Call not placed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Agent dispatched. call_id={handle.call_id} room={handle.room_name} dispatch={handle.dispatch_id}"
    )
    if args.no_wait:
        return 0

    print("Waiting for the call to finish and the post-call report to be written...")
    report = await wait_for_report(
        build_store(settings), handle.call_id, timeout_seconds=args.timeout
    )
    if report is None:
        print(
            "No report yet. Check the worker logs; the report appears under "
            f"{settings.artifacts_dir}/calls/{handle.call_id}/ when the call ends.",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "call_id": report.record.call_id,
                "dial_status": report.record.dial_status.value,
                "termination": report.record.termination_reason.value,
                "outcome": report.analysis.outcome.value,
                "booking_status": report.analysis.booking_status.value,
                "appointment_id": report.analysis.appointment_id,
                "recording": report.record.recording.status,
                "opik": report.telemetry.status,
                "opik_trace_id": report.telemetry.trace_id,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
