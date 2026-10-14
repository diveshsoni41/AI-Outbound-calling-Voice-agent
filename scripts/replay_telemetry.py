"""Re-send stored call reports to Opik (for example after an Opik outage).

    python scripts/replay_telemetry.py --call-id call_xxxxxxxx
    python scripts/replay_telemetry.py --all-unsent

Reports whose telemetry status is already "sent" are skipped unless --force is
given, so replaying never creates duplicate traces by accident. The stored
analysis is reused as-is; nothing is re-analyzed or re-booked.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ConfigurationError, Requirement, get_settings  # noqa: E402
from app.pipeline import finalize_call  # noqa: E402
from app.runtime import build_analyzer, build_observer, build_store  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--call-id")
    target.add_argument("--all-unsent", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-send even if already sent")
    args = parser.parse_args()

    try:
        settings = get_settings()
        settings.require(Requirement.OPIK)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    store = build_store(settings)
    if args.call_id:
        call_ids = [args.call_id]
    else:
        call_ids = [
            p.parent.name for p in sorted((store.root / "calls").glob("*/call_report.json"))
        ]

    observer = build_observer(settings)
    analyzer = build_analyzer(settings)
    exit_code = 0
    for call_id in call_ids:
        report = store.load(call_id)
        if report is None:
            print(f"{call_id}: no stored report", file=sys.stderr)
            exit_code = 1
            continue
        if report.telemetry.status == "sent" and not args.force:
            if args.call_id:
                print(
                    f"{call_id}: already sent (trace {report.telemetry.trace_id}); use --force to resend"
                )
            continue
        updated = await finalize_call(
            report.record,
            analyzer=analyzer,
            observer=observer,
            store=store,
            force_telemetry=args.force,
        )
        if updated is None:
            print(f"{call_id}: finalization in progress elsewhere; skipped")
            continue
        print(
            f"{call_id}: {updated.telemetry.status} trace={updated.telemetry.trace_id} {updated.telemetry.error or ''}"
        )
        if updated.telemetry.status != "sent":
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
