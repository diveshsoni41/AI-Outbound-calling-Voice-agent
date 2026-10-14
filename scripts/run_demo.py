"""Run the end-to-end flow in SIMULATION mode (text, no phone call).

Each scenario runs the real agent, tools and scheduling backend in a text-only
LiveKit AgentSession, then the real post-call analysis, local report and (if
configured) Opik trace.

    python scripts/run_demo.py                      # default scenarios, scripted LLM
    python scripts/run_demo.py --scenario all
    python scripts/run_demo.py --llm openai         # real model decides the agent's turns
    python scripts/run_demo.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ConfigurationError, Requirement, get_settings  # noqa: E402
from app.patient import load_patient  # noqa: E402
from app.pipeline import CallReport  # noqa: E402
from app.runtime import build_analyzer, build_observer, build_store  # noqa: E402
from app.scenarios import build_scenarios  # noqa: E402
from app.scheduling import SchedulingService  # noqa: E402
from app.simulation import run_simulated_call  # noqa: E402
from app.transcript import format_transcript  # noqa: E402

DEFAULT_SCENARIOS = [
    "appointment_booked",
    "callback_requested",
    "booking_failure",
    "false_booking_claim",
]


def print_report(name: str, report: CallReport) -> None:
    record, analysis, telemetry = report.record, report.analysis, report.telemetry
    bar = "=" * 78
    print(f"\n{bar}\nSCENARIO: {name}   [SIMULATION - text only, no phone call]\n{bar}")
    print(f"call_id:      {record.call_id}")
    print(f"patient:      {json.dumps(record.patient)}")
    print(f"agent model:  {record.agent.llm_model}")
    print("\n--- transcript ---")
    print(format_transcript(record.transcript) or "(empty)")
    print("\n--- tool calls (as executed by the backend) ---")
    for event in record.tool_events:
        status = "ok" if event.ok else f"FAILED ({event.error})"
        print(f"{event.tool_name}({json.dumps(event.arguments)}) -> {status}")
        print(f"    result: {json.dumps(event.result)[:300]}")
    print("\n--- post-call analysis ---")
    print(f"outcome:          {analysis.outcome.value} - {analysis.outcome_reason}")
    print(
        f"booking_status:   {analysis.booking_status.value}  appointment_id={analysis.appointment_id}"
    )
    print(
        f"booking claim:    {analysis.booking_claim_check.value:.0f} - {analysis.booking_claim_check.reason}"
    )
    print(f"safety flags:     {analysis.safety_flags or 'none'}")
    print(f"contradictions:   {analysis.contradictions or 'none'}")
    print(f"qualitative:      {analysis.qualitative_source}")
    if analysis.qualitative:
        print(f"summary:          {analysis.qualitative.summary}")
    print("\n--- Opik ---")
    print(f"status:   {telemetry.status}")
    if telemetry.trace_id:
        print(f"trace_id: {telemetry.trace_id}")
        print(f"project:  {telemetry.project_url or telemetry.project_name}")
    if telemetry.error:
        print(f"detail:   {telemetry.error}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenario", nargs="+", default=DEFAULT_SCENARIOS, help="scenario names, or 'all'"
    )
    parser.add_argument("--llm", choices=["scripted", "openai"], default="scripted")
    parser.add_argument("--patient", default="data/demo_patient.json")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--evidence-dir", default="artifacts/demo", help="where to save demo evidence"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        settings = get_settings()
        if args.llm == "openai":
            settings.require(Requirement.OPENAI)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    patient = load_patient(args.patient)
    scenarios = build_scenarios(patient.first_name)
    if args.list:
        for scenario in scenarios.values():
            print(f"{scenario.name:26} {scenario.description}")
        return 0

    names = list(scenarios) if args.scenario == ["all"] else args.scenario
    unknown = [name for name in names if name not in scenarios]
    if unknown:
        print(f"Unknown scenario(s): {', '.join(unknown)}. Use --list.", file=sys.stderr)
        return 2

    analyzer = build_analyzer(settings)
    observer = build_observer(settings)
    store = build_store(settings)
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for name in names:
        scenario = scenarios[name]
        # A fresh in-memory schedule per scenario keeps runs independent and repeatable.
        scheduler = SchedulingService(
            ":memory:", settings.timezone, simulation=scenario.scheduler_simulation
        )
        model = None
        if args.llm == "openai":
            from livekit.plugins import openai

            model = openai.LLM(model=settings.llm_model)
        report = await run_simulated_call(
            scenario,
            patient=patient,
            settings=settings,
            scheduler=scheduler,
            analyzer=analyzer,
            observer=observer,
            store=store,
            model=model,
        )
        print_report(name, report)
        (evidence_dir / f"{name}.json").write_text(
            json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        summary.append(
            {
                "scenario": name,
                "mode": "simulation",
                "llm": report.record.agent.llm_model,
                "call_id": report.record.call_id,
                "expected_outcome": scenario.expected_outcome if args.llm == "scripted" else None,
                "outcome": report.analysis.outcome.value,
                "booking_status": report.analysis.booking_status.value,
                "booking_claim_check": report.analysis.booking_claim_check.value,
                "opik_status": report.telemetry.status,
                "opik_trace_id": report.telemetry.trace_id,
            }
        )

    (evidence_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nEvidence written to {evidence_dir}/ (full reports also under {store.root}/calls/)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
