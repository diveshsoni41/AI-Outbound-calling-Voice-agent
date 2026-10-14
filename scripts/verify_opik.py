"""Check that a call's trace, spans, attachment and online evaluation scores
actually exist in Opik (reads them back from the Opik API).

    python scripts/verify_opik.py --call-id call_xxxxxxxx
    python scripts/verify_opik.py --trace-id 0192...      # without a local report
    python scripts/verify_opik.py --call-id call_x --wait 180

Online evaluation runs asynchronously on the Opik side, so --wait polls for
the feedback scores. Results are saved to artifacts/opik_verification/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ConfigurationError, Requirement, get_settings  # noqa: E402
from app.observability.evaluation import LLM_JUDGE_RULE_NAME, PYTHON_RULES  # noqa: E402
from app.observability.opik_integration import CALL_TRACE_TAG  # noqa: E402
from app.runtime import build_store  # noqa: E402

EXPECTED_SPANS = {"call_setup", "conversation", "recording", "post_call_analysis"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--call-id")
    target.add_argument("--trace-id")
    parser.add_argument(
        "--wait", type=int, default=120, help="seconds to wait for evaluation scores"
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
        settings.require(Requirement.OPIK)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    trace_id = args.trace_id
    if args.call_id:
        report = build_store(settings).load(args.call_id)
        if report is None or not report.telemetry.trace_id:
            print(f"No local report with an Opik trace ID for {args.call_id}.", file=sys.stderr)
            return 1
        trace_id = report.telemetry.trace_id

    import opik

    client = opik.Opik(
        project_name=settings.opik_project_name,
        workspace=settings.opik_workspace,
        api_key=settings.opik_api_key.get_secret_value() if settings.opik_api_key else None,
        host=settings.opik_url_override,
    )

    trace = client.get_trace_content(trace_id)
    output = trace.output or {}
    spans = client.search_spans(
        project_name=settings.opik_project_name, trace_id=trace_id, truncate=True
    )
    span_names = sorted({span.name for span in spans})
    attachments = client.get_attachment_client().get_attachment_list(
        project_name=settings.opik_project_name, entity_id=trace_id, entity_type="trace"
    )

    expected_scores = {rule.name for rule in PYTHON_RULES}
    deadline = time.monotonic() + args.wait
    scores = {}
    while True:
        trace = client.get_trace_content(trace_id)
        scores = {
            s.name: {"value": s.value, "reason": s.reason} for s in (trace.feedback_scores or [])
        }
        if expected_scores <= scores.keys() or time.monotonic() >= deadline:
            break
        print(f"waiting for online evaluation scores... have {sorted(scores) or 'none'}")
        time.sleep(10)

    checks = {
        "trace_found": True,
        "tagged_for_rules": CALL_TRACE_TAG in (trace.tags or []),
        "has_call_metadata": bool((trace.metadata or {}).get("call_id")),
        "has_transcript": isinstance(output.get("transcript"), list),
        "has_booking": isinstance(output.get("booking"), dict),
        "has_post_call_analysis": isinstance(output.get("post_call_analysis"), dict),
        "has_recording_reference": isinstance(output.get("recording"), dict),
        "has_tool_spans": any(name.startswith("tool:") for name in span_names),
        "has_expected_spans": set(span_names) >= EXPECTED_SPANS,
        "online_eval_scores_present": expected_scores <= scores.keys(),
    }
    result = {
        "trace_id": trace_id,
        "call_id": (trace.metadata or {}).get("call_id"),
        "mode": (trace.metadata or {}).get("mode"),
        "outcome": output.get("outcome"),
        "recording_status": (output.get("recording") or {}).get("status"),
        "attachments": [a.file_name for a in attachments],
        "spans": span_names,
        "feedback_scores": scores,
        "llm_judge_scores_present": any(
            name in scores for name in ("disclosed_ai", LLM_JUDGE_RULE_NAME)
        ),
        "checks": checks,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(result, indent=2))

    out_dir = settings.artifacts_dir / "opik_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{trace_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print(f"\nFAILED checks: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
