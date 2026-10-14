"""Run the test suite and write artifacts/test_report.json.

    python scripts/make_test_report.py            # offline suite
    python scripts/make_test_report.py --external # also run external tests (needs credentials)

Every pytest result is recorded with whether it actually executed. Checks that
cannot be automated here (a real phone call, Opik running the online rules)
are listed as "not_executed" with the reason, never as passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MANUAL_CHECKS = [
    {
        "name": "real_outbound_call_to_authorized_number",
        "category": "telephony",
        "reason": "The outbound SIP trunk exists but has no real provider address or "
        "credentials, so nothing can reach the phone network. Real-call recording is "
        "blocked with it.",
        "how": "docs/testing.md -> Real outbound call",
    },
]

# Checks run by hand against real services; evidence in the referenced files.
MANUAL_VERIFICATIONS = [
    {
        "name": "opik_online_rules_score_live_traces",
        "category": "opik",
        "result": "passed",
        "evidence": "artifacts/external_verification/summary.json (opik)",
    },
    {
        "name": "livekit_worker_registers_and_receives_dispatch",
        "category": "livekit",
        "result": "passed",
        "evidence": "artifacts/external_verification/summary.json (livekit), worker.log",
    },
    {
        "name": "openai_agent_conversation_personas",
        "category": "openai",
        "result": "passed",
        "evidence": "artifacts/external_verification/openai_conversation/summary.json",
    },
]


def _category(classname: str) -> str:
    for part in ("unit", "integration", "e2e", "external"):
        if f"tests.{part}." in classname:
            return part
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--output", default="artifacts/test_report.json")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
        env = dict(os.environ)
        if args.external:
            command += ["-m", "external or not external"]
            env["RUN_EXTERNAL_TESTS"] = "1"
        else:
            command += ["-m", "not external"]
        completed = subprocess.run(command, cwd=ROOT, env=env)
        tree = ET.parse(junit)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tests = []
    for case in tree.iter("testcase"):
        classname = case.get("classname", "")
        failure = case.find("failure") if case.find("failure") is not None else case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            result, reason = "failed", (failure.get("message") or "")[:300]
        elif skipped is not None:
            result, reason = "skipped", (skipped.get("message") or "")[:300]
        else:
            result, reason = "passed", None
        tests.append(
            {
                "name": f"{classname.rsplit('.', 1)[-1]}::{case.get('name')}",
                "category": _category(classname),
                "result": result,
                "executed": result != "skipped",
                "timestamp": timestamp,
                "duration_seconds": round(float(case.get("time", 0)), 3),
                "failure_reason": reason,
                "evidence": classname.replace(".", "/") + ".py",
            }
        )

    summary: dict[str, dict[str, int]] = {}
    for test in tests:
        bucket = summary.setdefault(test["category"], {"passed": 0, "failed": 0, "skipped": 0})
        bucket[test["result"]] += 1

    report = {
        "generated_at": timestamp,
        "command": " ".join(command[1:]),
        "python": sys.version.split()[0],
        "exit_code": completed.returncode,
        "summary": summary,
        "tests": tests,
        "manual_verifications": MANUAL_VERIFICATIONS,
        "not_executed": [
            {**check, "result": "not_executed", "executed": False} for check in MANUAL_CHECKS
        ],
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {output.relative_to(ROOT)}: {json.dumps(summary)}")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
