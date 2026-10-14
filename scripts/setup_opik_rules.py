"""Create the Opik online evaluation rules for the call project.

    python scripts/setup_opik_rules.py                 # Python code rules
    python scripts/setup_opik_rules.py --llm-judge     # also the LLM-as-a-judge rule
    python scripts/setup_opik_rules.py --replace       # recreate rules that already exist

Requires OPIK_API_KEY (Opik Cloud) or OPIK_URL_OVERRIDE (self-hosted). The
LLM-as-a-judge rule additionally needs an AI provider key configured in the
Opik workspace (Configuration -> AI Providers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ConfigurationError, Requirement, get_settings  # noqa: E402
from app.observability.evaluation import install_rules  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--llm-judge", action="store_true", help="also create the LLM-as-a-judge rule"
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    try:
        settings = get_settings()
        settings.require(Requirement.OPIK)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    import opik
    from opik.rest_api.core.api_error import ApiError

    client = opik.Opik(
        project_name=settings.opik_project_name,
        workspace=settings.opik_workspace,
        api_key=settings.opik_api_key.get_secret_value() if settings.opik_api_key else None,
        host=settings.opik_url_override,
    )
    try:
        results = install_rules(
            client,
            settings.opik_project_name,
            include_llm_judge=args.llm_judge,
            judge_model=args.judge_model,
            replace=args.replace,
        )
    except ApiError as exc:
        print(f"Opik rejected the request (HTTP {exc.status_code}): {exc.body}", file=sys.stderr)
        print(
            "If Python code rules are not available on your Opik plan or version, create the "
            "rule in the UI instead (see docs/opik_evaluation.md).",
            file=sys.stderr,
        )
        return 1

    print(f"Project: {settings.opik_project_name}")
    for result in results:
        print(f"  {result['name']:26} {result['status']:8} id={result['ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
