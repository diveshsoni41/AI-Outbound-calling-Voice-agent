"""Post-call pipeline: analyze, persist, send to Opik. Safe to run more than once.

A call can be finalized twice (a retried shutdown hook, a manual replay after
an Opik outage, a restarted worker). Each step is skipped if its result is
already stored, so a retry never produces a second analysis or a duplicate
trace, and it never touches the scheduling backend at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

from app.call_state import utc_now
from app.observability.opik_integration import OpikCallObserver, TelemetryResult
from app.post_call import PostCallAnalyzer
from app.schemas import CallRecord, PostCallAnalysis

logger = logging.getLogger("healthcare_agent.pipeline")

# Finalization (LLM summary + Opik flush) takes well under a minute.
STALE_LOCK_SECONDS = 600


class CallReport(BaseModel):
    record: CallRecord
    analysis: PostCallAnalysis
    telemetry: TelemetryResult


class CallReportStore:
    """One JSON report per call under <artifacts>/calls/<call_id>/."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def call_dir(self, call_id: str) -> Path:
        return self.root / "calls" / call_id

    def report_path(self, call_id: str) -> Path:
        return self.call_dir(call_id) / "call_report.json"

    def load(self, call_id: str) -> CallReport | None:
        path = self.report_path(call_id)
        if not path.is_file():
            return None
        return CallReport.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, report: CallReport) -> Path:
        path = self.report_path(report.record.call_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic: readers never see a half-written report
        return path

    @contextlib.contextmanager
    def lock(self, call_id: str) -> Iterator[bool]:
        """Exclusive per-call lock across processes. Yields False if already held."""
        path = self.call_dir(call_id) / ".finalize.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        # A lock left behind by a crashed worker would block retries forever.
        with contextlib.suppress(FileNotFoundError):
            if time.time() - path.stat().st_mtime > STALE_LOCK_SECONDS:
                logger.warning("removing stale finalize lock", extra={"call_id": call_id})
                path.unlink()
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            yield False
            return
        try:
            os.write(fd, str(os.getpid()).encode())
            yield True
        finally:
            os.close(fd)
            path.unlink(missing_ok=True)


async def finalize_call(
    record: CallRecord,
    *,
    analyzer: PostCallAnalyzer,
    observer: OpikCallObserver | None,
    store: CallReportStore,
    force_telemetry: bool = False,
) -> CallReport | None:
    with store.lock(record.call_id) as acquired:
        if not acquired:
            logger.warning(
                "finalization already in progress; skipping", extra={"call_id": record.call_id}
            )
            return None

        existing = store.load(record.call_id)
        if existing is not None:
            # Keep the first analysis: the logical result of a call must not change
            # because the pipeline ran again.
            record, analysis, telemetry = existing.record, existing.analysis, existing.telemetry
            if telemetry.status == "sent" and not force_telemetry:
                logger.info("call already finalized", extra={"call_id": record.call_id})
                return existing
        else:
            analysis = await analyzer.analyze(record)
            telemetry = TelemetryResult(status="pending", attempted_at=utc_now())

        # Persist before telemetry so an Opik outage or a crash cannot lose the result.
        store.save(CallReport(record=record, analysis=analysis, telemetry=telemetry))

        if observer is not None:
            telemetry = await asyncio.to_thread(observer.log_call, record, analysis)
        else:
            telemetry = TelemetryResult(
                status="disabled", error="Opik is not configured.", attempted_at=utc_now()
            )

        report = CallReport(record=record, analysis=analysis, telemetry=telemetry)
        store.save(report)
        logger.info(
            "call finalized",
            extra={
                "call_id": record.call_id,
                "outcome": analysis.outcome.value,
                "telemetry": telemetry.status,
            },
        )
        return report
