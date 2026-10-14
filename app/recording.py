"""Call recording storage.

LiveKit Agents' AgentSession records both sides of the conversation to
`<job session directory>/audio.ogg` when recording is enabled (RecorderIO).
That directory is temporary, so after the session closes we move the file into
the call's artifact folder and keep only a reference to it.

Local disk is the right destination for development. In production this is the
place to upload to access-controlled object storage (or switch to LiveKit
Egress) and return that object's URI instead.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import RecordingReference

logger = logging.getLogger("healthcare_agent.recording")


def store_recording(
    source: Path | None,
    destination_dir: Path,
    *,
    started_at: float | None = None,
) -> RecordingReference:
    if source is None:
        return RecordingReference(
            status="not_available",
            reason="Recording was not enabled for this session.",
        )
    if not source.is_file() or source.stat().st_size == 0:
        return RecordingReference(
            status="not_available",
            reason="The session produced no audio (for example, the call was never answered).",
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"recording{source.suffix}"
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        logger.error("could not store recording: %s", exc)
        return RecordingReference(status="failed", reason=f"Could not copy recording: {exc}")

    return RecordingReference(
        status="available",
        uri=str(destination.resolve()),
        content_type="audio/ogg" if source.suffix == ".ogg" else None,
        size_bytes=destination.stat().st_size,
        started_at=datetime.fromtimestamp(started_at, tz=timezone.utc) if started_at else None,
    )
