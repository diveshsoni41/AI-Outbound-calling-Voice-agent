"""Turn a LiveKit chat history into an ordered, speaker-labelled transcript."""

from __future__ import annotations

from datetime import datetime, timezone

from livekit.agents import llm

from app.schemas import TranscriptTurn

_SPEAKERS = {"assistant": "agent", "user": "patient"}


def transcript_from_chat_context(chat_ctx: llm.ChatContext) -> list[TranscriptTurn]:
    """Extract spoken turns from the session history.

    System/developer messages, tool calls and handoff markers are skipped: tool
    activity is recorded separately (and authoritatively) by the tools
    themselves. Items are de-duplicated by ID and ordered by creation time, so
    repeated or out-of-order delivery does not corrupt the transcript. Empty
    messages are dropped rather than replaced with anything.
    """
    seen: set[str] = set()
    turns: list[tuple[float, int, TranscriptTurn]] = []
    for position, item in enumerate(chat_ctx.items):
        if item.type != "message" or item.role not in _SPEAKERS or item.id in seen:
            continue
        seen.add(item.id)
        text = (item.text_content or "").strip()
        if not text:
            continue
        turns.append(
            (
                item.created_at,
                position,
                TranscriptTurn(
                    speaker=_SPEAKERS[item.role],
                    text=text,
                    timestamp=datetime.fromtimestamp(item.created_at, tz=timezone.utc),
                    interrupted=bool(item.interrupted),
                ),
            )
        )
    turns.sort(key=lambda entry: (entry[0], entry[1]))
    return [turn for _, _, turn in turns]


def format_transcript(turns: list[TranscriptTurn]) -> str:
    lines = []
    for turn in turns:
        label = "Agent" if turn.speaker == "agent" else "Patient"
        suffix = " [interrupted]" if turn.interrupted else ""
        lines.append(f"{label}: {turn.text}{suffix}")
    return "\n".join(lines)
