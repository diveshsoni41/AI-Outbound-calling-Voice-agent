"""Opik online metric: does what the agent said about booking match the backend?

This file is uploaded verbatim as an Opik "Python code" online evaluation rule
(see scripts/setup_opik_rules.py), so it must stay self-contained: standard
library plus `opik` only, and exactly one BaseMetric subclass.

Opik maps the trace's `output` field onto the `output` argument. Depending on
the Opik version it arrives as a dict or as a JSON string, so both are handled.

Scoring:
* backend booked, agent's last booking statement was not a failure  -> 1.0
* backend booked, agent's last booking statement said it failed     -> 0.0
* backend not booked, agent ever claimed a booking was made          -> 0.0
* backend not booked, agent never claimed a booking                  -> 1.0

A false "you're booked" is the costly error in this domain, so a false claim
fails the call even if the agent later corrected itself.
"""

import json
import re
from typing import Any, Optional

from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult

CLAIMED_BOOKED = "claimed_booked"
CLAIMED_NOT_BOOKED = "claimed_not_booked"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Statements that a booking now exists. Past/perfect tense only, so offers such
# as "I'll book that" or "shall I book it?" are not claims.
_POSITIVE = [
    re.compile(
        r"\b(?:i|we)(?:'ve| have)\s+(?:\w+\s+){0,2}(?:booked|scheduled|confirmed|reserved)\b"
    ),
    re.compile(
        r"\b(?:i|we)\s+(?:\w+\s+){0,1}(?:booked|scheduled|reserved)\s+(?:you|your|that|it|the)\b"
    ),
    re.compile(
        r"\b(?:appointment|consultation|visit|booking)\b[^.!?]{0,40}?\b(?:is|has been|was|are)\s+"
        r"(?:now\s+|all\s+)?(?:booked|confirmed|scheduled|set)\b"
    ),
    re.compile(r"\byou(?:'re| are)\s+(?:all\s+)?(?:set|booked|confirmed|scheduled)\b"),
]

# Statements that the booking did not happen.
_NEGATIVE = [
    re.compile(
        r"\b(?:couldn't|could not|can't|cannot|wasn't able to|was not able to|unable to|"
        r"not able to|didn't manage to)\s+(?:\w+\s+){0,2}"
        r"(?:book|schedule|complete|confirm|reserve|finish)"
    ),
    re.compile(
        r"\b(?:booking|appointment|reservation)\b[^.!?]{0,40}?\b(?:failed|didn't go through|"
        r"did not go through|wasn't successful|was not successful|wasn't completed|"
        r"was not completed|isn't confirmed|is not confirmed|isn't booked|is not booked)"
    ),
    re.compile(r"\b(?:no|not any)\s+(?:available\s+)?(?:slots|appointments|openings)\b"),
]

_NEGATION = re.compile(
    r"\b(?:not|never|unable|failed|unfortunately|couldn't|wasn't|isn't|didn't|cannot|can't)\b"
)


def classify_statement(sentence: str) -> Optional[str]:
    """Classify one sentence as a booking claim, a non-booking claim, or neither."""
    text = sentence.strip().lower().replace("’", "'")
    if not text or text.endswith("?"):
        return None
    if any(pattern.search(text) for pattern in _NEGATIVE):
        return CLAIMED_NOT_BOOKED
    if any(pattern.search(text) for pattern in _POSITIVE):
        return CLAIMED_NOT_BOOKED if _NEGATION.search(text) else CLAIMED_BOOKED
    return None


def extract_booking_claims(agent_texts: list) -> list:
    """Return every booking-related claim in order, with the sentence it came from."""
    claims = []
    for text in agent_texts:
        for sentence in _SENTENCE_SPLIT.split(str(text or "")):
            claim = classify_statement(sentence)
            if claim:
                claims.append({"claim": claim, "sentence": sentence.strip()})
    return claims


def evaluate_booking_claim(booking_status: str, agent_texts: list) -> dict:
    """Compare agent statements with the authoritative booking status.

    booking_status is the scheduling backend's verdict: "booked", "failed" or
    "not_attempted". Returns {"value", "reason", "claims"}.
    """
    claims = extract_booking_claims(agent_texts)
    booked = booking_status == "booked"
    positive = [c for c in claims if c["claim"] == CLAIMED_BOOKED]

    if booked:
        if claims and claims[-1]["claim"] == CLAIMED_NOT_BOOKED:
            return {
                "value": 0.0,
                "reason": "Backend confirmed the booking but the agent told the patient it "
                f'did not happen: "{claims[-1]["sentence"]}"',
                "claims": claims,
            }
        if positive:
            reason = (
                f'Backend confirmed the booking and the agent said so: "{positive[-1]["sentence"]}"'
            )
        else:
            reason = (
                "Backend confirmed the booking; the agent made no contradicting statement "
                "(for example, the call ended before confirmation was spoken)."
            )
        return {"value": 1.0, "reason": reason, "claims": claims}

    if positive:
        return {
            "value": 0.0,
            "reason": f"Backend status is '{booking_status}' but the agent claimed a booking: "
            f'"{positive[0]["sentence"]}"',
            "claims": claims,
        }
    if claims:
        reason = (
            f"Backend status is '{booking_status}' and the agent correctly said no booking was "
            f'made: "{claims[-1]["sentence"]}"'
        )
    else:
        reason = f"Backend status is '{booking_status}' and the agent never claimed a booking."
    return {"value": 1.0, "reason": reason, "claims": claims}


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class BookingClaimAccuracy(BaseMetric):
    def __init__(self, name: str = "booking_claim_accuracy", track: bool = False):
        super().__init__(name=name, track=track)

    def score(self, output: Any = None, **ignored_kwargs: Any) -> ScoreResult:
        data = _as_dict(output)
        booking = data.get("booking") or {}
        status = booking.get("status") if isinstance(booking, dict) else None
        transcript = data.get("transcript")
        if status not in ("booked", "failed", "not_attempted") or not isinstance(transcript, list):
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="Trace output lacks booking.status or transcript; cannot evaluate.",
                scoring_failed=True,
            )
        agent_texts = [
            turn.get("text", "")
            for turn in transcript
            if isinstance(turn, dict) and turn.get("speaker") == "agent"
        ]
        result = evaluate_booking_claim(status, agent_texts)
        return ScoreResult(name=self.name, value=result["value"], reason=result["reason"])
