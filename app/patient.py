"""Patient input model and biomarker interpretation.

Everything the agent says about a measurement comes from `describe_biomarker`,
which uses fixed, commonly published reference ranges and never draws a
diagnostic conclusion. The LLM only rephrases this text for speech.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import phonenumbers
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import E164_PATTERN

PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


class BiomarkerType(str, Enum):
    BLOOD_GLUCOSE = "blood_glucose"
    HBA1C = "hba1c"


GlucoseContext = Literal["fasting", "random", "post_meal", "unspecified"]


@dataclass(frozen=True)
class UnitSpec:
    unit: str
    spoken: str
    plausible_min: float
    plausible_max: float


# Plausible ranges reject typos and unit mix-ups (e.g. an HbA1c of 82), not
# clinically unusual values.
SUPPORTED_UNITS: dict[BiomarkerType, dict[str, UnitSpec]] = {
    BiomarkerType.BLOOD_GLUCOSE: {
        "mg/dL": UnitSpec("mg/dL", "milligrams per deciliter", 10, 1000),
        "mmol/L": UnitSpec("mmol/L", "millimoles per liter", 0.5, 55),
    },
    BiomarkerType.HBA1C: {
        "%": UnitSpec("%", "percent", 3, 20),
        "mmol/mol": UnitSpec("mmol/mol", "millimoles per mole", 9, 190),
    },
}

DISPLAY_NAMES = {
    BiomarkerType.BLOOD_GLUCOSE: "blood glucose",
    BiomarkerType.HBA1C: "HbA1c",
}

EXPLANATIONS = {
    BiomarkerType.BLOOD_GLUCOSE: "",
    BiomarkerType.HBA1C: "HbA1c reflects your average blood sugar over roughly the past three months.",
}


class Biomarker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    unit: str
    measured_at: datetime
    # Only meaningful for blood glucose. "unspecified" is an explicit statement that
    # the source did not say, and changes how the value is described.
    context: GlucoseContext | None = None

    @field_validator("measured_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("measured_at must include a timezone, e.g. 2026-10-01T09:00:00Z")
        if value > datetime.now(timezone.utc):
            raise ValueError("measured_at cannot be in the future")
        return value


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_id: str = Field(description="Pseudonymous identifier used in logs and traces")
    name: str = Field(min_length=1, max_length=100)
    phone_number: str
    biomarkers: dict[BiomarkerType, Biomarker] = Field(min_length=1)

    @field_validator("patient_id")
    @classmethod
    def _validate_patient_id(cls, value: str) -> str:
        if not PATIENT_ID_PATTERN.match(value):
            raise ValueError("patient_id must be 3-64 characters of letters, digits, '-' or '_'")
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("phone_number")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        value = value.strip()
        if not E164_PATTERN.match(value):
            raise ValueError("phone_number must be in E.164 format, e.g. +15555550123")
        # is_possible_number checks length and country code. We deliberately do not
        # require is_valid_number, which rejects reserved fictional ranges such as
        # 555-01xx that are exactly what development data should use.
        if not phonenumbers.is_possible_number(phonenumbers.parse(value, None)):
            raise ValueError("phone_number is not a possible telephone number")
        return value

    @model_validator(mode="after")
    def _validate_biomarkers(self) -> Patient:
        for kind, marker in self.biomarkers.items():
            units = SUPPORTED_UNITS[kind]
            spec = units.get(marker.unit)
            if spec is None:
                allowed = ", ".join(units)
                raise ValueError(f"{kind.value}: unsupported unit '{marker.unit}' (use {allowed})")
            if not spec.plausible_min <= marker.value <= spec.plausible_max:
                raise ValueError(
                    f"{kind.value}: value {marker.value} {marker.unit} is outside the plausible "
                    f"range {spec.plausible_min}-{spec.plausible_max}"
                )
            if kind is BiomarkerType.BLOOD_GLUCOSE and marker.context is None:
                raise ValueError(
                    "blood_glucose.context is required (fasting, random, post_meal or unspecified)"
                )
            if kind is not BiomarkerType.BLOOD_GLUCOSE and marker.context is not None:
                raise ValueError(f"{kind.value}: context only applies to blood_glucose")
        return self

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    def telemetry_view(self) -> dict[str, Any]:
        """Patient variables that are safe to send to logs and Opik.

        The name and full phone number are excluded; the pseudonymous patient_id is
        the join key. Biomarkers are included because the assignment asks for call
        variables to be traced. Demo data is synthetic (see docs/architecture.md).
        """
        return {
            "patient_id": self.patient_id,
            "phone_number_masked": mask_phone(self.phone_number),
            "biomarkers": {
                kind.value: {
                    "value": marker.value,
                    "unit": marker.unit,
                    "measured_at": marker.measured_at.isoformat(),
                    **({"context": marker.context} if marker.context else {}),
                }
                for kind, marker in self.biomarkers.items()
            },
        }


def mask_phone(number: str) -> str:
    """Keep the country code and last two digits: +1********23."""
    if len(number) <= 4:
        return "*" * len(number)
    country_len = 2 if number.startswith("+1") else 3
    return number[:country_len] + "*" * (len(number) - country_len - 2) + number[-2:]


def load_patient(path: str | Path) -> Patient:
    """Load and validate a patient JSON file. Raises pydantic.ValidationError."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Patient.model_validate(data)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

Level = Literal["below", "within", "above", "well_above", "needs_context"]


@dataclass(frozen=True)
class BiomarkerSummary:
    kind: BiomarkerType
    level: Level
    spoken_text: str


def _hba1c_percent(marker: Biomarker) -> float:
    if marker.unit == "%":
        return marker.value
    # IFCC (mmol/mol) to NGSP (%) master equation.
    return marker.value / 10.929 + 2.15


def _glucose_mg_dl(marker: Biomarker) -> float:
    return marker.value if marker.unit == "mg/dL" else marker.value * 18.016


def _classify(kind: BiomarkerType, marker: Biomarker) -> Level:
    if kind is BiomarkerType.HBA1C:
        percent = _hba1c_percent(marker)
        if percent < 5.7:
            return "within"
        return "above" if percent < 6.5 else "well_above"

    mg_dl = _glucose_mg_dl(marker)
    if mg_dl < 70:
        return "below"
    if marker.context == "fasting":
        if mg_dl < 100:
            return "within"
        return "above" if mg_dl < 126 else "well_above"
    # Without a fasting reading, only clearly high values can be described.
    return "well_above" if mg_dl >= 200 else "needs_context"


_REFERENCE_TEXT = {
    BiomarkerType.HBA1C: "for HbA1c, which is below 5.7 percent",
    BiomarkerType.BLOOD_GLUCOSE: (
        "for fasting blood glucose, which is 70 to 99 milligrams per deciliter"
    ),
}

_LEVEL_TEXT = {
    "below": "That is below the commonly used reference range",
    "within": "That is within the commonly used reference range",
    "above": "That is above the commonly used reference range",
    "well_above": "That is well above the commonly used reference range",
}


def _format_value(value: float) -> str:
    return f"{value:g}"


def describe_biomarker(kind: BiomarkerType, marker: Biomarker) -> BiomarkerSummary:
    spec = SUPPORTED_UNITS[kind][marker.unit]
    measured = marker.measured_at
    date_text = f"{measured:%B} {measured.day}, {measured.year}"
    level = _classify(kind, marker)

    context_text = ""
    if kind is BiomarkerType.BLOOD_GLUCOSE:
        context_text = {
            "fasting": " fasting",
            "random": " non-fasting",
            "post_meal": " after-meal",
            "unspecified": "",
        }[marker.context or "unspecified"]

    parts = [
        f"Your{context_text} {DISPLAY_NAMES[kind]} reading from {date_text} was "
        f"{_format_value(marker.value)} {spec.spoken}."
    ]
    if EXPLANATIONS[kind]:
        parts.append(EXPLANATIONS[kind])
    if level == "needs_context":
        parts.append(
            "Because it was not a fasting measurement, it can't be compared with the fasting "
            "reference range, so a doctor is the right person to interpret it."
        )
    elif level == "below" and kind is BiomarkerType.BLOOD_GLUCOSE:
        parts.append("That is below the commonly used lower limit of 70 milligrams per deciliter.")
    else:
        parts.append(f"{_LEVEL_TEXT[level]} {_REFERENCE_TEXT[kind]}.")
    return BiomarkerSummary(kind=kind, level=level, spoken_text=" ".join(parts))


def summarize_biomarkers(patient: Patient) -> list[BiomarkerSummary]:
    return [describe_biomarker(kind, marker) for kind, marker in patient.biomarkers.items()]
