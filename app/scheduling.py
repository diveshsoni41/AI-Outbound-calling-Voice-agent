"""Simulated appointment scheduling backend.

This stands in for a hospital scheduling API. It is deterministic (slots are
derived from an injectable clock) and persistent (SQLite), so it behaves like a
real system in the ways that matter for the agent:

* a slot can only be booked once (UNIQUE constraint, enforced in a transaction);
* repeating a request with the same idempotency key returns the original
  appointment instead of creating a second one;
* a booking only exists if this module says it does.

Replacing it with a real integration means implementing the same three methods
against the real API.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from app.config import SchedulerSimulation
from app.patient import PATIENT_ID_PATTERN

SLOT_TIMES = (time(9, 30), time(11, 0), time(14, 0), time(16, 30))
SLOT_MINUTES = 30
SCHEDULE_DAYS = 5
CLINICIAN = "General Physician"

PartOfDay = Literal["morning", "afternoon", "any"]


class BookingErrorCode(str, Enum):
    CONFIRMATION_REQUIRED = "confirmation_required"
    INVALID_PATIENT = "invalid_patient"
    SLOT_NOT_FOUND = "slot_not_found"
    SLOT_UNAVAILABLE = "slot_unavailable"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    BACKEND_ERROR = "backend_error"


class SlotView(BaseModel):
    slot_id: str
    start: datetime
    end: datetime
    timezone: str
    clinician: str
    spoken: str


class AvailabilityResult(BaseModel):
    ok: bool
    slots: list[SlotView]
    timezone: str
    error: str | None = None


class Appointment(BaseModel):
    appointment_id: str
    patient_id: str
    slot_id: str
    start: datetime
    timezone: str
    clinician: str
    status: Literal["confirmed"]
    idempotency_key: str
    created_at: datetime


class BookingResult(BaseModel):
    ok: bool
    appointment: Appointment | None = None
    error_code: BookingErrorCode | None = None
    error_message: str | None = None
    # True when an earlier identical request already created this appointment.
    idempotent_replay: bool = False


def _appointment_id(idempotency_key: str) -> str:
    # Derived from the key so a retried request maps to the same ID even if the
    # first response was lost after the write committed.
    return "apt_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:12]


def _spoken_time(start: datetime) -> str:
    hour = start.strftime("%I").lstrip("0")
    minute = "" if start.minute == 0 else f":{start.minute:02d}"
    return f"{start:%A}, {start:%B} {start.day} at {hour}{minute} {start:%p}"


class SchedulingService:
    def __init__(
        self,
        db_path: str | Path,
        timezone: ZoneInfo,
        *,
        clock: Callable[[], datetime] | None = None,
        simulation: SchedulerSimulation = SchedulerSimulation.NONE,
    ) -> None:
        self._tz = timezone
        self._clock = clock or (lambda: datetime.now(self._tz))
        self._simulation = simulation
        self._lock = threading.Lock()
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(db_path), isolation_level=None, check_same_thread=False, timeout=5
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                patient_id TEXT NOT NULL,
                slot_id TEXT NOT NULL UNIQUE,
                start TEXT NOT NULL,
                timezone TEXT NOT NULL,
                clinician TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    def close(self) -> None:
        self._db.close()

    # -- schedule -------------------------------------------------------------

    def _schedule(self) -> dict[str, SlotView]:
        """All slots on the next SCHEDULE_DAYS weekdays, starting tomorrow."""
        now = self._clock().astimezone(self._tz)
        slots: dict[str, SlotView] = {}
        day = now.date()
        while len(slots) < SCHEDULE_DAYS * len(SLOT_TIMES):
            day += timedelta(days=1)
            if day.weekday() >= 5:
                continue
            for slot_time in SLOT_TIMES:
                start = datetime.combine(day, slot_time, tzinfo=self._tz)
                slot_id = f"slot-{start:%Y%m%d-%H%M}"
                slots[slot_id] = SlotView(
                    slot_id=slot_id,
                    start=start,
                    end=start + timedelta(minutes=SLOT_MINUTES),
                    timezone=str(self._tz),
                    clinician=CLINICIAN,
                    spoken=_spoken_time(start),
                )
        return slots

    def _booked_slot_ids(self) -> set[str]:
        rows = self._db.execute("SELECT slot_id FROM appointments WHERE status = 'confirmed'")
        return {row["slot_id"] for row in rows}

    # -- public API -------------------------------------------------------------

    def check_availability(
        self, part_of_day: PartOfDay = "any", limit: int = 3
    ) -> AvailabilityResult:
        tz_name = str(self._tz)
        if self._simulation is SchedulerSimulation.NO_SLOTS:
            return AvailabilityResult(ok=True, slots=[], timezone=tz_name)
        try:
            with self._lock:
                booked = self._booked_slot_ids()
        except sqlite3.Error as exc:
            return AvailabilityResult(ok=False, slots=[], timezone=tz_name, error=str(exc))

        free = [slot for slot_id, slot in self._schedule().items() if slot_id not in booked]
        if part_of_day == "morning":
            free = [slot for slot in free if slot.start.hour < 12]
        elif part_of_day == "afternoon":
            free = [slot for slot in free if slot.start.hour >= 12]
        return AvailabilityResult(ok=True, slots=free[:limit], timezone=tz_name)

    def book_appointment(
        self,
        *,
        patient_id: str,
        slot_id: str,
        idempotency_key: str,
        patient_confirmed: bool,
    ) -> BookingResult:
        if not patient_confirmed:
            return _failure(
                BookingErrorCode.CONFIRMATION_REQUIRED,
                "The patient has not explicitly confirmed this slot.",
            )
        if not PATIENT_ID_PATTERN.match(patient_id):
            return _failure(BookingErrorCode.INVALID_PATIENT, "Unknown or malformed patient ID.")

        with self._lock:
            existing = self._find_by_key(idempotency_key)
            if existing is not None:
                if existing.slot_id != slot_id or existing.patient_id != patient_id:
                    return _failure(
                        BookingErrorCode.IDEMPOTENCY_CONFLICT,
                        "This idempotency key was already used for a different request.",
                    )
                return BookingResult(ok=True, appointment=existing, idempotent_replay=True)

            slot = self._schedule().get(slot_id)
            if slot is None:
                return _failure(
                    BookingErrorCode.SLOT_NOT_FOUND, f"Slot '{slot_id}' is not on the schedule."
                )
            if self._simulation is SchedulerSimulation.BOOKING_FAILURE:
                return _failure(
                    BookingErrorCode.BACKEND_ERROR,
                    "The scheduling system did not accept the booking (simulated outage).",
                )

            appointment = Appointment(
                appointment_id=_appointment_id(idempotency_key),
                patient_id=patient_id,
                slot_id=slot_id,
                start=slot.start,
                timezone=slot.timezone,
                clinician=slot.clinician,
                status="confirmed",
                idempotency_key=idempotency_key,
                created_at=self._clock(),
            )
            try:
                self._insert(appointment)
            except sqlite3.IntegrityError:
                return _failure(BookingErrorCode.SLOT_UNAVAILABLE, "That slot has just been taken.")
            except sqlite3.Error as exc:
                return _failure(BookingErrorCode.BACKEND_ERROR, f"Storage error: {exc}")
        return BookingResult(ok=True, appointment=appointment)

    def get_appointment(self, appointment_id: str) -> Appointment | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
            ).fetchone()
        return _row_to_appointment(row) if row else None

    # -- storage helpers ---------------------------------------------------------

    def _find_by_key(self, idempotency_key: str) -> Appointment | None:
        row = self._db.execute(
            "SELECT * FROM appointments WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return _row_to_appointment(row) if row else None

    def _insert(self, appointment: Appointment) -> None:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute(
                """
                INSERT INTO appointments (appointment_id, idempotency_key, patient_id, slot_id,
                                          start, timezone, clinician, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    appointment.appointment_id,
                    appointment.idempotency_key,
                    appointment.patient_id,
                    appointment.slot_id,
                    appointment.start.isoformat(),
                    appointment.timezone,
                    appointment.clinician,
                    appointment.status,
                    appointment.created_at.isoformat(),
                ),
            )
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")


def _failure(code: BookingErrorCode, message: str) -> BookingResult:
    return BookingResult(ok=False, error_code=code, error_message=message)


def _row_to_appointment(row: sqlite3.Row) -> Appointment:
    return Appointment(
        appointment_id=row["appointment_id"],
        patient_id=row["patient_id"],
        slot_id=row["slot_id"],
        start=datetime.fromisoformat(row["start"]),
        timezone=row["timezone"],
        clinician=row["clinician"],
        status=row["status"],
        idempotency_key=row["idempotency_key"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
