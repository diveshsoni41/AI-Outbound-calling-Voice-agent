"""Application configuration.

Settings come from environment variables (and a local `.env` file in development).
Loading settings never requires credentials: each entry point calls
`Settings.require(...)` for the integrations it actually needs, so importing the
package or running offline tests works without any accounts configured.
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")


class ConfigurationError(RuntimeError):
    """Raised when a required setting is missing or invalid.

    Messages name the environment variables involved, never their values.
    """


class Requirement(str, Enum):
    LIVEKIT = "livekit"
    TELEPHONY = "telephony"
    OPENAI = "openai"
    OPIK = "opik"


class SchedulerSimulation(str, Enum):
    """Failure modes the simulated scheduling backend can be put into for demos."""

    NONE = "none"
    NO_SLOTS = "no_slots"
    BOOKING_FAILURE = "booking_failure"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LiveKit server (the LiveKit SDKs read the same variable names).
    livekit_url: str | None = None
    livekit_api_key: SecretStr | None = None
    livekit_api_secret: SecretStr | None = None
    agent_name: str = "healthcare-outbound-agent"

    # Telephony
    sip_outbound_trunk_id: str | None = None
    # Outbound calls are refused unless the destination is on this list. This keeps
    # development and demo runs from ever dialing an arbitrary real person.
    # repr=False: real phone numbers must not appear in tracebacks or test output.
    authorized_test_numbers: Annotated[list[str], NoDecode] = Field(
        default_factory=list, repr=False
    )
    ring_timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_call_duration_seconds: int = Field(default=600, ge=60, le=3600)

    # Model providers (one OpenAI key covers STT, LLM, TTS and post-call analysis).
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4.1-mini"
    stt_model: str = "gpt-4o-mini-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "coral"
    analysis_model: str = "gpt-4.1-mini"
    post_call_llm_enabled: bool = True

    # Opik (the Opik SDK reads OPIK_API_KEY / OPIK_WORKSPACE / OPIK_URL_OVERRIDE too).
    opik_enabled: bool = True
    opik_api_key: SecretStr | None = None
    opik_workspace: str | None = None
    opik_project_name: str = "healthcare-outbound-calls"
    opik_url_override: str | None = None
    opik_flush_timeout_seconds: int = Field(default=30, ge=1, le=300)

    # Conversation
    organization_name: str | None = None
    clinic_timezone: str = "America/New_York"
    emergency_number: str = "911"

    # Local storage
    artifacts_dir: Path = Path("artifacts")
    appointments_db_path: Path = Path("artifacts/appointments.sqlite3")
    record_audio: bool = True
    scheduler_simulation: SchedulerSimulation = SchedulerSimulation.NONE

    @field_validator("authorized_test_numbers", mode="before")
    @classmethod
    def _split_numbers(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("authorized_test_numbers")
    @classmethod
    def _validate_numbers(cls, numbers: list[str]) -> list[str]:
        for index, number in enumerate(numbers):
            if not E164_PATTERN.match(number):
                raise ValueError(
                    f"AUTHORIZED_TEST_NUMBERS entry #{index + 1} is not an E.164 number "
                    "(expected a format like +15555550123)"
                )
        return numbers

    @field_validator("clinic_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"CLINIC_TIMEZONE '{value}' is not a valid IANA timezone") from exc
        return value

    @field_validator("livekit_url")
    @classmethod
    def _validate_livekit_url(cls, value: str | None) -> str | None:
        if value and not value.startswith(("ws://", "wss://", "http://", "https://")):
            raise ValueError("LIVEKIT_URL must start with wss://, ws://, https:// or http://")
        return value

    @field_validator("sip_outbound_trunk_id")
    @classmethod
    def _validate_trunk(cls, value: str | None) -> str | None:
        if value and not value.startswith("ST_"):
            raise ValueError("SIP_OUTBOUND_TRUNK_ID should look like ST_xxxxxxxx")
        return value

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.clinic_timezone)

    @property
    def opik_configured(self) -> bool:
        """Opik needs either a Cloud API key or a self-hosted URL."""
        return self.opik_enabled and bool(self.opik_api_key or self.opik_url_override)

    def missing(self, *requirements: Requirement) -> list[str]:
        """Return the environment variable names that a set of integrations still needs."""
        missing: list[str] = []
        for requirement in requirements:
            if requirement is Requirement.LIVEKIT:
                missing += _unset(
                    LIVEKIT_URL=self.livekit_url,
                    LIVEKIT_API_KEY=self.livekit_api_key,
                    LIVEKIT_API_SECRET=self.livekit_api_secret,
                )
            elif requirement is Requirement.TELEPHONY:
                missing += _unset(
                    SIP_OUTBOUND_TRUNK_ID=self.sip_outbound_trunk_id,
                    AUTHORIZED_TEST_NUMBERS=self.authorized_test_numbers,
                )
            elif requirement is Requirement.OPENAI:
                missing += _unset(OPENAI_API_KEY=self.openai_api_key)
            elif requirement is Requirement.OPIK and not self.opik_configured:
                missing.append("OPIK_API_KEY (or OPIK_URL_OVERRIDE for self-hosted Opik)")
        return missing

    def require(self, *requirements: Requirement) -> None:
        missing = self.missing(*requirements)
        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(
                f"Missing required configuration: {names}. "
                "Set these in your environment or .env file (see .env.example)."
            )


def _unset(**values: object) -> list[str]:
    return [name for name, value in values.items() if not value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process. Invalid values raise ConfigurationError."""
    from dotenv import load_dotenv
    from pydantic import ValidationError

    # The LiveKit, OpenAI and Opik SDKs read their credentials from os.environ
    # directly, so export .env there too. Real environment variables win.
    load_dotenv(override=False)
    try:
        return Settings()
    except ValidationError as exc:
        # Pydantic includes the offending input in its message; keep only the
        # field names and our own validator text so secrets never reach logs.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']).upper()}: {error['msg']}"
            for error in exc.errors(include_input=False)
        )
        raise ConfigurationError(f"Invalid configuration: {problems}") from None
