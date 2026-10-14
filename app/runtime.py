"""Builds the shared services from settings (used by the worker and the demo)."""

from __future__ import annotations

import logging

from app import __version__
from app.config import Settings
from app.observability.opik_integration import OpikCallObserver
from app.pipeline import CallReportStore
from app.post_call import OpenAIQualitativeAnalyzer, PostCallAnalyzer
from app.prompts import PROMPT_VERSION
from app.scheduling import SchedulingService
from app.schemas import AgentInfo

logger = logging.getLogger("healthcare_agent")


def agent_info(
    settings: Settings, *, llm_model: str | None = None, voice: bool = True
) -> AgentInfo:
    return AgentInfo(
        agent_name=settings.agent_name,
        agent_version=__version__,
        llm_model=llm_model or settings.llm_model,
        stt_model=settings.stt_model if voice else None,
        tts_model=settings.tts_model if voice else None,
        tts_voice=settings.tts_voice if voice else None,
        prompt_version=PROMPT_VERSION,
    )


def build_scheduler(settings: Settings) -> SchedulingService:
    return SchedulingService(
        settings.appointments_db_path,
        settings.timezone,
        simulation=settings.scheduler_simulation,
    )


def build_analyzer(settings: Settings) -> PostCallAnalyzer:
    if not settings.post_call_llm_enabled:
        return PostCallAnalyzer()
    if settings.openai_api_key is None:
        logger.warning("OPENAI_API_KEY not set; post-call analysis will skip the LLM summary")
        return PostCallAnalyzer()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    return PostCallAnalyzer(OpenAIQualitativeAnalyzer(client, settings.analysis_model))


def build_observer(settings: Settings) -> OpikCallObserver | None:
    observer = OpikCallObserver.from_settings(settings)
    if observer is None:
        logger.info("Opik is not configured; traces will not be sent")
    return observer


def build_store(settings: Settings) -> CallReportStore:
    return CallReportStore(settings.artifacts_dir)
