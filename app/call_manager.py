"""Starting an outbound call.

Order of operations (from LiveKit's outbound calling guide):

1. This module validates the request and creates an *explicit agent dispatch*
   for a fresh room, passing the call ID and patient data as job metadata.
   Dispatch metadata travels over the authenticated LiveKit server API; the
   room name contains only the call ID, never a phone number or health data.
2. LiveKit assigns the job to a running worker (app/agent.py), which joins
   the room and dials the patient with CreateSIPParticipant through the
   configured outbound SIP trunk, waiting until the call is answered.
3. The worker runs the conversation and, when the session ends, the post-call
   pipeline writes artifacts/calls/<call_id>/call_report.json.

So "placing a call" here means dispatching the agent; the dialing happens in
the worker, which is where the call's lifecycle can be observed end to end.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from livekit import api
from pydantic import BaseModel

from app.call_state import new_call_id
from app.config import Requirement, Settings
from app.patient import Patient, mask_phone
from app.pipeline import CallReport, CallReportStore

logger = logging.getLogger("healthcare_agent.call_manager")


class CallJob(BaseModel):
    """Job metadata handed from the call manager to the agent worker."""

    call_id: str
    patient: Patient


class CallInitiationError(RuntimeError):
    pass


class UnauthorizedDestinationError(CallInitiationError):
    pass


@dataclass(frozen=True)
class CallHandle:
    call_id: str
    room_name: str
    dispatch_id: str


def room_name_for(call_id: str) -> str:
    return f"outbound-{call_id}"


def ensure_authorized_destination(settings: Settings, phone_number: str) -> None:
    if phone_number not in settings.authorized_test_numbers:
        raise UnauthorizedDestinationError(
            f"Refusing to dial {mask_phone(phone_number)}: it is not listed in "
            "AUTHORIZED_TEST_NUMBERS."
        )


class OutboundCallManager:
    def __init__(
        self,
        settings: Settings,
        *,
        api_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._api_factory = api_factory or self._default_api

    def _default_api(self) -> api.LiveKitAPI:
        s = self._settings
        return api.LiveKitAPI(
            url=s.livekit_url,
            api_key=s.livekit_api_key.get_secret_value() if s.livekit_api_key else None,
            api_secret=s.livekit_api_secret.get_secret_value() if s.livekit_api_secret else None,
        )

    async def place_call(self, patient: Patient) -> CallHandle:
        """Validate, then dispatch the agent that will dial the patient."""
        self._settings.require(Requirement.LIVEKIT, Requirement.TELEPHONY)
        ensure_authorized_destination(self._settings, patient.phone_number)

        call_id = new_call_id()
        room_name = room_name_for(call_id)
        metadata = CallJob(call_id=call_id, patient=patient).model_dump_json()

        try:
            async with self._api_factory() as lkapi:
                dispatch = await lkapi.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(
                        agent_name=self._settings.agent_name,
                        room=room_name,
                        metadata=metadata,
                    )
                )
        except api.TwirpError as exc:
            raise CallInitiationError(f"Agent dispatch failed ({exc.code}): {exc.message}") from exc
        except OSError as exc:  # DNS / connection errors from aiohttp subclass OSError
            raise CallInitiationError(f"Could not reach LiveKit: {exc}") from exc

        logger.info(
            "agent dispatched",
            extra={"call_id": call_id, "room": room_name, "dispatch_id": dispatch.id},
        )
        return CallHandle(call_id=call_id, room_name=room_name, dispatch_id=dispatch.id)


async def wait_for_report(
    store: CallReportStore,
    call_id: str,
    *,
    timeout_seconds: float,
    poll_interval: float = 2.0,
) -> CallReport | None:
    """Wait for the worker to write the call report.

    This reads the local artifacts folder, so it only works when the worker
    runs on the same machine (the normal development setup).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        report = store.load(call_id)
        if report is not None and report.telemetry.status != "pending":
            return report
        await asyncio.sleep(poll_interval)
    return store.load(call_id)
