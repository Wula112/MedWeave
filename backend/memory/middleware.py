"""Capture one completed Run and submit its latest turn to Mem0 in the background."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from memory.prompts import CHINA_TIMEZONE, instructions_for

logger = logging.getLogger(__name__)

_AGENT_BY_NODE = {
    "advisor_node": "advisor_agent",
    "clinic_node": "clinic_agent",
    "insurance_node": "insurance_agent",
    "pharmacy_node": "pharmacy_agent",
    "report_node": "report_agent",
    # A resumed clinic graph may begin at a child node rather than clinic_node.
    "classify_reply": "clinic_agent",
    "emergency_gate": "clinic_agent",
    "extract_symptoms": "clinic_agent",
    "check_sufficiency": "clinic_agent",
    "ask_followup": "clinic_agent",
    "conclude": "clinic_agent",
}


@dataclass
class _Turn:
    user_text: str = field(repr=False)
    agent_name: str
    observed_at: datetime = field(default_factory=lambda: datetime.now(CHINA_TIMEZONE))
    assistant_chunks: list[str] = field(default_factory=list, repr=False)


class MemoryMiddleware:
    """Run-level hooks; the parent graph can interrupt or call several agents."""

    def __init__(self, api_key: str, user_id: str, data_dir: Path) -> None:
        self._api_key = api_key
        self._user_id = user_id
        self._data_dir = data_dir
        self._turns: dict[str, _Turn] = {}
        self._jobs: set[asyncio.Task] = set()
        self._limit = asyncio.Semaphore(4)

    @classmethod
    def from_env(cls, data_dir: Path) -> MemoryMiddleware | None:
        key = os.getenv("MEM0_API_KEY", "").strip()
        if not key:
            return None
        user_id = os.getenv("MEM0_USER_ID", "smart_health_user").strip()
        if not user_id:
            logger.warning("Mem0 capture disabled: MEM0_USER_ID is empty")
            return None
        return cls(key, user_id, data_dir)

    def before_run(self, run_id: str, user_text: str, agent_name: str) -> None:
        if user_text.strip():
            self._turns[run_id] = _Turn(user_text=user_text, agent_name=agent_name)

    def on_event(self, run_id: str, event: dict[str, Any]) -> None:
        turn = self._turns.get(run_id)
        if turn is None:
            return
        if event.get("type") == "node_start":
            turn.agent_name = _AGENT_BY_NODE.get(event.get("node"), turn.agent_name)
        elif event.get("type") == "text" and isinstance(event.get("content"), str):
            turn.assistant_chunks.append(event["content"])

    def after_run(self, run_id: str, status: str) -> None:
        turn = self._turns.pop(run_id, None)
        if turn is None or status not in {"success", "waiting_input"}:
            return
        assistant_text = "".join(turn.assistant_chunks).strip()
        if not assistant_text or not turn.agent_name:
            return
        messages = [
            {"role": "user", "content": turn.user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        job = asyncio.create_task(self._submit(run_id, turn.agent_name, messages, turn.observed_at))
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)

    async def _submit(self, run_id: str, agent_name: str, messages: list[dict[str, str]],
                      observed_at: datetime) -> None:
        try:
            async with self._limit:
                result = await asyncio.to_thread(self._add, agent_name, messages, observed_at)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never log the conversation, provider error body, or credential.
            logger.warning("Mem0 submission failed run_id=%s error_type=%s", run_id, type(exc).__name__)
        else:
            # The platform processes an accepted add asynchronously.
            try:
                event_id = str(UUID(str(result.get("event_id")))) if isinstance(result, dict) else None
            except (ValueError, TypeError, AttributeError):
                event_id = None
            logger.info("Mem0 submission accepted run_id=%s event_id=%s", run_id, event_id)

    def _add(self, agent_name: str, messages: list[dict[str, str]], observed_at: datetime) -> dict[str, Any]:
        # SDK import initializes its local config; keep it with other runtime data.
        os.environ.setdefault("MEM0_DIR", str(self._data_dir / "mem0"))
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        from mem0 import MemoryClient

        # MemoryClient validates its key synchronously during construction.
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as transport:
            client = MemoryClient(api_key=self._api_key, client=transport)
            result = client.add(
                messages,
                user_id=self._user_id,
                agent_id=agent_name,
                custom_instructions=instructions_for(observed_at),
                timezone="Asia/Shanghai",
                observation_datetime=observed_at.isoformat(),
            )
        if result.get("status") == "FAILED":
            raise RuntimeError("Mem0 rejected the submission")
        return result

    async def shutdown(self, timeout: float) -> None:
        if not self._jobs:
            return
        _, pending = await asyncio.wait(self._jobs, timeout=timeout)
        for job in pending:
            job.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning("Mem0 submissions did not finish before shutdown: count=%d", len(pending))
