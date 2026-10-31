"""Ephemeral execution state; deliberately kept out of LangGraph checkpoints."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from runtime.config import RuntimeSettings
from runtime.errors import RuntimeFailure


@dataclass
class RunContext:
    run_id: str
    thread_id: str
    channel: str
    deadline: float
    settings: RuntimeSettings
    mark_graph: Callable[[], Awaitable[None]]
    graph_started: bool = False
    ownership_lost: bool = False
    waiting_input: bool = False
    tool_calls: int = 0
    call_counts: dict[str, int] = field(default_factory=dict)
    tool_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    tool_results: dict[str, Any] = field(default_factory=dict)
    tool_failures: dict[str, Any] = field(default_factory=dict)
    resources: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    failure: RuntimeFailure | None = None

    def remaining(self, limit: float | None = None) -> float:
        if self.ownership_lost:
            raise RuntimeFailure("ownership_lost", "本次运行已停止，请稍后重试。", retryable=True)
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeFailure("run_timeout", "本次处理超时，请稍后重试。", retryable=True)
        return min(remaining, limit) if limit is not None else remaining

    async def begin_graph(self) -> None:
        self.remaining()
        await self.mark_graph()
        self.graph_started = True


current_run: ContextVar[RunContext | None] = ContextVar("health_run", default=None)


def operation_timeout(limit: float) -> float:
    context = current_run.get()
    return context.remaining(limit) if context else limit
