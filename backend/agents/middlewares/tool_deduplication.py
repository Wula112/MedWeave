"""Run-scoped result reuse and single-flight for explicitly read-only tools."""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agents.middlewares.execution_context import execution_context
from agents.middlewares.tool_error_handling import ToolHandler

ToolIdentity = Callable[[ToolCallRequest], tuple[str, dict, bool]]


def call_identity(request: ToolCallRequest, identify: ToolIdentity) -> tuple[str, str, bool]:
    name, arguments, reusable = identify(request)
    key = hashlib.sha256(json.dumps([name, arguments], sort_keys=True, ensure_ascii=False,
                                    default=str, separators=(",", ":")).encode()).hexdigest()
    return name, key, reusable


class ToolDeduplicationMiddleware(AgentMiddleware[AgentState]):
    def __init__(self, identify: ToolIdentity) -> None:
        self.identify = identify

    async def awrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolMessage | Command:
        context = execution_context(request.runtime)
        name, key, reusable = call_identity(request, self.identify)
        cached = key in context.tool_results or key in context.tool_failures
        if key in context.tool_results:
            result = context.tool_results[key]
        elif key in context.tool_failures:
            # A timed-out synchronous operation may still be running. Reusing
            # its error is distinct from treating it as a successful result.
            result = context.tool_failures[key]
        elif reusable:
            task = context.tool_tasks.get(key)
            if task is None:
                task = asyncio.create_task(handler(request))
                context.tool_tasks[key] = task
            else:
                cached = True
            try:
                result = await task
            finally:
                if task.done():
                    context.tool_tasks.pop(key, None)
        else:
            result = await handler(request)

        if isinstance(result, ToolMessage):
            result = result.model_copy(deep=True, update={
                "tool_call_id": request.tool_call["id"], "name": request.tool_call["name"],
            })
            successful = result.status != "error"
            try:
                payload = json.loads(result.content) if isinstance(result.content, str) else None
                if isinstance(payload, dict) and payload.get("success") is False:
                    successful = False
            except (ValueError, TypeError):
                pass
            if not successful:
                result.status = "error"
                context.tool_failures[key] = result.model_copy(deep=True)
            elif reusable:
                context.tool_results[key] = result.model_copy(deep=True)
            result.additional_kwargs.update({"runtime_cached": cached, "runtime_key": key, "runtime_skill": name})
        return result
