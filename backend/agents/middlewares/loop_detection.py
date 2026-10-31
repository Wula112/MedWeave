"""Bound tool requests and stop repetitive loops through native graph routing."""
from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agents.middlewares.execution_context import execution_context
from agents.middlewares.tool_deduplication import ToolIdentity, call_identity
from agents.middlewares.tool_error_handling import ToolHandler, tool_error
from runtime.errors import RuntimeFailure


class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    def __init__(self, identify: ToolIdentity) -> None:
        self.identify = identify

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        context = execution_context(runtime)
        context.remaining()
        if context.failure:
            # Every rejected tool call already received its paired ToolMessage.
            # create_agent's native conditional edge now ends the graph safely.
            return {"jump_to": "end"}
        return None

    async def awrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolMessage | Command:
        context = execution_context(request.runtime)
        context.remaining()
        if context.failure:
            return tool_error(request, context.failure.code, context.failure.message)
        _, key, _ = call_identity(request, self.identify)
        context.tool_calls += 1
        count = context.call_counts.get(key, 0) + 1
        context.call_counts[key] = count
        if context.tool_calls > context.settings.max_tool_calls or count >= context.settings.repeat_limit:
            code = "loop_detected" if count >= context.settings.repeat_limit else "tool_limit"
            context.failure = RuntimeFailure(code, "工具调用已达到本轮限制，请补充问题后重试。")
            return tool_error(request, code, context.failure.message)
        result = await handler(request)
        if count > 1 and isinstance(result, ToolMessage):
            result = result.model_copy(deep=True)
            content = result.content if isinstance(result.content, str) else json.dumps(result.content)
            result.content = content + "\n本轮已调用过相同工具和参数，请利用已有结果完成回答，避免重复调用。"
        return result
