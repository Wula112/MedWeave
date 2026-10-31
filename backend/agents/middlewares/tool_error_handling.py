"""Convert ordinary tool failures to paired, safe ToolMessages."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from agents.middlewares.execution_context import execution_context
from runtime.errors import RuntimeFailure

ToolHandler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]


def tool_error(request: ToolCallRequest, code: str, message: str) -> ToolMessage:
    return ToolMessage(
        name=request.tool_call["name"], tool_call_id=request.tool_call["id"], status="error",
        content=json.dumps({"success": False, "code": code, "error": message}, ensure_ascii=False))


class ToolErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    async def awrap_tool_call(self, request: ToolCallRequest, handler: ToolHandler) -> ToolMessage | Command:
        context = execution_context(request.runtime)
        try:
            async with asyncio.timeout(context.remaining(context.settings.tool_timeout)):
                result = await handler(request)
        except (GraphBubbleUp, RuntimeFailure):
            raise
        except TimeoutError:
            return tool_error(request, "tool_timeout", "工具响应超时，请换一种方式或向用户说明。")
        except Exception:
            return tool_error(request, "tool_failed", "工具调用失败，请检查参数或换一种方式。")
        if isinstance(result, ToolMessage) and result.status == "error":
            # ToolNode also converts validation/unknown-tool errors itself. Its
            # default message can contain raw arguments, so sanitize those too.
            return tool_error(request, "tool_failed", "工具调用失败，请检查参数或换一种方式。")
        return result
