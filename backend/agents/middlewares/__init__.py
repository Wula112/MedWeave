"""The sole composition point for LangChain execution middleware.

Wrap hooks are nested in list order: loop accounting -> result reuse ->
timeout/error handling -> tool. Thus hits still consume budget, and failures
are normalized before deduplication decides whether they can be reused.
"""
from langchain.agents.middleware import AgentMiddleware

from agents.middlewares.llm_error_handling import LLMErrorHandlingMiddleware
from agents.middlewares.loop_detection import LoopDetectionMiddleware
from agents.middlewares.tool_deduplication import ToolDeduplicationMiddleware, ToolIdentity
from agents.middlewares.tool_error_handling import ToolErrorHandlingMiddleware


def build_middlewares(*, tool_identity: ToolIdentity | None = None) -> list[AgentMiddleware]:
    middlewares: list[AgentMiddleware] = [LLMErrorHandlingMiddleware()]
    if tool_identity is not None:
        middlewares.extend([
            LoopDetectionMiddleware(tool_identity),
            ToolDeduplicationMiddleware(tool_identity),
            ToolErrorHandlingMiddleware(),
        ])
    return middlewares
