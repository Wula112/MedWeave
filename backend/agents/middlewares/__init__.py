"""The sole composition point for LangChain execution middleware.

Wrap hooks are nested in list order: loop accounting -> result reuse ->
timeout/error handling -> tool. Thus hits still consume budget, and failures
are normalized before deduplication decides whether they can be reused.
"""
from collections.abc import Sequence
import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from agents.middlewares.llm_error_handling import LLMErrorHandlingMiddleware
from agents.middlewares.loop_detection import LoopDetectionMiddleware
from agents.middlewares.summarization import (
    HealthSummarizationMiddleware, calculate_context_budget, model_context_window,
)
from agents.middlewares.tool_deduplication import ToolDeduplicationMiddleware, ToolIdentity
from agents.middlewares.tool_error_handling import ToolErrorHandlingMiddleware


logger = logging.getLogger(__name__)


def build_middlewares(
    *, tool_identity: ToolIdentity | None = None, model: BaseChatModel | None = None,
    system_prompt: str | SystemMessage = "", tools: Sequence[BaseTool] = (),
) -> list[AgentMiddleware]:
    middlewares: list[AgentMiddleware] = [LLMErrorHandlingMiddleware()]
    if tool_identity is not None:
        # Stop the graph before attempting summary generation after a hard limit.
        middlewares.append(LoopDetectionMiddleware(tool_identity))
    if model is not None:
        budget = calculate_context_budget(model_context_window(model), system_prompt, tools)
        logger.info(
            "Context budget window=%d fixed=%d trigger=%d keep=%d",
            budget.window, budget.fixed, budget.trigger, budget.keep,
        )
        middlewares.append(HealthSummarizationMiddleware(model, budget))
    if tool_identity is not None:
        middlewares.extend([
            ToolDeduplicationMiddleware(tool_identity),
            ToolErrorHandlingMiddleware(),
        ])
    return middlewares
