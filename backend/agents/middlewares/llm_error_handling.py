"""Model-boundary timeout/retry policy using LangChain's public middleware API."""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ModelCallResult, ModelRequest
from langchain.agents.structured_output import StructuredOutputError
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.errors import GraphBubbleUp

from agents.middlewares.execution_context import execution_context
from runtime.context import RunContext
from runtime.errors import RuntimeFailure, public_failure, transient_model_error

logger = logging.getLogger(__name__)
T = TypeVar("T")


class _StreamProgress(AsyncCallbackHandler):
    """Observe streamed chunks without replacing any model implementation."""
    run_inline = True

    def __init__(self) -> None:
        self.emitted = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.emitted = True


async def _backoff(exc: Exception, attempt: int, context: RunContext) -> None:
    settings = context.settings
    delay = min(settings.retry_cap, settings.retry_base * 2 ** (attempt - 1) * random.uniform(0.75, 1.25))
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        delay = min(settings.retry_cap, max(delay, float(headers.get("retry-after", 0))))
    except (TypeError, ValueError):
        pass
    if context.remaining() <= delay:
        raise RuntimeFailure("run_timeout", "本次处理超时，请稍后重试。", retryable=True)
    await asyncio.sleep(delay)


async def run_model_attempts(
    context: RunContext, invoke: Callable[[], Awaitable[T]],
    *, streamed: Callable[[], bool] = lambda: False,
) -> T:
    """Apply the same deadline, retry and error policy to model calls and summaries."""
    if context.failure:
        raise context.failure
    for attempt in range(1, context.settings.model_attempts + 1):
        try:
            async with asyncio.timeout(context.remaining(context.settings.model_timeout)):
                return await invoke()
        except (GraphBubbleUp, StructuredOutputError):
            raise
        except Exception as exc:
            if streamed() or not transient_model_error(exc) or attempt == context.settings.model_attempts:
                if isinstance(exc, TimeoutError):
                    raise RuntimeFailure("model_timeout", "模型响应超时，请稍后重试。", retryable=True) from None
                raise public_failure(exc) from None
            logger.info("Model retry run_id=%s attempt=%d error_type=%s", context.run_id, attempt, type(exc).__name__)
            await _backoff(exc, attempt, context)
    raise AssertionError("unreachable")


class LLMErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    async def awrap_model_call(
        self, request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        context = execution_context(request.runtime)
        progress = _StreamProgress()

        async def invoke() -> ModelCallResult:
            nonlocal progress
            progress = _StreamProgress()
            callbacks = request.model.callbacks
            if callbacks is not None and not isinstance(callbacks, list):
                callbacks = callbacks.copy()
                callbacks.add_handler(progress, inherit=True)
            else:
                callbacks = [*(callbacks or []), progress]
            # A shallow Pydantic copy retains provider clients and tool/schema
            # binding. Only this attempt's callback list changes.
            model = request.model.model_copy(update={"callbacks": callbacks})
            return await handler(request.override(model=model))

        return await run_model_attempts(context, invoke, streamed=lambda: progress.emitted)
