"""Inject run dependencies through LangChain Runtime.context, never graph state."""
from __future__ import annotations

from langgraph.runtime import Runtime

from runtime.config import get_settings
from runtime.context import RunContext, current_run


async def _noop() -> None:
    pass


def get_agent_context() -> RunContext:
    """Share a hosted Run or give a standalone invocation a fresh budget."""
    return current_run.get() or RunContext(
        "standalone", "", "text", float("inf"), get_settings(), _noop)


def execution_context(runtime: Runtime) -> RunContext:
    context = runtime.context
    if not isinstance(context, RunContext):
        raise RuntimeError("Pass context=get_agent_context() when invoking a middleware-enabled agent")
    return context
