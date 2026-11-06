"""Compose native LangChain agents with one consistent middleware chain."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agents.middlewares import build_middlewares
from agents.middlewares.execution_context import get_agent_context
from agents.middlewares.summarization import is_summary
from agents.tool_policy import tool_identity
from runtime.config import get_settings
from runtime.context import RunContext, current_run


def create_specialist_agent(
    model: BaseChatModel, *, tools: list[BaseTool], system_prompt: str | SystemMessage,
) -> CompiledStateGraph:
    context = current_run.get()
    policy = context.settings if context else get_settings()
    return create_agent(
        model=model, tools=tools, system_prompt=system_prompt,
        middleware=build_middlewares(
            tool_identity=tool_identity, model=model, system_prompt=system_prompt, tools=tools,
        ), context_schema=RunContext,
    ).with_config({"recursion_limit": policy.agent_recursion_limit})


@dataclass
class SpecialistResult:
    new_messages: list[BaseMessage]
    compacted_messages: list[BaseMessage] | None


async def invoke_specialist_agent(agent: CompiledStateGraph, messages: Sequence[BaseMessage]) -> SpecialistResult:
    """Separate this turn's events from history even when compaction changes its length."""
    for message in messages:
        if message.id is None:
            message.id = str(uuid4())
    input_ids = {message.id for message in messages}
    result = await agent.ainvoke({"messages": list(messages)}, context=get_agent_context())
    remaining = list(result["messages"])
    archived = result.get("summary_archived_messages", [])
    new_messages = [
        message for message in [*archived, *remaining]
        if message.id not in input_ids and not is_summary(message)
    ]
    compacted = any(is_summary(message) and message.id not in input_ids for message in remaining)
    return SpecialistResult(new_messages, remaining if compacted else None)


async def invoke_model(
    model: BaseChatModel, messages: list[BaseMessage], *, schema: type[BaseModel] | None = None,
) -> Any:
    """One model-only agent for routing, extraction, summaries and vision.

    ProviderStrategy retains provider-native JSON schema; business nodes own
    semantic validation, without an implicit tool-based repair loop.
    """
    system_prompt = messages[0] if messages and isinstance(messages[0], SystemMessage) else None
    agent = create_agent(
        model=model, tools=[], system_prompt=system_prompt,
        response_format=ProviderStrategy(schema) if schema is not None else None,
        middleware=build_middlewares(), context_schema=RunContext,
    )
    result = await agent.ainvoke(
        {"messages": messages[1:] if system_prompt else messages}, context=get_agent_context())
    return result["structured_response"] if schema is not None else result["messages"][-1]
