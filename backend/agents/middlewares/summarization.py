"""Budget and compact specialist-agent history without losing turn events."""
from __future__ import annotations

import json
import logging
import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Annotated, Any

from langchain.agents.middleware import AgentState, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from agents.middlewares.execution_context import execution_context
from agents.middlewares.llm_error_handling import run_model_attempts


# Unknown endpoint IDs do not reliably reveal their model's capacity.
DEFAULT_CONTEXT_WINDOW = 8192
MAX_TRIGGER_TOKENS = 32000
MAX_KEEP_TOKENS = 8000
logger = logging.getLogger(__name__)

# The same unscaled counter is used for whole histories and message suffixes.
# This is an intentionally conservative starting point for Chinese, not a tokenizer.
count_history_tokens = partial(
    count_tokens_approximately, chars_per_token=2.0, use_usage_metadata_scaling=False
)

SUMMARY_PROMPT = """请为后续健康咨询压缩下面的历史对话，只输出简洁、忠实的中文摘要。
保留用户的主要诉求、症状和时间线、重要的肯定及否定信息、既往史、过敏史、
用药名称与剂量、检查结果和单位、风险提示、已经采取的行动以及尚待解决的问题。
区分用户陈述、工具结果和助手建议；保留不确定性和前后矛盾，不要补造事实或诊断。
优先保留影响后续安全判断的事实，省略重复和无关内容。

历史对话：
{messages}"""


@dataclass(frozen=True)
class ContextBudget:
    window: int
    fixed: int
    tool_reserve: int
    safety_reserve: int
    output_reserve: int
    history: int
    trigger: int
    keep: int


def model_context_window(model: BaseChatModel) -> int:
    """Use the actual model profile, falling back safely for unknown models."""
    profile = getattr(model, "profile", None) or {}
    value = profile.get("max_input_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 1024:
            raise ValueError("Model profile max_input_tokens must be at least 1024")
        return value
    _warn_unknown_capacity(getattr(model, "model_name", "") or model._llm_type)
    return DEFAULT_CONTEXT_WINDOW


@lru_cache(maxsize=64)
def _warn_unknown_capacity(_model_name: str) -> None:
    logger.warning(
        "Model context capacity unavailable; using %d tokens. Set LLM_CONTEXT_WINDOW to the actual capacity.",
        DEFAULT_CONTEXT_WINDOW,
    )


def calculate_context_budget(
    window: int, system_prompt: str | SystemMessage, tools: Sequence[BaseTool]
) -> ContextBudget:
    """Reserve prompt, tool growth, estimation error and possible output space."""
    if window < 1024:
        raise ValueError("Model context window must be at least 1024 tokens")
    prompt = system_prompt if isinstance(system_prompt, SystemMessage) else SystemMessage(content=system_prompt)
    tool_chars = sum(
        len(json.dumps(convert_to_openai_tool(tool), ensure_ascii=False)) for tool in tools
    )
    estimated_fixed = count_history_tokens([prompt]) + math.ceil(tool_chars / 2.0)
    fixed = max(estimated_fixed, math.ceil(window * 0.09))
    tool_reserve = math.ceil(window * 0.15)
    safety_reserve = math.ceil(window * 0.09)
    output_reserve = math.ceil(window * 0.05)
    history = window - fixed - tool_reserve - safety_reserve - output_reserve
    if history <= 0:
        raise ValueError(
            "System prompt and reserved context exceed the model window; "
            "reduce prompt/tool size or configure the correct LLM_CONTEXT_WINDOW"
        )
    trigger = max(1, min(MAX_TRIGGER_TOKENS, int(history * 0.67)))
    keep = max(1, min(MAX_KEEP_TOKENS, int(trigger * 0.29)))
    return ContextBudget(window, fixed, tool_reserve, safety_reserve, output_reserve, history, trigger, keep)


def is_summary(message: BaseMessage) -> bool:
    return message.additional_kwargs.get("lc_source") == "summarization"


class SummaryState(AgentState, total=False):
    # The inner agent can compact messages produced during this very turn.
    # Retain them separately for handoffs and report-card side effects.
    summary_input_ids: list[str]
    summary_archived_messages: Annotated[list[BaseMessage], operator.add]


class HealthSummarizationMiddleware(SummarizationMiddleware):
    """LangChain compaction with failure propagation and a turn-event journal."""

    state_schema = SummaryState

    def __init__(self, model: BaseChatModel, budget: ContextBudget) -> None:
        super().__init__(
            model=model.model_copy(update={"disable_streaming": True}),
            trigger=("tokens", budget.trigger),
            keep=("tokens", budget.keep),
            token_counter=count_history_tokens,
            summary_prompt=SUMMARY_PROMPT,
            trim_tokens_to_summarize=None,
        )

    def _summary_input(self, messages: list[BaseMessage]) -> str:
        return self.summary_prompt.format(messages=get_buffer_string(messages)).rstrip()

    @staticmethod
    def _validated_summary(response: Any) -> str:
        summary = response.text.strip()
        if not summary:
            raise ValueError("The model returned an empty conversation summary")
        return summary

    # LangChain 1.2.13 catches summary exceptions and replaces the history with
    # an error string. Propagate errors instead; no state update is then applied.
    def _create_summary(self, messages_to_summarize: list[BaseMessage]) -> str:
        response = self.model.invoke(
            self._summary_input(messages_to_summarize),
            config={"metadata": {"lc_source": "summarization"}},
        )
        return self._validated_summary(response)

    async def _acreate_summary(self, messages_to_summarize: list[BaseMessage]) -> str:
        response = await self.model.ainvoke(
            self._summary_input(messages_to_summarize),
            config={"metadata": {"lc_source": "summarization"}},
        )
        return self._validated_summary(response)

    @staticmethod
    def _record_turn_events(state: dict, update: dict | None) -> dict | None:
        first_call = "summary_input_ids" not in state
        input_ids = state.get("summary_input_ids") if not first_call else [m.id for m in state["messages"]]
        if update is None:
            return {"summary_input_ids": input_ids} if first_call else None

        # The update contains RemoveMessage, then the new summary and preserved suffix.
        remaining_ids = {m.id for m in update["messages"][2:]}
        archived = [
            message for message in state["messages"]
            if message.id not in remaining_ids and message.id not in input_ids and not is_summary(message)
        ]
        return {**update, "summary_input_ids": input_ids, "summary_archived_messages": archived}

    def before_model(self, state: SummaryState, runtime: Any) -> dict | None:
        return self._record_turn_events(state, super().before_model(state, runtime))

    async def abefore_model(self, state: SummaryState, runtime: Any) -> dict | None:
        context = execution_context(runtime)
        before_model = super().abefore_model

        async def invoke() -> dict | None:
            return await before_model(state, runtime)

        update = await run_model_attempts(context, invoke)
        return self._record_turn_events(state, update)
