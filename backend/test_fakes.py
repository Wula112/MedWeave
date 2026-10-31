"""
Shared deterministic doubles for the backend test-suite.

`ScriptedChatModel` is a real `BaseChatModel`, so it can be handed straight to
`create_agent`: it replays a fixed list of `AIMessage`s (tool calls
included), one per invocation, and never touches the network.
"""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from typing import Any, AsyncIterator, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


@contextmanager
def isolated_api_runtime():
    """API tests must never share run admission or touch application databases."""
    with patch("dotenv.load_dotenv"):
        import main
    from runtime.config import RuntimeSettings
    from runtime.runs import RunManager

    with tempfile.TemporaryDirectory() as directory:
        settings = RuntimeSettings(data_dir=Path(directory))
        manager = RunManager(settings)
        with patch.object(main, "_runtime", manager), patch.object(main, "get_settings", return_value=settings):
            yield manager


class ScriptedChatModel(BaseChatModel):
    """Replays `script[i]` on the i-th invocation; repeats the last entry after."""

    script: list[AIMessage] = []
    invocations: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = len(self.invocations)
        self.invocations.append(list(messages))
        message = self.script[min(index, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the scripted reply so `on_chat_model_stream` events fire."""
        index = len(self.invocations)
        self.invocations.append(list(messages))
        message = self.script[min(index, len(self.script) - 1)]
        chunk = AIMessageChunk(
            content=message.content,
            tool_call_chunks=[
                {
                    "name": call["name"],
                    "args": json.dumps(call.get("args") or {}, ensure_ascii=False),
                    "id": call.get("id"),
                    "index": position,
                }
                for position, call in enumerate(message.tool_calls or [])
            ],
        )
        yield ChatGenerationChunk(message=chunk)


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    """Build a fresh `ScriptedChatModel` with its own invocation log."""
    return ScriptedChatModel(script=list(messages), invocations=[])


class StructuredChatModel(ScriptedChatModel):
    """Native ProviderStrategy double; uses the actual agent/middleware graph."""
    schema_name: str | None = None

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "StructuredChatModel":
        response_format = kwargs.get("response_format")
        if response_format:
            name = response_format["json_schema"]["name"]
            return self.model_copy(update={"schema_name": name, "disable_streaming": True})
        return self

    async def structured_reply(self, schema_name: str, messages: list[BaseMessage]):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.schema_name:
            result = await self.structured_reply(self.schema_name, messages)
            content = result.model_dump_json() if hasattr(result, "model_dump_json") else json.dumps(result)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name: str, args: dict | None = None, call_id: str = "call_1") -> AIMessage:
    """An `AIMessage` that asks for exactly one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id}],
    )


class FakeKnowledgeBase:
    """A knowledge base that always returns nothing, so RAG stays offline."""

    async def aretrieve(self, _query: str, k: int = 3) -> list:
        return []

    async def amulti_query_retrieve(self, _queries: list[str], k: int = 4) -> list:
        return []

    def format_context(self, _docs: list) -> str:
        return ""
