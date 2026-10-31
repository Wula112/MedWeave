"""Behavioral tests through real create_agent graphs; no provider calls."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime

from agents.factory import create_specialist_agent, invoke_model, get_agent_context
from agents.middlewares import build_middlewares
from agents.middlewares.llm_error_handling import LLMErrorHandlingMiddleware
from agents.middlewares.tool_error_handling import ToolErrorHandlingMiddleware
from runtime.config import RuntimeSettings
from runtime.context import RunContext, current_run
from runtime.errors import RuntimeFailure
from test_fakes import ScriptedChatModel, scripted, tool_call


async def noop():
    pass


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.policy = RuntimeSettings(data_dir=Path(self.temp.name), retry_base=0, retry_cap=0, model_timeout=0.3)
        self.scope_patch = patch("agents.middlewares.execution_context.get_settings", return_value=self.policy)
        self.scope_patch.start()

    async def asyncTearDown(self):
        self.scope_patch.stop()
        self.temp.cleanup()

    async def test_transient_model_retry_runs_inside_native_agent(self):
        class Model(ScriptedChatModel):
            async def _agenerate(self, messages, **kwargs):
                self.invocations.append(messages)
                if len(self.invocations) < 3:
                    raise ConnectionError("temporary")
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        model = Model(invocations=[])
        result = await invoke_model(model, [HumanMessage(content="test")])
        self.assertEqual(result.content, "ok")
        self.assertEqual(len(model.invocations), 3)

    async def test_permanent_model_failure_is_not_retried(self):
        class Model(ScriptedChatModel):
            async def _agenerate(self, messages, **kwargs):
                self.invocations.append(messages)
                raise ValueError("private provider details")

        model = Model(invocations=[])
        with self.assertRaises(RuntimeFailure) as raised:
            await invoke_model(model, [HumanMessage(content="test")])
        self.assertEqual(len(model.invocations), 1)
        self.assertNotIn("private", str(raised.exception))

    async def test_provider_status_and_quota_control_retry_count(self):
        class ProviderError(Exception):
            def __init__(self, status, code):
                self.status_code = status
                self.body = {"error": {"code": code}}

        for status, code, expected in ((401, "", 1), (400, "", 1), (429, "insufficient_quota", 1),
                                       (408, "", 3), (429, "rate_limit", 3), (503, "", 3)):
            with self.subTest(status=status, code=code):
                class Model(ScriptedChatModel):
                    async def _agenerate(self, messages, **kwargs):
                        self.invocations.append(messages)
                        raise ProviderError(status, code)

                model = Model(invocations=[])
                with self.assertRaises(RuntimeFailure):
                    await invoke_model(model, [HumanMessage(content="test")])
                self.assertEqual(len(model.invocations), expected)

    async def test_retry_budget_is_bounded_and_cancellation_propagates(self):
        class Model(ScriptedChatModel):
            async def _agenerate(self, messages, **kwargs):
                self.invocations.append(messages)
                await asyncio.Event().wait()

        model = Model(invocations=[])
        with patch("agents.middlewares.execution_context.get_settings",
                   return_value=self.policy.model_copy(update={"model_timeout": 0.02})):
            with self.assertRaises(RuntimeFailure) as raised:
                await invoke_model(model, [HumanMessage(content="test")])
        self.assertEqual(raised.exception.code, "model_timeout")
        self.assertEqual(len(model.invocations), 3)

        context = RunContext("t", "t", "text", float("inf"), self.policy, noop)
        request = ModelRequest(model=model, messages=[], state={"messages": []}, runtime=Runtime(context=context))
        for error in (asyncio.CancelledError(), GraphInterrupt(())):
            calls = []

            async def fail(request):
                calls.append(1)
                raise error

            with self.assertRaises(type(error)):
                await LLMErrorHandlingMiddleware().awrap_model_call(request, fail)
            self.assertEqual(calls, [1])

    async def test_partial_stream_is_not_replayed(self):
        class Model(ScriptedChatModel):
            async def _astream(self, messages, **kwargs):
                self.invocations.append(messages)
                yield ChatGenerationChunk(message=AIMessageChunk(content="partial"))
                raise ConnectionError("dropped stream")

        model = Model(invocations=[])
        agent = create_agent(model, middleware=build_middlewares(), context_schema=RunContext)
        chunks = []
        with self.assertRaises(RuntimeFailure):
            async for event in agent.astream_events({"messages": [HumanMessage(content="test")]}, version="v2", context=get_agent_context()):
                if event["event"] == "on_chat_model_stream":
                    chunks.append(event["data"]["chunk"].content)
        self.assertEqual(chunks, ["partial"])
        self.assertEqual(len(model.invocations), 1)

    async def test_parallel_duplicates_rebind_ids_and_loop_routes_to_end(self):
        calls = []

        @tool
        async def health_calculator(value: int = 1) -> str:
            """A read-only test calculation."""
            calls.append(value)
            await asyncio.sleep(0.01)
            return '{"success":true}'

        model = scripted(
            AIMessage(content="", tool_calls=[
                {"id": "one", "name": "health_calculator", "args": {}},
                {"id": "two", "name": "health_calculator", "args": {"value": 1}},
            ]),
            tool_call("health_calculator", {}, "three"),
            AIMessage(content="must not reach another model call"),
        )
        agent = create_specialist_agent(model, tools=[health_calculator], system_prompt="test")
        result = await agent.ainvoke({"messages": [HumanMessage(content="test")]}, context=get_agent_context())
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(calls, [1])
        self.assertEqual([m.tool_call_id for m in messages], ["one", "two", "three"])
        self.assertTrue(messages[1].additional_kwargs["runtime_cached"])
        self.assertEqual(messages[-1].status, "error")
        self.assertIn("loop_detected", messages[-1].content)
        self.assertEqual(len(model.invocations), 2)

    async def test_validation_errors_are_sanitized_and_agent_can_answer(self):
        @tool
        async def calculation(value: int) -> str:
            """Validate an integer."""
            return str(value)

        model = scripted(tool_call("calculation", {"value": "private-invalid-value"}, "bad"), AIMessage(content="cannot calculate"))
        result = await create_specialist_agent(model, tools=[calculation], system_prompt="test").ainvoke(
            {"messages": [HumanMessage(content="test")]}, context=get_agent_context())
        message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(message.tool_call_id, "bad")
        self.assertEqual(message.status, "error")
        self.assertNotIn("private-invalid-value", message.content)
        self.assertEqual(result["messages"][-1].content, "cannot calculate")

    async def test_failed_tool_is_not_reexecuted_by_dedup_layer(self):
        calls = []

        @tool
        async def health_calculator(value: int = 1) -> str:
            """A failing read-only calculation."""
            calls.append(1)
            raise OSError("private details")

        model = scripted(tool_call("health_calculator", {}, "a"), tool_call("health_calculator", {}, "b"), AIMessage(content="done"))
        result = await create_specialist_agent(model, tools=[health_calculator], system_prompt="test").ainvoke(
            {"messages": [HumanMessage(content="test")]}, context=get_agent_context())
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(calls, [1])
        self.assertEqual([m.status for m in messages], ["error", "error"])

    async def test_timed_out_tool_is_cancelled_and_not_started_again(self):
        calls = []
        cancelled = asyncio.Event()

        @tool
        async def health_calculator(value: int = 1) -> str:
            """Simulate an unresponsive read-only service."""
            calls.append(1)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return "unreachable"

        model = scripted(tool_call("health_calculator", {}, "a"), tool_call("health_calculator", {}, "b"), AIMessage(content="done"))
        policy = self.policy.model_copy(update={"tool_timeout": 0.02})
        with patch("agents.middlewares.execution_context.get_settings", return_value=policy):
            result = await create_specialist_agent(model, tools=[health_calculator], system_prompt="test").ainvoke(
                {"messages": [HumanMessage(content="test")]}, context=get_agent_context())
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(calls, [1])
        self.assertTrue(cancelled.is_set())
        self.assertTrue(all("tool_timeout" in m.content for m in messages))
        self.assertTrue(messages[1].additional_kwargs["runtime_cached"])

    async def test_unclassified_tools_do_not_reuse_successful_results(self):
        calls = []

        @tool
        async def write_record(value: int = 1) -> str:
            """Represent a future tool with an external side effect."""
            calls.append(value)
            return "ok"

        model = scripted(tool_call("write_record", {}, "a"), tool_call("write_record", {}, "b"), AIMessage(content="done"))
        await create_specialist_agent(model, tools=[write_record], system_prompt="test").ainvoke(
            {"messages": [HumanMessage(content="test")]}, context=get_agent_context())
        self.assertEqual(calls, [1, 1])

    async def test_load_skill_and_direct_skill_share_normalized_results(self):
        from skills import get_registry, load_skill

        registry = get_registry()
        direct = registry.get_tool("health_calculator")
        instance = registry._load_instance("health_calculator")
        arguments = {"height_cm": 170, "weight_kg": 65, "age": 30, "gender": "male"}
        model = scripted(
            tool_call("health_calculator", arguments, "direct"),
            tool_call("load_skill", {"skill_name": "health_calculator", "params_json": {
                **arguments, "activity_level": "sedentary"}}, "alias"), AIMessage(content="done"))
        with patch.object(instance, "arun", AsyncMock(wraps=instance.arun)) as execute:
            result = await create_specialist_agent(model, tools=[direct, load_skill], system_prompt="test").ainvoke(
                {"messages": [HumanMessage(content="test")]}, context=get_agent_context())
            self.assertEqual(execute.await_count, 1)
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in messages], ["direct", "alias"])
        self.assertEqual(messages[1].name, "load_skill")
        self.assertTrue(messages[1].additional_kwargs["runtime_cached"])

    async def test_hosted_run_shares_tool_budget_across_specialist_graphs(self):
        calls = []

        @tool
        async def health_calculator(value: int) -> str:
            """Count physical requests across separate specialists."""
            calls.append(value)
            return "ok"

        policy = self.policy.model_copy(update={"max_tool_calls": 1})
        context = RunContext("run", "thread", "text", float("inf"), policy, noop)
        token = current_run.set(context)
        try:
            for value in (1, 2):
                model = scripted(tool_call("health_calculator", {"value": value}, str(value)), AIMessage(content="done"))
                await create_specialist_agent(model, tools=[health_calculator], system_prompt="test").ainvoke(
                    {"messages": [HumanMessage(content="test")]}, context=get_agent_context())
            self.assertEqual(calls, [1])
            self.assertEqual(context.failure.code, "tool_limit")
            summary = scripted(AIMessage(content="must not execute after a hard stop"))
            with self.assertRaises(RuntimeFailure) as raised:
                await invoke_model(summary, [HumanMessage(content="summarize")])
            self.assertEqual(raised.exception.code, "tool_limit")
            self.assertEqual(summary.invocations, [])
        finally:
            current_run.reset(token)

    async def test_same_compiled_agent_gets_new_budget_and_never_checkpoints_tasks(self):
        calls = []

        @tool
        async def health_calculator(value: int = 1) -> str:
            """Count physical calls."""
            calls.append(1)
            return "ok"

        model = scripted(tool_call("health_calculator", {}, "a"), AIMessage(content="one"),
                         tool_call("health_calculator", {}, "b"), AIMessage(content="two"))
        from agents.tool_policy import tool_identity
        saver = MemorySaver()
        agent = create_agent(model, tools=[health_calculator], middleware=build_middlewares(tool_identity=tool_identity), checkpointer=saver, context_schema=RunContext)
        config = {"configurable": {"thread_id": "same"}}
        for text in ("first", "second"):
            result = await agent.ainvoke({"messages": [HumanMessage(content=text)]}, config, context=get_agent_context())
            self.assertNotIn("context", result)
            self.assertNotIn("context", saver.get_tuple(config).checkpoint["channel_values"])
        self.assertEqual(calls, [1, 1])

    async def test_tool_control_signals_are_not_ordinary_errors(self):
        context = RunContext("t", "t", "text", float("inf"), self.policy, noop)
        request = ToolCallRequest(tool_call={"name": "test", "args": {}, "id": "id"}, tool=None,
                                  state={"messages": []}, runtime=Runtime(context=context))

        async def interrupt(request):
            raise GraphInterrupt(())

        with self.assertRaises(GraphInterrupt):
            await ToolErrorHandlingMiddleware().awrap_tool_call(request, interrupt)


if __name__ == "__main__":
    unittest.main()
