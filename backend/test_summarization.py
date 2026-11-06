"""Offline budget, compaction, persistence and event-stream regression tests."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agents.factory import create_specialist_agent, invoke_specialist_agent
from agents.middlewares.execution_context import get_agent_context
from agents.graph import build_graph
from agents.llm import LLMConfigurationError, get_chat_llm, resolve_model_settings
from agents.middlewares.summarization import (
    DEFAULT_CONTEXT_WINDOW, calculate_context_budget, count_history_tokens, model_context_window,
)
from runtime.events import iter_graph_events
from runtime.errors import RuntimeFailure
from test_fakes import FakeKnowledgeBase, ScriptedChatModel, scripted, tool_call
from test_handoff import _ClinicChatModel
from test_clinic_graph import _FakeLLM
from agents import clinic


def model_with_window(*messages, window=8192):
    return scripted(*messages).model_copy(update={"profile": {"max_input_tokens": window}})


class ContextBudgetTests(unittest.TestCase):
    def test_scales_and_caps_budgets_by_model_capacity(self):
        small = calculate_context_budget(8192, "", [])
        medium = calculate_context_budget(32768, "", [])
        large = calculate_context_budget(131072, "", [])
        self.assertLess(small.trigger, medium.trigger)
        self.assertGreater(medium.trigger, 12000)
        self.assertLess(medium.trigger, 15000)
        self.assertEqual(large.trigger, 32000)
        self.assertEqual(large.keep, 8000)
        for budget in (small, medium, large):
            self.assertLess(budget.keep, budget.trigger)
            self.assertLessEqual(budget.trigger, budget.history)
            self.assertEqual(budget.window, budget.fixed + budget.tool_reserve +
                             budget.safety_reserve + budget.output_reserve + budget.history)

    def test_long_system_prompt_reduces_history_budget(self):
        short = calculate_context_budget(8192, "提示", [])
        long = calculate_context_budget(8192, "检索结果" * 500, [])
        self.assertGreater(long.fixed, short.fixed)
        self.assertLess(long.trigger, short.trigger)
        with self.assertRaisesRegex(ValueError, "System prompt"):
            calculate_context_budget(8192, "内容" * 10000, [])

    def test_profile_and_unknown_endpoint_fallback(self):
        known = model_with_window(AIMessage(content="ok"), window=32768)
        unknown = scripted(AIMessage(content="ok"))
        self.assertEqual(model_context_window(known), 32768)
        self.assertEqual(model_context_window(unknown), DEFAULT_CONTEXT_WINDOW)
        with self.assertRaisesRegex(ValueError, "max_input_tokens"):
            model_context_window(model_with_window(AIMessage(content="ok"), window=512))

    def test_chinese_counter_is_more_conservative(self):
        from langchain_core.messages.utils import count_tokens_approximately
        messages = [HumanMessage(content="长期用药和过敏史" * 100)]
        self.assertGreater(count_history_tokens(messages), count_tokens_approximately(messages))

    def test_explicit_capacity_is_validated_and_attached_to_model(self):
        env = {"LLM_PROVIDER": "custom", "LLM_API_KEY": "test", "LLM_MODEL": "endpoint",
               "LLM_BASE_URL": "https://example.invalid/v1", "LLM_CONTEXT_WINDOW": "32768"}
        self.assertEqual(resolve_model_settings(env).context_window, 32768)
        with patch.dict(os.environ, env, clear=True):
            model = get_chat_llm()
        self.assertEqual(model_context_window(model), 32768)
        with patch.dict(os.environ, {**env, "LLM_MODEL": "gpt-4o-mini"}, clear=True):
            known = get_chat_llm()
        self.assertEqual(model_context_window(known), 32768)
        self.assertTrue(known.profile.get("tool_calling"))
        for bad in ("abc", "0", "1023"):
            with self.subTest(value=bad), self.assertRaisesRegex(LLMConfigurationError, "LLM_CONTEXT_WINDOW"):
                resolve_model_settings({**env, "LLM_CONTEXT_WINDOW": bad})


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_agent_summarizes_complete_history_and_keeps_latest_message(self):
        model = model_with_window(AIMessage(content="过敏史：青霉素。"), AIMessage(content="已了解"), window=4096)
        history = [HumanMessage(content="青霉素过敏" + "历史" * 1800),
                   AIMessage(content="已记录"), HumanMessage(content="今天应该注意什么？")]
        agent = create_specialist_agent(model, tools=[], system_prompt="健康助手")
        result = await invoke_specialist_agent(agent, history)
        self.assertIsNotNone(result.compacted_messages)
        self.assertEqual([m.content for m in result.new_messages], ["已了解"])
        self.assertTrue(result.compacted_messages[0].additional_kwargs.get("lc_source") == "summarization")
        self.assertIn("青霉素过敏", model.invocations[0][0].content)
        self.assertTrue(any(m.content == "今天应该注意什么？" for m in model.invocations[1]))

    async def test_turn_tool_events_survive_compaction_for_downstream_consumers(self):
        @tool
        def first() -> str:
            """Return a short first result."""
            return "甲" * 200

        @tool
        def second() -> str:
            """Return a long second result."""
            return "乙" * 4500

        model = model_with_window(
            tool_call("first", {}, "one"), tool_call("second", {}, "two"),
            AIMessage(content="已检查工具结果"), AIMessage(content="完成"), window=4096,
        )
        agent = create_specialist_agent(model, tools=[first, second], system_prompt="健康助手")
        result = await invoke_specialist_agent(agent, [HumanMessage(content="请检查")])
        self.assertIsNotNone(result.compacted_messages)
        tools = [m.name for m in result.new_messages if isinstance(m, ToolMessage)]
        self.assertEqual(tools, ["first", "second"])
        self.assertEqual(result.new_messages[-1].content, "完成")
        self.assertFalse(any(isinstance(m, ToolMessage) and m.name == "first"
                             for m in result.compacted_messages))

    async def test_summary_receives_more_than_four_thousand_tokens_of_old_history(self):
        model = model_with_window(AIMessage(content="既往病史摘要"), AIMessage(content="已了解"), window=32768)
        history = [HumanMessage(content="最早记录" + "史" * 15000),
                   AIMessage(content="中间记录" + "药" * 15000), HumanMessage(content="当前问题")]
        agent = create_specialist_agent(model, tools=[], system_prompt="健康助手")
        await invoke_specialist_agent(agent, history)
        summary_input = model.invocations[0][0].content
        self.assertIn("最早记录", summary_input)
        self.assertIn("中间记录", summary_input)

    async def test_summary_failure_does_not_replace_history_with_error_text(self):
        class BrokenSummary(ScriptedChatModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("summary failed")

        model = BrokenSummary(script=[AIMessage(content="unused")], invocations=[],
                              profile={"max_input_tokens": 4096})
        history = [HumanMessage(content="重要病史" * 1000), HumanMessage(content="继续")]
        agent = create_specialist_agent(model, tools=[], system_prompt="健康助手")
        with self.assertRaises(RuntimeFailure) as error:
            await invoke_specialist_agent(agent, history)
        self.assertEqual(error.exception.code, "execution_failed")
        self.assertEqual(len(history), 2)
        self.assertIn("重要病史", history[0].content)

    async def test_summary_retries_transient_failures_using_run_policy(self):
        class RetrySummary(ScriptedChatModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                if not self.invocations:
                    self.invocations.append(list(messages))
                    raise ConnectionError("temporary")
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        model = RetrySummary(script=[AIMessage(content="unused"), AIMessage(content="保留病史"),
                                     AIMessage(content="完成")], invocations=[],
                             profile={"max_input_tokens": 4096})
        agent = create_specialist_agent(model, tools=[], system_prompt="健康助手")
        with patch("agents.middlewares.llm_error_handling._backoff", new_callable=AsyncMock) as backoff:
            result = await invoke_specialist_agent(
                agent, [HumanMessage(content="病史" * 2000), HumanMessage(content="继续")]
            )
        self.assertEqual(result.new_messages[-1].content, "完成")
        backoff.assert_awaited_once()

    async def test_run_failure_blocks_summary_model_call(self):
        model = model_with_window(AIMessage(content="不应调用"), window=4096)
        context = get_agent_context()
        context.failure = RuntimeFailure("loop_detected", "已停止")
        agent = create_specialist_agent(model, tools=[], system_prompt="健康助手")
        with patch("agents.factory.get_agent_context", return_value=context):
            result = await invoke_specialist_agent(agent, [HumanMessage(content="病史" * 2000),
                                                          HumanMessage(content="继续")])
        self.assertEqual(result.new_messages, [])
        self.assertEqual(context.failure.code, "loop_detected")
        self.assertEqual(model.invocations, [])

    async def test_summary_obeys_model_timeout(self):
        class SlowSummary(ScriptedChatModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                await asyncio.sleep(1)
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

        model = SlowSummary(script=[AIMessage(content="摘要")], invocations=[],
                            profile={"max_input_tokens": 4096})
        context = get_agent_context()
        context.settings = context.settings.model_copy(update={"model_timeout": 0.01, "model_attempts": 1})
        agent = create_specialist_agent(model, tools=[], system_prompt="健康助手")
        with patch("agents.factory.get_agent_context", return_value=context):
            with self.assertRaises(RuntimeFailure) as error:
                await invoke_specialist_agent(agent, [HumanMessage(content="病史" * 2000),
                                                      HumanMessage(content="继续")])
        self.assertEqual(error.exception.code, "model_timeout")

    async def test_compacted_history_is_persisted_across_turns(self):
        model = model_with_window(AIMessage(content="用户对青霉素过敏"),
                                  AIMessage(content="第一次回答"), AIMessage(content="第二次回答"))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "compaction-test"}}
        state = {"messages": [HumanMessage(content="青霉素过敏" + "既往史" * 4000),
                              AIMessage(content="已记录"), HumanMessage(content="请继续")],
                 "user_info": {}, "active_agent": "advisor_agent", "next_agent": "", "handoff_count": 0}
        with patch("agents.router.get_chat_llm", return_value=scripted(AIMessage(content="advisor_agent"))), \
             patch("agents.advisor.get_chat_llm", return_value=model), \
             patch("agents.advisor.get_knowledge_base", return_value=FakeKnowledgeBase()):
            first = await graph.ainvoke(state, config)
            second = await graph.ainvoke({"messages": [HumanMessage(content="还有什么建议？")]}, config)
        self.assertTrue(first["messages"][0].additional_kwargs.get("lc_source") == "summarization")
        self.assertEqual(first["messages"][-1].content, "第一次回答")
        self.assertEqual(second["messages"][-1].content, "第二次回答")
        self.assertEqual(len(model.invocations), 3)

    async def test_handoff_after_compaction_keeps_the_current_request(self):
        insurance = model_with_window(AIMessage(content="用户有青霉素过敏史"),
                                      tool_call("transfer_to_pharmacy", {}, "transfer"),
                                      AIMessage(content="为您转接"))
        pharmacy = scripted(AIMessage(content="请先咨询药师。"))
        graph = build_graph(checkpointer=MemorySaver())
        state = {"messages": [HumanMessage(content="青霉素过敏" + "既往史" * 4000),
                              AIMessage(content="已记录"), HumanMessage(content="这个药怎么用？")],
                 "user_info": {}, "active_agent": "insurance_agent", "next_agent": "", "handoff_count": 0}
        with patch("agents.insurance.get_chat_llm", return_value=insurance), \
             patch("agents.pharmacy.get_chat_llm", return_value=pharmacy):
            result = await graph.ainvoke(state, {"configurable": {"thread_id": "compacted-handoff"}})
        self.assertEqual(result["active_agent"], "pharmacy_agent")
        self.assertEqual(result["messages"][-1].content, "请先咨询药师。")
        self.assertTrue(result["messages"][0].additional_kwargs.get("lc_source") == "summarization")
        self.assertTrue(any(m.content == "这个药怎么用？" for m in pharmacy.invocations[0]))
        self.assertFalse(any(isinstance(m, ToolMessage) and m.name == "transfer_to_pharmacy"
                             for m in result["messages"]))

    async def test_clinic_compaction_is_returned_to_parent_after_conclusion(self):
        model = _ClinicChatModel(
            script=[AIMessage(content="用户对青霉素过敏"), AIMessage(content="建议到呼吸内科就诊。")],
            invocations=[], profile={"max_input_tokens": 8192},
            facts=clinic.SymptomFacts(chief_complaint="咳嗽", duration="三天", severity="轻微"),
            recommendation=clinic.TriageRecommendation(department=["呼吸内科"], urgency="routine"),
        )
        graph = build_graph(checkpointer=MemorySaver())
        state = {"messages": [HumanMessage(content="青霉素过敏" + "既往史" * 4000),
                              AIMessage(content="已记录"), HumanMessage(content="咳嗽三天了")],
                 "user_info": {}, "active_agent": "clinic_agent", "next_agent": "", "handoff_count": 0}
        config = {"configurable": {"thread_id": "clinic-compaction"}}
        with patch("agents.clinic.get_chat_llm", return_value=model), \
             patch("agents.clinic.get_knowledge_base", return_value=FakeKnowledgeBase()):
            result = await graph.ainvoke(state, config)
        self.assertEqual(result["messages"][-1].content, "建议到呼吸内科就诊。")
        self.assertTrue(result["messages"][0].additional_kwargs.get("lc_source") == "summarization")
        self.assertIsNone(result.get("history_replacement"))
        self.assertEqual((await graph.aget_state(config)).values["messages"], result["messages"])

    async def test_clinic_compaction_survives_handoff_to_parent(self):
        model = _ClinicChatModel(
            script=[AIMessage(content="用户对青霉素过敏"),
                    tool_call("transfer_to_pharmacy", {}, "transfer"), AIMessage(content="为您转接")],
            invocations=[], profile={"max_input_tokens": 8192},
            facts=clinic.SymptomFacts(chief_complaint="头痛", duration="一天", severity="轻微"),
            recommendation=clinic.TriageRecommendation(department=["神经内科"], urgency="routine"),
        )
        pharmacy = scripted(AIMessage(content="请咨询药师。"))
        graph = build_graph(checkpointer=MemorySaver())
        state = {"messages": [HumanMessage(content="青霉素过敏" + "既往史" * 4000),
                              AIMessage(content="已记录"), HumanMessage(content="头痛吃什么药？")],
                 "user_info": {}, "active_agent": "clinic_agent", "next_agent": "", "handoff_count": 0}
        with patch("agents.clinic.get_chat_llm", return_value=model), \
             patch("agents.clinic.get_knowledge_base", return_value=FakeKnowledgeBase()), \
             patch("agents.pharmacy.get_chat_llm", return_value=pharmacy):
            result = await graph.ainvoke(state, {"configurable": {"thread_id": "clinic-compacted-handoff"}})
        self.assertEqual(result["active_agent"], "pharmacy_agent")
        self.assertEqual(result["messages"][-1].content, "请咨询药师。")
        self.assertTrue(result["messages"][0].additional_kwargs.get("lc_source") == "summarization")
        self.assertTrue(any(m.content == "头痛吃什么药？" for m in pharmacy.invocations[0]))

    async def test_compaction_after_clinic_resume_exports_followup_once(self):
        model = _FakeLLM(
            [clinic.SymptomFacts(chief_complaint="咳嗽"),
             clinic.SymptomFacts(chief_complaint="咳嗽", duration="三天", severity="轻微")],
            clinic.TriageRecommendation(department=["呼吸内科"], urgency="routine"),
            "咳嗽多久了？",
        ).model_copy(update={"profile": {"max_input_tokens": 8192},
                             "script": [AIMessage(content="青霉素过敏"), AIMessage(content="建议就诊。")],
                             "invocations": []})
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "compacted-clinic-resume"}}
        state = {"messages": [HumanMessage(content="青霉素过敏" + "既往史" * 4000),
                              AIMessage(content="已记录"), HumanMessage(content="我咳嗽")],
                 "user_info": {}, "active_agent": "clinic_agent", "next_agent": "", "handoff_count": 0}
        with patch("agents.clinic.get_chat_llm", return_value=model), \
             patch("agents.clinic.get_knowledge_base", return_value=FakeKnowledgeBase()):
            first = await graph.ainvoke(state, config)
            self.assertIn("__interrupt__", first)
            result = await graph.ainvoke(Command(resume="三天，轻微"), config)
        texts = [message.content for message in result["messages"]]
        self.assertEqual(texts.count("咳嗽多久了？"), 1)
        self.assertEqual(texts.count("三天，轻微"), 1)
        self.assertEqual(texts[-1], "建议就诊。")
        self.assertTrue(result["messages"][0].additional_kwargs.get("lc_source") == "summarization")

    async def test_summary_model_tokens_are_not_streamed_to_the_user(self):
        class Graph:
            async def astream_events(self, *args, **kwargs):
                yield {"event": "on_chat_model_stream", "metadata": {"lc_source": "summarization"},
                       "data": {"chunk": SimpleNamespace(content="内部摘要")}}
                yield {"event": "on_chat_model_stream", "metadata": {},
                       "data": {"chunk": SimpleNamespace(content="正常回答")}}

        events = [event async for event in iter_graph_events(Graph(), {}, {}, node_labels={},
                                                           tool_labels={}, custom_types=set())]
        self.assertEqual(events, [{"type": "text", "content": "正常回答"}, {"type": "finish"}])

    async def test_real_graph_stream_excludes_summary(self):
        model = model_with_window(AIMessage(content="内部病史摘要"), AIMessage(content="这是正式回答"))
        graph = build_graph(checkpointer=MemorySaver())
        state = {"messages": [HumanMessage(content="青霉素过敏" + "既往史" * 4000),
                              AIMessage(content="已记录"), HumanMessage(content="请继续")],
                 "user_info": {}, "active_agent": "advisor_agent", "next_agent": "", "handoff_count": 0}
        with patch("agents.router.get_chat_llm", return_value=scripted(AIMessage(content="advisor_agent"))), \
             patch("agents.advisor.get_chat_llm", return_value=model), \
             patch("agents.advisor.get_knowledge_base", return_value=FakeKnowledgeBase()):
            events = [event async for event in iter_graph_events(
                graph, state, {"configurable": {"thread_id": "summary-stream"}},
                node_labels={}, tool_labels={}, custom_types=set(),
            )]
        text = "".join(event["content"] for event in events if event["type"] == "text")
        self.assertNotIn("内部病史摘要", text)
        self.assertIn("这是正式回答", text)
