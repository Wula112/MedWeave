"""Offline regressions for interrupted clinic handoff, re-entry and completion."""
from __future__ import annotations

import json
import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agents import clinic
from agents.checkpointing import create_checkpointer
from agents.graph import build_graph
from test_clinic_graph import _FakeLLM, _FakeReactAgent
from test_fakes import FakeKnowledgeBase, scripted, tool_call


PARTIAL = clinic.SymptomFacts(chief_complaint="咳嗽")
COMPLETE = clinic.SymptomFacts(chief_complaint="咳嗽", duration="三天", severity="中等")
QUESTION = "咳嗽多久了呢？"
ANSWER = "建议您到呼吸内科就诊。"


class ReplyModel(_FakeLLM):
    decisions: list[Any] = []
    routing_inputs: list[Any] = []

    def __init__(self, facts, decisions=()):
        super().__init__(facts, clinic.TriageRecommendation(
            department=["呼吸内科"], urgency="soon", summary=ANSWER), QUESTION)
        self.decisions = list(decisions)
        self.routing_inputs = []

    async def structured_reply(self, schema_name, messages):
        if schema_name == "ClinicReplyDecision":
            self.routing_inputs.append(messages)
            return self.decisions.pop(0) if self.decisions else clinic.ClinicReplyDecision(action="answer")
        return await super().structured_reply(schema_name, messages)


def state(text: str, *, active="clinic_agent", channel="text") -> dict:
    return {"messages": [HumanMessage(content=text)], "user_info": {},
            "active_agent": active, "next_agent": "", "handoff_count": 0, "channel": channel}


async def clinic_state(graph, config) -> dict:
    child_config = {"configurable": {**config["configurable"], "checkpoint_ns": "clinic_node"}}
    return (await graph.aget_state(child_config)).values


class ClinicPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def use_models(self, model, *, insurance=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("agents.clinic.get_chat_llm", return_value=model))
        stack.enter_context(patch("agents.clinic.get_agent_tools", return_value=[]))
        stack.enter_context(patch("agents.clinic.get_knowledge_base", return_value=FakeKnowledgeBase()))
        stack.enter_context(patch("agents.clinic.create_specialist_agent", return_value=_FakeReactAgent(ANSWER)))
        stack.enter_context(patch("agents.insurance.get_chat_llm", return_value=insurance or scripted(
            AIMessage(content="医保账户余额已查询。"))))
        stack.enter_context(patch("agents.advisor.get_chat_llm", return_value=scripted(
            AIMessage(content="已结束本次问诊。"))))
        stack.enter_context(patch("agents.advisor.get_knowledge_base", return_value=FakeKnowledgeBase()))

    async def test_handoff_return_and_new_interview_keep_history_without_duplicates(self):
        model = ReplyModel([PARTIAL, COMPLETE], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")])
        insurance = scripted(
            AIMessage(content="医保账户余额已查询。"),
            tool_call("transfer_to_clinic", {"reason": "继续刚才问诊"}),
            AIMessage(content="为您转接问诊。"),
        )
        self.use_models(model, insurance=insurance)
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "clinic-roundtrip"}}
        first = await graph.ainvoke(state("我最近有点咳嗽"), config)
        self.assertIn("__interrupt__", first)

        second = await graph.ainvoke(Command(resume="先查一下医保余额"), config)
        self.assertEqual(second["active_agent"], "insurance_agent")
        saved = await clinic_state(graph, config)
        self.assertEqual(saved["interview_status"], "paused")
        self.assertEqual(saved["collected"]["chief_complaint"], "咳嗽")
        self.assertEqual(saved["pending_question"], QUESTION)
        self.assertEqual(saved["followup_count"], 0)
        self.assertIn("先查一下医保余额", [m.content for m in insurance.invocations[0]])

        third = await graph.ainvoke(state("继续刚才的问诊", active="insurance_agent"), config)
        self.assertEqual(third["__interrupt__"][0].value["question"], QUESTION)
        final = await graph.ainvoke(Command(resume="三天了，中等程度"), config)
        self.assertNotIn("__interrupt__", final)
        texts = [m.content for m in final["messages"]]
        self.assertEqual(texts.count(QUESTION), 2)  # Original question and intentional re-ask.
        for text in ("我最近有点咳嗽", "先查一下医保余额", "医保账户余额已查询。",
                     "继续刚才的问诊", "三天了，中等程度", ANSWER):
            self.assertEqual(texts.count(text), 1, texts)
        # Routing and extraction still receive the complete shared history.
        self.assertIn("医保账户余额已查询。", [m.content for m in model.routing_inputs[-1]])
        saved = await clinic_state(graph, config)
        self.assertEqual(saved["interview_status"], "completed")
        for field, empty in (("collected", {}), ("turn_messages", []), ("pending_question", ""),
                             ("followup_count", 0), ("confirm_count", 0), ("facts_confirmed", False)):
            self.assertEqual(saved[field], empty)

        model.facts_queue.append(clinic.SymptomFacts(chief_complaint="头疼"))
        await graph.ainvoke(state("现在我头疼"), config)
        saved = await clinic_state(graph, config)
        self.assertEqual(saved["collected"]["chief_complaint"], "头疼")
        self.assertEqual(saved["collected"].get("duration", ""), "")
        self.assertEqual(saved["followup_count"], 0)

    async def test_direct_answer_on_reentry_is_not_appended_twice(self):
        self.use_models(ReplyModel([PARTIAL, COMPLETE], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")]))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "direct-reentry"}}
        await graph.ainvoke(state("我咳嗽"), config)
        await graph.ainvoke(Command(resume="医保余额是多少"), config)
        result = await graph.ainvoke(state("三天了，中等程度"), config)
        texts = [m.content for m in result["messages"]]
        self.assertEqual(texts.count("三天了，中等程度"), 1)
        self.assertEqual(texts.count(QUESTION), 1)
        self.assertEqual(texts[-1], ANSWER)

    async def test_returning_after_an_ordinary_handoff_without_a_pending_question_concludes(self):
        class LateHandoffAgent:
            calls = 0

            async def ainvoke(self, payload, **kwargs):
                self.calls += 1
                produced = [AIMessage(content=ANSWER)]
                if self.calls == 1:
                    produced = [tool_call("transfer_to_insurance", call_id="late"),
                                ToolMessage(content='{"handoff":"insurance_agent"}',
                                            name="transfer_to_insurance", tool_call_id="late")]
                return {"messages": [*payload["messages"], *produced]}

        self.use_models(ReplyModel([COMPLETE]))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "late-handoff-return"}}
        with patch("agents.clinic.create_specialist_agent", return_value=LateHandoffAgent()):
            await graph.ainvoke(state("咳嗽三天，中等，另外查医保"), config)
            saved = await clinic_state(graph, config)
            self.assertEqual(saved["interview_status"], "paused")
            self.assertEqual(saved["pending_question"], "")
            final = await graph.ainvoke(state("继续问诊"), config)
        self.assertNotIn("__interrupt__", final)
        self.assertEqual(final["messages"][-1].content, ANSWER)

    async def test_cancelling_before_reentry_does_not_reexport_previous_history(self):
        self.use_models(ReplyModel([COMPLETE]))
        entered = asyncio.Event()
        original_entry = clinic.enter_interview

        async def blocked_entry(values):
            if values.get("interview_status") == "completed":
                entered.set()
                await asyncio.Event().wait()
            return original_entry(values)

        with patch.object(clinic, "enter_interview", blocked_entry):
            child = clinic.build_clinic_graph()
        with patch("agents.graph.clinic_node", child):
            graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "cancel-before-entry"}}
        await graph.ainvoke(state("我咳嗽三天了，中等"), config)
        task = asyncio.create_task(graph.ainvoke(state("我有一个新问题"), config))
        try:
            await asyncio.wait_for(entered.wait(), 3)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        with patch("dotenv.load_dotenv"):
            import main
        with patch.object(main, "_master_app", graph):
            await main._abandon_pending_run(config)
        snapshot = await graph.aget_state(config)
        self.assertEqual(snapshot.next, ())
        self.assertEqual([m.content for m in snapshot.values["messages"]],
                         ["我咳嗽三天了，中等", ANSWER, "我有一个新问题"])

    async def test_voice_confirmation_handoff_does_not_consume_confirmation_budget(self):
        self.use_models(ReplyModel([COMPLETE], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")]))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "confirmation-handoff", "channel": "voice"}}
        first = await graph.ainvoke(state("我咳嗽三天了，中等", channel="voice"), config)
        question = first["__interrupt__"][0].value["question"]
        await graph.ainvoke(Command(resume="先查医保余额"), config)
        saved = await clinic_state(graph, config)
        self.assertEqual(saved["pending_kind"], "confirm")
        self.assertEqual(saved["pending_question"], question)
        self.assertEqual(saved["confirm_count"], 0)
        self.assertFalse(saved.get("facts_confirmed", False))
        resumed = await graph.ainvoke(state("继续问诊", channel="voice"), config)
        self.assertEqual(resumed["__interrupt__"][0].value["kind"], "confirm")
        final = await graph.ainvoke(Command(resume="对"), config)
        texts = [m.content for m in final["messages"]]
        self.assertEqual(texts.count(question), 2)
        self.assertEqual(texts.count("对"), 1)
        self.assertEqual(texts[-1], ANSWER)

    async def test_paused_voice_confirmation_can_finish_in_text_without_another_readback(self):
        self.use_models(ReplyModel([COMPLETE], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")]))
        graph = build_graph(checkpointer=MemorySaver())
        voice = {"configurable": {"thread_id": "voice-to-text", "channel": "voice"}}
        await graph.ainvoke(state("我咳嗽三天了，中等", channel="voice"), voice)
        await graph.ainvoke(Command(resume="先查医保余额"), voice)
        text = {"configurable": {"thread_id": "voice-to-text", "channel": "text"}}
        final = await graph.ainvoke(state("继续问诊"), text)
        self.assertNotIn("__interrupt__", final)
        self.assertEqual(final["messages"][-1].content, ANSWER)

    async def test_ambiguous_input_clarifies_without_consuming_a_followup(self):
        self.use_models(ReplyModel([PARTIAL, COMPLETE], [clinic.ClinicReplyDecision(
            action="clarify", question="您想继续问诊吗？")]))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "clarify"}}
        await graph.ainvoke(state("我咳嗽"), config)
        clarification = await graph.ainvoke(Command(resume="嗯？"), config)
        self.assertEqual(clarification["__interrupt__"][0].value["question"], "您想继续问诊吗？")
        saved = await clinic_state(graph, config)
        self.assertEqual(saved["followup_count"], 0)
        self.assertEqual(saved["pending_question"], QUESTION)
        final = await graph.ainvoke(Command(resume="三天了，中等"), config)
        texts = [m.content for m in final["messages"]]
        self.assertEqual(texts.count("嗯？"), 1)
        self.assertEqual(texts.count("您想继续问诊吗？"), 1)

    async def test_cancel_clears_clinic_but_preserves_the_conversation(self):
        self.use_models(ReplyModel([PARTIAL]))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "cancel-clinic"}}
        await graph.ainvoke(state("我咳嗽"), config)
        final = await graph.ainvoke(Command(resume="结束问诊"), config)
        self.assertEqual(final["active_agent"], "advisor_agent")
        self.assertEqual([m.content for m in final["messages"]].count("结束问诊"), 1)
        saved = await clinic_state(graph, config)
        self.assertEqual(saved["interview_status"], "completed")
        self.assertEqual(saved["collected"], {})
        self.assertEqual(saved["turn_messages"], [])

    async def test_emergency_in_confirmation_runs_before_reply_routing_and_resets_state(self):
        self.use_models(ReplyModel([COMPLETE]))
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "confirmation-emergency", "channel": "voice"}}
        await graph.ainvoke(state("我咳嗽三天了，中等", channel="voice"), config)
        with patch("agents.clinic.get_chat_llm", side_effect=AssertionError("No routing or conclusion model")):
            final = await graph.ainvoke(Command(resume="现在胸口剧痛，喘不过气"), config)
        self.assertIn("120", final["messages"][-1].content)
        self.assertEqual((await clinic_state(graph, config))["collected"], {})

    async def test_new_run_resets_the_handoff_budget_after_entering_from_another_agent(self):
        insurance = scripted(tool_call("transfer_to_clinic"), AIMessage(content="转接问诊。"),
                             AIMessage(content="医保账户余额已查询。"))
        self.use_models(ReplyModel([PARTIAL], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")]), insurance=insurance)
        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "handoff-budget"}}
        first = await graph.ainvoke(state("我咳嗽", active="insurance_agent"), config)
        self.assertEqual(first["handoff_count"], 1)
        final = await graph.ainvoke(Command(resume="先查医保余额"), config)
        self.assertEqual(final["active_agent"], "insurance_agent")
        self.assertEqual(final["handoff_count"], 1)
        self.assertEqual((await clinic_state(graph, config))["interview_status"], "paused")

    async def test_sqlite_preserves_paused_state_across_graph_instances_and_isolates_threads(self):
        self.use_models(ReplyModel([PARTIAL, clinic.SymptomFacts(chief_complaint="头疼"), COMPLETE],
                                  [clinic.ClinicReplyDecision(action="handoff", target_agent="insurance_agent")]))
        with tempfile.TemporaryDirectory() as directory:
            env = {"CHECKPOINTER": "sqlite", "CHECKPOINT_DB_PATH": str(Path(directory) / "checkpoints.sqlite3")}
            savers = [create_checkpointer(env), create_checkpointer(env)]
            first, second = [build_graph(checkpointer=saver) for saver in savers]
            config = {"configurable": {"thread_id": "shared-clinic"}}
            other = {"configurable": {"thread_id": "other-clinic"}}
            try:
                await first.ainvoke(state("我咳嗽"), config)
                await second.ainvoke(Command(resume="查医保余额"), config)
                await second.ainvoke(state("我头疼"), other)
                self.assertEqual((await clinic_state(first, config))["collected"]["chief_complaint"], "咳嗽")
                self.assertEqual((await clinic_state(first, other))["collected"]["chief_complaint"], "头疼")
                final = await first.ainvoke(state("三天了，中等"), config)
                self.assertEqual(final["messages"][-1].content, ANSWER)
                self.assertEqual((await clinic_state(second, config))["interview_status"], "completed")
                self.assertEqual((await clinic_state(second, other))["interview_status"], "collecting")
            finally:
                for saver in savers:
                    await saver.conn.close()

    async def test_chat_sse_roundtrip_releases_each_run_and_preserves_the_thread(self):
        from test_fakes import isolated_api_runtime
        insurance = scripted(AIMessage(content="医保账户余额已查询。"),
                             tool_call("transfer_to_clinic"), AIMessage(content="转接问诊。"))
        self.use_models(ReplyModel([PARTIAL, COMPLETE], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")]), insurance=insurance)
        graph = build_graph(checkpointer=MemorySaver())
        with isolated_api_runtime() as manager:
            import main
            async def submit(text, mode):
                response = await main.chat(main.ChatRequest(
                    messages=[main.ChatMessage(role="user", content=text)], chat_mode=mode,
                    thread_id="api-roundtrip", request_id=uuid4().hex,
                ))
                chunks = [chunk async for chunk in response.body_iterator]
                return [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines()
                        if line.startswith("data: ")]

            with patch.object(main, "_master_app", graph), patch.object(main, "resolve_model_settings"):
                batches = [await submit("我咳嗽", "clinic"),
                           await submit("先查医保余额", "clinic"),
                           await submit("继续刚才的问诊", "insurance"),
                           await submit("三天了，中等", "clinic")]
            self.assertEqual([batch[-1]["status"] for batch in batches],
                             ["waiting_input", "success", "waiting_input", "success"])
            run_ids = [batch[-1]["run_id"] for batch in batches]
            self.assertEqual(len(set(run_ids)), 4)
            for batch, run_id in zip(batches, run_ids):
                self.assertEqual(batch[0]["thread_id"], "api-roundtrip")
                self.assertEqual(manager.store.get(run_id)["thread_id"], "api-roundtrip")
            self.assertFalse(manager.store.active_thread("api-roundtrip"))
            self.assertFalse(manager.store.dirty("api-roundtrip"))


if __name__ == "__main__":
    unittest.main()
