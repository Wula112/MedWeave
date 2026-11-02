"""Offline tests for one-turn Mem0 capture at the outer Run boundary."""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from memory.middleware import MemoryMiddleware
from memory.prompts import CUSTOM_INSTRUCTIONS, instructions_for
from runtime.config import RuntimeSettings
from runtime.runs import RunManager


async def noop(*args) -> None:
    pass


class PromptTests(unittest.TestCase):
    def test_china_date_is_appended_without_changing_user_instructions(self):
        observed = datetime(2026, 10, 31, 16, 30, tzinfo=timezone.utc)
        result = instructions_for(observed)
        self.assertTrue(result.startswith(CUSTOM_INSTRUCTIONS + "\n\n## Observation Date(China)\n"))
        self.assertIn("2026-11-01 00:30:00", result)
        self.assertIn("Asia/Shanghai", result)
        self.assertIn("UTC+08:00", result)

    def test_missing_key_disables_capture(self):
        with patch.dict(os.environ, {"MEM0_API_KEY": ""}, clear=True):
            self.assertIsNone(MemoryMiddleware.from_env(Path(".")))

    def test_api_runtime_enables_memory_from_environment(self):
        with patch("dotenv.load_dotenv"):
            import main
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"MEM0_API_KEY": "test-key", "MEM0_USER_ID": "test-user"}), \
             patch.object(main, "_runtime", None), \
             patch.object(main, "get_settings", return_value=RuntimeSettings(data_dir=Path(directory))):
            manager = main.get_runtime()
            self.assertIsInstance(manager.middleware, MemoryMiddleware)


class Mem0SdkTests(unittest.TestCase):
    def test_real_sdk_sends_turn_scope_instructions_and_observation_time(self):
        payloads = []

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/ping/":
                return httpx.Response(200, json={"org_id": "org", "project_id": "project"})
            self.assertEqual(request.url.path, "/v3/memories/add/")
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "PENDING", "event_id": "test-event"})

        messages = [{"role": "user", "content": "我对坚果过敏。"},
                    {"role": "assistant", "content": "我会记下您提到的过敏信息。"}]
        observed = datetime(2026, 11, 1, 0, 30, tzinfo=timezone.utc)
        real_client = httpx.Client
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "MEM0_DIR": directory, "MEM0_TELEMETRY": "false",
        }), patch("memory.middleware.httpx.Client", side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(respond), timeout=kwargs.get("timeout"))):
            MemoryMiddleware("test-key", "test-user", Path(directory))._add("clinic_agent", messages, observed)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["messages"], messages)
        self.assertEqual(payloads[0]["user_id"], "test-user")
        self.assertEqual(payloads[0]["agent_id"], "clinic_agent")
        self.assertNotIn("run_id", payloads[0])
        self.assertNotIn("app_id", payloads[0])
        self.assertEqual(payloads[0]["timezone"], "Asia/Shanghai")
        self.assertEqual(payloads[0]["observation_datetime"], observed.isoformat())
        self.assertEqual(payloads[0]["custom_instructions"], instructions_for(observed))


class RunMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = MemoryMiddleware("test-key", "test-user", Path(self.temp.name))
        self.manager = RunManager(RuntimeSettings(data_dir=Path(self.temp.name)), middleware=self.memory)

    async def asyncTearDown(self):
        await self.manager.shutdown()
        self.temp.cleanup()

    async def start(self, source, *, text="本轮问题", agent="clinic_agent", thread="thread"):
        return await self.manager.start(
            thread_id=thread, request_id=None, fingerprint="test", channel="text",
            source=source, cleanup=noop, user_text=text, agent_name=agent,
        )

    async def test_completed_turn_is_dispatched_once_without_delaying_response(self):
        entered = threading.Event()
        release = threading.Event()
        submitted = []

        def add(agent, messages, observed):
            entered.set()
            release.wait(3)
            submitted.append((agent, messages))

        async def source(context):
            yield {"type": "node_start", "node": "clinic_node"}
            yield {"type": "text", "content": "前半句，"}
            yield {"type": "node_start", "node": "insurance_node"}
            yield {"type": "text", "content": "后半句。"}

        with patch.object(self.memory, "_add", side_effect=add):
            try:
                handle = await self.start(source, text="请查询医保")
                events = await asyncio.wait_for(self.collect(handle), 2)
                self.assertEqual(events[-1]["type"], "finish")
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.assertEqual(submitted, [])
            finally:
                release.set()
            await self.memory.shutdown(3)
        self.assertEqual(submitted, [("insurance_agent", [
            {"role": "user", "content": "请查询医保"},
            {"role": "assistant", "content": "前半句，后半句。"},
        ])])

    async def test_interrupt_is_a_completed_run_and_failures_do_not_send_partial_text(self):
        submitted = []

        def add(agent, messages, observed):
            submitted.append((agent, messages))

        async def interrupted(context):
            yield {"type": "node_start", "node": "ask_followup"}
            yield {"type": "text", "content": "持续多久了？"}
            yield {"type": "interrupt", "content": "持续多久了？"}

        async def failure(context):
            yield {"type": "text", "content": "未完成的回答"}
            raise RuntimeError("private details")

        async def cancelled(context):
            yield {"type": "text", "content": "未完成的回答"}
            await asyncio.Event().wait()

        with patch.object(self.memory, "_add", side_effect=add):
            first = await self.start(interrupted, text="我头疼")
            events = await self.collect(first)
            self.assertEqual(events[-1]["status"], "waiting_input")
            failed = await self.start(failure, text="第二条")
            self.assertEqual((await self.collect(failed))[-1]["type"], "error")
            third = await self.start(cancelled, text="第三条")
            consumer = asyncio.create_task(self.collect(third))
            await asyncio.sleep(0.03)
            await third.cancel()
            await consumer
            await self.memory.shutdown(3)
        self.assertEqual(submitted, [("clinic_agent", [
            {"role": "user", "content": "我头疼"},
            {"role": "assistant", "content": "持续多久了？"},
        ])])
        self.assertEqual(self.memory._turns, {})

    async def test_provider_failure_is_isolated_and_does_not_log_message_or_key(self):
        async def source(context):
            yield {"type": "text", "content": "助手回答"}

        with self.assertLogs("memory.middleware", level="WARNING") as captured:
            with patch.object(self.memory, "_add", side_effect=ValueError("test-key private-health-text")):
                handle = await self.start(source, text="private-health-text")
                self.assertEqual((await self.collect(handle))[-1]["type"], "finish")
                await self.memory.shutdown(3)
        self.assertNotIn("test-key", " ".join(captured.output))
        self.assertNotIn("private-health-text", " ".join(captured.output))

    async def test_accepted_submission_logs_event_id_for_status_tracking(self):
        async def source(context):
            yield {"type": "text", "content": "助手回答"}

        event_id = "0f3e88d7-079a-4ac5-a192-36fc0893b1f7"
        with self.assertLogs("memory.middleware", level="INFO") as captured:
            with patch.object(self.memory, "_add", return_value={"status": "PENDING", "event_id": event_id}):
                handle = await self.start(source)
                await self.collect(handle)
                await self.memory.shutdown(3)
        self.assertIn(event_id, " ".join(captured.output))
        self.assertNotIn("本轮问题", " ".join(captured.output))

    async def test_observation_time_is_fixed_when_the_turn_starts(self):
        observed = datetime(2026, 10, 31, 15, 59, 30, tzinfo=timezone.utc)
        submitted = []
        with patch("memory.middleware.datetime") as clock, \
             patch.object(self.memory, "_add", side_effect=lambda agent, messages, at: submitted.append(at)):
            clock.now.return_value = observed
            self.memory.before_run("run-time", "测试问题", "clinic_agent")
            self.memory.on_event("run-time", {"type": "text", "content": "测试回答"})
            self.memory.after_run("run-time", "success")
            await self.memory.shutdown(3)
            clock.now.assert_called_once()
        self.assertEqual(submitted, [observed])

    async def test_real_graph_interruption_and_handoff_capture_the_current_turn(self):
        with patch("dotenv.load_dotenv"):
            import main
        from agents import clinic
        from agents.graph import build_graph
        from test_clinic_persistence import ANSWER, COMPLETE, PARTIAL, QUESTION, ReplyModel
        from test_fakes import FakeKnowledgeBase, scripted, tool_call

        model = ReplyModel([PARTIAL, COMPLETE], [clinic.ClinicReplyDecision(
            action="handoff", target_agent="insurance_agent")])
        insurance = scripted(
            AIMessage(content="医保账户余额已查询。"),
            tool_call("transfer_to_clinic", {"reason": "继续刚才问诊"}),
            AIMessage(content="为您转接问诊。"),
        )
        graph = build_graph(checkpointer=MemorySaver())
        submitted = []

        def add(agent, messages, observed):
            submitted.append((agent, messages))

        with patch.object(main, "_runtime", self.manager), patch.object(main, "_master_app", graph), \
             patch("agents.clinic.get_chat_llm", side_effect=lambda preset, **kwargs:
                   model if preset in {"precise", "router"} else scripted(AIMessage(content=ANSWER))), \
             patch("agents.clinic.get_agent_tools", return_value=[]), \
             patch("agents.clinic.get_knowledge_base", return_value=FakeKnowledgeBase()), \
             patch("agents.insurance.get_chat_llm", return_value=insurance), \
             patch.object(self.memory, "_add", side_effect=add):
            first = [e async for e in main._voice_run_events("我最近有点咳嗽", "real", {}, 8)]
            self.assertTrue(any(e["type"] == "interrupt" for e in first))
            await self.memory.shutdown(3)
            second = [e async for e in main._voice_run_events("先查一下医保余额", "real", {}, 8)]
            self.assertEqual(second[-1]["type"], "finish")
            await self.memory.shutdown(3)
            third = [e async for e in main._voice_run_events("继续刚才的问诊", "real", {}, 8)]
            self.assertTrue(any(e["type"] == "interrupt" for e in third))
            await self.memory.shutdown(3)
            fourth = [e async for e in main._voice_run_events("三天了，中等程度", "real", {}, 8)]
            self.assertTrue(any(e["type"] == "interrupt" for e in fourth))
            await self.memory.shutdown(3)
            fifth = [e async for e in main._voice_run_events("对", "real", {}, 8)]
            self.assertEqual(fifth[-1]["type"], "finish")
            await self.memory.shutdown(3)

        self.assertEqual(len(submitted), 5)
        self.assertEqual(submitted[0], ("clinic_agent", [
            {"role": "user", "content": "我最近有点咳嗽"},
            {"role": "assistant", "content": QUESTION},
        ]))
        self.assertEqual(submitted[1], ("insurance_agent", [
            {"role": "user", "content": "先查一下医保余额"},
            {"role": "assistant", "content": "医保账户余额已查询。"},
        ]))
        self.assertEqual(submitted[2][0], "clinic_agent")
        self.assertEqual(submitted[2][1][0]["content"], "继续刚才的问诊")
        self.assertIn(QUESTION, submitted[2][1][1]["content"])
        self.assertEqual(submitted[3][0], "clinic_agent")
        self.assertEqual(submitted[3][1][0]["content"], "三天了，中等程度")
        self.assertIn("确认一下", submitted[3][1][1]["content"])
        self.assertEqual(submitted[4][0], "clinic_agent")
        self.assertEqual(submitted[4][1][0]["content"], "对")
        self.assertIn(ANSWER, submitted[4][1][1]["content"])

    async def test_vision_uses_latest_user_text_and_never_sends_image_data(self):
        with patch("dotenv.load_dotenv"):
            import main
        from agents.graph import build_graph
        from starlette.datastructures import Headers, UploadFile
        from test_fakes import FakeKnowledgeBase, scripted

        graph = build_graph(checkpointer=MemorySaver())
        submitted = []

        def add(agent, messages, observed):
            submitted.append((agent, messages))

        upload = UploadFile(file=io.BytesIO(b"test-image-bytes"), filename="test.png",
                            headers=Headers({"content-type": "image/png"}))
        history = json.dumps([
            {"role": "user", "content": "上一轮用户消息"},
            {"role": "assistant", "content": "上一轮助手答复"},
            {"role": "user", "content": "请帮我看看这张报告"},
        ], ensure_ascii=False)
        with patch.object(main, "_runtime", self.manager), patch.object(main, "_master_app", graph), \
             patch("main.resolve_model_settings"), \
             patch("main.recognize_image", return_value="视觉识别内容"), \
             patch("agents.report.get_chat_llm", return_value=scripted(AIMessage(content="报告解读答复。"))), \
             patch("agents.report.get_agent_tools", return_value=[]), \
             patch("agents.report.get_knowledge_base", return_value=FakeKnowledgeBase()), \
             patch.object(self.memory, "_add", side_effect=add):
            response = await main.vision_chat(file=upload, scan_type="report", user_info="{}",
                                              messages=history, thread_id="vision", request_id="")
            [chunk async for chunk in response.body_iterator]
            await self.memory.shutdown(3)

        self.assertEqual(submitted, [("report_agent", [
            {"role": "user", "content": "请帮我看看这张报告"},
            {"role": "assistant", "content": "报告解读答复。"},
        ])])

    async def test_text_api_with_history_only_submits_the_newest_turn(self):
        with patch("dotenv.load_dotenv"):
            import main
        from agents.graph import build_graph
        from test_fakes import scripted

        graph = build_graph(checkpointer=MemorySaver())
        submitted = []

        def add(agent, messages, observed):
            submitted.append((agent, messages))

        request = main.ChatRequest(messages=[
            main.ChatMessage(role="user", content="先前的问题"),
            main.ChatMessage(role="assistant", content="先前的回答"),
            main.ChatMessage(role="user", content="最新的问题"),
        ], chat_mode="insurance", thread_id="text-history")
        with patch.object(main, "_runtime", self.manager), patch.object(main, "_master_app", graph), \
             patch("main.resolve_model_settings"), \
             patch("agents.insurance.get_chat_llm", return_value=scripted(AIMessage(content="最新的回答。"))), \
             patch.object(self.memory, "_add", side_effect=add):
            response = await main.chat(request)
            [chunk async for chunk in response.body_iterator]
            await self.memory.shutdown(3)

        self.assertEqual(submitted, [("insurance_agent", [
            {"role": "user", "content": "最新的问题"},
            {"role": "assistant", "content": "最新的回答。"},
        ])])

    @staticmethod
    async def collect(handle):
        return [event async for event in handle.events()]


if __name__ == "__main__":
    unittest.main()
