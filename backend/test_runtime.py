"""Offline execution tests, including real independent worker processes."""
from __future__ import annotations

import asyncio
import multiprocessing
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage

from test_fakes import ScriptedChatModel, StructuredChatModel
from runtime.config import RuntimeSettings
from runtime.errors import RuntimeFailure
from runtime.runs import RunManager
from runtime.store import RunStore, ThreadLock


def settings(directory: str, **overrides) -> RuntimeSettings:
    return RuntimeSettings(data_dir=Path(directory), heartbeat_seconds=0.05,
                           lease_seconds=2, retry_base=0, retry_cap=0, **overrides)


async def noop(*args) -> None:
    pass


async def response(context):
    yield {"type": "text", "content": "ok"}


def independent_writer(directory, ready, release):
    """Spawn-safe worker that deliberately stays alive after lease expiry."""
    store = RunStore(settings(directory))
    lock = ThreadLock(store.directory / "locks", "shared")
    row = store.admit("shared", "original", "input", "child", "text", 120)
    store.owned_update(row["run_id"], "child", graph=True, status="running")
    ready.put(row["run_id"])
    release.wait(30)
    lock.close()


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.first = RunManager(settings(self.temp.name))
        self.second = RunManager(settings(self.temp.name))

    async def asyncTearDown(self):
        await self.first.shutdown()
        await self.second.shutdown()
        self.temp.cleanup()

    async def start(self, manager=None, thread="t", request=None, fingerprint="input", source=response, **kwargs):
        return await (manager or self.first).start(thread_id=thread, request_id=request,
            fingerprint=fingerprint, channel="text", source=source, cleanup=noop, **kwargs)

    async def test_workers_share_thread_exclusion_and_different_threads_can_run(self):
        first = await self.start()
        with self.assertRaises(RuntimeFailure) as raised:
            await self.start(self.second)
        self.assertEqual(raised.exception.code, "thread_busy")
        other = await self.start(self.second, thread="other")
        self.assertEqual([e["type"] async for e in other.events()], ["text", "finish"])
        self.assertEqual([e["type"] async for e in first.events()], ["text", "finish"])

    async def test_header_send_disconnect_releases_admitted_run(self):
        with patch("dotenv.load_dotenv"):
            import main
        from starlette.requests import ClientDisconnect

        handle = await self.start()

        async def send(message):
            raise OSError("connection closed")

        with self.assertRaises(ClientDisconnect):
            await main._run_response(handle)({"type": "http", "asgi": {"spec_version": "2.4"}}, noop, send)
        self.assertTrue(handle.done.is_set())
        self.assertEqual(self.first.store.get(handle.context.run_id)["status"], "cancelled")
        await self.collect(await self.start(self.second))

    async def test_asgi_response_delivers_one_success_terminal(self):
        with patch("dotenv.load_dotenv"):
            import main
        handle = await self.start()
        sent = []

        async def send(message):
            sent.append(message)

        await main._run_response(handle)({"type": "http", "asgi": {"spec_version": "2.4"}}, noop, send)
        body = b"".join(message.get("body", b"") for message in sent).decode()
        self.assertEqual(body.count('"type": "finish"'), 1)
        self.assertNotIn('"type": "error"', body)
        self.assertEqual(self.first.store.get(handle.context.run_id)["status"], "success")

    async def test_duplicate_and_changed_request_are_rejected_across_workers(self):
        first = await self.start(request="id")
        for fingerprint, code in (("input", "duplicate_request"), ("changed", "request_id_conflict")):
            with self.assertRaises(RuntimeFailure) as raised:
                await self.start(self.second, request="id", fingerprint=fingerprint)
            self.assertEqual(raised.exception.code, code)
        await first.cancel()
        with self.assertRaises(RuntimeFailure):
            await self.start(self.second, request="id")

    async def test_timeout_cleans_checkpoint_before_releasing_thread(self):
        cleanup = []

        async def hung(context):
            await context.begin_graph()
            await asyncio.Event().wait()
            yield {}

        async def repair(thread):
            cleanup.append(thread)

        handle = await self.first.start(thread_id="t", request_id=None, fingerprint="input",
            channel="text", source=hung, cleanup=repair, timeout=0.5)
        events = [event async for event in handle.events()]
        self.assertEqual(events[-1]["code"], "run_timeout")
        self.assertEqual(cleanup, ["t"])
        self.assertEqual(self.first.store.get(handle.context.run_id)["status"], "timeout")
        next_run = await self.start(self.second)
        self.assertEqual([e["type"] async for e in next_run.events()][-1], "finish")

    async def test_cross_worker_cancel_wins_over_success(self):
        entered = asyncio.Event()

        async def hung(context):
            entered.set()
            await asyncio.Event().wait()
            yield {}

        handle = await self.start(source=hung)
        consumer = asyncio.create_task(self.collect(handle))
        await entered.wait()
        result = await self.second.cancel(handle.context.run_id, "t")
        self.assertTrue(result["cancel_requested"])
        events = await asyncio.wait_for(consumer, 2)
        self.assertEqual(events[-1]["code"], "run_cancelled")
        self.assertEqual(self.first.store.get(handle.context.run_id)["status"], "cancelled")

    async def test_cancelled_router_cleanup_does_not_leave_a_scheduled_specialist(self):
        with patch("dotenv.load_dotenv"):
            import main
        from agents.graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage

        entered = asyncio.Event()

        class BlockedModel(ScriptedChatModel):
            disable_streaming: bool = True

            async def _agenerate(self, messages, **kwargs):
                entered.set()
                await asyncio.Event().wait()

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "router-cancel"}}
        with patch("agents.router.get_chat_llm", return_value=BlockedModel()), patch.object(main, "_master_app", graph):
            task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage(content="hello")],
                "active_agent": "", "next_agent": "", "user_info": {}, "handoff_count": 0}, config))
            await entered.wait()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await main._abandon_pending_run(config)
            snapshot = await graph.aget_state(config)
            self.assertEqual(snapshot.next, ())
            self.assertEqual([m.content for m in snapshot.values["messages"]], ["hello"])

    async def test_repeated_cancellation_cannot_release_a_cleaning_thread(self):
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def hung(context):
            await context.begin_graph()
            entered.set()
            await asyncio.Event().wait()
            yield {}

        async def cleanup(thread):
            cleaning.set()
            await release.wait()

        handle = await self.first.start(thread_id="t", request_id=None, fingerprint="i",
            channel="text", source=hung, cleanup=cleanup)
        consumer = asyncio.create_task(self.collect(handle))
        await entered.wait()
        handle.task.cancel()
        await cleaning.wait()
        handle.task.cancel()
        with self.assertRaises(RuntimeFailure):
            await self.start(self.second)
        release.set()
        await consumer
        next_run = await self.start(self.second)
        await self.collect(next_run)

    async def test_global_capacity_is_shared_and_recovers_after_completion(self):
        self.first.settings.max_concurrent_runs = 1
        self.second.settings.max_concurrent_runs = 1
        first = await self.start()
        with self.assertRaises(RuntimeFailure) as raised:
            await self.start(self.second, thread="other")
        self.assertEqual(raised.exception.code, "runtime_busy")
        await first.cancel()
        await self.collect(await self.start(self.second, thread="other"))

    async def test_failed_cleanup_is_repaired_before_next_graph_starts(self):
        async def failure(context):
            await context.begin_graph()
            raise ValueError("private medical input must not reach the client")
            yield {}

        async def broken(thread):
            raise OSError("unavailable")

        handle = await self.first.start(thread_id="t", request_id=None, fingerprint="i", channel="text",
                                        source=failure, cleanup=broken)
        events = await self.collect(handle)
        self.assertEqual(events[-1]["code"], "checkpoint_cleanup_failed")
        self.assertNotIn("private", str(events))
        self.assertTrue(self.first.store.dirty("t"))
        repaired = []

        async def cleanup(thread):
            repaired.append(thread)

        handle = await self.second.start(thread_id="t", request_id=None, fingerprint="j", channel="text",
                                         source=response, cleanup=cleanup)
        await self.collect(handle)
        self.assertEqual(repaired, ["t"])
        self.assertFalse(self.second.store.dirty("t"))

    async def test_model_clients_close_before_run_completion(self):
        from agents.llm import get_chat_llm
        client = SimpleNamespace(root_async_client=SimpleNamespace(close=AsyncMock()),
                                 root_client=SimpleNamespace(close=Mock()))

        async def source(context):
            get_chat_llm()
            yield {"type": "text", "content": "ok"}

        with patch.dict("os.environ", {"LLM_PROVIDER": "custom", "LLM_API_KEY": "test", "LLM_MODEL": "test",
                        "LLM_BASE_URL": "https://model.invalid"}), patch("agents.llm.ChatOpenAI", return_value=client):
            await self.collect(await self.start(source=source))
        client.root_async_client.close.assert_awaited_once()
        client.root_client.close.assert_called_once()

    async def test_live_expired_process_is_fenced_and_dead_owner_is_recovered(self):
        mp = multiprocessing.get_context("spawn")
        ready, release = mp.Queue(), mp.Event()
        process = mp.Process(target=independent_writer, args=(self.temp.name, ready, release))
        process.start()
        try:
            run_id = await asyncio.to_thread(ready.get, True, 20)
            with closing(sqlite3.connect(self.first.store.path)) as db:
                with db:
                    db.execute("UPDATE runs SET lease_expires=? WHERE run_id=?", (time.time() - 1, run_id))
            await self.first.recover_expired()
            self.assertEqual(self.first.store.get(run_id)["status"], "running")
            with self.assertRaises(RuntimeFailure):
                await self.start(thread="shared")
            process.terminate()
            await asyncio.to_thread(process.join, 5)
            await self.first.recover_expired()
            self.assertEqual(self.first.store.get(run_id)["stop_reason"], "orphan_recovered")
            await self.collect(await self.start(thread="shared"))
            with self.assertRaises(RuntimeFailure):
                self.first.store.owned_update(run_id, "child", status="success")
        finally:
            # A deliberately terminated process may have held Event's internal
            # condition lock. Never touch that IPC object after simulating crash.
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 5)
            ready.close()

    async def test_shared_sqlite_checkpoint_resumes_on_another_worker(self):
        from typing import TypedDict
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import Command, interrupt
        from agents.checkpointing import create_checkpointer
        from runtime.events import iter_graph_events

        class State(TypedDict, total=False):
            answer: str

        def interview(state):
            return {"answer": interrupt({"question": "多久了？"})}

        env = {"CHECKPOINTER": "sqlite", "CHECKPOINT_DB_PATH": str(Path(self.temp.name) / "checkpoints.sqlite")}
        savers = [create_checkpointer(env), create_checkpointer(env)]
        graphs = []
        for saver in savers:
            builder = StateGraph(State)
            builder.add_node("interview", interview)
            builder.add_edge(START, "interview")
            builder.add_edge("interview", END)
            graphs.append(builder.compile(checkpointer=saver))

        def source(graph, value):
            async def execute(context):
                await context.begin_graph()
                async for event in iter_graph_events(graph, value,
                    {"configurable": {"thread_id": "shared-interview"}}, node_labels={}, tool_labels={}, custom_types=set()):
                    yield event
            return execute

        try:
            first = await self.start(thread="shared-interview", source=source(graphs[0], {"answer": ""}))
            events = await self.collect(first)
            self.assertEqual(events[-1]["status"], "waiting_input")
            second = await self.start(self.second, thread="shared-interview", source=source(graphs[1], Command(resume="三天")))
            events = await self.collect(second)
            self.assertEqual(events[-1]["status"], "success")
            snapshot = await graphs[1].aget_state({"configurable": {"thread_id": "shared-interview"}})
            self.assertEqual(snapshot.values["answer"], "三天")
        finally:
            for saver in savers:
                await saver.conn.close()

    async def test_managed_voice_stream_preserves_confirmation_interrupt(self):
        with patch("dotenv.load_dotenv"):
            import main
        from agents.graph import build_graph
        from agents.clinic import InterviewStep, SymptomFacts
        from langgraph.checkpoint.memory import MemorySaver

        class InterviewModel(StructuredChatModel):
            async def structured_reply(self, schema_name, messages):
                return InterviewStep(facts=SymptomFacts(chief_complaint="咳嗽", duration="三天", severity="中等"))

        graph = build_graph(checkpointer=MemorySaver())
        with patch.object(main, "_master_app", graph), patch.object(main, "_runtime", self.first), patch(
            "agents.clinic.get_chat_llm", return_value=InterviewModel()
        ):
            events = await asyncio.wait_for(self.collect_events(main._voice_run_events("咳嗽三天", "voice-test", {}, 5)), 4)
        self.assertTrue(any(e["type"] == "interrupt" and e.get("kind") == "confirm" for e in events))

    @staticmethod
    async def collect_events(events):
        return [event async for event in events]

    @staticmethod
    async def collect(handle):
        return [event async for event in handle.events()]


class RetrievalPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_timed_out_waiters_do_not_restart_shared_initialization(self):
        import threading
        from rag.knowledge_base import HealthKnowledgeBase

        kb = HealthKnowledgeBase()
        release = threading.Event()
        calls = []

        def initialize():
            calls.append(1)
            release.wait(2)
            kb._initialized = True

        with patch.object(kb, "_sync_init", initialize), patch("rag.knowledge_base.get_settings",
                return_value=RuntimeSettings(retrieval_timeout=0.02)):
            try:
                for _ in range(2):
                    with self.assertRaises(RuntimeFailure) as raised:
                        await kb.aretrieve("query")
                    self.assertEqual(raised.exception.code, "retrieval_timeout")
            finally:
                release.set()
                if kb._init_task:
                    await kb._init_task
        self.assertEqual(calls, [1])


class EventConversionTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_blocks_are_forwarded_as_strings(self):
        from runtime.events import iter_graph_events

        class Graph:
            async def astream_events(self, *args, **kwargs):
                yield {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content=[
                    {"type": "text", "text": "hello"}, {"type": "output_text", "text": " world"},
                    {"type": "reasoning", "text": "hidden"},
                ])}}

        events = [event async for event in iter_graph_events(Graph(), {}, {},
            node_labels={}, tool_labels={}, custom_types=set())]
        self.assertEqual(events, [{"type": "text", "content": "hello world"}, {"type": "finish"}])


if __name__ == "__main__":
    unittest.main()
