"""Transport-independent workers. A run owns its task, lease and thread lock.

SQLite decides cross-process admission. OS locks remain held through stream
closure/checkpoint cleanup, even if cancellation takes longer than the lease.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from runtime.config import RuntimeSettings, get_settings
from runtime.context import RunContext, current_run
from runtime.errors import RuntimeFailure, public_failure
from runtime.store import RunStore, ThreadLock

logger = logging.getLogger(__name__)
EventSource = Callable[[RunContext], AsyncIterator[dict]]
Cleanup = Callable[[str], Awaitable[None]]


def request_fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


async def drain(task: asyncio.Task):
    """Wait for owned work to settle; repeated cancel must not release its lock."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@dataclass
class RunHandle:
    context: RunContext
    lock: ThreadLock
    queue: asyncio.Queue
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    terminal: dict | None = None
    cancel_reason: str = "run_cancelled"
    lease_expires: float = 0

    async def cancel(self, reason: str = "run_cancelled") -> None:
        self.cancel_reason = reason
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await drain(self.task)

    async def events(self) -> AsyncIterator[dict]:
        self.ready.set()
        try:
            while True:
                if not self.queue.empty():
                    yield self.queue.get_nowait()
                    continue
                if self.done.is_set():
                    break
                receive = asyncio.create_task(self.queue.get())
                completed = asyncio.create_task(self.done.wait())
                try:
                    await asyncio.wait((receive, completed), return_when=asyncio.FIRST_COMPLETED)
                    if receive.done():
                        yield receive.result()
                finally:
                    for task in (receive, completed):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(receive, completed, return_exceptions=True)
            if self.terminal:
                yield self.terminal
        finally:
            if not self.done.is_set():
                await self.cancel("client_disconnected")


class RunManager:
    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = RunStore(self.settings)
        self.owner = uuid4().hex
        self.live: dict[str, RunHandle] = {}
        self._closed = False

    async def start(self, *, thread_id: str, request_id: str | None, fingerprint: str,
                    channel: str, source: EventSource, cleanup: Cleanup,
                    timeout: float | None = None) -> RunHandle:
        if self._closed:
            raise RuntimeFailure("runtime_stopping", "服务正在重启，请稍后重试。", http_status=503, retryable=True)
        await asyncio.to_thread(self.store.find_request, request_id, fingerprint)
        await self.recover_expired()
        lock = ThreadLock(self.store.directory / "locks", thread_id)
        run_id = uuid4().hex
        timeout = timeout or self.settings.run_timeout
        admission = asyncio.create_task(asyncio.to_thread(
            self.store.admit, thread_id, request_id, fingerprint, self.owner, channel, timeout, run_id))
        try:
            row = await asyncio.shield(admission)
        except asyncio.CancelledError:
            try:
                await drain(admission)
                await asyncio.to_thread(self.store.owned_update, run_id, self.owner,
                                        status="cancelled", reason="admission_cancelled")
            finally:
                lock.close()
            raise
        except Exception:
            lock.close()
            raise
        context = RunContext(run_id, thread_id, channel,
                             asyncio.get_running_loop().time() + timeout, self.settings,
                             lambda: asyncio.to_thread(self.store.owned_update, run_id, self.owner, graph=True))
        handle = RunHandle(context, lock, asyncio.Queue(self.settings.event_buffer), lease_expires=row["lease_expires"])
        self.live[run_id] = handle
        handle.task = asyncio.create_task(self._drive(handle, source, cleanup), name=f"health-run-{run_id}")
        started = asyncio.create_task(handle.started.wait())
        try:
            await asyncio.shield(started)
        except asyncio.CancelledError:
            await drain(started)
            await handle.cancel("admission_cancelled")
            raise
        return handle

    async def _repair(self, context: RunContext, cleanup: Cleanup) -> None:
        async with asyncio.timeout(self.settings.cleanup_timeout):
            await cleanup(context.thread_id)
        await asyncio.to_thread(self.store.owned_update, context.run_id, self.owner, clean=True)

    async def _drive(self, handle: RunHandle, source: EventSource, cleanup: Cleanup) -> None:
        context = handle.context
        token = current_run.set(context)
        heartbeat = asyncio.create_task(self._heartbeat(handle))
        handle.started.set()
        status, failure = "error", None
        try:
            async with asyncio.timeout(context.remaining()):
                await handle.ready.wait()
                if await asyncio.to_thread(self.store.dirty, context.thread_id):
                    await self._repair(context, cleanup)
                await asyncio.to_thread(self.store.owned_update, context.run_id, self.owner, status="running")
                stream = source(context)
                try:
                    async for event in stream:
                        context.remaining()
                        if event.get("type") == "finish":
                            continue
                        if event.get("type") == "error":
                            raise RuntimeFailure(event.get("code", "execution_failed"), event.get("content", "处理失败。"),
                                                 retryable=bool(event.get("retryable")))
                        if event.get("type") == "interrupt":
                            context.waiting_input = True
                        await handle.queue.put({**event, "run_id": context.run_id})
                finally:
                    await stream.aclose()
                context.remaining()
                if context.failure:
                    raise context.failure
                status = "waiting_input" if context.waiting_input else "success"
        except asyncio.CancelledError:
            status = "cancelled"
            failure = RuntimeFailure(handle.cancel_reason, "本次处理已停止。")
        except Exception as exc:
            failure = public_failure(exc)
            status = "timeout" if failure.code == "run_timeout" else "error"
            logger.warning("Run failed run_id=%s error_type=%s code=%s", context.run_id, type(exc).__name__, failure.code)
        finally:
            # Finalization is a separate owned task so further disconnect/shutdown
            # cancellation cannot interrupt it halfway through a checkpoint write.
            finalizer = asyncio.create_task(self._finalize(handle, cleanup, status, failure))
            try:
                await drain(finalizer)
            finally:
                heartbeat.cancel()
                try:
                    await asyncio.gather(heartbeat, return_exceptions=True)
                finally:
                    current_run.reset(token)
                    self.live.pop(context.run_id, None)
                    handle.done.set()

    async def _finalize(self, handle: RunHandle, cleanup: Cleanup, status: str,
                        failure: RuntimeFailure | None) -> None:
        context = handle.context
        try:
            if status not in {"success", "waiting_input"} and context.graph_started:
                try:
                    await self._repair(context, cleanup)
                except Exception as exc:
                    logger.warning("Checkpoint cleanup failed run_id=%s error_type=%s", context.run_id, type(exc).__name__)
                    failure = RuntimeFailure("checkpoint_cleanup_failed", "会话状态暂时不可用，请稍后重试。", retryable=True)
                    status = "error"
            try:
                await asyncio.to_thread(self.store.owned_update, context.run_id, self.owner,
                                        status=status, reason=failure.code if failure else None,
                                        clean=status in {"success", "waiting_input"})
            except Exception as exc:
                failure = public_failure(exc)
                status = "error"
                if failure.code == "run_cancelled":
                    status = "cancelled"
                    await asyncio.to_thread(self.store.owned_update, context.run_id, self.owner,
                                            status=status, reason=failure.code)
                logger.warning("Run finalization failed run_id=%s error_type=%s", context.run_id, type(exc).__name__)
            handle.terminal = failure.event(context.run_id) if failure else {
                "type": "finish", "run_id": context.run_id, "status": status,
            }
        finally:
            # Tool tasks are owned by this run, including in-flight dedup waiters.
            tasks = list(context.tool_tasks.values())
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                async with asyncio.timeout(self.settings.cleanup_timeout):
                    for close in context.resources:
                        await close()
            except Exception as exc:
                logger.warning("Resource close failed run_id=%s error_type=%s", context.run_id, type(exc).__name__)
            context.resources.clear()
            handle.lock.close()

    async def _heartbeat(self, handle: RunHandle) -> None:
        context = handle.context
        while True:
            await asyncio.sleep(self.settings.heartbeat_seconds)
            try:
                owned, cancelled = await asyncio.to_thread(self.store.renew, context.run_id, self.owner)
                if owned:
                    handle.lease_expires = time.time() + self.settings.lease_seconds
                if not owned or cancelled:
                    context.ownership_lost = not owned
                    handle.cancel_reason = "ownership_lost" if not owned else "run_cancelled"
                    if handle.task and not handle.task.done():
                        handle.task.cancel()
                    return
                await self.recover_expired()
            except Exception as exc:
                logger.warning("Lease renewal failed run_id=%s error_type=%s", context.run_id, type(exc).__name__)
                if time.time() >= handle.lease_expires:
                    context.ownership_lost = True
                    handle.cancel_reason = "ownership_lost"
                    if handle.task:
                        handle.task.cancel()
                    return

    async def recover_expired(self) -> None:
        for thread_id in await asyncio.to_thread(self.store.expired):
            try:
                lock = ThreadLock(self.store.directory / "locks", thread_id)
            except RuntimeFailure:
                continue  # An expired lease is not proof that its writer stopped.
            try:
                await asyncio.to_thread(self.store.recover, thread_id)
            finally:
                lock.close()
        await asyncio.to_thread(self.store.prune)

    async def move_thread(self, context: RunContext, thread_id: str) -> None:
        """Exit from a suspended interview into a fresh, exclusively held thread."""
        lock = ThreadLock(self.store.directory / "locks", thread_id)
        mutation = asyncio.create_task(asyncio.to_thread(
            self.store.owned_update, context.run_id, self.owner, thread_id=thread_id))
        cancelled = None
        try:
            try:
                await asyncio.shield(mutation)
            except asyncio.CancelledError as exc:
                await drain(mutation)
                cancelled = exc
        except BaseException:
            lock.close()
            raise
        handle = self.live[context.run_id]
        handle.lock.close()
        handle.lock = lock
        context.thread_id = thread_id
        if cancelled is not None:
            raise cancelled

    async def read_thread(self, thread_id: str, read: Callable[[], Awaitable], cleanup: Cleanup):
        """Reconnects wait across workers, then read under the same writer fence."""
        async with asyncio.timeout(self.settings.run_timeout + self.settings.lease_seconds):
            while True:
                try:
                    lock = ThreadLock(self.store.directory / "locks", thread_id)
                except RuntimeFailure:
                    await asyncio.sleep(0.05)
                    continue
                try:
                    await asyncio.to_thread(self.store.recover, thread_id)
                    if await asyncio.to_thread(self.store.active_thread, thread_id):
                        continue
                    if await asyncio.to_thread(self.store.dirty, thread_id):
                        async with asyncio.timeout(self.settings.cleanup_timeout):
                            await cleanup(thread_id)
                        await asyncio.to_thread(self.store.clean_thread, thread_id)
                    return await read()
                finally:
                    lock.close()
                    await asyncio.sleep(0.05)

    async def cancel(self, run_id: str, thread_id: str) -> dict:
        result = await asyncio.to_thread(self.store.request_cancel, run_id, thread_id)
        if handle := self.live.get(run_id):
            if handle.task and not handle.task.done():
                handle.task.cancel()
        return result

    async def shutdown(self) -> bool:
        self._closed = True
        tasks = [h.task for h in self.live.values() if h.task and not h.task.done()]
        for task in tasks:
            task.cancel()
        if not tasks:
            return True
        _, pending = await asyncio.wait(tasks, timeout=self.settings.cleanup_timeout + 1)
        return not pending
