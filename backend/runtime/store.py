"""SQLite admission/ownership, plus OS locks fencing live checkpoint writers.

All workers must share this local directory. Locks are never deleted: unlinking
a locked inode would create two independent locks for the same thread on Unix.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

from runtime.config import RuntimeSettings
from runtime.errors import RuntimeFailure

ACTIVE = ("pending", "running")


class ThreadLock:
    def __init__(self, directory: Path, thread_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(thread_id.encode()).hexdigest()
        self._file = open(directory / f"{name}.lock", "a+b")
        try:
            self._file.seek(0, 2)
            if not self._file.tell():
                self._file.write(b"0")
                self._file.flush()
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            raise RuntimeFailure("thread_busy", "此会话仍在处理中，请等待本轮结束。", http_status=409)

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()


class RunStore:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.directory = settings.data_dir.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "runs.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL,
                    request_id TEXT UNIQUE, fingerprint TEXT NOT NULL,
                    owner TEXT NOT NULL, channel TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    lease_expires REAL NOT NULL, deadline REAL NOT NULL,
                    stop_reason TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS active_thread ON runs(thread_id)
                    WHERE status IN ('pending', 'running');
                CREATE INDEX IF NOT EXISTS run_expiry ON runs(status, lease_expires);
                CREATE TABLE IF NOT EXISTS thread_health (
                    thread_id TEXT PRIMARY KEY, dirty INTEGER NOT NULL DEFAULT 0
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=1, isolation_level="IMMEDIATE")
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def find_request(self, request_id: str | None, fingerprint: str) -> None:
        if not request_id:
            return
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE request_id=?", (request_id,)).fetchone()
        if row:
            same = row["fingerprint"] == fingerprint
            raise RuntimeFailure(
                "duplicate_request" if same else "request_id_conflict",
                "该请求已经提交，请勿重复发送。" if same else "请求标识已用于其他内容。",
                http_status=409,
                details={"run_id": row["run_id"], "status": row["status"], "thread_id": row["thread_id"]},
            )

    def admit(self, thread_id: str, request_id: str | None, fingerprint: str,
              owner: str, channel: str, timeout: float, run_id: str | None = None) -> dict:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.find_request(request_id, fingerprint)
            # Caller holds the OS lock, proving no prior writer can still run.
            db.execute("""UPDATE runs SET status='error', stop_reason='orphan_recovered', updated_at=?
                WHERE thread_id=? AND status IN ('pending','running') AND lease_expires<=?""",
                       (now, thread_id, now))
            row = db.execute("SELECT run_id FROM runs WHERE thread_id=? AND status IN ('pending','running')",
                             (thread_id,)).fetchone()
            if row:
                raise RuntimeFailure("thread_busy", "此会话仍在处理中，请等待本轮结束。", http_status=409)
            count = db.execute("SELECT count(*) FROM runs WHERE status IN ('pending','running')").fetchone()[0]
            if count >= self.settings.max_concurrent_runs:
                raise RuntimeFailure("runtime_busy", "服务正在处理较多请求，请稍后再试。", retryable=True, http_status=503)
            run_id = run_id or uuid4().hex
            db.execute("""INSERT INTO runs
                (run_id,thread_id,request_id,fingerprint,owner,channel,status,created_at,updated_at,lease_expires,deadline)
                VALUES (?,?,?,?,?,?,'pending',?,?,?,?)""",
                       (run_id, thread_id, request_id, fingerprint, owner, channel, now, now,
                        now + self.settings.lease_seconds, now + timeout))
            db.execute("INSERT OR IGNORE INTO thread_health(thread_id) VALUES (?)", (thread_id,))
            return dict(db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())

    def renew(self, run_id: str, owner: str) -> tuple[bool, bool]:
        now = time.time()
        with self.connect() as db:
            updated = db.execute("""UPDATE runs SET lease_expires=?,updated_at=?
                WHERE run_id=? AND owner=? AND status IN ('pending','running') AND lease_expires>?""",
                                 (now + self.settings.lease_seconds, now, run_id, owner, now)).rowcount
            row = db.execute("SELECT cancel_requested FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return bool(updated), bool(row and row[0])

    def owned_update(self, run_id: str, owner: str, *, graph: bool = False,
                     status: str | None = None, reason: str | None = None,
                     clean: bool = False, thread_id: str | None = None) -> None:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM runs WHERE run_id=? AND owner=?
                AND status IN ('pending','running') AND lease_expires>?""", (run_id, owner, now)).fetchone()
            if not row:
                raise RuntimeFailure("ownership_lost", "本次运行已停止，请稍后重试。", retryable=True)
            if row["cancel_requested"] and status in {"success", "waiting_input"}:
                raise RuntimeFailure("run_cancelled", "本次处理已取消。")
            if thread_id:
                db.execute("UPDATE runs SET thread_id=? WHERE run_id=?", (thread_id, run_id))
                db.execute("INSERT OR IGNORE INTO thread_health(thread_id) VALUES (?)", (thread_id,))
            if graph:
                db.execute("UPDATE thread_health SET dirty=1 WHERE thread_id=?", (thread_id or row["thread_id"],))
            if clean:
                db.execute("UPDATE thread_health SET dirty=0 WHERE thread_id=?", (thread_id or row["thread_id"],))
            if status:
                db.execute("UPDATE runs SET status=?,stop_reason=?,updated_at=? WHERE run_id=?",
                           (status, reason, now, run_id))

    def dirty(self, thread_id: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT dirty FROM thread_health WHERE thread_id=?", (thread_id,)).fetchone()
            return bool(row and row[0])

    def clean_thread(self, thread_id: str) -> None:
        """Only called while the caller holds this thread's OS lock."""
        with self.connect() as db:
            db.execute("UPDATE thread_health SET dirty=0 WHERE thread_id=?", (thread_id,))

    def active_thread(self, thread_id: str) -> bool:
        with self.connect() as db:
            return db.execute("SELECT 1 FROM runs WHERE thread_id=? AND status IN ('pending','running')",
                              (thread_id,)).fetchone() is not None

    def get(self, run_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def request_cancel(self, run_id: str, thread_id: str) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=? AND thread_id=?", (run_id, thread_id)).fetchone()
            if not row:
                raise RuntimeFailure("run_not_found", "未找到该次运行。", http_status=404)
            if row["status"] in ACTIVE:
                db.execute("UPDATE runs SET cancel_requested=1 WHERE run_id=?", (run_id,))
            return {"run_id": run_id, "status": row["status"], "cancel_requested": row["status"] in ACTIVE}

    def expired(self) -> list[str]:
        with self.connect() as db:
            return [row[0] for row in db.execute("""SELECT thread_id FROM runs
                WHERE status IN ('pending','running') AND lease_expires<=? LIMIT 100""", (time.time(),))]

    def recover(self, thread_id: str) -> None:
        with self.connect() as db:
            db.execute("""UPDATE runs SET status='error',stop_reason='orphan_recovered',updated_at=?
                WHERE thread_id=? AND status IN ('pending','running') AND lease_expires<=?""",
                       (time.time(), thread_id, time.time()))

    def prune(self) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM runs WHERE status NOT IN ('pending','running') AND updated_at<?",
                       (time.time() - self.settings.retention_seconds,))
            db.execute("""DELETE FROM thread_health WHERE dirty=0 AND NOT EXISTS
                (SELECT 1 FROM runs WHERE runs.thread_id=thread_health.thread_id)""")
