"""Durable observation receipts and reliable event identities, separate from media retention."""
from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .credentials import private_directory, private_file
from .protocol import ProtocolError, canonical, loads, timestamp


class ReceiptJournal:
    def __init__(self, root: Path, *, identity: str | None = None):
        root = private_directory(root)
        lock_path = root / "process.lock"
        self._process_lock = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        private_file(lock_path)
        try:
            fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._process_lock)
            raise ProtocolError("BUSY", "client_already_running", "This V1 state directory is already in use.") from None
        path = root / "receipts.sqlite3"
        if not path.exists():
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        private_file(path)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        # DELETE journal uses owner-only files; FULL fsync before effects/results.
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT, retain_until REAL NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (key TEXT PRIMARY KEY, payload TEXT NOT NULL, retain_until REAL NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), binding TEXT NOT NULL)")
        if identity is not None:
            row = self.db.execute("SELECT binding FROM identity WHERE id=1").fetchone()
            if row is not None and row[0] != identity:
                self.close()
                raise ProtocolError("PERMISSION_DENIED", "state_identity_mismatch", "V1 receipts belong to another identity.")
            self.db.execute("INSERT OR IGNORE INTO identity VALUES (1, ?)", (identity,))
        self.db.commit()

    def begin(self, command: dict[str, Any], now: datetime) -> tuple[bool, dict[str, Any] | None]:
        with self._lock:
            key = command["request_id"]  # authenticated issuer is always Home Cortex, one body per journal
            frozen = canonical(command)
            row = self.db.execute("SELECT request, response FROM requests WHERE id=?", (key,)).fetchone()
            if row is not None:
                if row[0] != frozen:
                    raise ProtocolError("CONFLICT", "idempotency_conflict", "Request ID was reused with different content.")
                return False, None if row[1] is None else loads(row[1].encode())
            deadline = timestamp(command["deadline_at"])
            if now >= deadline:
                raise ProtocolError("TIMEOUT", "deadline", "The command deadline has expired.")
            self.db.execute("INSERT INTO requests VALUES (?, ?, NULL, ?)",
                            (key, frozen, (deadline + timedelta(hours=24)).timestamp()))
            self.db.commit()
            return True, None

    def finish(self, command_id: str, reply: dict[str, Any]) -> None:
        with self._lock:
            self.db.execute("UPDATE requests SET response=? WHERE id=?", (canonical(reply), command_id))
            self.db.commit()

    def event(self, key: str, make, now: datetime) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute("SELECT payload FROM events WHERE key=?", (key,)).fetchone()
            if row is not None:
                return loads(row[0].encode())
            value = make()
            self.db.execute("INSERT INTO events VALUES (?, ?, ?)",
                            (key, canonical(value), (now + timedelta(hours=24)).timestamp()))
            self.db.commit()
            return value

    def cleanup(self, now: datetime) -> None:
        with self._lock:
            self.db.execute("DELETE FROM requests WHERE retain_until<?", (now.timestamp(),))
            self.db.execute("DELETE FROM events WHERE retain_until<?", (now.timestamp(),))
            self.db.commit()

    def close(self) -> None:
        self.db.close()
        os.close(self._process_lock)
