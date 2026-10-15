"""Audit trail. Stores decisions only - never embeddings or images."""
import sqlite3
import threading
import time


class AuditLog:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events (ts REAL, device_id TEXT, zone TEXT, "
            "status TEXT, reason TEXT, person_id TEXT, score REAL)"
        )
        self._conn.commit()

    def record(self, device_id, zone, status, reason, person_id, score):
        with self._lock:
            self._conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
                (time.time(), device_id, zone, status, reason, person_id, score),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
