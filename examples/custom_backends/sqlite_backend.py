"""SQLite logger backend — zero-dependency local persistent store.

Stores both decision and outcome events in a single DB file.
Events are stored in one table keyed on (decision_id, event_type) so you can
query decisions, outcomes, or JOIN them in plain SQL.

Install: no extra dependencies (sqlite3 is stdlib)

Usage:
    from examples.custom_backends.sqlite_backend import SQLiteBackend
    from classifier import Router

    backend = SQLiteBackend()  # writes to dmr_telemetry.db in cwd
    router = Router(decision_logger=backend, outcome_logger=backend)

    # After running some classifications:
    import sqlite3, json
    conn = sqlite3.connect("dmr_telemetry.db")
    for row in conn.execute("SELECT payload FROM events WHERE event_type='decision' ORDER BY timestamp DESC LIMIT 5"):
        print(json.loads(row[0]))
"""

import json
import sqlite3
import threading
from pathlib import Path


class SQLiteBackend:
    def __init__(self, path: str = "dmr_telemetry.db"):
        self._path = Path(path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id  TEXT,
                    event_type   TEXT NOT NULL,
                    timestamp    TEXT,
                    tier         TEXT,
                    model        TEXT,
                    layer        TEXT,
                    payload      TEXT NOT NULL
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_ts      ON events(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tier    ON events(tier)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_did     ON events(decision_id)")
            c.commit()

    def log(self, entry: dict) -> None:
        event_type = "outcome" if "tokens_in" in entry else "decision"
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT INTO events (decision_id, event_type, timestamp, tier, model, layer, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.get("decision_id"),
                    event_type,
                    entry.get("timestamp"),
                    entry.get("tier"),
                    entry.get("model"),
                    entry.get("layer"),
                    json.dumps(entry),
                ),
            )
            c.commit()

    def read(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        decision_ids: set | None = None,
    ) -> list[dict]:
        event_type = "outcome" if decision_ids is not None else "decision"
        sql = "SELECT payload FROM events WHERE event_type = ?"
        params: list = [event_type]
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        if until:
            sql += " AND timestamp < ?"
            params.append(until)
        if decision_ids:
            placeholders = ",".join("?" * len(decision_ids))
            sql += f" AND decision_id IN ({placeholders})"
            params.extend(decision_ids)
        sql += " ORDER BY timestamp ASC"
        rows = self._conn().execute(sql, params).fetchall()
        return [json.loads(r[0]) for r in rows]


if __name__ == "__main__":
    import os

    os.environ.setdefault("DMR_TELEMETRY", "1")
    backend = SQLiteBackend("_demo.db")
    backend.log(
        {
            "decision_id": "demo01",
            "timestamp": "2026-05-10T12:00:00+00:00",
            "tier": "low",
            "model": "gemini-2.0-flash-lite",
            "layer": "layer1",
            "confidence": 0.91,
            "latency_ms": 2.1,
        }
    )
    print("Stored. Reading back:", backend.read(since="2026-01-01T00:00:00+00:00"))
