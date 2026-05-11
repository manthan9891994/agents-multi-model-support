"""PostgreSQL logger backend — upsert per decision/outcome event.

Uses a single `events` table with an `event_type` column (decision | outcome).
Connection pooling via psycopg2's SimpleConnectionPool.

Install: pip install psycopg2-binary

Table DDL (run once):
    CREATE TABLE IF NOT EXISTS dmr_events (
        id          SERIAL PRIMARY KEY,
        decision_id TEXT,
        event_type  TEXT NOT NULL,
        timestamp   TEXT,
        tier        TEXT,
        model       TEXT,
        layer       TEXT,
        payload     JSONB NOT NULL,
        UNIQUE (decision_id, event_type)
    );
    CREATE INDEX IF NOT EXISTS idx_dmr_ts   ON dmr_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_dmr_tier ON dmr_events(tier);

Usage:
    from examples.custom_backends.postgres_backend import PostgresBackend
    from classifier import Router

    backend = PostgresBackend(dsn="postgresql://user:pw@localhost/mydb")
    router = Router(decision_logger=backend, outcome_logger=backend)
"""

import json
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_INSERT_SQL = """
    INSERT INTO dmr_events (decision_id, event_type, timestamp, tier, model, layer, payload)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (decision_id, event_type)
    DO UPDATE SET payload = EXCLUDED.payload, timestamp = EXCLUDED.timestamp
"""

_READ_SQL = "SELECT payload FROM dmr_events WHERE event_type = %s"


class PostgresBackend:
    def __init__(
        self,
        *,
        dsn: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "dmr_telemetry",
        user: str | None = None,
        password: str | None = None,
        pool_size: int = 5,
    ):
        try:
            from psycopg2 import pool as pg_pool
        except ImportError as exc:
            raise ImportError("psycopg2-binary is required: pip install psycopg2-binary") from exc
        connect_kwargs = {"dsn": dsn} if dsn else {
            "host": host, "port": port, "dbname": dbname, "user": user, "password": password
        }
        self._pool = pg_pool.SimpleConnectionPool(1, pool_size, **connect_kwargs)
        self._lock = threading.Lock()

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def log(self, entry: dict) -> None:
        event_type = "outcome" if "tokens_in" in entry else "decision"
        try:
            with self._lock, self._conn() as conn:
                conn.cursor().execute(
                    _INSERT_SQL,
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
        except Exception as exc:
            logger.warning("PostgresBackend: insert failed: %s", exc)

    def read(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        decision_ids: set | None = None,
    ) -> list[dict]:
        event_type = "outcome" if decision_ids is not None else "decision"
        sql, params = _READ_SQL, [event_type]
        if since:
            sql += " AND timestamp >= %s"
            params.append(since)
        if until:
            sql += " AND timestamp < %s"
            params.append(until)
        if decision_ids:
            placeholders = ",".join(["%s"] * len(decision_ids))
            sql += f" AND decision_id IN ({placeholders})"
            params.extend(decision_ids)
        sql += " ORDER BY timestamp ASC"
        try:
            with self._conn() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                return [json.loads(r[0]) for r in cur.fetchall()]
        except Exception as exc:
            logger.warning("PostgresBackend: read failed: %s", exc)
            return []
