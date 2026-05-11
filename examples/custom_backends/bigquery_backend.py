"""BigQuery logger backend — streaming inserts to a BQ table.

Each decision/outcome is streamed as a row via insertAll (no batch delay).
Use separate tables for decisions and outcomes, or a single table with an
event_type column (shown below).

Install: pip install google-cloud-bigquery

BQ table schema (create once):
    decision_id  STRING
    event_type   STRING  (decision | outcome)
    timestamp    TIMESTAMP
    tier         STRING
    model        STRING
    layer        STRING
    payload      JSON    (full event, forward-compat)

Usage:
    from examples.custom_backends.bigquery_backend import BigQueryBackend
    from classifier import Router

    backend = BigQueryBackend(project="my-proj", dataset="dmr", table="events")
    router = Router(decision_logger=backend, outcome_logger=backend)
"""

import json
import logging

logger = logging.getLogger(__name__)


class BigQueryBackend:
    def __init__(self, *, project: str, dataset: str, table: str):
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise ImportError("google-cloud-bigquery is required: pip install google-cloud-bigquery") from exc
        self._client = bigquery.Client(project=project)
        self._table_id = f"{project}.{dataset}.{table}"

    def log(self, entry: dict) -> None:
        row = {
            "decision_id": entry.get("decision_id"),
            "event_type": "outcome" if "tokens_in" in entry else "decision",
            "timestamp": entry.get("timestamp"),
            "tier": entry.get("tier"),
            "model": entry.get("model"),
            "layer": entry.get("layer"),
            "payload": json.dumps(entry),
        }
        try:
            errors = self._client.insert_rows_json(self._table_id, [row])
            if errors:
                logger.warning("BigQueryBackend: insert errors: %s", errors)
        except Exception as exc:
            logger.warning("BigQueryBackend: insert failed: %s", exc)
