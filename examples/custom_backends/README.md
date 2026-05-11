# Custom Logger Backend Examples

These are copy-paste reference implementations of the `LoggerBackend` protocol
for common databases and cloud stores.

## Protocol

Any object with a `log(entry: dict) -> None` method works as a backend:

```python
class MyBackend:
    def log(self, entry: dict) -> None:
        # entry is a flat dict with all decision or outcome fields
        # decisions have: decision_id, timestamp, tier, model, layer, confidence, ...
        # outcomes have:  decision_id, timestamp, tokens_in, tokens_out, cost_usd, success, ...
        ...

    # Optional: implement read() to support dmr stats / auto-labeler
    def read(self, *, since=None, until=None, decision_ids=None) -> list[dict]:
        ...
```

Wire it:
```python
from classifier import Router
router = Router(decision_logger=MyBackend(), outcome_logger=MyBackend())
```

## Files

| File | Storage | Extra deps |
|------|---------|------------|
| [sqlite_backend.py](sqlite_backend.py) | SQLite (local) | none (stdlib) |
| [postgres_backend.py](postgres_backend.py) | PostgreSQL | `psycopg2-binary` |
| [bigquery_backend.py](bigquery_backend.py) | Google BigQuery | `google-cloud-bigquery` |
| [dynamodb_backend.py](dynamodb_backend.py) | AWS DynamoDB | `boto3` |
| [gcs_backend.py](gcs_backend.py) | Google Cloud Storage | `google-cloud-storage` |

The built-in backends (Kafka, S3, Webhook, Stdout) are in `classifier/logger_backends.py`.

## Fan-out to multiple backends

```python
from classifier import Router, MultiLoggerBackend
from examples.custom_backends.sqlite_backend import SQLiteBackend
from classifier import StdoutLoggerBackend

backend = MultiLoggerBackend([
    SQLiteBackend("local.db"),       # local queryable copy
    StdoutLoggerBackend(),           # also stream to stdout for log collectors
])
router = Router(decision_logger=backend, outcome_logger=backend)
```

## Enable full structured telemetry

Set `DMR_TELEMETRY=1` in your environment to emit complete JSON events via Python
logging at `DEBUG` level (logger name: `dmr.decisions` / `dmr.outcomes`):

```bash
DMR_TELEMETRY=1 python your_app.py
```

Without this env var, only a minimal one-line INFO message is emitted — no files
written, no backend required.
