"""DynamoDB logger backend — PutItem per decision/outcome event.

Table requirements:
    Hash key: decision_id (String)
    Sort key: event_type  (String)  ← allows one item per (decision, event_type)

Install: pip install boto3

Usage:
    from examples.custom_backends.dynamodb_backend import DynamoDBBackend
    from classifier import Router

    backend = DynamoDBBackend(table_name="dmr-events", region_name="us-east-1")
    router = Router(decision_logger=backend, outcome_logger=backend)

Query example (boto3):
    table = boto3.resource("dynamodb").Table("dmr-events")
    resp = table.query(
        KeyConditionExpression=Key("decision_id").eq("abc123"),
    )
"""

import json
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


def _to_decimal(obj):
    """Recursively convert floats to Decimal (DynamoDB requirement)."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_decimal(v) for v in obj]
    return obj


class DynamoDBBackend:
    def __init__(self, *, table_name: str, region_name: str = "us-east-1"):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required: pip install boto3") from exc
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)

    def log(self, entry: dict) -> None:
        event_type = "outcome" if "tokens_in" in entry else "decision"
        item = _to_decimal(dict(entry))
        item["event_type"] = event_type
        try:
            self._table.put_item(Item=item)
        except Exception as exc:
            logger.warning("DynamoDBBackend: put_item failed: %s", exc)
