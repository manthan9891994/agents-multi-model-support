import random
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai

from classifier.infra.config import settings
from classifier.config.feature_flags import feature_flags
from .prompt import _SCHEMA, _build_contents

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="layer2")
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _call_with_retry(fn, *args, max_attempts: int = 3):
    """Exponential backoff retry for retryable HTTP errors only."""
    delay    = 0.2
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args)
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status is not None and status not in _RETRYABLE_STATUSES:
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(delay + random.uniform(0, 0.1))
                delay *= 3
    raise last_exc


def _call_api(task: str, history: list[str] | None = None):
    contents = _build_contents(task, history)
    client   = genai.Client(api_key=settings.google_api_key)
    cfg      = genai.types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
        response_schema=_SCHEMA,
    )
    if feature_flags.l2_retry_with_backoff:
        return _call_with_retry(client.models.generate_content, settings.layer2_model, contents, cfg)
    return client.models.generate_content(settings.layer2_model, contents, cfg)


def _call_api_with_model(task: str, history: list[str] | None, model: str):
    contents = _build_contents(task, history)
    client   = genai.Client(api_key=settings.google_api_key)
    cfg      = genai.types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
        response_schema=_SCHEMA,
    )
    if feature_flags.l2_retry_with_backoff:
        return _call_with_retry(client.models.generate_content, model, contents, cfg)
    return client.models.generate_content(model, contents, cfg)
