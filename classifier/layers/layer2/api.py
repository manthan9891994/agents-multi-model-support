import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

from classifier.config.feature_flags import feature_flags
from classifier.infra.config import settings
from classifier.infra.pii_scrubber import scrub

from .prompt import _SCHEMA, _build_contents

# google-genai is an optional dep. Import lazily so the package stays usable
# without it (e.g. user only configures Anthropic/OpenAI providers).
try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _scrub_for_external(task: str, history: list[str] | None) -> tuple[str, list[str] | None, list[str]]:
    """Strip PII from task + history before sending to external LLM. Returns (task, history, matches)."""
    if not feature_flags.l2_pii_scrub:
        return task, history, []
    res_task = scrub(task, strict=settings.pii_scrub_strict)
    matches = list(res_task.matches)
    new_history = None
    if history:
        new_history = []
        for h in history:
            r = scrub(h, strict=settings.pii_scrub_strict)
            new_history.append(r.text)
            for m in r.matches:
                if m not in matches:
                    matches.append(m)
    if matches:
        logger.info("layer2: PII scrubbed before API call — tokens=%s", matches)
    return res_task.text, new_history, matches


_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="layer2")
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


# ── Circuit breaker ──────────────────────────────────────────────────────────
# After N consecutive failures, trip OPEN and skip L2 for COOLDOWN seconds.
# Prevents runaway latency/cost during provider outages.
class _CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_secs: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_secs = cooldown_secs
        self._failures = 0
        self._opened_at = 0.0
        import threading as _t

        self._lock = _t.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._failures < self.failure_threshold:
                return False
            if time.time() - self._opened_at >= self.cooldown_secs:
                # half-open: allow one trial through
                self._failures = self.failure_threshold - 1
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures == self.failure_threshold:
                self._opened_at = time.time()
                logger.warning(
                    "layer2: circuit breaker OPEN — %d consecutive failures, skipping L2 for %.0fs",
                    self._failures,
                    self.cooldown_secs,
                )


_circuit_breaker = _CircuitBreaker()


# ── Connection pooling — share one client across calls ───────────────────────
_client_lock = None
_shared_client = None


def _get_client():
    """Return a shared genai.Client (one connection pool process-wide)."""
    global _client_lock, _shared_client
    if _client_lock is None:
        import threading as _t

        _client_lock = _t.Lock()
    with _client_lock:
        if _shared_client is None:
            if genai is None:
                raise ImportError(
                    "google-genai not installed. Install with: pip install 'dynamic-model-router[google]'"
                )
            _shared_client = genai.Client(api_key=settings.google_api_key)
        return _shared_client


def _retry_after_seconds(exc) -> float | None:
    """Pull a Retry-After hint out of common exception shapes (genai, requests, httpx)."""
    for attr in ("retry_after", "headers"):
        val = getattr(exc, attr, None)
        if val is None:
            continue
        if isinstance(val, dict):
            ra = val.get("Retry-After") or val.get("retry-after")
            if ra:
                try:
                    return float(ra)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


# Retry policy can be overridden via Router(l2_retry_policy={...}) — module global.
_retry_policy: dict = {"max_attempts": 3, "initial_delay": 0.2, "backoff": 3.0}


def configure_retry_policy(
    *, max_attempts: int = 3, initial_delay: float = 0.2, backoff: float = 3.0
) -> None:
    _retry_policy["max_attempts"] = max_attempts
    _retry_policy["initial_delay"] = initial_delay
    _retry_policy["backoff"] = backoff


def _call_with_retry(fn, *args, max_attempts: int | None = None, **kwargs):
    """Exponential backoff retry for retryable HTTP errors. Honors Retry-After."""
    max_attempts = max_attempts or _retry_policy["max_attempts"]
    delay = _retry_policy["initial_delay"]
    backoff = _retry_policy["backoff"]
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status is not None and status not in _RETRYABLE_STATUSES:
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                ra = _retry_after_seconds(exc)
                sleep_for = ra if ra is not None else delay + random.uniform(0, 0.1)
                logger.info(
                    "layer2: retryable error (status=%s attempt=%d) — sleeping %.2fs",
                    status,
                    attempt + 1,
                    sleep_for,
                )
                time.sleep(sleep_for)
                delay *= backoff
    raise last_exc


def _resolve_l2_provider() -> str:
    """L2 provider falls back to default_provider if layer2_provider is unset."""
    return getattr(settings, "layer2_provider", "") or settings.default_provider


def _call_api(task: str, history: list[str] | None = None):
    return _call_api_with_model(task, history, settings.layer2_model)


def _call_api_with_model(task: str, history: list[str] | None, model: str):
    if _circuit_breaker.is_open():
        raise RuntimeError("layer2: circuit breaker OPEN — provider degraded")

    task, history, _ = _scrub_for_external(task, history)
    contents = _build_contents(task, history)

    provider = _resolve_l2_provider()
    from classifier.layers.layer2.providers import get_l2_caller

    caller = get_l2_caller(provider)
    if caller is None:
        # Fall back to legacy in-line Google path so existing setups keep working
        caller = _legacy_google_caller

    try:
        if feature_flags.l2_retry_with_backoff:
            result = _call_with_retry(caller, contents, history, model, _SCHEMA)
        else:
            result = caller(contents, history, model, _SCHEMA)
        _circuit_breaker.record_success()
        return result
    except Exception:
        _circuit_breaker.record_failure()
        raise


def _legacy_google_caller(task, history, model, schema):
    """Original inlined Google caller — kept for back-compat when providers/ pkg
    isn't importable for some reason."""
    client = _get_client()
    cfg = genai.types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
        response_schema=schema,
    )
    return client.models.generate_content(model=model, contents=task, config=cfg)
