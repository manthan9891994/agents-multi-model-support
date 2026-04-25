import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

from google import genai

from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.core.registry import TIER_MATRIX
from classifier.infra.config import settings

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES  = {t.value for t in TaskType}
_VALID_COMPLEXITIES = {c.value for c in TaskComplexity}

_PROMPT = (
    'Classify this task. JSON only: task_type, complexity, confidence (0-1), reason (≤8 words).\n\n'
    'task_type: reasoning|thinking|analyzing|code_creation|doc_creation|translation|math|conversation|multimodal\n'
    'complexity: simple|standard|complex|research\n\n'
    'Task: "{task}"'
)

_SCHEMA = genai.types.Schema(
    type=genai.types.Type.OBJECT,
    properties={
        "task_type":  genai.types.Schema(type=genai.types.Type.STRING),
        "complexity": genai.types.Schema(type=genai.types.Type.STRING),
        "confidence": genai.types.Schema(type=genai.types.Type.NUMBER),
        "reason":     genai.types.Schema(type=genai.types.Type.STRING),
    },
    required=["task_type", "complexity", "confidence", "reason"],
)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="layer2")


class _RateLimiter:
    def __init__(self, max_rpm: int):
        self._lock  = threading.Lock()
        self._calls: list[float] = []
        self._max   = max_rpm

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < 60]
            if len(self._calls) >= self._max:
                return False
            self._calls.append(now)
            return True


_rate_limiter: _RateLimiter | None = None


def _get_rate_limiter() -> _RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = _RateLimiter(settings.layer2_max_rpm)
    return _rate_limiter


def _call_api(task: str):
    client = genai.Client(api_key=settings.google_api_key)
    return client.models.generate_content(
        model=settings.layer2_model,
        contents=_PROMPT.format(task=task[:500]),
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=300,
            response_mime_type="application/json",
            response_schema=_SCHEMA,
        ),
    )


def classify_layer2(
    task: str,
) -> tuple[TaskType, TaskComplexity, ModelTier, float, str] | None:
    """LLM classifier using Gemini Flash Lite. Returns None on any failure → Layer 1 fallback."""
    if not _get_rate_limiter().allow():
        logger.warning("layer2: rate limit reached (%d rpm)", settings.layer2_max_rpm)
        return None

    try:
        future   = _executor.submit(_call_api, task)
        response = future.result(timeout=settings.layer2_timeout_ms / 1000)
    except _FuturesTimeout:
        logger.warning("layer2: timeout after %dms", settings.layer2_timeout_ms)
        return None
    except Exception as exc:
        logger.warning("layer2: API error: %s", exc)
        return None

    try:
        data    = json.loads(response.text.replace("\n", " "))  # model may emit literal newlines inside string values
        tt_val  = str(data.get("task_type",  "")).lower().strip()
        cx_val  = str(data.get("complexity", "")).lower().strip()
        conf    = float(data.get("confidence", 0.5))
        reason  = str(data.get("reason", "llm classifier"))

        if tt_val not in _VALID_TASK_TYPES:
            logger.warning("layer2: unknown task_type=%r", tt_val)
            return None
        if cx_val not in _VALID_COMPLEXITIES:
            logger.warning("layer2: unknown complexity=%r", cx_val)
            return None

        task_type  = TaskType(tt_val)
        complexity = TaskComplexity(cx_val)
        tier       = TIER_MATRIX[(task_type, complexity)]
        reasoning  = f"layer2 | {reason} | conf={conf:.2f}"

        return task_type, complexity, tier, conf, reasoning

    except Exception as exc:
        logger.warning("layer2: parse error: %s", exc)
        return None
