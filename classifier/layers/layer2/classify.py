import json
import logging
from concurrent.futures import TimeoutError as _FuturesTimeout

from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.core.registry import TIER_MATRIX
from classifier.infra.config import settings
from classifier.infra.cost_tracker import cost_tracker
from classifier.config.feature_flags import feature_flags
from .api import _executor, _call_api, _call_api_with_model
from .rate_limiter import _get_rate_limiter
from .validation import _validate_l2_output

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES   = {t.value for t in TaskType}
_VALID_COMPLEXITIES = {c.value for c in TaskComplexity}


def classify_layer2(
    task: str,
    history: list[str] | None = None,
) -> tuple[TaskType, TaskComplexity, ModelTier, float, str] | None:
    """LLM classifier using Gemini Flash Lite. Returns None on any failure → Layer 1 fallback."""

    if feature_flags.l2_rate_limiter and not _get_rate_limiter().allow():
        logger.warning("layer2: rate limit reached (%d rpm)", settings.layer2_max_rpm)
        return None

    response = None
    try:
        future   = _executor.submit(_call_api, task, history)
        response = future.result(timeout=settings.layer2_timeout_ms / 1000)
    except (_FuturesTimeout, Exception) as exc:
        logger.warning("layer2: primary model failed: %s", exc)
        if feature_flags.l2_fallback_model and settings.layer2_fallback_model:
            logger.info("layer2: trying fallback model %s", settings.layer2_fallback_model)
            try:
                future   = _executor.submit(_call_api_with_model, task, history, settings.layer2_fallback_model)
                response = future.result(timeout=settings.layer2_timeout_ms / 1000)
            except Exception as fallback_exc:
                logger.warning("layer2: fallback also failed: %s", fallback_exc)
                return None
        else:
            return None

    if response is None:
        return None

    try:
        data    = json.loads(response.text.replace("\n", " "))
        tt_val  = str(data.get("task_type",  "")).lower().strip()
        cx_val  = str(data.get("complexity", "")).lower().strip()
        conf    = min(float(data.get("confidence", 0.5)), 0.85)
        reason  = str(data.get("reason", "llm classifier"))

        if tt_val not in _VALID_TASK_TYPES:
            logger.warning("layer2: unknown task_type=%r", tt_val)
            return None
        if cx_val not in _VALID_COMPLEXITIES:
            logger.warning("layer2: unknown complexity=%r", cx_val)
            return None

        task_type  = TaskType(tt_val)
        complexity = TaskComplexity(cx_val)

        if feature_flags.l2_output_validation and not _validate_l2_output(task, task_type, complexity, conf):
            logger.warning(
                "layer2: output validation rejected response "
                "(type=%s complexity=%s conf=%.2f) — using L1 fallback",
                tt_val, cx_val, conf,
            )
            return None

        tier      = TIER_MATRIX[(task_type, complexity)]
        reasoning = f"layer2 | {reason} | conf={conf:.2f}"

        input_tokens = len(task) // 4 + 200
        cost_tracker.record(settings.layer2_model, input_tokens, output_tokens=50, category="layer2")

        return task_type, complexity, tier, conf, reasoning

    except Exception as exc:
        logger.warning("layer2: parse error: %s", exc)
        return None
