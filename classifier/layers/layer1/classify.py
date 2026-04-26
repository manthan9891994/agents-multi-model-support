import logging
import time
from typing import Optional

from classifier.core.exceptions import ClassificationError
from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.core.registry import TIER_MATRIX
from classifier.config.feature_flags import feature_flags
from .constants import _TIER_ORDER, _TASK_KEYWORDS
from .helpers import _is_trivial, _detect_language, _count_tokens, _CONTINUATION_RE
from .pii import detect_pii
from .scoring import _score_task_type, _detect_task_type, _detect_complexity, _domain_min_tier
from .keyword_packs import _load_keyword_packs

logger = logging.getLogger(__name__)

if feature_flags.keyword_packs:
    _load_keyword_packs()


def classify_layer1(
    task: str,
    history: Optional[list[str]] = None,
    provider: str = "google",
) -> tuple[TaskType, TaskComplexity, ModelTier, float, str]:
    """Layer 1 — keyword + heuristic classifier (<1ms, no external calls)."""
    if not task or not task.strip():
        raise ClassificationError("Layer 1 received an empty task string.")

    if feature_flags.trivial_input_guard and _is_trivial(task):
        return (
            TaskType.CONVERSATION,
            TaskComplexity.SIMPLE,
            ModelTier.LOW,
            0.95,
            "layer1 | trivial input → CONVERSATION/SIMPLE",
        )

    t0     = time.perf_counter()
    lower  = task.lower()
    tokens = _count_tokens(task, provider=provider)

    if feature_flags.continuation_detection and history and _CONTINUATION_RE.match(task.strip()):
        recent = " ".join(h.lower() for h in history[-3:])
        history_scores: dict[TaskType, float] = {t: 0.0 for t in TaskType}
        for ht, groups in _TASK_KEYWORDS.items():
            for weight, group_key in [(3.0, "primary"), (1.0, "secondary")]:
                for kw in groups.get(group_key, []):
                    if kw in recent:
                        history_scores[ht] += weight
        best_history = max(history_scores, key=history_scores.get)
        if history_scores[best_history] >= 3:
            task_type    = best_history
            complexity   = _detect_complexity(lower, tokens)
            tier         = TIER_MATRIX.get((task_type, complexity), ModelTier.MEDIUM)
            domain_floor = _domain_min_tier(lower) if feature_flags.domain_escalation else None
            if domain_floor and _TIER_ORDER.index(domain_floor) > _TIER_ORDER.index(tier):
                tier = domain_floor
            elapsed = (time.perf_counter() - t0) * 1000
            return (
                task_type, complexity, tier, 0.80,
                f"layer1 | type={task_type.value} complexity={complexity.value} "
                f"tokens={tokens} tier={tier.value} ({elapsed:.1f}ms) [continuation]",
            )

    task_type, confidence, is_ambiguous = _detect_task_type(lower)

    if feature_flags.history_bias and history:
        recent = " ".join(h.lower() for h in history[-3:])
        h_scores = {t: 0.0 for t in TaskType}
        for ht, groups in _TASK_KEYWORDS.items():
            for weight, group_key in [(3.0, "primary"), (1.0, "secondary")]:
                for kw in groups.get(group_key, []):
                    if kw in recent:
                        h_scores[ht] += weight
        best_history = max(h_scores, key=h_scores.get)
        if h_scores[best_history] >= 4 and best_history != task_type:
            task_type  = best_history
            confidence = min(confidence + 0.1, 1.0)

    lang = "en"
    if feature_flags.language_detection:
        lang = _detect_language(task)
        if lang != "en" and confidence < 0.60:
            confidence   = min(confidence, 0.40)
            is_ambiguous = True

    complexity   = _detect_complexity(lower, tokens)
    tier         = TIER_MATRIX.get((task_type, complexity), ModelTier.MEDIUM)
    domain_floor = _domain_min_tier(lower) if feature_flags.domain_escalation else None

    if domain_floor and _TIER_ORDER.index(domain_floor) > _TIER_ORDER.index(tier):
        tier = domain_floor

    elapsed = (time.perf_counter() - t0) * 1000

    flags: list[str] = []
    if is_ambiguous:
        flags.append("ambiguous")
    if lang != "en":
        flags.append(f"lang={lang}")
    if domain_floor:
        flags.append(f"domain_min={domain_floor.value}")

    reason = (
        f"layer1 | type={task_type.value} complexity={complexity.value} "
        f"tokens={tokens} tier={tier.value} ({elapsed:.1f}ms)"
        + (f" [{', '.join(flags)}]" if flags else "")
    )
    logger.debug(reason)
    return task_type, complexity, tier, confidence, reason
