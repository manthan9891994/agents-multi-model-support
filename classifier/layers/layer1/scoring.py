import re
from typing import Optional

from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.config.feature_flags import feature_flags
from .constants import (
    _TASK_KEYWORDS, _NEGATIVE_KEYWORDS, _ESCALATORS, _DEESCALATORS,
    _ALGORITHM_NAMES, _FORMAT_REQUESTS, _DOMAIN_MIN_TIER,
    _TASK_TYPE_TIER_WEIGHT, _COMPLEXITY_LEVELS, _TIER_ORDER, _LOW_TIER_CONTEXT_TOKENS,
)
from .helpers import _negation_positions, _extract_instruction

_CODE_SNIPPET_RE = re.compile(
    r"(^(def |class |import |from |function |const |let |var |public |private |#include)"
    r"|[\{\}];\s*$"
    r"|```[\w]*\n)"
    , re.MULTILINE,
)


def _pick_higher_tier_type(positives: list[tuple[TaskType, float]]) -> TaskType:
    top_score = positives[0][1]
    threshold = top_score * 0.80
    ambiguous_set = [t for t, s in positives if s >= threshold]
    return max(ambiguous_set, key=lambda t: _TASK_TYPE_TIER_WEIGHT.get(t, 0))


def _score_task_type(lower: str) -> dict[TaskType, float]:
    """Raw keyword scores for all task types. Exported for L2 cross-layer output validation."""
    scores: dict[TaskType, float] = {t: 0.0 for t in TaskType}
    negated = _negation_positions(lower) if feature_flags.negation_suppression else set()

    for task_type, groups in _TASK_KEYWORDS.items():
        for weight, group_key in [(3.0, "primary"), (1.0, "secondary")]:
            sorted_kws = sorted(groups.get(group_key, []), key=len, reverse=True)
            consumed: set[int] = set()
            for kw in sorted_kws:
                for m in re.finditer(r"\b" + re.escape(kw) + r"\b", lower):
                    s, e = m.start(), m.end()
                    region = set(range(s, e))
                    if region & consumed:
                        continue
                    mid = (s + e) // 2
                    if mid in negated:
                        scores[task_type] -= weight * 0.5
                    else:
                        scores[task_type] += weight
                        consumed |= region

    for task_type, neg_kws in _NEGATIVE_KEYWORDS.items():
        for kw in sorted(neg_kws, key=len, reverse=True):
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                scores[task_type] -= 2.0

    if feature_flags.code_snippet_detection and _CODE_SNIPPET_RE.search(lower):
        scores[TaskType.CODE_CREATION] += 4.0

    return scores


def _detect_task_type(lower: str) -> tuple[TaskType, float, bool]:
    scores = _score_task_type(lower)

    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return TaskType.DOC_CREATION, 0.3, False

    positives = sorted(
        [(t, s) for t, s in scores.items() if s > 0],
        key=lambda x: x[1], reverse=True,
    )
    is_ambiguous = (
        len(positives) >= 2
        and positives[1][1] / positives[0][1] >= 0.80
    )

    if is_ambiguous and feature_flags.multi_task_detection:
        best = _pick_higher_tier_type(positives)

    total = sum(v for v in scores.values() if v > 0) or 1
    confidence = min(scores[best] / total + 0.2, 1.0)
    if is_ambiguous:
        confidence = min(confidence, 0.45)

    return best, confidence, is_ambiguous


def _detect_complexity(lower: str, tokens: int) -> TaskComplexity:
    is_format_only = feature_flags.format_request_deescalation and any(fr in lower for fr in _FORMAT_REQUESTS)
    has_algorithm  = feature_flags.algorithm_detection and any(alg in lower for alg in _ALGORITHM_NAMES)

    instruction = _extract_instruction(lower)
    instruction_tokens = len(instruction.split()) * 4 // 3

    if instruction_tokens > 15_000:
        base = TaskComplexity.RESEARCH
    elif instruction_tokens > 5_000:
        base = TaskComplexity.COMPLEX
    elif instruction_tokens > 500:
        base = TaskComplexity.STANDARD
    else:
        base = TaskComplexity.SIMPLE

    if base == TaskComplexity.SIMPLE and tokens > _LOW_TIER_CONTEXT_TOKENS * 0.5:
        base = TaskComplexity.STANDARD

    idx = _COMPLEXITY_LEVELS.index(base)

    if has_algorithm:
        idx = max(idx, 2)

    if feature_flags.escalator_scoring and not is_format_only:
        weight = sum(w for kw, w in _ESCALATORS.items() if kw in lower)
        has_deescalator = sum(1 for kw in _DEESCALATORS if kw in lower) >= 1
        if has_deescalator:
            weight = weight * 0.5
        if weight >= 5:
            idx = min(idx + 3, 3)
        elif weight >= 3:
            idx = min(idx + 2, 3)
        elif weight >= 1:
            idx = min(idx + 1, 3)

    complexity = _COMPLEXITY_LEVELS[idx]

    if sum(1 for kw in _DEESCALATORS if kw in lower) >= 1 and idx > 0:
        weight_raw = sum(w for kw, w in _ESCALATORS.items() if kw in lower)
        if weight_raw == 0:
            complexity = _COMPLEXITY_LEVELS[idx - 1]

    if feature_flags.question_type_override:
        from .helpers import _WHAT_IS_RE, _YES_NO_RE
        stripped = lower.strip()
        if _WHAT_IS_RE.match(stripped):
            if complexity in (TaskComplexity.STANDARD, TaskComplexity.COMPLEX, TaskComplexity.RESEARCH):
                complexity = TaskComplexity.SIMPLE
        elif _YES_NO_RE.match(stripped):
            escalator_weight = sum(w for kw, w in _ESCALATORS.items() if kw in lower)
            if escalator_weight < 3 and complexity in (TaskComplexity.STANDARD, TaskComplexity.COMPLEX):
                complexity = TaskComplexity.SIMPLE

    return complexity


def _domain_min_tier(lower: str) -> Optional[ModelTier]:
    result: Optional[ModelTier] = None
    for kw, min_t in _DOMAIN_MIN_TIER.items():
        if kw in lower:
            if result is None or _TIER_ORDER.index(min_t) > _TIER_ORDER.index(result):
                result = min_t
    return result
