import logging

from classifier.core.types import TaskType, TaskComplexity
from classifier.layers.layer1 import _score_task_type, _count_tokens, _CODE_SNIPPET_RE, _TASK_TYPE_TIER_WEIGHT

logger = logging.getLogger(__name__)

_KEYWORD_OPTIONAL_TYPES = {TaskType.MULTIMODAL, TaskType.THINKING, TaskType.CONVERSATION}


def _validate_l2_output(
    task: str,
    l2_type: TaskType,
    l2_complexity: TaskComplexity,
    l2_conf: float,
) -> bool:
    """Return False if L2's response is structurally implausible.

    Validates the OUTPUT only — words like "ignore", "disregard", "act as"
    in task content never cause false positives here.

    Three independent checks:
      1. Keyword cross-check  — L2's returned type has some keyword support.
      2. Complexity sanity    — long tasks shouldn't be conversation/simple/high-conf.
      3. Code-in-task check   — code snippet present + doc_creation/simple = implausible.
    """
    lower = task.lower()

    if l2_type not in _KEYWORD_OPTIONAL_TYPES:
        scores    = _score_task_type(lower)
        l2_score  = scores.get(l2_type, 0.0)
        top_score = max(scores.values(), default=0.0)
        top_type  = max(scores, key=scores.get)

        if (
            l2_score <= 0
            and top_score > 3
            and top_type != l2_type
        ):
            l2_weight  = _TASK_TYPE_TIER_WEIGHT.get(l2_type, 0)
            top_weight = _TASK_TYPE_TIER_WEIGHT.get(top_type, 0)
            if abs(top_weight - l2_weight) >= 2:
                logger.debug(
                    "l2_validation: keyword cross-check failed — "
                    "type=%s (score=%.1f) but %s scores %.1f (tier diff=%d)",
                    l2_type.value, l2_score, top_type.value, top_score,
                    abs(top_weight - l2_weight),
                )
                return False

    is_trivial_classification = (
        l2_type == TaskType.CONVERSATION
        and l2_complexity == TaskComplexity.SIMPLE
    )
    if is_trivial_classification and l2_conf > 0.80:
        tokens = _count_tokens(task)
        if tokens > 60:
            logger.debug(
                "l2_validation: complexity sanity failed — "
                "conversation/simple conf=%.2f for %d-token task",
                l2_conf, tokens,
            )
            return False

    if (
        _CODE_SNIPPET_RE.search(task)
        and l2_type == TaskType.DOC_CREATION
        and l2_complexity == TaskComplexity.SIMPLE
    ):
        logger.debug(
            "l2_validation: code-in-task check failed — "
            "code snippet present but classified as doc_creation/simple"
        )
        return False

    return True
