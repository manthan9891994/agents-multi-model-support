"""Layer 3 — strategy router for embedding/ML classifiers.

Three strategies, each shippable independently:

| Strategy   | Stage | Latency  | Accuracy | Training data |
|------------|-------|----------|----------|---------------|
| zeroshot   | 1     | ~80ms    | ~80%     | None          |
| head       | 2     | ~15ms    | ~90%     | 1,500+        |
| distilbert | 3     | ~12ms    | ~95%     | 5,000+        |

Selection is controlled by `settings.layer3_strategy`. All strategies share
the same return signature `(TaskType, TaskComplexity, ModelTier, float, str) | None`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from classifier.core.types import _TIER_ORDER, ModelTier, TaskComplexity, TaskType
from classifier.infra.config import settings

logger = logging.getLogger(__name__)


def _apply_savings_level(tier: ModelTier, level: int) -> ModelTier:
    """Bias an L3 tier toward cheaper models by `level` steps (clamped at lowest).

    The quality<->savings dial (settings.l3_dmr_savings_level / env
    L3_DMR_SAVINGS_LEVEL): 0 keeps L3's natural tier; 1 = one tier cheaper
    (HIGH->MEDIUM, MEDIUM->LOW); 2+ = floor at LOW. Lets one trained head run
    anywhere on the cost/quality frontier without retraining.
    """
    if level <= 0:
        return tier
    i = _TIER_ORDER.index(tier)
    return _TIER_ORDER[max(0, i - level)]


# Strategy registry: {name: callable(task, history) -> tuple | None}
_STRATEGIES: dict[str, Callable] = {}


def register_strategy(name: str, fn: Callable) -> None:
    """Register a custom L3 strategy.

    Args:
        name: Strategy name. Set settings.layer3_strategy to this value to use it.
        fn:   Callable(task, history) → (TaskType, TaskComplexity, ModelTier, conf, reason) | None.

    Example:
        from classifier.layers.layer3 import register_strategy

        def my_hf_strategy(task, history=None):
            # use a HuggingFace pipeline...
            return TaskType.REASONING, TaskComplexity.STANDARD, ModelTier.MEDIUM, 0.92, "hf"

        register_strategy("hf_pipeline", my_hf_strategy)
        # Then settings.layer3_strategy = "hf_pipeline"
    """
    _STRATEGIES[name] = fn


def list_strategies() -> list[str]:
    """All registered L3 strategy names (built-ins are lazy-loaded)."""
    return sorted(set(list(_STRATEGIES) + ["zeroshot", "head", "distilbert"]))


def _builtin(name: str) -> Callable | None:
    if name == "zeroshot":
        from .zeroshot import classify_layer3_zeroshot

        return classify_layer3_zeroshot
    if name == "head":
        from .embed_classifier import classify_layer3_head

        return classify_layer3_head
    return None


def classify_layer3(
    task: str,
    history: list[str] | None = None,
) -> tuple[TaskType, TaskComplexity, ModelTier, float, str] | None:
    """Dispatch to the configured L3 strategy. Returns None on abstain/failure."""
    strategy = settings.layer3_strategy

    fn = _STRATEGIES.get(strategy) or _builtin(strategy)
    if fn is None:
        if strategy == "distilbert":
            logger.debug("layer3: 'distilbert' strategy not yet implemented — skipping")
        else:
            logger.warning("layer3: unknown strategy=%r — skipping", strategy)
        return None

    result = fn(task, history=history)
    if result is None:
        return None

    # Quality<->savings dial: bias L3's chosen tier toward cheaper models.
    level = settings.l3_dmr_savings_level
    if level > 0:
        task_type, complexity, tier, confidence, reasoning = result
        shifted = _apply_savings_level(tier, level)
        if shifted is not tier:
            reasoning = f"{reasoning} | savings_level={level} ({tier.value}->{shifted.value})"
        result = (task_type, complexity, shifted, confidence, reasoning)
    return result


__all__ = ["classify_layer3", "register_strategy", "list_strategies"]
