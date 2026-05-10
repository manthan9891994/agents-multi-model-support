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

from classifier.core.types import ModelTier, TaskComplexity, TaskType
from classifier.infra.config import settings

logger = logging.getLogger(__name__)

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

    if strategy in _STRATEGIES:
        return _STRATEGIES[strategy](task, history=history)

    fn = _builtin(strategy)
    if fn is not None:
        return fn(task, history=history)

    if strategy == "distilbert":
        logger.debug("layer3: 'distilbert' strategy not yet implemented — skipping")
        return None

    logger.warning("layer3: unknown strategy=%r — skipping", strategy)
    return None


__all__ = ["classify_layer3", "register_strategy", "list_strategies"]
