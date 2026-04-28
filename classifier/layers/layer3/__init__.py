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
from typing import Optional

from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.infra.config import settings

logger = logging.getLogger(__name__)


def classify_layer3(
    task: str,
    history: Optional[list[str]] = None,
) -> Optional[tuple[TaskType, TaskComplexity, ModelTier, float, str]]:
    """Dispatch to the configured L3 strategy. Returns None on abstain/failure."""
    strategy = settings.layer3_strategy

    if strategy == "zeroshot":
        from .zeroshot import classify_layer3_zeroshot
        return classify_layer3_zeroshot(task, history=history)

    if strategy == "head":
        from .embed_classifier import classify_layer3_head
        return classify_layer3_head(task, history=history)

    if strategy == "distilbert":
        # Stage 3 — fine-tuned DistilBERT ONNX (not yet implemented)
        logger.debug("layer3: 'distilbert' strategy not yet implemented — skipping")
        return None

    logger.warning("layer3: unknown strategy=%r — skipping", strategy)
    return None


__all__ = ["classify_layer3"]
