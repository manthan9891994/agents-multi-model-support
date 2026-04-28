"""Layer 3 — Stage 1 zero-shot classifier (no training data required).

Uses Hugging Face NLI-based zero-shot classification (`facebook/bart-large-mnli`)
to classify task_type and complexity in two parallel calls. Returns None when
confidence falls below threshold → cascade falls through to Layer 2.

Latency:  ~80ms per call (CPU)        Model size: ~400MB
Accuracy: ~80% on confident decisions  Training data: zero
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.core.registry import TIER_MATRIX
from classifier.infra.config import settings

logger = logging.getLogger(__name__)

# Hypothesis templates — descriptive strings give the NLI model better signal
# than bare category names. Map each hypothesis back to its enum value.
_TASK_TYPE_HYPOTHESES: dict[str, TaskType] = {
    "this is a casual conversation, greeting, or small talk":           TaskType.CONVERSATION,
    "this is a request to write code, implement, or fix a function":    TaskType.CODE_CREATION,
    "this is a request to write documentation, summary, or report":     TaskType.DOC_CREATION,
    "this is a reasoning task comparing options or analyzing trade-offs": TaskType.REASONING,
    "this is a planning task designing a system, architecture, or strategy": TaskType.THINKING,
    "this is a data analysis task finding patterns, trends, or insights":   TaskType.ANALYZING,
    "this is a translation between human languages":                    TaskType.TRANSLATION,
    "this is a math, calculation, or numerical problem":                TaskType.MATH,
    "this is a multimodal task involving images, audio, or video":      TaskType.MULTIMODAL,
}

_COMPLEXITY_HYPOTHESES: dict[str, TaskComplexity] = {
    "this task is trivial requiring a one or two sentence answer":      TaskComplexity.SIMPLE,
    "this task is standard requiring a structured multi-paragraph response": TaskComplexity.STANDARD,
    "this task is complex requiring a multi-part deeply reasoned response":  TaskComplexity.COMPLEX,
    "this task is research-level requiring extensive expert reasoning":  TaskComplexity.RESEARCH,
}

# Module-level singletons — pipeline takes 5–10s to load (~400MB model)
_lock = threading.Lock()
_pipeline = None
_load_failed = False


def _load_pipeline():
    """Lazy-load the transformers zero-shot pipeline. Returns None if unavailable."""
    global _pipeline, _load_failed
    if _pipeline is not None:
        return _pipeline
    if _load_failed:
        return None
    with _lock:
        if _pipeline is not None:
            return _pipeline
        if _load_failed:
            return None
        try:
            from transformers import pipeline
            _pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
            logger.info("layer3.zeroshot: loaded bart-large-mnli pipeline")
            return _pipeline
        except ImportError:
            logger.warning(
                "layer3.zeroshot: transformers not installed — "
                "run `pip install transformers torch` to enable Stage 1 L3"
            )
            _load_failed = True
            return None
        except Exception as exc:
            logger.warning("layer3.zeroshot: pipeline load failed — %s", exc)
            _load_failed = True
            return None


def classify_layer3_zeroshot(
    task: str,
    history: Optional[list[str]] = None,
) -> Optional[tuple[TaskType, TaskComplexity, ModelTier, float, str]]:
    """Zero-shot classification using NLI. Returns None on low confidence or failure."""
    pipe = _load_pipeline()
    if pipe is None:
        return None

    try:
        # Truncate to BART's effective context window. Combine with last history turn
        # if available — gives the model continuation context for short follow-ups.
        truncated = task[:512]
        if history:
            truncated = (history[-1][:200] + " | " + truncated)[:512]

        tt_labels = list(_TASK_TYPE_HYPOTHESES.keys())
        cx_labels = list(_COMPLEXITY_HYPOTHESES.keys())

        tt_result = pipe(truncated, candidate_labels=tt_labels, multi_label=False)
        cx_result = pipe(truncated, candidate_labels=cx_labels, multi_label=False)

        top_tt_label = tt_result["labels"][0]
        top_tt_prob  = float(tt_result["scores"][0])
        top_cx_label = cx_result["labels"][0]
        top_cx_prob  = float(cx_result["scores"][0])

        task_type  = _TASK_TYPE_HYPOTHESES[top_tt_label]
        complexity = _COMPLEXITY_HYPOTHESES[top_cx_label]

        # Geometric mean — penalises asymmetric confidence (one head sure, other guessing)
        confidence = (top_tt_prob * top_cx_prob) ** 0.5

        threshold = settings.layer3_zeroshot_threshold
        if confidence < threshold:
            logger.debug(
                "layer3.zeroshot: abstaining — conf=%.2f < %.2f (tt=%.2f cx=%.2f)",
                confidence, threshold, top_tt_prob, top_cx_prob,
            )
            return None

        tier = TIER_MATRIX[(task_type, complexity)]
        reasoning = (
            f"layer3 | zeroshot | {task_type.value}/{complexity.value} "
            f"| conf={confidence:.2f} (tt={top_tt_prob:.2f} cx={top_cx_prob:.2f})"
        )
        return task_type, complexity, tier, confidence, reasoning

    except Exception as exc:
        logger.warning("layer3.zeroshot: classification failed — %s", exc)
        return None
