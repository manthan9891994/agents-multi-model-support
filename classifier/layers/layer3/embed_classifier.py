"""Layer 3 — Stage 2 runtime: frozen MiniLM + sklearn MLP heads.

Loads the trained bundle from `classifier/ml/models/head_v1.joblib`. If the model
file is missing → returns None gracefully (cascade falls through to L2).

Latency:  ~10–15ms per call (CPU)
Accuracy: ~85–90% on confident decisions (depends on training data)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.core.registry import TIER_MATRIX
from classifier.infra.config import settings

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent.parent / "ml" / "models" / "head_v1.joblib"

_lock = threading.Lock()
_bundle = None
_load_failed = False


def _load_bundle():
    """Lazy-load the joblib bundle. Returns None if missing or load failed."""
    global _bundle, _load_failed
    if _bundle is not None:
        return _bundle
    if _load_failed:
        return None
    with _lock:
        if _bundle is not None:
            return _bundle
        if _load_failed:
            return None
        if not _MODEL_PATH.exists():
            logger.warning(
                "layer3.head: model not found at %s — "
                "run `python -m classifier.ml.train_head` first",
                _MODEL_PATH,
            )
            _load_failed = True
            return None
        try:
            import joblib
            _bundle = joblib.load(_MODEL_PATH)
            logger.info("layer3.head: loaded model from %s", _MODEL_PATH.name)
            return _bundle
        except Exception as exc:
            logger.warning("layer3.head: model load failed — %s", exc)
            _load_failed = True
            return None


def classify_layer3_head(
    task: str,
    history: Optional[list[str]] = None,
) -> Optional[tuple[TaskType, TaskComplexity, ModelTier, float, str]]:
    """Classify with frozen-encoder + MLP heads. Returns None on abstain or failure."""
    bundle = _load_bundle()
    if bundle is None:
        return None

    try:
        from classifier.ml.embeddings import encode_one
        # Combine with last history turn for continuation context
        text = task[:512]
        if history:
            text = (history[-1][:200] + " | " + text)[:512]

        vec = encode_one(text)
        if vec is None:
            return None
        vec = np.asarray(vec).reshape(1, -1)

        tt_clf = bundle["task_type_clf"]
        cx_clf = bundle["complexity_clf"]

        tt_probs = tt_clf.predict_proba(vec)[0]
        cx_probs = cx_clf.predict_proba(vec)[0]

        tt_idx = int(np.argmax(tt_probs))
        cx_idx = int(np.argmax(cx_probs))
        tt_prob = float(tt_probs[tt_idx])
        cx_prob = float(cx_probs[cx_idx])

        tt_label = bundle["task_type_classes"][tt_idx]
        cx_label = bundle["complexity_classes"][cx_idx]

        try:
            task_type  = TaskType(tt_label)
            complexity = TaskComplexity(cx_label)
        except ValueError as exc:
            logger.warning("layer3.head: unknown label in bundle — %s", exc)
            return None

        # Geometric mean of both head probabilities
        confidence = (tt_prob * cx_prob) ** 0.5

        if confidence < settings.layer3_confidence_threshold:
            logger.debug(
                "layer3.head: abstaining — conf=%.2f < %.2f (tt=%.2f cx=%.2f)",
                confidence, settings.layer3_confidence_threshold, tt_prob, cx_prob,
            )
            return None

        tier = TIER_MATRIX[(task_type, complexity)]
        reasoning = (
            f"layer3 | head | {task_type.value}/{complexity.value} "
            f"| conf={confidence:.2f} (tt={tt_prob:.2f} cx={cx_prob:.2f})"
        )
        return task_type, complexity, tier, confidence, reasoning

    except Exception as exc:
        logger.warning("layer3.head: classification failed — %s", exc)
        return None
