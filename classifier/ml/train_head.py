"""Train Stage 2 classifier heads — frozen MiniLM embeddings + sklearn MLPs.

Two MLPs (one per output: task_type, complexity) trained on top of frozen
`all-MiniLM-L6-v2` embeddings. Saves to:

    classifier/ml/models/head_v1.joblib
    classifier/ml/models/head_v1.metadata.json

Usage:
    python -m classifier.ml.train_head

Trains in <30s on CPU for ≤5,000 examples.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from classifier.ml.data_loader import load_examples
from classifier.ml.embeddings import encode

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODELS_DIR / "head_v1.joblib"
_META_PATH  = _MODELS_DIR / "head_v1.metadata.json"


def _train_mlp(X: np.ndarray, y: list[str], name: str) -> tuple[MLPClassifier, float]:
    """Train one sklearn MLP. Returns (model, validation_accuracy)."""
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_va)
    acc = float(accuracy_score(y_va, y_pred))
    logger.info("[%s] val accuracy: %.3f", name, acc)
    logger.debug("\n%s", classification_report(y_va, y_pred, zero_division=0))
    return clf, acc


def main() -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load labeled data
    texts, task_types, complexities = load_examples(include_synthetic=True)
    if len(texts) < 50:
        raise SystemExit(
            f"Only {len(texts)} labeled examples found — need at least 50. "
            f"Run `python -m classifier.ml.generate_synthetic --per-slot 30` first."
        )
    logger.info("Loaded %d examples", len(texts))

    # 2. Encode → frozen 384-dim embeddings
    logger.info("Encoding embeddings...")
    X = encode(texts)
    if X is None:
        raise SystemExit("Encoder unavailable — sentence-transformers not installed?")
    X = np.asarray(X)
    logger.info("Embedding shape: %s", X.shape)

    # 3. Train two heads independently
    tt_clf, tt_acc = _train_mlp(X, task_types,   "task_type")
    cx_clf, cx_acc = _train_mlp(X, complexities, "complexity")

    # 4. Bundle and save
    bundle = {
        "task_type_clf":  tt_clf,
        "complexity_clf": cx_clf,
        "task_type_classes":  list(tt_clf.classes_),
        "complexity_classes": list(cx_clf.classes_),
    }
    joblib.dump(bundle, _MODEL_PATH)
    metadata = {
        "trained_at":             datetime.now(timezone.utc).isoformat(),
        "n_examples":             len(texts),
        "task_type_val_accuracy": tt_acc,
        "complexity_val_accuracy": cx_acc,
        "encoder":                "all-MiniLM-L6-v2",
        "architecture":           "MLPClassifier(256,) per head",
    }
    _META_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model → %s", _MODEL_PATH)
    logger.info("Metadata → %s", _META_PATH)
    logger.info("Combined val accuracy (geo-mean): %.3f", (tt_acc * cx_acc) ** 0.5)


if __name__ == "__main__":
    main()
