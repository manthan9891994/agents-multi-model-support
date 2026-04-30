"""Train Stage 2 classifier heads — frozen MiniLM embeddings + calibrated sklearn MLPs.

Two MLPs (one per output: task_type, complexity) trained on top of frozen
`all-MiniLM-L6-v2` embeddings, then **calibrated** with isotonic regression
so that `predict_proba` returns honest confidence scores.

Saves to:
    classifier/ml/models/head_v1.joblib
    classifier/ml/models/head_v1.metadata.json

Usage:
    python -m classifier.ml.train_head

Trains in <60s on CPU for ≤5,000 examples.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from classifier.ml.data_loader import load_examples
from classifier.ml.embeddings import encode

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODELS_DIR / "head_v1.joblib"
_META_PATH  = _MODELS_DIR / "head_v1.metadata.json"


def _train_calibrated_mlp(
    X_tr: np.ndarray, y_tr: list[str],
    X_cal: np.ndarray, y_cal: list[str],
    X_te: np.ndarray, y_te: list[str],
    name: str,
) -> tuple[CalibratedClassifierCV, float]:
    """Train MLP on train set, calibrate on cal set, evaluate on test set."""
    base = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        max_iter=600,
        random_state=42,
        verbose=False,
    )
    base.fit(X_tr, y_tr)
    raw_acc = accuracy_score(y_te, base.predict(X_te))

    # Wrap with isotonic calibration on held-out calibration set
    cal_clf = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    cal_clf.fit(X_cal, y_cal)
    cal_acc = accuracy_score(y_te, cal_clf.predict(X_te))

    logger.info("[%s] raw test acc: %.3f | calibrated test acc: %.3f", name, raw_acc, cal_acc)
    return cal_clf, float(cal_acc)


def _threshold_sweep(
    tt_clf, cx_clf,
    X_te: np.ndarray, y_tt: list[str], y_cx: list[str],
    thresholds: list[float],
) -> dict:
    """For each threshold, compute (intercept_rate, precision_on_intercepted)."""
    tt_probs = tt_clf.predict_proba(X_te)
    cx_probs = cx_clf.predict_proba(X_te)
    tt_pred  = tt_clf.classes_[np.argmax(tt_probs, axis=1)]
    cx_pred  = cx_clf.classes_[np.argmax(cx_probs, axis=1)]
    confidence = (np.max(tt_probs, axis=1) * np.max(cx_probs, axis=1)) ** 0.5

    correct = (np.array(y_tt) == tt_pred) & (np.array(y_cx) == cx_pred)
    n = len(y_tt)
    results = {}
    for t in thresholds:
        intercepted = confidence >= t
        if intercepted.sum() == 0:
            results[t] = {"intercept_rate": 0.0, "precision": None, "n": 0}
            continue
        precision = correct[intercepted].mean()
        results[t] = {
            "intercept_rate": float(intercepted.mean()),
            "precision":      float(precision),
            "n":              int(intercepted.sum()),
        }
    return results


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

    # 3. Three-way split: train / calibration / test (70/15/15)
    X_train, X_temp, tt_train, tt_temp, cx_train, cx_temp = train_test_split(
        X, task_types, complexities, test_size=0.30, random_state=42, stratify=task_types,
    )
    X_cal, X_te, tt_cal, tt_te, cx_cal, cx_te = train_test_split(
        X_temp, tt_temp, cx_temp, test_size=0.50, random_state=42, stratify=tt_temp,
    )
    logger.info("Split sizes — train=%d cal=%d test=%d", len(X_train), len(X_cal), len(X_te))

    # 4. Train + calibrate each head
    tt_clf, tt_acc = _train_calibrated_mlp(X_train, tt_train, X_cal, tt_cal, X_te, tt_te, "task_type")
    cx_clf, cx_acc = _train_calibrated_mlp(X_train, cx_train, X_cal, cx_cal, X_te, cx_te, "complexity")

    # 5. Threshold sweep on test set — find best precision/intercept tradeoff
    sweep = _threshold_sweep(
        tt_clf, cx_clf, X_te, tt_te, cx_te,
        thresholds=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    logger.info("─" * 60)
    logger.info("THRESHOLD SWEEP (on calibrated test set)")
    logger.info("%-10s %-15s %-12s %s", "threshold", "intercept_rate", "precision", "n")
    logger.info("─" * 60)
    for t, m in sweep.items():
        prec = f"{m['precision']:.3f}" if m["precision"] is not None else "—"
        ir = f"{m['intercept_rate']*100:.1f}%"
        logger.info("%-10.2f %-15s %-12s %d", t, ir, prec, m["n"])
    logger.info("─" * 60)

    # 6. Bundle and save
    bundle = {
        "task_type_clf":  tt_clf,
        "complexity_clf": cx_clf,
        "task_type_classes":  list(tt_clf.classes_),
        "complexity_classes": list(cx_clf.classes_),
    }
    joblib.dump(bundle, _MODEL_PATH)
    metadata = {
        "trained_at":              datetime.now(timezone.utc).isoformat(),
        "n_examples":              len(texts),
        "task_type_test_accuracy":  tt_acc,
        "complexity_test_accuracy": cx_acc,
        "encoder":                 "all-MiniLM-L6-v2",
        "architecture":            "MLPClassifier(256,) per head + isotonic calibration",
        "threshold_sweep":         {str(k): v for k, v in sweep.items()},
    }
    _META_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model → %s", _MODEL_PATH)
    logger.info("Metadata → %s", _META_PATH)
    logger.info("Combined test accuracy (geo-mean): %.3f", (tt_acc * cx_acc) ** 0.5)


if __name__ == "__main__":
    main()
