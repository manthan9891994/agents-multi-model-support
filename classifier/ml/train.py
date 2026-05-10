"""User-facing training API. Backs `Router.train()` and `dmr train`.

Wraps the existing `train_head.py` pipeline with a simpler interface:

    from classifier.ml.train import train_from_data
    metadata = train_from_data("my_data.jsonl")
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def train_from_data(
    data_path: Path,
    *,
    output_path: Path | None = None,
    max_iter: int = 600,
    test_size: float = 0.30,
    cal_fraction: float = 0.50,
    random_state: int = 42,
) -> dict:
    """Train Stage 2 (frozen MiniLM + calibrated MLPs) from a JSONL file.

    The JSONL file must have one object per line with keys:
        task        (str)
        task_type   (str — must match TaskType enum value)
        complexity  (str — must match TaskComplexity enum value)

    Args:
        data_path:    Input JSONL.
        output_path:  Where to save model bundle (default: classifier/ml/models/head_v1.joblib).
        max_iter:     sklearn MLP max iterations.
        test_size:    Fraction held out for calibration + test (default 0.30 → 70/15/15 split).
        cal_fraction: Of held-out, fraction used for calibration (default 0.50 → cal=test).
        random_state: Reproducibility seed.

    Returns:
        Metadata dict with training accuracy, threshold sweep, etc.
    """
    try:
        import joblib
        import numpy as np
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
    except ImportError as exc:
        raise ImportError(
            "Training requires the [ml] extra. Install with:\n"
            "    pip install 'dynamic-model-router[ml]'"
        ) from exc

    from classifier.ml.embeddings import encode

    # 1. Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    texts, task_types, complexities = [], [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if all(k in obj for k in ("task", "task_type", "complexity")):
                    texts.append(obj["task"])
                    task_types.append(obj["task_type"])
                    complexities.append(obj["complexity"])
            except json.JSONDecodeError:
                continue

    if len(texts) < 50:
        raise ValueError(
            f"Need at least 50 examples to train, got {len(texts)}. "
            f"Try `dmr generate-data --per-slot 30` to bootstrap."
        )
    logger.info("Loaded %d examples from %s", len(texts), data_path)

    # 2. Encode with frozen MiniLM
    logger.info("Encoding embeddings (frozen all-MiniLM-L6-v2)...")
    X = encode(texts)
    if X is None:
        raise RuntimeError(
            "Encoder unavailable. Install with: pip install 'dynamic-model-router[ml]'"
        )
    X = np.asarray(X)
    logger.info("Embedding shape: %s", X.shape)

    # 3. Three-way split
    X_train, X_temp, tt_train, tt_temp, cx_train, cx_temp = train_test_split(
        X, task_types, complexities,
        test_size=test_size, random_state=random_state, stratify=task_types,
    )
    X_cal, X_te, tt_cal, tt_te, cx_cal, cx_te = train_test_split(
        X_temp, tt_temp, cx_temp,
        test_size=cal_fraction, random_state=random_state, stratify=tt_temp,
    )
    logger.info("Split: train=%d cal=%d test=%d", len(X_train), len(X_cal), len(X_te))

    def _train_head(name, X_tr, y_tr, X_cal, y_cal, X_te, y_te):
        base = MLPClassifier(
            hidden_layer_sizes=(256,), activation="relu", solver="adam",
            max_iter=max_iter, random_state=random_state, verbose=False,
        )
        base.fit(X_tr, y_tr)
        cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        cal.fit(X_cal, y_cal)
        acc = float(accuracy_score(y_te, cal.predict(X_te)))
        logger.info("[%s] calibrated test acc: %.3f", name, acc)
        return cal, acc

    tt_clf, tt_acc = _train_head("task_type",  X_train, tt_train, X_cal, tt_cal, X_te, tt_te)
    cx_clf, cx_acc = _train_head("complexity", X_train, cx_train, X_cal, cx_cal, X_te, cx_te)

    # 4. Threshold sweep
    sweep = _threshold_sweep(tt_clf, cx_clf, X_te, tt_te, cx_te)
    logger.info("Threshold sweep:")
    for t, m in sweep.items():
        prec = f"{m['precision']:.3f}" if m["precision"] is not None else "—"
        logger.info("  %.2f: intercept=%.1f%% precision=%s n=%d",
                    t, m["intercept_rate"]*100, prec, m["n"])

    # 5. Save bundle
    if output_path is None:
        output_path = Path(__file__).parent / "models" / "head_v1.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "task_type_clf":  tt_clf,
        "complexity_clf": cx_clf,
        "task_type_classes":  list(tt_clf.classes_),
        "complexity_classes": list(cx_clf.classes_),
    }
    joblib.dump(bundle, output_path)

    metadata = {
        "trained_at":              datetime.now(timezone.utc).isoformat(),
        "n_examples":              len(texts),
        "task_type_test_accuracy":  tt_acc,
        "complexity_test_accuracy": cx_acc,
        "geo_mean_accuracy":       round((tt_acc * cx_acc) ** 0.5, 3),
        "encoder":                 "all-MiniLM-L6-v2",
        "architecture":            "MLPClassifier(256,) per head + isotonic calibration",
        "threshold_sweep":         {str(k): v for k, v in sweep.items()},
        "model_path":              str(output_path),
    }
    meta_path = output_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model → %s", output_path)
    return metadata


def _threshold_sweep(tt_clf, cx_clf, X_te, y_tt, y_cx) -> dict:
    import numpy as np
    tt_probs = tt_clf.predict_proba(X_te)
    cx_probs = cx_clf.predict_proba(X_te)
    tt_pred  = tt_clf.classes_[np.argmax(tt_probs, axis=1)]
    cx_pred  = cx_clf.classes_[np.argmax(cx_probs, axis=1)]
    confidence = (np.max(tt_probs, axis=1) * np.max(cx_probs, axis=1)) ** 0.5
    correct = (np.array(y_tt) == tt_pred) & (np.array(y_cx) == cx_pred)
    results: dict = {}
    for t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = confidence >= t
        if mask.sum() == 0:
            results[t] = {"intercept_rate": 0.0, "precision": None, "n": 0}
            continue
        results[t] = {
            "intercept_rate": float(mask.mean()),
            "precision":      float(correct[mask].mean()),
            "n":              int(mask.sum()),
        }
    return results
