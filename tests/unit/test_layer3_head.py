"""Unit tests for Layer 3 Stage 2 — frozen MiniLM + sklearn MLP heads.

The encoder and MLPs are mocked at the module level. Sentence-transformers does
NOT need to be installed for these tests.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def reset_head_singletons():
    """Reset module-level cache between tests."""
    import classifier.layers.layer3.embed_classifier as eh
    eh._bundle = None
    eh._load_failed = False
    yield
    eh._bundle = None
    eh._load_failed = False


def _make_mock_bundle(tt_label: str, tt_prob: float, cx_label: str, cx_prob: float):
    """Build a fake bundle matching the joblib structure."""
    tt_clf = MagicMock()
    tt_clf.predict_proba.return_value = np.array([[1.0 - tt_prob, tt_prob]])
    tt_clf.classes_ = ["filler", tt_label]

    cx_clf = MagicMock()
    cx_clf.predict_proba.return_value = np.array([[1.0 - cx_prob, cx_prob]])
    cx_clf.classes_ = ["filler", cx_label]

    return {
        "task_type_clf":      tt_clf,
        "complexity_clf":     cx_clf,
        "task_type_classes":  ["filler", tt_label],
        "complexity_classes": ["filler", cx_label],
    }


def _patch_encoder(monkeypatch, vec=None):
    """Patch encode_one to return a fixed 384-dim vector."""
    if vec is None:
        vec = np.random.rand(384).astype(np.float32)
    import classifier.ml.embeddings as emb
    monkeypatch.setattr(emb, "encode_one", lambda text: vec)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_confident_head_returns_decision(monkeypatch):
    from classifier.core.types import TaskType, TaskComplexity, ModelTier
    import classifier.layers.layer3.embed_classifier as eh

    eh._bundle = _make_mock_bundle("code_creation", 0.95, "standard", 0.90)
    _patch_encoder(monkeypatch)

    result = eh.classify_layer3_head("Implement a REST API endpoint")
    assert result is not None

    task_type, complexity, tier, conf, reasoning = result
    assert task_type  == TaskType.CODE_CREATION
    assert complexity == TaskComplexity.STANDARD
    assert tier       == ModelTier.MEDIUM
    assert 0.92 < conf < 0.93  # geo-mean of 0.95 and 0.90
    assert reasoning.startswith("layer3 | head | code_creation/standard")


def test_reasoning_includes_both_head_probs(monkeypatch):
    import classifier.layers.layer3.embed_classifier as eh

    eh._bundle = _make_mock_bundle("reasoning", 0.93, "complex", 0.88)
    _patch_encoder(monkeypatch)

    _, _, _, _, reasoning = eh.classify_layer3_head("Compare X and Y in depth")
    assert "tt=0.93" in reasoning
    assert "cx=0.88" in reasoning


# ── Abstain logic ─────────────────────────────────────────────────────────────

def test_low_confidence_abstains(monkeypatch):
    import classifier.layers.layer3.embed_classifier as eh

    # 0.70 * 0.65 → sqrt ≈ 0.675, below default 0.85 threshold
    eh._bundle = _make_mock_bundle("code_creation", 0.70, "simple", 0.65)
    _patch_encoder(monkeypatch)

    result = eh.classify_layer3_head("ambiguous task")
    assert result is None


def test_asymmetric_confidence_abstains(monkeypatch):
    """One head sure, other unsure → geometric mean penalises → abstain."""
    import classifier.layers.layer3.embed_classifier as eh

    # 0.99 * 0.55 → sqrt ≈ 0.738, below 0.85 threshold
    eh._bundle = _make_mock_bundle("reasoning", 0.99, "standard", 0.55)
    _patch_encoder(monkeypatch)

    result = eh.classify_layer3_head("a task")
    assert result is None


# ── Failure modes ─────────────────────────────────────────────────────────────

def test_missing_model_returns_none(tmp_path, monkeypatch):
    """No trained model on disk → returns None gracefully → cascade falls to L2."""
    import classifier.layers.layer3.embed_classifier as eh

    monkeypatch.setattr(eh, "_MODEL_PATH", tmp_path / "nonexistent.joblib")
    eh._bundle = None
    eh._load_failed = False

    result = eh.classify_layer3_head("any task")
    assert result is None
    assert eh._load_failed is True  # short-circuit subsequent calls


def test_load_failed_short_circuits():
    import classifier.layers.layer3.embed_classifier as eh
    eh._load_failed = True
    assert eh._load_bundle() is None


def test_encoder_unavailable_returns_none(monkeypatch):
    """When sentence-transformers isn't available, encode_one returns None."""
    import classifier.layers.layer3.embed_classifier as eh
    import classifier.ml.embeddings as emb

    eh._bundle = _make_mock_bundle("math", 0.95, "simple", 0.92)
    monkeypatch.setattr(emb, "encode_one", lambda text: None)

    result = eh.classify_layer3_head("calculate 2+2")
    assert result is None


def test_unknown_label_returns_none(monkeypatch):
    """If trained model emits an unknown enum value, classifier returns None."""
    import classifier.layers.layer3.embed_classifier as eh

    eh._bundle = _make_mock_bundle("nonsense_type", 0.95, "simple", 0.95)
    _patch_encoder(monkeypatch)

    result = eh.classify_layer3_head("any task")
    assert result is None


# ── History context ──────────────────────────────────────────────────────────

def test_history_prepended_to_input(monkeypatch):
    import classifier.layers.layer3.embed_classifier as eh
    import classifier.ml.embeddings as emb

    eh._bundle = _make_mock_bundle("code_creation", 0.95, "simple", 0.92)

    captured: list[str] = []
    def fake_encode(text):
        captured.append(text)
        return np.random.rand(384).astype(np.float32)
    monkeypatch.setattr(emb, "encode_one", fake_encode)

    eh.classify_layer3_head("now make it faster", history=["implement binary search"])

    assert len(captured) == 1
    assert "implement binary search" in captured[0]
    assert "now make it faster" in captured[0]


# ── Strategy router ──────────────────────────────────────────────────────────

def test_strategy_router_dispatches_head(monkeypatch):
    """settings.layer3_strategy='head' → calls embed_classifier implementation."""
    from classifier.layers.layer3 import classify_layer3
    import classifier.layers.layer3.embed_classifier as eh
    from classifier.infra.config import settings
    from classifier.core.types import TaskType

    eh._bundle = _make_mock_bundle("math", 0.94, "simple", 0.92)
    _patch_encoder(monkeypatch)

    original = settings.layer3_strategy
    try:
        settings.layer3_strategy = "head"
        result = classify_layer3("solve x + 2 = 5")
        assert result is not None
        assert result[0] == TaskType.MATH
    finally:
        settings.layer3_strategy = original


# ── Cascade integration ──────────────────────────────────────────────────────

def test_cascade_l3_head_skips_l2(monkeypatch):
    """When L3 head is confident, L2 should not fire."""
    from unittest.mock import patch
    from classifier.core.types import TaskType, TaskComplexity
    from classifier.infra.config import settings
    from classifier.infra.cache import cache
    import classifier.layers.layer3.embed_classifier as eh

    eh._bundle = _make_mock_bundle("reasoning", 0.95, "complex", 0.90)
    _patch_encoder(monkeypatch)
    cache.clear()

    o_l3 = settings.layer3_enabled
    o_l2 = settings.layer2_enabled
    o_strat = settings.layer3_strategy
    try:
        settings.layer3_enabled = True
        settings.layer2_enabled = True
        settings.layer3_strategy = "head"

        with patch("classifier.layers.layer2.api.genai.Client") as mock_l2:
            from classifier import classify_task
            decision = classify_task("xyzzy frobnicate quuxify wibble", provider="google")

        assert decision.layer_used == "layer3"
        mock_l2.assert_not_called()
    finally:
        settings.layer3_enabled = o_l3
        settings.layer2_enabled = o_l2
        settings.layer3_strategy = o_strat
        cache.clear()
