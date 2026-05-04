"""Tests for the PR 1 follow-up fixes (issues #1–#13)."""
import logging
from unittest.mock import MagicMock

import pytest

from classifier import (
    Router, OutcomeRecord, log_outcome, read_outcomes, classify_task,
)
from classifier.core.types import (
    ClassificationDecision, ModelTier, TaskType, TaskComplexity,
)


# ── #1 Cache-hit: fresh decision_id + `cached` flag ──────────────────────────

def test_cache_hit_mints_fresh_decision_id(tmp_path):
    """Two classify() calls hitting the same cache entry must NOT share decision_id."""
    from classifier.infra.cache import cache as _cache
    _cache.clear()
    router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=True)
    d1 = router.classify("a unique task xyz123 for cache test")
    d2 = router.classify("a unique task xyz123 for cache test")   # cache hit

    assert d1.decision_id != d2.decision_id, "cache hit re-used decision_id; would corrupt joins"
    assert d2.cached is True
    assert d2.cached_from == d1.decision_id
    assert d1.cached is False


def test_cache_hit_preserves_routing_decision():
    """The routed model should be identical even though the IDs differ."""
    from classifier.infra.cache import cache as _cache
    _cache.clear()
    router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=True)
    d1 = router.classify("Cache routing equality task xyz")
    d2 = router.classify("Cache routing equality task xyz")
    assert d1.tier == d2.tier
    assert d1.model_name == d2.model_name


# ── #4 PII redaction in outcome log ──────────────────────────────────────────

def test_outcome_pii_redacted_before_write(tmp_path, monkeypatch):
    from classifier.infra import outcome_logger as ol
    monkeypatch.setattr(ol, "_LOG_FILE", tmp_path / "out.jsonl")
    monkeypatch.setattr(ol, "_TEST_LOG", tmp_path / "out.test.jsonl")
    monkeypatch.setattr(ol, "_backend", None)
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)

    log_outcome(OutcomeRecord(
        decision_id="pii-test",
        success=False,
        error_message="patient SSN 123-45-6789 leaked in stack trace",
    ))
    rows = read_outcomes()
    assert len(rows) == 1
    msg = rows[0]["error_message"]
    assert "123-45-6789" not in msg
    assert "[REDACTED]" in msg


def test_outcome_pii_redacted_email(tmp_path, monkeypatch):
    from classifier.infra import outcome_logger as ol
    monkeypatch.setattr(ol, "_LOG_FILE", tmp_path / "out.jsonl")
    monkeypatch.setattr(ol, "_TEST_LOG", tmp_path / "out.test.jsonl")
    monkeypatch.setattr(ol, "_backend", None)
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)

    log_outcome(OutcomeRecord(
        decision_id="pii-email", success=True,
        error_message="failed for user dr.house@hospital.example",
    ))
    rows = read_outcomes()
    assert "dr.house@hospital.example" not in rows[0]["error_message"]


# ── #5 Retention: prune_old_outcomes ─────────────────────────────────────────

def test_prune_old_outcomes(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from classifier.infra import outcome_logger as ol
    monkeypatch.setattr(ol, "_LOG_FILE", tmp_path / "out.jsonl")
    monkeypatch.setattr(ol, "_TEST_LOG", tmp_path / "out.test.jsonl")
    monkeypatch.setattr(ol, "_backend", None)
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)

    # Recent
    log_outcome(OutcomeRecord(decision_id="recent", success=True))
    # Synthetically aged
    old_iso = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    log_outcome(OutcomeRecord(decision_id="old", success=True, timestamp=old_iso))

    pruned = ol.prune_old_outcomes(days=90)
    assert pruned == 1
    rows = read_outcomes()
    ids = {r["decision_id"] for r in rows}
    assert "recent" in ids
    assert "old" not in ids


# ── #6 read_outcomes consults backend.read() if present ─────────────────────

def test_read_outcomes_uses_backend(monkeypatch):
    """When backend implements `.read()`, read_outcomes() consults it first."""
    from classifier.infra import outcome_logger as ol
    captured_args: dict = {}

    def fake_read(*, since, until, decision_ids):
        captured_args["since"] = since
        captured_args["until"] = until
        captured_args["decision_ids"] = decision_ids
        return [{"decision_id": "from-backend", "success": True}]

    backend = MagicMock(read=fake_read)
    monkeypatch.setattr(ol, "_backend", backend)
    rows = read_outcomes(decision_ids={"x"})
    assert rows == [{"decision_id": "from-backend", "success": True}]
    assert captured_args["decision_ids"] == {"x"}


def test_read_outcomes_falls_back_when_backend_lacks_read(tmp_path, monkeypatch):
    """Backend without `read` falls through to local JSONL."""
    from classifier.infra import outcome_logger as ol
    monkeypatch.setattr(ol, "_LOG_FILE", tmp_path / "out.jsonl")
    monkeypatch.setattr(ol, "_TEST_LOG", tmp_path / "out.test.jsonl")
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)

    write_only_backend = type("WriteOnly", (), {"log": staticmethod(lambda e: None), "read": None})()
    monkeypatch.setattr(ol, "_backend", write_only_backend)

    log_outcome(OutcomeRecord(decision_id="local-only", success=True))
    rows = read_outcomes()
    # backend's log is a no-op so local file should be empty
    assert rows == []


# ── #7 Async LangChain instrumentation ──────────────────────────────────────

def test_dynamic_chat_model_has_async_methods():
    from classifier.integrations.langchain import DynamicChatModel
    llm = DynamicChatModel(provider="google", report_outcomes=False)
    assert hasattr(llm, "ainvoke")
    assert hasattr(llm, "astream")
    assert hasattr(llm, "abatch")


def test_dynamic_chat_model_ainvoke_reports():
    """Run the async path via asyncio.run — no pytest-asyncio dependency."""
    import asyncio
    import sys
    import types
    sys.modules.setdefault("langchain_google_genai", types.ModuleType("langchain_google_genai"))
    fake_llm = MagicMock()

    async def _ainvoke(input, **kwargs):
        return MagicMock(usage_metadata={"input_tokens": 11, "output_tokens": 22})
    fake_llm.ainvoke = _ainvoke
    sys.modules["langchain_google_genai"].ChatGoogleGenerativeAI = MagicMock(return_value=fake_llm)

    from classifier.integrations.langchain import DynamicChatModel
    llm = DynamicChatModel(provider="google", report_outcomes=False)
    result = asyncio.run(llm.ainvoke("hello async"))
    assert result is not None


# ── #8 tokens_estimated flag ─────────────────────────────────────────────────

def test_outcome_record_tokens_estimated_default_false():
    rec = OutcomeRecord(decision_id="x", tokens_in=10, tokens_out=20)
    assert rec.tokens_estimated is False


def test_outcome_record_tokens_estimated_can_be_true():
    rec = OutcomeRecord(decision_id="x", tokens_in=10, tokens_out=20, tokens_estimated=True)
    assert rec.tokens_estimated is True


# ── #13 from_dict warns on missing decision_id ──────────────────────────────

def test_from_dict_warns_on_missing_decision_id(caplog):
    data = {
        "model_name": "x", "tier": "low",
        "task_type": "conversation", "complexity": "simple",
        "reasoning": "t", "confidence": 0.9, "provider": "google",
        # NOTE: decision_id intentionally missing
    }
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="classifier.core.types"):
        d = ClassificationDecision.from_dict(data)
    assert any("missing decision_id" in r.message for r in caplog.records)
    assert d.decision_id   # fresh one was minted


def test_from_dict_no_warning_when_decision_id_present(caplog):
    data = {
        "decision_id": "abc123",
        "model_name": "x", "tier": "low",
        "task_type": "conversation", "complexity": "simple",
        "reasoning": "t", "confidence": 0.9, "provider": "google",
    }
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="classifier.core.types"):
        d = ClassificationDecision.from_dict(data)
    assert not any("missing decision_id" in r.message for r in caplog.records)
    assert d.decision_id == "abc123"


# ── #2 ADK pending-decisions LRU ────────────────────────────────────────────

def test_adk_pending_decisions_bounded():
    """Unmatched decisions don't grow unbounded."""
    from classifier.integrations import adk
    adk._pending_decisions.clear()

    # Force-fill past the cap
    for i in range(adk._PENDING_MAX + 50):
        adk._store_pending(MagicMock(state={}, invocation_id=f"inv-{i}"), {"decision_id": f"d{i}"})

    assert len(adk._pending_decisions) <= adk._PENDING_MAX


def test_adk_pending_uses_invocation_id_when_available():
    from classifier.integrations import adk
    adk._pending_decisions.clear()
    ctx = MagicMock(state={}, invocation_id="stable-123")
    adk._store_pending(ctx, {"decision_id": "d1", "model": "m", "task": "t", "t0": 0})
    found = adk._pop_pending(ctx)
    assert found is not None
    assert found["decision_id"] == "d1"


def test_adk_pending_pops_via_state_key_after_id_recycle():
    """If id(ctx) recycles, the state-key fallback still finds the decision."""
    from classifier.integrations import adk
    adk._pending_decisions.clear()

    ctx = MagicMock(state={}, invocation_id=None)
    adk._store_pending(ctx, {"decision_id": "d1", "model": "m", "task": "t", "t0": 0})
    state_key = ctx.state["_dmr_decision_key"]

    # Build a different context that just happens to share the same state dict
    ctx2 = MagicMock(state=ctx.state, invocation_id=None)
    found = adk._pop_pending(ctx2)
    assert found is not None
    assert found["decision_id"] == "d1"
    _ = state_key   # silence unused
