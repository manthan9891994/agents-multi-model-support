"""Tests for the outcome logger and the decision_id ⨝ outcome_id join key."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from classifier import (
    OutcomeRecord,
    Router,
    join_decisions_outcomes,
    log_outcome,
    read_outcomes,
)
from classifier.core.types import ClassificationDecision, ModelTier, TaskComplexity, TaskType


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    """Redirect outcome log to a per-test file and clear backend."""
    from classifier.infra import outcome_logger as ol
    monkeypatch.setattr(ol, "_LOG_FILE", tmp_path / "outcomes.jsonl")
    monkeypatch.setattr(ol, "_TEST_LOG", tmp_path / "outcomes.test.jsonl")
    monkeypatch.setattr(ol, "_backend", None)
    yield tmp_path


# ── Decision ID generation ──────────────────────────────────────────────────

def test_decision_has_unique_id():
    d1 = ClassificationDecision(
        model_name="x", tier=ModelTier.LOW,
        task_type=TaskType.CONVERSATION, complexity=TaskComplexity.SIMPLE,
        reasoning="t", confidence=0.9, provider="google",
    )
    d2 = ClassificationDecision(
        model_name="x", tier=ModelTier.LOW,
        task_type=TaskType.CONVERSATION, complexity=TaskComplexity.SIMPLE,
        reasoning="t", confidence=0.9, provider="google",
    )
    assert d1.decision_id and d2.decision_id
    assert d1.decision_id != d2.decision_id
    assert len(d1.decision_id) == 16


def test_decision_to_dict_includes_decision_id():
    d = ClassificationDecision(
        model_name="x", tier=ModelTier.LOW,
        task_type=TaskType.CONVERSATION, complexity=TaskComplexity.SIMPLE,
        reasoning="t", confidence=0.9, provider="google",
    )
    out = d.to_dict()
    assert "decision_id" in out
    assert out["decision_id"] == d.decision_id


def test_decision_from_dict_round_trip():
    d = ClassificationDecision(
        model_name="x", tier=ModelTier.LOW,
        task_type=TaskType.CONVERSATION, complexity=TaskComplexity.SIMPLE,
        reasoning="t", confidence=0.9, provider="google",
    )
    raw = d.to_dict()
    d2 = ClassificationDecision.from_dict(raw)
    assert d2.decision_id == d.decision_id


# ── log_outcome / read_outcomes ─────────────────────────────────────────────

def test_log_and_read_outcome(isolated_log, monkeypatch):
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)
    log_outcome(OutcomeRecord(
        decision_id="abc123", tokens_in=10, tokens_out=20, wall_ms=50.0, success=True,
    ))
    log_outcome(OutcomeRecord(
        decision_id="def456", tokens_in=5,  tokens_out=8,  wall_ms=12.0,
        success=False, error_message="500 internal",
    ))
    rows = read_outcomes()
    assert len(rows) == 2
    by_id = {r["decision_id"]: r for r in rows}
    assert by_id["abc123"]["tokens_in"] == 10
    assert by_id["def456"]["success"] is False


def test_read_outcomes_filter_by_decision_id(isolated_log, monkeypatch):
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)
    for did in ["a", "b", "c"]:
        log_outcome(OutcomeRecord(decision_id=did, success=True))
    rows = read_outcomes(decision_ids={"b", "c"})
    assert {r["decision_id"] for r in rows} == {"b", "c"}


def test_outcome_logger_pluggable_backend(monkeypatch):
    """Setting _backend routes log_outcome through it instead of the file."""
    captured = []
    backend = MagicMock(log=lambda entry: captured.append(entry))

    from classifier.infra import outcome_logger as ol
    monkeypatch.setattr(ol, "_backend", backend)

    log_outcome(OutcomeRecord(decision_id="xyz", tokens_in=1))
    assert len(captured) == 1
    assert captured[0]["decision_id"] == "xyz"


def test_router_outcome_logger_constructor_arg(monkeypatch):
    """Router(outcome_logger=...) wires the global backend."""
    captured = []
    backend = MagicMock(log=lambda entry: captured.append(entry))

    Router(outcome_logger=backend, layer2_enabled=False, layer3_enabled=False)

    log_outcome(OutcomeRecord(decision_id="r-test", tokens_in=42))
    assert len(captured) == 1
    assert captured[0]["decision_id"] == "r-test"


# ── Router.report_outcome ───────────────────────────────────────────────────

def test_router_report_outcome(isolated_log, monkeypatch):
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)
    router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
    decision = router.classify("Hello, how are you?")
    router.report_outcome(
        decision.decision_id,
        tokens_in=8, tokens_out=15, wall_ms=180.0, success=True,
        user_feedback="up",
    )

    rows = read_outcomes(decision_ids={decision.decision_id})
    assert len(rows) == 1
    assert rows[0]["tokens_in"] == 8
    assert rows[0]["user_feedback"] == "up"


# ── join_decisions_outcomes ─────────────────────────────────────────────────

def test_join_decisions_outcomes_drops_orphan_decisions():
    decisions = [
        {"decision_id": "a", "model": "m1"},
        {"decision_id": "b", "model": "m2"},
        {"decision_id": "c", "model": "m3"},
    ]
    outcomes = [
        {"decision_id": "a", "tokens_in": 10},
        {"decision_id": "c", "tokens_in": 20},
    ]
    joined = join_decisions_outcomes(decisions, outcomes)
    assert len(joined) == 2
    by_id = {row["decision"]["decision_id"]: row for row in joined}
    assert "a" in by_id and "c" in by_id
    assert "b" not in by_id   # no outcome → dropped
    assert by_id["a"]["outcome"]["tokens_in"] == 10


def test_join_handles_empty_inputs():
    assert join_decisions_outcomes([], []) == []
    assert join_decisions_outcomes([{"decision_id": "x"}], []) == []
    assert join_decisions_outcomes([], [{"decision_id": "x"}]) == []


# ── Decision logger writes decision_id (the join key) ──────────────────────

def test_decision_logger_writes_decision_id(tmp_path, monkeypatch):
    """The decision log entry must include decision_id so we can later join."""
    from classifier.infra import decision_logger as dl
    monkeypatch.setattr(dl, "_LOG_FILE", tmp_path / "dec.jsonl")
    monkeypatch.setattr(dl, "_TEST_LOG", tmp_path / "dec.test.jsonl")
    monkeypatch.setattr(dl, "_backend", None)
    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)

    d = ClassificationDecision(
        model_name="m", tier=ModelTier.LOW,
        task_type=TaskType.CONVERSATION, complexity=TaskComplexity.SIMPLE,
        reasoning="r", confidence=0.9, provider="google",
    )
    dl.log_decision("hello", d, layer_used="layer1", latency_ms=1.0)

    text = (tmp_path / "dec.jsonl").read_text(encoding="utf-8")
    rec = json.loads(text.strip())
    assert rec["decision_id"] == d.decision_id


# ── LangChain auto-instrumentation reports outcomes ─────────────────────────

def test_langchain_dynamic_chat_model_reports_outcome(isolated_log, monkeypatch):
    """DynamicChatModel.invoke calls log_outcome with the decision id."""
    import sys
    import types
    sys.modules.setdefault("langchain_google_genai", types.ModuleType("langchain_google_genai"))
    sys.modules["langchain_google_genai"].ChatGoogleGenerativeAI = MagicMock(
        return_value=MagicMock(invoke=MagicMock(return_value=MagicMock(usage_metadata={"input_tokens": 12, "output_tokens": 30}))),
    )

    captured = []
    import classifier as _c
    monkeypatch.setattr(_c, "log_outcome", lambda rec: captured.append(rec))

    monkeypatch.delenv("CLASSIFIER_TEST_MODE", raising=False)
    from classifier.integrations.langchain import DynamicChatModel
    llm = DynamicChatModel(provider="google")
    llm.invoke("Hello world")

    assert len(captured) == 1
    assert captured[0].tokens_in  == 12
    assert captured[0].tokens_out == 30
    assert captured[0].success is True


# ── ADK paired callback reports outcome ─────────────────────────────────────

def test_adk_paired_callback_reports(monkeypatch):
    captured = []
    import classifier as _c
    monkeypatch.setattr(_c, "log_outcome", lambda rec: captured.append(rec))

    # Build fake ADK request shape
    fake_part = MagicMock(text="What is the capital of France?", inline_data=None, file_data=None)
    fake_content = MagicMock(role="user", parts=[fake_part])
    fake_request = MagicMock(model="gemini-2.5-flash", contents=[fake_content], tools=[])
    fake_ctx     = MagicMock(agent_name="test_agent")

    from classifier.integrations.adk import dynamic_model_selector, report_model_outcome
    dynamic_model_selector(fake_ctx, fake_request)

    fake_response = MagicMock(usage_metadata={"prompt_token_count": 6, "candidates_token_count": 18})
    report_model_outcome(fake_ctx, fake_response)

    assert len(captured) == 1
    assert captured[0].tokens_in  == 6
    assert captured[0].tokens_out == 18
