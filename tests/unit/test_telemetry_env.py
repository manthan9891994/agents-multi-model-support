"""Tests for DMR_TELEMETRY env-var gating in decision_logger and outcome_logger."""
import json
import logging

import pytest


@pytest.fixture(autouse=True)
def reset_loggers():
    """Ensure dmr.decisions / dmr.outcomes loggers are clean for each test."""
    for name in ("dmr.decisions", "dmr.outcomes"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = False
    yield
    for name in ("dmr.decisions", "dmr.outcomes"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class _MockBackend:
    def __init__(self):
        self.logged = []

    def log(self, entry: dict) -> None:
        self.logged.append(entry)


def _make_decision():
    from unittest.mock import MagicMock
    from classifier.core.types import ModelTier, TaskType, TaskComplexity

    d = MagicMock()
    d.decision_id = "test001"
    d.model_name = "gemini-2.0-flash-lite"
    d.tier = ModelTier.LOW
    d.task_type = TaskType.CONVERSATION
    d.complexity = TaskComplexity.SIMPLE
    d.confidence = 0.91
    d.provider = "google"
    d.compliance_flag = False
    d.disagreement = False
    d.exploration = False
    d.cached = False
    d.cached_from = ""
    return d


# ── decision_logger tests ────────────────────────────────────────────────────

def test_default_mode_emits_info_not_debug(monkeypatch):
    """No DMR_TELEMETRY → INFO line, no backend called."""
    import classifier.infra.decision_logger as dl

    monkeypatch.setattr(dl, "_TELEMETRY_ENABLED", False)
    monkeypatch.setattr(dl, "_backend", None)

    cap = _LogCapture()
    cap.setLevel(logging.DEBUG)
    dl._dmr_logger.addHandler(cap)
    dl._dmr_logger.setLevel(logging.DEBUG)

    dl.log_decision("explain recursion", _make_decision(), "layer1", 2.1)

    info_recs = [r for r in cap.records if r.levelno == logging.INFO]
    debug_recs = [r for r in cap.records if r.levelno == logging.DEBUG]
    assert len(info_recs) == 1
    assert len(debug_recs) == 0
    assert "DMR decision:" in info_recs[0].getMessage()


def test_default_mode_no_backend_called(monkeypatch, tmp_path):
    import classifier.infra.decision_logger as dl

    monkeypatch.setattr(dl, "_TELEMETRY_ENABLED", False)
    backend = _MockBackend()
    monkeypatch.setattr(dl, "_backend", None)

    dl.log_decision("hello", _make_decision(), "layer1", 1.5)
    assert len(backend.logged) == 0


def test_telemetry_enabled_emits_debug_only(monkeypatch, tmp_path):
    """DMR_TELEMETRY=1 with no backend → DEBUG log ONLY. No file ever written."""
    import classifier.infra.decision_logger as dl

    monkeypatch.setattr(dl, "_TELEMETRY_ENABLED", True)
    monkeypatch.setattr(dl, "_backend", None)
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(dl, "_LOG_FILE", log_file)
    monkeypatch.setattr(dl, "_TEST_LOG", log_file)

    cap = _LogCapture()
    cap.setLevel(logging.DEBUG)
    dl._dmr_logger.addHandler(cap)
    dl._dmr_logger.setLevel(logging.DEBUG)

    dl.log_decision("write a parser", _make_decision(), "layer3", 15.0)

    debug_recs = [r for r in cap.records if r.levelno == logging.DEBUG]
    assert len(debug_recs) == 1
    msg = debug_recs[0].getMessage()
    assert "DMR telemetry:" in msg
    payload = json.loads(msg.split("DMR telemetry: ", 1)[1])
    assert payload["tier"] == "low"
    assert payload["decision_id"] == "test001"
    assert "router_version" in payload
    # CRITICAL: package must NEVER write files automatically
    assert not log_file.exists()


def test_telemetry_with_backend_calls_backend(monkeypatch, tmp_path):
    """DMR_TELEMETRY=1 + backend → backend.log() called AND DEBUG line emitted."""
    import classifier.infra.decision_logger as dl

    monkeypatch.setattr(dl, "_TELEMETRY_ENABLED", True)
    backend = _MockBackend()
    monkeypatch.setattr(dl, "_backend", backend)
    log_file = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(dl, "_LOG_FILE", log_file)

    dl.log_decision("summarize a PDF", _make_decision(), "layer1", 1.8)

    assert len(backend.logged) == 1
    assert backend.logged[0]["decision_id"] == "test001"
    assert "router_version" in backend.logged[0]
    assert not log_file.exists()  # never auto-written


def test_router_version_present_in_entry(monkeypatch, tmp_path):
    import classifier.infra.decision_logger as dl
    import classifier

    monkeypatch.setattr(dl, "_TELEMETRY_ENABLED", True)
    backend = _MockBackend()
    monkeypatch.setattr(dl, "_backend", backend)

    dl.log_decision("test task", _make_decision(), "layer1", 1.0)
    assert backend.logged[0]["router_version"] == classifier.__version__


# ── outcome_logger tests ─────────────────────────────────────────────────────

def test_outcome_default_emits_info(monkeypatch):
    import classifier.infra.outcome_logger as ol
    from classifier.infra.outcome_logger import OutcomeRecord

    monkeypatch.setattr(ol, "_TELEMETRY_ENABLED", False)
    monkeypatch.setattr(ol, "_backend", None)

    cap = _LogCapture()
    cap.setLevel(logging.DEBUG)
    ol._dmr_logger.addHandler(cap)
    ol._dmr_logger.setLevel(logging.DEBUG)

    ol.log_outcome(OutcomeRecord(decision_id="out001", tokens_in=10, tokens_out=20, wall_ms=400, success=True))

    info_recs = [r for r in cap.records if r.levelno == logging.INFO]
    debug_recs = [r for r in cap.records if r.levelno == logging.DEBUG]
    assert len(info_recs) == 1
    assert len(debug_recs) == 0
    assert "DMR outcome:" in info_recs[0].getMessage()


def test_outcome_telemetry_calls_backend(monkeypatch):
    import classifier.infra.outcome_logger as ol
    from classifier.infra.outcome_logger import OutcomeRecord

    monkeypatch.setattr(ol, "_TELEMETRY_ENABLED", True)
    backend = _MockBackend()
    monkeypatch.setattr(ol, "_backend", backend)

    ol.log_outcome(OutcomeRecord(decision_id="out002", tokens_in=50, tokens_out=30, cost_usd=0.0002))

    assert len(backend.logged) == 1
    assert backend.logged[0]["decision_id"] == "out002"
    assert "router_version" in backend.logged[0]
    assert backend.logged[0]["tokens_in"] == 50
