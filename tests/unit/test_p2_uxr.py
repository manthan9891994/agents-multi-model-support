"""Tests for P2 UX features: layer3 auto mode, dmr keywords, keyword miner, dmr config."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from classifier.cli import main as cli_main


@pytest.fixture
def isolated_keywords(tmp_path, monkeypatch):
    """Point ~/.dmr/keywords at a tmp dir for the test."""
    monkeypatch.setenv("DMR_KEYWORDS_DIR", str(tmp_path))
    yield tmp_path


# ── Router(layer3_enabled='auto') ────────────────────────────────────────────


def test_layer3_auto_resolves_to_false_when_no_model():
    """If no L3 model file exists, 'auto' silently disables L3."""
    from classifier import Router
    from classifier.router import _l3_model_available

    with patch.object(_l3_model_available.__globals__["_l3_model_available"], "__call__", return_value=False):
        # Direct attribute check is more reliable than mocking the function
        pass

    with patch("classifier.router._l3_model_available", return_value=False):
        r = Router(layer3_enabled="auto")
        assert r.layer3_enabled is False


def test_layer3_auto_resolves_to_true_when_model_present():
    """If an L3 model file exists, 'auto' enables L3."""
    from classifier import Router

    with patch("classifier.router._l3_model_available", return_value=True):
        r = Router(layer3_enabled="auto")
        assert r.layer3_enabled is True


def test_layer3_explicit_bool_overrides_auto_logic():
    from classifier import Router

    with patch("classifier.router._l3_model_available", return_value=False):
        r = Router(layer3_enabled=True)
        assert r.layer3_enabled is True
    with patch("classifier.router._l3_model_available", return_value=True):
        r = Router(layer3_enabled=False)
        assert r.layer3_enabled is False


# ── dmr keywords add/list/remove ─────────────────────────────────────────────


def test_keywords_add_persists_to_yaml(isolated_keywords):
    rc = cli_main(
        [
            "keywords",
            "add",
            "--domain",
            "legal_test",
            "--type",
            "reasoning",
            "--keywords",
            "tort,liable,precedent",
        ]
    )
    assert rc == 0
    yaml_path = isolated_keywords / "legal_test.yaml"
    assert yaml_path.exists()

    import yaml as _yaml

    data = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["name"] == "legal_test"
    assert "tort" in data["task_keywords"]["reasoning"]["primary"]
    assert "liable" in data["task_keywords"]["reasoning"]["primary"]


def test_keywords_add_dedupes_existing(isolated_keywords):
    cli_main(["keywords", "add", "--domain", "x", "--type", "reasoning", "--keywords", "alpha,beta"])
    cli_main(["keywords", "add", "--domain", "x", "--type", "reasoning", "--keywords", "beta,gamma"])
    import yaml as _yaml

    data = _yaml.safe_load((isolated_keywords / "x.yaml").read_text(encoding="utf-8"))
    kws = data["task_keywords"]["reasoning"]["primary"]
    # No duplicate of "beta"
    assert kws.count("beta") == 1
    assert sorted(kws) == ["alpha", "beta", "gamma"]


def test_keywords_add_rejects_unknown_task_type(isolated_keywords):
    rc = cli_main(["keywords", "add", "--domain", "x", "--type", "not_a_real_type", "--keywords", "foo"])
    assert rc == 2


def test_keywords_remove_strips_keyword(isolated_keywords):
    cli_main(["keywords", "add", "--domain", "x", "--type", "reasoning", "--keywords", "alpha,beta"])
    rc = cli_main(["keywords", "remove", "--domain", "x", "--keyword", "alpha"])
    assert rc == 0
    import yaml as _yaml

    data = _yaml.safe_load((isolated_keywords / "x.yaml").read_text(encoding="utf-8"))
    assert "alpha" not in data["task_keywords"]["reasoning"]["primary"]
    assert "beta" in data["task_keywords"]["reasoning"]["primary"]


def test_keywords_remove_returns_1_when_not_found(isolated_keywords):
    cli_main(["keywords", "add", "--domain", "x", "--type", "reasoning", "--keywords", "alpha"])
    rc = cli_main(["keywords", "remove", "--domain", "x", "--keyword", "nonexistent"])
    assert rc == 1


def test_keywords_list_with_no_packs(isolated_keywords, capsys):
    rc = cli_main(["keywords", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No user keyword packs yet" in out


def test_keywords_list_shows_added_pack(isolated_keywords, capsys):
    cli_main(["keywords", "add", "--domain", "x", "--type", "reasoning", "--keywords", "uniqkw"])
    capsys.readouterr()  # flush
    rc = cli_main(["keywords", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "uniqkw" in out
    assert "[x]" in out


# ── auto-load user packs into Router() ──────────────────────────────────────


def test_router_autoloads_user_packs_from_dmr_dir(isolated_keywords):
    """A pack persisted via `dmr keywords add` should affect the next Router()."""
    cli_main(
        [
            "keywords",
            "add",
            "--domain",
            "auto_load_test",
            "--type",
            "reasoning",
            "--keywords",
            "xyzzy_marker_word",
        ]
    )
    # Re-import / construct Router after the pack file exists
    from classifier import Router
    from classifier.layers.layer1.keyword_pack import clear_registered

    clear_registered()  # clear in-memory state from prior tests
    Router()  # construction triggers auto_load_user_packs

    from classifier.core.types import TaskType
    from classifier.layers.layer1.constants import _TASK_KEYWORDS

    found = False
    for groups in (_TASK_KEYWORDS.get(TaskType.REASONING) or {}).values():
        if "xyzzy_marker_word" in (groups or []):
            found = True
            break
    assert found, "auto-loaded user pack should have injected the keyword into L1"


# ── keyword_miner.suggest_keywords ───────────────────────────────────────────


def test_keyword_miner_returns_distinctive_ngrams(monkeypatch):
    """Tasks dominated by a unique word should produce that word as a top suggestion."""
    fake_decisions = (
        # 30 reasoning tasks talking about "indemnification"
        [{"task_type": "reasoning", "task_preview": "draft indemnification clause for vendor"}] * 30
        # 30 doc_creation tasks talking about "summarize"
        + [{"task_type": "doc_creation", "task_preview": "summarize the meeting transcript briefly"}] * 30
    )
    monkeypatch.setattr(
        "classifier.infra.decision_logger.read_decisions",
        lambda **_: fake_decisions,
    )

    from classifier.ml.keyword_miner import suggest_keywords

    out = suggest_keywords(top_per_type=10, min_occurrences=3)
    assert "reasoning" in out
    assert "doc_creation" in out

    reasoning_kws = [kw for kw, _, _ in out["reasoning"]]
    assert "indemnification" in reasoning_kws
    assert "summarize" not in reasoning_kws  # belongs to the other class


def test_keyword_miner_skips_words_already_in_pack(monkeypatch):
    """Suggestions must not duplicate existing built-in keywords."""
    fake_decisions = [
        {"task_type": "reasoning", "task_preview": "analyze and reason about this problem"}
    ] * 20
    monkeypatch.setattr(
        "classifier.infra.decision_logger.read_decisions",
        lambda **_: fake_decisions,
    )

    from classifier.ml.keyword_miner import _existing_keywords, suggest_keywords

    existing = _existing_keywords()
    out = suggest_keywords(top_per_type=20, min_occurrences=3)
    suggested = {kw for items in out.values() for kw, _, _ in items}
    assert not (suggested & existing), "suggestions overlapped with existing keywords"


def test_keyword_miner_returns_empty_when_no_data(monkeypatch):
    monkeypatch.setattr(
        "classifier.infra.decision_logger.read_decisions",
        lambda **_: [],
    )
    from classifier.ml.keyword_miner import suggest_keywords

    assert suggest_keywords() == {}


# ── dmr config show ──────────────────────────────────────────────────────────


def test_dmr_config_show_runs(capsys):
    rc = cli_main(["config", "show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "default_provider" in out
    assert "registry" in out


# ── dmr train --auto error path ──────────────────────────────────────────────


def test_dmr_train_auto_with_no_data_returns_1(monkeypatch, capsys):
    """When AutoLabeler returns 0 confident rows, exit cleanly with hint."""
    from classifier.ml.auto_labeler import AutoLabeler

    # Make AutoLabeler always return empty
    monkeypatch.setattr(AutoLabeler, "run", lambda self, **_: [])

    rc = cli_main(["train", "--auto", "--since", "7d"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Need >= 50" in err or "need >= 50" in err.lower()


def test_dmr_train_without_data_or_auto_errors_clearly(capsys):
    rc = cli_main(["train"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--data" in err or "--auto" in err
