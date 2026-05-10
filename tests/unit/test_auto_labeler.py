"""Unit tests for the AutoLabeler — labeling functions, aggregation, end-to-end."""
from datetime import datetime, timedelta, timezone

import pytest

from classifier.ml.auto_labeler import (
    DEFAULT_LFS,
    AutoLabeler,
    Label,
    lf_code_keywords_strong,
    lf_high_output_ratio,
    lf_l2_ground_truth,
    lf_latency_breach,
    lf_short_short,
    lf_thumbs_down,
    lf_thumbs_up_trust_layer1,
    lf_user_escalated,
)


def _decision(**overrides):
    base = {
        "decision_id":   "d1",
        "task_preview":  "default task preview",
        "task_length":   20,
        "layer":         "layer1",
        "tier":          "low",
        "task_type":     "conversation",
        "complexity":    "simple",
        "confidence":    0.85,
        "model":         "gemini-2.5-flash",
        "provider":      "google",
        "cached":        False,
        "exploration":   False,
    }
    base.update(overrides)
    return base


def _outcome(**overrides):
    base = {
        "decision_id":          "d1",
        "tokens_in":            10,
        "tokens_out":           20,
        "tokens_estimated":     False,
        "wall_ms":              100.0,
        "success":              True,
        "user_retried":         False,
        "user_escalated_model": None,
        "user_feedback":        None,
        "edit_distance":        None,
        "error_message":        None,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


# ── Per-LF tests ────────────────────────────────────────────────────────────

def test_lf_short_short_fires_for_simple():
    lab = lf_short_short(_decision(task_length=20), _outcome(tokens_out=15, user_retried=False))
    assert lab is not None
    assert lab.complexity == "simple"


def test_lf_short_short_skips_long_prompts():
    lab = lf_short_short(_decision(task_length=200), _outcome(tokens_out=10))
    assert lab is None


def test_lf_short_short_skips_when_user_retried():
    lab = lf_short_short(_decision(task_length=20), _outcome(tokens_out=10, user_retried=True))
    assert lab is None


def test_lf_user_escalated_returns_complex():
    lab = lf_user_escalated(_decision(), _outcome(user_escalated_model="gpt-4-turbo"))
    assert lab is not None
    assert lab.task_type == "reasoning"
    assert lab.complexity == "complex"
    assert lab.confidence >= 0.9


def test_lf_user_escalated_no_signal():
    assert lf_user_escalated(_decision(), _outcome(user_escalated_model=None)) is None


def test_lf_high_output_ratio_doc_creation():
    lab = lf_high_output_ratio(_decision(), _outcome(tokens_in=20, tokens_out=200))
    assert lab is not None
    assert lab.task_type == "doc_creation"


def test_lf_high_output_ratio_ignores_estimated_counts():
    """Estimated token counts (CrewAI heuristic) should NOT trigger this LF."""
    lab = lf_high_output_ratio(
        _decision(),
        _outcome(tokens_in=20, tokens_out=200, tokens_estimated=True),
    )
    assert lab is None


def test_lf_high_output_ratio_skips_low_volume():
    # Need tokens_in >= 10 to fire
    assert lf_high_output_ratio(_decision(), _outcome(tokens_in=5, tokens_out=100)) is None


def test_lf_thumbs_down_bumps_complexity():
    lab = lf_thumbs_down(_decision(), _outcome(user_feedback="down"))
    assert lab is not None
    assert lab.complexity == "complex"


def test_lf_thumbs_down_silent_when_no_feedback():
    assert lf_thumbs_down(_decision(), _outcome(user_feedback=None)) is None


def test_lf_thumbs_up_trust_layer1_fires():
    lab = lf_thumbs_up_trust_layer1(
        _decision(layer="layer1", confidence=0.92, task_type="reasoning", complexity="standard"),
        _outcome(user_feedback="up"),
    )
    assert lab is not None
    assert lab.task_type == "reasoning"
    assert lab.complexity == "standard"


def test_lf_thumbs_up_skips_when_l1_low_conf():
    lab = lf_thumbs_up_trust_layer1(
        _decision(layer="layer1", confidence=0.6),
        _outcome(user_feedback="up"),
    )
    assert lab is None


def test_lf_thumbs_up_skips_when_not_layer1():
    lab = lf_thumbs_up_trust_layer1(
        _decision(layer="layer2", confidence=0.95),
        _outcome(user_feedback="up"),
    )
    assert lab is None


def test_lf_latency_breach_bumps_when_low_tier_slow():
    lab = lf_latency_breach(_decision(tier="low"), _outcome(wall_ms=8000))
    assert lab is not None
    assert lab.complexity == "standard"


def test_lf_latency_breach_silent_for_high_tier_slow():
    assert lf_latency_breach(_decision(tier="high"), _outcome(wall_ms=8000)) is None


def test_lf_latency_breach_silent_when_fast():
    assert lf_latency_breach(_decision(tier="low"), _outcome(wall_ms=200)) is None


def test_lf_code_keywords_strong_fires():
    lab = lf_code_keywords_strong(
        _decision(task_type="code_creation", complexity="standard", confidence=0.9),
        _outcome(success=True, user_retried=False),
    )
    assert lab is not None
    assert lab.task_type == "code_creation"
    assert lab.complexity == "standard"


def test_lf_code_keywords_skips_when_user_retried():
    lab = lf_code_keywords_strong(
        _decision(task_type="code_creation", confidence=0.9),
        _outcome(user_retried=True),
    )
    assert lab is None


def test_lf_l2_ground_truth_strongest():
    lab = lf_l2_ground_truth(
        _decision(layer="layer2", task_type="reasoning", complexity="complex", confidence=0.78),
        _outcome(success=True),
    )
    assert lab is not None
    assert lab.task_type == "reasoning"
    assert lab.complexity == "complex"
    assert lab.confidence > 0.78


def test_lf_l2_ground_truth_skips_when_user_escalated():
    lab = lf_l2_ground_truth(
        _decision(layer="layer2", confidence=0.85),
        _outcome(user_escalated_model="gpt-4-turbo"),
    )
    assert lab is None


# ── Aggregation tests ──────────────────────────────────────────────────────

def test_aggregator_majority_vote_picks_highest_score():
    al = AutoLabeler(min_confidence=0.0)   # don't filter for this test
    votes = [
        Label(task_type="reasoning", complexity="complex", confidence=0.9),
        Label(task_type="reasoning", complexity="standard", confidence=0.6),
        Label(task_type="doc_creation", complexity="complex", confidence=0.4),
    ]
    out = al._aggregate(votes)
    assert out["task_type"] == "reasoning"      # 0.9 + 0.6 = 1.5 vs 0.4
    assert out["complexity"] == "complex"        # 0.9 + 0.4 = 1.3 vs 0.6


def test_aggregator_drops_below_min_confidence():
    al = AutoLabeler(min_confidence=0.95)
    votes = [
        Label(task_type="reasoning",    complexity="standard", confidence=0.6),
        Label(task_type="doc_creation", complexity="complex",  confidence=0.5),
    ]
    out = al._aggregate(votes)
    assert out is None   # close split → low share → below 0.95


def test_aggregator_returns_none_when_no_dimension_voted():
    al = AutoLabeler(min_confidence=0.0)
    votes = [Label(confidence=0.9, reason="vote with no fields")]
    assert al._aggregate(votes) is None


def test_label_one_no_lf_fires_returns_none():
    al = AutoLabeler(lfs=[lambda d, o: None], min_confidence=0.0)
    assert al.label_one(_decision(), _outcome()) is None
    assert al.stats["no_lf_fired"] == 1


def test_label_one_below_confidence_returns_none():
    al = AutoLabeler(min_confidence=0.99)
    # Conflicting weak votes → shares ~0.5 → below 0.99
    al.lfs = [
        lambda d, o: Label(task_type="reasoning",    complexity="simple",   confidence=0.5),
        lambda d, o: Label(task_type="doc_creation", complexity="standard", confidence=0.5),
    ]
    assert al.label_one(_decision(), _outcome()) is None
    assert al.stats["below_confidence"] == 1


# ── End-to-end run() tests ─────────────────────────────────────────────────

def test_run_with_supplied_lists():
    """End-to-end with hand-built decision + outcome lists (no log files).

    Each row picks up multiple LFs so the aggregated confidence clears the
    threshold (single-LF rows have share=1.0 on the dimension they vote, but
    only 0.5 average if the other dimension was empty).
    """
    decisions = [
        # esc-1: lf_user_escalated → both dims voted (reasoning, complex)
        _decision(decision_id="esc-1", layer="layer1"),
        # code-1: lf_code_keywords_strong → both dims voted (code_creation, standard)
        _decision(decision_id="code-1", task_type="code_creation",
                  complexity="standard", confidence=0.9, layer="layer1"),
        # l2-1: lf_l2_ground_truth → both dims voted
        _decision(decision_id="l2-1", layer="layer2", task_type="reasoning",
                  complexity="complex", confidence=0.8),
    ]
    outcomes = [
        _outcome(decision_id="esc-1", user_escalated_model="gpt-4-turbo"),
        _outcome(decision_id="code-1"),
        _outcome(decision_id="l2-1"),
    ]
    labeler = AutoLabeler(min_confidence=0.5)
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 3
    by_id = {r["_decision_id"]: r for r in rows}
    assert by_id["esc-1"]["task_type"]   == "reasoning"
    assert by_id["esc-1"]["complexity"]  == "complex"
    assert by_id["code-1"]["task_type"]  == "code_creation"
    assert by_id["l2-1"]["task_type"]    == "reasoning"


def test_run_skips_cache_hits_by_default():
    decisions = [_decision(decision_id="d-cache", cached=True)]
    outcomes  = [_outcome(decision_id="d-cache", user_escalated_model="gpt-4-turbo")]
    labeler = AutoLabeler()
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 0
    assert labeler.stats["skipped_cached"] == 1


def test_run_includes_cache_hits_when_opted_in():
    decisions = [_decision(decision_id="d-cache", cached=True)]
    outcomes  = [_outcome(decision_id="d-cache", user_escalated_model="gpt-4-turbo")]
    labeler = AutoLabeler(skip_cached=False)
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 1


def test_run_skips_exploration_by_default():
    decisions = [_decision(decision_id="d-explore", exploration=True)]
    outcomes  = [_outcome(decision_id="d-explore", user_escalated_model="gpt-4-turbo")]
    labeler = AutoLabeler()
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 0
    assert labeler.stats["skipped_exploration"] == 1


def test_run_skips_failed_outcomes_by_default():
    decisions = [_decision(decision_id="d-fail")]
    outcomes  = [_outcome(decision_id="d-fail", success=False, user_escalated_model="gpt-4-turbo")]
    labeler = AutoLabeler()
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 0
    assert labeler.stats["skipped_failed"] == 1


def test_run_drops_orphaned_decisions():
    decisions = [_decision(decision_id="d-orphan")]
    outcomes  = []   # no outcome reported
    labeler = AutoLabeler()
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 0
    assert labeler.stats["joined"] == 0


def test_run_output_schema_matches_train_head():
    """Output must include task / task_type / complexity for train_head consumption."""
    decisions = [_decision(decision_id="d1", task_preview="What is the capital of France?")]
    outcomes  = [_outcome(decision_id="d1", user_escalated_model="gpt-4-turbo")]
    labeler = AutoLabeler()
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 1
    row = rows[0]
    assert "task" in row
    assert "task_type" in row
    assert "complexity" in row
    assert "_label_confidence" in row
    assert "_decision_id" in row
    assert row["task"] == "What is the capital of France?"


def test_run_skips_rows_with_empty_task_preview():
    decisions = [_decision(decision_id="d1", task_preview="")]
    outcomes  = [_outcome(decision_id="d1", user_escalated_model="gpt-4-turbo")]
    labeler = AutoLabeler()
    rows = labeler.run(decisions=decisions, outcomes=outcomes)
    assert len(rows) == 0
    assert labeler.stats["skipped_empty_preview"] >= 1


def test_default_lfs_are_eight():
    """Doc 05 commits to 8 default labeling functions."""
    assert len(DEFAULT_LFS) == 8


def test_run_stats_reports_full_funnel():
    decisions = [
        _decision(decision_id="d1"),
        _decision(decision_id="d2", cached=True),
        _decision(decision_id="d3", exploration=True),
        _decision(decision_id="d-orphan"),   # no outcome
    ]
    outcomes = [
        _outcome(decision_id="d1", user_escalated_model="gpt-4-turbo"),
        _outcome(decision_id="d2", user_escalated_model="gpt-4-turbo"),
        _outcome(decision_id="d3", user_escalated_model="gpt-4-turbo"),
        _outcome(decision_id="d-extra"),     # no decision
    ]
    labeler = AutoLabeler()
    labeler.run(decisions=decisions, outcomes=outcomes)
    s = labeler.stats
    assert s["decisions_read"] == 4
    assert s["outcomes_read"] == 4
    assert s["joined"] == 3
    assert s["skipped_cached"] == 1
    assert s["skipped_exploration"] == 1
    assert s["labeled"] == 1


# ── CLI smoke test ─────────────────────────────────────────────────────────

def test_cli_relabel_smoke(tmp_path, monkeypatch):
    """`dmr relabel` end-to-end with synthetic logs."""
    import json

    from classifier.cli import main as cli_main
    from classifier.infra import decision_logger as dl
    from classifier.infra import outcome_logger as ol

    monkeypatch.setattr(ol, "_TEST_LOG", tmp_path / "out.test.jsonl")
    monkeypatch.setattr(ol, "_LOG_FILE", tmp_path / "out.jsonl")
    monkeypatch.setattr(dl, "_TEST_LOG", tmp_path / "dec.test.jsonl")
    monkeypatch.setattr(dl, "_LOG_FILE", tmp_path / "dec.jsonl")
    monkeypatch.setenv("CLASSIFIER_TEST_MODE", "1")

    # Seed one decision + one outcome that the user_escalated LF will pick up
    (tmp_path / "dec.test.jsonl").write_text(
        json.dumps(_decision(decision_id="cli-1",
                             task_preview="What is the capital of France?")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "out.test.jsonl").write_text(
        json.dumps(_outcome(decision_id="cli-1",
                            user_escalated_model="gpt-4-turbo")) + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "labeled.jsonl"
    rc = cli_main(["relabel", "--out", str(out_path), "--min-confidence", "0.5"])
    assert rc == 0
    assert out_path.exists()
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["task_type"] == "reasoning"
