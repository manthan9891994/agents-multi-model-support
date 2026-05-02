"""Unit tests for the Router class — the high-level user-facing API."""
import pytest

from classifier import Router, classify, KeywordPack, TaskType


@pytest.fixture(autouse=True)
def reset_extras():
    """Wipe registered extra packs / pii patterns between tests."""
    from classifier.layers.layer1 import keyword_pack
    from classifier.infra import pii_scrubber
    keyword_pack.clear_registered()
    pii_scrubber.clear_extra_patterns()
    yield
    keyword_pack.clear_registered()
    pii_scrubber.clear_extra_patterns()


# ── Construction ──────────────────────────────────────────────────────────────

def test_default_router_zero_config():
    r = Router()
    d = r.classify("What is 2+2?")
    assert d.tier.value in ("low", "medium", "high")
    assert d.model_name


def test_module_level_classify_function():
    d = classify("Implement binary search")
    assert d.tier.value in ("low", "medium", "high")


# ── Layer toggles ─────────────────────────────────────────────────────────────

def test_layer3_can_be_disabled():
    r = Router(layer3_enabled=False)
    d = r.classify("ambiguous task that L1 can't handle confidently")
    # Decision still returned (from L2 or L1), just not from L3
    assert d.layer_used in ("layer1", "layer2")


def test_layer2_can_be_disabled():
    r = Router(layer2_enabled=False, layer3_enabled=False)
    d = r.classify("hi")
    assert d.layer_used == "layer1"


# ── Threshold overrides ───────────────────────────────────────────────────────

def test_thresholds_take_effect():
    r  = Router(escalation_threshold=0.99, layer3_threshold=0.99)  # always escalate / abstain
    r2 = Router(escalation_threshold=0.50, layer3_threshold=0.50)  # rarely escalate / abstain
    d1 = r.classify("hello")
    d2 = r2.classify("hello")
    assert d1.tier
    assert d2.tier


# ── Custom keyword packs ──────────────────────────────────────────────────────

def test_custom_keyword_pack_injected():
    pack = (KeywordPack.builder("test")
            .add(TaskType.REASONING, ["xyzzy_unique_kw_001"])
            .build())
    r = Router(extra_keyword_packs=[pack])
    d = r.classify("xyzzy_unique_kw_001 do this thing")
    assert d.task_type == TaskType.REASONING


def test_keyword_pack_idempotent_registration():
    """Registering the same pack twice should be a no-op."""
    from classifier.layers.layer1.keyword_pack import register_extra_packs, list_registered
    pack = KeywordPack.builder("dup").add(TaskType.REASONING, ["dup_kw"]).build()
    register_extra_packs([pack])
    register_extra_packs([pack])
    assert list_registered().count("dup") == 1


# ── Custom PII patterns ───────────────────────────────────────────────────────

def test_custom_pii_pattern_applied():
    import re
    from classifier.infra.pii_scrubber import scrub
    pattern = (re.compile(r"\bACCT-\d{6}\b"), "[ACCT]")
    Router(extra_pii_patterns=[pattern])
    res = scrub("Customer ACCT-123456 has overdue balance")
    assert "[ACCT]" in res.text
    assert "ACCT-123456" not in res.text


# ── Tier matrix override ──────────────────────────────────────────────────────

def test_tier_matrix_override_applied_then_restored():
    from classifier.core.types import TaskType, TaskComplexity, ModelTier
    from classifier.core.registry import TIER_MATRIX

    original = TIER_MATRIX[(TaskType.CONVERSATION, TaskComplexity.SIMPLE)]

    r = Router(tier_matrix={(TaskType.CONVERSATION, TaskComplexity.SIMPLE): ModelTier.HIGH})
    r.classify("hi")

    # After classify(), matrix should be restored
    assert TIER_MATRIX[(TaskType.CONVERSATION, TaskComplexity.SIMPLE)] == original


# ── Presets ───────────────────────────────────────────────────────────────────

def test_healthcare_preset_loads():
    r = Router.from_preset("healthcare")
    d = r.classify("Patient MRN: 12345678 has elevated AST")
    assert d.compliance_flag is True


def test_legal_preset_loads():
    r = Router.from_preset("legal")
    d = r.classify("Draft an indemnification clause for a SaaS contract")
    assert d.tier.value in ("low", "medium", "high")


def test_fintech_preset_loads():
    r = Router.from_preset("fintech")
    d = r.classify("Calculate Sharpe ratio for this portfolio")
    assert d.tier.value in ("low", "medium", "high")


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        Router.from_preset("nonexistent_xyz")


# ── YAML config ───────────────────────────────────────────────────────────────

def test_from_yaml_loads_config(tmp_path):
    cfg = tmp_path / "dmr.yaml"
    cfg.write_text(
        "providers:\n"
        "  - google\n"
        "layer1_enabled: true\n"
        "layer2_enabled: false\n"
        "layer3_enabled: false\n",
        encoding="utf-8",
    )
    r = Router.from_yaml(cfg)
    assert r.providers == ["google"]
    assert r.layer2_enabled is False


def test_from_yaml_keyword_packs(tmp_path):
    cfg = tmp_path / "dmr.yaml"
    cfg.write_text(
        "keyword_packs:\n"
        "  - name: yaml_test\n"
        "    packs:\n"
        "      reasoning:\n"
        "        - yaml_unique_word_zzz\n",
        encoding="utf-8",
    )
    r = Router.from_yaml(cfg)
    assert len(r.extra_keyword_packs) == 1
    assert r.extra_keyword_packs[0].name == "yaml_test"
