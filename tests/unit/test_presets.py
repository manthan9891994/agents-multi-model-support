"""Unit tests for built-in domain presets."""
import pytest

from classifier.presets import available, load_preset


def test_three_built_in_presets():
    names = available()
    assert "healthcare" in names
    assert "legal" in names
    assert "fintech" in names


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        load_preset("nonexistent")


@pytest.mark.parametrize("name", ["healthcare", "legal", "fintech"])
def test_each_preset_returns_valid_config(name):
    cfg = load_preset(name)
    assert isinstance(cfg, dict)
    # Each preset must provide at least one of these
    assert any(k in cfg for k in ("extra_keyword_packs", "extra_pii_patterns", "providers"))


def test_healthcare_preset_has_pii_patterns():
    from classifier.presets.healthcare import config
    cfg = config()
    assert cfg.get("extra_pii_patterns")
    assert len(cfg["extra_pii_patterns"]) >= 1


def test_healthcare_preset_has_keyword_pack():
    from classifier.presets.healthcare import config
    cfg = config()
    assert cfg.get("extra_keyword_packs")
    assert cfg["extra_keyword_packs"][0].name == "healthcare"


def test_fintech_preset_has_card_pattern():
    """Fintech preset should redact credit card numbers."""
    from classifier.presets.fintech import config
    cfg = config()
    tokens = [tok for _, tok in cfg["extra_pii_patterns"]]
    assert "[CARD]" in tokens or "[ACCT]" in tokens
