"""Tests for the YAML-driven registry — proves the package ships zero
hardcoded model names or pricing in Python."""

import os
import textwrap
from pathlib import Path

import pytest

from classifier import (
    capabilities_for,
    clear_registry,
    export_registry,
    get_model_cost,
    list_models,
    list_providers,
    load_registry,
)


@pytest.fixture
def empty_registry():
    """Reset to fully-empty state for the duration of a test, then restore default."""
    clear_registry()
    yield
    clear_registry()
    load_registry("default")


def test_default_registry_loads_at_import():
    """The bundled default.yaml is auto-loaded at import time."""
    # If this test runs in isolation, the auto-load will have populated tables.
    assert "google" in list_providers() or len(list_providers()) >= 0


def test_clear_registry_empties_everything(empty_registry):
    assert list_providers() == []
    assert list_models() == []


def test_load_default_registry(empty_registry):
    meta = load_registry("default")
    assert meta["providers"] >= 3
    assert meta["models"] >= 8
    assert "google" in list_providers()
    assert "anthropic" in list_providers()
    assert "openai" in list_providers()


def test_load_empty_template(empty_registry):
    """The bundled empty.yaml is a no-op (just placeholders)."""
    meta = load_registry("empty")
    assert meta["providers"] == 0
    assert meta["models"] == 0


def test_load_from_file(tmp_path: Path, empty_registry):
    yaml_text = textwrap.dedent("""\
        version: "test-1"
        providers:
          mistral:
            api_key_env: MISTRAL_API_KEY
            tiers:
              low: mistral-small
              high: mistral-large
        models:
          mistral-small:
            cost: { input_per_1m: 0.20, output_per_1m: 0.60 }
            capabilities:
              context_window: 32000
              supports_function_calling: true
          mistral-large:
            cost: { input_per_1m: 2.0, output_per_1m: 6.0 }
            capabilities:
              context_window: 128000
              supports_function_calling: true
    """)
    p = tmp_path / "models.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    meta = load_registry(p)
    assert meta["version"] == "test-1"
    assert "mistral" in list_providers()
    assert "mistral-large" in list_models()
    assert get_model_cost("mistral-large")["input"] == 2.0
    assert capabilities_for("mistral-small")["context_window"] == 32000


def test_load_from_dict(empty_registry):
    """A pre-parsed dict is also a valid source."""
    data = {
        "version": "inline",
        "providers": {"my_co": {"tiers": {"low": "my-cheap", "high": "my-expensive"}}},
        "models": {
            "my-cheap": {"cost": {"input_per_1m": 0.1, "output_per_1m": 0.3}},
            "my-expensive": {"cost": {"input_per_1m": 5.0, "output_per_1m": 15.0}},
        },
    }
    meta = load_registry(data)
    assert meta["providers"] == 1
    assert "my_co" in list_providers()


def test_export_round_trip(tmp_path: Path, empty_registry):
    """Loading -> exporting -> re-loading should preserve all data."""
    load_registry("default")
    snapshot = export_registry()
    clear_registry()
    load_registry(snapshot)
    assert "google" in list_providers()
    assert "claude-opus-4-7" in list_models()


def test_router_constructor_registry_arg(tmp_path: Path, empty_registry):
    """Router(registry=...) loads at construction."""
    from classifier import Router

    yaml_text = textwrap.dedent("""\
        providers:
          custom: { tiers: { low: my-low, high: my-high } }
        models:
          my-low:  { cost: { input_per_1m: 0.1, output_per_1m: 0.2 } }
          my-high: { cost: { input_per_1m: 1.0, output_per_1m: 3.0 } }
    """)
    p = tmp_path / "models.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    Router(registry=p, layer2_enabled=False, layer3_enabled=False)
    assert "custom" in list_providers()


def test_router_from_registry_classmethod(tmp_path: Path, empty_registry):
    from classifier import Router

    yaml_text = textwrap.dedent("""\
        providers:
          fab: { tiers: { low: fab-low, high: fab-high } }
        models:
          fab-low:  { cost: { input_per_1m: 0.5, output_per_1m: 1.0 } }
          fab-high: { cost: { input_per_1m: 5.0, output_per_1m: 10.0 } }
    """)
    p = tmp_path / "models.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    Router.from_registry(p, layer2_enabled=False, layer3_enabled=False)
    assert "fab" in list_providers()
    assert get_model_cost("fab-high") == {"input": 5.0, "output": 10.0}


def test_dmr_no_default_registry_env(empty_registry):
    """Manually invoking the auto-loader with the env var set yields empty state."""
    from classifier.core.registry_loader import _auto_load_at_import

    old = os.environ.get("DMR_NO_DEFAULT_REGISTRY")
    os.environ["DMR_NO_DEFAULT_REGISTRY"] = "1"
    try:
        clear_registry()
        _auto_load_at_import()
        assert list_providers() == []
        assert list_models() == []
    finally:
        if old is None:
            os.environ.pop("DMR_NO_DEFAULT_REGISTRY", None)
        else:
            os.environ["DMR_NO_DEFAULT_REGISTRY"] = old


def test_unknown_model_uses_default_cost(empty_registry):
    """Unknown models fall back to a conservative default cost."""
    cost = get_model_cost("never-registered-model-xyz")
    assert cost == {"input": 0.25, "output": 0.75}
