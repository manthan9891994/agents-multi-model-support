"""Loads model/cost/capability data from YAML into the runtime registries.

The package ships zero hardcoded model names or prices in Python — everything
lives in `classifier/data/registry/*.yaml` and is loaded at import time.
Users can override entirely via:

    1. Environment variable:    DMR_REGISTRY=/path/to/my-models.yaml
    2. Programmatically:        Router.load_registry("my-models.yaml")
    3. Constructor argument:    Router(registry="my-models.yaml")
    4. Remote URL:              Router.load_registry("https://example.com/r.yaml")
    5. Disable bundled data:    DMR_NO_DEFAULT_REGISTRY=1   (start empty)

Schema (see classifier/data/registry/default.yaml for full example):

    version: "YYYY.MM.DD"
    providers:
      <name>:
        api_key_env: ENV_VAR_NAME
        tiers:
          low:    model-name
          medium: model-name
          high:   model-name
    models:
      <name>:
        cost: { input_per_1m: 0.0, output_per_1m: 0.0 }
        capabilities:
          context_window: 128000
          supports_vision: bool
          supports_function_calling: bool
          supports_streaming: bool
          supports_json_mode: bool
          region: str | null
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BUNDLED_DEFAULT = Path(__file__).parent.parent / "data" / "registry" / "default.yaml"
_BUNDLED_EMPTY   = Path(__file__).parent.parent / "data" / "registry" / "empty.yaml"


def _load_yaml(source: str | Path) -> dict:
    """Load YAML from a path, URL, or string. Returns the parsed dict."""
    import yaml
    text: str
    src = str(source)

    if src.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(src, timeout=10) as r:
            text = r.read().decode("utf-8")
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {source}")
        text = path.read_text(encoding="utf-8")

    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Registry YAML must be a dict, got {type(parsed).__name__}")
    return parsed


def apply_registry(data: dict) -> dict:
    """Apply a parsed registry dict to the runtime tables.

    Returns metadata about what was loaded:
        {"version": str, "providers": int, "models": int}

    Does NOT clear existing entries — call clear_registry() first for a fresh slate.
    """
    from classifier.core.registry import (
        MODEL_CAPABILITIES,
        register_provider,
    )
    from classifier.infra.cost_tracker import register_model_cost

    providers = data.get("providers") or {}
    models    = data.get("models")    or {}

    # Apply providers
    for prov_name, prov_data in providers.items():
        tiers = prov_data.get("tiers") or {}
        if not tiers:
            continue
        register_provider(prov_name, dict(tiers))

    # Apply costs + capabilities
    for model_name, model_data in models.items():
        cost = (model_data or {}).get("cost") or {}
        if cost:
            register_model_cost(
                model_name,
                input_per_1m=float(cost.get("input_per_1m", 0.0)),
                output_per_1m=float(cost.get("output_per_1m", 0.0)),
            )
        caps = (model_data or {}).get("capabilities") or {}
        if caps:
            MODEL_CAPABILITIES.setdefault(model_name, {}).update(caps)

    return {
        "version":   data.get("version", "unknown"),
        "providers": len(providers),
        "models":    len(models),
    }


def clear_registry() -> None:
    """Wipe all registered providers, models, costs, and capabilities."""
    from classifier.core.registry import MODEL_CAPABILITIES, MODEL_REGISTRY
    from classifier.infra.cost_tracker import COST_TABLE
    MODEL_REGISTRY.clear()
    MODEL_CAPABILITIES.clear()
    COST_TABLE.clear()


def load_registry(source: str | Path = "default") -> dict:
    """Load a registry into the global tables.

    Args:
        source: One of:
            - "default"    : bundled default.yaml
            - "empty"      : bundled empty.yaml
            - path/to/file : local YAML
            - https://...  : remote URL
            - dict         : pre-parsed registry data

    Returns metadata about what was loaded. Does NOT clear existing tables —
    new entries merge in (last-write-wins).
    """
    if isinstance(source, dict):
        return apply_registry(source)

    src = str(source)
    if src == "default":
        path = _BUNDLED_DEFAULT
    elif src == "empty":
        path = _BUNDLED_EMPTY
    else:
        path = source

    data = _load_yaml(path)
    meta = apply_registry(data)
    logger.info("registry: loaded %s — %d providers, %d models, version=%s",
                path, meta["providers"], meta["models"], meta["version"])
    return meta


def export_registry() -> dict:
    """Snapshot the current runtime registries as a dict in registry-YAML shape."""
    from classifier.core.registry import MODEL_CAPABILITIES, MODEL_REGISTRY
    from classifier.infra.cost_tracker import COST_TABLE

    providers_out: dict[str, Any] = {}
    for prov_name, tier_map in MODEL_REGISTRY.items():
        providers_out[prov_name] = {
            "tiers": {
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in tier_map.items()
            },
        }

    models_out: dict[str, Any] = {}
    all_models = set(COST_TABLE.keys()) | set(MODEL_CAPABILITIES.keys())
    for m in sorted(all_models):
        entry: dict[str, Any] = {}
        if m in COST_TABLE:
            c = COST_TABLE[m]
            # Re-emit in registry-YAML schema (input_per_1m / output_per_1m)
            entry["cost"] = {
                "input_per_1m":  c.get("input", 0.0),
                "output_per_1m": c.get("output", 0.0),
            }
        if m in MODEL_CAPABILITIES:
            entry["capabilities"] = dict(MODEL_CAPABILITIES[m])
        models_out[m] = entry

    return {
        "version":   "exported",
        "providers": providers_out,
        "models":    models_out,
    }


def export_to_yaml(path: str | Path) -> None:
    """Save the current runtime registry as YAML."""
    import yaml
    data = export_registry()
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _auto_load_at_import() -> None:
    """Loaded once at package import. Honors env vars."""
    if os.environ.get("DMR_NO_DEFAULT_REGISTRY", "").lower() in ("1", "true", "yes"):
        logger.info("registry: DMR_NO_DEFAULT_REGISTRY=1 — starting with empty registry")
        return

    custom = os.environ.get("DMR_REGISTRY")
    if custom:
        try:
            load_registry(custom)
            return
        except Exception as exc:
            logger.warning("registry: DMR_REGISTRY=%r failed (%s) — falling back to bundled default",
                           custom, exc)

    # Fall back to the bundled default
    try:
        load_registry("default")
    except Exception as exc:
        logger.warning("registry: bundled default load failed: %s — runtime will be empty", exc)
