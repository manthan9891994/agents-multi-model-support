"""Domain presets — pre-built bundles of keyword packs, PII patterns, and config.

Usage:
    from classifier import Router
    router = Router.from_preset("healthcare")

Available presets:
    healthcare — HIPAA PII patterns, clinical vocab keywords (fully populated)
    legal      — skeleton — extend with your firm's vocab
    fintech    — skeleton — extend with your domain vocab
"""
from collections.abc import Callable

_REGISTRY: dict[str, Callable[[], dict]] = {}


def register(name: str, factory: Callable[[], dict]) -> None:
    """Register a preset factory by name."""
    _REGISTRY[name] = factory


def load_preset(name: str) -> dict:
    """Load a preset config dict by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown preset: {name!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def available() -> list[str]:
    """Return list of registered preset names."""
    return sorted(_REGISTRY.keys())


# Auto-register built-in presets on import
from classifier.presets import fintech, healthcare, legal  # noqa: E402, F401

register("healthcare", healthcare.config)
register("legal",      legal.config)
register("fintech",    fintech.config)
