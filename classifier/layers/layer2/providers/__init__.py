"""Layer 2 provider callers — pluggable backends for the LLM classifier.

Each provider exposes:
    call(task: str, history: list[str] | None, model: str, schema: dict) -> Response
where Response has `.text` (the JSON string) and `.usage_metadata` (optional).

Providers are looked up by name from a registry. Custom providers can be
registered via `register_l2_provider(name, fn)`.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# {provider_name: caller_fn}
_L2_PROVIDERS: dict[str, Callable] = {}


def register_l2_provider(name: str, caller: Callable) -> None:
    """Register a callable that implements the L2 caller protocol.

    Signature: `caller(task, history, model, schema) -> Response`
    """
    _L2_PROVIDERS[name] = caller


def get_l2_caller(provider: str) -> Callable | None:
    """Return registered L2 caller or lazy-load a built-in one."""
    if provider in _L2_PROVIDERS:
        return _L2_PROVIDERS[provider]
    # Lazy-load built-ins
    try:
        mod = importlib.import_module(f"classifier.layers.layer2.providers.{provider}")
        if hasattr(mod, "call"):
            register_l2_provider(provider, mod.call)
            return mod.call
    except ImportError:
        pass
    return None


def list_l2_providers() -> list[str]:
    """Return all registered L2 provider names."""
    return sorted(_L2_PROVIDERS.keys())
