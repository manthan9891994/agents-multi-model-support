"""Layer plugin interface — register custom classification layers.

A Layer plugin is anything that implements:

    class MyLayer:
        name = "my_layer_name"
        runs_after = "layer1"   # one of: "pre", "layer1", "layer3", "layer2", "post"

        def classify(self, task: str, history: list[str] | None = None) -> tuple | None:
            # Return (TaskType, TaskComplexity, ModelTier, confidence, reasoning)
            # Or None to abstain — cascade falls through to the next layer.
            ...

Then register:

    from classifier.layers.plugin import register_layer
    register_layer(MyLayer())

The cascade fires plugins in the order they were registered, at the position
declared by `runs_after`. If a plugin returns a non-None result, the cascade
treats it like a normal layer outcome.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Layer(Protocol):
    name: str
    runs_after: str

    def classify(self, task: str, history: list[str] | None = None) -> tuple | None: ...


# {position: [layers_in_registration_order]}
_PLUGINS: dict[str, list[Layer]] = {
    "pre": [],
    "layer1": [],
    "layer3": [],
    "layer2": [],
    "post": [],
}


def register_layer(layer: Layer) -> None:
    """Register a layer plugin. Idempotent on `name`."""
    pos = getattr(layer, "runs_after", "layer1")
    if pos not in _PLUGINS:
        raise ValueError(f"runs_after must be one of {list(_PLUGINS)}, got {pos!r}")
    name = getattr(layer, "name", repr(layer))
    if any(getattr(p, "name", None) == name for p in _PLUGINS[pos]):
        return
    _PLUGINS[pos].append(layer)
    logger.info("layers.plugin: registered %r (runs_after=%s)", name, pos)


def unregister_layer(name: str) -> None:
    for layers in _PLUGINS.values():
        for i, p in enumerate(layers):
            if getattr(p, "name", None) == name:
                layers.pop(i)
                return


def list_layers() -> dict[str, list[str]]:
    return {pos: [getattr(p, "name", repr(p)) for p in layers] for pos, layers in _PLUGINS.items()}


def run_layers_at(position: str, task: str, history=None) -> tuple | None:
    """Run plugins at `position`, returning the first non-None result, or None."""
    for layer in _PLUGINS.get(position, []):
        try:
            result = layer.classify(task, history=history)
            if result is not None:
                return result
        except Exception as exc:
            logger.warning("layer plugin %r raised: %s", getattr(layer, "name", "?"), exc)
    return None


def clear_layers() -> None:
    for v in _PLUGINS.values():
        v.clear()
