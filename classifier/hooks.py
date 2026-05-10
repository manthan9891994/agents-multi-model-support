"""Middleware / hook system — extension points around the classify() cascade.

Six hook points fire during a classify() call:
    pre_classify       — task and ctx, returns task (or raises to block)
    post_layer1        — after L1, before L3
    post_layer3        — after L3, before L2
    post_layer2        — after L2, before final adjustments
    post_classify      — final decision before return
    on_error           — when any layer raises (returns recovery decision or re-raises)

Hooks are how users build A/B testing, shadow mode, multi-tenancy, custom audit
logging, capability filtering, custom budget enforcement — without us
anticipating every use case.

Example:
    def audit_hook(task, decision, ctx):
        my_db.insert(task=task, model=decision.model_name, tenant=ctx.get("tenant_id"))
        return decision

    router = Router(post_classify_hooks=[audit_hook])
    decision = router.classify(task, hook_context={"tenant_id": "acme"})
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from classifier.core.types import ClassificationDecision

logger = logging.getLogger(__name__)

# Hook signatures (informal — Protocol for typing only):
#   PreHook  = Callable[[str, dict], str]
#   PostHook = Callable[[str, ClassificationDecision, dict], ClassificationDecision]
#   ErrorHook = Callable[[str, BaseException, dict], "ClassificationDecision | None"]


class HookManager:
    """Process-wide hook registry. Per-Router hooks are merged on each classify."""

    def __init__(self) -> None:
        self.pre_classify:  list[Callable] = []
        self.post_layer1:   list[Callable] = []
        self.post_layer3:   list[Callable] = []
        self.post_layer2:   list[Callable] = []
        self.post_classify: list[Callable] = []
        self.on_error:      list[Callable] = []
        self._lock = threading.RLock()

    def register(self, kind: str, fn: Callable) -> None:
        with self._lock:
            getattr(self, kind).append(fn)

    def unregister(self, kind: str, fn: Callable) -> None:
        with self._lock:
            try:
                getattr(self, kind).remove(fn)
            except ValueError:
                pass

    def clear(self, kind: str | None = None) -> None:
        """Clear hooks for one kind, or all kinds if None."""
        with self._lock:
            if kind is None:
                self.pre_classify.clear()
                self.post_layer1.clear()
                self.post_layer3.clear()
                self.post_layer2.clear()
                self.post_classify.clear()
                self.on_error.clear()
            else:
                getattr(self, kind).clear()

    def run_pre(self, task: str, ctx: dict[str, Any]) -> str:
        for fn in list(self.pre_classify):
            try:
                task = fn(task, ctx)
            except Exception:
                raise   # blocking exceptions intentionally propagate
        return task

    def run_post(
        self, kind: str, task: str, decision: ClassificationDecision, ctx: dict[str, Any],
    ) -> ClassificationDecision:
        for fn in list(getattr(self, kind)):
            try:
                result = fn(task, decision, ctx)
                if result is not None:
                    decision = result
            except Exception as exc:
                logger.warning("hook %s.%s raised: %s — keeping previous decision",
                               kind, getattr(fn, "__name__", "?"), exc)
        return decision

    def run_error(self, task: str, exc: BaseException, ctx: dict[str, Any]):
        """Returns a recovery decision if any handler returns one; else None."""
        for fn in list(self.on_error):
            try:
                result = fn(task, exc, ctx)
                if result is not None:
                    return result
            except Exception:
                continue
        return None


# Process-wide singleton
hook_manager = HookManager()


def register_hook(kind: str, fn: Callable) -> None:
    """Register a hook globally for the rest of the process.

    Args:
        kind: One of: "pre_classify", "post_layer1", "post_layer3", "post_layer2",
              "post_classify", "on_error".
        fn:   Hook callable matching the kind's signature.
    """
    hook_manager.register(kind, fn)


def unregister_hook(kind: str, fn: Callable) -> None:
    hook_manager.unregister(kind, fn)


def clear_hooks(kind: str | None = None) -> None:
    hook_manager.clear(kind)
