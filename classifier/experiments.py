"""A/B testing and shadow-mode helpers for safe routing rollouts.

Two patterns:

**A/B test** — split traffic between control and treatment routers.
Use a sticky_key (e.g. user_id) to keep each user on a consistent variant.

    from classifier import Router
    from classifier.experiments import ABTest

    ab = ABTest(
        control=Router(),
        treatment=Router(tier_matrix=NEW_MATRIX),
        split=0.05,                                   # 5% to treatment
        sticky_key=lambda ctx: ctx.get("user_id"),
    )
    decision = ab.classify("Summarise this", ctx={"user_id": "u123"})

**Shadow mode** — primary router serves the decision; shadow router runs in
parallel and you log diffs. Used for safely validating a config change before
flipping the switch.

    from classifier.experiments import ShadowMode

    sm = ShadowMode(
        primary=Router(),
        shadow=Router(layer3_threshold=0.6),
        on_diff=lambda task, primary_d, shadow_d: log_diff(...),
    )
    decision = sm.classify("Summarise this")    # always returns primary's decision
"""
from __future__ import annotations

import hashlib
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ClassificationDecision
    from classifier.router import Router

logger = logging.getLogger(__name__)


def _stable_bucket(key: str) -> float:
    """Hash a key to a stable [0, 1) float for consistent A/B assignment."""
    h = hashlib.md5(key.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return n / (1 << 64)


class ABTest:
    """Route traffic between control and treatment Routers.

    Args:
        control:    Router for the control variant (default behaviour).
        treatment:  Router for the treatment variant (new config under test).
        split:      Fraction (0..1) of traffic sent to treatment. Default 0.5.
        sticky_key: Callable(ctx) -> str — extracts a stable key (e.g. user_id)
                    so each user consistently lands on the same variant.
                    If None, assignment is random per call.
        on_assign:  Optional callback invoked as on_assign(variant_name, ctx).
    """

    def __init__(
        self,
        control: "Router",
        treatment: "Router",
        *,
        split: float = 0.5,
        sticky_key: Optional[Callable[[dict], str]] = None,
        on_assign: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        if not 0.0 <= split <= 1.0:
            raise ValueError(f"split must be in [0, 1], got {split}")
        self.control     = control
        self.treatment   = treatment
        self.split       = split
        self.sticky_key  = sticky_key
        self.on_assign   = on_assign

    def assign(self, ctx: dict) -> str:
        if self.sticky_key:
            try:
                key = self.sticky_key(ctx) or ""
            except Exception:
                key = ""
            if key:
                bucket = _stable_bucket(key)
                return "treatment" if bucket < self.split else "control"
        return "treatment" if random.random() < self.split else "control"

    def classify(self, task: str, ctx: dict | None = None, **kwargs) -> "ClassificationDecision":
        ctx = ctx or {}
        variant = self.assign(ctx)
        if self.on_assign:
            try: self.on_assign(variant, ctx)
            except Exception: pass
        router = self.treatment if variant == "treatment" else self.control
        ctx_with_variant = {**ctx, "ab_variant": variant}
        return router.classify(task, hook_context=ctx_with_variant, **kwargs)


class ShadowMode:
    """Run a shadow router in parallel; return the primary's decision.

    Args:
        primary: Router whose decision is actually used.
        shadow:  Router whose output is computed for comparison only.
        on_diff: Callback fired when primary != shadow.
                 Signature: on_diff(task, primary_decision, shadow_decision).
        on_match: Optional callback fired when primary == shadow.
        timeout_secs: Max time to wait for shadow before discarding (default 2s).
    """

    def __init__(
        self,
        primary: "Router",
        shadow: "Router",
        *,
        on_diff: Optional[Callable[[str, Any, Any], None]] = None,
        on_match: Optional[Callable[[str, Any, Any], None]] = None,
        timeout_secs: float = 2.0,
    ) -> None:
        self.primary  = primary
        self.shadow   = shadow
        self.on_diff  = on_diff
        self.on_match = on_match
        self._timeout = timeout_secs
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dmr-shadow")
        self._stats = {"calls": 0, "matches": 0, "diffs": 0, "shadow_errors": 0}
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _is_match(self, a, b) -> bool:
        if a is None or b is None:
            return False
        return (
            a.tier      == b.tier
            and a.task_type  == b.task_type
            and a.complexity == b.complexity
        )

    def classify(self, task: str, ctx: dict | None = None, **kwargs) -> "ClassificationDecision":
        ctx = ctx or {}
        # Fire shadow in parallel
        shadow_future = self._executor.submit(
            lambda: self.shadow.classify(task, hook_context=ctx, **kwargs)
        )
        primary_decision = self.primary.classify(task, hook_context=ctx, **kwargs)

        # Try to compare without blocking the response significantly
        shadow_decision = None
        try:
            shadow_decision = shadow_future.result(timeout=self._timeout)
        except Exception as exc:
            with self._lock:
                self._stats["shadow_errors"] += 1
            logger.warning("ShadowMode: shadow call failed/timeout: %s", exc)

        with self._lock:
            self._stats["calls"] += 1
            if self._is_match(primary_decision, shadow_decision):
                self._stats["matches"] += 1
                if self.on_match:
                    try: self.on_match(task, primary_decision, shadow_decision)
                    except Exception: pass
            elif shadow_decision is not None:
                self._stats["diffs"] += 1
                if self.on_diff:
                    try: self.on_diff(task, primary_decision, shadow_decision)
                    except Exception: pass
        return primary_decision
