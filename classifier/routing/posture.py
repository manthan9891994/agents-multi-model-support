"""Posture dial — preset the composable cost levers from one number (0..4).

`DMR_SAVINGS_LEVEL` / `Router(savings_level=...)` composes the levers in
increasing aggressiveness, quality-neutral first. Individual lever flags
(`DMR_CACHE_AWARE`, …) still override afterward.
"""

from __future__ import annotations


def apply_posture(level: int) -> None:
    """Set the lever settings for posture `level` (cumulative). Process-global —
    this is a coarse operating-point choice, not a per-call override."""
    from classifier.infra.config import settings

    if not level or level <= 0:
        return
    # 1 Saver — quality-neutral, model-stable input-cost levers
    settings.dmr_cache_aware = True
    settings.dmr_effort_routing = True
    # 2 Balanced — trim accumulating context
    if level >= 2:
        settings.dmr_context_reduction = "prune"
    # 3 Aggressive — let cheaper models drive tools (gated) + recover on failure
    if level >= 3:
        settings.dmr_model_routing = "dispatch_downgrade"
        settings.dmr_escalate_on_failure = True
    # 4 Max — same levers, most aggressive (capability gate still protects tools)
    # (reserved for future flash-lite-where-safe; no extra toggles yet)
