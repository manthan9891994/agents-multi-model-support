"""Pack registry — the single mutation path for L1's keyword globals.

Every loader (programmatic, bundled YAML, user-authored YAML) ends here.
Centralising mutation in one place keeps the data flow obvious and makes
debugging "where did this keyword come from?" tractable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .keyword_pack import KeywordPack


_registered_packs: list[KeywordPack] = []


def register_extra_packs(packs: list[KeywordPack]) -> None:
    """Merge packs into Layer 1's keyword dictionaries.

    Idempotent — re-registering a pack with the same ``name`` is a no-op.
    Called by ``Router(__init__)`` and by every loader in ``pack_loaders``.
    """
    from classifier.layers.layer1.constants import (
        _DOMAIN_MIN_TIER,
        _ESCALATORS,
        _TASK_KEYWORDS,
    )

    for pack in packs:
        if any(p.name == pack.name for p in _registered_packs):
            continue  # already registered — dedupe by pack.name

        for tt, groups in pack.task_keywords.items():
            slot = _TASK_KEYWORDS.setdefault(tt, {})
            for group_key, kws in groups.items():
                existing = slot.setdefault(group_key, [])
                for kw in kws:
                    if kw not in existing:
                        existing.append(kw)

        for kw, weight in pack.escalators.items():
            _ESCALATORS[kw] = weight

        for kw, tier in pack.domain_min_tier.items():
            _DOMAIN_MIN_TIER[kw] = tier

        _registered_packs.append(pack)


def list_registered() -> list[str]:
    """Return names of currently-registered extra packs (for debugging)."""
    return [p.name for p in _registered_packs]


def clear_registered() -> None:
    """Test helper — wipe registered packs (does NOT undo their effect on L1 dicts)."""
    _registered_packs.clear()
