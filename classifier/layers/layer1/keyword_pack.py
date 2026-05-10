"""KeywordPack — programmatic API for adding domain keywords to Layer 1.

Use this when you don't want to hand-edit YAML files. Build packs in code,
register them on a `Router`, and L1 will see them on the next classification.

Example:
    from classifier import KeywordPack, TaskType, Router

    pack = (KeywordPack.builder("legal")
            .add(TaskType.REASONING,    ["differential", "argue", "precedent"])
            .add(TaskType.DOC_CREATION, ["clause", "indemnification"])
            .escalator("statute", weight=2)
            .build())

    router = Router(extra_keyword_packs=[pack])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ModelTier, TaskType


@dataclass(frozen=True)
class KeywordPack:
    """Immutable bundle of L1 keyword overrides.

    Use `KeywordPack.builder(name)` to construct one.
    """

    name: str
    task_keywords: dict = field(default_factory=dict)  # {TaskType: {"primary": [kw, ...]}}
    escalators: dict = field(default_factory=dict)  # {keyword: weight}
    domain_min_tier: dict = field(default_factory=dict)  # {keyword: ModelTier}

    @staticmethod
    def builder(name: str) -> _KeywordPackBuilder:
        return _KeywordPackBuilder(name)


class _KeywordPackBuilder:
    def __init__(self, name: str):
        self._name = name
        self._task_keywords: dict = {}
        self._escalators: dict = {}
        self._domain_min_tier: dict = {}

    def add(self, task_type: TaskType, keywords: list[str], group: str = "primary") -> _KeywordPackBuilder:
        """Add keywords for a task type. `group` defaults to 'primary' (highest weight)."""
        slot = self._task_keywords.setdefault(task_type, {})
        existing = slot.setdefault(group, [])
        for kw in keywords:
            if kw not in existing:
                existing.append(kw)
        return self

    def escalator(self, keyword: str, weight: int = 1) -> _KeywordPackBuilder:
        """Add a complexity-escalator keyword (e.g. 'distributed' bumps complexity)."""
        self._escalators[keyword] = int(weight)
        return self

    def min_tier(self, keyword: str, tier: ModelTier) -> _KeywordPackBuilder:
        """Force a minimum tier when this keyword appears (e.g. 'compliance' → MEDIUM)."""
        self._domain_min_tier[keyword] = tier
        return self

    def build(self) -> KeywordPack:
        return KeywordPack(
            name=self._name,
            task_keywords=dict(self._task_keywords),
            escalators=dict(self._escalators),
            domain_min_tier=dict(self._domain_min_tier),
        )


# ── Runtime registration ──────────────────────────────────────────────────────

_registered_packs: list[KeywordPack] = []


def register_extra_packs(packs: list[KeywordPack]) -> None:
    """Merge user-supplied packs into Layer 1's keyword dictionaries.

    Idempotent — re-registering the same pack name is a no-op.
    Called by Router(__init__) when `extra_keyword_packs=[...]` is passed.
    """
    from classifier.layers.layer1.constants import (
        _DOMAIN_MIN_TIER,
        _ESCALATORS,
        _TASK_KEYWORDS,
    )

    for pack in packs:
        if any(p.name == pack.name for p in _registered_packs):
            continue  # already registered

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
    """Test helper — wipe all registered packs (does NOT undo their effect on L1 dicts)."""
    _registered_packs.clear()


def _load_pack_from_yaml(path) -> KeywordPack | None:
    """Build a KeywordPack from a YAML file written by `dmr keywords add`."""
    import yaml

    from classifier.core.types import task_type_for

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    name = data.get("name") or path.stem
    builder = KeywordPack.builder(name)
    for tt_str, groups in (data.get("task_keywords") or {}).items():
        try:
            tt = task_type_for(tt_str)
        except Exception:
            continue
        for grp, kws in (groups or {}).items():
            if kws:
                builder.add(tt, list(kws), group=grp or "primary")
    for kw, w in (data.get("escalators") or {}).items():
        try:
            builder.escalator(kw, weight=int(w))
        except Exception:
            continue
    return builder.build()


def auto_load_user_packs() -> list[str]:
    """Auto-discover and register packs from ~/.dmr/keywords/*.yaml.

    Called once on Router() construction. Idempotent — re-registration is a
    no-op (packs dedupe by name).

    Honors $DMR_KEYWORDS_DIR for test/CI isolation.
    """
    import os
    from pathlib import Path

    env = os.environ.get("DMR_KEYWORDS_DIR")
    base = Path(env) if env else Path.home() / ".dmr" / "keywords"
    if not base.exists():
        return []
    loaded: list[str] = []
    packs: list[KeywordPack] = []
    for yml in sorted(base.glob("*.yaml")):
        pack = _load_pack_from_yaml(yml)
        if pack is None:
            continue
        if any(p.name == pack.name for p in _registered_packs):
            continue
        packs.append(pack)
        loaded.append(pack.name)
    if packs:
        register_extra_packs(packs)
    return loaded
