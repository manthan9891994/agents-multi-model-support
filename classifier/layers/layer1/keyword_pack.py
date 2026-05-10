"""KeywordPack — domain model for L1 keyword overrides.

Pure data: an immutable dataclass and a fluent builder. No I/O, no side effects.

For YAML loading see ``pack_loaders``.
For runtime registration into L1's keyword dictionaries see ``pack_registry``.

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

    Use ``KeywordPack.builder(name)`` to construct one.
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
        """Add keywords for a task type. ``group`` defaults to 'primary' (highest weight)."""
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


# ── Backwards-compatibility re-exports ───────────────────────────────────────
# Pre-v0.3.0 callers imported register/list/clear from this module. The clean
# home is now ``pack_registry`` (mutation) and ``pack_loaders`` (I/O), but the
# old import paths continue to work so we don't break any user code.
from .pack_loaders import load_pack_from_file as _load_pack_from_yaml  # noqa: E402, F401
from .pack_loaders import load_user_packs as auto_load_user_packs  # noqa: E402, F401
from .pack_registry import clear_registered, list_registered, register_extra_packs  # noqa: E402, F401
