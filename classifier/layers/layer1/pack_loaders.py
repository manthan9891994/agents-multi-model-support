"""Pack loaders — the single I/O surface for L1 keyword YAML files.

Three sources, one normalised path:

    bundled    classifier/data/keyword_packs/<name>.yaml   (env: KEYWORD_PACKS=...)
    user       ~/.dmr/keywords/<name>.yaml                 (or $DMR_KEYWORDS_DIR)
    arbitrary  load_pack_from_file(path)                   (caller-controlled)

Every loader produces a ``KeywordPack`` and routes through
``pack_registry.register_extra_packs`` for the actual mutation. This file owns
all schema-normalisation logic so the registry never sees raw YAML.

Two YAML schemas are accepted (for back-compat with older bundled files):

    NEW (preferred):
        escalators:        {clinical: 2, distributed: 3}
        task_keywords:
          reasoning: {primary: [foo, bar]}

    LEGACY (still supported):
        escalators: [{kw: clinical, weight: 2}]
        domain_min_tier: [{kw: hipaa, tier: high}]
        task_keywords:
          reasoning: {primary: [foo, bar]}

Both forms parse to the same in-memory ``KeywordPack`` shape.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .pack_registry import _registered_packs, register_extra_packs

if TYPE_CHECKING:
    from .keyword_pack import KeywordPack

logger = logging.getLogger(__name__)


# ── Public loaders ────────────────────────────────────────────────────────────


def load_pack_from_file(path: Path) -> KeywordPack | None:
    """Read a single YAML file and return a ``KeywordPack`` (no registration).

    Returns ``None`` if the file is missing, unreadable, or YAML is invalid.
    The caller chooses whether to register the result.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — cannot load %s", path)
        return None

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read keyword pack %s: %s", path, exc)
        return None

    return _build_pack_from_dict(data, fallback_name=path.stem)


def load_user_packs() -> list[str]:
    """Auto-discover packs from ``~/.dmr/keywords/*.yaml`` and register them.

    Honors ``$DMR_KEYWORDS_DIR`` for test/CI isolation. Idempotent.

    Returns the names of the packs registered on this call (deduped — a pack
    that's already registered is skipped silently).
    """
    base = Path(os.environ.get("DMR_KEYWORDS_DIR") or Path.home() / ".dmr" / "keywords")
    if not base.exists():
        return []
    return _scan_dir_and_register(base)


def load_bundled_packs() -> list[str]:
    """Load packs named in ``settings.keyword_packs`` from the bundled data dir.

    The setting (``KEYWORD_PACKS=healthcare,legal`` env var or ``dmr.yaml``
    field) is a comma-separated list of pack names. Each name is resolved to
    ``classifier/data/keyword_packs/<name>.yaml``.

    Idempotent — already-registered packs are skipped.
    """
    try:
        from classifier.infra.config import settings

        raw = getattr(settings, "keyword_packs", "") or ""
        names = [n.strip() for n in raw.split(",") if n.strip()]
    except Exception:
        return []
    if not names:
        return []

    from .keyword_pack import KeywordPack  # noqa: F401 — runtime use below

    bundled_dir = Path(__file__).parent.parent.parent / "data" / "keyword_packs"
    loaded: list[str] = []
    packs: list[KeywordPack] = []
    for name in names:
        f = bundled_dir / f"{name}.yaml"
        if not f.exists():
            logger.warning("Bundled keyword pack not found: %s", f)
            continue
        pack = load_pack_from_file(f)
        if pack is None:
            continue
        if any(p.name == pack.name for p in _registered_packs):
            continue
        packs.append(pack)
        loaded.append(pack.name)
    if packs:
        register_extra_packs(packs)
    return loaded


# ── Internal helpers ──────────────────────────────────────────────────────────


def _scan_dir_and_register(base: Path) -> list[str]:
    """Read every ``*.yaml`` in ``base``, build packs, register the new ones."""
    from .keyword_pack import KeywordPack  # noqa: F401 — runtime use below

    loaded: list[str] = []
    packs: list[KeywordPack] = []
    for yml in sorted(base.glob("*.yaml")):
        pack = load_pack_from_file(yml)
        if pack is None:
            continue
        if any(p.name == pack.name for p in _registered_packs):
            continue
        packs.append(pack)
        loaded.append(pack.name)
    if packs:
        register_extra_packs(packs)
    return loaded


def _build_pack_from_dict(data: dict, *, fallback_name: str) -> KeywordPack:
    """Convert a YAML-parsed dict (either schema) to a ``KeywordPack``."""
    from classifier.core.types import ModelTier, task_type_for

    from .keyword_pack import KeywordPack

    name = data.get("name") or fallback_name
    builder = KeywordPack.builder(name)

    # task_keywords — same shape in both schemas
    for tt_str, groups in (data.get("task_keywords") or {}).items():
        try:
            tt = task_type_for(tt_str)
        except Exception:
            continue
        for grp, kws in (groups or {}).items():
            if kws:
                builder.add(tt, list(kws), group=grp or "primary")

    # escalators — accept either dict (new) or list-of-dicts (legacy)
    escalators = data.get("escalators")
    if isinstance(escalators, dict):
        for kw, w in escalators.items():
            try:
                builder.escalator(kw, weight=int(w))
            except Exception:
                continue
    elif isinstance(escalators, list):
        for item in escalators:
            kw = (item or {}).get("kw")
            if not kw:
                continue
            try:
                builder.escalator(kw, weight=int(item.get("weight", 1)))
            except Exception:
                continue

    # domain_min_tier — legacy list-of-dicts form (new schema doesn't expose this yet)
    _tier_map = {"low": ModelTier.LOW, "medium": ModelTier.MEDIUM, "high": ModelTier.HIGH}
    domain_min = data.get("domain_min_tier")
    if isinstance(domain_min, list):
        for item in domain_min:
            kw = (item or {}).get("kw")
            tier = _tier_map.get((item or {}).get("tier", ""))
            if kw and tier:
                builder.min_tier(kw, tier)
    elif isinstance(domain_min, dict):
        for kw, tier_str in domain_min.items():
            tier = _tier_map.get(str(tier_str).lower())
            if tier:
                builder.min_tier(kw, tier)

    return builder.build()
