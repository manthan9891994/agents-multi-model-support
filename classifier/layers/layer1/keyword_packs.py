import logging
from pathlib import Path

from classifier.core.types import TaskType, ModelTier
from .constants import _ESCALATORS, _DOMAIN_MIN_TIER, _TASK_KEYWORDS

logger = logging.getLogger(__name__)


def _load_keyword_packs() -> None:
    try:
        from classifier.infra.config import settings
        pack_names = [p.strip() for p in settings.keyword_packs.split(",") if p.strip()]
    except Exception:
        return
    if not pack_names:
        return

    packs_dir = Path(__file__).parent.parent.parent / "data" / "keyword_packs"
    for name in pack_names:
        pack_file = packs_dir / f"{name}.yaml"
        if not pack_file.exists():
            logger.warning("keyword pack not found: %s", pack_file)
            continue
        try:
            import yaml
            with open(pack_file, encoding="utf-8") as f:
                pack = yaml.safe_load(f)

            for item in pack.get("escalators", []):
                _ESCALATORS[item["kw"]] = int(item["weight"])

            _tier_name_map = {"low": ModelTier.LOW, "medium": ModelTier.MEDIUM, "high": ModelTier.HIGH}
            for item in pack.get("domain_min_tier", []):
                tier = _tier_name_map.get(item.get("tier", ""), ModelTier.MEDIUM)
                _DOMAIN_MIN_TIER[item["kw"]] = tier

            for tt_name, groups in pack.get("task_keywords", {}).items():
                try:
                    tt = TaskType(tt_name)
                    for group_key, kws in groups.items():
                        existing = _TASK_KEYWORDS.setdefault(tt, {}).setdefault(group_key, [])
                        existing.extend(kw for kw in kws if kw not in existing)
                except ValueError:
                    pass

            logger.info("Loaded keyword pack: %s", name)
        except ImportError:
            logger.warning("PyYAML not installed — skipping keyword pack: %s", name)
        except Exception as exc:
            logger.warning("Failed to load keyword pack %s: %s", name, exc)
