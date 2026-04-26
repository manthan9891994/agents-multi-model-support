"""Loads classifier/config/features.yaml into a typed FeatureFlags dataclass.

Usage anywhere in the classifier package:
    from classifier.config.feature_flags import feature_flags
    if feature_flags.pii_detection:
        ...
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(__file__).parent / "features.yaml"


@dataclass
class FeatureFlags:
    # ── Layer 1 ────────────────────────────────────────────────────────────────
    trivial_input_guard:        bool = True
    pii_detection:              bool = True
    continuation_detection:     bool = True
    history_bias:               bool = True
    domain_escalation:          bool = True
    question_type_override:     bool = True
    code_snippet_detection:     bool = True
    language_detection:         bool = True
    multi_task_detection:       bool = True
    negation_suppression:       bool = True
    keyword_packs:              bool = True
    format_request_deescalation: bool = True
    algorithm_detection:        bool = True
    escalator_scoring:          bool = True

    # ── Layer 2 ────────────────────────────────────────────────────────────────
    l2_rate_limiter:        bool = True
    l2_retry_with_backoff:  bool = True
    l2_output_validation:   bool = True   # output plausibility check (replaces input blocking)
    l2_fallback_model:      bool = True

    # ── System ─────────────────────────────────────────────────────────────────
    single_flight_coalescing:  bool = True
    health_tracker:            bool = True
    l1_l2_agreement:           bool = True
    per_user_personalization:  bool = False
    semantic_cache:            bool = False
    calibration:               bool = False


def _extract_enabled(val) -> Optional[bool]:
    """Accept both `key: true` and `key: {enabled: true, doc: ...}` forms."""
    if isinstance(val, bool):
        return val
    if isinstance(val, dict):
        raw = val.get("enabled")
        if isinstance(raw, bool):
            return raw
    return None


def _load() -> FeatureFlags:
    flags = FeatureFlags()
    if not _CONFIG_FILE.exists():
        logger.debug("features.yaml not found — using all defaults")
        return flags
    try:
        import yaml
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        changed: list[str] = []
        for section in ("layer1", "layer2", "system"):
            for key, val in (raw.get(section) or {}).items():
                enabled = _extract_enabled(val)
                if enabled is not None and hasattr(flags, key):
                    current = getattr(flags, key)
                    if enabled != current:
                        setattr(flags, key, enabled)
                        changed.append(f"{key}={'on' if enabled else 'OFF'}")

        if changed:
            logger.info("Feature flags overridden: %s", ", ".join(changed))
        else:
            logger.debug("Feature flags: all defaults active")

    except ImportError:
        logger.warning(
            "PyYAML not installed — feature flags using defaults. "
            "Run: pip install pyyaml"
        )
    except Exception as exc:
        logger.warning("Failed to load features.yaml (%s) — using defaults", exc)

    return flags


feature_flags = _load()
