"""Legal preset — skeleton. Extend with your firm's vocabulary.

This is a starter kit. To make it production-grade, add:
  - Domain-specific keywords (jurisdictions, practice areas)
  - Citation patterns for your jurisdiction (e.g., Bluebook, OSCOLA)
  - Privilege markers as PII patterns
  - Training data from past matters
"""

from __future__ import annotations

import re

from classifier.core.types import ModelTier, TaskType
from classifier.layers.layer1.keyword_pack import KeywordPack

_LEGAL_PACK = (
    KeywordPack.builder("legal")
    .add(
        TaskType.REASONING,
        [
            "precedent",
            "argue",
            "interpret statute",
            "burden of proof",
            "standing",
            "jurisdiction",
            "doctrine",
        ],
    )
    .add(
        TaskType.DOC_CREATION,
        [
            "clause",
            "indemnification",
            "non-compete",
            "termination",
            "warranty",
            "covenant",
            "memorandum",
            "brief",
            "motion",
            "deposition",
            "affidavit",
        ],
    )
    .add(
        TaskType.ANALYZING,
        [
            "case citation",
            "discovery",
            "review documents",
            "redline",
        ],
    )
    .escalator("constitutional", weight=2)
    .escalator("multi-jurisdictional", weight=2)
    .min_tier("privileged", ModelTier.MEDIUM)
    .min_tier("attorney work product", ModelTier.MEDIUM)
    .build()
)


_LEGAL_PII = [
    # Case docket numbers — varies by jurisdiction; example for US federal
    (re.compile(r"\b\d{1,2}:\d{2}-cv-\d{4,5}\b"), "[CASE_NO]"),
    # Bar numbers — TODO: jurisdiction-specific
]


def config() -> dict:
    """Return Router kwargs for the legal domain."""
    return {
        "extra_keyword_packs": [_LEGAL_PACK],
        "extra_pii_patterns": _LEGAL_PII,
    }
