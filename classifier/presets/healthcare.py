"""Healthcare preset — HIPAA-style PII patterns + clinical vocabulary."""

from __future__ import annotations

import re

from classifier.core.types import ModelTier, TaskType
from classifier.layers.layer1.keyword_pack import KeywordPack

_HEALTHCARE_PACK = (
    KeywordPack.builder("healthcare")
    .add(
        TaskType.REASONING,
        [
            "differential",
            "diagnosis",
            "etiology",
            "pathophysiology",
            "contraindication",
            "drug interaction",
            "side effect",
            "prognosis",
            "comorbidity",
            "clinical decision",
        ],
    )
    .add(
        TaskType.ANALYZING,
        [
            "lab result",
            "trend",
            "panel",
            "interpret",
            "abnormal",
            "elevated",
            "decreased",
            "reference range",
        ],
    )
    .add(
        TaskType.DOC_CREATION,
        [
            "soap note",
            "discharge summary",
            "progress note",
            "admission note",
            "icd-10",
            "cpt",
            "encounter note",
            "h&p",
            "consult note",
            "prior authorization",
            "medical necessity",
        ],
    )
    .add(
        TaskType.MATH,
        [
            "egfr",
            "bmi",
            "meld",
            "child-pugh",
            "homa-ir",
            "creatinine clearance",
            "cha2ds2-vasc",
            "curb-65",
        ],
    )
    .escalator("multi-system", weight=2)
    .escalator("acute on chronic", weight=2)
    .escalator("polypharmacy", weight=1)
    .min_tier("compliance", ModelTier.MEDIUM)
    .min_tier("hipaa", ModelTier.MEDIUM)
    .min_tier("phi", ModelTier.MEDIUM)
    .build()
)


# HIPAA-relevant PII patterns (extras on top of the built-in MRN/SSN/DOB/email/phone)
_HEALTHCARE_PII = [
    # NPI (National Provider Identifier — 10 digits)
    (re.compile(r"\bNPI[:\s]*\d{10}\b"), "[NPI]"),
    # FIN / CSN (encounter numbers)
    (re.compile(r"\b(?:FIN|CSN|Encounter\s+#?)[:\s]*\d{6,12}\b", re.IGNORECASE), "[ENCOUNTER]"),
    # ICD-10 codes are NOT PII but we keep them visible — no scrubbing
]


def config() -> dict:
    """Return Router kwargs for the healthcare domain."""
    return {
        "extra_keyword_packs": [_HEALTHCARE_PACK],
        "extra_pii_patterns": _HEALTHCARE_PII,
    }
