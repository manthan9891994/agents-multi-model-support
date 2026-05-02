"""Fintech preset — skeleton. Extend with your firm's vocabulary.

To make it production-grade, add:
  - Trading instruments / asset classes you cover
  - Compliance frameworks (PCI-DSS, KYC/AML, SOX)
  - Account number / card patterns as PII
  - Training data from past tickets / queries
"""
from __future__ import annotations

import re

from classifier.core.types import TaskType, ModelTier
from classifier.layers.layer1.keyword_pack import KeywordPack


_FINTECH_PACK = (
    KeywordPack.builder("fintech")
    .add(TaskType.REASONING, [
        "risk-adjusted", "hedging strategy", "portfolio rebalance",
        "options pricing", "yield curve", "duration",
    ])
    .add(TaskType.MATH, [
        "var", "value at risk", "sharpe ratio", "alpha", "beta",
        "expected return", "volatility", "correlation matrix",
    ])
    .add(TaskType.ANALYZING, [
        "market data", "trade reconciliation", "p&l", "exposure",
        "compliance check", "kyc", "aml",
    ])
    .add(TaskType.DOC_CREATION, [
        "compliance filing", "audit report", "trade confirmation",
        "policy document",
    ])
    .escalator("multi-asset", weight=1)
    .escalator("regulatory filing", weight=2)
    .min_tier("kyc", ModelTier.MEDIUM)
    .min_tier("aml", ModelTier.MEDIUM)
    .min_tier("pci", ModelTier.MEDIUM)
    .build()
)


_FINTECH_PII = [
    # Credit card patterns (PCI-DSS) — major networks 13–19 digits, common forms
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[CARD]"),
    # IBAN
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"), "[IBAN]"),
    # Generic account number marker
    (re.compile(r"\b(?:Account|Acct)[:\s#]*\d{6,16}\b", re.IGNORECASE), "[ACCT]"),
]


def config() -> dict:
    """Return Router kwargs for the fintech domain."""
    return {
        "extra_keyword_packs": [_FINTECH_PACK],
        "extra_pii_patterns":  _FINTECH_PII,
    }
