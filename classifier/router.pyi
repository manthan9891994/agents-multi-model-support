"""Type stubs for classifier.router."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from classifier.core.types import (
    ClassificationDecision,
    ContextSignals,
)
from classifier.layers.layer1.keyword_pack import KeywordPack

class Router:
    providers: List[str]
    extra_keyword_packs: List[KeywordPack]
    extra_pii_patterns: List[tuple]
    tier_matrix: Dict
    model_registry: Dict
    layer1_enabled: Optional[bool]
    layer2_enabled: Optional[bool]
    layer3_enabled: Optional[bool]
    escalation_threshold: Optional[float]
    layer3_threshold: Optional[float]
    budget_usd: Optional[float]
    cache_enabled: Optional[bool]

    def __init__(
        self,
        *,
        providers: Optional[List[str]] = ...,
        extra_keyword_packs: Optional[List[KeywordPack]] = ...,
        extra_pii_patterns: Optional[List[tuple]] = ...,
        tier_matrix: Optional[Dict] = ...,
        model_registry: Optional[Dict] = ...,
        layer1_enabled: Optional[bool] = ...,
        layer2_enabled: Optional[bool] = ...,
        layer3_enabled: Optional[bool] = ...,
        escalation_threshold: Optional[float] = ...,
        layer3_threshold: Optional[float] = ...,
        budget_usd: Optional[float] = ...,
        cache_enabled: Optional[bool] = ...,
    ) -> None: ...

    def classify(
        self,
        task: str,
        history: Optional[List[str]] = ...,
        context_signals: Optional[ContextSignals] = ...,
        provider: Optional[str] = ...,
    ) -> ClassificationDecision: ...

    def estimate_cost(
        self,
        task: str,
        *,
        provider: Optional[str] = ...,
        estimated_output_tokens: int = ...,
    ) -> Dict[str, Any]: ...

    def train(
        self,
        data: str | Path,
        *,
        output_path: Optional[str | Path] = ...,
        max_iter: int = ...,
    ) -> Dict[str, Any]: ...

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Router": ...

    @classmethod
    def from_preset(cls, name: str) -> "Router": ...


def classify(task: str, **kwargs: Any) -> ClassificationDecision: ...
