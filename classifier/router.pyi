"""Type stubs for classifier.router."""
from pathlib import Path
from typing import Any

from classifier.core.types import (
    ClassificationDecision,
    ContextSignals,
)
from classifier.layers.layer1.keyword_pack import KeywordPack

class Router:
    providers: list[str]
    extra_keyword_packs: list[KeywordPack]
    extra_pii_patterns: list[tuple]
    tier_matrix: dict
    model_registry: dict
    layer1_enabled: bool | None
    layer2_enabled: bool | None
    layer3_enabled: bool | None
    escalation_threshold: float | None
    layer3_threshold: float | None
    budget_usd: float | None
    cache_enabled: bool | None

    def __init__(
        self,
        *,
        providers: list[str] | None = ...,
        extra_keyword_packs: list[KeywordPack] | None = ...,
        extra_pii_patterns: list[tuple] | None = ...,
        tier_matrix: dict | None = ...,
        model_registry: dict | None = ...,
        layer1_enabled: bool | None = ...,
        layer2_enabled: bool | None = ...,
        layer3_enabled: bool | None = ...,
        escalation_threshold: float | None = ...,
        layer3_threshold: float | None = ...,
        budget_usd: float | None = ...,
        cache_enabled: bool | None = ...,
    ) -> None: ...

    def classify(
        self,
        task: str,
        history: list[str] | None = ...,
        context_signals: ContextSignals | None = ...,
        provider: str | None = ...,
    ) -> ClassificationDecision: ...

    def estimate_cost(
        self,
        task: str,
        *,
        provider: str | None = ...,
        estimated_output_tokens: int = ...,
    ) -> dict[str, Any]: ...

    def train(
        self,
        data: str | Path,
        *,
        output_path: str | Path | None = ...,
        max_iter: int = ...,
    ) -> dict[str, Any]: ...

    @classmethod
    def from_yaml(cls, path: str | Path) -> Router: ...

    @classmethod
    def from_preset(cls, name: str) -> Router: ...


def classify(task: str, **kwargs: Any) -> ClassificationDecision: ...
