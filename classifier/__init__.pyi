"""Type stubs for the classifier package public API."""
from typing import Any, Callable, List, Optional, TypeVar

from classifier.core.exceptions import (
    ClassificationError as ClassificationError,
    ClassifierError as ClassifierError,
    ConfigurationError as ConfigurationError,
    LayerNotAvailableError as LayerNotAvailableError,
    UnsupportedProviderError as UnsupportedProviderError,
)
from classifier.core.types import (
    ClassificationDecision as ClassificationDecision,
    ContextSignals as ContextSignals,
    ModelTier as ModelTier,
    TaskComplexity as TaskComplexity,
    TaskType as TaskType,
)
from classifier.core.registry import (
    MODEL_REGISTRY as MODEL_REGISTRY,
    TIER_MATRIX as TIER_MATRIX,
)
from classifier.router import Router as Router, classify as classify
from classifier.layers.layer1.keyword_pack import KeywordPack as KeywordPack

F = TypeVar("F", bound=Callable[..., Any])

def classify_task(
    task: str,
    provider: Optional[str] = ...,
    history: Optional[List[str]] = ...,
    context_signals: Optional[ContextSignals] = ...,
    task_stable: bool = ...,
    user_id: Optional[str] = ...,
) -> ClassificationDecision: ...

def route_model(
    provider: Optional[str] = ...,
    *,
    task_arg: str = ...,
    fallback_model: Optional[str] = ...,
    inject_as: str = ...,
) -> Callable[[F], F]: ...

def record_feedback(
    task: str,
    expected_type: str,
    expected_complexity: str,
    original_type: str,
    original_complexity: str,
) -> None: ...
