"""Type stubs for the classifier package public API."""
from collections.abc import Callable
from typing import Any, TypeVar

from classifier.core.exceptions import (
    ClassificationError as ClassificationError,
)
from classifier.core.exceptions import (
    ClassifierError as ClassifierError,
)
from classifier.core.exceptions import (
    ConfigurationError as ConfigurationError,
)
from classifier.core.exceptions import (
    LayerNotAvailableError as LayerNotAvailableError,
)
from classifier.core.exceptions import (
    UnsupportedProviderError as UnsupportedProviderError,
)
from classifier.core.registry import (
    MODEL_REGISTRY as MODEL_REGISTRY,
)
from classifier.core.registry import (
    TIER_MATRIX as TIER_MATRIX,
)
from classifier.core.types import (
    ClassificationDecision as ClassificationDecision,
)
from classifier.core.types import (
    ContextSignals as ContextSignals,
)
from classifier.core.types import (
    ModelTier as ModelTier,
)
from classifier.core.types import (
    TaskComplexity as TaskComplexity,
)
from classifier.core.types import (
    TaskType as TaskType,
)
from classifier.layers.layer1.keyword_pack import KeywordPack as KeywordPack
from classifier.router import Router as Router
from classifier.router import classify as classify

F = TypeVar("F", bound=Callable[..., Any])

def classify_task(
    task: str,
    provider: str | None = ...,
    history: list[str] | None = ...,
    context_signals: ContextSignals | None = ...,
    task_stable: bool = ...,
    user_id: str | None = ...,
) -> ClassificationDecision: ...

def route_model(
    provider: str | None = ...,
    *,
    task_arg: str = ...,
    fallback_model: str | None = ...,
    inject_as: str = ...,
) -> Callable[[F], F]: ...

def record_feedback(
    task: str,
    expected_type: str,
    expected_complexity: str,
    original_type: str,
    original_complexity: str,
) -> None: ...
