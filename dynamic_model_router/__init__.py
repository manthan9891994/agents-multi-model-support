"""dynamic_model_router — public package alias for `classifier`.

The package is internally organized as `classifier/...` for historical reasons.
Users should import from `dynamic_model_router` for the stable public API:

    from dynamic_model_router import Router, classify, KeywordPack
    from dynamic_model_router import TaskType, TaskComplexity, ModelTier
"""
from classifier import (
    Router,
    classify,
    classify_task,
    KeywordPack,
    ClassificationDecision,
    ContextSignals,
    ModelTier,
    TaskType,
    TaskComplexity,
    MODEL_REGISTRY,
    TIER_MATRIX,
    ClassificationError,
    ConfigurationError,
    UnsupportedProviderError,
    LayerNotAvailableError,
    record_feedback,
)

__version__ = "0.1.0"

__all__ = [
    "Router",
    "classify",
    "classify_task",
    "KeywordPack",
    "ClassificationDecision",
    "ContextSignals",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    "MODEL_REGISTRY",
    "TIER_MATRIX",
    "ClassificationError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "LayerNotAvailableError",
    "record_feedback",
]
