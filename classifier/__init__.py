import logging

from classifier.exceptions import (
    ClassificationError,
    ConfigurationError,
    LayerNotAvailableError,
    UnsupportedProviderError,
)
from classifier.types import ClassificationDecision, TaskComplexity, TaskType, ModelTier
from classifier.layer1 import classify_layer1
from classifier.registry import MODEL_REGISTRY, TIER_MATRIX
from classifier.config import settings

logger = logging.getLogger(__name__)


def classify_task(task: str, provider: str = None) -> ClassificationDecision:
    """Classify a task and return the best model for it.

    Args:
        task:     The user's input text.
        provider: One of 'google', 'openai', 'anthropic'.
                  Defaults to DEFAULT_PROVIDER from .env.

    Returns:
        ClassificationDecision with model_name, tier, task_type, complexity.

    Raises:
        ClassificationError:    If task is empty or classification fails.
        UnsupportedProviderError: If provider is not in the registry.
    """
    resolved_provider = provider or settings.default_provider

    if resolved_provider not in MODEL_REGISTRY:
        raise UnsupportedProviderError(
            f"Provider '{resolved_provider}' is not supported. "
            f"Choose from: {sorted(MODEL_REGISTRY)}"
        )

    if not task or not task.strip():
        raise ClassificationError(
            "Task cannot be empty. Provide a non-empty string to classify."
        )

    try:
        task_type, complexity, tier, confidence, reasoning = classify_layer1(task)
    except Exception as exc:
        raise ClassificationError(
            f"Layer 1 classification failed: {exc}"
        ) from exc

    model_name = MODEL_REGISTRY[resolved_provider][tier]

    logger.debug(
        "Classified | provider=%s type=%s complexity=%s tier=%s model=%s confidence=%.2f",
        resolved_provider,
        task_type.value,
        complexity.value,
        tier.value,
        model_name,
        confidence,
    )

    return ClassificationDecision(
        model_name=model_name,
        tier=tier,
        task_type=task_type,
        complexity=complexity,
        reasoning=reasoning,
        confidence=confidence,
        provider=resolved_provider,
    )


__all__ = [
    "classify_task",
    "ClassificationDecision",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    "MODEL_REGISTRY",
    "TIER_MATRIX",
    # exceptions — re-exported so callers don't need to import sub-modules
    "ClassifierError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "ClassificationError",
    "LayerNotAvailableError",
]
