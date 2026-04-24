import logging
import time

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
from classifier.cache import cache
from classifier.cost_tracker import cost_tracker

logger = logging.getLogger(__name__)


def classify_task(task: str, provider: str = None) -> ClassificationDecision:
    """Classify a task and return the best model for it.

    Args:
        task:     The user's input text.
        provider: One of 'google', 'openai', 'anthropic'.
                  Defaults to DEFAULT_PROVIDER from .env.

    Returns:
        ClassificationDecision with model_name, tier, task_type, complexity,
        layer_used, latency_ms.

    Raises:
        ClassificationError:      Task is empty or classification fails.
        UnsupportedProviderError: Provider not in registry.
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

    # ── Budget guard ──────────────────────────────────────────────────────────
    if cost_tracker.is_exhausted():
        tier = ModelTier.LOW
        model_name = MODEL_REGISTRY[resolved_provider][tier]
        return ClassificationDecision(
            model_name=model_name, tier=tier,
            task_type=TaskType.DOC_CREATION,
            complexity=TaskComplexity.SIMPLE,
            reasoning="budget exhausted — forced LOW",
            confidence=1.0, provider=resolved_provider,
            layer_used="budget_guard", latency_ms=0.0,
        )

    if cost_tracker.should_downgrade():
        # Cap at MEDIUM — do not allow HIGH when budget is near limit
        max_tier = ModelTier.MEDIUM
    else:
        max_tier = None  # no cap

    # ── Cache lookup ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if settings.cache_enabled:
        cached = cache.get(task, resolved_provider)
        if cached is not None:
            return cached

    # ── Layer 1: keyword + token heuristic (always available) ────────────────
    try:
        task_type, complexity, tier, confidence, reasoning = classify_layer1(task)
    except Exception as exc:
        raise ClassificationError(f"Layer 1 classification failed: {exc}") from exc

    # Apply budget cap
    if max_tier is not None and tier == ModelTier.HIGH:
        tier = max_tier
        reasoning += " [capped to MEDIUM: budget >80%]"

    latency_ms = (time.perf_counter() - t0) * 1000
    model_name  = MODEL_REGISTRY[resolved_provider][tier]

    decision = ClassificationDecision(
        model_name=model_name,
        tier=tier,
        task_type=task_type,
        complexity=complexity,
        reasoning=reasoning,
        confidence=confidence,
        provider=resolved_provider,
        layer_used="layer1",
        latency_ms=round(latency_ms, 2),
    )

    logger.info(
        "Classified | %s => %s [%s | %s | %s | %.1fms]",
        resolved_provider,
        model_name,
        tier.value.upper(),
        task_type.value,
        complexity.value,
        latency_ms,
    )

    # ── Cache store ───────────────────────────────────────────────────────────
    if settings.cache_enabled:
        cache.set(task, resolved_provider, decision)

    # ── Decision log ──────────────────────────────────────────────────────────
    if settings.log_decisions:
        from classifier.decision_logger import log_decision
        log_decision(task, decision, layer_used="layer1", latency_ms=latency_ms)

    return decision


__all__ = [
    "classify_task",
    "ClassificationDecision",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    "MODEL_REGISTRY",
    "TIER_MATRIX",
    "ClassifierError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "ClassificationError",
    "LayerNotAvailableError",
]
