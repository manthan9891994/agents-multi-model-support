from classifier.types import ModelTier, TaskType, TaskComplexity, ClassificationDecision
from classifier.layer1 import classify_layer1
from classifier.registry import MODEL_REGISTRY, TIER_MATRIX
from classifier.config import DEFAULT_PROVIDER


def classify_task(task: str, provider: str = DEFAULT_PROVIDER) -> ClassificationDecision:
    if provider not in MODEL_REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(MODEL_REGISTRY)}")

    task_type, complexity, tier, confidence, reasoning = classify_layer1(task)
    model_name = MODEL_REGISTRY[provider][tier]

    return ClassificationDecision(
        model_name=model_name,
        tier=tier,
        task_type=task_type,
        complexity=complexity,
        reasoning=reasoning,
        confidence=confidence,
        provider=provider,
    )


__all__ = [
    "classify_task",
    "ClassificationDecision",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    "MODEL_REGISTRY",
    "TIER_MATRIX",
]
