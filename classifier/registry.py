from classifier.types import ModelTier, TaskType, TaskComplexity

TIER_MATRIX = {
    (TaskType.REASONING,     TaskComplexity.SIMPLE):   ModelTier.MEDIUM,
    (TaskType.REASONING,     TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.REASONING,     TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.REASONING,     TaskComplexity.RESEARCH): ModelTier.HIGH,

    (TaskType.THINKING,      TaskComplexity.SIMPLE):   ModelTier.MEDIUM,
    (TaskType.THINKING,      TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.THINKING,      TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.THINKING,      TaskComplexity.RESEARCH): ModelTier.HIGH,

    (TaskType.ANALYZING,     TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.ANALYZING,     TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.ANALYZING,     TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.ANALYZING,     TaskComplexity.RESEARCH): ModelTier.HIGH,

    (TaskType.CODE_CREATION, TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.CODE_CREATION, TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.CODE_CREATION, TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.CODE_CREATION, TaskComplexity.RESEARCH): ModelTier.HIGH,

    (TaskType.DOC_CREATION,  TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.DOC_CREATION,  TaskComplexity.STANDARD): ModelTier.LOW,
    (TaskType.DOC_CREATION,  TaskComplexity.COMPLEX):  ModelTier.MEDIUM,
    (TaskType.DOC_CREATION,  TaskComplexity.RESEARCH): ModelTier.MEDIUM,
}

MODEL_REGISTRY = {
    "google": {
        ModelTier.LOW:    "gemini-2.5-flash-lite",
        ModelTier.MEDIUM: "gemini-2.5-flash",
        ModelTier.HIGH:   "gemini-2.5-pro",
    },
    "anthropic": {
        ModelTier.LOW:    "claude-haiku-4-5-20251001",
        ModelTier.MEDIUM: "claude-sonnet-4-6",
        ModelTier.HIGH:   "claude-opus-4-7",
    },
    "openai": {
        ModelTier.LOW:    "gpt-4o-mini",
        ModelTier.MEDIUM: "gpt-4o",
        ModelTier.HIGH:   "gpt-4-turbo",
    },
}
