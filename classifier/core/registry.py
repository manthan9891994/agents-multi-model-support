from classifier.core.types import ModelTier, TaskType, TaskComplexity

TIER_MATRIX = {
    # ── Reasoning ─────────────────────────────────────────────────────────────
    (TaskType.REASONING,     TaskComplexity.SIMPLE):   ModelTier.MEDIUM,
    (TaskType.REASONING,     TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.REASONING,     TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.REASONING,     TaskComplexity.RESEARCH): ModelTier.HIGH,

    # ── Thinking / design ─────────────────────────────────────────────────────
    (TaskType.THINKING,      TaskComplexity.SIMPLE):   ModelTier.MEDIUM,
    (TaskType.THINKING,      TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.THINKING,      TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.THINKING,      TaskComplexity.RESEARCH): ModelTier.HIGH,

    # ── Data analysis ─────────────────────────────────────────────────────────
    (TaskType.ANALYZING,     TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.ANALYZING,     TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.ANALYZING,     TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.ANALYZING,     TaskComplexity.RESEARCH): ModelTier.HIGH,

    # ── Code creation / debugging ─────────────────────────────────────────────
    (TaskType.CODE_CREATION, TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.CODE_CREATION, TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.CODE_CREATION, TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.CODE_CREATION, TaskComplexity.RESEARCH): ModelTier.HIGH,

    # ── Documentation / writing ───────────────────────────────────────────────
    (TaskType.DOC_CREATION,  TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.DOC_CREATION,  TaskComplexity.STANDARD): ModelTier.LOW,
    (TaskType.DOC_CREATION,  TaskComplexity.COMPLEX):  ModelTier.MEDIUM,
    (TaskType.DOC_CREATION,  TaskComplexity.RESEARCH): ModelTier.MEDIUM,

    # ── Translation ───────────────────────────────────────────────────────────
    (TaskType.TRANSLATION,   TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.TRANSLATION,   TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.TRANSLATION,   TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.TRANSLATION,   TaskComplexity.RESEARCH): ModelTier.HIGH,

    # ── Math / computation ────────────────────────────────────────────────────
    (TaskType.MATH,          TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.MATH,          TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.MATH,          TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.MATH,          TaskComplexity.RESEARCH): ModelTier.HIGH,

    # ── Casual conversation ───────────────────────────────────────────────────
    (TaskType.CONVERSATION,  TaskComplexity.SIMPLE):   ModelTier.LOW,
    (TaskType.CONVERSATION,  TaskComplexity.STANDARD): ModelTier.LOW,
    (TaskType.CONVERSATION,  TaskComplexity.COMPLEX):  ModelTier.LOW,
    (TaskType.CONVERSATION,  TaskComplexity.RESEARCH): ModelTier.LOW,

    # ── Multimodal (image/audio/vision) ──────────────────────────────────────
    (TaskType.MULTIMODAL,    TaskComplexity.SIMPLE):   ModelTier.MEDIUM,
    (TaskType.MULTIMODAL,    TaskComplexity.STANDARD): ModelTier.MEDIUM,
    (TaskType.MULTIMODAL,    TaskComplexity.COMPLEX):  ModelTier.HIGH,
    (TaskType.MULTIMODAL,    TaskComplexity.RESEARCH): ModelTier.HIGH,
}

MODEL_REGISTRY = {
    "google": {
        ModelTier.LOW:    "gemini-2.5-flash",       # flash-lite quota exhausted on free tier
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
