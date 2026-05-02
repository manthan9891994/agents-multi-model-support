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

# Runtime tables — populated from YAML at import time.
# The package ships zero hardcoded model names or pricing. See registry_loader.py.
MODEL_REGISTRY:     dict[str, dict[ModelTier, str]] = {}
MODEL_CAPABILITIES: dict[str, dict]                 = {}


def register_provider(
    name: str,
    tier_map: dict,
    *,
    capabilities: dict[str, dict] | None = None,
) -> None:
    """Register a new provider (or update an existing one) with its tier→model mapping.

    Args:
        name: Provider name (e.g. "mistral", "groq", "cohere").
        tier_map: {ModelTier.LOW: "model-name", ...} — accepts ModelTier enum or string.
        capabilities: Optional {model_name: capabilities_dict}. See MODEL_CAPABILITIES schema.

    Example:
        register_provider("groq", {
            ModelTier.LOW: "llama-3.3-8b-instant",
            ModelTier.HIGH: "llama-3.3-70b-versatile",
        }, capabilities={
            "llama-3.3-70b-versatile": {"context_window": 128_000, "supports_function_calling": True},
        })
    """
    # Accept string keys; map to ModelTier where possible
    normalized: dict = {}
    for k, v in tier_map.items():
        if isinstance(k, ModelTier):
            normalized[k] = v
        elif isinstance(k, str):
            try:
                normalized[ModelTier(k)] = v
            except ValueError:
                # Custom tier — leave as-is (works once #8 is implemented)
                normalized[k] = v
        else:
            normalized[k] = v
    MODEL_REGISTRY[name] = normalized

    if capabilities:
        for model, caps in capabilities.items():
            MODEL_CAPABILITIES.setdefault(model, {}).update(caps)


def list_providers() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def list_models() -> list[str]:
    """All registered model names, deduped."""
    seen = set()
    for tier_map in MODEL_REGISTRY.values():
        for m in tier_map.values():
            seen.add(m)
    return sorted(seen)


def capabilities_for(model: str) -> dict:
    """Return capability dict for a model (empty dict if unknown)."""
    return MODEL_CAPABILITIES.get(model, {})
