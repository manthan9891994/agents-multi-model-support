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

MODEL_REGISTRY: dict[str, dict[ModelTier, str]] = {
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


# ── Capability metadata per model ────────────────────────────────────────────
# Used for context-window escalation and capability filtering.
# Schema: {"context_window": int, "supports_vision": bool, "supports_function_calling": bool,
#          "supports_streaming": bool, "supports_json_mode": bool, "region": str | None}
MODEL_CAPABILITIES: dict[str, dict] = {
    "gemini-2.5-flash":       {"context_window": 1_000_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": True, "region": None},
    "gemini-2.5-flash-lite":  {"context_window": 1_000_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": True, "region": None},
    "gemini-2.5-pro":         {"context_window": 2_000_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": True, "region": None},
    "claude-haiku-4-5-20251001": {"context_window": 200_000, "supports_vision": True, "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": False, "region": None},
    "claude-sonnet-4-6":      {"context_window": 200_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": False, "region": None},
    "claude-opus-4-7":        {"context_window": 200_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": False, "region": None},
    "gpt-4o-mini":            {"context_window": 128_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": True, "region": None},
    "gpt-4o":                 {"context_window": 128_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": True, "region": None},
    "gpt-4-turbo":            {"context_window": 128_000, "supports_vision": True,  "supports_function_calling": True, "supports_streaming": True, "supports_json_mode": True, "region": None},
}


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
