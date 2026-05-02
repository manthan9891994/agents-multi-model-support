class ClassifierError(Exception):
    """Base exception for the classifier package."""


class ConfigurationError(ClassifierError):
    """Raised when .env config is missing, invalid, or incomplete."""


class UnsupportedProviderError(ClassifierError):
    """Raised when the requested provider is not in MODEL_REGISTRY."""


class ClassificationError(ClassifierError):
    """Raised when classification fails at runtime.

    Attributes:
        layer:      Which layer failed ('layer1', 'layer2', 'layer3').
        task_preview: First 80 chars of the task that caused the failure.
        suggestion: Human-readable fix hint.
    """

    def __init__(
        self,
        message: str,
        *,
        layer: str = "unknown",
        task: str = "",
        suggestion: str = "",
    ) -> None:
        self.layer = layer
        self.task_preview = (task[:80] + "…") if len(task) > 80 else task
        self.suggestion = suggestion

        parts = [message]
        if layer and layer != "unknown":
            parts.append(f"layer={layer}")
        if self.task_preview:
            parts.append(f"task={self.task_preview!r}")
        if suggestion:
            parts.append(f"hint: {suggestion}")

        super().__init__(" | ".join(parts))


class LayerNotAvailableError(ClassifierError):
    """Raised when a classification layer cannot be used.

    Example: Layer 3 embedding model not built yet.
    """
