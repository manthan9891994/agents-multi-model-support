class ClassifierError(Exception):
    """Base exception for the classifier package."""


class ConfigurationError(ClassifierError):
    """Raised when .env config is missing, invalid, or incomplete.

    Example: GOOGLE_API_KEY not set, or DEFAULT_PROVIDER is unknown.
    """


class UnsupportedProviderError(ClassifierError):
    """Raised when the requested provider is not in MODEL_REGISTRY.

    Example: classify_task(task, provider="unknown")
    """


class ClassificationError(ClassifierError):
    """Raised when classification fails unexpectedly at runtime."""


class LayerNotAvailableError(ClassifierError):
    """Raised when a classification layer cannot be used.

    Example: Layer 2 requested but Ollama is not running.
             Layer 4 requested but the model is not trained yet.
    """
