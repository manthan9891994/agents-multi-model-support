class ClassifierError(Exception):
    """Base exception for the classifier package."""


class ConfigurationError(ClassifierError):
    """Raised when .env config is missing, invalid, or incomplete."""


class UnsupportedProviderError(ClassifierError):
    """Raised when the requested provider is not in MODEL_REGISTRY."""


class ClassificationError(ClassifierError):
    """Raised when classification fails unexpectedly at runtime."""


class LayerNotAvailableError(ClassifierError):
    """Raised when a classification layer cannot be used.

    Example: Layer 3 embedding model not built yet.
    """
