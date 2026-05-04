"""Microsoft Semantic Kernel integration — dynamic model routing for SK Kernels.

Install with:
    pip install semantic-kernel
    # plus provider connectors as needed

Pattern:

**`get_chat_service(task)`** — returns the right Semantic Kernel chat service.

    import semantic_kernel as sk
    from classifier.integrations.semantic_kernel import get_chat_service

    kernel = sk.Kernel()
    kernel.add_service(get_chat_service("Summarise this contract"))
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Provider -> (package, class name)
_PROVIDER_MAP = {
    "google":    ("semantic_kernel.connectors.ai.google.google_ai", "GoogleAIChatCompletion"),
    "anthropic": ("semantic_kernel.connectors.ai.anthropic",        "AnthropicChatCompletion"),
    "openai":    ("semantic_kernel.connectors.ai.open_ai",          "OpenAIChatCompletion"),
}


def _build_service(model_name: str, provider: str, **kwargs: Any) -> Any:
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Provider '{provider}' not supported by Semantic Kernel integration. "
            f"Choose from: {sorted(_PROVIDER_MAP)}"
        )
    pkg, cls_name = _PROVIDER_MAP[provider]
    try:
        mod = importlib.import_module(pkg)
        cls = getattr(mod, cls_name)
    except ImportError as exc:
        raise ImportError(
            f"Semantic Kernel provider connector '{pkg}' is not installed. "
            f"Install: pip install semantic-kernel"
        ) from exc

    # Each SK service uses different kwarg names — try both common patterns
    try:
        return cls(ai_model_id=model_name, **kwargs)
    except TypeError:
        return cls(model=model_name, **kwargs)


def get_chat_service(
    task: str,
    *,
    provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
    **service_kwargs: Any,
) -> Any:
    """Classify a task and return the Semantic Kernel chat service pinned to the routed model."""
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved_provider = provider or settings.default_provider
    try:
        decision = classify_task(task, provider=resolved_provider)
        model_name = decision.model_name
        logger.info(
            "SemanticKernel: routed [%s | %s/%s] -> %s",
            decision.tier.value.upper(),
            decision.task_type.value, decision.complexity.value, model_name,
        )
    except ClassificationError:
        if not fallback_model:
            raise
        model_name = fallback_model

    return _build_service(model_name, resolved_provider, **service_kwargs)
