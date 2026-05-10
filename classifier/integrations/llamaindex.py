"""LlamaIndex integration — dynamic model routing for LlamaIndex agents/queries.

Install with:
    pip install llama-index-llms-google-genai     # or -anthropic / -openai

Two patterns:

1. **`get_llm(task)`** — returns the right LlamaIndex LLM for a single task.

    from classifier.integrations.llamaindex import get_llm
    llm = get_llm("Summarise this 10-page contract")
    response = llm.complete("Summarise this 10-page contract")

2. **`DynamicLLM`** — a LlamaIndex-compatible LLM that classifies each prompt
   on `.complete()` / `.chat()` and dispatches to the right underlying model.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Provider → (package, class name) for the appropriate LlamaIndex LLM
_PROVIDER_MAP = {
    "google": ("llama_index.llms.google_genai", "GoogleGenAI"),
    "anthropic": ("llama_index.llms.anthropic", "Anthropic"),
    "openai": ("llama_index.llms.openai", "OpenAI"),
}


def _build_llm(model_name: str, provider: str, **kwargs: Any) -> Any:
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Provider '{provider}' not supported by LlamaIndex integration. "
            f"Choose from: {sorted(_PROVIDER_MAP)}"
        )
    pkg, cls_name = _PROVIDER_MAP[provider]
    try:
        mod = importlib.import_module(pkg)
        cls = getattr(mod, cls_name)
    except ImportError as exc:
        raise ImportError(
            f"LlamaIndex provider package '{pkg}' is not installed. "
            f"Install with: pip install {pkg.replace('_', '-')}"
        ) from exc
    return cls(model=model_name, **kwargs)


def get_llm(
    task: str,
    *,
    provider: str | None = None,
    fallback_model: str | None = None,
    **llm_kwargs: Any,
) -> Any:
    """Classify a task and return the appropriate LlamaIndex LLM instance."""
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved_provider = provider or settings.default_provider
    try:
        decision = classify_task(task, provider=resolved_provider)
        model_name = decision.model_name
        logger.info(
            "LlamaIndex: routed [%s | %s/%s | conf=%.2f] -> %s",
            decision.tier.value.upper(),
            decision.task_type.value,
            decision.complexity.value,
            decision.confidence,
            model_name,
        )
    except ClassificationError:
        if not fallback_model:
            raise
        model_name = fallback_model

    return _build_llm(model_name, resolved_provider, **llm_kwargs)


class DynamicLLM:
    """LlamaIndex-compatible LLM that classifies each call's prompt.

    Usage:
        from classifier.integrations.llamaindex import DynamicLLM
        from llama_index.core import Settings

        Settings.llm = DynamicLLM(provider="google")

    The wrapper picks an underlying LlamaIndex LLM per call. Methods are
    forwarded to whichever LLM was selected for the prompt.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        fallback_model: str | None = None,
        **llm_kwargs: Any,
    ) -> None:
        self._provider = provider
        self._fallback_model = fallback_model
        self._llm_kwargs = llm_kwargs

    def _llm_for(self, prompt: str) -> Any:
        return get_llm(
            prompt,
            provider=self._provider,
            fallback_model=self._fallback_model,
            **self._llm_kwargs,
        )

    @staticmethod
    def _as_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            for msg in reversed(prompt):
                content = getattr(msg, "content", None) or (
                    msg.get("content") if isinstance(msg, dict) else None
                )
                if content:
                    return str(content)
        return str(prompt)

    def complete(self, prompt, **kwargs):
        return self._llm_for(self._as_text(prompt)).complete(prompt, **kwargs)

    def chat(self, messages, **kwargs):
        return self._llm_for(self._as_text(messages)).chat(messages, **kwargs)

    def stream_complete(self, prompt, **kwargs):
        return self._llm_for(self._as_text(prompt)).stream_complete(prompt, **kwargs)

    async def acomplete(self, prompt, **kwargs):
        return await self._llm_for(self._as_text(prompt)).acomplete(prompt, **kwargs)

    async def achat(self, messages, **kwargs):
        return await self._llm_for(self._as_text(messages)).achat(messages, **kwargs)
