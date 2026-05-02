"""LangChain integration — dynamic model routing for LangChain / LangGraph agents.

Install with:
    pip install langchain-google-genai   # or langchain-anthropic / langchain-openai

Two patterns:

1. **`get_chat_model(task)`** — returns the right ChatModel for a single task.

    from classifier.integrations.langchain import get_chat_model
    llm = get_chat_model("Summarise this 10-page contract")
    chain = llm | StrOutputParser()

2. **`DynamicChatModel`** — a LangChain-compatible BaseChatModel that classifies
   each prompt on `.invoke()` and dispatches to the right underlying model.

    from classifier.integrations.langchain import DynamicChatModel
    llm = DynamicChatModel(provider="google")
    chain = llm | StrOutputParser()
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Provider → langchain package + class name
_PROVIDER_MAP = {
    "google":    ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "anthropic": ("langchain_anthropic",    "ChatAnthropic"),
    "openai":    ("langchain_openai",       "ChatOpenAI"),
}


def _build_chat_model(model_name: str, provider: str, **kwargs) -> Any:
    """Instantiate the correct LangChain ChatModel for a provider/model pair."""
    if provider not in _PROVIDER_MAP:
        raise ValueError(
            f"Provider '{provider}' not supported by LangChain integration. "
            f"Choose from: {sorted(_PROVIDER_MAP)}"
        )
    pkg, cls_name = _PROVIDER_MAP[provider]
    try:
        import importlib
        mod = importlib.import_module(pkg)
        cls = getattr(mod, cls_name)
    except ImportError as exc:
        raise ImportError(
            f"LangChain provider package '{pkg}' is not installed. "
            f"Install with: pip install {pkg}"
        ) from exc

    return cls(model=model_name, **kwargs)


def get_chat_model(
    task: str,
    *,
    provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
    **model_kwargs: Any,
) -> Any:
    """Classify a task and return the appropriate LangChain ChatModel.

    Args:
        task:           Task text to classify.
        provider:       "google" | "anthropic" | "openai". Defaults to DEFAULT_PROVIDER.
        fallback_model: Model name used only on classification failure.
        **model_kwargs: Passed through to the underlying LangChain ChatModel constructor.

    Returns:
        A LangChain ChatModel (ChatGoogleGenerativeAI / ChatAnthropic / ChatOpenAI).

    Example:
        llm = get_chat_model("Explain CRISPR gene editing for a 5th grader")
        response = llm.invoke("Explain CRISPR gene editing for a 5th grader")
    """
    from classifier import classify_task
    from classifier.core.exceptions import ClassificationError
    from classifier.infra.config import settings

    resolved_provider = provider or settings.default_provider

    try:
        decision = classify_task(task, provider=resolved_provider)
        model_name = decision.model_name
        logger.info(
            "LangChain: routed [%s | %s/%s | conf=%.2f] → %s",
            decision.tier.value.upper(),
            decision.task_type.value, decision.complexity.value,
            decision.confidence, model_name,
        )
    except ClassificationError as exc:
        if fallback_model:
            logger.warning("LangChain: classification failed (%s) — using fallback %s",
                           exc, fallback_model)
            model_name = fallback_model
        else:
            raise

    return _build_chat_model(model_name, resolved_provider, **model_kwargs)


class DynamicChatModel:
    """LangChain-compatible chat model that classifies each prompt on invoke().

    Wraps any LangChain ChatModel. On each call it runs the router, picks the
    right model, builds a ChatModel, and forwards the call.

    Usage:
        from classifier.integrations.langchain import DynamicChatModel
        from langchain_core.messages import HumanMessage

        llm = DynamicChatModel(provider="anthropic")
        response = llm.invoke([HumanMessage(content="Write a Python binary search")])

    Works with LangChain Expression Language (LCEL):
        chain = DynamicChatModel() | StrOutputParser()
        chain.invoke("Translate to French: Hello world")
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        self._provider = provider
        self._fallback_model = fallback_model
        self._model_kwargs = model_kwargs

    def _get_llm(self, task_text: str) -> Any:
        return get_chat_model(
            task_text,
            provider=self._provider,
            fallback_model=self._fallback_model,
            **self._model_kwargs,
        )

    def invoke(self, input, **kwargs) -> Any:
        task_text = self._extract_text(input)
        return self._get_llm(task_text).invoke(input, **kwargs)

    def stream(self, input, **kwargs) -> Iterator:
        task_text = self._extract_text(input)
        yield from self._get_llm(task_text).stream(input, **kwargs)

    def batch(self, inputs: List, **kwargs) -> List:
        return [self.invoke(inp, **kwargs) for inp in inputs]

    # LCEL pipe operator
    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda, RunnableSequence
        return RunnableSequence(RunnableLambda(self.invoke), other)

    @staticmethod
    def _extract_text(input) -> str:
        """Pull task text from a string, list of messages, or HumanMessage."""
        if isinstance(input, str):
            return input
        if isinstance(input, list):
            for msg in reversed(input):
                # LangChain message objects
                if hasattr(msg, "content"):
                    return str(msg.content)
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", ""))
            return str(input[-1]) if input else ""
        if hasattr(input, "content"):
            return str(input.content)
        return str(input)
