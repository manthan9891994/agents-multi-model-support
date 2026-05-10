"""Unit tests for the CrewAI integration.

CrewAI is an optional dependency — these tests mock the `crewai` module so
they run without it installed.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def fake_crewai():
    """Inject a fake `crewai` module into sys.modules for tests that need it."""
    if "crewai" in sys.modules:
        # Real CrewAI is installed — leave it alone
        yield
        return

    fake = types.ModuleType("crewai")
    fake.LLM = MagicMock(spec=["__call__"], return_value=MagicMock(name="LLMInstance"))
    sys.modules["crewai"] = fake
    yield
    del sys.modules["crewai"]


def test_pick_llm_returns_correct_provider_prefix():
    """pick_llm_for_task should call crewai.LLM with provider-prefixed model name."""
    from classifier.integrations.crewai import pick_llm_for_task

    with patch("classifier.classify_task") as mock_classify:
        decision = MagicMock()
        decision.model_name = "gemini-2.5-flash"
        decision.tier.value = "medium"
        decision.task_type.value = "reasoning"
        decision.complexity.value = "standard"
        decision.confidence = 0.85
        mock_classify.return_value = decision

        pick_llm_for_task("test task", provider="google")

        # crewai.LLM should have been called with provider-prefixed model
        import crewai

        crewai.LLM.assert_called_once()
        call_kwargs = crewai.LLM.call_args.kwargs
        assert "gemini/" in call_kwargs.get("model", "")


def test_pick_llm_anthropic_prefix():
    from classifier.integrations.crewai import pick_llm_for_task

    with patch("classifier.classify_task") as mock_classify:
        decision = MagicMock()
        decision.model_name = "claude-opus-4-7"
        decision.tier.value = "high"
        decision.task_type.value = "reasoning"
        decision.complexity.value = "complex"
        decision.confidence = 0.92
        mock_classify.return_value = decision

        pick_llm_for_task("complex task", provider="anthropic")

        import crewai

        call_kwargs = crewai.LLM.call_args.kwargs
        assert "anthropic/" in call_kwargs.get("model", "")


def test_pick_llm_uses_fallback_on_classification_error():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.crewai import pick_llm_for_task

    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        pick_llm_for_task("task", provider="google", fallback_model="gemini-2.5-flash-lite")
        import crewai

        call_kwargs = crewai.LLM.call_args.kwargs
        assert "gemini" in call_kwargs.get("model", "")


def test_pick_llm_raises_without_fallback():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.crewai import pick_llm_for_task

    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        with pytest.raises(ClassificationError):
            pick_llm_for_task("task", provider="google", fallback_model=None)


def test_dynamic_llm_extracts_user_message():
    from classifier.integrations.crewai import DynamicLLM

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert DynamicLLM._extract_task_text(messages) == "What is 2+2?"


def test_dynamic_llm_extracts_from_string():
    from classifier.integrations.crewai import DynamicLLM

    assert DynamicLLM._extract_task_text("plain string") == "plain string"


def test_dynamic_llm_handles_empty_messages():
    from classifier.integrations.crewai import DynamicLLM

    assert DynamicLLM._extract_task_text([]) == "[]"


def test_qualify_model_idempotent():
    """If already prefixed, don't double-prefix."""
    from classifier.integrations.crewai import _qualify_model

    assert _qualify_model("gemini/gemini-2.5-flash", "google") == "gemini/gemini-2.5-flash"


def test_qualify_model_adds_prefix():
    from classifier.integrations.crewai import _qualify_model

    assert _qualify_model("gemini-2.5-flash", "google") == "gemini/gemini-2.5-flash"
    assert _qualify_model("gpt-4o", "openai") == "openai/gpt-4o"
