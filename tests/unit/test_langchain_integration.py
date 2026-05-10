"""Unit tests for the LangChain integration.

langchain packages are optional â€” mocked so tests run without them installed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _inject_fake_langchain():
    """Inject minimal fake langchain packages into sys.modules."""
    for pkg in [
        "langchain_google_genai",
        "langchain_anthropic",
        "langchain_openai",
        "langchain_core",
        "langchain_core.runnables",
    ]:
        if pkg not in sys.modules:
            fake = types.ModuleType(pkg)
            if pkg == "langchain_google_genai":
                fake.ChatGoogleGenerativeAI = MagicMock(name="ChatGoogleGenerativeAI")
            elif pkg == "langchain_anthropic":
                fake.ChatAnthropic = MagicMock(name="ChatAnthropic")
            elif pkg == "langchain_openai":
                fake.ChatOpenAI = MagicMock(name="ChatOpenAI")
            sys.modules[pkg] = fake


@pytest.fixture(autouse=True)
def fake_langchain_pkgs():
    _inject_fake_langchain()
    yield


def _mock_decision(tier="medium", model="gemini-2.5-flash", task_type="reasoning",
                   complexity="standard", confidence=0.85):
    d = MagicMock()
    d.tier.value = tier
    d.task_type.value = task_type
    d.complexity.value = complexity
    d.confidence = confidence
    d.model_name = model
    return d


def test_get_chat_model_google():
    from classifier.integrations.langchain import get_chat_model
    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="gemini-2.5-flash")
        llm = get_chat_model("Write a short summary", provider="google")  # noqa: F841
        import langchain_google_genai
        langchain_google_genai.ChatGoogleGenerativeAI.assert_called_once_with(
            model="gemini-2.5-flash"
        )


def test_get_chat_model_anthropic():
    from classifier.integrations.langchain import get_chat_model
    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="claude-opus-4-7", tier="high")
        llm = get_chat_model("Complex analysis", provider="anthropic")  # noqa: F841
        import langchain_anthropic
        langchain_anthropic.ChatAnthropic.assert_called_once_with(model="claude-opus-4-7")


def test_get_chat_model_openai():
    from classifier.integrations.langchain import get_chat_model
    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="gpt-4o", tier="medium")
        llm = get_chat_model("Translate to Spanish", provider="openai")  # noqa: F841
        import langchain_openai
        langchain_openai.ChatOpenAI.assert_called_once_with(model="gpt-4o")


def test_get_chat_model_uses_fallback_on_error():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.langchain import get_chat_model
    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        llm = get_chat_model("task", provider="google", fallback_model="gemini-2.5-flash")  # noqa: F841
        import langchain_google_genai
        langchain_google_genai.ChatGoogleGenerativeAI.assert_called_with(
            model="gemini-2.5-flash"
        )


def test_get_chat_model_raises_without_fallback():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.langchain import get_chat_model
    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        with pytest.raises(ClassificationError):
            get_chat_model("task", provider="google", fallback_model=None)


def test_get_chat_model_unsupported_provider():
    from classifier.integrations.langchain import get_chat_model
    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision()
        with pytest.raises(ValueError, match="not supported"):
            get_chat_model("task", provider="unknown_provider")


def test_dynamic_chat_model_invoke():
    from classifier.integrations.langchain import DynamicChatModel
    inner_llm = MagicMock()
    inner_llm.invoke.return_value = "response"
    with patch("classifier.integrations.langchain._build_chat_model", return_value=inner_llm) as mock_build:
        with patch("classifier.classify_task", return_value=_mock_decision()):
            llm = DynamicChatModel(provider="google", report_outcomes=False)
            result = llm.invoke("What is 2+2?")

            assert result == "response"
            mock_build.assert_called_once()
            inner_llm.invoke.assert_called_once()


def test_dynamic_chat_model_extract_text_string():
    from classifier.integrations.langchain import DynamicChatModel
    assert DynamicChatModel._extract_text("plain string") == "plain string"


def test_dynamic_chat_model_extract_text_messages():
    from classifier.integrations.langchain import DynamicChatModel
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "What time is it?"},
    ]
    assert DynamicChatModel._extract_text(msgs) == "What time is it?"


def test_dynamic_chat_model_extract_text_object():
    from classifier.integrations.langchain import DynamicChatModel
    msg = MagicMock()
    msg.content = "Message content here"
    assert DynamicChatModel._extract_text(msg) == "Message content here"


def test_dynamic_chat_model_batch():
    from classifier.integrations.langchain import DynamicChatModel
    inner_llm = MagicMock()
    inner_llm.invoke.side_effect = ["r1", "r2", "r3"]
    with patch("classifier.integrations.langchain._build_chat_model", return_value=inner_llm):
        with patch("classifier.classify_task", return_value=_mock_decision()):
            llm = DynamicChatModel(provider="google", report_outcomes=False)
            results = llm.batch(["task1", "task2", "task3"])

            assert results == ["r1", "r2", "r3"]
