"""Unit tests for the AutoGen / OpenAI Agents SDK integration."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_decision(
    tier="medium", model="gpt-4o", task_type="reasoning", complexity="standard", confidence=0.85
):
    d = MagicMock()
    d.tier.value = tier
    d.task_type.value = task_type
    d.complexity.value = complexity
    d.confidence = confidence
    d.model_name = model
    return d


def test_get_autogen_llm_config_returns_config_list():
    from classifier.integrations.autogen import get_autogen_llm_config

    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="gpt-4o")
        config = get_autogen_llm_config("Analyse revenue trends", provider="openai")
        assert "config_list" in config
        assert config["config_list"][0]["model"] == "gpt-4o"
        assert config["config_list"][0]["api_type"] == "openai"


def test_get_autogen_llm_config_anthropic():
    from classifier.integrations.autogen import get_autogen_llm_config

    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="claude-opus-4-7", tier="high")
        config = get_autogen_llm_config("Complex legal analysis", provider="anthropic")
        assert config["config_list"][0]["model"] == "claude-opus-4-7"
        assert config["config_list"][0]["api_type"] == "anthropic"


def test_get_autogen_llm_config_extra_config():
    from classifier.integrations.autogen import get_autogen_llm_config

    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision()
        config = get_autogen_llm_config(
            "task", provider="openai", extra_config={"temperature": 0, "timeout": 30}
        )
        assert config["temperature"] == 0
        assert config["timeout"] == 30


def test_get_autogen_llm_config_fallback_on_error():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.autogen import get_autogen_llm_config

    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        config = get_autogen_llm_config("task", provider="openai", fallback_model="gpt-4o-mini")
        assert config["config_list"][0]["model"] == "gpt-4o-mini"


def test_get_autogen_llm_config_raises_without_fallback():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.autogen import get_autogen_llm_config

    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        with pytest.raises(ClassificationError):
            get_autogen_llm_config("task", provider="openai")


def test_get_openai_agent_model_returns_string():
    from classifier.integrations.autogen import get_openai_agent_model

    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="gpt-4-turbo", tier="high")
        model = get_openai_agent_model("Build a compiler", provider="openai")
        assert model == "gpt-4-turbo"


def test_get_openai_agent_model_fallback():
    from classifier.core.exceptions import ClassificationError
    from classifier.integrations.autogen import get_openai_agent_model

    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        model = get_openai_agent_model("task", provider="openai", fallback_model="gpt-4o-mini")
        assert model == "gpt-4o-mini"


def test_dynamic_model_router_llm_config():
    from classifier.integrations.autogen import DynamicModelRouter

    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="gpt-4o")
        router = DynamicModelRouter(provider="openai")
        config = router.llm_config("Summarise this document")
        assert config["config_list"][0]["model"] == "gpt-4o"


def test_dynamic_model_router_model():
    from classifier.integrations.autogen import DynamicModelRouter

    with patch("classifier.classify_task") as mock_classify:
        mock_classify.return_value = _mock_decision(model="claude-sonnet-4-6")
        router = DynamicModelRouter(provider="anthropic")
        model = router.model("Write a blog post")
        assert model == "claude-sonnet-4-6"
