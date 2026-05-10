"""Unit tests for the 6 new integrations:
LlamaIndex, Pydantic AI, DSPy, Haystack, Semantic Kernel, smolagents.

Each framework is optional — tests mock the underlying SDK so they run on a
bare install.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _mock_decision(model="gemini-2.5-flash", tier="medium",
                   task_type="reasoning", complexity="standard", confidence=0.85):
    d = MagicMock()
    d.tier.value = tier
    d.task_type.value = task_type
    d.complexity.value = complexity
    d.confidence = confidence
    d.model_name = model
    return d


# ── Helper to inject fake framework modules into sys.modules ─────────────────

def _inject_fake_module(name: str, attrs: dict):
    """Create a fake module with given attrs and put in sys.modules."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
    mod = sys.modules[name]
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
#  LlamaIndex
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def llamaindex_fakes():
    _inject_fake_module("llama_index.llms.google_genai", {"GoogleGenAI": MagicMock(name="GoogleGenAI")})
    _inject_fake_module("llama_index.llms.anthropic",    {"Anthropic":   MagicMock(name="Anthropic")})
    _inject_fake_module("llama_index.llms.openai",       {"OpenAI":      MagicMock(name="OpenAI")})


def test_llamaindex_get_llm_google(llamaindex_fakes):
    from classifier.integrations.llamaindex import get_llm
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-flash")):
        llm = get_llm("test", provider="google")  # noqa: F841
        import llama_index.llms.google_genai as g
        g.GoogleGenAI.assert_called_once_with(model="gemini-2.5-flash")


def test_llamaindex_get_llm_anthropic(llamaindex_fakes):
    from classifier.integrations.llamaindex import get_llm
    with patch("classifier.classify_task", return_value=_mock_decision(model="claude-opus-4-7", tier="high")):
        llm = get_llm("test", provider="anthropic")  # noqa: F841
        import llama_index.llms.anthropic as a
        a.Anthropic.assert_called_once_with(model="claude-opus-4-7")


def test_llamaindex_dynamic_llm(llamaindex_fakes):
    from classifier.integrations.llamaindex import DynamicLLM
    with patch("classifier.integrations.llamaindex.get_llm") as mock_get:
        inner = MagicMock()
        inner.complete.return_value = "ok"
        mock_get.return_value = inner

        llm = DynamicLLM(provider="google")
        result = llm.complete("Hello")
        assert result == "ok"
        mock_get.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic AI
# ─────────────────────────────────────────────────────────────────────────────

def test_pydantic_ai_qualify():
    from classifier.integrations.pydantic_ai import _qualify
    assert _qualify("gemini-2.5-flash", "google")    == "google-gla:gemini-2.5-flash"
    assert _qualify("claude-opus-4-7",  "anthropic") == "anthropic:claude-opus-4-7"
    assert _qualify("gpt-4o",           "openai")    == "openai:gpt-4o"
    assert _qualify("openai:gpt-4o",    "openai")    == "openai:gpt-4o"     # already qualified


def test_pydantic_ai_get_model_string():
    from classifier.integrations.pydantic_ai import get_model_string
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-pro", tier="high")):
        s = get_model_string("Hard reasoning task", provider="google")
        assert s == "google-gla:gemini-2.5-pro"


def test_pydantic_ai_get_agent():
    """get_agent returns an Agent constructed with the qualified model."""
    fake_agent_cls = MagicMock(name="Agent")
    _inject_fake_module("pydantic_ai", {"Agent": fake_agent_cls})

    from classifier.integrations.pydantic_ai import get_agent
    with patch("classifier.classify_task", return_value=_mock_decision(model="claude-opus-4-7", tier="high")):
        agent = get_agent("Complex task", provider="anthropic", system_prompt="You are helpful")  # noqa: F841
        fake_agent_cls.assert_called_once_with("anthropic:claude-opus-4-7", system_prompt="You are helpful")


# ─────────────────────────────────────────────────────────────────────────────
#  DSPy
# ─────────────────────────────────────────────────────────────────────────────

def test_dspy_qualify():
    from classifier.integrations.dspy import _qualify
    assert _qualify("gemini-2.5-flash", "google")    == "gemini/gemini-2.5-flash"
    assert _qualify("claude-opus-4-7",  "anthropic") == "anthropic/claude-opus-4-7"
    assert _qualify("gemini/gemini-2.5-flash", "google") == "gemini/gemini-2.5-flash"


def test_dspy_get_model_string():
    from classifier.integrations.dspy import get_model_string
    with patch("classifier.classify_task", return_value=_mock_decision(model="gpt-4o", tier="medium")):
        s = get_model_string("test", provider="openai")
        assert s == "openai/gpt-4o"


def test_dspy_get_lm():
    fake_lm_cls = MagicMock(name="LM")
    fake_dspy = types.ModuleType("dspy")
    fake_dspy.LM = fake_lm_cls
    fake_dspy.settings = MagicMock(lm=None)
    fake_dspy.configure = MagicMock()
    sys.modules["dspy"] = fake_dspy

    from classifier.integrations.dspy import get_lm
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-flash")):
        lm = get_lm("test", provider="google")  # noqa: F841
        fake_lm_cls.assert_called_once_with("gemini/gemini-2.5-flash")


def test_dspy_route_context_manager():
    fake_lm_cls = MagicMock(name="LM", return_value=MagicMock(name="LMInstance"))
    fake_dspy = types.ModuleType("dspy")
    fake_dspy.LM = fake_lm_cls
    fake_dspy.settings = MagicMock()
    fake_dspy.settings.lm = "previous_lm"
    fake_dspy.configure = MagicMock()
    sys.modules["dspy"] = fake_dspy

    from classifier.integrations.dspy import route
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-pro", tier="high")):
        with route("Hard task", provider="google"):
            # Inside the block, configure was called with the new LM
            assert fake_dspy.configure.called
        # On exit, the previous LM is restored
        last_call = fake_dspy.configure.call_args_list[-1]
        assert last_call.kwargs.get("lm") == "previous_lm"


# ─────────────────────────────────────────────────────────────────────────────
#  Haystack
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def haystack_fakes():
    _inject_fake_module(
        "haystack_integrations.components.generators.google_genai",
        {"GoogleGenAIGenerator": MagicMock(name="GoogleGenAIGenerator")},
    )
    _inject_fake_module(
        "haystack_integrations.components.generators.anthropic",
        {"AnthropicGenerator": MagicMock(name="AnthropicGenerator")},
    )
    _inject_fake_module(
        "haystack.components.generators",
        {"OpenAIGenerator": MagicMock(name="OpenAIGenerator")},
    )


def test_haystack_get_generator_google(haystack_fakes):
    from classifier.integrations.haystack import get_generator
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-flash")):
        gen = get_generator("test", provider="google")  # noqa: F841
        from haystack_integrations.components.generators import google_genai as g
        g.GoogleGenAIGenerator.assert_called_once_with(model="gemini-2.5-flash")


def test_haystack_get_generator_openai(haystack_fakes):
    from classifier.integrations.haystack import get_generator
    with patch("classifier.classify_task", return_value=_mock_decision(model="gpt-4o", tier="medium")):
        gen = get_generator("test", provider="openai")  # noqa: F841
        from haystack.components import generators as g
        g.OpenAIGenerator.assert_called_once_with(model="gpt-4o")


def test_haystack_unsupported_provider_raises(haystack_fakes):
    from classifier.integrations.haystack import get_generator
    with patch("classifier.classify_task", return_value=_mock_decision()):
        with pytest.raises(ValueError, match="not supported"):
            get_generator("test", provider="unknown_provider")


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic Kernel
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sk_fakes():
    _inject_fake_module(
        "semantic_kernel.connectors.ai.google.google_ai",
        {"GoogleAIChatCompletion": MagicMock(name="GoogleAIChatCompletion")},
    )
    _inject_fake_module(
        "semantic_kernel.connectors.ai.anthropic",
        {"AnthropicChatCompletion": MagicMock(name="AnthropicChatCompletion")},
    )
    _inject_fake_module(
        "semantic_kernel.connectors.ai.open_ai",
        {"OpenAIChatCompletion": MagicMock(name="OpenAIChatCompletion")},
    )


def test_semantic_kernel_get_chat_service_google(sk_fakes):
    from classifier.integrations.semantic_kernel import get_chat_service
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-flash")):
        svc = get_chat_service("test", provider="google")  # noqa: F841
        from semantic_kernel.connectors.ai.google import google_ai as g
        g.GoogleAIChatCompletion.assert_called_once()


def test_semantic_kernel_get_chat_service_anthropic(sk_fakes):
    from classifier.integrations.semantic_kernel import get_chat_service
    with patch("classifier.classify_task", return_value=_mock_decision(model="claude-opus-4-7", tier="high")):
        svc = get_chat_service("test", provider="anthropic")  # noqa: F841
        from semantic_kernel.connectors.ai import anthropic as a
        a.AnthropicChatCompletion.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  smolagents
# ─────────────────────────────────────────────────────────────────────────────

def test_smolagents_qualify():
    from classifier.integrations.smolagents import _qualify
    assert _qualify("gemini-2.5-flash", "google")    == "gemini/gemini-2.5-flash"
    assert _qualify("claude-opus-4-7",  "anthropic") == "anthropic/claude-opus-4-7"


def test_smolagents_get_model():
    fake_lite_llm_cls = MagicMock(name="LiteLLMModel")
    _inject_fake_module("smolagents", {"LiteLLMModel": fake_lite_llm_cls})

    from classifier.integrations.smolagents import get_model
    with patch("classifier.classify_task", return_value=_mock_decision(model="gemini-2.5-flash")):
        model = get_model("test", provider="google")  # noqa: F841
        fake_lite_llm_cls.assert_called_once_with(model_id="gemini/gemini-2.5-flash")


def test_smolagents_dynamic_model_extracts_text():
    from classifier.integrations.smolagents import DynamicModel
    assert DynamicModel._as_text("plain") == "plain"
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    assert DynamicModel._as_text(msgs) == "What is 2+2?"
    blocks = [{"role": "user", "content": [{"type": "text", "text": "Hello block"}]}]
    assert DynamicModel._as_text(blocks) == "Hello block"


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-cutting: each integration honors fallback_model
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_path,fn,fake_setup", [
    ("classifier.integrations.llamaindex",       "get_llm",          "llamaindex"),
    ("classifier.integrations.pydantic_ai",      "get_model_string", "pydantic_ai"),
    ("classifier.integrations.dspy",             "get_model_string", "dspy"),
    ("classifier.integrations.haystack",         "get_generator",    "haystack"),
    ("classifier.integrations.semantic_kernel",  "get_chat_service", "sk"),
    ("classifier.integrations.smolagents",       "get_model",        "smolagents"),
])
def test_fallback_model_used_on_classification_error(module_path, fn, fake_setup):
    """When classification raises, fallback_model is used."""
    import importlib

    from classifier.core.exceptions import ClassificationError

    # Provide framework fakes for those paths that need them
    if fake_setup == "llamaindex":
        _inject_fake_module("llama_index.llms.google_genai", {"GoogleGenAI": MagicMock()})
    elif fake_setup == "haystack":
        _inject_fake_module(
            "haystack_integrations.components.generators.google_genai",
            {"GoogleGenAIGenerator": MagicMock()},
        )
    elif fake_setup == "sk":
        _inject_fake_module(
            "semantic_kernel.connectors.ai.google.google_ai",
            {"GoogleAIChatCompletion": MagicMock()},
        )
    elif fake_setup == "smolagents":
        _inject_fake_module("smolagents", {"LiteLLMModel": MagicMock()})

    mod = importlib.import_module(module_path)
    func = getattr(mod, fn)

    with patch("classifier.classify_task", side_effect=ClassificationError("boom")):
        # Should not raise — falls back
        result = func("task", provider="google", fallback_model="gemini-2.5-flash")
        assert result is not None
