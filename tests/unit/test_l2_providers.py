"""Tests for the lazy-loaded L2 provider callers (anthropic, openai).

These cover the alternate L2 backends that fire when Router(layer2_provider=...)
selects a non-default provider. They were previously at 0% coverage.
"""

import json
from unittest.mock import MagicMock, patch


def _schema():
    return {"type": "object", "properties": {"task_type": {"type": "string"}}}


def test_anthropic_caller_extracts_tool_use_json():
    fake_client = MagicMock()
    fake_block = MagicMock()
    fake_block.type = "tool_use"
    fake_block.input = {"task_type": "reasoning", "complexity": "complex"}
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    fake_resp.usage.input_tokens = 42
    fake_resp.usage.output_tokens = 7
    fake_client.messages.create.return_value = fake_resp

    with patch("anthropic.Anthropic", return_value=fake_client):
        from classifier.layers.layer2.providers.anthropic import call

        result = call("classify this", None, "claude-haiku-4-5-20251001", _schema())

    parsed = json.loads(result.text)
    assert parsed["task_type"] == "reasoning"
    assert result.usage_metadata.prompt_token_count == 42
    assert result.usage_metadata.candidates_token_count == 7
    fake_client.messages.create.assert_called_once()


def test_anthropic_caller_returns_empty_object_when_no_tool_use():
    fake_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"  # not tool_use → caller skips it
    fake_resp = MagicMock()
    fake_resp.content = [text_block]
    fake_resp.usage.input_tokens = 0
    fake_resp.usage.output_tokens = 0
    fake_client.messages.create.return_value = fake_resp

    with patch("anthropic.Anthropic", return_value=fake_client):
        from classifier.layers.layer2.providers.anthropic import call

        result = call("task", None, "claude-haiku-4-5-20251001", _schema())

    assert result.text == "{}"


def test_openai_caller_extracts_json_response():
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = '{"task_type": "code_creation"}'
    fake_resp.usage.prompt_tokens = 30
    fake_resp.usage.completion_tokens = 12
    fake_client.chat.completions.create.return_value = fake_resp

    with patch("openai.OpenAI", return_value=fake_client):
        from classifier.layers.layer2.providers.openai import call

        result = call("write a function", None, "gpt-4o-mini", _schema())

    parsed = json.loads(result.text)
    assert parsed["task_type"] == "code_creation"
    assert result.usage_metadata.prompt_token_count == 30
    assert result.usage_metadata.candidates_token_count == 12

    # Verify schema was passed in the system prompt
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert "Schema:" in call_kwargs["messages"][0]["content"]


def test_openai_caller_handles_null_content():
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = None  # OpenAI sometimes returns None
    fake_resp.usage.prompt_tokens = 0
    fake_resp.usage.completion_tokens = 0
    fake_client.chat.completions.create.return_value = fake_resp

    with patch("openai.OpenAI", return_value=fake_client):
        from classifier.layers.layer2.providers.openai import call

        result = call("task", None, "gpt-4o-mini", _schema())

    assert result.text == "{}"
