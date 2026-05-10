"""Anthropic Claude L2 caller — JSON mode via tool-use trick."""

from __future__ import annotations

import json
from types import SimpleNamespace

from classifier.infra.config import settings


def call(task: str, history, model: str, schema):
    """Call Claude with a forced-tool to extract structured JSON output."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool = {
        "name": "classify_task",
        "description": "Classify the task into structured fields.",
        "input_schema": schema,
    }
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0.0,
        messages=[{"role": "user", "content": task}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "classify_task"},
    )
    # Extract tool_use input as JSON string for unified downstream parsing
    text = "{}"
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            text = json.dumps(block.input)
            break
    usage = SimpleNamespace(
        prompt_token_count=getattr(resp.usage, "input_tokens", 0),
        candidates_token_count=getattr(resp.usage, "output_tokens", 0),
    )
    return SimpleNamespace(text=text, usage_metadata=usage)
