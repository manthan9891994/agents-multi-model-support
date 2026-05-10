"""OpenAI L2 caller — uses JSON mode."""

from __future__ import annotations

import json
from types import SimpleNamespace

from classifier.infra.config import settings


def call(task: str, history, model: str, schema):
    import openai

    client = openai.OpenAI(api_key=settings.openai_api_key)
    sys_prompt = (
        "You classify user tasks into structured fields. "
        "Respond with a JSON object that matches the provided schema. "
        f"Schema: {json.dumps(schema)}"
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=300,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task},
        ],
    )
    text = resp.choices[0].message.content or "{}"
    usage = SimpleNamespace(
        prompt_token_count=getattr(resp.usage, "prompt_tokens", 0),
        candidates_token_count=getattr(resp.usage, "completion_tokens", 0),
    )
    return SimpleNamespace(text=text, usage_metadata=usage)
