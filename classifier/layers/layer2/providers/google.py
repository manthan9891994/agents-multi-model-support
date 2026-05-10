"""Google Gemini L2 caller."""

from __future__ import annotations

from classifier.infra.config import settings


def call(task: str, history, model: str, schema):
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    cfg = genai.types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
        response_schema=schema,
    )
    return client.models.generate_content(model=model, contents=task, config=cfg)
