# Dynamic Model Selection for Google ADK

Automatically picks the right Gemini model (cheap → powerful) based on task
complexity — before the agent runs. One agent, zero framework changes.

---

## The Problem

ADK agents are hardcoded to one model. A simple "write a README" and a complex
"design a distributed system" both hit `gemini-2.5-pro` — wasting money.

## The Solution

Google ADK has a built-in hook — `before_model_callback` — that fires before
every LLM call. It receives the raw `LlmRequest` object. The Gemini client
uses `llm_request.model` for the actual API call, so mutating it in the
callback genuinely changes which model handles the request.

```
User message
  → ADK fires before_model_callback
  → Classifier reads llm_request.contents (user's text)
  → Picks model: gemini-2.5-flash-lite / gemini-2.5-flash / gemini-2.5-pro
  → llm_request.model = selected_model   ← mutated in place
  → Gemini API called with the new model
```

No ADK source changes. No pre-built agent variants. One agent.

---

## Structure

```
classifier/
  __init__.py     # classify_task(task, provider) → ClassificationDecision
  types.py        # ModelTier, TaskType, TaskComplexity, ClassificationDecision
  layer1.py       # Python keyword heuristic (<1ms, no API call)
  registry.py     # TIER_MATRIX + MODEL_REGISTRY (3 providers, 3 tiers)

examples_adk/
  agent.py        # ONE LlmAgent with before_model_callback
  run.py          # Demo through real ADK Runner
  __init__.py     # ADK discovery (exports root_agent)
  .env            # GOOGLE_API_KEY
```

---

## Model Tiers

| Tier   | Google Model          | When |
|--------|-----------------------|------|
| LOW    | gemini-2.5-flash-lite | Simple docs, trivial code, short answers |
| MEDIUM | gemini-2.5-flash      | Reasoning, standard APIs, analysis |
| HIGH   | gemini-2.5-pro        | Complex architecture, research, hard code |

### Decision Matrix

```
                 SIMPLE    STANDARD    COMPLEX    RESEARCH
REASONING        MEDIUM    MEDIUM      HIGH       HIGH
THINKING         MEDIUM    MEDIUM      HIGH       HIGH
ANALYZING        LOW       MEDIUM      HIGH       HIGH
CODE_CREATION    LOW       MEDIUM      HIGH       HIGH
DOC_CREATION     LOW       LOW         MEDIUM     MEDIUM
```

---

## Quick Start

```bash
# Install
pip install google-adk

# Set your key
echo "GOOGLE_API_KEY=your_key_here" > examples_adk/.env

# Run demo (see model selection printed before each API call)
adk run examples_adk

# Or web UI
adk web examples_adk
```

---

## How It Works

```python
# examples_adk/agent.py

def _dynamic_model_selector(callback_context, llm_request):
    task = ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            task = content.parts[0].text or ""
            break

    if task:
        decision = classify_task(task, provider="google")
        llm_request.model = decision.model_name   # the key line

    return None  # None = proceed with the (mutated) request


# ONE agent — model= is just the fallback
root_agent = LlmAgent(
    name="DynamicModelAgent",
    model="gemini-2.5-flash",                      # fallback only
    before_model_callback=_dynamic_model_selector,  # swaps model per request
    instruction="...",
)
```

What you see at runtime:

```
[classifier] task   : Write a README for this project
[classifier] model  : gemini-2.5-flash => gemini-2.5-flash-lite
[classifier] reason : doc_creation / simple => LOW

[classifier] task   : Design a distributed cache with LRU eviction...
[classifier] model  : gemini-2.5-flash => gemini-2.5-pro
[classifier] reason : thinking / research => HIGH
```

---

## Extending

**Add a provider** — update `MODEL_REGISTRY` in `classifier/registry.py`:
```python
"openai": {
    ModelTier.LOW:    "gpt-4o-mini",
    ModelTier.MEDIUM: "gpt-4o",
    ModelTier.HIGH:   "gpt-4-turbo",
}
```

**Add a smarter classification layer** — the classifier is layered:
- Layer 1 (built): Python keyword heuristic, <1ms
- Layer 2 (planned): Local Gemma via Ollama, 50-200ms
- Layer 3 (planned): Cheap API model (Haiku / GPT-4o-mini) for ambiguous tasks
- Layer 4 (planned): sklearn ML ensemble, enabled after 500+ logged samples

**CrewAI** — same classifier, swap the last line:
```python
from crewai import Agent
decision = classify_task(task, provider="openai")
agent = Agent(llm=decision.model_name, ...)  # llm=, not model=
```
