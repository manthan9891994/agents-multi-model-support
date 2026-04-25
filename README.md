# Dynamic Model Selection — Multi-Model Task Router

Automatically picks the right model (cheap → powerful) **before the agent runs**,
based on task complexity, domain, and intent. One agent. Zero framework changes.

Supports Google ADK, with CrewAI and direct-API usage also possible.

---

## The Problem

Agents are hardcoded to one model. A simple "write a README" and a complex
"design a distributed system" both hit `gemini-2.5-pro` — wasting money and
adding latency where it isn't needed.

## The Solution

Classify the task **before** the model call. Route to the cheapest model that
can handle it well.

```
User message
  → Layer 1: keyword + heuristic (<1ms, no API call)
      └─ conf < 0.75 → Layer 2: Gemini Flash Lite (~2s, $0.00001/call)
  → Picks: LOW / MEDIUM / HIGH tier
  → Maps tier to model name for the chosen provider
  → ADK: llm_request.model = selected_model  (mutated before API call)
```

No ADK source changes. No pre-built agent variants. One agent.

---

## Project Structure

```
agents_multi_model_support/
│
├── classifier/                    # Core package
│   ├── __init__.py               # Public API: classify_task()
│   ├── core/
│   │   ├── types.py              # ModelTier, TaskType, TaskComplexity, ClassificationDecision
│   │   ├── registry.py           # TIER_MATRIX + MODEL_REGISTRY (3 providers × 3 tiers)
│   │   └── exceptions.py        # ClassifierError hierarchy
│   ├── layers/
│   │   ├── layer1.py             # Keyword + heuristic classifier (<1ms, no API call)
│   │   └── layer2.py             # LLM classifier via Gemini Flash Lite (~2s, only on low-conf)
│   ├── infra/
│   │   ├── config.py             # Pydantic Settings (.env)
│   │   ├── cache.py              # LRU + TTL classification cache
│   │   ├── cost_tracker.py       # Budget tracking + downgrade signals
│   │   └── decision_logger.py    # JSONL routing decision log
│   └── data/
│       └── reference_tasks.jsonl # Labeled examples for future Layer 3
│
├── integrations/
│   └── adk/
│       ├── agent.py              # LlmAgent with before_model_callback
│       └── run.py                # Demo runner
│
├── tests/
│   ├── unit/                     # test_layer1, test_cache, test_cost_tracker
│   └── integration/              # test_classifier (full pipeline)
│
├── .env.example
└── requirements.txt
```

---

## Model Tiers

| Tier | Google | Anthropic | OpenAI | When |
|------|--------|-----------|--------|------|
| LOW | gemini-2.5-flash-lite | claude-haiku-4-5 | gpt-4o-mini | Simple docs, trivial code, conversation |
| MEDIUM | gemini-2.5-flash | claude-sonnet-4-6 | gpt-4o | Reasoning, standard APIs, analysis |
| HIGH | gemini-2.5-pro | claude-opus-4-7 | gpt-4-turbo | Complex architecture, research, hard code |

### Decision Matrix

```
                  SIMPLE    STANDARD    COMPLEX    RESEARCH
REASONING         MEDIUM    MEDIUM      HIGH       HIGH
THINKING          MEDIUM    MEDIUM      HIGH       HIGH
ANALYZING         LOW       MEDIUM      HIGH       HIGH
CODE_CREATION     LOW       MEDIUM      HIGH       HIGH
DOC_CREATION      LOW       LOW         MEDIUM     MEDIUM
TRANSLATION       LOW       MEDIUM      HIGH       HIGH
MATH              LOW       MEDIUM      HIGH       HIGH
CONVERSATION      LOW       LOW         LOW        LOW      ← always cheapest
MULTIMODAL        MEDIUM    MEDIUM      HIGH       HIGH
```

---

## Quick Start

```bash
# Install
pip install google-adk google-genai pydantic-settings python-dotenv

# Copy env and fill in your key
cp .env.example .env

# Run demo (model selection printed before each API call)
adk run integrations/adk

# Or web UI
adk web integrations/adk

# Run tests
python -m pytest tests/unit/ tests/integration/ -v
```

---

## How It Works

### ADK Integration

```python
# integrations/adk/agent.py

def _dynamic_model_selector(callback_context, llm_request):
    task = ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            task = content.parts[0].text or ""
            break

    if task:
        decision = classify_task(task, provider="google")
        llm_request.model = decision.model_name  # the key line

    return None  # None = proceed with the (now mutated) request


root_agent = LlmAgent(
    name="DynamicModelAgent",
    model="gemini-2.5-flash",                      # fallback only
    before_model_callback=_dynamic_model_selector,
    instruction="You are a helpful expert assistant.",
)
```

### Direct Usage

```python
from classifier import classify_task

decision = classify_task("Design a distributed cache with LRU eviction", provider="google")
print(decision.model_name)      # "gemini-2.5-pro"
print(decision.tier.value)      # "high"
print(decision.task_type.value) # "code_creation"
print(decision.confidence)      # 0.72
print(decision.layer_used)      # "layer1" or "layer2"
print(decision.reasoning)       # "layer1 | type=code_creation complexity=complex ..."
#                                  "layer2 | multi-part implementation required | conf=0.90"

# With conversation history
decision = classify_task(
    "Make it faster",
    provider="google",
    history=["implement binary search", "debug this function"],
)
# history bias nudges toward CODE_CREATION
```

### What You See at Runtime

```
[classifier] task   : Write a README for this project
[classifier] model  : gemini-2.5-flash => gemini-2.5-flash-lite  (LOW)

[classifier] task   : Design a distributed cache with LRU eviction...
[classifier] model  : gemini-2.5-flash => gemini-2.5-pro          (HIGH)

[classifier] task   : Implement a REST API endpoint with validation
[classifier] model  : gemini-2.5-flash => gemini-2.5-flash        (MEDIUM)
```

---

## Layer 1 — Features

Pure Python, no API calls, <1ms. 15 detection features:

**Task classification**
- Weighted keyword scoring — primary keywords score 3pts, secondary 1pt, across 9 task types
- Greedy multi-word phrase matching — longer phrases matched first; consumed regions prevent double-counting
- Negation awareness — `don't write`, `without code` → penalises the matched category score
- Negative keywords — `explain` suppresses CODE_CREATION; `write` suppresses MATH

**Complexity detection**
- Weighted escalators — keyword weight sum: `distributed`=3, `microservices`=3, `rest api`=1 → escalates tier
- De-escalators — `simple`, `basic`, `quick`, `tldr`, `one-liner` → push complexity down one level
- Algorithm names — `raft`, `paxos`, `bloom filter`, `b-tree` → forces COMPLEX minimum
- Domain escalation — `hipaa`, `gdpr` → HIGH tier minimum; `clinical`, `contract` → MEDIUM minimum
- Format request suppression — `return json`, `as a table`, `in yaml` → suppresses escalation
- Question type detection — yes/no questions, "what is X" → forces SIMPLE
- Context window check — tokens > 50% of LOW tier limit → SIMPLE bumped to STANDARD

**Confidence & signals**
- Ambiguity detection — top-2 task types within 20% score → confidence capped at 0.45 (cascade signal)
- Language detection — non-English Unicode ranges → confidence 0.40 (cascade signal)
- Conversation history bias — recent code-heavy turns bias toward CODE_CREATION
- Token counting — tiktoken if installed, word-estimate fallback

---

## Configuration (.env)

```bash
# Provider
DEFAULT_PROVIDER=google          # google | anthropic | openai
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Classification layers
LAYER2_ENABLED=false             # LLM classifier via Gemini Flash Lite (set true + GOOGLE_API_KEY)
LAYER2_MODEL=gemini-2.5-flash-lite
LAYER2_TIMEOUT_MS=5000
LAYER2_MAX_RPM=100
LAYER3_ENABLED=false             # Embedding-based classifier (planned)

# Cache
CACHE_ENABLED=true
CACHE_MAX_SIZE=10000
CACHE_TTL_SECS=3600

# Budget (at 80% spend: cap MEDIUM. At 100%: force LOW)
MONTHLY_BUDGET_USD=1000.0

# Logging
LOG_DECISIONS=true               # writes routing_decisions.jsonl
```

---

## Extending

**Add a provider** — update `MODEL_REGISTRY` in `classifier/core/registry.py`:
```python
"mistral": {
    ModelTier.LOW:    "mistral-small",
    ModelTier.MEDIUM: "mistral-medium",
    ModelTier.HIGH:   "mistral-large",
}
```

**CrewAI** — same classifier, swap the last line:
```python
from crewai import Agent
decision = classify_task(task, provider="openai")
agent = Agent(llm=decision.model_name, ...)  # llm=, not model=
```

**Classification layers** (cascading — only called when previous layer is low-confidence):

| Layer | Status | Latency | When triggered |
|-------|--------|---------|----------------|
| Layer 1 | ✅ Built | <1ms | Always (fallback) |
| Layer 2 | ✅ Built | ~2s | Layer 1 confidence < 0.75 |
| Layer 3 | Planned | ~20ms | Layer 2 confidence < 0.85 |

---

## Tests

```
77 tests, 0 failures

tests/unit/test_layer1.py            22 tests  — all Layer 1 features
tests/unit/test_layer2.py            23 tests  — Layer 2: mocked API, rate limiter, all types
tests/unit/test_cache.py              8 tests  — LRU cache behaviour
tests/unit/test_cost_tracker.py       6 tests  — budget tracking
tests/integration/test_classifier.py 13 tests  — full classify_task() pipeline
```

Run: `python -m pytest tests/unit/ tests/integration/ -v`
