# Dynamic Model Router — Multi-Model Task Classifier

Automatically routes each task to the **cheapest model that can handle it well** — before the agent makes an API call. One classifier, three providers, zero framework changes.

Built for Google ADK. Works with CrewAI, direct API, or any Python agent.

---

## The Problem

Every agent is hardcoded to one model. A "write a README" and a "design a distributed fault-tolerant system" both hit `gemini-2.5-pro` — wasting money and adding latency where it isn't needed.

## The Solution

Classify the task **before** the model call. Route to the cheapest model that can handle it well.

```
User message
  → Layer 1: keyword + heuristic  (<1ms,  no API call, always runs)
      └─ conf < 0.75 → Layer 2: Gemini Flash Lite  (~2s, $0.00001/call)
  → Context signals from agent loop → mid-flight tier adjustment
  → Tier: LOW / MEDIUM / HIGH
  → Maps tier → model name for chosen provider
  → ADK: llm_request.model = selected_model  (mutated before API call)
```

---

## Quick Start

```bash
pip install -r requirements.txt

cp .env.example .env
# Fill in DEFAULT_PROVIDER and the matching API key

# ADK demo
adk run integrations/adk

# Run tests
python -m pytest tests/ -v   # 104 tests, 0 failures

# Direct usage
python -c "
from classifier import classify_task
d = classify_task('Design a distributed cache with LRU eviction', provider='google')
print(d.model_name, d.tier.value, d.layer_used, d.confidence)
"
```

---

## Model Tiers

| Tier | Google | Anthropic | OpenAI | When |
|------|--------|-----------|--------|------|
| LOW | gemini-2.5-flash-lite | claude-haiku-4-5 | gpt-4o-mini | Conversation, simple docs, trivial code |
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
CONVERSATION      LOW       LOW         LOW        LOW
MULTIMODAL        MEDIUM    MEDIUM      HIGH       HIGH
```

---

## Project Structure

```
agents_multi_model_support/
├── classifier/
│   ├── __init__.py                   # Public API: classify_task()
│   ├── core/
│   │   ├── types.py                  # ModelTier, TaskType, TaskComplexity, ClassificationDecision
│   │   ├── registry.py               # TIER_MATRIX, MODEL_REGISTRY (3 providers × 3 tiers)
│   │   └── exceptions.py             # ClassifierError hierarchy
│   ├── layers/
│   │   ├── layer1/                   # Keyword + heuristic classifier (<1ms)
│   │   │   ├── constants.py          # Keyword tables, escalators, domain tiers
│   │   │   ├── helpers.py            # Token counting, language detection, negation
│   │   │   ├── pii.py                # PII/PHI pattern detection
│   │   │   ├── scoring.py            # _score_task_type, _detect_complexity
│   │   │   ├── keyword_packs.py      # Domain YAML pack loader
│   │   │   └── classify.py           # classify_layer1() entry point
│   │   └── layer2/                   # LLM reclassifier (Gemini Flash Lite)
│   │       ├── prompt.py             # Injection-resistant prompt + schema
│   │       ├── api.py                # Client calls, retry logic, executor
│   │       ├── rate_limiter.py       # Thread-safe sliding-window limiter
│   │       ├── validation.py         # Output plausibility validation
│   │       └── classify.py           # classify_layer2() entry point
│   ├── infra/
│   │   ├── config.py                 # Pydantic Settings (.env)
│   │   ├── cache.py                  # LRU + TTL exact-match cache
│   │   ├── semantic_cache.py         # Embedding-based similarity cache (optional)
│   │   ├── cost_tracker.py           # Budget tracking, per-category limits
│   │   ├── decision_logger.py        # JSONL routing log with PII redaction
│   │   ├── coalescer.py              # Single-flight cache stampede protection
│   │   ├── health_tracker.py         # p95 latency SLO tracker per (provider, tier)
│   │   ├── personalization.py        # Per-user tier bias (30-day decay)
│   │   └── feedback.py               # Feedback recording for L3 training
│   ├── config/
│   │   ├── features.yaml             # Feature flag configuration (24 flags)
│   │   └── feature_flags.py          # Typed FeatureFlags dataclass + YAML loader
│   ├── data/
│   │   ├── reference_tasks.jsonl     # Labeled examples for Layer 3
│   │   └── keyword_packs/
│   │       └── healthcare.yaml       # Healthcare domain keyword pack
│   ├── calibrate.py                  # Offline confidence calibration
│   └── stats.py                      # CLI: routing decisions analytics
├── integrations/
│   └── adk/
│       ├── agent.py                  # LlmAgent with before_model_callback
│       └── run.py
├── tests/
│   ├── conftest.py
│   ├── unit/                         # test_layer1, test_layer2, test_cache, test_cost_tracker
│   └── integration/                  # test_classifier (full pipeline)
└── plan_docs/
    ├── 00_status.md                  # Current implementation status
    ├── 01_layer3_embedding.md        # Next: Layer 3 embedding KNN classifier
    ├── 02_layer3_ml_pipeline.md      # Future: fine-tuned ML classifier
    └── 03_enterprise_scale.md        # Future: enterprise & global scale
```

---

## Layer 1 — Keyword + Heuristic (14 feature flags)

Pure Python, no API calls, <1ms. Handles ~78% of production traffic at zero LLM cost.

| Feature | What it does |
|---------|-------------|
| Weighted keyword scoring | Primary keywords score 3pts, secondary 1pt, 9 task types |
| Negation suppression | `don't implement`, `without code` → penalises matched category |
| Code snippet detection | `def`/`class`/triple-backtick → +4pts toward CODE_CREATION |
| Multi-task detection | Ambiguous tasks (top-2 within 80%) → picks higher-tier type |
| Weighted escalator scoring | `distributed`=3, `thread-safe`=2, `rest api`=1 → bumps complexity |
| Algorithm detection | `raft`, `bloom filter`, `dijkstra` → forces COMPLEX minimum |
| Domain escalation | `hipaa`/`gdpr` → HIGH tier; `clinical`/`contract` → MEDIUM |
| Format request de-escalation | `as JSON`, `in markdown` → suppresses escalation |
| Question type override | "What is X" → always SIMPLE; yes/no → SIMPLE when weight < 3 |
| Language detection | Non-English Unicode → caps conf at 0.40 (cascade signal) |
| Continuation detection | "Now make it faster" → inherits type from history |
| History bias | Last 3 turns score ≥ 4pts → overrides current task type |
| Trivial-input guard | `"k"`, `"👍"`, single chars → CONVERSATION/SIMPLE instantly |
| Pluggable keyword packs | Domain YAML packs (healthcare, fintech, legal) via `KEYWORD_PACKS=` |
| PII/PHI detection | SSN, email, credit card, JWT, MRN, DOB → forces MEDIUM+, sets `compliance_flag=True` |

---

## Layer 2 — LLM Reclassifier (4 feature flags)

Fires only when Layer 1 confidence < 0.75. Uses Gemini Flash Lite for ~$0.00001/call.

| Feature | What it does |
|---------|-------------|
| Injection-resistant prompt | Task wrapped in `<task>` tags; model told to treat as untrusted data |
| Output plausibility validation | Rejects structurally implausible responses (not input blocking — zero false positives) |
| Exponential backoff retry | 3 attempts on 429/5xx; 200ms/600ms/1.8s; non-retryable errors fail fast |
| Thread-safe rate limiter | Sliding-window 100 rpm; blocks excess calls, falls back to L1 |
| Fallback model | `LAYER2_FALLBACK_MODEL` retried if primary fails |
| ThreadPoolExecutor timeout | API hangs never block the caller (default 2s) |

**Output validation checks** (on L2's JSON response, never on input):
1. Keyword cross-check — returned type must have keyword support if L1 has strong counter-evidence
2. Complexity sanity — `conversation/simple/conf>0.80` rejected for tasks > 60 tokens
3. Code-in-task — code blocks + `doc_creation/simple` = structural mismatch → rejected

---

## System Features (6 feature flags)

| Feature | What it does |
|---------|-------------|
| L1+L2 agreement boost | Both agree → confidence boosted to `min(0.95, max+0.10)` |
| L1+L2 disagreement flag | Disagree → higher-tier wins, `disagreement=True` logged |
| Single-flight coalescing | 100 concurrent identical tasks → 1 classification, others wait |
| Health tracker | p95 latency SLO; demotes tier when provider degrades |
| Per-user personalization | Tier bias per user_id, 30-day half-life decay |
| Confidence calibration | Offline-computed curves from labeled decisions |

---

## Feature Flags

All 24 features can be toggled in `classifier/config/features.yaml` without touching code:

```yaml
layer1:
  pii_detection: false         # disable if PII is redacted upstream
  trivial_input_guard: false   # disable for voice pipelines ("stop", "go" = real commands)
  domain_escalation: false     # disable if all traffic is pre-routed to HIGH
  keyword_packs: false         # disable if no domain packs configured

layer2:
  l2_output_validation: false  # disable to trust raw LLM output
  l2_rate_limiter: false       # disable if gateway already enforces rate limits

system:
  semantic_cache: true         # enable once sentence-transformers is installed
  per_user_personalization: true  # enable for multi-tenant systems
```

---

## ADK Integration

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

    return None  # proceed with the (now mutated) request

root_agent = LlmAgent(
    name="DynamicModelAgent",
    model="gemini-2.5-flash",                      # fallback only
    before_model_callback=_dynamic_model_selector,
    instruction="You are a helpful expert assistant.",
)
```

**Context-aware mid-flight switching** (auto, no config needed):

| Call | Condition | Action |
|------|-----------|--------|
| Any | `total_context_tokens > 100K` | Bump to MEDIUM minimum |
| Any | `has_error=True` | Bump to MEDIUM minimum |
| ≥ 2nd | `last=tool, no error` | Step down one tier |
| ≥ 3rd | `last=model, no error` | Drop to LOW |
| 1st | `available_tools ≥ 3` | Bump one tier (planning call) |
| Any | Multimodal content detected | Force MULTIMODAL task type |

---

## Direct Usage

```python
from classifier import classify_task, ContextSignals

# Basic
decision = classify_task("Design a distributed cache with LRU eviction", provider="google")
print(decision.model_name)      # "gemini-2.5-pro"
print(decision.tier.value)      # "high"
print(decision.task_type.value) # "code_creation"
print(decision.confidence)      # 0.92
print(decision.layer_used)      # "layer1" | "layer2"
print(decision.reasoning)       # "layer1 | type=code_creation complexity=complex ..."
print(decision.compliance_flag) # True if PII/PHI detected

# With history
decision = classify_task(
    "Make it faster",
    provider="google",
    history=["implement binary search", "debug this function"],
)

# With context signals (agent loop)
decision = classify_task(
    "summarize the results",
    provider="google",
    context_signals=ContextSignals(
        call_number=3,
        last_role="tool",
        has_error=False,
        total_context_tokens=12000,
    ),
)

# Per-user personalization
decision = classify_task("help me with this", provider="google", user_id="alice")

# Streaming debounce (user still typing)
decision = classify_task(task, provider="google", task_stable=False)
```

---

## Configuration (.env)

```bash
# Provider
DEFAULT_PROVIDER=google
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# Layer 2
LAYER2_ENABLED=true
LAYER2_MODEL=gemini-2.5-flash-lite
LAYER2_TIMEOUT_MS=2000
LAYER2_MAX_RPM=100
LAYER2_CONFIDENCE_THRESHOLD=0.75
LAYER2_FALLBACK_MODEL=           # e.g. gpt-4o-mini

# Budget (80% spend → cap MEDIUM; 100% → force LOW)
MONTHLY_BUDGET_USD=1000.0
LAYER2_MONTHLY_BUDGET_USD=50.0   # defaults to 5% of main budget

# Cache
CACHE_ENABLED=true
CACHE_MAX_SIZE=10000
CACHE_TTL_SECS=3600
SEMANTIC_CACHE_ENABLED=false     # requires sentence-transformers

# Domain keyword packs
KEYWORD_PACKS=healthcare          # comma-separated pack names

# Logging
LOG_DECISIONS=true               # writes routing_decisions.jsonl
DEBUG_AB_MODE=false              # run L1 and L2 unconditionally, log both
```

---

## Tests

```
104 tests, 0 failures

tests/unit/test_layer1.py            ~47 tests  — all 14 L1 features
tests/unit/test_layer2.py            ~30 tests  — output validation, injection defense, retry, rate limiter
tests/unit/test_cache.py              8 tests   — LRU, TTL, hit rate
tests/unit/test_cost_tracker.py       6 tests   — budget thresholds, category budgets
tests/integration/test_classifier.py 13 tests   — full classify_task() pipeline
```

```bash
python -m pytest tests/ -v
```

---

## Observability

```bash
# Routing summary (last 24 hours)
python -m classifier.stats summary --since 24h
# total: 12,403 | L1-only: 78% | L2-fired: 22% | L2-agreement: 86%
# Avg latency: LOW=120ms MEDIUM=180ms HIGH=250ms (classifier overhead)

# L1/L2 disagreements (review candidates for L3 training)
python -m classifier.stats disagreements --since 7d --limit 20

# Cost breakdown
python -m classifier.stats cost --since 30d
# L2 spend: $4.32 / budget $50.00 (8.6%)
```

---

## Healthcare / Domain Notes

This system was built with healthcare and high-compliance domains in mind:

- **PII/PHI detection** — SSN, email, MRN, DOB, credit card, API key, JWT patterns → forces MEDIUM+ tier, sets `compliance_flag=True`, redacts spans in decision log
- **HIPAA/GDPR domain escalation** — any task containing these keywords is forced to HIGH tier regardless of complexity score
- **Domain keyword packs** — extend the classifier with clinical vocabularies (ICD-10, SNOMED terms, drug names) via `classifier/data/keyword_packs/healthcare.yaml`
- **Audit trail** — every routing decision logged to `routing_decisions.jsonl` with tier, confidence, layer used, latency
- **Governance** — feature flags let compliance teams disable specific classifier behaviors per deployment

---

## Roadmap

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | Layer 1 (keyword + heuristic) + infra + ADK | ✅ Done |
| 2 | Layer 2 (LLM classifier) + reliability hardening | ✅ Done |
| 3 | 20 production hardening features + feature flags | ✅ Done |
| 4 | Layer 3 (embedding KNN, ~10ms, $0) | `plan_docs/01_layer3_embedding.md` |
| 5 | In-house fine-tuned ML classifier | `plan_docs/02_layer3_ml_pipeline.md` |
| 6 | Enterprise scale (multi-region, multi-tenant, REST API) | `plan_docs/03_enterprise_scale.md` |

---

## Extending

**Add a provider:**
```python
# classifier/core/registry.py
MODEL_REGISTRY["mistral"] = {
    ModelTier.LOW:    "mistral-small",
    ModelTier.MEDIUM: "mistral-medium",
    ModelTier.HIGH:   "mistral-large",
}
```

**Add a domain pack:**
```yaml
# classifier/data/keyword_packs/fintech.yaml
escalators:
  - {kw: "derivative pricing", weight: 3}
domain_min_tier:
  - {kw: "sec filing", tier: high}
task_keywords:
  analyzing:
    primary: ["portfolio attribution", "risk-adjusted return"]
```
```bash
# .env
KEYWORD_PACKS=healthcare,fintech
```

**CrewAI:**
```python
from crewai import Agent
from classifier import classify_task

decision = classify_task(task, provider="openai")
agent = Agent(llm=decision.model_name, ...)  # llm=, not model=
```
