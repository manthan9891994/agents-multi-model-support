# dynamic-model-router

[![CI](https://github.com/manthan9891994/dynamic-model-router/actions/workflows/ci.yml/badge.svg)](https://github.com/manthan9891994/dynamic-model-router/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![Coverage](https://img.shields.io/badge/coverage-70%25%2B-brightgreen.svg)](#)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-287%20passing-brightgreen.svg)](#)

> A 3-layer cascade classifier that routes each task to the cheapest model that can handle it well — **before** the agent makes an API call.

```python
from classifier import classify

decision = classify("What is 2+2?")                    # → low tier (cheap)
decision = classify("Design a CQRS architecture for…") # → high tier (capable)
print(decision.tier, decision.model_name)
```

That's the whole pitch. Cost goes down 60–80% on real workloads with no quality loss.

---

## 📚 Table of contents

- [Why](#why)
- [How it works](#how-it-works)
- [Install](#install)
- [Step-by-step quickstart](#step-by-step-quickstart)
- [Configuration — layer by layer](#configuration--layer-by-layer)
- [The model registry](#the-model-registry)
- [Integrations](#integrations)
- [CLI reference](#cli-reference)
- [Telemetry](#telemetry)
- [Production checklist](#production-checklist)
- [License](#license)

---

## Why

You're paying for `gpt-4o` or `claude-opus-4-7` to answer "Hello, how are you?". An LLM router should pick the right model per task. Existing routers are either:

- **Hardcoded** ("if `len(prompt) > X` use big model") — too dumb
- **LLM-based** (every routing decision is itself an LLM call) — adds latency + cost
- **Single-vendor** (LiteLLM, etc.) — locked in

`dynamic-model-router` is **3 cascading classifiers** that get progressively more accurate but more expensive, stopping at the first one that's confident. Most calls never leave Layer 1 (free, <1ms).

## How it works

```
┌─────────┐   high confidence   ┌──────────┐
│ Layer 1 │ ──────────────────▶ │  Pick    │
│ keyword │                     │  model   │
│  <1ms   │                     │  & GO    │
└────┬────┘                     └──────────┘
     │ low confidence
     ▼
┌─────────┐   high confidence
│ Layer 3 │ ──────────────────▶ (same)
│   ML    │
│ ~15ms   │
└────┬────┘
     │ low confidence
     ▼
┌─────────┐
│ Layer 2 │ ──────────────────▶ (same)
│   LLM   │
│ ~500ms  │
└─────────┘
```

Each layer outputs `(task_type, complexity, confidence)` — together those map to `(provider, tier, model)` via a configurable matrix.

---

## Install

```bash
# Core (Layer 1 only — keyword router, no ML, no LLM fallback)
pip install dynamic-model-router

# With Layer 3 (ML head) — recommended
pip install 'dynamic-model-router[ml]'

# With one or more providers
pip install 'dynamic-model-router[google,anthropic,openai]'

# With agent framework integrations
pip install 'dynamic-model-router[ml,crewai]'         # CrewAI
pip install 'dynamic-model-router[ml,adk,google]'     # Google ADK

# Production extras
pip install 'dynamic-model-router[redis,kafka,s3,otel,tokenizers]'

# Everything
pip install 'dynamic-model-router[all_extensions]'
```

---

## Step-by-step quickstart

### 1️⃣ Install + set an API key

```bash
pip install 'dynamic-model-router[ml,google]'

# Choose any provider — Google's free tier is the easiest start.
echo 'GOOGLE_API_KEY=your-key-here' > .env
```

### 2️⃣ Verify your install

```bash
dmr doctor
```

You should see all green / yellow checks. Any red `[FAIL]` should be fixed before going further.

### 3️⃣ Classify your first task

```python
from classifier import classify

decision = classify("Write a Python function to merge two sorted lists.")
print(f"Use model: {decision.model_name}")
print(f"Tier:      {decision.tier.value}")
print(f"Why:       {decision.reasoning}")
```

### 4️⃣ Route an actual LLM call

```python
from classifier import Router
from google import genai

router = Router()

def smart_completion(task: str) -> str:
    decision = router.classify(task)
    client   = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(model=decision.model_name, contents=task)
    return response.text

print(smart_completion("Hi"))                            # gemini-2.5-flash
print(smart_completion("Design a distributed lock…"))    # gemini-2.5-pro
```

### 5️⃣ Train Layer 3 on your domain (optional but recommended)

```bash
# Generate sample data (or bring your own JSONL with task/task_type/complexity)
dmr generate-data --domain healthcare --per-slot 50 --out healthcare.jsonl

# Train a domain-specific classifier head (~30 seconds on CPU)
dmr train --data healthcare.jsonl
```

### 6️⃣ Customize per-domain

```python
from classifier import Router, KeywordPack, TaskType

# Healthcare keywords + HIPAA PII patterns
router = Router.from_preset("healthcare")

# Or build your own
legal_pack = (
    KeywordPack.builder("legal")
    .add(TaskType.REASONING, ["precedent", "tort", "indemnification"])
    .add(TaskType.DOC_CREATION, ["clause", "agreement", "NDA"])
    .build()
)
router = Router(extra_keyword_packs=[legal_pack])
```

### 7️⃣ Production: drop in a `dmr.yaml`

```bash
dmr init                    # scaffolds dmr.yaml in cwd
$EDITOR dmr.yaml            # tweak providers, layers, thresholds, costs
```

```python
router = Router.from_yaml("dmr.yaml")
```

---

## Configuration — layer by layer

The package ships **zero hardcoded** model names, prices, or capabilities — everything is overridable. Below is the cheat sheet, organised by layer.

### 🔵 Layer 1 — Keyword Heuristics (always on, <1ms)

| What | How |
|------|-----|
| Add domain keywords | `Router(extra_keyword_packs=[KeywordPack.builder("…").add(...).build()])` |
| Tune scoring weights | `Router(l1_weights={"primary": 5.0, "secondary": 1.0, "escalator": 2.0})` |
| Disable entirely | `Router(layer1_enabled=False)` |
| Set escalation threshold | `Router(escalation_threshold=0.75)` (below this, fall through to L3/L2) |

```python
pack = (KeywordPack.builder("biotech")
        .add(TaskType.REASONING, ["protein", "CRISPR", "in-vitro"])
        .escalator("genome-wide", weight=2)
        .build())
router = Router(extra_keyword_packs=[pack])
```

### 🟢 Layer 3 — ML Classifier (frozen MiniLM + MLP head, ~15ms)

| What | How |
|------|-----|
| Train on your data | `router.train(data="my_examples.jsonl")` or `dmr train --data ...` |
| Swap the embedding model | `Router(layer3_embedding_model="BAAI/bge-large-en-v1.5")` |
| Plug in a custom strategy | `register_l3_strategy("my_pipeline", lambda task, hist: ...)` |
| Set abstain threshold | `Router(layer3_threshold=0.85)` |
| Disable | `Router(layer3_enabled=False)` |

JSONL format for training:

```jsonl
{"task": "Implement Dijkstra in Python", "task_type": "code_creation", "complexity": "standard"}
{"task": "Hello", "task_type": "conversation", "complexity": "simple"}
```

### 🟡 Layer 2 — LLM Fallback (Gemini Flash by default, ~500ms)

| What | How |
|------|-----|
| Switch provider | `Router(layer2_provider="anthropic", layer2_model="claude-haiku-4-5-20251001")` |
| Custom prompt | `Router(layer2_prompt_template=open("my_prompt.txt").read())` |
| Retry policy | `Router(l2_retry_policy={"max_attempts": 5, "initial_delay": 0.5, "backoff": 2.0})` |
| Circuit breaker | `Router(l2_circuit_breaker={"failure_threshold": 3, "cooldown_secs": 120})` |
| Disable | `Router(layer2_enabled=False)` |
| Budget cap | `Router(budget_usd=100)` (auto-downgrades to MEDIUM at 80%, halts at 100%) |

### ⚙️ Cross-cutting

| What | How |
|------|-----|
| Per-instance overrides | `Router(provider=..., tier_matrix=..., model_registry=...)` |
| Hooks | `Router(pre_classify_hooks=[…], post_classify_hooks=[…], on_error_hooks=[…])` |
| Custom router escape hatch | `Router(custom_classifier=lambda task, ctx: my_decision)` |
| Cache backend | `Router(cache_backend=RedisCacheBackend(host="…"))` |
| Decision logger | `Router(decision_logger=KafkaLoggerBackend(brokers=[…], topic="…"))` |
| Multi-tenant per-call | `router.classify(task, tenant_config={"providers":["anthropic"], …})` |
| A/B testing | `ABTest(control=Router(), treatment=Router(...), split=0.05)` |
| Shadow mode | `ShadowMode(primary=current, shadow=new, on_diff=log_diff)` |
| PII policy | `Router(pii_policy={"min_tier": ModelTier.HIGH, "block": False})` |
| Latency SLA | `Router(latency_budget_ms=1500)` |
| Data residency | `Router(residency="EU")` |
| Custom tokenizer | `register_tokenizer("model-name", lambda t: my_count(t))` |
| Layer plugin | `register_layer(MyCustomLayer())` |

---

## The model registry

**No model name or price is hardcoded.** All of it lives in YAML — bundled `default.yaml` is a snapshot you should override in production.

### Inspect what's registered

```bash
dmr models                    # list providers + models + costs + capabilities
```

### Override entirely with your own YAML

```bash
dmr models load my-models.yaml --replace
```

```yaml
# my-models.yaml
version: "2026.05.01"
providers:
  groq:
    api_key_env: GROQ_API_KEY
    tiers:
      low:    llama-3.3-8b-instant
      medium: llama-3.3-70b-versatile
      high:   llama-3.3-70b-versatile
  bedrock:
    api_key_env: AWS_ACCESS_KEY_ID
    tiers:
      low:    anthropic.claude-haiku-4-5-20251001
      high:   anthropic.claude-opus-4-7

models:
  llama-3.3-8b-instant:
    cost: { input_per_1m: 0.05, output_per_1m: 0.08 }
    capabilities:
      context_window: 128000
      supports_function_calling: true
  llama-3.3-70b-versatile:
    cost: { input_per_1m: 0.59, output_per_1m: 0.79 }
    capabilities:
      context_window: 128000
      supports_function_calling: true
```

### Or programmatically

```python
from classifier import register_provider, register_model_cost, ModelTier

register_provider("groq", {
    ModelTier.LOW:    "llama-3.3-8b-instant",
    ModelTier.HIGH:   "llama-3.3-70b-versatile",
})
register_model_cost("llama-3.3-70b-versatile", input_per_1m=0.59, output_per_1m=0.79)
```

### Override sources (priority order)

1. `Router(registry="path-or-url")`
2. `Router.from_registry("path-or-url")`
3. `DMR_REGISTRY=/path/to/my-models.yaml` env var (loaded at import)
4. `DMR_NO_DEFAULT_REGISTRY=1` env var (start completely empty)
5. Bundled `default.yaml` (snapshot — verify before production!)

---

## Integrations

| Framework | Module | Pattern |
|-----------|--------|---------|
| **LangChain** | `classifier.integrations.langchain` | `get_chat_model(task)` or `DynamicChatModel()` |
| **CrewAI** | `classifier.integrations.crewai` | `pick_llm_for_task(task)` or `DynamicLLM()` |
| **AutoGen** | `classifier.integrations.autogen` | `get_autogen_llm_config(task)` |
| **OpenAI Agents SDK** | `classifier.integrations.autogen` | `get_openai_agent_model(task)` |
| **Google ADK** | `classifier.integrations.adk` | `before_model_callback=dynamic_model_selector` |
| **LlamaIndex** | `classifier.integrations.llamaindex` | `get_llm(task)` or `DynamicLLM()` |
| **Pydantic AI** | `classifier.integrations.pydantic_ai` | `get_model_string(task)` or `get_agent(task, **kw)` |
| **DSPy** | `classifier.integrations.dspy` | `get_lm(task)` or `with route(task): ...` |
| **Haystack** | `classifier.integrations.haystack` | `get_generator(task)` |
| **Semantic Kernel** | `classifier.integrations.semantic_kernel` | `get_chat_service(task)` |
| **smolagents (HF)** | `classifier.integrations.smolagents` | `get_model(task)` or `DynamicModel()` |

```python
# CrewAI example
from crewai import Agent
from classifier.integrations.crewai import DynamicLLM

agent = Agent(role="Analyst", goal="...", llm=DynamicLLM())
# Each call this agent makes is routed to the right tier dynamically.
```

```python
# Decorator — any function gets dynamic model selection
from classifier import route_model

@route_model(provider="anthropic")
def call_claude(task: str, model_name: str = "claude-haiku-4-5-20251001"):
    # model_name is auto-injected by the router
    ...
```

---

## CLI reference

```bash
dmr classify "task text"            # one-shot classification
dmr classify --preset healthcare "Patient MRN 12345 has chest pain"

dmr train --data examples.jsonl     # train Layer 3 on your data
dmr eval  --data test.jsonl         # accuracy + tier distribution
dmr generate-data --domain legal --per-slot 50    # synthetic training data via Gemini

dmr models                          # list registered providers/models/costs
dmr models load my-models.yaml --replace
dmr models export --output snapshot.yaml
dmr models pull https://example.com/community-registry.yaml

dmr stats                           # routing distribution from decision log
dmr stats cost --since 7d           # cost breakdown over last week

dmr doctor                          # diagnose env / config / dependencies
dmr version                         # package + Python + dep versions
dmr benchmark                       # local p50/p95/p99 latency
dmr init                            # scaffold dmr.yaml in cwd
dmr presets                         # list domain presets
```

---

## Telemetry

> **`dynamic-model-router` does not collect any telemetry. No usage data, no model names, no error reports leave your machine. Ever.**

The package never makes a network call you didn't ask for. The only network calls happen when:

1. You explicitly construct a `Router` and call `.classify()` with `layer2_enabled=True` — then Layer 2 calls the provider you chose.
2. You explicitly call `Router.load_registry("https://...")` — then we fetch that URL.
3. Your decision-logger backend is configured to forward (e.g. `WebhookLoggerBackend`).

If you discover any unexpected outbound traffic, **that is a security bug** — please file a [security advisory](SECURITY.md).

---

## Production checklist

Before going live with serious traffic:

- [ ] **Override the bundled registry.** `dmr models export > my-models.yaml`, edit, then `Router.from_registry("my-models.yaml")`. Bundled prices go stale fast.
- [ ] **Set up secrets properly.** Use a secret manager — not `.env` in your repo. Rotate quarterly.
- [ ] **Train Layer 3 on your data.** A `head_v1.joblib` trained on your domain reduces L2 (LLM) calls by another 60–80%.
- [ ] **Pin a small budget initially** (`Router(budget_usd=100)`) and watch `dmr stats cost`.
- [ ] **Enable strict PII scrubbing** (`pii_scrub_strict=true` in settings, plus domain-specific `extra_pii_patterns`).
- [ ] **Set a tight L2 circuit breaker** (`failure_threshold=3, cooldown_secs=120`) so a provider outage doesn't drain your wallet.
- [ ] **Configure decision logging** to an immutable backend (S3 with object lock, or a write-only Kafka topic) for audit trails.
- [ ] **Run `dmr doctor`** in CI — fail the build if any check is FAIL.
- [ ] **Use `ShadowMode`** to validate every routing change before flipping the switch.
- [ ] **Subscribe to the [security advisory](SECURITY.md)** for vulnerability notifications.
- [ ] **Pin the package version** in your lock file. The package follows semver; minor bumps may include behaviour changes for unset config defaults.

---

## License

MIT — see [LICENSE](LICENSE).

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md). Please **do not** open a public issue.

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). All contributors agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for upcoming features and the path from 0.1 → 1.0.
