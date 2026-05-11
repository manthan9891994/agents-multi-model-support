# dynamic-model-router

[![CI](https://github.com/manthan9891994/agents-multi-model-support/actions/workflows/ci.yml/badge.svg)](https://github.com/manthan9891994/agents-multi-model-support/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![Coverage](https://img.shields.io/badge/coverage-78%25-brightgreen.svg)](#)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-397%20passing-brightgreen.svg)](#)

> A 3-layer cascade classifier that routes each task to the cheapest model that can handle it well — **before** the agent makes an API call.

```python
from classifier import classify

decision = classify("What is 2+2?")                    # → low tier (cheap)
decision = classify("Design a CQRS architecture for…") # → high tier (capable)
print(decision.tier, decision.model_name)
```

That's the whole pitch. Cost goes down 60–80% on real workloads with no quality loss.

---

## Why not just use LiteLLM / RouteLLM / Portkey?

Short answer: **we decide *which* model to call** — those tools manage *how* to call a model you've already chosen. They're complementary, not competitive. Use this router to pick the tier, then use LiteLLM (or your provider SDK) to actually make the call.

| | **dynamic-model-router** | LiteLLM | RouteLLM (lmsys) | Portkey | OpenRouter |
|---|---|---|---|---|---|
| **Category** | Pre-call routing classifier | Unified SDK + proxy | LLM-to-LLM routing | AI gateway (commercial) | Hosted routing API (paid) |
| Decides model **before** any API call | ✅ | ❌ (you specify) | ✅ | ❌ | ✅ |
| Sub-1ms keyword tier | ✅ | ❌ | ❌ | ❌ | ❌ |
| Local ML classifier (frozen-encoder, no API) | ✅ | ❌ | ✅ | ❌ | ❌ |
| Three independent decision layers | ✅ | n/a | ❌ (single MLP) | n/a | ❌ |
| Train on your own logs (zero data → SMEs → auto) | ✅ | n/a | partial | ❌ | ❌ |
| Framework adapters bundled | **11** | 4 | 1 | 6 | n/a (provider API) |
| PII auto-redaction in decision logs | ✅ | ❌ | ❌ | partial | ❌ |
| Healthcare/HIPAA keyword pack out of the box | ✅ | ❌ | ❌ | ❌ | ❌ |
| Self-hosted, no data leaves your infra | ✅ | ✅ | ✅ | partial | ❌ (SaaS) |
| Open source, MIT | ✅ | ✅ | ✅ | ❌ | ❌ |
| Free | ✅ | ✅ | ✅ | freemium | pay-per-token |

If you want to **call a specific model reliably**, use LiteLLM.
If you want to **pick the right model first**, use this router.
If you want **both**, stack them — `router.classify(task)` → pick tier → hand the model name to LiteLLM.

---

## 📚 Table of contents

- [Why not LiteLLM / RouteLLM / Portkey?](#why-not-just-use-litellm--routellm--portkey)
- [Install](#install)
- [The 3 layers — in plain English](#the-3-layers--in-plain-english)
- [60-second quickstart](#60-second-quickstart)
- [Layer 1 — Add your own keywords](#layer-1--add-your-own-keywords-no-code-needed)
- [Layer 3 — Train on your data](#layer-3--train-on-your-data-one-command)
- [Track & inspect what's happening](#track--inspect-whats-happening)
- [Decision log — three modes](#decision-log--three-modes)
- [Layer 2 — LLM fallback (advanced)](#layer-2--llm-fallback-advanced)
- [Model registry](#model-registry)
- [Integrations](#integrations)
- [CLI reference](#cli-reference)
- [Production checklist](#production-checklist)
- [FAQ](#faq)

---

## Install

```bash
pip install dynamic-model-router          # core
pip install 'dynamic-model-router[ml]'    # + Layer 3 (recommended)
pip install 'dynamic-model-router[ml,google]'           # + Gemini provider
pip install 'dynamic-model-router[ml,google,anthropic,openai]'   # all 3
```

Set one API key in `.env` (Google has a free tier — easiest start):

```bash
echo 'GOOGLE_API_KEY=your-key-here' > .env
```

Verify your install:

```bash
dmr doctor
```

---

## The 3 layers — in plain English

Every task you classify walks down a ladder. The first layer that's confident wins. Most tasks stop at Layer 1.

| | Layer | What it does | Cost | Speed |
|---|---|---|---|---|
| 🟦 | **Layer 1 — Keywords** | Looks at the words in your task. "implement", "function" → coding. "summarize" → doc creation. "diagnose", "patient" → medical reasoning. | Free | <1 ms |
| 🟩 | **Layer 3 — ML model** | A small neural net trained on your data (or our defaults). Catches things keywords miss — like sentence structure, intent, complexity. | Free | ~15 ms |
| 🟨 | **Layer 2 — LLM fallback** | When the first two are unsure, asks an LLM to classify the task. Same provider you'll route to. | $$ | ~500 ms |

**The cascade:** `keywords confident? → ship.` Otherwise: `ML confident? → ship.` Otherwise: ask an LLM. So every customization you make to Layer 1 (cheap, deterministic) saves you Layer 2 calls (slow, billed).

What each layer outputs is the same: `(task_type, complexity, confidence)`. Together those map to `(provider, tier, model)` via a configurable matrix.

---

## 60-second quickstart

```python
from classifier import Router

# Zero config. Layer 3 turns on automatically once you've trained it.
router = Router(layer3_enabled="auto")

decision = router.classify("Implement Dijkstra's algorithm in Python")
print(decision.model_name)   # → gemini-2.5-flash
print(decision.tier.value)   # → low
print(decision.layer_used)   # → layer1
print(decision.reasoning)    # → keyword match: "implement"
```

Drop that `decision.model_name` into whatever SDK you use:

```python
from google import genai
client = genai.Client()
response = client.models.generate_content(
    model=decision.model_name,
    contents="Implement Dijkstra's algorithm in Python",
)
```

Or use one of the [11 framework integrations](#integrations) — LangChain, CrewAI, AutoGen, ADK, LlamaIndex, Pydantic AI, DSPy, Haystack, Semantic Kernel, smolagents, OpenAI Agents.

---

## Layer 1 — Add your own keywords (no code needed)

Layer 1 is just: *"if the task contains these words, it's probably this kind of task."* Adding domain vocabulary is the single highest-leverage customization you can make.

### The easy way — `dmr keywords`

```bash
# Add a few legal-domain keywords
dmr keywords add --domain legal --type reasoning \
                 --keywords "tort,liable,precedent,indemnification"

# See what you've added
dmr keywords list

# Found a wrong one?
dmr keywords remove --domain legal --keyword "tort"
```

That's it. Packs are saved to `~/.dmr/keywords/<domain>.yaml` and **auto-loaded by every new `Router()`** — no code change.

### Don't know what keywords to add? Mine them from your logs

Once your router has handled some real traffic, ask it which words it's seeing:

```bash
dmr keywords suggest --since 30d --top 15
```

```
Top distinctive n-grams per task_type (not already in any pack):

  [reasoning]
     2.41   n=37    differential diagnosis
     2.18   n=29    clinical scenario
     1.94   n=42    contraindication

  [doc_creation]
     2.05   n=51    progress note
     1.78   n=33    discharge summary
```

Pick the strong ones and `dmr keywords add` them.

### Or build a pack programmatically

```python
from classifier import KeywordPack, TaskType, Router

biotech = (KeywordPack.builder("biotech")
           .add(TaskType.REASONING, ["protein", "CRISPR", "in-vitro"])
           .escalator("genome-wide", weight=2)   # bumps complexity
           .build())

router = Router(extra_keyword_packs=[biotech])
```

---

## Layer 3 — Train on your data (one command)

You don't need labeled data to start. The package logs every routing decision to `routing_decisions.jsonl`, and `dmr train --auto` turns that log into training data using **8 weak-supervision rules** (Snorkel-style — short prompts, user retries, model escalations, etc.).

### Workflow

**Day 1.** Install. Use the router. L1 + L2 work immediately. L3 is silently disabled.

```python
router = Router(layer3_enabled="auto")    # auto = enable when a model exists
```

**Day 30.** You've logged a few hundred decisions. `dmr doctor` notices:

```
[!] L3 model file  WARN  missing, but 547 decisions logged
                          → run `dmr train --auto` to enable Layer 3
```

**One command** to bootstrap Layer 3 from those logs:

```bash
dmr train --auto
```

```
[1/3] Auto-labeling decision/outcome telemetry since 2026-04-09...
  Got 312 confident labels:
    task_type   reasoning            104
    task_type   doc_creation          98
    task_type   code_creation         67
    complexity  simple                86
    complexity  standard             162
    complexity  complex               64

[2/3] Training Layer 3 head (frozen MiniLM + calibrated MLPs)...
[3/3] Done.

  task_type accuracy:    0.831
  complexity accuracy:   0.776

  Layer 3 is now active. New `Router()` instances will pick it up
  automatically when constructed with `layer3_enabled='auto'` (default).
```

That's it. Re-run any time you want — each run replaces the model.

### Already have labeled data?

```bash
dmr train --data my_examples.jsonl
```

JSONL format:

```jsonl
{"task": "Implement Dijkstra in Python", "task_type": "code_creation", "complexity": "standard"}
{"task": "Hello", "task_type": "conversation", "complexity": "simple"}
```

### No production data and want a head start?

```bash
dmr generate-data --domain healthcare --per-slot 50 --out healthcare.jsonl
dmr train --data healthcare.jsonl
```

(Uses Gemini to synthesize realistic examples for your domain.)

### Tune Layer 3 in code

```python
Router(
    layer3_enabled="auto",                                     # default
    layer3_threshold=0.85,                                      # higher = stricter
    layer3_embedding_model="BAAI/bge-large-en-v1.5",           # swap encoder
)
```

---

## Track & inspect what's happening

Every classification is logged. The package gives you simple commands to inspect what the router is doing.

### `dmr doctor` — health check + readiness

```bash
dmr doctor
```

```
  [+] Python version              OK   3.12.7
  [+] dep:pydantic_settings       OK   installed
  [+] opt:google.genai            OK   installed (Layer 2 fallback)
  [+] opt:sentence_transformers   OK   installed (Layer 3 ML head)
  [+] key:google                  OK   configured
  [!] key:anthropic               WARN ANTHROPIC_API_KEY not set
  [+] L3 model file               OK   head_v1.joblib (3,166 KB)
  [+] classify smoke test         OK   tier=low model=gemini-2.5-flash

  Result: 12 ok, 1 warning(s), 0 failure(s)
```

Run it in CI — fail your build on `[x]`.

### `dmr config show` — what's actually loaded

```bash
dmr config show
```

```
  dynamic-model-router  v0.2.0

  [settings]
    default_provider          google
    layer1_enabled            True
    layer2_enabled            True
    layer3_enabled            True
    cache_enabled             True
    monthly_budget_usd        $1000.0

  [registry]
    providers                 google, anthropic, openai
    models                    8

  [layer 3]
    model file                head_v1.joblib (3,166 KB)
    trained on                2026 examples
    task_type accuracy        0.789
    complexity accuracy       0.796

  [keyword packs]
    registered                healthcare, legal, your_custom
```

### `dmr stats` — what's it actually routing?

```bash
dmr stats              # tier distribution + layer hit rates (default 24h)
dmr stats cost --since 7d
dmr stats disagreements
```

```
Routing summary — last 24 hours
  Total decisions          1,247
  Layer 1 (free)           892   (71.5%)
  Layer 3 (ML)             231   (18.5%)
  Layer 2 (LLM)            124   (10.0%)

  Tier distribution
    low                    687   (55.1%)   $0.86
    medium                 478   (38.3%)   $4.12
    high                    82   ( 6.6%)   $9.74
                                            ─────
                                            $14.72
```

### `dmr config validate` — schema-check your `dmr.yaml`

```bash
dmr config validate
```

### Decision log — three modes

The router emits two streams: **decisions** (what was routed where) and **outcomes** (what happened — tokens, cost, success). How they're delivered depends on what you turn on.

#### Mode 1 — Default (no setup)

One quiet `INFO` line per event via standard Python logging. No files. No DB. Just like any well-behaved library:

```
INFO dmr.decisions: DMR decision: tier=low  model=gemini-2.5-flash layer=layer1 conf=0.91 lat=2ms
INFO dmr.outcomes:  DMR outcome:  tokens=42/180 wall=412ms success=True cost=$0.000023
```

Silence it: `logging.getLogger("dmr").setLevel(logging.WARNING)`.

#### Mode 2 — Full structured telemetry

Set `DMR_TELEMETRY=1`. Same logger, richer payload — now every event is a full JSON event at `logging.DEBUG`. **Still no files written.** If you want persistence, see Mode 3.

```bash
DMR_TELEMETRY=1 python app.py
```

```json
{"timestamp": "2026-05-09T14:23:11Z", "decision_id": "abc123...", "router_version": "0.4.0",
 "task_preview": "Implement…", "tier": "low", "model": "gemini-2.5-flash", "task_type": "code_creation",
 "complexity": "standard", "confidence": 0.91, "layer": "layer1", "latency_ms": 0.4,
 "provider": "google", "compliance_flag": false, "cached": false}
```

PII (SSNs, emails, API keys, JWTs, phone numbers, etc.) is auto-redacted from `task_preview` and `error_message`. Route the `dmr.decisions` and `dmr.outcomes` Python loggers wherever you want — file handler, syslog, OTLP, Datadog, etc.

#### Mode 3 — Pluggable backend (you own the storage)

**The package never writes files automatically.** If you want persistence, wire a backend — that's the *only* way data lands anywhere outside Python logging.

Any object with a `log(entry: dict)` method works:

```python
from classifier import Router
from examples.custom_backends.sqlite_backend import SQLiteBackend

backend = SQLiteBackend("my_telemetry.db")
router = Router(decision_logger=backend, outcome_logger=backend)
```

Ready-made backends in [`examples/custom_backends/`](examples/custom_backends/):

| Storage | File | Extra deps |
|---------|------|------------|
| **SQLite** (local, zero-dep) | [sqlite_backend.py](examples/custom_backends/sqlite_backend.py) | none |
| **PostgreSQL** | [postgres_backend.py](examples/custom_backends/postgres_backend.py) | `psycopg2-binary` |
| **Google BigQuery** | [bigquery_backend.py](examples/custom_backends/bigquery_backend.py) | `google-cloud-bigquery` |
| **AWS DynamoDB** | [dynamodb_backend.py](examples/custom_backends/dynamodb_backend.py) | `boto3` |
| **Google Cloud Storage** | [gcs_backend.py](examples/custom_backends/gcs_backend.py) | `google-cloud-storage` |

Built-in (no extra files needed): `JSONLLoggerBackend`, `StdoutLoggerBackend`, `WebhookLoggerBackend`, `KafkaLoggerBackend`, `S3LoggerBackend`.

**Fan out to multiple sinks** with `MultiLoggerBackend`:

```python
from classifier import Router, MultiLoggerBackend, StdoutLoggerBackend
from examples.custom_backends.sqlite_backend import SQLiteBackend

backend = MultiLoggerBackend([
    SQLiteBackend("local.db"),     # local queryable copy
    StdoutLoggerBackend(),         # also stream to stdout for log collectors
])
router = Router(decision_logger=backend, outcome_logger=backend)
```

A broken backend never blocks the others — failures are caught and logged at `WARNING`.

#### What's in each event

**Decision event** (one per `router.classify()`):

| Field | Type | Notes |
|-------|------|-------|
| `decision_id` | str | 16-char hex — join key to outcomes |
| `timestamp` | ISO 8601 | UTC |
| `router_version` | str | package `__version__` |
| `task_preview` | str | first 200 chars, PII-redacted |
| `task_length` | int | full task length |
| `tier` | str | `low`/`medium`/`high` |
| `model`, `provider` | str | the routed model |
| `task_type`, `complexity` | str | classifier output |
| `confidence` | float | 0–1 |
| `layer` | str | which layer decided: `layer1`/`layer2`/`layer3` |
| `latency_ms` | float | classification time |
| `compliance_flag` | bool | PII/PHI detected in task |
| `disagreement` | bool | L1 vs L3 disagree |
| `exploration` | bool | random sample for drift detection |
| `cached`, `cached_from` | bool, str | cache-hit metadata |

**Outcome event** (call `router.report_outcome(...)` after your LLM call returns):

| Field | Type | Notes |
|-------|------|-------|
| `decision_id` | str | join key |
| `tokens_in`, `tokens_out` | int | usage |
| `tokens_estimated` | bool | True if heuristic (vs provider-reported) |
| `wall_ms` | float | full LLM call time |
| `success` | bool | call completed |
| `cost_usd` | float | computed from model rates |
| `user_feedback` | str | `up`/`down`/None |
| `user_retried`, `user_escalated_model`, `edit_distance` | mixed | optional signals |
| `error_message` | str | PII-redacted |

Join decisions to outcomes via `decision_id` for cost-per-tier / accuracy / cache-hit-rate dashboards.

#### Try it in 30 seconds

```bash
python examples/test_telemetry.py              # Mode 1 — quiet
DMR_TELEMETRY=1 python examples/test_telemetry.py   # Mode 2 — full JSON
python examples/test_telemetry.py --db         # Mode 3 — SQLite backend + analytics
```

---

## Layer 2 — LLM fallback (advanced)

Layer 2 only fires when L1 + L3 are both uncertain (~10% of traffic in practice). Defaults to Gemini Flash, but everything is overridable:

```python
Router(
    layer2_provider="anthropic",
    layer2_model="claude-haiku-4-5-20251001",
    l2_retry_policy={"max_attempts": 5, "initial_delay": 0.5, "backoff": 2.0},
    l2_circuit_breaker={"failure_threshold": 3, "cooldown_secs": 120},
    layer2_prompt_template=open("my_prompt.txt").read(),
    budget_usd=100,           # auto-downgrades at 80%, halts at 100%
)
```

Disable it entirely if you want a pure offline router:

```python
Router(layer2_enabled=False)
```

---

## Model registry

**No model name or price is hardcoded.** Everything lives in YAML.

```bash
dmr models                              # see what's loaded
dmr models load my-models.yaml --replace
dmr models export --output snapshot.yaml
```

```yaml
# my-models.yaml
providers:
  groq:
    api_key_env: GROQ_API_KEY
    tiers:
      low:    llama-3.3-8b-instant
      medium: llama-3.3-70b-versatile
      high:   llama-3.3-70b-versatile
models:
  llama-3.3-8b-instant:
    cost: { input_per_1m: 0.05, output_per_1m: 0.08 }
    capabilities: { context_window: 128000, supports_function_calling: true }
```

Or programmatically:

```python
from classifier import register_provider, register_model_cost, ModelTier

register_provider("groq", {
    ModelTier.LOW:  "llama-3.3-8b-instant",
    ModelTier.HIGH: "llama-3.3-70b-versatile",
})
register_model_cost("llama-3.3-70b-versatile", input_per_1m=0.59, output_per_1m=0.79)
```

Override priority: `Router(registry=...)` → `DMR_REGISTRY` env var → bundled `default.yaml`.

---

## Integrations

| Framework | Module | One-line use |
|-----------|--------|-------------|
| **LangChain** | `classifier.integrations.langchain` | `get_chat_model(task)` or `DynamicChatModel()` |
| **CrewAI** | `classifier.integrations.crewai` | `pick_llm_for_task(task)` or `DynamicLLM()` |
| **AutoGen** | `classifier.integrations.autogen` | `get_autogen_llm_config(task)` |
| **OpenAI Agents** | `classifier.integrations.autogen` | `get_openai_agent_model(task)` |
| **Google ADK** | `classifier.integrations.adk` | `before_model_callback=dynamic_model_selector` |
| **LlamaIndex** | `classifier.integrations.llamaindex` | `get_llm(task)` or `DynamicLLM()` |
| **Pydantic AI** | `classifier.integrations.pydantic_ai` | `get_model_string(task)` or `get_agent(task)` |
| **DSPy** | `classifier.integrations.dspy` | `get_lm(task)` or `with route(task): ...` |
| **Haystack** | `classifier.integrations.haystack` | `get_generator(task)` |
| **Semantic Kernel** | `classifier.integrations.semantic_kernel` | `get_chat_service(task)` |
| **smolagents (HF)** | `classifier.integrations.smolagents` | `get_model(task)` or `DynamicModel()` |

```python
# CrewAI example — every call this agent makes is routed dynamically
from crewai import Agent
from classifier.integrations.crewai import DynamicLLM

agent = Agent(role="Analyst", goal="...", llm=DynamicLLM())
```

---

## CLI reference

```bash
# Classify
dmr classify "task text"                       # one-shot
dmr classify --preset healthcare "Patient MRN…"

# Train Layer 3
dmr train --auto                               # bootstrap from logs
dmr train --data examples.jsonl                # train on labeled JSONL
dmr generate-data --domain legal --per-slot 50 # synthesize via Gemini

# Customize Layer 1 keywords
dmr keywords add --domain legal --type reasoning --keywords "tort,liable"
dmr keywords list
dmr keywords remove --domain legal --keyword "tort"
dmr keywords suggest --since 30d               # mine from your logs

# Inspect
dmr config show                                # effective config + L3 status
dmr config validate                            # validate dmr.yaml
dmr doctor                                     # env / dep / readiness check
dmr stats                                      # routing distribution
dmr stats cost --since 7d                      # cost breakdown
dmr models                                     # registry inventory

# Eval
dmr eval --data test.jsonl                     # accuracy + tier distribution

# Other
dmr init                                       # scaffold dmr.yaml
dmr presets                                    # list domain presets
dmr benchmark                                  # local p50/p95/p99 latency
dmr version
```

---

## Production checklist

Before going live with serious traffic:

- [ ] **Override the bundled registry.** Bundled prices go stale fast. `dmr models export > my-models.yaml`, edit, then `Router.from_registry("my-models.yaml")`.
- [ ] **Train Layer 3 on your data.** Run `dmr train --auto` after a few hundred logged decisions. Reduces L2 calls another 60–80%.
- [ ] **Pin a small budget initially.** `Router(budget_usd=100)` and watch `dmr stats cost`.
- [ ] **Set a tight L2 circuit breaker.** `failure_threshold=3, cooldown_secs=120` so a provider outage doesn't drain your wallet.
- [ ] **Configure decision logging** to an immutable backend (S3 + object lock, or write-only Kafka) for audit trails.
- [ ] **Run `dmr doctor` in CI.** Fail the build on any `[x]`.
- [ ] **Use `ShadowMode`** when changing routing config — runs old and new in parallel, logs diffs without affecting users.
- [ ] **Pin the package version** in your lock file. Semver — minor bumps may include behavior changes for unset config defaults.

---

## FAQ

### 1. Why not just use LiteLLM?
LiteLLM is a unified SDK — you specify which model to call and it handles the API call to that provider. We do the step *before*: deciding *which* model is the right one in the first place. They stack perfectly: `decision = router.classify(task)` → pass `decision.model_name` to LiteLLM. We are not a proxy or an SDK.

### 2. How does Layer 3 (the ML classifier) actually work?
A frozen `sentence-transformers/all-MiniLM-L6-v2` encoder produces a 384-dim embedding of the task. A small calibrated MLP head (~150 KB on disk) maps that embedding to `(task_type, complexity, confidence)`. Inference is ~15 ms on CPU, no API calls. You can train it three ways:
- **zero-shot** (default, no data needed) — uses bundled reference examples
- **labeled data** you provide via `dmr train --data your.jsonl`
- **auto-labels** mined from your production logs via `dmr train --auto` (uses 8 Snorkel-style weak labeling functions)

### 3. Is this safe for healthcare / HIPAA?
The package itself is built with PHI exposure in mind:
- PII spans (SSN, MRN, DOB, email, phone, JWT, API keys) are **auto-redacted** from `task_preview` and `error_message` before any logging
- Healthcare keyword pack (`HIPAA`, `clinical`, `prior_auth`, `lab analysis`) bundled out of the box
- All telemetry is **opt-in** — nothing leaves your machine unless you wire a backend
- Self-hosted — no SaaS dependency

That said: **using this package does not, by itself, make your application HIPAA-compliant.** You still need a BAA with your LLM provider, encrypted storage for any logs you keep, and the usual access controls. The router reduces the attack surface but doesn't replace your compliance work.

### 4. What does Layer 2 (the LLM fallback) cost?
Layer 2 only fires when both Layer 1 and Layer 3 are uncertain — typically **~10% of traffic** in production. Default L2 model is `gemini-2.5-flash-lite` at ~$0.000075 per 1k input tokens. A classification call uses ~100 tokens of context → **~$0.0000075 per L2 call**. Routing 1 million tasks costs roughly $7.50 in L2 fees, worst case. The savings on the 1 million downstream LLM calls dwarf this by ~1000×.

### 5. Can I run fully offline (no LLM API at all)?
Yes. Set `layer2_enabled=False` in your `.env` or pass it to `Router(layer2_enabled=False)`. The router now uses only Layer 1 (keyword heuristics, fully deterministic) and Layer 3 (local `sentence-transformers` inference, no API). Tasks never leave your machine. Useful for air-gapped environments, regulated workloads, or running the router on a private model gateway.

---

## We don't phone home

> **`dynamic-model-router` collects zero telemetry on its own.** No usage data, model names, error reports — nothing about your usage ever leaves your machine to us or anyone else.

The only network calls happen when **you** ask for them: Layer 2 → your LLM provider, `Router(registry="https://...")` → that URL, or your configured logger backend forwarding decisions to *your* DB.

(Not to be confused with `DMR_TELEMETRY=1` — that's a flag *you* set to get richer logs about *your own* routing. The data stays in your environment.)

---

## License

MIT — see [LICENSE](LICENSE).

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md). **Do not** open a public issue.

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). All contributors agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Changelog & roadmap

[CHANGELOG.md](CHANGELOG.md) · [ROADMAP.md](ROADMAP.md)
