# dynamic-model-router

[![CI](https://github.com/manthanvaghela/dynamic-model-router/actions/workflows/ci.yml/badge.svg)](https://github.com/manthanvaghela/dynamic-model-router/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![Python versions](https://img.shields.io/pypi/pyversions/dynamic-model-router.svg)](https://pypi.org/project/dynamic-model-router/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/dynamic-model-router/month)](https://pepy.tech/project/dynamic-model-router)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A 3-layer cascade classifier that routes each task to **the cheapest model that can handle it well** — before the agent makes an API call.

One classifier, three providers, zero framework changes. Drop it into Google ADK, CrewAI, LangChain, AutoGen, or any Python agent.

> **No telemetry.** This package never phones home. `dmr doctor` and `dmr stats` only read local files.

```python
from dynamic_model_router import classify

decision = classify("Implement a binary search")
print(decision.tier, decision.model_name, decision.layer_used)
# medium  gemini-2.5-flash  layer3
```

---

## The problem

Every agent is hardcoded to one model. "Write a README" and "design a distributed fault-tolerant system" both hit `gemini-2.5-pro` — wasting money and adding latency where it isn't needed.

## The solution

Classify the task **before** the model call. Route to the cheapest model that can handle it.

```
User message
  → Layer 1: keyword + heuristic     (<1ms,    free,    always runs)
      └─ conf < 0.75 → Layer 3: ML classifier  (~15ms, free, abstain-capable)
          └─ conf < 0.75 → Layer 2: Gemini Flash Lite  (~500ms, $0.0001, LLM fallback)
  → Tier: LOW / MEDIUM / HIGH → model name for chosen provider
```

---

## Install

```bash
pip install dynamic-model-router                  # core (L1 + L2 only)
pip install 'dynamic-model-router[ml]'            # add Layer 3 ML classifier
pip install 'dynamic-model-router[adk]'           # add Google ADK integration
pip install 'dynamic-model-router[all]'           # everything
```

Set provider API key in `.env`:

```env
GOOGLE_API_KEY=your_key_here
DEFAULT_PROVIDER=google         # or anthropic, openai
```

---

## Quickstart

### Zero-config

```python
from dynamic_model_router import classify
decision = classify("Compare microservices vs monolith for a 10-engineer team")
print(decision.tier.value, decision.model_name)
```

### Customized router

```python
from dynamic_model_router import Router, KeywordPack, TaskType

# Add domain vocabulary
legal_pack = (KeywordPack.builder("legal")
              .add(TaskType.DOC_CREATION, ["clause", "indemnification", "non-compete"])
              .add(TaskType.REASONING,    ["precedent", "statute", "doctrine"])
              .escalator("constitutional", weight=2)
              .build())

router = Router(
    providers=["anthropic", "google"],     # failover order
    extra_keyword_packs=[legal_pack],
    layer3_enabled=True,
    escalation_threshold=0.75,
)

decision = router.classify("Draft a non-compete clause with reasonable scope")
```

### Domain presets

```python
from dynamic_model_router import Router

router = Router.from_preset("healthcare")
decision = router.classify("Patient MRN: 12345678 has elevated AST")
print(decision.compliance_flag)   # True — PII detected, MEDIUM tier minimum
```

Available: `healthcare`, `legal`, `fintech`. The healthcare preset ships fully populated; legal and fintech are skeletons you extend with your domain vocab.

### YAML config

```yaml
# dmr.yaml
providers: [anthropic, google]
layer3_enabled: true
escalation_threshold: 0.75

keyword_packs:
  - name: my_domain
    packs:
      reasoning: [my_term_1, my_term_2]
      doc_creation: [report_type_a]
```

```python
router = Router.from_yaml("dmr.yaml")
```

### Train on your own data

```python
from dynamic_model_router import Router

router = Router()
metadata = router.train(data="my_domain_examples.jsonl")
print(metadata["geo_mean_accuracy"])
```

The JSONL must have `{"task": str, "task_type": str, "complexity": str}` per line. Need at least 50 examples; 1,500+ recommended.

---

## CLI

```bash
dmr classify "Implement a REST API endpoint with input validation"
dmr classify "Patient MRN: 12345 has elevated AST" --preset healthcare

dmr train --data my_examples.jsonl
dmr generate-data --domain healthcare --per-slot 30   # synthetic data via Gemini

dmr stats --since 24h
dmr presets               # list available presets
dmr init                  # scaffold dmr.yaml
```

---

## Model Tiers

| Tier   | Google                | Anthropic           | OpenAI       | When                                              |
|--------|-----------------------|---------------------|--------------|---------------------------------------------------|
| LOW    | gemini-2.5-flash-lite | claude-haiku-4-5    | gpt-4o-mini  | Conversation, simple docs, trivial code           |
| MEDIUM | gemini-2.5-flash      | claude-sonnet-4-6   | gpt-4o       | Reasoning, standard APIs, analysis                |
| HIGH   | gemini-2.5-pro        | claude-opus-4-7     | gpt-4-turbo  | Complex architecture, research, hard code         |

Override per-instance:

```python
router = Router(model_registry={
    "google": {ModelTier.LOW: "gemini-1.5-flash"}
})
```

---

## Cascade Architecture

| Layer | What it does | Latency | Cost | Coverage |
|---|---|---|---|---|
| **L1** Keyword + heuristic | Regex / keyword matching, structural signals | <1ms | $0 | ~33% |
| **L3** ML classifier (frozen MiniLM + sklearn MLPs) | Real classification model | ~15ms | $0 | ~25% |
| **L2** Gemini Flash Lite | LLM with structured JSON output | ~500ms | $0.0001 | ~42% |

L1 → L3 → L2: L1 catches obvious cases for free, L3 catches the next slice without paying for an LLM call, L2 is the fallback for anything ambiguous to both.

---

## Layer 3 — what the ML classifier actually is

Not cosine similarity, not KNN, not vector search. A real **supervised multi-class classifier** with two heads.

```
User task: "What are contraindications for ACE inhibitors?"
         │
         ▼
┌──────────────────────────────────────────────┐
│ STEP 1 — Sentence Embedder (FROZEN)          │
│   Model: all-MiniLM-L6-v2 (22M params)       │
│   Output: 384-dim dense vector               │
└──────────────────────────────────────────────┘
         │
         ├──────────────────────┬───────────────┐
         ▼                      ▼               │
┌──────────────────┐   ┌──────────────────┐     │
│ HEAD 1 — MLP     │   │ HEAD 2 — MLP     │     │
│ task_type        │   │ complexity       │     │
│ 384 → 256 → 9    │   │ 384 → 256 → 4    │     │
└──────────────────┘   └──────────────────┘     │
         │                      │               │
         └──────────┬───────────┘               │
                    ▼                           │
       confidence = √(p_tt × p_cx)              │
                    │                           │
                    ▼                           │
          if conf >= 0.75: return  ─────────────┘
          else: abstain → cascade to L2
```

| Component | Type | Trainable? |
|---|---|---|
| MiniLM encoder | Pre-trained transformer | ❌ Frozen |
| Task type head | sklearn MLPClassifier (256) | ✅ On your data |
| Complexity head | sklearn MLPClassifier (256) | ✅ On your data |
| Sigmoid calibrator | Platt scaling | ✅ On held-out cal set |

**Why two heads:** predicting `(task_type × complexity)` jointly = 36 sparse classes. Two heads (9 + 4) need far less data per class. Geometric mean of probabilities penalizes asymmetric confidence.

**Why abstain:** L3 returns `None` when confidence < threshold. Better to spend $0.0001 on L2 than to misroute. L3 wins by being **conservative and confident**, not by intercepting everything.

### Training pipeline

```bash
dmr generate-data --domain healthcare --per-slot 30   # bootstrap (~$0.05 Gemini)
dmr train --data classifier/data/synthetic_tasks.jsonl
```

The training script:
1. Encodes all examples with frozen MiniLM
2. Three-way split: 70% train / 15% calibration / 15% test
3. Trains MLPs on train set
4. Wraps each MLP with `CalibratedClassifierCV(method="sigmoid")` on calibration set
5. Sweeps thresholds [0.50–0.95] on test, reports (intercept_rate, precision)
6. Saves bundle to `classifier/ml/models/head_v1.joblib`

### Strategies (`LAYER3_STRATEGY` env var)

| Strategy | Latency | Accuracy | Training data | Status |
|---|---|---|---|---|
| `zeroshot` | ~80ms | ~80% | None | ✅ Built |
| `head` | ~15ms | ~80% (calibrated) | 1,500+ | ✅ Built — **default** |
| `distilbert` | ~12ms | ~95% target | 5,000+ | ⏳ Roadmap |

---

## ADK integration

```python
from google.adk.agents import LlmAgent
from classifier.integrations.adk import dynamic_model_selector

agent = LlmAgent(
    name="MyAgent",
    model="gemini-2.5-flash",   # placeholder — replaced per-request
    before_model_callback=dynamic_model_selector,
)
```

That's the entire integration surface. The callback inspects each request, classifies the user's task, and overwrites `llm_request.model` before the API call.

See [`examples/adk_healthcare/`](examples/adk_healthcare/README.md) for four working healthcare agents (clinical Q&A, prior auth, lab analyzer, clinical note) and an `adk web` runner.

---

## Customization Hooks

| Hook | Constructor arg | Example |
|---|---|---|
| Layer toggle | `layer1_enabled` / `layer2_enabled` / `layer3_enabled` | `Router(layer3_enabled=False)` |
| Custom keywords | `extra_keyword_packs` | `Router(extra_keyword_packs=[my_pack])` |
| Custom PII patterns | `extra_pii_patterns` | `Router(extra_pii_patterns=[(regex, "[ACCT]")])` |
| Tier matrix override | `tier_matrix` | `Router(tier_matrix={(REASONING, SIMPLE): LOW})` |
| Model registry override | `model_registry` | `Router(model_registry={"google": {LOW: "..."}})` |
| Provider failover | `providers` | `Router(providers=["anthropic", "google"])` |
| Abstain threshold | `layer3_threshold` | `Router(layer3_threshold=0.85)` |
| Escalation threshold | `escalation_threshold` | `Router(escalation_threshold=0.65)` |
| Domain preset | `Router.from_preset(name)` | `Router.from_preset("healthcare")` |

All overrides are **per-instance**. Multiple Router instances with different configs coexist without polluting global state.

---

## PII Scrubbing

Patient data, account numbers, and personal identifiers must never leak to external LLMs. Layer 2 scrubs before every API call.

Built-in patterns: MRN, SSN, DOB, phone, email, names with title (Dr./Mr./Mrs.). Strict mode adds all-caps names and addresses.

Add your own:

```python
import re

router = Router(extra_pii_patterns=[
    (re.compile(r"\bACCT-\d{6}\b"),     "[ACCT]"),
    (re.compile(r"\bMyAppUserID:\d+"),  "[USERID]"),
])
```

The healthcare preset adds NPI and encounter number patterns. The fintech preset adds credit card and IBAN patterns.

---

## Project structure

```
classifier/                 # the package (also exported as `dynamic_model_router`)
├── router.py               # Router class — main API
├── core/                   # types, registry, exceptions
├── layers/                 # L1, L2, L3
│   └── layer1/keyword_pack.py    # KeywordPack builder
├── ml/                     # training pipeline (optional [ml] extra)
├── infra/                  # cache, config, pii_scrubber, decision_logger
├── presets/                # healthcare / legal / fintech
├── integrations/
│   └── adk.py              # Google ADK callback
└── cli.py                  # `dmr` command

examples/adk_healthcare/    # NOT installed — demo agents
tests/                      # 180 passing tests
plan_docs/                  # design docs
```

---

## Status

- 180 passing tests
- Stage 2 ML classifier (frozen MiniLM + calibrated sklearn MLPs) trained on 2,026 examples → 79% accuracy
- 3 domain presets (healthcare populated, legal/fintech skeletons)
- Google ADK integration shipped; CrewAI/LangChain on roadmap
- Stage 3 fine-tuned DistilBERT not yet built — frozen-MLP approach is sufficient at current data scale

## License

MIT
