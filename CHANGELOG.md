# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0](https://github.com/manthan9891994/agents-multi-model-support/compare/v0.4.0...v0.5.0) (2026-06-07)


### Features

* agentic-efficiency redesign + L3 savings dial (0.5.0) ([7083a27](https://github.com/manthan9891994/agents-multi-model-support/commit/7083a273416f49debe1b33e7e14c7414340e20d5))
* **ml:** bundle MiniLM encoder for fully offline Layer 3 ([a3d3c4c](https://github.com/manthan9891994/agents-multi-model-support/commit/a3d3c4c119b2ae08c04091073702b2d7e9bb6647))


### Documentation

* add competitor comparison table + FAQ ([071d10b](https://github.com/manthan9891994/agents-multi-model-support/commit/071d10b993dc00176a7edc9676ebb8d72ef1b9ad))
* add demo GIF showing 94% cost reduction on real traffic ([2999c9c](https://github.com/manthan9891994/agents-multi-model-support/commit/2999c9c406b5a333395257a548ae35035260e32d))
* reframe cost comparison from 'GPT-4o' to 'frontier reasoning models' ([0cf0659](https://github.com/manthan9891994/agents-multi-model-support/commit/0cf0659675c1253aae3d5b8feb5223f1a3e52139))
* replace specific-percentage claim in demo GIF alt text with benchmark-source description ([b6b27e8](https://github.com/manthan9891994/agents-multi-model-support/commit/b6b27e85b4d2eaf5f7b997f08e0526571accdfd0))

## [Unreleased]

### Added
- **Agentic cost levers + framework-neutral universal API (0.5.0).** A new layer for *agentic* workloads, where the cost is dominated by input context (not model tier) and cheap models fail at tool-driving. All opt-in; defaults preserve today's behavior.
  - **Universal API:** `from classifier import route_scope, route, report` — works for any framework or bespoke agent loop (contextvar-based, async-safe). Framework adapters (ADK, …) are now thin translators over the neutral core `classifier.integrations._agentic` (`AgentCallContext`, `route_agent_call`, `report_agent_outcome`).
  - **Posture dial `DMR_SAVINGS_LEVEL` (0–4)** / `Router(savings_level=…)` composes the levers: **1** cache+effort, **2** +context-prune, **3** +capability-gated dispatch-downgrade+escalate, **4** max. Per-lever flags `DMR_CACHE_AWARE`, `DMR_CONTEXT_REDUCTION`, `DMR_EFFORT_ROUTING`, `DMR_MODEL_ROUTING`, `DMR_ESCALATE_ON_FAILURE`, `DMR_ROUTING_SCOPE`.
  - **Capability gate** — a tool-driving call is never routed to a model with `tool_calling: basic` (registry `capabilities`); fixes the agentic "no-answer" failures.
  - **Effort routing** — `decision.effort` (none/low/high) for reasoning-capable models; adapters map to provider thinking budgets.
  - **Scope stickiness** — one decision per turn (default scope keeps `call`) so models don't thrash mid-loop (preserves provider prompt caches).
  - **Escalate-on-failure** — `classifier.quality.failure_detect` flags refusals/no-answer; the scope escalates to the ceiling.
  - **Context reduction** — `classifier.context.reduce.prune_context` trims stale tool outputs (the biggest input-cost lever).
  - **Cache-aware cost** — `cost_tracker.estimate_cost(..., cached_fraction)` + `switch_penalty`; registry `cache` block per model.
  - **Configured model = ceiling** — routing never exceeds the agent's configured model.
  - CLI: `dmr frontier` shows the cost↔posture frontier. Registry models gain `capabilities.tool_calling`/`reasoning` + `cache`.
- **Layer 3 quality↔savings dial (`L3_DMR_SAVINGS_LEVEL`).** A single integer that biases L3's chosen tier toward cheaper models without retraining: `0` = quality (L3's natural tier), each step shifts the tier one notch cheaper (HIGH→MEDIUM→LOW), clamped at LOW. Set via env `L3_DMR_SAVINGS_LEVEL` or `Router(layer3_savings_level=…)`. Default `0` (no behavior change). Lets one trained head run anywhere on the cost/quality frontier. Applied at the L3 dispatcher (`classify_layer3`), so it covers every strategy (head/zeroshot/custom).

### Fixed
- **`Router(layer1_enabled=...)` no longer crashes.** `Settings` was missing the `layer1_enabled` field, so `_apply_overrides` raised `ValueError: "Settings" object has no field "layer1_enabled"` on every `classify()` when L1 was toggled. Added the field (default `True`) and made the cascade honor `layer1_enabled=False` — it falls through to L3/L2, or returns a safe MEDIUM default if those are off too.

### Added
- **Bundled Layer 3 encoder (offline by default):** `all-MiniLM-L6-v2` now ships inside the package (`classifier/ml/models/all-MiniLM-L6-v2/`) and is included in the wheel/sdist. `ml.embeddings` resolves the bundled copy first — no Hugging Face download on first use, works on air-gapped / restricted networks. A custom `DMR_EMBEDDING_MODEL` (local dir or HF id) is still honored. The `[ml]` extra remains required for the runtime libraries (`sentence-transformers`, `torch`, `scikit-learn`); only the model *download* is removed.
- **Continual learning — PR 1 of 4 (outcome logger):**
  - `decision_id: str` (UUID4 hex prefix) field on `ClassificationDecision` — joins decision ⨝ outcome streams.
  - `Router.report_outcome(decision_id, tokens_in, tokens_out, wall_ms, success, user_retried, user_escalated_model, user_feedback, edit_distance, error_message)` API.
  - `OutcomeRecord` dataclass with append-only JSONL storage at `routing_outcomes.jsonl`.
  - Pluggable backend via `Router(outcome_logger=KafkaLoggerBackend(...))` — same backends as `decision_logger`.
  - `read_outcomes(since=, until=, decision_ids=)` and `join_decisions_outcomes(decisions, outcomes)` helpers.
  - **Auto-instrumentation** of three integrations — outcomes are reported automatically:
    - `DynamicChatModel` (LangChain) — wraps `invoke` / `stream` with token-count + wall-time capture.
    - `DynamicLLM` (CrewAI) — wraps `call` with success/error reporting.
    - ADK — `report_model_outcome` paired with `dynamic_model_selector` via `after_model_callback`.
  - 13 new unit tests proving the join logic, pluggable backends, and auto-instrumentation.
- `examples/benchmark_cost_savings.ipynb` — runs 1,000 representative prompts and produces a cost-comparison bar chart + tier-distribution pie. No live LLM calls in default mode.
- `CITATION.cff` — academic citation metadata (CFF 1.2.0 spec).
- Six new agent-framework integrations:
  - **LlamaIndex** (`classifier.integrations.llamaindex`) — `get_llm(task)` + `DynamicLLM`
  - **Pydantic AI** (`classifier.integrations.pydantic_ai`) — `get_model_string(task)` + `get_agent(task, **kw)`
  - **DSPy** (`classifier.integrations.dspy`) — `get_lm(task)` + `with route(task): …` context manager
  - **Haystack** (`classifier.integrations.haystack`) — `get_generator(task)`
  - **Semantic Kernel** (`classifier.integrations.semantic_kernel`) — `get_chat_service(task)`
  - **smolagents** (`classifier.integrations.smolagents`) — `get_model(task)` + `DynamicModel`
- Optional extras: `[llamaindex]`, `[pydanticai]`, `[dspy_ext]`, `[haystack]`, `[semantickernel]`, `[smolagents]`
- 24 new mocked unit tests proving each integration honors `provider=`, `fallback_model=`, and DSPy's context-manager restore.
- YAML-driven model registry: providers, models, costs, and capabilities live in `classifier/data/registry/default.yaml`. Zero hardcoded model names or prices in Python.
- `dmr models {list, export, load, pull, clear}` CLI for runtime registry management.
- `Router.from_registry(path | URL | dict)` and `Router.load_registry(...)` classmethods.
- `Router(registry=...)` constructor argument.
- Environment overrides: `DMR_REGISTRY=<path>` and `DMR_NO_DEFAULT_REGISTRY=1`.
- Async API: `await router.aclassify(task)` and `await router.aclassify_batch([...])`
- `Router.estimate_cost(task)` — dry-run cost preview before any API call
- `route_model` decorator — wrap existing functions with dynamic model selection
- `dmr doctor` — diagnose configuration / dependency / model-file issues
- `dmr version` — package + Python + dependency versions
- `dmr benchmark` — measure routing latency p50/p95/p99
- `dmr eval --data file.jsonl` — evaluate routing accuracy on a labeled dataset
- LangChain integration (`get_chat_model`, `DynamicChatModel`)
- AutoGen / OpenAI Agents SDK integration (`get_autogen_llm_config`, `get_openai_agent_model`, `DynamicModelRouter`)
- `ClassificationDecision.to_dict()` / `to_json()` / `from_dict()` / `from_json()` serde
- `KeywordPack` builder API for programmatic L1 vocabulary injection
- Domain presets: `healthcare` (full), `legal` and `fintech` (skeletons)
- Sample datasets: 30 legal + 30 fintech labeled tasks (`classifier/data/`)
- Quickstart Colab notebook (`examples/quickstart.ipynb`)
- Type stubs (`classifier/__init__.pyi`, `router.pyi`) + `py.typed` marker (PEP 561)
- `__version__` attribute
- OpenTelemetry trace spans (no-op when `opentelemetry-api` isn't installed)
- Layer 2 circuit breaker (5 failures in 30s → trip OPEN for 60s)
- Layer 2 connection pooling (single shared `genai.Client`)
- Layer 2 retry with `Retry-After` header support
- Regex DoS protection on user-supplied PII patterns
- Input length guard (`DMR_MAX_TASK_CHARS`, default 32K)
- API key validation with actionable `ConfigurationError`
- L3 missing-model warning with concrete remediation hints
- Real token counts from API response (replaces word-count proxy in cost tracking)
- GitHub Actions CI: pytest matrix (3.10/3.11/3.12) + Linux/macOS/Windows + ruff + mypy + pip-audit + nbmake + coverage threshold
- PyPI publish workflow with trusted publishing (OIDC)
- Dependabot config

### Changed
- Library code no longer calls `logging.basicConfig` at import time (PEP 282 compliance) — host applications retain control of logging.
- Router keyword pack / PII pattern registration deferred from `__init__` to first
  `classify()` call — prevents state leak between Router instances.
- `ClassificationError` now carries `layer`, `task_preview`, and `suggestion` fields.

### Fixed
- `_last_decision` global now lock-protected and clearable via `reset_last_decision()`.
- Cost tracker tests no longer leak `CLASSIFIER_TEST_MODE` into other tests.
- Layer 2 timeout test patches the correct module path.

## [0.1.0] - 2026-04-15

Initial release: 3-layer cascade classifier (L1 keyword → L3 ML head → L2 Gemini).
