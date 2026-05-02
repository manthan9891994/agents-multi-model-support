# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
