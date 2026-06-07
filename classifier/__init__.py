"""dynamic-model-router — public package entry point.

This module is intentionally tiny: it only re-exports symbols. All business
logic lives in named submodules so the cascade is easy to navigate:

    classifier/
    ├── pipeline/                cascade business logic (classify_task)
    ├── router.py                Router OO API
    ├── core/                    domain types, exceptions, model registry
    ├── layers/                  L1 (keywords), L2 (LLM), L3 (ML head)
    ├── ml/                      training pipeline + auto-labeler + miner
    ├── infra/                   config, cache, telemetry, cost, PII
    ├── integrations/            framework adapters (LangChain, CrewAI, ...)
    ├── presets/                 domain bundles (healthcare, legal, fintech)
    └── cli.py                   `dmr` command-line interface
"""

from __future__ import annotations

__version__ = "0.4.0"

import logging as _logging

# PEP 282: libraries must NOT configure logging — only attach a NullHandler so
# "no handlers" warnings don't fire if the host app doesn't configure logging.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# ── Auto-load the bundled / configured registry at import time ───────────────
# Honors DMR_REGISTRY env var and DMR_NO_DEFAULT_REGISTRY=1 to opt out.
from classifier.core.registry_loader import _auto_load_at_import as _registry_auto_load

_registry_auto_load()

# ── Public re-exports (alphabetised by source module within sections) ────────

# Core types and exceptions
from classifier.core.exceptions import (
    ClassificationError,
    ConfigurationError,
    LayerNotAvailableError,
    UnsupportedProviderError,
)
from classifier.core.registry import (
    MODEL_REGISTRY,
    TIER_MATRIX,
    capabilities_for,
    list_models,
    list_providers,
    register_provider,
)
from classifier.core.registry_loader import (
    clear_registry,
    export_registry,
    export_to_yaml,
    load_registry,
)
from classifier.core.types import (
    ClassificationDecision,
    ContextSignals,
    ModelTier,
    TaskComplexity,
    TaskType,
    list_tier_levels,
    register_complexity,
    register_task_type,
    set_tier_levels,
)

# Pluggable extensions and pipeline
from classifier.experiments import ABTest, ShadowMode
from classifier.hooks import (
    clear_hooks,
    hook_manager,
    register_hook,
    unregister_hook,
)

# Apply the env-configured posture (DMR_SAVINGS_LEVEL) once at import — this is a
# coarse, process-global operating point. Router(savings_level=...) is per-call.
from classifier.infra.config import settings as _settings
from classifier.infra.cost_tracker import get_model_cost, register_model_cost
from classifier.infra.decision_logger import read_decisions
from classifier.infra.feedback import record_feedback
from classifier.infra.outcome_logger import (
    OutcomeRecord,
    join_decisions_outcomes,
    log_outcome,
    prune_old_outcomes,
    read_outcomes,
)
from classifier.infra.tokenizers import count_tokens, register_tokenizer

# Framework-neutral agentic routing + universal API (works for any agent loop).
from classifier.integrations._agentic import (
    AgentCallContext,
    report,
    report_agent_outcome,
    route,
    route_agent_call,
    route_scope,
)
from classifier.layers.layer1 import classify_layer1, detect_pii  # re-exported (test fixtures)
from classifier.layers.layer1.keyword_pack import KeywordPack
from classifier.layers.layer3 import register_strategy as register_l3_strategy
from classifier.layers.plugin import list_layers, register_layer, unregister_layer
from classifier.logger_backends import (
    JSONLLoggerBackend,
    KafkaLoggerBackend,
    MultiLoggerBackend,
    NullLoggerBackend,
    S3LoggerBackend,
    StdoutLoggerBackend,
    WebhookLoggerBackend,
)
from classifier.ml.auto_labeler import (
    DEFAULT_LFS,
    AutoLabeler,
    Label,
    LabelingFunction,
)
from classifier.ml.embeddings import current_embedding_model, set_embedding_model

# The cascade itself
from classifier.pipeline.classify_task import (
    MAX_TASK_CHARS,
    classify_task,
    reset_last_decision,
)
from classifier.router import Router, classify

if getattr(_settings, "dmr_savings_level", 0):
    from classifier.routing.posture import apply_posture as _apply_posture

    _apply_posture(_settings.dmr_savings_level)

# ── @route_model decorator ───────────────────────────────────────────────────


def route_model(
    provider: str | None = None,
    *,
    task_arg: str = "task",
    fallback_model: str | None = None,
    inject_as: str = "model_name",
):
    """Decorator that classifies the task argument and injects the model name.

    The decorated function receives an extra keyword argument (default:
    ``model_name``) with the router-selected model name. Original positional
    and keyword args are passed through unchanged.

    Example:
        @route_model(provider="anthropic")
        def call_llm(task: str, model_name: str = "claude-sonnet-4-6"):
            client = anthropic.Anthropic()
            return client.messages.create(model=model_name, ...)

        result = call_llm("Compare metformin vs GLP-1 agonists")
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(fn)
            params = list(sig.parameters)

            if task_arg in kwargs:
                task_text = kwargs[task_arg]
            elif task_arg in params:
                idx = params.index(task_arg)
                task_text = args[idx] if idx < len(args) else ""
            else:
                task_text = args[0] if args else ""

            try:
                decision = classify_task(str(task_text), provider=provider)
                model = decision.model_name
            except Exception:
                if fallback_model:
                    model = fallback_model
                else:
                    raise

            kwargs.setdefault(inject_as, model)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "__version__",
    # Public OO API
    "Router",
    "classify",
    "KeywordPack",
    "route_model",
    "reset_last_decision",
    "MAX_TASK_CHARS",
    # Agentic routing (framework-neutral core + universal API)
    "route_scope",
    "route",
    "report",
    "route_agent_call",
    "report_agent_outcome",
    "AgentCallContext",
    # Domain
    "ClassificationDecision",
    "ContextSignals",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    "MODEL_REGISTRY",
    "TIER_MATRIX",
    # Exceptions
    "ClassificationError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "LayerNotAvailableError",
    # Registry management
    "register_provider",
    "list_providers",
    "list_models",
    "capabilities_for",
    "load_registry",
    "clear_registry",
    "export_registry",
    "export_to_yaml",
    "register_model_cost",
    "get_model_cost",
    # Open enums
    "register_task_type",
    "register_complexity",
    "set_tier_levels",
    "list_tier_levels",
    # Embeddings (Layer 3)
    "set_embedding_model",
    "current_embedding_model",
    # Hooks + experiments
    "register_hook",
    "unregister_hook",
    "clear_hooks",
    "hook_manager",
    "ABTest",
    "ShadowMode",
    # Outcomes / decision telemetry
    "OutcomeRecord",
    "log_outcome",
    "read_outcomes",
    "join_decisions_outcomes",
    "prune_old_outcomes",
    "read_decisions",
    # Auto-labeler (weak supervision)
    "AutoLabeler",
    "Label",
    "LabelingFunction",
    "DEFAULT_LFS",
    # Plugin system
    "register_layer",
    "unregister_layer",
    "list_layers",
    "register_l3_strategy",
    # Tokenizer
    "register_tokenizer",
    "count_tokens",
    # Logger backends (pluggable telemetry sinks)
    "JSONLLoggerBackend",
    "StdoutLoggerBackend",
    "WebhookLoggerBackend",
    "KafkaLoggerBackend",
    "S3LoggerBackend",
    "MultiLoggerBackend",
    "NullLoggerBackend",
    # Free function (kept for back-compat)
    "classify_task",
    # Misc
    "record_feedback",
]
