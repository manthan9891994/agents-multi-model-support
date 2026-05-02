"""High-level Router API — the main user-facing entry point.

Wraps the flat `classify_task()` function with an OO interface that supports
per-instance overrides for keyword packs, tier matrix, model registry,
PII patterns, providers, and layer toggles.

Example:
    from classifier import Router

    router = Router(
        providers=["anthropic", "google"],
        layer3_enabled=False,
    )
    decision = router.classify("Implement binary search")
    print(decision.tier, decision.model_name)
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ClassificationDecision, ContextSignals, ModelTier, TaskComplexity, TaskType
    from classifier.layers.layer1.keyword_pack import KeywordPack

# Single lock guards mutation of global state during a classify() call
_router_lock = threading.RLock()


class Router:
    """Configurable cascade router. Each instance has its own overrides.

    All keyword args are optional — defaults match the package-wide defaults.

    Args:
        providers:           List of providers in failover order, e.g. ["anthropic", "google"].
        extra_keyword_packs: Custom L1 KeywordPack instances merged with built-ins.
        extra_pii_patterns:  List of (compiled regex, replacement-token) tuples.
        tier_matrix:         Override default (TaskType, TaskComplexity) → ModelTier mapping.
        model_registry:      Override default {provider: {tier: model_name}} mapping.
        layer1_enabled:      Disable L1 keyword/heuristic layer.
        layer2_enabled:      Disable L2 LLM fallback layer.
        layer3_enabled:      Disable L3 ML head layer.
        layer1_threshold:    Confidence below which L1 escalates (default 0.75).
        layer3_threshold:    Confidence below which L3 abstains (default 0.75).
        budget_usd:          Monthly budget cap (USD).
        cache_enabled:       In-memory result caching (default True).
    """

    def __init__(
        self,
        *,
        providers: Optional[list[str]] = None,
        extra_keyword_packs: Optional[list["KeywordPack"]] = None,
        extra_pii_patterns: Optional[list[tuple]] = None,
        tier_matrix: Optional[dict] = None,
        model_registry: Optional[dict] = None,
        layer1_enabled: Optional[bool] = None,
        layer2_enabled: Optional[bool] = None,
        layer3_enabled: Optional[bool] = None,
        escalation_threshold: Optional[float] = None,
        layer3_threshold: Optional[float] = None,
        budget_usd: Optional[float] = None,
        cache_enabled: Optional[bool] = None,
        # ── Extensibility hooks (v2) ──────────────────────────────────────────
        layer2_provider: Optional[str] = None,
        layer2_model:    Optional[str] = None,
        layer3_embedding_model: Optional[str] = None,
        model_costs:     Optional[dict] = None,
        custom_classifier: Optional[Any] = None,
        pre_classify_hooks:  Optional[list[Any]] = None,
        post_classify_hooks: Optional[list[Any]] = None,
        on_error_hooks:      Optional[list[Any]] = None,
        pii_policy: Optional[dict] = None,
        l2_retry_policy: Optional[dict] = None,
        l2_circuit_breaker: Optional[dict] = None,
        l1_weights: Optional[dict] = None,
        tokenizer: Optional[Any] = None,
        latency_budget_ms: Optional[float] = None,
        residency: Optional[str] = None,
        cache_backend: Optional[Any] = None,
        decision_logger: Optional[Any] = None,
        layer2_prompt_template: Optional[str] = None,
        registry: Optional[Any] = None,    # path / URL / dict to load at construction
    ):
        self.providers           = providers or []
        self.extra_keyword_packs = extra_keyword_packs or []
        self.extra_pii_patterns  = extra_pii_patterns or []
        self.tier_matrix         = tier_matrix or {}
        self.model_registry      = model_registry or {}
        self.layer1_enabled       = layer1_enabled
        self.layer2_enabled       = layer2_enabled
        self.layer3_enabled       = layer3_enabled
        self.escalation_threshold = escalation_threshold
        self.layer3_threshold     = layer3_threshold
        self.budget_usd          = budget_usd
        self.cache_enabled       = cache_enabled

        # Extensibility v2
        self.layer2_provider = layer2_provider
        self.layer2_model    = layer2_model
        self.layer3_embedding_model = layer3_embedding_model
        self.model_costs     = model_costs or {}
        self.custom_classifier   = custom_classifier
        self.pre_classify_hooks  = pre_classify_hooks  or []
        self.post_classify_hooks = post_classify_hooks or []
        self.on_error_hooks      = on_error_hooks      or []
        self.pii_policy          = pii_policy
        self.l2_retry_policy     = l2_retry_policy
        self.l2_circuit_breaker  = l2_circuit_breaker
        self.l1_weights          = l1_weights
        self.tokenizer           = tokenizer
        self.latency_budget_ms   = latency_budget_ms
        self.residency           = residency
        self.cache_backend       = cache_backend
        self.decision_logger     = decision_logger
        self.layer2_prompt_template = layer2_prompt_template
        self.registry = registry

        # Apply registry override (path / URL / dict) at construction
        if self.registry is not None:
            from classifier.core.registry_loader import load_registry
            load_registry(self.registry)

        # Apply extensibility settings at construction
        if self.model_costs:
            from classifier.infra.cost_tracker import register_model_cost
            for model_name, rates in self.model_costs.items():
                register_model_cost(
                    model_name,
                    input_per_1m=rates.get("input", 0.25),
                    output_per_1m=rates.get("output", 0.75),
                )
        if self.layer3_embedding_model:
            from classifier.ml.embeddings import set_embedding_model
            set_embedding_model(self.layer3_embedding_model)

        # Inject extras at construction. Both registries dedupe by identity/name,
        # so callers can safely instantiate multiple Routers with overlapping
        # packs. Two Routers with DIFFERENT packs will see each other's packs
        # in L1 — this is documented (see Router.classify docstring); use
        # `Router.from_preset()` if you want isolation per use case.
        if self.extra_keyword_packs:
            from classifier.layers.layer1.keyword_pack import register_extra_packs
            register_extra_packs(self.extra_keyword_packs)
        if self.extra_pii_patterns:
            from classifier.infra import pii_scrubber
            pii_scrubber.register_extra_patterns(self.extra_pii_patterns)

    # ── Primary API ──────────────────────────────────────────────────────────

    def classify(
        self,
        task: str,
        history: Optional[list[str]] = None,
        context_signals: "Optional[ContextSignals]" = None,
        provider: Optional[str] = None,
        hook_context: Optional[dict] = None,
        tenant_config: Optional[dict] = None,
    ) -> "ClassificationDecision":
        """Classify a task and return the routing decision.

        Args:
            tenant_config: Per-call config overrides (multi-tenant deployments).
                Supported keys: providers, budget_usd, layer1_enabled,
                layer2_enabled, layer3_enabled, latency_budget_ms,
                residency, pii_policy, model_registry, tier_matrix.

        If `providers` was set at construction, tries each in order on failure.
        """
        from classifier import classify_task

        # Per-call tenant overrides — build a temporary Router for just this call
        if tenant_config:
            tenant_router = self.with_overrides(**tenant_config)
            return tenant_router.classify(
                task, history=history, context_signals=context_signals,
                provider=provider, hook_context=hook_context,
            )

        resolved = provider or (self.providers[0] if self.providers else None)

        # Inject Router-level config into hook_context so the cascade can read it
        merged_ctx: dict = dict(hook_context or {})
        if self.pii_policy is not None:
            merged_ctx.setdefault("pii_policy", self.pii_policy)
        if self.latency_budget_ms is not None:
            merged_ctx.setdefault("latency_budget_ms", self.latency_budget_ms)
        if self.residency is not None:
            merged_ctx.setdefault("residency", self.residency)

        with self._apply_overrides():
            with self._apply_hooks():
                try:
                    return classify_task(
                        task,
                        provider=resolved,
                        history=history,
                        context_signals=context_signals,
                        hook_context=merged_ctx,
                        custom_classifier=self.custom_classifier,
                    )
                except Exception as exc:
                    if not self.providers or len(self.providers) <= 1:
                        raise
                    for fallback in self.providers[1:]:
                        try:
                            return classify_task(
                                task, provider=fallback,
                                history=history, context_signals=context_signals,
                                hook_context=merged_ctx,
                                custom_classifier=self.custom_classifier,
                            )
                        except Exception:
                            continue
                    raise exc

    @contextmanager
    def _apply_hooks(self):
        """Register Router-scoped hooks and unregister on exit."""
        from classifier.hooks import hook_manager
        registered: list[tuple[str, Any]] = []
        try:
            for fn in self.pre_classify_hooks:
                hook_manager.register("pre_classify", fn)
                registered.append(("pre_classify", fn))
            for fn in self.post_classify_hooks:
                hook_manager.register("post_classify", fn)
                registered.append(("post_classify", fn))
            for fn in self.on_error_hooks:
                hook_manager.register("on_error", fn)
                registered.append(("on_error", fn))
            yield
        finally:
            for kind, fn in registered:
                hook_manager.unregister(kind, fn)

    async def aclassify(
        self,
        task: str,
        history: Optional[list[str]] = None,
        context_signals: "Optional[ContextSignals]" = None,
        provider: Optional[str] = None,
    ) -> "ClassificationDecision":
        """Async version of classify(). Runs the synchronous classifier in a
        threadpool to avoid blocking the event loop.

        Use this from FastAPI / aiohttp / asyncio agent frameworks.

        Example:
            decision = await router.aclassify("Summarise this contract")
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.classify(task, history=history,
                                  context_signals=context_signals, provider=provider),
        )

    async def aclassify_batch(
        self,
        tasks: list[str],
        *,
        provider: Optional[str] = None,
        concurrency: int = 8,
    ) -> "list[ClassificationDecision]":
        """Async batch classify. Runs up to `concurrency` classifications in parallel."""
        import asyncio
        sem = asyncio.Semaphore(concurrency)

        async def _one(t: str):
            async with sem:
                return await self.aclassify(t, provider=provider)

        return await asyncio.gather(*[_one(t) for t in tasks])

    def estimate_cost(
        self,
        task: str,
        *,
        provider: Optional[str] = None,
        estimated_output_tokens: int = 500,
    ) -> dict:
        """Dry-run: classify task and return estimated cost without making an API call.

        Args:
            task:                     The task text to classify.
            provider:                 Provider override (defaults to router's first provider).
            estimated_output_tokens:  Assumed output length for cost estimate (default 500).

        Returns:
            Dict with keys: tier, model, provider, input_tokens, output_tokens, est_usd_per_call.

        Example:
            router = Router()
            info = router.estimate_cost("Summarise this 2-page document")
            print(info["tier"], info["est_usd_per_call"])
        """
        from classifier.infra.cost_tracker import get_model_cost
        from classifier.infra.tokenizers import count_tokens

        decision = self.classify(task, provider=provider)
        model = decision.model_name
        input_tokens = count_tokens(task, model=model)

        rates = get_model_cost(model)
        est_usd = (input_tokens / 1_000_000) * rates["input"] + \
                  (estimated_output_tokens / 1_000_000) * rates["output"]

        return {
            "tier":               decision.tier.value,
            "model":              model,
            "provider":           decision.provider,
            "layer_used":         decision.layer_used,
            "input_tokens":       input_tokens,
            "output_tokens":      estimated_output_tokens,
            "est_usd_per_call":   round(est_usd, 8),
            "input_rate_per_1m":  rates["input"],
            "output_rate_per_1m": rates["output"],
        }

    def train(
        self,
        data: str | Path,
        *,
        output_path: Optional[str | Path] = None,
        max_iter: int = 600,
    ) -> dict:
        """Retrain L3 head on user-supplied data. Returns metadata dict.

        Args:
            data:        Path to JSONL file with {"task", "task_type", "complexity"} per line.
            output_path: Where to save model bundle. Default: classifier/ml/models/head_v1.joblib.
            max_iter:    sklearn MLP max iterations.

        Returns:
            Metadata dict (training accuracy, threshold sweep, etc.).
        """
        from classifier.ml.train import train_from_data
        return train_from_data(
            data_path=Path(data),
            output_path=Path(output_path) if output_path else None,
            max_iter=max_iter,
        )

    # ── Alternative constructors ─────────────────────────────────────────────

    def with_overrides(self, **kwargs) -> "Router":
        """Create a new Router that inherits this one's config, with overrides.

        For multi-tenant deployments where a base Router serves many tenants
        each with their own per-tenant config:

            base = Router(layer3_enabled=True)
            tenant_a = base.with_overrides(providers=["anthropic"], budget_usd=100)
            tenant_b = base.with_overrides(layer2_enabled=False)
        """
        merged = self.to_dict()
        merged.update(kwargs)
        return Router(**merged)

    def merge(self, other: "Router") -> "Router":
        """Compose two Routers — last-wins semantics on every field.

        Useful when combining a domain preset with custom user overrides:

            base   = Router.from_preset("healthcare")
            custom = Router(extra_keyword_packs=[my_pack])
            router = base.merge(custom)
        """
        a = self.to_dict()
        b = other.to_dict()
        for k, v in b.items():
            # Lists merge additively; scalars overwrite
            if isinstance(v, list) and isinstance(a.get(k), list):
                a[k] = a[k] + [x for x in v if x not in a[k]]
            elif isinstance(v, dict) and isinstance(a.get(k), dict):
                a[k] = {**a[k], **v}
            elif v is not None:
                a[k] = v
        return Router(**a)

    def to_dict(self) -> dict:
        """Serialise constructor args back to a dict (for merge/with_overrides)."""
        return {
            "providers":           list(self.providers),
            "extra_keyword_packs": list(self.extra_keyword_packs),
            "extra_pii_patterns":  list(self.extra_pii_patterns),
            "tier_matrix":         dict(self.tier_matrix),
            "model_registry":      dict(self.model_registry),
            "layer1_enabled":      self.layer1_enabled,
            "layer2_enabled":      self.layer2_enabled,
            "layer3_enabled":      self.layer3_enabled,
            "escalation_threshold": self.escalation_threshold,
            "layer3_threshold":    self.layer3_threshold,
            "budget_usd":          self.budget_usd,
            "cache_enabled":       self.cache_enabled,
            "layer2_provider":     self.layer2_provider,
            "layer2_model":        self.layer2_model,
            "layer3_embedding_model": self.layer3_embedding_model,
            "model_costs":         dict(self.model_costs),
            "custom_classifier":   self.custom_classifier,
            "pre_classify_hooks":  list(self.pre_classify_hooks),
            "post_classify_hooks": list(self.post_classify_hooks),
            "on_error_hooks":      list(self.on_error_hooks),
            "pii_policy":          self.pii_policy,
            "l2_retry_policy":     self.l2_retry_policy,
            "l2_circuit_breaker":  self.l2_circuit_breaker,
            "l1_weights":          self.l1_weights,
            "tokenizer":           self.tokenizer,
            "latency_budget_ms":   self.latency_budget_ms,
            "residency":           self.residency,
            "cache_backend":       self.cache_backend,
            "decision_logger":     self.decision_logger,
            "layer2_prompt_template": self.layer2_prompt_template,
        }

    @classmethod
    def from_registry(cls, source: "str | Path | dict", **router_kwargs) -> "Router":
        """Construct a Router after loading a model registry.

        Equivalent to:
            from classifier import load_registry
            load_registry(source)
            return Router(**router_kwargs)
        """
        from classifier.core.registry_loader import load_registry
        load_registry(source)
        return cls(**router_kwargs)

    @staticmethod
    def load_registry(source: "str | Path | dict") -> dict:
        """Load (and merge) a model registry into the runtime tables."""
        from classifier.core.registry_loader import load_registry
        return load_registry(source)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Router":
        """Construct a Router from a YAML config file."""
        import yaml
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

        # Translate dotted YAML keys → kwargs
        kwargs: dict[str, Any] = {}
        for key in (
            "providers", "layer1_enabled", "layer2_enabled", "layer3_enabled",
            "layer1_threshold", "layer3_threshold", "budget_usd", "cache_enabled",
            "tier_matrix", "model_registry",
        ):
            if key in cfg:
                kwargs[key] = cfg[key]

        # Keyword packs from YAML (list of {name, packs: {task_type: [keywords]}})
        if "keyword_packs" in cfg:
            from classifier.layers.layer1.keyword_pack import KeywordPack
            from classifier.core.types import TaskType
            packs = []
            for pack_def in cfg["keyword_packs"]:
                builder = KeywordPack.builder(pack_def.get("name", "custom"))
                for tt_name, words in (pack_def.get("packs") or {}).items():
                    builder.add(TaskType(tt_name), list(words))
                packs.append(builder.build())
            kwargs["extra_keyword_packs"] = packs

        return cls(**kwargs)

    @classmethod
    def from_preset(cls, name: str) -> "Router":
        """Construct a Router from a built-in domain preset."""
        from classifier.presets import load_preset
        cfg = load_preset(name)
        return cls(**cfg)

    # ── Internals ────────────────────────────────────────────────────────────

    @contextmanager
    def _apply_overrides(self):
        """Apply per-instance overrides to global state for the duration of one call.

        Mutates settings, feature_flags, TIER_MATRIX, MODEL_REGISTRY under a lock
        and restores on exit. Per-instance keyword_packs and pii_patterns are
        merged at construction (they're additive, not replacement).
        """
        from classifier.infra.config import settings
        from classifier.config.feature_flags import feature_flags
        from classifier.core import registry

        with _router_lock:
            saved: dict[str, Any] = {}

            try:
                # Layer toggles
                if self.layer1_enabled is not None:
                    saved["l1_enabled"] = getattr(settings, "layer1_enabled", True)
                    settings.layer1_enabled = self.layer1_enabled
                if self.layer2_enabled is not None:
                    saved["l2_enabled"] = settings.layer2_enabled
                    settings.layer2_enabled = self.layer2_enabled
                if self.layer3_enabled is not None:
                    saved["l3_enabled"] = settings.layer3_enabled
                    settings.layer3_enabled = self.layer3_enabled

                # Thresholds
                if self.escalation_threshold is not None:
                    saved["esc_thresh"] = settings.layer2_confidence_threshold
                    settings.layer2_confidence_threshold = self.escalation_threshold
                if self.layer3_threshold is not None:
                    saved["l3_thresh"] = settings.layer3_confidence_threshold
                    settings.layer3_confidence_threshold = self.layer3_threshold

                # Cache
                if self.cache_enabled is not None:
                    saved["cache"] = settings.cache_enabled
                    settings.cache_enabled = self.cache_enabled

                # L2 provider / model override
                if self.layer2_provider is not None:
                    saved["l2_provider"] = settings.layer2_provider
                    settings.layer2_provider = self.layer2_provider
                if self.layer2_model is not None:
                    saved["l2_model"] = settings.layer2_model
                    settings.layer2_model = self.layer2_model

                # Cache backend
                if self.cache_backend is not None:
                    from classifier.infra.cache import cache as _cache
                    saved["cache_backend"] = _cache._backend
                    _cache.set_backend(self.cache_backend)

                # Decision logger backend
                if self.decision_logger is not None:
                    from classifier.infra import decision_logger as _dl_mod
                    saved["decision_logger"] = getattr(_dl_mod, "_backend", None)
                    _dl_mod._backend = self.decision_logger

                # L1 weights
                if self.l1_weights is not None:
                    from classifier.layers.layer1 import scoring as _scoring
                    if hasattr(_scoring, "_WEIGHTS"):
                        saved["l1_weights"] = dict(_scoring._WEIGHTS)
                        _scoring._WEIGHTS.update(self.l1_weights)

                # L2 prompt template
                if self.layer2_prompt_template is not None:
                    from classifier.layers.layer2 import prompt as _prompt
                    if hasattr(_prompt, "_PROMPT"):
                        saved["l2_prompt"] = _prompt._PROMPT
                        _prompt._PROMPT = self.layer2_prompt_template

                # L2 retry policy
                if self.l2_retry_policy is not None:
                    from classifier.layers.layer2 import api as l2api
                    saved["retry_policy"] = dict(l2api._retry_policy)
                    l2api.configure_retry_policy(**{
                        k: v for k, v in self.l2_retry_policy.items()
                        if k in ("max_attempts", "initial_delay", "backoff")
                    })

                # L2 circuit breaker policy
                if self.l2_circuit_breaker is not None:
                    from classifier.layers.layer2 import api as l2api
                    saved["cb_threshold"] = l2api._circuit_breaker.failure_threshold
                    saved["cb_cooldown"]  = l2api._circuit_breaker.cooldown_secs
                    l2api._circuit_breaker.failure_threshold = self.l2_circuit_breaker.get("failure_threshold", 5)
                    l2api._circuit_breaker.cooldown_secs     = self.l2_circuit_breaker.get("cooldown_secs", 60.0)

                # Budget
                if self.budget_usd is not None:
                    saved["budget"] = settings.monthly_budget_usd
                    settings.monthly_budget_usd = self.budget_usd

                # Tier matrix override (merge into module-level dict)
                if self.tier_matrix:
                    saved["tier_matrix"] = dict(registry.TIER_MATRIX)
                    registry.TIER_MATRIX.update(self.tier_matrix)

                # Model registry override
                if self.model_registry:
                    saved["model_registry"] = {
                        k: dict(v) for k, v in registry.MODEL_REGISTRY.items()
                    }
                    for prov, tier_map in self.model_registry.items():
                        registry.MODEL_REGISTRY.setdefault(prov, {}).update(tier_map)

                yield

            finally:
                # Restore in reverse order
                if "l1_enabled" in saved: settings.layer1_enabled = saved["l1_enabled"]
                if "l2_enabled" in saved: settings.layer2_enabled = saved["l2_enabled"]
                if "l3_enabled" in saved: settings.layer3_enabled = saved["l3_enabled"]
                if "esc_thresh" in saved: settings.layer2_confidence_threshold = saved["esc_thresh"]
                if "l3_thresh"  in saved: settings.layer3_confidence_threshold = saved["l3_thresh"]
                if "cache"     in saved:  settings.cache_enabled = saved["cache"]
                if "budget"    in saved:  settings.monthly_budget_usd = saved["budget"]
                if "l2_provider" in saved: settings.layer2_provider = saved["l2_provider"]
                if "l2_model"    in saved: settings.layer2_model = saved["l2_model"]
                if "cb_threshold" in saved:
                    from classifier.layers.layer2 import api as l2api
                    l2api._circuit_breaker.failure_threshold = saved["cb_threshold"]
                    l2api._circuit_breaker.cooldown_secs     = saved["cb_cooldown"]
                if "retry_policy" in saved:
                    from classifier.layers.layer2 import api as l2api
                    l2api._retry_policy.update(saved["retry_policy"])
                if "cache_backend" in saved:
                    from classifier.infra.cache import cache as _cache
                    _cache.set_backend(saved["cache_backend"])
                if "decision_logger" in saved:
                    from classifier.infra import decision_logger as _dl_mod
                    _dl_mod._backend = saved["decision_logger"]
                if "l1_weights" in saved:
                    from classifier.layers.layer1 import scoring as _scoring
                    _scoring._WEIGHTS.clear()
                    _scoring._WEIGHTS.update(saved["l1_weights"])
                if "l2_prompt" in saved:
                    from classifier.layers.layer2 import prompt as _prompt
                    _prompt._PROMPT = saved["l2_prompt"]
                if "tier_matrix" in saved:
                    registry.TIER_MATRIX.clear()
                    registry.TIER_MATRIX.update(saved["tier_matrix"])
                if "model_registry" in saved:
                    registry.MODEL_REGISTRY.clear()
                    registry.MODEL_REGISTRY.update(saved["model_registry"])


# ── Module-level convenience function ────────────────────────────────────────

_default_router: Optional[Router] = None


def classify(task: str, **kwargs) -> "ClassificationDecision":
    """Zero-config classify using a process-wide default Router.

    Equivalent to:
        Router().classify(task, **kwargs)

    For repeated calls, prefer creating a Router instance and reusing it.
    """
    global _default_router
    if _default_router is None:
        _default_router = Router()
    return _default_router.classify(task, **kwargs)
