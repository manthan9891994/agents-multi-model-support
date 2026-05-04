__version__ = "0.1.0"

import logging
import threading
import time

from classifier.core.exceptions import (
    ClassificationError,
    ConfigurationError,
    LayerNotAvailableError,
    UnsupportedProviderError,
)
from classifier.core.types import (
    ClassificationDecision, ContextSignals, TaskComplexity, TaskType, ModelTier,
    register_task_type, register_complexity, set_tier_levels, list_tier_levels,
)
from classifier.infra.feedback import record_feedback
from classifier.core.registry import MODEL_REGISTRY, TIER_MATRIX

# Auto-load the bundled / configured registry at import time.
# Honors DMR_REGISTRY env var and DMR_NO_DEFAULT_REGISTRY=1 to opt out.
from classifier.core.registry_loader import (
    _auto_load_at_import as _registry_auto_load,
    load_registry, clear_registry, export_registry, export_to_yaml,
)
_registry_auto_load()
from classifier.layers.layer1 import classify_layer1, detect_pii  # noqa: F401 — re-exported
from classifier.infra.config import settings
from classifier.infra.cache import cache
from classifier.infra.cost_tracker import cost_tracker
from classifier.config.feature_flags import feature_flags

logger = logging.getLogger(__name__)
# PEP 282 best practice: libraries must NOT configure logging — only attach
# a NullHandler so that "no handlers" warnings don't fire if the host app
# doesn't configure logging itself.
logger.addHandler(logging.NullHandler())

# Use the dynamic tier order from core.types so set_tier_levels() takes effect.
from classifier.core.types import _TIER_ORDER as _DYN_TIER_ORDER

def _tier_order():
    """Always return the latest tier order (live reference, not snapshot)."""
    return _DYN_TIER_ORDER

# Keep _TIER_ORDER as a module-level name for back-compat but route through helper.
_TIER_ORDER = _DYN_TIER_ORDER

# Maximum input length — guards L3 OOM, L2 timeout, runaway costs.
# Override via DMR_MAX_TASK_CHARS env var if you really need longer inputs.
import os as _os
MAX_TASK_CHARS = int(_os.environ.get("DMR_MAX_TASK_CHARS", "32000"))

# Item 20: Streaming debounce — last known good decision (stateless fallback).
# Bounded + lock-protected so long-running processes don't accumulate state.
_last_decision: ClassificationDecision | None = None
_last_decision_lock = threading.Lock()

# Item 11: Calibration data (loaded once at first use)
_calibration: dict | None = None


def _get_calibration() -> dict:
    global _calibration
    if _calibration is None:
        try:
            from classifier.calibrate import load_calibration
            _calibration = load_calibration()
        except Exception:
            _calibration = {}
    return _calibration


def _apply_calibration(layer: str, raw_conf: float) -> float:
    cal = _get_calibration()
    if not cal:
        return raw_conf
    try:
        from classifier.calibrate import calibrated_confidence
        return calibrated_confidence(layer, raw_conf, cal)
    except Exception:
        return raw_conf


def _adjust_tier_for_context(
    tier: ModelTier,
    reasoning: str,
    ctx: "ContextSignals",
) -> tuple[ModelTier, str]:
    """Adjust tier based on agent-loop context signals for mid-flight model switching."""
    idx = _TIER_ORDER.index(tier)

    if ctx.call_number <= 1:
        return tier, reasoning

    if ctx.total_context_tokens > 100_000 and idx < 1:
        idx = 1
        reasoning += f" [ctx={ctx.total_context_tokens} tokens → bumped to MEDIUM]"

    if ctx.has_error and idx < 1:
        idx = 1
        reasoning += " [error detected → bumped to MEDIUM]"
    elif not ctx.has_error and ctx.call_number >= 3 and ctx.last_role == "model":
        idx = 0
        reasoning += f" [call={ctx.call_number}, last=model, no error → dropped to LOW]"
    elif not ctx.has_error and ctx.call_number >= 2 and ctx.last_role == "tool":
        idx = max(idx - 1, 0)
        reasoning += f" [call={ctx.call_number}, last=tool, no error → stepped down]"

    return _TIER_ORDER[idx], reasoning


def _setup_l2_budget() -> None:
    """Configure L2 category budget in cost_tracker (called once at startup)."""
    l2_budget = settings.layer2_monthly_budget_usd
    if l2_budget <= 0:
        l2_budget = settings.monthly_budget_usd * 0.05  # default: 5% of main budget
    cost_tracker.set_category_budget("layer2", l2_budget)


_setup_l2_budget()


def classify_task(
    task: str,
    provider: str = None,
    history: list[str] | None = None,
    context_signals: "ContextSignals | None" = None,
    task_stable: bool = True,
    user_id: str | None = None,
    hook_context: dict | None = None,
    custom_classifier: callable = None,
) -> ClassificationDecision:
    """Classify a task and return the best model for it.

    Args:
        task:            The user's input text.
        provider:        One of 'google', 'openai', 'anthropic'. Defaults to DEFAULT_PROVIDER.
        history:         Optional prior conversation turns (most-recent last).
        context_signals: Agent mid-flight signals (call number, errors, context size).
        task_stable:     Item 20 — set False while user is still typing to return last known decision.
        user_id:         Item 17 — enables per-user tier personalization.

    Returns:
        ClassificationDecision with model_name, tier, task_type, complexity,
        layer_used, latency_ms, compliance_flag, disagreement.
    """
    global _last_decision

    # Hook context: per-call user data passed to all hooks
    ctx: dict = dict(hook_context) if hook_context else {}
    ctx.setdefault("provider", provider)
    ctx.setdefault("user_id",  user_id)

    # Pre-classify hooks — can modify or reject the task
    from classifier.hooks import hook_manager
    task = hook_manager.run_pre(task, ctx)

    # Custom classifier escape hatch — if the user provided one and it returns
    # a decision, skip the cascade entirely.
    if custom_classifier is not None:
        try:
            custom = custom_classifier(task, ctx)
            if custom is not None:
                return hook_manager.run_post("post_classify", task, custom, ctx)
        except Exception as exc:
            logger.warning("custom_classifier raised: %s — falling back to cascade", exc)

    # Item 20: Streaming debounce — return last decision while input is in-flight
    if not task_stable:
        with _last_decision_lock:
            if _last_decision is not None:
                return _stamp_cache_hit(_last_decision)

    resolved_provider = provider or settings.default_provider

    if resolved_provider not in MODEL_REGISTRY:
        raise UnsupportedProviderError(
            f"Provider '{resolved_provider}' is not supported. "
            f"Choose from: {sorted(MODEL_REGISTRY)}"
        )

    if not task or not task.strip():
        raise ClassificationError(
            "Task cannot be empty.",
            layer="input",
            suggestion="Pass a non-empty string, e.g. classify('Write a Python function')",
        )

    # Item: input length guard — protects L3 OOM, L2 timeout, runaway costs
    if len(task) > MAX_TASK_CHARS:
        raise ClassificationError(
            f"Task length {len(task)} chars exceeds DMR_MAX_TASK_CHARS={MAX_TASK_CHARS}.",
            layer="input",
            task=task,
            suggestion=(
                "Truncate input or set DMR_MAX_TASK_CHARS to a higher value. "
                "Long inputs are usually a sign you should split the work."
            ),
        )

    # Item: API key validation — only validate the GOOGLE key when L2 is enabled
    # (L2 always uses Gemini Flash Lite regardless of `provider`). The package
    # itself never calls Anthropic/OpenAI — it only returns model names that
    # the user's own SDK will use, so we don't validate those keys here.
    if settings.layer2_enabled:
        try:
            settings.api_key_for("google")
        except ConfigurationError:
            raise

    # ── Budget guard ──────────────────────────────────────────────────────────
    if cost_tracker.is_exhausted():
        tier = ModelTier.LOW
        return ClassificationDecision(
            model_name=MODEL_REGISTRY[resolved_provider][tier],
            tier=tier,
            task_type=TaskType.DOC_CREATION,
            complexity=TaskComplexity.SIMPLE,
            reasoning="budget exhausted — forced LOW",
            confidence=1.0,
            provider=resolved_provider,
            layer_used="budget_guard",
            latency_ms=0.0,
        )

    max_tier = ModelTier.MEDIUM if cost_tracker.should_downgrade() else None

    # ── Cache lookup (exact match) ────────────────────────────────────────────
    t0 = time.perf_counter()

    if settings.cache_enabled:
        cached = cache.get(task, resolved_provider)
        if cached is not None:
            return _stamp_cache_hit(cached)

    # ── Semantic cache lookup (Item 5) ────────────────────────────────────────
    if settings.semantic_cache_enabled:
        try:
            from classifier.infra.semantic_cache import semantic_cache
            sem_hit = semantic_cache.get(task)
            if sem_hit is not None:
                return _stamp_cache_hit(sem_hit)
        except Exception:
            pass

    # ── Single-flight coalescing (Item 7) — compute once per unique task ──────
    cache_key = f"{resolved_provider}::{task[:200]}"

    def _compute() -> ClassificationDecision:
        try:
            return _classify_inner(
                task, resolved_provider, history, context_signals, max_tier, t0, user_id, ctx,
            )
        except Exception as exc:
            recovery = hook_manager.run_error(task, exc, ctx)
            if recovery is not None:
                return recovery
            raise

    if feature_flags.single_flight_coalescing:
        from classifier.infra.coalescer import single_flight
        decision = single_flight.do(cache_key, _compute)
    else:
        decision = _compute()

    # Final post-classify hooks (after all cascade logic, including PII bumps)
    decision = hook_manager.run_post("post_classify", task, decision, ctx)

    # ── Store for streaming debounce (Item 20) ────────────────────────────────
    with _last_decision_lock:
        _last_decision = decision

    return decision


def reset_last_decision() -> None:
    """Clear the streaming-debounce cache. Call between independent classify sessions."""
    global _last_decision
    with _last_decision_lock:
        _last_decision = None


def _stamp_cache_hit(original: ClassificationDecision) -> ClassificationDecision:
    """Return a copy of `original` with a fresh decision_id + cached=True.

    Cache hits represent real LLM-call events (each one needs its own outcome
    row), but the *routing decision* didn't run again. So we mint a new
    decision_id, point `cached_from` at the original, and let the auto-labeler
    decide whether to dedupe by `cached_from` during training.
    """
    from dataclasses import replace
    from classifier.core.types import _new_decision_id
    return replace(
        original,
        decision_id=_new_decision_id(),
        cached=True,
        cached_from=original.decision_id,
    )


def _classify_inner(
    task: str,
    resolved_provider: str,
    history: list[str] | None,
    context_signals: "ContextSignals | None",
    max_tier: ModelTier | None,
    t0: float,
    user_id: str | None,
    ctx: dict | None = None,
) -> ClassificationDecision:
    from classifier.infra.telemetry import span as _span, set_attribute as _attr

    with _span("dmr.classify", **{"task.length": len(task), "provider": resolved_provider}) as _s:
        return _classify_inner_traced(
            task, resolved_provider, history, context_signals, max_tier, t0, user_id, _s, _attr, ctx or {},
        )


def _classify_inner_traced(
    task, resolved_provider, history, context_signals, max_tier, t0, user_id, _s, _attr, ctx,
):
    from classifier.hooks import hook_manager
    from classifier.layers.plugin import run_layers_at as _run_plugins

    # Pre-cascade plugins — first non-None tuple short-circuits the cascade.
    # The decision is built at the end (after PII bumps, capability filtering, etc).
    plugin_pre = _run_plugins("pre", task, history)
    plugin_pre_used = plugin_pre is not None
    # ── Layer 1 (or pre-plugin) ──────────────────────────────────────────────
    if plugin_pre_used:
        task_type, complexity, tier, confidence, reasoning = plugin_pre
        layer_used = "plugin:pre"
    else:
        layer_used = "layer1"
        try:
            task_type, complexity, tier, confidence, reasoning = classify_layer1(
                task, history=history, provider=resolved_provider
            )
        except Exception as exc:
            raise ClassificationError(
                f"Layer 1 classification failed: {exc}",
                layer="layer1",
                task=task,
                suggestion="Check that task text is valid UTF-8 and not excessively long (>32K chars).",
            ) from exc

    # ── Item 11: Apply calibration to L1 confidence ───────────────────────────
    if feature_flags.calibration:
        confidence = _apply_calibration("layer1", confidence)

    # ── Layer 3 (between L1 and L2 — fast ML classifier with abstain) ─────────
    if not plugin_pre_used and settings.layer3_enabled and confidence < settings.layer2_confidence_threshold:
        try:
            from classifier.layers.layer3 import classify_layer3
            l3 = classify_layer3(task, history=history)
            if l3 is not None and l3[3] >= settings.layer3_confidence_threshold:
                task_type, complexity, tier, confidence, reasoning = l3
                layer_used = "layer3"
                if feature_flags.calibration:
                    confidence = _apply_calibration("layer3", confidence)
        except ImportError:
            logger.warning("layer3: transformers not installed — skipping")
        except Exception as exc:
            logger.warning("layer3: failed: %s — skipping", exc)

    # ── Layer 2 (Item 10: check L2 budget before firing) ──────────────────────
    l2_result = None
    l2_fired = (not plugin_pre_used) and settings.layer2_enabled and not cost_tracker.is_exhausted_for("layer2")
    if l2_fired and (
        confidence < settings.layer2_confidence_threshold
        or settings.debug_ab_mode
    ):
        try:
            from classifier.layers.layer2 import classify_layer2
            l2 = classify_layer2(task, history=history)
            if l2 is not None:
                l2_result = l2
                if confidence < settings.layer2_confidence_threshold:
                    task_type, complexity, tier, confidence, reasoning = l2
                    layer_used = "layer2"
                    # Item 11: calibrate L2 confidence too
                    confidence = _apply_calibration("layer2", confidence)
        except ImportError:
            logger.warning("layer2: google-genai not installed — falling back to layer1")

    # ── Item 12: L1 + L2 agreement boost / disagreement flag ─────────────────
    disagreement = False
    if feature_flags.l1_l2_agreement and l2_result is not None and layer_used == "layer1":
        # Both layers ran; compare results
        l2_tt, l2_cx, l2_tier, l2_conf, l2_reason = l2_result
        if l2_tt == task_type and l2_cx == complexity:
            # Both agree → boost confidence
            confidence = min(0.95, max(confidence, l2_conf) + 0.10)
            reasoning += " | L1∩L2 agree"
        else:
            # Disagree → pick higher-tier (safer); flag for review
            disagreement = True
            if _TIER_ORDER.index(l2_tier) > _TIER_ORDER.index(tier):
                task_type, complexity, tier = l2_tt, l2_cx, l2_tier
                reasoning += f" | L1≠L2 disagree → L2 tier higher, using L2 ({l2_reason})"
            else:
                reasoning += f" | L1≠L2 disagree → L1 tier ≥ L2, keeping L1"
            confidence = min(confidence, l2_conf, 0.55)
            # Auto-record disagreement as feedback candidate for L3 training
            try:
                record_feedback(
                    task,
                    expected_type=task_type.value,
                    expected_complexity=complexity.value,
                    original_type=task_type.value,
                    original_complexity=complexity.value,
                )
            except Exception:
                pass

    # ── A/B debug logging ─────────────────────────────────────────────────────
    if settings.debug_ab_mode and l2_result is not None:
        l2_tt2, l2_cx2, _, l2_conf2, _ = l2_result
        logger.info(
            "A/B | L1: %s/%s (%.2f) | L2: %s/%s (%.2f)",
            task_type.value if layer_used == "layer1" else "—",
            complexity.value if layer_used == "layer1" else "—",
            confidence if layer_used == "layer1" else 0,
            l2_tt2.value, l2_cx2.value, l2_conf2,
        )

    # ── Item 3: Multimodal content inspection ─────────────────────────────────
    if context_signals is not None and context_signals.has_multimodal:
        if task_type != TaskType.MULTIMODAL:
            task_type = TaskType.MULTIMODAL
            tier = TIER_MATRIX.get((task_type, complexity), tier)
            reasoning += " [multimodal content detected → forced MULTIMODAL]"

    # ── Item 4: Tool-aware routing — bump tier for first planning call ─────────
    if (
        context_signals is not None
        and context_signals.available_tools >= 3
        and context_signals.call_number == 1
    ):
        idx = _TIER_ORDER.index(tier)
        if idx < 2:
            tier = _TIER_ORDER[idx + 1]
            reasoning += f" [tools={context_signals.available_tools} → planning call bumped]"

    # ── Context-signal tier adjustment ────────────────────────────────────────
    if context_signals is not None:
        tier, reasoning = _adjust_tier_for_context(tier, reasoning, context_signals)

    # ── Item 9: Adaptive latency routing ─────────────────────────────────────
    if feature_flags.health_tracker:
        try:
            from classifier.infra.health_tracker import health_tracker
            if health_tracker.is_degraded(resolved_provider, tier):
                idx = _TIER_ORDER.index(tier)
                if idx > 0:
                    tier = _TIER_ORDER[idx - 1]
                    reasoning += f" [degraded: {resolved_provider} p95 SLO exceeded → demoted]"
        except Exception:
            pass

    # ── Budget cap ────────────────────────────────────────────────────────────
    if max_tier is not None and tier == ModelTier.HIGH:
        tier = max_tier
        reasoning += " [capped to MEDIUM: budget >80%]"

    # ── Item 17: Per-user personalization ─────────────────────────────────────
    if feature_flags.per_user_personalization and user_id:
        try:
            from classifier.infra.personalization import get_user_bias
            bias = get_user_bias(user_id)
            idx = _TIER_ORDER.index(tier)
            if bias > 0.3 and idx < 2:
                tier = _TIER_ORDER[idx + 1]
                reasoning += f" [user_bias={bias:.2f} → bumped]"
            elif bias < -0.3 and idx > 0:
                tier = _TIER_ORDER[idx - 1]
                reasoning += f" [user_bias={bias:.2f} → demoted]"
        except Exception:
            pass

    latency_ms = (time.perf_counter() - t0) * 1000
    model_name = MODEL_REGISTRY[resolved_provider][tier]

    # ── Item 1: PII detection → policy-driven tier bump and compliance_flag ──
    compliance_flag = feature_flags.pii_detection and detect_pii(task)
    if compliance_flag:
        # Default policy: bump to MEDIUM minimum, no block
        policy = (ctx or {}).get("pii_policy") or {"min_tier": ModelTier.MEDIUM, "block": False}
        if policy.get("block"):
            raise ClassificationError(
                "PII detected and pii_policy.block=True", layer="pii", task=task,
                suggestion="Disable pii_policy.block, scrub upstream, or use a different model.",
            )
        min_tier = policy.get("min_tier", ModelTier.MEDIUM)
        if isinstance(min_tier, str):
            min_tier = ModelTier(min_tier)
        idx_now = _TIER_ORDER.index(tier)
        idx_min = _TIER_ORDER.index(min_tier)
        if idx_now < idx_min:
            tier = min_tier
            model_name = MODEL_REGISTRY[resolved_provider][tier]
            reasoning += f" [PII/PHI detected → bumped to {tier.value.upper()} minimum]"
        logger.warning("PII/PHI detected in task — compliance_flag=True")

    # Post-cascade plugins — last-wins
    plugin_post = _run_plugins("post", task, history)
    if plugin_post is not None:
        task_type, complexity, tier, confidence, reasoning = plugin_post
        layer_used = "plugin:post"

    # ── Context window escalation (#19) ───────────────────────────────────────
    # If the picked model's context window is smaller than estimated total
    # tokens, escalate to a tier whose model has more headroom.
    from classifier.core.registry import capabilities_for as _caps
    from classifier.infra.tokenizers import count_tokens as _count_tokens
    candidate_model = MODEL_REGISTRY[resolved_provider][tier]
    candidate_caps  = _caps(candidate_model)
    total_input_tokens = _count_tokens(task, model=candidate_model)
    if history:
        for h in history:
            total_input_tokens += _count_tokens(h, model=candidate_model)
    cw = candidate_caps.get("context_window")
    if cw and total_input_tokens > cw * 0.9:   # 10% headroom for output
        for higher in _TIER_ORDER[_TIER_ORDER.index(tier) + 1:]:
            higher_model = MODEL_REGISTRY[resolved_provider].get(higher)
            higher_cw = _caps(higher_model).get("context_window") if higher_model else None
            if higher_cw and total_input_tokens <= higher_cw * 0.9:
                tier = higher
                reasoning += f" [context {total_input_tokens} > {cw} → escalated to {higher.value.upper()}]"
                break

    # ── Capability filtering (#20) ────────────────────────────────────────────
    # If the request needs vision / function-calling and the picked model
    # doesn't support it, escalate within the same provider.
    needed: list[str] = []
    if context_signals is not None and context_signals.has_multimodal:
        needed.append("supports_vision")
    if context_signals is not None and getattr(context_signals, "available_tools", 0) > 0:
        needed.append("supports_function_calling")
    if needed:
        candidate_model = MODEL_REGISTRY[resolved_provider][tier]
        c = _caps(candidate_model)
        if not all(c.get(flag, True) for flag in needed):
            for higher in _TIER_ORDER[_TIER_ORDER.index(tier) + 1:]:
                hm = MODEL_REGISTRY[resolved_provider].get(higher)
                if hm and all(_caps(hm).get(flag, True) for flag in needed):
                    tier = higher
                    reasoning += f" [needed {needed} → escalated to {higher.value.upper()}]"
                    break

    # ── Latency SLA budget (#21) ──────────────────────────────────────────────
    sla_ms = (ctx or {}).get("latency_budget_ms")
    if sla_ms and feature_flags.health_tracker:
        try:
            from classifier.infra.health_tracker import health_tracker
            # If the chosen model's recent p95 exceeds SLA, drop a tier
            current_model = MODEL_REGISTRY[resolved_provider][tier]
            p95 = getattr(health_tracker, "get_p95", lambda *_: 0)(resolved_provider, tier)
            if p95 and p95 > sla_ms:
                idx = _TIER_ORDER.index(tier)
                if idx > 0:
                    tier = _TIER_ORDER[idx - 1]
                    reasoning += f" [p95={p95}ms > SLA={sla_ms}ms → demoted]"
        except Exception:
            pass

    # ── Data residency (#22) ──────────────────────────────────────────────────
    residency = (ctx or {}).get("residency")
    if residency:
        candidate_model = MODEL_REGISTRY[resolved_provider][tier]
        c = _caps(candidate_model)
        if c.get("region") and c["region"] != residency:
            # Find any model in this provider that matches the residency
            for try_tier in _TIER_ORDER:
                try_model = MODEL_REGISTRY[resolved_provider].get(try_tier)
                if try_model and _caps(try_model).get("region") in (None, residency):
                    tier = try_tier
                    reasoning += f" [residency={residency} → {try_tier.value.upper()}]"
                    break

    model_name = MODEL_REGISTRY[resolved_provider][tier]

    decision = ClassificationDecision(
        model_name=model_name,
        tier=tier,
        task_type=task_type,
        complexity=complexity,
        reasoning=reasoning,
        confidence=confidence,
        provider=resolved_provider,
        layer_used=layer_used,
        latency_ms=round(latency_ms, 2),
        compliance_flag=compliance_flag,
        disagreement=disagreement,
    )

    logger.info(
        "Classified | %s => %s [%s | %s | %s | %s | %.1fms%s%s]",
        resolved_provider, model_name,
        tier.value.upper(), task_type.value, complexity.value, layer_used, latency_ms,
        " | PII" if compliance_flag else "",
        " | DISAGREE" if disagreement else "",
    )

    # Annotate trace span with the final outcome (no-op if OTel not installed)
    _attr(_s, "tier", tier.value)
    _attr(_s, "model", model_name)
    _attr(_s, "task_type", task_type.value)
    _attr(_s, "complexity", complexity.value)
    _attr(_s, "layer_used", layer_used)
    _attr(_s, "confidence", confidence)
    _attr(_s, "latency_ms", round(latency_ms, 2))
    _attr(_s, "compliance_flag", compliance_flag)
    _attr(_s, "disagreement", disagreement)

    if settings.cache_enabled:
        cache.set(task, resolved_provider, decision)

    if settings.semantic_cache_enabled:
        try:
            from classifier.infra.semantic_cache import semantic_cache
            semantic_cache.set(task, decision)
        except Exception:
            pass

    if settings.log_decisions:
        from classifier.infra.decision_logger import log_decision
        log_decision(task, decision, layer_used=layer_used, latency_ms=latency_ms)

    return decision


from classifier.router import Router, classify
from classifier.layers.layer1.keyword_pack import KeywordPack
from classifier.core.registry import register_provider, list_providers, list_models, capabilities_for
from classifier.infra.cost_tracker import register_model_cost, get_model_cost
from classifier.ml.embeddings import set_embedding_model, current_embedding_model
from classifier.hooks import register_hook, unregister_hook, clear_hooks, hook_manager
from classifier.experiments import ABTest, ShadowMode
from classifier.infra.outcome_logger import (
    OutcomeRecord, log_outcome, read_outcomes, join_decisions_outcomes,
    prune_old_outcomes,
)
from classifier.infra.decision_logger import read_decisions
from classifier.ml.auto_labeler import (
    AutoLabeler, Label, LabelingFunction, DEFAULT_LFS,
)
from classifier.layers.plugin import register_layer, unregister_layer, list_layers
from classifier.layers.layer3 import register_strategy as register_l3_strategy
from classifier.infra.tokenizers import register_tokenizer, count_tokens


def route_model(
    provider: str | None = None,
    *,
    task_arg: str = "task",
    fallback_model: str | None = None,
    inject_as: str = "model_name",
):
    """Decorator that classifies the task argument and injects the model name.

    The decorated function receives an extra keyword argument (default: `model_name`)
    with the router-selected model name. The original positional/keyword args are
    passed through unchanged.

    Args:
        provider:       Provider to use ("google" | "anthropic" | "openai").
        task_arg:       Name of the argument that holds the task text (default "task").
        fallback_model: Model used if classification fails.
        inject_as:      Name of the kwarg injected into the function (default "model_name").

    Example:
        @route_model(provider="anthropic")
        def call_llm(task: str, model_name: str = "claude-sonnet-4-6"):
            client = anthropic.Anthropic()
            return client.messages.create(model=model_name, ...)

        # model_name is auto-filled by the router:
        result = call_llm("Compare metformin vs GLP-1 agonists")
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(fn)
            params = list(sig.parameters)

            # Resolve task text: check kwargs first, then positional
            if task_arg in kwargs:
                task_text = kwargs[task_arg]
            elif task_arg in params:
                idx = params.index(task_arg)
                task_text = args[idx] if idx < len(args) else ""
            else:
                task_text = args[0] if args else ""

            # Classify and inject
            try:
                decision = classify_task(str(task_text), provider=provider)
                model = decision.model_name
            except Exception as exc:
                if fallback_model:
                    model = fallback_model
                else:
                    raise

            kwargs.setdefault(inject_as, model)
            return fn(*args, **kwargs)

        return wrapper
    return decorator


__all__ = [
    # New high-level API
    "__version__",
    "Router",
    "classify",
    "KeywordPack",
    "route_model",
    "reset_last_decision",
    "MAX_TASK_CHARS",
    # Extensibility v2
    "register_provider", "list_providers", "list_models", "capabilities_for",
    "register_task_type", "register_complexity",
    "set_tier_levels", "list_tier_levels",
    "register_model_cost", "get_model_cost",
    "set_embedding_model", "current_embedding_model",
    "register_hook", "unregister_hook", "clear_hooks", "hook_manager",
    "ABTest", "ShadowMode",
    "OutcomeRecord", "log_outcome", "read_outcomes", "join_decisions_outcomes",
    "prune_old_outcomes", "read_decisions",
    "AutoLabeler", "Label", "LabelingFunction", "DEFAULT_LFS",
    "register_layer", "unregister_layer", "list_layers",
    "register_l3_strategy",
    "register_tokenizer", "count_tokens",
    # Free function (kept for backwards compat)
    "classify_task",
    # Types
    "ClassificationDecision",
    "ContextSignals",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    # Registries (for advanced overrides)
    "MODEL_REGISTRY",
    "TIER_MATRIX",
    # Exceptions
    "ClassificationError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "LayerNotAvailableError",
    # Misc
    "record_feedback",
]
