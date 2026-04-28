import logging
import time

from classifier.core.exceptions import (
    ClassificationError,
    ConfigurationError,
    LayerNotAvailableError,
    UnsupportedProviderError,
)
from classifier.core.types import ClassificationDecision, ContextSignals, TaskComplexity, TaskType, ModelTier
from classifier.infra.feedback import record_feedback
from classifier.core.registry import MODEL_REGISTRY, TIER_MATRIX
from classifier.layers.layer1 import classify_layer1, detect_pii  # noqa: F401 — re-exported
from classifier.infra.config import settings
from classifier.infra.cache import cache
from classifier.infra.cost_tracker import cost_tracker
from classifier.config.feature_flags import feature_flags

logger = logging.getLogger(__name__)

_TIER_ORDER = [ModelTier.LOW, ModelTier.MEDIUM, ModelTier.HIGH]

# Item 20: Streaming debounce — last known good decision (stateless fallback)
_last_decision: ClassificationDecision | None = None

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

    # Item 20: Streaming debounce — return last decision while input is in-flight
    if not task_stable and _last_decision is not None:
        return _last_decision

    resolved_provider = provider or settings.default_provider

    if resolved_provider not in MODEL_REGISTRY:
        raise UnsupportedProviderError(
            f"Provider '{resolved_provider}' is not supported. "
            f"Choose from: {sorted(MODEL_REGISTRY)}"
        )

    if not task or not task.strip():
        raise ClassificationError(
            "Task cannot be empty. Provide a non-empty string to classify."
        )

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
            return cached

    # ── Semantic cache lookup (Item 5) ────────────────────────────────────────
    if settings.semantic_cache_enabled:
        try:
            from classifier.infra.semantic_cache import semantic_cache
            sem_hit = semantic_cache.get(task)
            if sem_hit is not None:
                return sem_hit
        except Exception:
            pass

    # ── Single-flight coalescing (Item 7) — compute once per unique task ──────
    cache_key = f"{resolved_provider}::{task[:200]}"

    def _compute() -> ClassificationDecision:
        return _classify_inner(
            task, resolved_provider, history, context_signals, max_tier, t0, user_id
        )

    if feature_flags.single_flight_coalescing:
        from classifier.infra.coalescer import single_flight
        decision = single_flight.do(cache_key, _compute)
    else:
        decision = _compute()

    # ── Store for streaming debounce (Item 20) ────────────────────────────────
    _last_decision = decision

    return decision


def _classify_inner(
    task: str,
    resolved_provider: str,
    history: list[str] | None,
    context_signals: "ContextSignals | None",
    max_tier: ModelTier | None,
    t0: float,
    user_id: str | None,
) -> ClassificationDecision:
    # ── Layer 1 ───────────────────────────────────────────────────────────────
    layer_used = "layer1"
    try:
        task_type, complexity, tier, confidence, reasoning = classify_layer1(
            task, history=history, provider=resolved_provider
        )
    except Exception as exc:
        raise ClassificationError(f"Layer 1 classification failed: {exc}") from exc

    # ── Item 11: Apply calibration to L1 confidence ───────────────────────────
    if feature_flags.calibration:
        confidence = _apply_calibration("layer1", confidence)

    # ── Layer 3 (between L1 and L2 — fast ML classifier with abstain) ─────────
    if settings.layer3_enabled and confidence < settings.layer2_confidence_threshold:
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
    l2_fired = settings.layer2_enabled and not cost_tracker.is_exhausted_for("layer2")
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

    # ── Item 1: PII detection → force MEDIUM+ and set compliance_flag ─────────
    compliance_flag = feature_flags.pii_detection and detect_pii(task)
    if compliance_flag:
        idx = _TIER_ORDER.index(tier)
        if idx < 1:
            tier = ModelTier.MEDIUM
            model_name = MODEL_REGISTRY[resolved_provider][tier]
            reasoning += " [PII/PHI detected → bumped to MEDIUM minimum]"
        logger.warning("PII/PHI detected in task — compliance_flag=True")

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


__all__ = [
    "classify_task",
    "ClassificationDecision",
    "ContextSignals",
    "ModelTier",
    "TaskType",
    "TaskComplexity",
    "MODEL_REGISTRY",
    "TIER_MATRIX",
    "ClassificationError",
    "ConfigurationError",
    "UnsupportedProviderError",
    "LayerNotAvailableError",
    "record_feedback",
]
