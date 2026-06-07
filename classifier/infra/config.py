import logging
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from classifier.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API keys ──────────────────────────────────────────────────────────────
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Classifier behaviour ──────────────────────────────────────────────────
    default_provider: str = "google"
    layer1_enabled: bool = True
    layer2_enabled: bool = False
    layer3_enabled: bool = False
    layer4_enabled: bool = False

    # ── Cascade confidence thresholds ─────────────────────────────────────────
    layer3_confidence_threshold: float = 0.85
    layer2_confidence_threshold: float = 0.75

    # ── Layer 3 settings ──────────────────────────────────────────────────────
    # Strategy: zeroshot (no data, ~80ms) | head (1.5K examples, ~15ms) | distilbert (5K, ~12ms)
    layer3_strategy: str = "zeroshot"
    # Quality<->savings dial for Layer 3 (env: L3_DMR_SAVINGS_LEVEL). 0 = quality
    # (L3's natural tier); each step shifts the chosen tier one notch cheaper
    # (HIGH->MEDIUM->LOW), clamped at LOW. Lets one trained head run anywhere on
    # the cost/quality frontier without retraining. Default 0 = no change.
    l3_dmr_savings_level: int = 0

    # ── Agentic cost levers (router-wide; all opt-in, default = today's behavior) ─
    # Field names are dmr_-prefixed so the env var is DMR_* (pydantic maps field→UPPER).
    # Posture dial 0..4 (Off/Saver/Balanced/Aggressive/Max) — presets the levers below.
    dmr_savings_level: int = 0  # DMR_SAVINGS_LEVEL
    dmr_cache_aware: bool = False  # DMR_CACHE_AWARE — account for prompt-cache economics
    dmr_context_reduction: str = "off"  # DMR_CONTEXT_REDUCTION: off | prune
    dmr_effort_routing: bool = False  # DMR_EFFORT_ROUTING — vary thinking budget per call
    dmr_model_routing: str = "off"  # DMR_MODEL_ROUTING: off | dispatch_downgrade | turn
    dmr_escalate_on_failure: bool = False  # DMR_ESCALATE_ON_FAILURE — escalate on refusals
    dmr_routing_scope: str = "call"  # DMR_ROUTING_SCOPE: call | turn | agent | conversation
    dmr_scope_decision_ttl_s: int = 300  # how long a sticky per-scope decision is reused
    # Higher abstain threshold than `layer3_confidence_threshold` because zero-shot
    # is uncalibrated and tends to be over-confident on out-of-distribution inputs.
    layer3_zeroshot_threshold: float = 0.85

    # ── Layer 2 settings ──────────────────────────────────────────────────────
    layer2_provider: str = ""  # "" = same as default_provider
    layer2_model: str = "gemini-2.5-flash-lite"
    layer2_timeout_ms: int = 3500
    layer2_max_rpm: int = 100
    layer2_fallback_model: str = ""
    layer2_monthly_budget_usd: float = 0.0  # 0 = auto (5% of monthly_budget_usd)

    # ── PII / compliance ──────────────────────────────────────────────────────
    pii_scrub_strict: bool = False  # also scrub all-caps names + addresses

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_max_size: int = 10_000
    cache_ttl_secs: int = 3600
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = 0.92

    # ── Cost / budget ─────────────────────────────────────────────────────────
    monthly_budget_usd: float = 1000.0

    # ── Decision logging ──────────────────────────────────────────────────────
    log_decisions: bool = True
    debug_ab_mode: bool = False

    # ── Test mode ─────────────────────────────────────────────────────────────
    classifier_test_mode: bool = False  # set CLASSIFIER_TEST_MODE=1 in env

    # ── Domain keyword packs ──────────────────────────────────────────────────
    keyword_packs: str = ""  # comma-separated names: "healthcare,fintech"

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        # Trust the registry — custom providers can be registered via
        # classifier.register_provider() before this validator runs in user code.
        # Empty string isn't valid.
        if not v:
            raise ValueError("DEFAULT_PROVIDER cannot be empty.")
        return v

    def api_key_for(self, provider: str) -> str:
        key_map = {
            "google": self.google_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        if provider not in key_map:
            raise ConfigurationError(f"Unknown provider '{provider}'. Supported: {sorted(key_map)}")
        key = key_map[provider]
        if not key or key.startswith("your_"):
            raise ConfigurationError(
                f"API key for '{provider}' is not configured. "
                f"Set {provider.upper()}_API_KEY in your .env file."
            )
        return key


try:
    settings = Settings()
    logger.debug(
        "Config loaded: provider=%s layer2=%s layer3=%s cache=%s budget=$%.0f",
        settings.default_provider,
        settings.layer2_enabled,
        settings.layer3_enabled,
        settings.cache_enabled,
        settings.monthly_budget_usd,
    )
except Exception as exc:
    raise ConfigurationError(
        f"Failed to load settings from {_ENV_FILE}: {exc}\n"
        "Hint: copy .env.example to .env and fill in your API keys."
    ) from exc
