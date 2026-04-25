import logging
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from classifier.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
_VALID_PROVIDERS = {"google", "openai", "anthropic"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API keys ──────────────────────────────────────────────────────────────
    google_api_key:    str = ""
    openai_api_key:    str = ""
    anthropic_api_key: str = ""

    # ── Classifier behaviour ──────────────────────────────────────────────────
    default_provider: str  = "google"
    layer2_enabled:   bool = False
    layer3_enabled:   bool = False
    layer4_enabled:   bool = False

    # ── Cascade confidence thresholds ─────────────────────────────────────────
    layer3_confidence_threshold: float = 0.85
    layer2_confidence_threshold: float = 0.75

    # ── Layer 2 settings ──────────────────────────────────────────────────────
    layer2_model:      str = "gemini-2.5-flash-lite"
    layer2_timeout_ms: int = 2000
    layer2_max_rpm:    int = 100

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_enabled:   bool = True
    cache_max_size:  int  = 10_000
    cache_ttl_secs:  int  = 3600

    # ── Cost / budget ─────────────────────────────────────────────────────────
    monthly_budget_usd: float = 1000.0

    # ── Decision logging ──────────────────────────────────────────────────────
    log_decisions: bool = True

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in _VALID_PROVIDERS:
            raise ValueError(
                f"DEFAULT_PROVIDER='{v}' is invalid. "
                f"Must be one of: {sorted(_VALID_PROVIDERS)}"
            )
        return v

    def api_key_for(self, provider: str) -> str:
        key_map = {
            "google":    self.google_api_key,
            "openai":    self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }
        if provider not in key_map:
            raise ConfigurationError(
                f"Unknown provider '{provider}'. Supported: {sorted(key_map)}"
            )
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
