"""Configuration management for codex-local-proxy-launcher."""

import logging
import os
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _parse_csv(value: str) -> list[str]:
    """Parse comma-separated environment values, preserving order."""
    items: list[str] = []
    seen = set()
    for raw_item in (value or "").split(","):
        item = raw_item.strip()
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items


def _parse_model_mapping(mapping_str: str) -> dict[str, str]:
    """Parse model mapping from env: public-model=target-model,..."""
    mapping: dict[str, str] = {}
    for raw_pair in (mapping_str or "").split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning("Ignoring invalid MODEL_MAPPING entry %r; expected public=target", pair)
            continue
        public_model, target_model = pair.split("=", 1)
        public_model = public_model.strip()
        target_model = target_model.strip()
        if public_model and target_model:
            mapping[public_model] = target_model
    return mapping


def _parse_model_pricing(pricing_str: str, env_name: str = "MODEL_PRICING_USD_PER_1K") -> dict[str, dict[str, float]]:
    """
    Parse model pricing from env.

    Format: model=input_per_1k/output_per_1k,model2=input_per_1k/output_per_1k
    Costs are USD per 1,000 tokens.
    """
    pricing: dict[str, dict[str, float]] = {}
    for raw_pair in (pricing_str or "").split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning("Ignoring invalid %s entry %r; expected model=input/output", env_name, pair)
            continue

        model, raw_costs = pair.split("=", 1)
        model = model.strip()
        raw_costs = raw_costs.strip()
        if not model or not raw_costs:
            continue

        if "/" not in raw_costs:
            logger.warning("Ignoring invalid pricing for %r; expected input/output", model)
            continue

        raw_input, raw_output = raw_costs.split("/", 1)
        try:
            input_cost = float(raw_input.strip())
            output_cost = float(raw_output.strip())
        except ValueError:
            logger.warning("Ignoring invalid pricing for %r; costs must be numbers", model)
            continue

        if input_cost < 0 or output_cost < 0:
            logger.warning("Ignoring invalid pricing for %r; costs must be non-negative", model)
            continue

        pricing[model] = {
            "input": input_cost,
            "output": output_cost,
        }

    return pricing


def _convert_pricing_per_million_to_1k(pricing: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Convert legacy USD-per-million-token pricing to USD-per-1K-token pricing."""
    return {
        model: {
            "input": costs["input"] / 1000,
            "output": costs["output"] / 1000,
        }
        for model, costs in pricing.items()
    }


def _parse_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(name: str, default: str) -> int:
    raw_value = os.getenv(name, default).strip()
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, raw_value, default)
        return int(default)


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def _codex_home() -> Path:
    return Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser()


DEFAULT_MODEL_OPTIONS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
]


class Config:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        # Local proxy settings.
        self.port = int(_env_first("CODEX_PROXY_PORT", "PROXY_PORT", default="5051"))
        self.bind_host = os.getenv("BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
        self.codex_home = _codex_home()
        self.launcher_state_dir = Path(
            _env_first(
                "CODEX_LAUNCHER_STATE_DIR",
                default=str(self.codex_home / "codex-launcher"),
            )
        ).expanduser()
        self.codex_config_path = Path(
            _env_first("CODEX_CONFIG_PATH", default=str(self.codex_home / "config.toml"))
        ).expanduser()
        self.proxy_token_file = Path(
            _env_first(
                "CODEX_PROXY_TOKEN_FILE",
                default=str(self.launcher_state_dir / "proxy_token"),
            )
        ).expanduser()
        self.proxy_access_token = (
            _env_first("CODEX_PROXY_ACCESS_TOKEN", "PROXY_ACCESS_TOKEN")
            or self._load_or_generate_proxy_token()
        )
        self.dashboard_access_token = (
            _env_first("CODEX_DASHBOARD_ACCESS_TOKEN", "DASHBOARD_ACCESS_TOKEN")
            or self.proxy_access_token
        )

        # Upstream OpenAI-compatible endpoint.
        self.target_endpoint = (
            _env_first("CODEX_TARGET_ENDPOINT", "TARGET_ENDPOINT", default="https://api.openai.com/v1").strip()
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.target_api_key = _env_first("CODEX_TARGET_API_KEY", "TARGET_API_KEY", "OPENAI_API_KEY")
        self.upstream_wire_api = (
            _env_first("CODEX_UPSTREAM_WIRE_API", default="chat_completions").strip().lower()
            or "chat_completions"
        )
        if self.upstream_wire_api not in {"chat_completions", "responses"}:
            logger.warning("Invalid CODEX_UPSTREAM_WIRE_API=%r; using chat_completions", self.upstream_wire_api)
            self.upstream_wire_api = "chat_completions"
        self.use_placeholder_mode = _parse_bool("USE_PLACEHOLDER_MODE")
        self.dev_mode = _parse_bool("DEV_MODE")
        self.skip_ssl_verify = _parse_bool("SKIP_SSL_VERIFY")
        self.auto_open_browser = _parse_bool("AUTO_OPEN_BROWSER", "true")
        self.auto_apply_codex_config = _parse_bool("AUTO_APPLY_CODEX_CONFIG", "true")
        self.auto_restart_codex_desktop = _parse_bool("AUTO_RESTART_CODEX_DESKTOP", "true")
        self.codex_app_path = Path(
            _env_first("CODEX_APP_PATH", default="/Applications/Codex.app")
        ).expanduser()

        # Request behavior.
        self.request_timeout_seconds = _parse_int("OPENAI_REQUEST_TIMEOUT_SECONDS", "600")
        self.streaming_timeout_seconds = _parse_int("OPENAI_STREAMING_TIMEOUT_SECONDS", "900")
        self.default_max_completion_tokens = _parse_int(
            "DEFAULT_MAX_COMPLETION_TOKENS",
            os.getenv("DEFAULT_MAX_TOKENS", "16384"),
        )
        self.default_max_tokens = self.default_max_completion_tokens
        self.inject_default_max_completion_tokens = _parse_bool(
            "INJECT_DEFAULT_MAX_COMPLETION_TOKENS",
            os.getenv("INJECT_DEFAULT_MAX_TOKENS", "true"),
        )
        self.inject_default_max_tokens = self.inject_default_max_completion_tokens
        self.completion_token_limit_field = (
            os.getenv("COMPLETION_TOKEN_LIMIT_FIELD", "max_completion_tokens").strip()
            or "max_completion_tokens"
        )
        if self.completion_token_limit_field not in {"max_completion_tokens", "max_tokens"}:
            logger.warning(
                "Invalid COMPLETION_TOKEN_LIMIT_FIELD=%r; using max_completion_tokens",
                self.completion_token_limit_field,
            )
            self.completion_token_limit_field = "max_completion_tokens"
        self.convert_max_tokens_to_max_completion_tokens = _parse_bool(
            "CONVERT_MAX_TOKENS_TO_MAX_COMPLETION_TOKENS",
            "true",
        )
        self.enable_duplicate_request_guard = _parse_bool("ENABLE_DUPLICATE_REQUEST_GUARD", "true")
        self.duplicate_request_ttl_seconds = _parse_int("DUPLICATE_REQUEST_TTL_SECONDS", "300")

        # Codex-facing model configuration.
        self.model_mapping = _parse_model_mapping(os.getenv("MODEL_MAPPING", ""))
        pricing_per_1k = os.getenv("MODEL_PRICING_USD_PER_1K", "")
        legacy_pricing_per_million = os.getenv("MODEL_PRICING_USD_PER_MILLION", "")
        if pricing_per_1k.strip():
            self.model_pricing = _parse_model_pricing(pricing_per_1k, "MODEL_PRICING_USD_PER_1K")
            self.model_pricing_unit = "usd_per_1k_tokens"
        elif legacy_pricing_per_million.strip():
            logger.warning(
                "MODEL_PRICING_USD_PER_MILLION is deprecated; use MODEL_PRICING_USD_PER_1K instead"
            )
            legacy_pricing = _parse_model_pricing(
                legacy_pricing_per_million,
                "MODEL_PRICING_USD_PER_MILLION",
            )
            self.model_pricing = _convert_pricing_per_million_to_1k(legacy_pricing)
            self.model_pricing_unit = "usd_per_1k_tokens"
        else:
            self.model_pricing = {}
            self.model_pricing_unit = "usd_per_1k_tokens"
        configured_models = (
            os.getenv("CODEX_MODEL_OPTIONS")
            or os.getenv("MODEL_OPTIONS")
            or os.getenv("OPENAI_MODEL_OPTIONS")
            or ""
        )
        self.model_options = _parse_csv(configured_models)
        if not self.model_options:
            self.model_options = list(self.model_mapping.keys()) or list(DEFAULT_MODEL_OPTIONS)

        self.default_model = _env_first("CODEX_DEFAULT_MODEL", "DEFAULT_MODEL").strip() or None
        if self.default_model and self.model_options and self.default_model not in self.model_options:
            logger.warning(
                "Ignoring DEFAULT_MODEL=%r because it is not present in MODEL_OPTIONS",
                self.default_model,
            )
            self.default_model = None
        if not self.default_model and self.model_options:
            self.default_model = self.model_options[0]

        self.strict_model_allowlist = _parse_bool("STRICT_MODEL_ALLOWLIST")
        self.codex_provider_id = (
            _env_first("CODEX_PROVIDER_ID", default="codex-local-proxy").strip()
            or "codex-local-proxy"
        )
        self.codex_provider_name = (
            _env_first("CODEX_PROVIDER_NAME", default="Codex Local Proxy").strip()
            or "Codex Local Proxy"
        )

        # OAuth settings.
        self.oauth_token_endpoint = os.getenv("OAUTH_TOKEN_ENDPOINT")
        self.oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
        self.oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")
        self.oauth_scope = os.getenv("OAUTH_SCOPE")
        self.oauth_refresh_buffer_minutes = _parse_int("OAUTH_REFRESH_BUFFER_MINUTES", "5")

        # SSL verification state is set by setup_ssl().
        self.ssl_enabled = True

    def _generate_token(self) -> str:
        """Generate a random access token for local proxy clients."""
        return f"codex-launcher-{secrets.token_hex(32)}"

    def _load_or_generate_proxy_token(self) -> str:
        """Load the durable local proxy token used by Codex command auth."""
        try:
            if self.proxy_token_file.exists():
                token = self.proxy_token_file.read_text(encoding="utf-8").strip()
                if token:
                    return token

            token = self._generate_token()
            self.proxy_token_file.parent.mkdir(parents=True, exist_ok=True)
            self.proxy_token_file.write_text(token + "\n", encoding="utf-8")
            self.proxy_token_file.chmod(0o600)
            return token
        except Exception as e:
            logger.warning("Failed to persist proxy token at %s: %s", self.proxy_token_file, e)
            return self._generate_token()

    def get_public_model_names(self) -> list[str]:
        """Return Codex-facing model names exposed by /v1/models."""
        return list(self.model_options)

    def is_known_public_model(self, model: str | None) -> bool:
        """Return whether the model is allowed by MODEL_OPTIONS."""
        if not model:
            return False
        if model in self.model_options:
            return True
        # Some config values are provider/model, but providers usually send only model.
        _, _, suffix = model.partition("/")
        return bool(suffix and suffix in self.model_options)

    def resolve_target_model(self, requested_model: str | None) -> tuple[str, str]:
        """
        Resolve the model sent upstream.

        Returns (public_model, target_model). public_model is the Codex-facing
        model used for logs and response rewriting; target_model is sent to the
        upstream OpenAI-compatible endpoint.
        """
        public_model = (requested_model or self.default_model or "").strip()
        if not public_model:
            raise ValueError("Request body must include a model, or DEFAULT_MODEL must be configured")

        lookup_keys = [public_model]
        if "/" in public_model:
            lookup_keys.append(public_model.split("/", 1)[1])

        if self.strict_model_allowlist and not any(key in self.model_options for key in lookup_keys):
            raise ValueError(
                f"Model {public_model!r} is not in MODEL_OPTIONS: {', '.join(self.model_options)}"
            )

        for key in lookup_keys:
            if key in self.model_mapping:
                return key, self.model_mapping[key]

        # No mapping means the public model is already the upstream model name.
        return lookup_keys[-1], lookup_keys[-1]

    def apply_completion_token_limit(self, request_payload: dict[str, Any]) -> None:
        """
        Normalize OpenAI token-limit fields for the upstream endpoint.

        GPT-5-style Chat Completions endpoints commonly expect
        max_completion_tokens, so the default behavior converts max_tokens
        before forwarding upstream.
        """
        target_field = self.completion_token_limit_field

        if target_field == "max_completion_tokens":
            if "max_completion_tokens" in request_payload:
                if self.convert_max_tokens_to_max_completion_tokens:
                    request_payload.pop("max_tokens", None)
                return

            if (
                self.convert_max_tokens_to_max_completion_tokens
                and "max_tokens" in request_payload
            ):
                request_payload["max_completion_tokens"] = request_payload.pop("max_tokens")
                return

            if (
                self.inject_default_max_completion_tokens
                and self.default_max_completion_tokens > 0
            ):
                request_payload["max_completion_tokens"] = self.default_max_completion_tokens
            return

        if "max_tokens" in request_payload or "max_completion_tokens" in request_payload:
            return

        if self.inject_default_max_completion_tokens and self.default_max_completion_tokens > 0:
            request_payload["max_tokens"] = self.default_max_completion_tokens

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate request cost from configured USD-per-1K-token pricing."""
        pricing = self.model_pricing.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000) * pricing["input"]
        output_cost = (output_tokens / 1_000) * pricing["output"]
        return input_cost + output_cost

    def get_model_pricing_table(self) -> list[dict[str, Any]]:
        """Return model pricing and mapping rows for dashboard display."""
        rows = []
        for model in self.model_options:
            pricing = self.model_pricing.get(model, {})
            rows.append(
                {
                    "model": model,
                    "target_model": self.model_mapping.get(model, model),
                    "input_cost_per_1k": pricing.get("input"),
                    "output_cost_per_1k": pricing.get("output"),
                    "configured": model in self.model_pricing,
                }
            )
        return rows

    def is_oauth_configured(self) -> bool:
        """Check if OAuth client-credentials auth is configured for upstream."""
        return bool(
            self.oauth_token_endpoint
            and self.oauth_client_id
            and self.oauth_client_secret
        )

    def is_api_key_configured(self) -> bool:
        """Check if direct upstream API key auth is configured."""
        return bool(self.target_api_key)

    def is_upstream_auth_configured(self) -> bool:
        """Return whether the proxy can authenticate to the upstream endpoint."""
        return self.dev_mode or self.is_oauth_configured() or self.is_api_key_configured()

    def get_verify_ssl(self) -> bool:
        """Get SSL verification setting for outbound upstream requests."""
        if self.skip_ssl_verify:
            return False
        return self.ssl_enabled

    def get_local_base_url(self) -> str:
        """Return the localhost base URL clients should use."""
        return f"http://localhost:{self.port}"

    def get_openai_base_url(self) -> str:
        """Return the OpenAI-compatible base URL Codex should use."""
        return f"{self.get_local_base_url()}/v1"

    def get_responses_url(self) -> str:
        """Return the Codex-facing Responses API URL."""
        return f"{self.get_openai_base_url()}/responses"

    def get_chat_completions_url(self) -> str:
        """Return the debug/backward-compatible Chat Completions URL."""
        return f"{self.get_openai_base_url()}/chat/completions"

    def get_models_url(self) -> str:
        """Return the model listing URL."""
        return f"{self.get_openai_base_url()}/models"

    def get_codex_config_snippet(self) -> dict[str, Any]:
        """Build the config.toml fragment managed by this launcher."""
        return {
            "model": self.default_model,
            "model_provider": self.codex_provider_id,
            "model_providers": {
                self.codex_provider_id: {
                    "name": self.codex_provider_name,
                    "base_url": self.get_openai_base_url(),
                    "wire_api": "responses",
                    "supports_websockets": False,
                    "stream_idle_timeout_ms": self.streaming_timeout_seconds * 1000,
                    "auth": {
                        "command": "/bin/cat",
                        "args": [str(self.proxy_token_file)],
                        "refresh_interval_ms": 0,
                    },
                }
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return non-sensitive configuration for API responses."""
        return {
            "port": self.port,
            "bind_host": self.bind_host,
            "target_endpoint": self.target_endpoint,
            "openai_base_url": self.get_openai_base_url(),
            "responses_url": self.get_responses_url(),
            "chat_completions_url": self.get_chat_completions_url(),
            "models_url": self.get_models_url(),
            "use_placeholder_mode": self.use_placeholder_mode,
            "upstream_wire_api": self.upstream_wire_api,
            "enable_duplicate_request_guard": self.enable_duplicate_request_guard,
            "duplicate_request_ttl_seconds": self.duplicate_request_ttl_seconds,
            "model_options": self.get_public_model_names(),
            "model_mapping": self.model_mapping,
            "model_pricing": self.model_pricing,
            "model_pricing_unit": self.model_pricing_unit,
            "model_pricing_table": self.get_model_pricing_table(),
            "default_model": self.default_model,
            "strict_model_allowlist": self.strict_model_allowlist,
            "default_max_completion_tokens": self.default_max_completion_tokens,
            "default_max_tokens": self.default_max_completion_tokens,
            "inject_default_max_completion_tokens": self.inject_default_max_completion_tokens,
            "inject_default_max_tokens": self.inject_default_max_completion_tokens,
            "completion_token_limit_field": self.completion_token_limit_field,
            "convert_max_tokens_to_max_completion_tokens": self.convert_max_tokens_to_max_completion_tokens,
            "request_timeout_seconds": self.request_timeout_seconds,
            "streaming_timeout_seconds": self.streaming_timeout_seconds,
            "oauth_configured": self.is_oauth_configured(),
            "api_key_configured": self.is_api_key_configured(),
            "dev_mode": self.dev_mode,
            "ssl_enabled": self.ssl_enabled,
            "codex_provider_id": self.codex_provider_id,
            "codex_provider_name": self.codex_provider_name,
            "codex_home": str(self.codex_home),
            "codex_config_path": str(self.codex_config_path),
            "codex_app_path": str(self.codex_app_path),
            "proxy_token_file": str(self.proxy_token_file),
            "auto_apply_codex_config": self.auto_apply_codex_config,
            "auto_restart_codex_desktop": self.auto_restart_codex_desktop,
            "codex_config_snippet": self.get_codex_config_snippet(),
        }


def setup_ssl() -> bool:
    """
    Enable enterprise certificate support when available.

    Returns True when SSL verification should remain enabled. Missing optional
    enterprise certificate tooling is non-fatal and falls back to the system
    certificate store.
    """
    try:
        import rbc_security

        rbc_security.enable_certs()
        logger.info("Enterprise certificate support enabled")
        return True
    except ImportError:
        logger.warning("rbc_security not available - using system certificate store")
        return True
    except Exception as e:
        logger.warning("rbc_security setup failed: %s - using system certificate store", e)
        return True
