"""Dashboard API endpoints for codex-local-proxy-launcher."""

import hmac
from functools import wraps

import httpx
from flask import Blueprint, jsonify, request, session

from codex_config_manager import (
    apply_codex_config,
    get_codex_status,
    restart_codex_desktop,
    restore_codex_config,
)

dashboard_bp = Blueprint("dashboard", __name__)


def get_config():
    """Get config from Flask app context."""
    from flask import current_app

    return current_app.config["KL_CONFIG"]


def get_log_manager():
    """Get log manager from Flask app context."""
    from flask import current_app

    return current_app.config["LOG_MANAGER"]


def _extract_api_key_from_request() -> str:
    """Extract API key from x-api-key or Authorization header."""
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        return api_key

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return ""


def _redact_token(token: str) -> str:
    """Return a safe token preview for UI display."""
    if not token:
        return ""
    if len(token) <= 12:
        return token[:4] + "..." if len(token) > 4 else token
    return f"{token[:16]}...{token[-6:]}"


def _verify_dashboard_api_key():
    """Verify dashboard API key."""
    if session.get("dashboard_authenticated"):
        return True, None

    config = get_config()
    expected_token = config.dashboard_access_token
    provided_token = _extract_api_key_from_request()

    if not provided_token:
        return False, {"error": "Missing dashboard API key"}

    if not hmac.compare_digest(provided_token, expected_token):
        return False, {"error": "Invalid dashboard API key"}

    return True, None


def require_dashboard_auth(handler):
    """Protect dashboard API endpoints with token auth."""

    @wraps(handler)
    def wrapper(*args, **kwargs):
        valid, error = _verify_dashboard_api_key()
        if not valid:
            return jsonify(error), 401
        return handler(*args, **kwargs)

    return wrapper


@dashboard_bp.route("/api/config", methods=["GET"])
@require_dashboard_auth
def get_configuration():
    """Get current configuration with sensitive data redacted."""
    config = get_config()
    config_dict = config.to_dict()
    codex_status = get_codex_status(config)
    config_dict.update(
        {
            "localBaseUrl": config.get_local_base_url(),
            "openaiBaseUrl": config.get_openai_base_url(),
            "accessTokenPreview": _redact_token(config.proxy_access_token),
            "codexStatus": codex_status,
            "codexConfig": config.get_codex_config_snippet(),
            "proxyTokenFile": str(config.proxy_token_file),
        }
    )
    return jsonify(config_dict)


@dashboard_bp.route("/api/setup", methods=["GET"])
@require_dashboard_auth
def get_setup():
    """Return setup values for Codex."""
    config = get_config()
    return jsonify(
        {
            "provider": config.codex_provider_name,
            "providerId": config.codex_provider_id,
            "baseUrl": config.get_openai_base_url(),
            "apiKeyPreview": _redact_token(config.proxy_access_token),
            "models": config.get_public_model_names(),
            "codexConfig": config.get_codex_config_snippet(),
            "codexStatus": get_codex_status(config),
        }
    )


@dashboard_bp.route("/api/models", methods=["GET"])
@require_dashboard_auth
def get_models():
    """Get model options and mappings."""
    config = get_config()
    return jsonify(
        {
            "models": config.get_public_model_names(),
            "defaultModel": config.default_model,
            "mapping": config.model_mapping,
            "pricing": config.model_pricing,
            "pricingTable": config.get_model_pricing_table(),
            "strictAllowlist": config.strict_model_allowlist,
        }
    )


@dashboard_bp.route("/api/status", methods=["GET"])
@require_dashboard_auth
def get_status():
    """Get current proxy status."""
    config = get_config()
    return jsonify(
        {
            "proxy": {
                "running": True,
                "port": config.port,
                "mode": "placeholder" if config.use_placeholder_mode else "proxy",
                "baseUrl": config.get_openai_base_url(),
                "responsesUrl": config.get_responses_url(),
                "upstreamWireApi": config.upstream_wire_api,
            },
            "target": {
                "endpoint": config.target_endpoint,
                "authentication": "oauth"
                if config.is_oauth_configured()
                else ("api_key" if config.is_api_key_configured() else ("dev" if config.dev_mode else "none")),
                "configured": config.is_upstream_auth_configured() or config.use_placeholder_mode,
            },
            "models": config.get_public_model_names(),
            "codex": get_codex_status(config),
        }
    )


@dashboard_bp.route("/api/codex/status", methods=["GET"])
@require_dashboard_auth
def get_codex_configuration_status():
    """Return Codex Desktop/config status."""
    return jsonify(get_codex_status(get_config()))


@dashboard_bp.route("/api/codex/apply-config", methods=["POST"])
@require_dashboard_auth
def apply_codex_configuration():
    """Apply the managed Codex config."""
    return jsonify(apply_codex_config(get_config()))


@dashboard_bp.route("/api/codex/restore-config", methods=["POST"])
@require_dashboard_auth
def restore_codex_configuration():
    """Restore Codex config captured before this launcher applied changes."""
    return jsonify(restore_codex_config(get_config()))


@dashboard_bp.route("/api/codex/restart-desktop", methods=["POST"])
@require_dashboard_auth
def restart_codex_desktop_endpoint():
    """Restart or launch Codex Desktop."""
    return jsonify(restart_codex_desktop(get_config()))


@dashboard_bp.route("/api/codex/smoke-test", methods=["POST"])
@require_dashboard_auth
def run_codex_proxy_smoke_test():
    """Run a local Responses API smoke test through the proxy."""
    config = get_config()
    payload = {
        "model": config.default_model,
        "input": "Reply with exactly: codex-proxy-ok",
        "stream": False,
        "max_output_tokens": 32,
    }
    try:
        response = httpx.post(
            config.get_responses_url(),
            headers={"Authorization": f"Bearer {config.proxy_access_token}"},
            json=payload,
            timeout=min(config.request_timeout_seconds, 30),
        )
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text[:1000]}
        return jsonify(
            {
                "ok": response.status_code < 400,
                "status": response.status_code,
                "request": payload,
                "response": body,
            }
        ), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "request": payload}), 200


@dashboard_bp.route("/api/logs", methods=["GET"])
@require_dashboard_auth
def get_logs():
    """Get all logs."""
    log_manager = get_log_manager()
    limit = request.args.get("limit", 50, type=int)

    return jsonify(
        {
            "apiCalls": log_manager.get_api_calls(limit),
            "serverEvents": log_manager.get_server_events(limit),
        }
    )


@dashboard_bp.route("/api/logs/api-calls", methods=["GET"])
@require_dashboard_auth
def get_api_logs():
    """Get API call logs."""
    log_manager = get_log_manager()
    limit = request.args.get("limit", 50, type=int)
    return jsonify(log_manager.get_api_calls(limit))


@dashboard_bp.route("/api/logs/server-events", methods=["GET"])
@require_dashboard_auth
def get_server_logs():
    """Get server event logs."""
    log_manager = get_log_manager()
    limit = request.args.get("limit", 50, type=int)
    return jsonify(log_manager.get_server_events(limit))


@dashboard_bp.route("/api/logs", methods=["DELETE"])
@require_dashboard_auth
def clear_logs():
    """Clear all logs."""
    log_manager = get_log_manager()
    log_manager.clear_logs()
    return jsonify({"success": True, "message": "Logs cleared"})


@dashboard_bp.route("/api/usage", methods=["GET"])
@require_dashboard_auth
def get_usage():
    """Get usage statistics."""
    log_manager = get_log_manager()
    return jsonify(log_manager.get_usage_stats())


@dashboard_bp.route("/api/usage/reset", methods=["POST"])
@require_dashboard_auth
def reset_usage():
    """Reset usage statistics."""
    log_manager = get_log_manager()
    log_manager.reset_usage()
    return jsonify({"success": True, "message": "Usage statistics reset"})


@dashboard_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "codex-local-proxy-launcher"})
