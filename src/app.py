#!/usr/bin/env python3
"""Codex Local Proxy Launcher."""

import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, session

SRC_DIR = Path(__file__).parent.resolve()
load_dotenv(SRC_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from config import Config, setup_ssl
from codex_config_manager import maybe_apply_and_launch
from handlers import dashboard_bp, proxy_bp
from logger_manager import LoggerManager
from oauth_manager import OAuthManager


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder=str(SRC_DIR / "templates"))
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)

    config = Config()
    config.ssl_enabled = setup_ssl()
    app.config["KL_CONFIG"] = config

    log_manager = LoggerManager()
    app.config["LOG_MANAGER"] = log_manager

    oauth_manager = None
    if config.is_oauth_configured() and not config.dev_mode:
        try:
            oauth_manager = OAuthManager(
                token_endpoint=config.oauth_token_endpoint,
                client_id=config.oauth_client_id,
                client_secret=config.oauth_client_secret,
                scope=config.oauth_scope,
                refresh_buffer_minutes=config.oauth_refresh_buffer_minutes,
                verify_ssl=config.get_verify_ssl(),
            )
            logger.info("Attempting initial OAuth token fetch...")
            token = oauth_manager.get_token()
            if token:
                logger.info("OAuth token obtained successfully")
            else:
                logger.warning("Failed to obtain OAuth token")
        except Exception as e:
            logger.error("OAuth initialization failed: %s", e)
            oauth_manager = None

    app.config["OAUTH_MANAGER"] = oauth_manager

    app.register_blueprint(proxy_bp)
    app.register_blueprint(dashboard_bp)

    @app.route("/")
    def dashboard():
        # Local dashboard is intentionally frictionless. The proxy API itself
        # still requires the configured bearer token.
        session["dashboard_authenticated"] = True
        return render_template("index.html")

    log_manager.log_server_event(
        "info",
        "Codex Local Proxy Launcher started",
        {
            "port": config.port,
            "mode": "placeholder" if config.use_placeholder_mode else "proxy",
            "target": config.target_endpoint,
            "upstream_wire_api": config.upstream_wire_api,
            "oauth": config.is_oauth_configured(),
            "ssl": config.ssl_enabled,
            "models": config.get_public_model_names(),
            "codex_provider_id": config.codex_provider_id,
        },
    )

    return app


def open_browser(port: int):
    """Open the dashboard after a short delay."""
    import time

    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{port}")


def wait_for_proxy_health(config, log_manager, timeout_seconds: float = 30.0) -> bool:
    """Wait until the local proxy answers health and model checks."""
    import time

    import httpx

    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            health_response = httpx.get(f"{config.get_local_base_url()}/health", timeout=2)
            models_response = httpx.get(
                config.get_models_url(),
                headers={"Authorization": f"Bearer {config.proxy_access_token}"},
                timeout=2,
            )
            if health_response.status_code == 200 and models_response.status_code == 200:
                log_manager.log_server_event(
                    "info",
                    "Proxy health checks passed",
                    {
                        "health_status": health_response.status_code,
                        "models_status": models_response.status_code,
                    },
                )
                return True
            last_error = f"health={health_response.status_code}, models={models_response.status_code}"
        except Exception as e:
            last_error = str(e)
        time.sleep(0.5)

    log_manager.log_server_event("error", f"Proxy health checks failed before Codex launch: {last_error}")
    return False


def apply_config_and_launch_codex(app: Flask):
    """Apply Codex config and launch desktop after the proxy starts."""
    import time

    time.sleep(1.5)
    with app.app_context():
        config = app.config["KL_CONFIG"]
        log_manager = app.config["LOG_MANAGER"]
        try:
            if not wait_for_proxy_health(config, log_manager):
                return
            maybe_apply_and_launch(config, log_manager)
        except Exception as e:
            logger.exception("Failed to apply Codex config or launch desktop")
            log_manager.log_server_event("error", f"Failed to apply Codex config or launch desktop: {e}")


def main():
    """Main entry point."""
    app = create_app()
    config = app.config["KL_CONFIG"]

    print()
    print("=" * 68)
    print("  Codex Local Proxy Launcher")
    print("=" * 68)
    print()
    print(f"  Dashboard:       http://localhost:{config.port}")
    print(f"  OpenAI Base URL: {config.get_openai_base_url()}")
    print(f"  Responses API:   {config.get_responses_url()}")
    print(f"  Chat Debug API:  {config.get_chat_completions_url()}")
    print(f"  Models Endpoint: {config.get_models_url()}")
    print()
    print(f"  Target:          {config.target_endpoint}")
    print(f"  Upstream API:    {config.upstream_wire_api}")
    print(f"  Mode:            {'Placeholder' if config.use_placeholder_mode else 'Proxy'}")
    print(f"  SSL:             {'Enabled' if config.ssl_enabled else 'Disabled'}")
    print(f"  Bind Host:       {config.bind_host}")
    print(f"  Models:          {', '.join(config.get_public_model_names()) or '(none configured)'}")
    print()
    print("  Codex managed provider settings:")
    print()
    print(f"    Provider:   {config.codex_provider_id}")
    print(f"    Model:      {config.default_model}")
    print(f"    Base URL:   {config.get_openai_base_url()}")
    print(f"    Token file: {config.proxy_token_file}")
    print(f"    Config:     {config.codex_config_path}")
    print()
    print("=" * 68)
    print()

    if config.auto_open_browser:
        threading.Thread(target=open_browser, args=(config.port,), daemon=True).start()
    if config.auto_apply_codex_config or config.auto_restart_codex_desktop:
        threading.Thread(target=apply_config_and_launch_codex, args=(app,), daemon=True).start()

    try:
        app.run(
            host=config.bind_host,
            port=config.port,
            debug=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Server error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
