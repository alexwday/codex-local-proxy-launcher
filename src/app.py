#!/usr/bin/env python3
"""Kilo-Launcher - local OpenAI-compatible proxy for Kilo Code."""

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
        "Kilo-Launcher started",
        {
            "port": config.port,
            "mode": "placeholder" if config.use_placeholder_mode else "proxy",
            "target": config.target_endpoint,
            "oauth": config.is_oauth_configured(),
            "ssl": config.ssl_enabled,
            "models": config.get_public_model_names(),
        },
    )

    return app


def open_browser(port: int):
    """Open the dashboard after a short delay."""
    import time

    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{port}")


def main():
    """Main entry point."""
    app = create_app()
    config = app.config["KL_CONFIG"]

    print()
    print("=" * 68)
    print("  Kilo-Launcher - OpenAI-compatible proxy for Kilo Code")
    print("=" * 68)
    print()
    print(f"  Dashboard:       http://localhost:{config.port}")
    print(f"  OpenAI Base URL: {config.get_openai_base_url()}")
    print(f"  Chat Endpoint:   {config.get_openai_base_url()}/chat/completions")
    print(f"  Models Endpoint: {config.get_openai_base_url()}/models")
    print()
    print(f"  Target:          {config.target_endpoint}")
    print(f"  Mode:            {'Placeholder' if config.use_placeholder_mode else 'Proxy'}")
    print(f"  SSL:             {'Enabled' if config.ssl_enabled else 'Disabled'}")
    print(f"  Bind Host:       {config.bind_host}")
    print(f"  Models:          {', '.join(config.get_public_model_names()) or '(none configured)'}")
    print()
    print("  Kilo Code OpenAI-compatible provider settings:")
    print()
    print(f"    Base URL: {config.get_openai_base_url()}")
    print(f"    API Key:  {config.proxy_access_token}")
    print()
    print("=" * 68)
    print()

    if config.auto_open_browser:
        threading.Thread(target=open_browser, args=(config.port,), daemon=True).start()

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
