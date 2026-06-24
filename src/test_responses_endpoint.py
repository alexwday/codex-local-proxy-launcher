#!/usr/bin/env python3
"""Smoke-test native Responses API support using the Codex proxy launcher env."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent.resolve()
load_dotenv(SRC_DIR / ".env")

from config import Config, setup_ssl
from oauth_manager import OAuthManager
from usage_extractor import extract_usage_tokens


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the configured OpenAI-compatible /responses endpoint works."
    )
    parser.add_argument(
        "--local-proxy",
        action="store_true",
        help="Call the running local proxy at /v1/responses instead of the upstream target endpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CODEX_RESPONSES_TEST_MODEL"),
        help=(
            "Model to send. Defaults to CODEX_RESPONSES_TEST_MODEL when set; otherwise the "
            "mapped upstream model for upstream tests, or the Codex-facing default model for --local-proxy."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("CODEX_RESPONSES_TEST_PROMPT", "Reply with exactly: pong"),
        help="Prompt/input text to send.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.getenv("CODEX_RESPONSES_TEST_MAX_OUTPUT_TOKENS", "32")),
        help="max_output_tokens for the smoke request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("CODEX_RESPONSES_TEST_TIMEOUT_SECONDS", "60")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Send stream=true and verify the endpoint can open an SSE response.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved endpoint and payload without making the HTTP request.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def _redact(value: str | None) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return value[:2] + "..."
    return value[:6] + "..." + value[-4:]


def _endpoint_for(config: Config, local_proxy: bool) -> str:
    if local_proxy:
        return config.get_responses_url()
    return f"{config.target_endpoint}/responses"


def _model_for(config: Config, explicit_model: str | None, local_proxy: bool) -> tuple[str, str | None]:
    if explicit_model:
        return explicit_model, None

    default_model = config.default_model
    if local_proxy:
        return default_model or "gpt-5.5", None

    public_model, target_model = config.resolve_target_model(default_model)
    return target_model, public_model


def _auth_for(config: Config, local_proxy: bool) -> tuple[str, str, OAuthManager | None]:
    if local_proxy:
        return config.proxy_access_token, "local proxy token", None

    if config.dev_mode:
        return "dev-mock-token", "dev mock token", None

    if config.is_oauth_configured():
        oauth_manager = OAuthManager(
            token_endpoint=config.oauth_token_endpoint,
            client_id=config.oauth_client_id,
            client_secret=config.oauth_client_secret,
            scope=config.oauth_scope,
            refresh_buffer_minutes=config.oauth_refresh_buffer_minutes,
            verify_ssl=config.get_verify_ssl(),
        )
        token = oauth_manager.get_token()
        if not token:
            oauth_manager.destroy()
            raise RuntimeError("OAuth is configured, but no access token was returned.")
        return token, "oauth client credentials", oauth_manager

    if config.target_api_key:
        return config.target_api_key, "target api key", None

    raise RuntimeError("No upstream auth configured. Set OAuth credentials or TARGET_API_KEY in src/.env.")


def _auth_preview_for(config: Config, local_proxy: bool) -> tuple[str, str]:
    """Return non-network auth preview values for dry-runs."""
    if local_proxy:
        return config.proxy_access_token, "local proxy token"
    if config.dev_mode:
        return "dev-mock-token", "dev mock token"
    if config.is_oauth_configured():
        return "(not fetched in dry-run)", "oauth client credentials"
    if config.target_api_key:
        return config.target_api_key, "target api key"
    return "(none)", "not configured"


def _payload(model: str, prompt: str, max_output_tokens: int, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if stream:
        payload["stream"] = True
    return payload


def _extract_output_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    text_parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                text_parts.append(content["text"])
    return "".join(text_parts)


def _print_request_summary(
    *,
    config: Config,
    endpoint: str,
    auth_mode: str,
    bearer_token: str,
    payload: dict[str, Any],
    mapped_from: str | None,
    local_proxy: bool,
) -> None:
    mode = "local proxy" if local_proxy else "upstream target"
    print()
    print("Responses API smoke test")
    print("=" * 64)
    print(f"Mode:        {mode}")
    print(f"Endpoint:    {endpoint}")
    print(f"Env file:    {SRC_DIR / '.env'}")
    print(f"Auth:        {auth_mode} ({_redact(bearer_token)})")
    print(f"SSL verify:  {config.get_verify_ssl()}")
    if mapped_from:
        print(f"Model:       {mapped_from} -> {payload['model']}")
    else:
        print(f"Model:       {payload['model']}")
    print(f"Stream:      {bool(payload.get('stream'))}")
    print("=" * 64)


def _post_non_streaming(endpoint: str, headers: dict[str, str], payload: dict[str, Any], timeout: float, verify: bool) -> int:
    started = time.time()
    with httpx.Client(timeout=timeout, verify=verify) as client:
        response = client.post(endpoint, headers=headers, json=payload)
    elapsed_ms = int((time.time() - started) * 1000)

    print(f"HTTP status: {response.status_code} ({elapsed_ms}ms)")
    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text[:2000]}

    if response.status_code >= 400:
        print("Response error body:")
        print(json.dumps(body, indent=2, sort_keys=True)[:4000])
        return 1

    input_tokens, output_tokens = extract_usage_tokens(body)
    output_text = _extract_output_text(body)

    print(f"Response id: {body.get('id', '-')}")
    print(f"Model:       {body.get('model', '-')}")
    print(f"Status:      {body.get('status', '-')}")
    print(f"Usage:       {input_tokens} input / {output_tokens} output")
    print(f"Output:      {(output_text or '(no output_text found)')[:500]}")
    print()
    print("OK: /responses returned a successful non-streaming response.")
    return 0


def _post_streaming(endpoint: str, headers: dict[str, str], payload: dict[str, Any], timeout: float, verify: bool) -> int:
    started = time.time()
    events_seen: list[str] = []
    completed_payload: dict[str, Any] | None = None

    with httpx.Client(timeout=timeout, verify=verify) as client:
        with client.stream("POST", endpoint, headers=headers, json=payload) as response:
            elapsed_ms = int((time.time() - started) * 1000)
            print(f"HTTP status: {response.status_code} ({elapsed_ms}ms)")
            if response.status_code >= 400:
                print("Response error body:")
                print(response.read().decode("utf-8", errors="replace")[:4000])
                return 1

            current_event = ""
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    events_seen.append(current_event)
                    continue
                if not line.startswith("data:"):
                    continue
                raw_data = line.split(":", 1)[1].strip()
                if raw_data == "[DONE]":
                    break
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue
                event_type = data.get("type") or current_event
                if event_type and event_type not in events_seen:
                    events_seen.append(str(event_type))
                if event_type == "response.completed":
                    response_payload = data.get("response")
                    if isinstance(response_payload, dict):
                        completed_payload = response_payload

    if not events_seen:
        print("No SSE events received.")
        return 1

    print(f"Events:      {', '.join(events_seen[:12])}")
    if completed_payload:
        input_tokens, output_tokens = extract_usage_tokens(completed_payload)
        print(f"Response id: {completed_payload.get('id', '-')}")
        print(f"Status:      {completed_payload.get('status', '-')}")
        print(f"Usage:       {input_tokens} input / {output_tokens} output")
    print()
    print("OK: /responses opened a successful streaming response.")
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    config = Config()
    config.ssl_enabled = setup_ssl()

    endpoint = _endpoint_for(config, args.local_proxy)
    model, mapped_from = _model_for(config, args.model, args.local_proxy)
    payload = _payload(model, args.prompt, args.max_output_tokens, args.stream)
    oauth_manager = None

    try:
        if args.dry_run:
            bearer_token, auth_mode = _auth_preview_for(config, args.local_proxy)
        else:
            bearer_token, auth_mode, oauth_manager = _auth_for(config, args.local_proxy)
        _print_request_summary(
            config=config,
            endpoint=endpoint,
            auth_mode=auth_mode,
            bearer_token=bearer_token,
            payload=payload,
            mapped_from=mapped_from,
            local_proxy=args.local_proxy,
        )

        if args.dry_run:
            print("Dry run payload:")
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }
        if args.stream:
            return _post_streaming(endpoint, headers, payload, args.timeout, config.get_verify_ssl())
        return _post_non_streaming(endpoint, headers, payload, args.timeout, config.get_verify_ssl())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    finally:
        if oauth_manager is not None:
            oauth_manager.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
