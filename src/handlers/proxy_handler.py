"""OpenAI-compatible proxy handler for Kilo Code."""

import hashlib
import hmac
import json
import logging
import threading
import time
from collections.abc import Iterable
from typing import Any

import httpx
from flask import Blueprint, Response, jsonify, request, stream_with_context
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

proxy_bp = Blueprint("proxy", __name__)
_inflight_lock = threading.Lock()
_inflight_requests: dict[str, float] = {}

_OPENAI_CHAT_COMPLETION_KEYS = {
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "metadata",
    "modalities",
    "model",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "web_search_options",
}


def get_config():
    """Get config from Flask app context."""
    from flask import current_app

    return current_app.config["KL_CONFIG"]


def get_oauth_manager():
    """Get OAuth manager from Flask app context."""
    from flask import current_app

    return current_app.config.get("OAUTH_MANAGER")


def get_log_manager():
    """Get log manager from Flask app context."""
    from flask import current_app

    return current_app.config["LOG_MANAGER"]


def _extract_request_api_key() -> str:
    """Extract API key from x-api-key or Authorization header."""
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        return api_key

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return ""


def verify_proxy_api_key():
    """Verify the request API key matches the local proxy access token."""
    config = get_config()
    api_key = _extract_request_api_key()

    if not api_key:
        return False, openai_error("Missing API key", "authentication_error")

    if not hmac.compare_digest(api_key, config.proxy_access_token):
        return False, openai_error("Invalid API key", "authentication_error")

    return True, None


def openai_error(message: str, error_type: str = "api_error", code: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-compatible error payload."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _build_request_fingerprint(openai_request: dict[str, Any]) -> str:
    """Build stable fingerprint for duplicate in-flight detection."""
    payload = {
        "body": openai_request,
        "api_key": _extract_request_api_key(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cleanup_stale_inflight(now: float, ttl_seconds: int):
    """Drop stale in-flight entries to avoid unbounded growth."""
    stale = [key for key, started_at in _inflight_requests.items() if (now - started_at) > ttl_seconds]
    for key in stale:
        _inflight_requests.pop(key, None)


def _register_inflight_request(config, openai_request: dict[str, Any]):
    """Register request as in-flight; return (fingerprint, accepted)."""
    if not getattr(config, "enable_duplicate_request_guard", True):
        return None, True

    fingerprint = _build_request_fingerprint(openai_request)
    now = time.time()

    with _inflight_lock:
        ttl_seconds = max(
            30,
            int(config.duplicate_request_ttl_seconds),
            int(config.streaming_timeout_seconds),
        )
        _cleanup_stale_inflight(now, ttl_seconds)
        if fingerprint in _inflight_requests:
            return fingerprint, False
        _inflight_requests[fingerprint] = now

    return fingerprint, True


def _release_inflight_request(fingerprint: str | None):
    """Release in-flight registration."""
    if not fingerprint:
        return
    with _inflight_lock:
        _inflight_requests.pop(fingerprint, None)


def _get_upstream_api_key(config, oauth_manager) -> str:
    """Resolve upstream bearer token/API key."""
    if config.dev_mode:
        return "dev-mock-token"

    if oauth_manager:
        token = oauth_manager.get_token()
        if token:
            return token

    if config.target_api_key:
        return config.target_api_key

    raise RuntimeError("No upstream authentication configured. Set OAuth credentials or TARGET_API_KEY.")


def _build_openai_client(config, oauth_manager, *, timeout_seconds: int) -> OpenAI:
    """Build an OpenAI SDK client pointed at the target endpoint."""
    api_key = _get_upstream_api_key(config, oauth_manager)
    http_client = httpx.Client(verify=config.get_verify_ssl(), timeout=timeout_seconds)
    return OpenAI(
        api_key=api_key,
        base_url=config.target_endpoint,
        timeout=timeout_seconds,
        http_client=http_client,
    )


def _close_client(client: OpenAI):
    """Close OpenAI SDK HTTP resources if supported by the installed SDK."""
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _split_sdk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Move non-SDK-known fields into extra_body while preserving wire payload."""
    sdk_payload: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    for key, value in payload.items():
        if key in _OPENAI_CHAT_COMPLETION_KEYS:
            sdk_payload[key] = value
        else:
            extra_body[key] = value

    if extra_body:
        sdk_payload["extra_body"] = extra_body
    return sdk_payload


def _prepare_upstream_request(config, openai_request: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    """Return (sdk_payload, public_model, target_model)."""
    upstream_request = dict(openai_request)
    public_model, target_model = config.resolve_target_model(upstream_request.get("model"))
    upstream_request["model"] = target_model

    config.apply_completion_token_limit(upstream_request)

    return _split_sdk_payload(upstream_request), public_model, target_model


def _object_to_dict(value: Any) -> dict[str, Any]:
    """Convert OpenAI SDK models or plain dicts to JSON-serializable dicts."""
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return json.loads(json.dumps(value, default=lambda obj: getattr(obj, "__dict__", str(obj))))


def _restore_response_model(payload: dict[str, Any], public_model: str) -> dict[str, Any]:
    """Rewrite upstream model name to the Kilo-facing model name."""
    if public_model and payload.get("model"):
        payload = dict(payload)
        payload["model"] = public_model
    return payload


def _extract_usage_tokens(payload: dict[str, Any]) -> tuple[int, int]:
    """Extract prompt/completion usage from OpenAI-compatible response shapes."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return 0, 0

    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("promptTokens")
        or usage.get("inputTokens")
        or 0
    )
    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completionTokens")
        or usage.get("outputTokens")
        or 0
    )
    return int(input_tokens or 0), int(output_tokens or 0)


def _make_model_object(model_id: str) -> dict[str, Any]:
    """Build an OpenAI-compatible model list item."""
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "kilo-launcher",
    }


@proxy_bp.route("/v1/models", methods=["GET"])
def models():
    """Return env-configured Kilo-facing model options."""
    start_time = time.time()
    config = get_config()
    log_manager = get_log_manager()

    valid, error = verify_proxy_api_key()
    if not valid:
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("GET", "/v1/models", 401, duration_ms, None, error)
        return jsonify(error), 401

    payload = {
        "object": "list",
        "data": [_make_model_object(model) for model in config.get_public_model_names()],
    }
    duration_ms = int((time.time() - start_time) * 1000)
    log_manager.log_api_call("GET", "/v1/models", 200, duration_ms, None, payload)
    return jsonify(payload), 200


@proxy_bp.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Handle OpenAI-compatible Chat Completions requests from Kilo Code."""
    start_time = time.time()
    config = get_config()
    log_manager = get_log_manager()

    valid, error = verify_proxy_api_key()
    if not valid:
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 401, duration_ms, None, error)
        return jsonify(error), 401

    try:
        openai_request = request.get_json()
    except Exception as e:
        error = openai_error(f"Invalid JSON: {e}", "invalid_request_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 400, duration_ms, None, error)
        return jsonify(error), 400

    if not isinstance(openai_request, dict) or not openai_request:
        error = openai_error("Request body must be a non-empty JSON object", "invalid_request_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 400, duration_ms, openai_request, error)
        return jsonify(error), 400

    if "messages" not in openai_request:
        error = openai_error("Request body must include messages", "invalid_request_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 400, duration_ms, openai_request, error)
        return jsonify(error), 400

    is_streaming = bool(openai_request.get("stream", False))
    request_fingerprint, accepted = _register_inflight_request(config, openai_request)
    if not accepted:
        error = openai_error(
            "Duplicate request already in progress. Wait for the active response or retry shortly.",
            "invalid_request_error",
            "duplicate_request",
        )
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 409, duration_ms, openai_request, error)
        logger.warning("Rejected duplicate in-flight request")
        return jsonify(error), 409

    release_inflight_on_return = True
    try:
        try:
            sdk_payload, public_model, target_model = _prepare_upstream_request(config, openai_request)
        except ValueError as e:
            error = openai_error(str(e), "invalid_request_error")
            duration_ms = int((time.time() - start_time) * 1000)
            log_manager.log_api_call("POST", "/v1/chat/completions", 400, duration_ms, openai_request, error)
            return jsonify(error), 400

        logger.info(
            "-> %s mapped_to=%s | msgs=%s | stream=%s",
            public_model,
            target_model,
            len(openai_request.get("messages", [])),
            is_streaming,
        )

        if config.use_placeholder_mode:
            if is_streaming:
                release_inflight_on_return = False
                return _handle_placeholder_stream(
                    public_model,
                    openai_request,
                    start_time,
                    log_manager,
                    request_fingerprint,
                )
            response_payload = _placeholder_completion(public_model)
            duration_ms = int((time.time() - start_time) * 1000)
            input_tokens, output_tokens = _extract_usage_tokens(response_payload)
            log_manager.log_api_call(
                "POST",
                "/v1/chat/completions",
                200,
                duration_ms,
                openai_request,
                response_payload,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            return jsonify(response_payload), 200

        if not config.is_upstream_auth_configured():
            error = openai_error(
                "No upstream authentication configured. Set OAuth credentials or TARGET_API_KEY.",
                "authentication_error",
            )
            duration_ms = int((time.time() - start_time) * 1000)
            log_manager.log_api_call("POST", "/v1/chat/completions", 500, duration_ms, openai_request, error)
            return jsonify(error), 500

        if is_streaming:
            release_inflight_on_return = False
            return _handle_streaming(
                sdk_payload,
                public_model,
                openai_request,
                start_time,
                config,
                log_manager,
                request_fingerprint,
            )

        return _handle_non_streaming(
            sdk_payload,
            public_model,
            openai_request,
            start_time,
            config,
            log_manager,
        )
    finally:
        if release_inflight_on_return:
            _release_inflight_request(request_fingerprint)


def _handle_non_streaming(sdk_payload, public_model, openai_request, start_time, config, log_manager):
    """Handle a non-streaming Chat Completions request."""
    client = None
    try:
        client = _build_openai_client(
            config,
            get_oauth_manager(),
            timeout_seconds=config.request_timeout_seconds,
        )
        completion = client.chat.completions.create(**sdk_payload)
        response_payload = _restore_response_model(_object_to_dict(completion), public_model)
    except APIStatusError as e:
        return _handle_openai_status_error(e, openai_request, start_time, log_manager)
    except (APITimeoutError, APIConnectionError) as e:
        error = openai_error(f"Upstream connection error: {e}", "api_connection_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 502, duration_ms, openai_request, error)
        return jsonify(error), 502
    except Exception as e:
        logger.exception("Unexpected proxy error")
        error = openai_error(f"Internal proxy error: {e}", "api_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 500, duration_ms, openai_request, error)
        return jsonify(error), 500
    finally:
        if client is not None:
            _close_client(client)

    duration_ms = int((time.time() - start_time) * 1000)
    input_tokens, output_tokens = _extract_usage_tokens(response_payload)
    cost = config.calculate_cost(public_model, input_tokens, output_tokens)
    log_manager.log_api_call(
        "POST",
        "/v1/chat/completions",
        200,
        duration_ms,
        openai_request,
        response_payload,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )

    logger.info("<- %s tokens=%s+%s", public_model, input_tokens, output_tokens)
    return jsonify(response_payload), 200


def _handle_streaming(
    sdk_payload,
    public_model,
    openai_request,
    start_time,
    config,
    log_manager,
    request_fingerprint,
):
    """Handle a streaming Chat Completions request."""
    client = None
    stream = None

    try:
        client = _build_openai_client(
            config,
            get_oauth_manager(),
            timeout_seconds=config.streaming_timeout_seconds,
        )
        stream = client.chat.completions.create(**sdk_payload)
    except APIStatusError as e:
        _release_inflight_request(request_fingerprint)
        return _handle_openai_status_error(e, openai_request, start_time, log_manager)
    except (APITimeoutError, APIConnectionError) as e:
        _release_inflight_request(request_fingerprint)
        if client is not None:
            _close_client(client)
        error = openai_error(f"Upstream connection error: {e}", "api_connection_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 502, duration_ms, openai_request, error)
        return jsonify(error), 502
    except Exception as e:
        _release_inflight_request(request_fingerprint)
        if client is not None:
            _close_client(client)
        logger.exception("Unexpected streaming setup error")
        error = openai_error(f"Internal proxy error: {e}", "api_error")
        duration_ms = int((time.time() - start_time) * 1000)
        log_manager.log_api_call("POST", "/v1/chat/completions", 500, duration_ms, openai_request, error)
        return jsonify(error), 500

    def generate():
        nonlocal stream, client
        input_tokens = 0
        output_tokens = 0
        status = 200
        response_for_log: dict[str, Any] = {"streaming": True}
        try:
            for chunk in _iter_stream(stream):
                payload = _restore_response_model(_object_to_dict(chunk), public_model)
                chunk_input_tokens, chunk_output_tokens = _extract_usage_tokens(payload)
                input_tokens = chunk_input_tokens or input_tokens
                output_tokens = chunk_output_tokens or output_tokens
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            status = 499
            response_for_log = {"streaming": True, "cancelled": True}
            raise
        except Exception as e:
            status = 500
            logger.exception("Streaming proxy error")
            error_payload = openai_error(str(e), "api_error")
            response_for_log = error_payload
            yield f"data: {json.dumps(error_payload, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if client is not None:
                _close_client(client)
            duration_ms = int((time.time() - start_time) * 1000)
            cost = config.calculate_cost(public_model, input_tokens, output_tokens)
            log_manager.log_api_call(
                "POST",
                "/v1/chat/completions",
                status,
                duration_ms,
                openai_request,
                response_for_log,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
            _release_inflight_request(request_fingerprint)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    ), 200


def _iter_stream(stream: Any) -> Iterable[Any]:
    """Return an iterable over OpenAI SDK stream chunks."""
    if isinstance(stream, Iterable):
        return stream
    raise TypeError("Upstream stream response is not iterable")


def _handle_openai_status_error(e: APIStatusError, openai_request, start_time, log_manager):
    """Forward an OpenAI SDK status error as an OpenAI-compatible payload."""
    status_code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", 502)
    error_payload = None
    response = getattr(e, "response", None)
    if response is not None:
        try:
            error_payload = response.json()
        except Exception:
            error_payload = None
    if not isinstance(error_payload, dict):
        error_payload = openai_error(str(e), "upstream_error")

    duration_ms = int((time.time() - start_time) * 1000)
    log_manager.log_api_call(
        "POST",
        "/v1/chat/completions",
        status_code,
        duration_ms,
        openai_request,
        error_payload,
    )
    return jsonify(error_payload), status_code


def _placeholder_completion(public_model: str) -> dict[str, Any]:
    """Build a small OpenAI-compatible placeholder response."""
    created = int(time.time())
    return {
        "id": f"chatcmpl-placeholder-{created}",
        "object": "chat.completion",
        "created": created,
        "model": public_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Kilo-Launcher placeholder response.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
        },
    }


def _handle_placeholder_stream(public_model, openai_request, start_time, log_manager, request_fingerprint):
    """Handle placeholder streaming response."""
    created = int(time.time())

    def generate():
        try:
            chunks = [
                {
                    "id": f"chatcmpl-placeholder-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": f"chatcmpl-placeholder-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Kilo-Launcher placeholder response."},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": f"chatcmpl-placeholder-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 6,
                        "total_tokens": 18,
                    },
                },
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"

            duration_ms = int((time.time() - start_time) * 1000)
            log_manager.log_api_call(
                "POST",
                "/v1/chat/completions",
                200,
                duration_ms,
                openai_request,
                {"streaming": True, "placeholder": True},
                input_tokens=12,
                output_tokens=6,
            )
        finally:
            _release_inflight_request(request_fingerprint)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    ), 200
