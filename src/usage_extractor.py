"""Usage token extraction helpers for heterogeneous upstream response formats."""

from __future__ import annotations

from typing import Any, Optional


def extract_usage_tokens(payload: dict[str, Any] | None) -> tuple[int, int]:
    """Extract input/output tokens, defaulting missing values to 0."""
    input_tokens, output_tokens = extract_usage_tokens_optional(payload)
    return input_tokens or 0, output_tokens or 0


def discover_usage_candidates(
    payload: dict[str, Any] | None,
    max_results: int = 20,
    max_depth: int = 8,
) -> list[str]:
    """Discover likely usage/token fields in an arbitrary JSON-like payload."""
    if not isinstance(payload, dict):
        return []

    results: list[str] = []
    seen: set[str] = set()

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > max_depth or len(results) >= max_results:
            return

        if isinstance(node, dict):
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                if _is_usage_key(key):
                    hint = f"{child_path}={_summarize_value(value)}"
                    if hint not in seen:
                        seen.add(hint)
                        results.append(hint)
                        if len(results) >= max_results:
                            return
                walk(value, child_path, depth + 1)
        elif isinstance(node, list):
            for index, value in enumerate(node[:10]):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                walk(value, child_path, depth + 1)

    walk(payload, "", 0)
    return results


def extract_usage_tokens_optional(payload: dict[str, Any] | None) -> tuple[Optional[int], Optional[int]]:
    """
    Extract input/output tokens from a response object or usage object.

    Supports common variants:
    - prompt_tokens/completion_tokens
    - input_tokens/output_tokens
    - prompt_token_count/completion_token_count
    - token_usage/tokenUsage containers
    - total_tokens fallback when only one side is present
    """
    if not isinstance(payload, dict):
        return None, None

    candidates = []
    if isinstance(payload.get("usage"), dict):
        candidates.append(payload["usage"])
    if isinstance(payload.get("token_usage"), dict):
        candidates.append(payload["token_usage"])
    if isinstance(payload.get("tokenUsage"), dict):
        candidates.append(payload["tokenUsage"])
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if isinstance(choice.get("usage"), dict):
                candidates.append(choice["usage"])
            candidates.append(choice)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                if isinstance(delta.get("usage"), dict):
                    candidates.append(delta["usage"])
                candidates.append(delta)
            message = choice.get("message")
            if isinstance(message, dict):
                if isinstance(message.get("usage"), dict):
                    candidates.append(message["usage"])
                candidates.append(message)
    candidates.append(payload)

    for usage in candidates:
        input_tokens = _first_int(
            usage,
            [
                "prompt_tokens",
                "input_tokens",
                "prompt_token_count",
                "input_token_count",
                "promptTokens",
                "inputTokens",
                "promptTokenCount",
                "inputTokenCount",
            ],
        )
        output_tokens = _first_int(
            usage,
            [
                "completion_tokens",
                "output_tokens",
                "completion_token_count",
                "output_token_count",
                "completionTokens",
                "outputTokens",
                "completionTokenCount",
                "outputTokenCount",
            ],
        )
        total_tokens = _first_int(usage, ["total_tokens", "total_token_count", "totalTokens", "totalTokenCount"])

        if output_tokens is None and total_tokens is not None and input_tokens is not None:
            output_tokens = max(total_tokens - input_tokens, 0)
        if input_tokens is None and total_tokens is not None and output_tokens is not None:
            input_tokens = max(total_tokens - output_tokens, 0)

        if input_tokens is not None or output_tokens is not None:
            return input_tokens, output_tokens

    return None, None


def _first_int(data: dict[str, Any], keys: list[str]) -> Optional[int]:
    """Return first parseable non-negative integer value among keys."""
    for key in keys:
        if key in data:
            value = _to_non_negative_int(data.get(key))
            if value is not None:
                return value
    return None


def _to_non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return max(int(float(value)), 0)
        except ValueError:
            return None
    return None


def _is_usage_key(key: Any) -> bool:
    """Heuristic: likely usage field names, while avoiding auth token fields."""
    if not isinstance(key, str):
        return False

    key_lower = key.lower()
    if "usage" in key_lower:
        return True
    if "token" not in key_lower:
        return False
    return any(term in key_lower for term in ["prompt", "completion", "input", "output", "total", "count"])


def _summarize_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f"<str len={len(value)}>"
    if isinstance(value, dict):
        keys = list(value.keys())[:6]
        return f"<dict keys={','.join(str(key) for key in keys)}>"
    if isinstance(value, list):
        return f"<list len={len(value)}>"
    if value is None:
        return "null"
    return f"<{type(value).__name__}>"
