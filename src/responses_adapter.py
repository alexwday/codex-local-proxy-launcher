"""Translate between Codex-facing Responses API and Chat Completions upstreams."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any


CHAT_MAPPABLE_RESPONSE_TOOL_TYPES = {"function", "custom"}
CHAT_IGNORED_RESPONSE_TOOL_TYPES = {"tool_search", "web_search"}
HOSTED_RESPONSE_TOOL_TYPES_REQUIRING_NATIVE = {
    "code_interpreter",
    "computer_use_preview",
    "file_search",
    "image_generation",
    "mcp",
    "shell",
    "web_search_preview",
    "web_search_preview_2025_03_11",
}
SUPPORTED_CHAT_UPSTREAM_CONTENT_TYPES = {"input_text", "output_text", "text"}
DEFAULT_FUNCTION_PARAMETERS = {"type": "object", "properties": {}}
DEFAULT_CUSTOM_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "Freeform input for the custom tool.",
        }
    },
    "required": ["input"],
    "additionalProperties": False,
}


def make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def make_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def make_function_call_id() -> str:
    return f"fc_{uuid.uuid4().hex}"


def make_custom_tool_call_id() -> str:
    return f"ctc_{uuid.uuid4().hex}"


def unsupported_tool_types(request_payload: dict[str, Any]) -> list[str]:
    unsupported = []
    for tool in request_payload.get("tools") or []:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if tool_type in HOSTED_RESPONSE_TOOL_TYPES_REQUIRING_NATIVE:
            unsupported.append(tool_type)
    return unsupported


def unsupported_input_content_types(input_value: Any) -> list[str]:
    """Return Responses input content block types this chat adapter cannot preserve."""
    unsupported: list[str] = []

    def inspect_content(content: Any) -> None:
        if isinstance(content, dict):
            block_type = content.get("type")
            if block_type and block_type not in SUPPORTED_CHAT_UPSTREAM_CONTENT_TYPES:
                unsupported.append(str(block_type))
            return
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type and block_type not in SUPPORTED_CHAT_UPSTREAM_CONTENT_TYPES:
                unsupported.append(str(block_type))

    def inspect_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        item_type = item.get("type")
        if item_type == "message" or "role" in item:
            inspect_content(item.get("content"))
        elif item_type and item_type not in {
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
            "reasoning",
            "web_search_call",
            "file_search_call",
            "tool_search_call",
            "tool_search_output",
            "additional_tools",
        }:
            unsupported.append(str(item_type))

    if isinstance(input_value, list):
        for item in input_value:
            inspect_item(item)
    else:
        inspect_item(input_value)

    return unsupported


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _jsonish(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            parts.append(_jsonish(block))
            continue
        block_type = block.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            parts.append(str(block.get("text", "")))
        elif block_type == "input_image":
            parts.append("[image input omitted by local chat-completions adapter]")
        elif "text" in block:
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _response_item_to_chat_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    item_type = item.get("type")
    if item_type == "message" or "role" in item:
        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        return [{"role": role, "content": _content_to_text(item.get("content"))}]

    if item_type in {"function_call_output", "custom_tool_call_output"}:
        output = item["output"] if "output" in item else item.get("content")
        return [
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "call_unknown",
                "content": _jsonish(output),
            }
        ]

    if item_type in {"function_call", "custom_tool_call"}:
        call_id = item.get("call_id") or item.get("id") or "call_unknown"
        arguments = item.get("arguments")
        if item_type == "custom_tool_call" and arguments is None:
            arguments = json.dumps({"input": item.get("input", "")}, ensure_ascii=False)
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "unknown_function",
                            "arguments": arguments or "{}",
                        },
                    }
                ],
            }
        ]

    if item_type in {
        "reasoning",
        "web_search_call",
        "file_search_call",
        "tool_search_call",
        "tool_search_output",
        "additional_tools",
    }:
        return []

    return [{"role": "user", "content": _content_to_text(item.get("content", item))}]


def responses_input_to_chat_messages(input_value: Any) -> list[dict[str, Any]]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if isinstance(input_value, dict):
        return _response_item_to_chat_messages(input_value)
    if not isinstance(input_value, list):
        return [{"role": "user", "content": _jsonish(input_value)}]

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if isinstance(item, dict):
            messages.extend(_response_item_to_chat_messages(item))
        else:
            messages.append({"role": "user", "content": _jsonish(item)})
    return messages


def responses_tools_to_chat_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    chat_tools: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for tool, namespace in _iter_chat_mappable_response_tools(tools):
        function = _response_tool_to_chat_function(tool, namespace)
        if not function:
            continue
        name = function["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        chat_tools.append({"type": "function", "function": function})
    return chat_tools


def responses_tool_choice_to_chat(tool_choice: Any, available_tool_names: set[str] | None = None) -> Any:
    if tool_choice is None or (isinstance(tool_choice, str) and tool_choice in {"auto", "none", "required"}):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") in CHAT_MAPPABLE_RESPONSE_TOOL_TYPES:
        name = tool_choice.get("name") or tool_choice.get("function", {}).get("name")
        if name:
            chat_name = _chat_tool_name(name)
            if chat_name and (available_tool_names is None or chat_name in available_tool_names):
                return {"type": "function", "function": {"name": chat_name}}
    return "auto"


def _iter_chat_mappable_response_tools(
    tools: list[dict[str, Any]] | None,
) -> list[tuple[dict[str, Any], str | None]]:
    mappable: list[tuple[dict[str, Any], str | None]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "namespace":
            namespace = str(tool.get("name") or "").strip() or None
            for nested_tool in tool.get("tools") or []:
                if not isinstance(nested_tool, dict):
                    continue
                nested_type = nested_tool.get("type")
                if nested_type in CHAT_MAPPABLE_RESPONSE_TOOL_TYPES:
                    mappable.append((nested_tool, namespace))
            continue
        if tool_type in CHAT_IGNORED_RESPONSE_TOOL_TYPES or tool_type in HOSTED_RESPONSE_TOOL_TYPES_REQUIRING_NATIVE:
            continue
        if tool_type in CHAT_MAPPABLE_RESPONSE_TOOL_TYPES:
            mappable.append((tool, None))
    return mappable


def _chat_tool_name(name: Any) -> str | None:
    if not name:
        return None
    chat_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(name).strip())
    chat_name = chat_name.strip("_")[:64]
    return chat_name or None


def _response_tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    for key in ("parameters", "input_schema", "schema", "json_schema"):
        parameters = tool.get(key)
        if isinstance(parameters, dict):
            return parameters
    if tool.get("type") == "custom":
        return dict(DEFAULT_CUSTOM_TOOL_PARAMETERS)
    return dict(DEFAULT_FUNCTION_PARAMETERS)


def _response_tool_to_chat_function(tool: dict[str, Any], namespace: str | None) -> dict[str, Any] | None:
    name = _chat_tool_name(tool.get("name"))
    if not name:
        return None

    function: dict[str, Any] = {
        "name": name,
        "parameters": _response_tool_parameters(tool),
    }
    description = tool.get("description")
    if namespace:
        prefix = f"{namespace} namespace"
        description = f"{prefix}: {description}" if description else prefix
    if description:
        function["description"] = str(description)
    if "strict" in tool:
        function["strict"] = bool(tool["strict"])
    return function


def response_tool_metadata_by_chat_name(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for tool, namespace in _iter_chat_mappable_response_tools(tools):
        chat_name = _chat_tool_name(tool.get("name"))
        if not chat_name or chat_name in metadata:
            continue
        item = {
            "type": str(tool.get("type") or "function"),
            "name": str(tool.get("name") or chat_name),
        }
        if namespace:
            item["namespace"] = namespace
        metadata[chat_name] = item
    return metadata


def _custom_tool_input_from_chat_arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    if not isinstance(arguments, str):
        return _jsonish(arguments)
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments
    if isinstance(parsed, dict):
        if "input" in parsed:
            return str(parsed["input"])
        if len(parsed) == 1:
            return str(next(iter(parsed.values())))
    return _jsonish(parsed)


def _tool_call_id(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    call_id = tool_call.get("id")
    return str(call_id) if call_id else None


def normalize_chat_messages_for_upstream(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make Responses-style histories valid for Chat Completions tool rules."""
    normalized: list[dict[str, Any]] = []
    moved_tool_message_indexes: set[int] = set()

    for index, message in enumerate(messages):
        if index in moved_tool_message_indexes:
            continue

        if message.get("role") == "tool":
            continue

        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
            normalized.append(message)
            continue

        matched_outputs: dict[str, tuple[int, dict[str, Any]]] = {}
        for output_index in range(index + 1, len(messages)):
            if output_index in moved_tool_message_indexes:
                continue
            candidate = messages[output_index]
            if candidate.get("role") != "tool":
                continue
            output_call_id = candidate.get("tool_call_id")
            if output_call_id and output_call_id not in matched_outputs:
                matched_outputs[str(output_call_id)] = (output_index, candidate)

        paired_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if (call_id := _tool_call_id(tool_call)) and call_id in matched_outputs
        ]
        if not paired_tool_calls:
            content = message.get("content")
            if content:
                assistant_without_tools = dict(message)
                assistant_without_tools.pop("tool_calls", None)
                normalized.append(assistant_without_tools)
            continue

        paired_assistant = dict(message)
        paired_assistant["tool_calls"] = paired_tool_calls
        normalized.append(paired_assistant)
        for tool_call in paired_tool_calls:
            call_id = _tool_call_id(tool_call)
            if not call_id:
                continue
            output_index, tool_message = matched_outputs[call_id]
            normalized.append(tool_message)
            moved_tool_message_indexes.add(output_index)

    return normalized


def responses_text_format_to_chat_response_format(text_format: Any) -> dict[str, Any] | None:
    if not isinstance(text_format, dict):
        return None

    format_type = text_format.get("type")
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type != "json_schema":
        return None

    if isinstance(text_format.get("json_schema"), dict):
        json_schema = dict(text_format["json_schema"])
    else:
        json_schema = {
            key: text_format[key]
            for key in ("name", "description", "schema", "strict")
            if key in text_format
        }

    if "name" not in json_schema:
        json_schema["name"] = "response"
    if not isinstance(json_schema.get("schema"), dict):
        json_schema["schema"] = {"type": "object", "additionalProperties": True}

    return {"type": "json_schema", "json_schema": json_schema}


def build_chat_request_from_responses(
    config,
    request_payload: dict[str, Any],
    previous_messages: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str, str, list[dict[str, Any]]]:
    """Return (chat_request, public_model, target_model, full_chat_messages)."""
    public_model, target_model = config.resolve_target_model(request_payload.get("model"))
    messages: list[dict[str, Any]] = []
    if previous_messages:
        messages.extend(
            message
            for message in previous_messages
            if message.get("role") not in {"system", "developer"}
        )

    instructions = request_payload.get("instructions")
    if instructions:
        messages.insert(0, {"role": "system", "content": str(instructions)})

    messages.extend(responses_input_to_chat_messages(request_payload.get("input")))
    messages = normalize_chat_messages_for_upstream(messages)

    chat_request: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": bool(request_payload.get("stream", False)),
    }

    if request_payload.get("max_output_tokens") is not None:
        chat_request["max_completion_tokens"] = request_payload["max_output_tokens"]
    if request_payload.get("temperature") is not None:
        chat_request["temperature"] = request_payload["temperature"]
    if request_payload.get("top_p") is not None:
        chat_request["top_p"] = request_payload["top_p"]
    if request_payload.get("parallel_tool_calls") is not None:
        chat_request["parallel_tool_calls"] = request_payload["parallel_tool_calls"]
    if isinstance(request_payload.get("reasoning"), dict) and request_payload["reasoning"].get("effort"):
        chat_request["reasoning_effort"] = request_payload["reasoning"]["effort"]

    tools = responses_tools_to_chat_tools(request_payload.get("tools"))
    if tools:
        available_tool_names = {tool["function"]["name"] for tool in tools if tool.get("function", {}).get("name")}
        chat_request["tools"] = tools
        tool_choice = responses_tool_choice_to_chat(request_payload.get("tool_choice"), available_tool_names)
        if tool_choice is not None:
            chat_request["tool_choice"] = tool_choice

    text_config = request_payload.get("text")
    if isinstance(text_config, dict) and isinstance(text_config.get("format"), dict):
        response_format = responses_text_format_to_chat_response_format(text_config["format"])
        if response_format:
            chat_request["response_format"] = response_format

    config.apply_completion_token_limit(chat_request)
    return chat_request, public_model, target_model, messages


def extract_response_usage_tokens(payload: dict[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return int(input_tokens or 0), int(output_tokens or 0)


def chat_usage_to_response_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": int(usage.get("cached_tokens") or 0)},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": int(usage.get("reasoning_tokens") or 0)},
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


def chat_message_to_response_output(
    message: dict[str, Any],
    original_request: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    tool_metadata = response_tool_metadata_by_chat_name((original_request or {}).get("tools"))
    content = message.get("content")
    if content:
        output.append(
            {
                "id": make_message_id(),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content if isinstance(content, str) else _jsonish(content),
                        "annotations": [],
                    }
                ],
            }
        )

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        chat_name = function.get("name") or "unknown_function"
        metadata = tool_metadata.get(chat_name, {})
        response_name = metadata.get("name") or chat_name
        call_id = tool_call.get("id") or make_function_call_id()
        if metadata.get("type") == "custom":
            output.append(
                {
                    "id": make_custom_tool_call_id(),
                    "type": "custom_tool_call",
                    "call_id": call_id,
                    "name": response_name,
                    "input": _custom_tool_input_from_chat_arguments(function.get("arguments")),
                    "status": "completed",
                }
            )
            continue

        item = {
            "id": make_function_call_id(),
            "type": "function_call",
            "call_id": call_id,
            "name": response_name,
            "arguments": function.get("arguments") or "{}",
            "status": "completed",
        }
        if metadata.get("namespace"):
            item["namespace"] = metadata["namespace"]
        output.append(
            item
        )
    return output


def response_payload_from_chat_completion(
    completion_payload: dict[str, Any],
    public_model: str,
    original_request: dict[str, Any],
    response_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (Responses payload, assistant chat message for state storage)."""
    created = int(completion_payload.get("created") or time.time())
    response_id = response_id or make_response_id()
    choice = (completion_payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = chat_message_to_response_output(message, original_request)
    finish_reason = choice.get("finish_reason")
    status = "incomplete" if finish_reason == "length" else "completed"
    usage = chat_usage_to_response_usage(completion_payload.get("usage"))
    response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "completed_at": int(time.time()) if status == "completed" else None,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": original_request.get("instructions"),
        "max_output_tokens": original_request.get("max_output_tokens"),
        "model": public_model,
        "output": output,
        "parallel_tool_calls": original_request.get("parallel_tool_calls", True),
        "previous_response_id": original_request.get("previous_response_id"),
        "reasoning": original_request.get("reasoning") or {"effort": None, "summary": None},
        "store": original_request.get("store", True),
        "temperature": original_request.get("temperature"),
        "text": original_request.get("text") or {"format": {"type": "text"}},
        "tool_choice": original_request.get("tool_choice", "auto"),
        "tools": original_request.get("tools") or [],
        "top_p": original_request.get("top_p"),
        "truncation": original_request.get("truncation", "disabled"),
        "usage": usage,
        "user": original_request.get("user"),
        "metadata": original_request.get("metadata") or {},
    }
    return response, message


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def response_shell(
    response_id: str,
    public_model: str,
    original_request: dict[str, Any],
    created: int,
    *,
    output: list[dict[str, Any]] | None = None,
    status: str = "in_progress",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = status in {"completed", "incomplete", "failed"}
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "completed_at": int(time.time()) if completed and status == "completed" else None,
        "error": None,
        "incomplete_details": None,
        "instructions": original_request.get("instructions"),
        "max_output_tokens": original_request.get("max_output_tokens"),
        "model": public_model,
        "output": output or [],
        "parallel_tool_calls": original_request.get("parallel_tool_calls", True),
        "previous_response_id": original_request.get("previous_response_id"),
        "reasoning": original_request.get("reasoning") or {"effort": None, "summary": None},
        "store": original_request.get("store", True),
        "temperature": original_request.get("temperature"),
        "text": original_request.get("text") or {"format": {"type": "text"}},
        "tool_choice": original_request.get("tool_choice", "auto"),
        "tools": original_request.get("tools") or [],
        "top_p": original_request.get("top_p"),
        "truncation": original_request.get("truncation", "disabled"),
        "usage": usage,
        "user": original_request.get("user"),
        "metadata": original_request.get("metadata") or {},
    }


class ResponseStreamAdapter:
    """Build Responses SSE events from Chat Completions chunks."""

    def __init__(self, public_model: str, original_request: dict[str, Any], response_id: str | None = None):
        self.public_model = public_model
        self.original_request = original_request
        self.response_id = response_id or make_response_id()
        self.created = int(time.time())
        self.tool_metadata = response_tool_metadata_by_chat_name(original_request.get("tools"))
        self.text_item_id: str | None = None
        self.text_started = False
        self.text = ""
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.next_output_index = 0
        self.usage: dict[str, Any] | None = None

    def initial_events(self) -> list[str]:
        response = response_shell(self.response_id, self.public_model, self.original_request, self.created)
        return [
            sse_event("response.created", {"type": "response.created", "response": response}),
            sse_event("response.in_progress", {"type": "response.in_progress", "response": response}),
        ]

    def _ensure_text_started(self) -> list[str]:
        if self.text_started:
            return []
        self.text_started = True
        self.text_item_id = make_message_id()
        item = {
            "id": self.text_item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        part = {"type": "output_text", "text": "", "annotations": []}
        return [
            sse_event(
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": self.next_output_index, "item": item},
            ),
            sse_event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": self.text_item_id,
                    "output_index": self.next_output_index,
                    "content_index": 0,
                    "part": part,
                },
            ),
        ]

    def _ensure_tool_started(self, index: int, delta: dict[str, Any]) -> list[str]:
        if index in self.tool_calls and self.tool_calls[index].get("started"):
            return []
        state = self.tool_calls.get(index)
        if state is None:
            call_id = delta.get("id") or f"call_{uuid.uuid4().hex}"
            output_index = self.next_output_index + len(self.tool_calls) + (1 if self.text_started else 0)
            state = {
                "id": None,
                "output_index": output_index,
                "call_id": call_id,
                "name": "",
                "arguments": "",
                "started": False,
                "response_tool_type": None,
            }
            self.tool_calls[index] = state
        if state.get("started"):
            return []
        if not state.get("name"):
            return []

        metadata = self.tool_metadata.get(state["name"], {})
        is_custom = metadata.get("type") == "custom"
        state["response_tool_type"] = "custom_tool_call" if is_custom else "function_call"
        state["id"] = make_custom_tool_call_id() if is_custom else make_function_call_id()
        state["started"] = True

        if is_custom:
            item = {
                "id": state["id"],
                "type": "custom_tool_call",
                "call_id": state["call_id"],
                "name": metadata.get("name") or state["name"],
                "input": "",
                "status": "in_progress",
            }
        else:
            item = {
                "id": state["id"],
                "type": "function_call",
                "call_id": state["call_id"],
                "name": metadata.get("name") or state["name"],
                "arguments": "",
                "status": "in_progress",
            }
            if metadata.get("namespace"):
                item["namespace"] = metadata["namespace"]

        return [
            sse_event(
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": state["output_index"], "item": item},
            )
        ]

    def _ensure_tool_state(self, index: int, delta: dict[str, Any]) -> dict[str, Any]:
        if index not in self.tool_calls:
            call_id = delta.get("id") or f"call_{uuid.uuid4().hex}"
            self.tool_calls[index] = {
                "id": None,
                "output_index": self.next_output_index + len(self.tool_calls) + (1 if self.text_started else 0),
                "call_id": call_id,
                "name": "",
                "arguments": "",
                "started": False,
                "response_tool_type": None,
            }
        elif delta.get("id") and not self.tool_calls[index].get("call_id"):
            self.tool_calls[index]["call_id"] = delta["id"]
        return self.tool_calls[index]

    def _tool_delta_event(self, state: dict[str, Any], arg_delta: str) -> str | None:
        if not state.get("started"):
            return None
        if state.get("response_tool_type") == "custom_tool_call":
            return None
        return sse_event(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": state["id"],
                "output_index": state["output_index"],
                "delta": arg_delta,
            },
        )

    def _final_tool_events(self, state: dict[str, Any], item: dict[str, Any]) -> list[str]:
        if state.get("response_tool_type") == "custom_tool_call":
            input_text = item.get("input", "")
            return [
                sse_event(
                    "response.custom_tool_call_input.delta",
                    {
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "delta": input_text,
                    },
                ),
                sse_event(
                    "response.custom_tool_call_input.done",
                    {
                        "type": "response.custom_tool_call_input.done",
                        "item_id": state["id"],
                        "output_index": state["output_index"],
                        "input": input_text,
                    },
                ),
                sse_event(
                    "response.output_item.done",
                    {"type": "response.output_item.done", "output_index": state["output_index"], "item": item},
                ),
            ]

        return [
            sse_event(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": state["id"],
                    "output_index": state["output_index"],
                    "arguments": item["arguments"],
                },
            ),
            sse_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": state["output_index"], "item": item},
            ),
        ]

    def _tool_output_item(self, state: dict[str, Any]) -> dict[str, Any]:
        metadata = self.tool_metadata.get(state["name"], {})
        if state.get("response_tool_type") == "custom_tool_call":
            if not state.get("id"):
                state["id"] = make_custom_tool_call_id()
            return {
                "id": state["id"],
                "type": "custom_tool_call",
                "call_id": state["call_id"],
                "name": metadata.get("name") or state["name"] or "unknown_function",
                "input": _custom_tool_input_from_chat_arguments(state["arguments"]),
                "status": "completed",
            }

        if not state.get("id"):
            state["id"] = make_function_call_id()
        item = {
            "id": state["id"],
            "type": "function_call",
            "call_id": state["call_id"],
            "name": metadata.get("name") or state["name"] or "unknown_function",
            "arguments": state["arguments"] or "{}",
            "status": "completed",
        }
        if metadata.get("namespace"):
            item["namespace"] = metadata["namespace"]
        return item

    def _tool_call_for_chat_history(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": state["call_id"],
            "type": "function",
            "function": {
                "name": state["name"] or "unknown_function",
                "arguments": state["arguments"] or "{}",
            },
        }

    def chunk_events(self, chunk_payload: dict[str, Any]) -> list[str]:
        events: list[str] = []
        if isinstance(chunk_payload.get("usage"), dict):
            self.usage = chat_usage_to_response_usage(chunk_payload["usage"])

        for choice in chunk_payload.get("choices") or []:
            delta = choice.get("delta") or {}
            content_delta = delta.get("content")
            if content_delta:
                events.extend(self._ensure_text_started())
                self.text += str(content_delta)
                events.append(
                    sse_event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": self.text_item_id,
                            "output_index": self.next_output_index,
                            "content_index": 0,
                            "delta": str(content_delta),
                        },
                    )
                )

            for tool_delta in delta.get("tool_calls") or []:
                index = int(tool_delta.get("index", 0))
                state = self._ensure_tool_state(index, tool_delta)
                function = tool_delta.get("function") or {}
                if function.get("name"):
                    state["name"] += function["name"]
                events.extend(self._ensure_tool_started(index, tool_delta))
                if function.get("arguments"):
                    arg_delta = str(function["arguments"])
                    state["arguments"] += arg_delta
                    event = self._tool_delta_event(state, arg_delta)
                    if event:
                        events.append(event)
        return events

    def final_events(self) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
        events: list[str] = []
        output: list[dict[str, Any]] = []
        assistant_message: dict[str, Any] = {"role": "assistant", "content": None}

        if self.text_started and self.text_item_id:
            part = {"type": "output_text", "text": self.text, "annotations": []}
            item = {
                "id": self.text_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
            output.append(item)
            assistant_message["content"] = self.text
            events.extend(
                [
                    sse_event(
                        "response.output_text.done",
                        {
                            "type": "response.output_text.done",
                            "item_id": self.text_item_id,
                            "output_index": self.next_output_index,
                            "content_index": 0,
                            "text": self.text,
                        },
                    ),
                    sse_event(
                        "response.content_part.done",
                        {
                            "type": "response.content_part.done",
                            "item_id": self.text_item_id,
                            "output_index": self.next_output_index,
                            "content_index": 0,
                            "part": part,
                        },
                    ),
                    sse_event(
                        "response.output_item.done",
                        {"type": "response.output_item.done", "output_index": self.next_output_index, "item": item},
                    ),
                ]
            )

        tool_calls: list[dict[str, Any]] = []
        for index in sorted(self.tool_calls):
            state = self.tool_calls[index]
            events.extend(self._ensure_tool_started(index, {}))
            item = self._tool_output_item(state)
            output.append(item)
            tool_calls.append(self._tool_call_for_chat_history(state))
            events.extend(self._final_tool_events(state, item))

        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        response = response_shell(
            self.response_id,
            self.public_model,
            self.original_request,
            self.created,
            output=output,
            status="completed",
            usage=self.usage,
        )
        events.append(sse_event("response.completed", {"type": "response.completed", "response": response}))
        return events, response, assistant_message
