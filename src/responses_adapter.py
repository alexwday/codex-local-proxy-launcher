"""Translate between Codex-facing Responses API and Chat Completions upstreams."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


SUPPORTED_CHAT_UPSTREAM_RESPONSE_TOOL_TYPES = {"function"}
SUPPORTED_CHAT_UPSTREAM_CONTENT_TYPES = {"input_text", "output_text", "text"}


def make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def make_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def make_function_call_id() -> str:
    return f"fc_{uuid.uuid4().hex}"


def unsupported_tool_types(request_payload: dict[str, Any]) -> list[str]:
    unsupported = []
    for tool in request_payload.get("tools") or []:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if tool_type and tool_type not in SUPPORTED_CHAT_UPSTREAM_RESPONSE_TOOL_TYPES:
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
            "reasoning",
            "web_search_call",
            "file_search_call",
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

    if item_type == "function_call_output":
        return [
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "call_unknown",
                "content": _jsonish(item.get("output")),
            }
        ]

    if item_type == "function_call":
        call_id = item.get("call_id") or item.get("id") or "call_unknown"
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
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                ],
            }
        ]

    if item_type in {"reasoning", "web_search_call", "file_search_call"}:
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
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function: dict[str, Any] = {
            "name": tool.get("name"),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        if "strict" in tool:
            function["strict"] = bool(tool["strict"])
        chat_tools.append({"type": "function", "function": function})
    return chat_tools


def responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    if tool_choice in {None, "auto", "none", "required"}:
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name") or tool_choice.get("function", {}).get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return "auto"


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
        chat_request["tools"] = tools
        chat_request["tool_choice"] = responses_tool_choice_to_chat(request_payload.get("tool_choice"))

    text_config = request_payload.get("text")
    if isinstance(text_config, dict) and isinstance(text_config.get("format"), dict):
        fmt = text_config["format"]
        if fmt.get("type") in {"json_object", "json_schema"}:
            chat_request["response_format"] = fmt

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


def chat_message_to_response_output(message: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
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
        output.append(
            {
                "id": make_function_call_id(),
                "type": "function_call",
                "call_id": tool_call.get("id") or make_function_call_id(),
                "name": function.get("name") or "unknown_function",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
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
    output = chat_message_to_response_output(message)
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
        if index in self.tool_calls:
            return []
        item_id = make_function_call_id()
        call_id = delta.get("id") or f"call_{uuid.uuid4().hex}"
        function = delta.get("function") or {}
        state = {
            "id": item_id,
            "output_index": self.next_output_index + len(self.tool_calls) + (1 if self.text_started else 0),
            "call_id": call_id,
            "name": function.get("name") or "",
            "arguments": "",
        }
        self.tool_calls[index] = state
        item = {
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": state["name"],
            "arguments": "",
            "status": "in_progress",
        }
        return [
            sse_event(
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": state["output_index"], "item": item},
            )
        ]

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
                events.extend(self._ensure_tool_started(index, tool_delta))
                state = self.tool_calls[index]
                function = tool_delta.get("function") or {}
                if function.get("name"):
                    state["name"] += function["name"]
                if function.get("arguments"):
                    arg_delta = str(function["arguments"])
                    state["arguments"] += arg_delta
                    events.append(
                        sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": state["id"],
                                "output_index": state["output_index"],
                                "delta": arg_delta,
                            },
                        )
                    )
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
            item = {
                "id": state["id"],
                "type": "function_call",
                "call_id": state["call_id"],
                "name": state["name"] or "unknown_function",
                "arguments": state["arguments"] or "{}",
                "status": "completed",
            }
            output.append(item)
            tool_calls.append(
                {
                    "id": state["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                }
            )
            events.extend(
                [
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
            )

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
