import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import create_app
from codex_config_manager import apply_codex_config, restore_codex_config
from config import Config
from handlers import proxy_handler
from logger_manager import LoggerManager
from oauth_manager import OAuthManager


class _FakeCompletion:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, exclude_none=True):
        return dict(self.payload)


class _FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class _FakeOpenAIClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def close(self):
        self.closed = True


class _FakeOAuthTokenResponse:
    status_code = 200
    ok = True
    text = ""

    def __init__(self, access_token="oauth-test-token", expires_in=1):
        self._payload = {"access_token": access_token, "expires_in": expires_in}

    def raise_for_status(self):
        return None

    def json(self):
        return dict(self._payload)


class ProxyTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self._env = mock.patch.dict(
            os.environ,
            {
                "CODEX_HOME": str(self.tmp_path / "codex-home"),
                "CODEX_CONFIG_PATH": str(self.tmp_path / "codex-home" / "config.toml"),
                "CODEX_PROXY_TOKEN_FILE": str(self.tmp_path / "codex-home" / "codex-launcher" / "proxy_token"),
                "CODEX_PROXY_PORT": "5051",
                "CODEX_PROVIDER_ID": "codex-local-proxy",
                "CODEX_DEFAULT_MODEL": "codex-gpt",
                "AUTO_APPLY_CODEX_CONFIG": "false",
                "AUTO_RESTART_CODEX_DESKTOP": "false",
                "PROXY_ACCESS_TOKEN": "proxy-test-token",
                "DASHBOARD_ACCESS_TOKEN": "dashboard-test-token",
                "TARGET_API_KEY": "upstream-test-key",
                "TARGET_ENDPOINT": "https://internal.example.test/v1",
                "MODEL_OPTIONS": "codex-gpt,codex-mini",
                "MODEL_MAPPING": "codex-gpt=internal-gpt,codex-mini=internal-mini",
                "MODEL_PRICING_USD_PER_1K": "codex-gpt=1/2,codex-mini=0.1/0.2",
                "MODEL_PRICING_USD_PER_MILLION": "",
                "DEFAULT_MODEL": "codex-gpt",
                "DEFAULT_MAX_COMPLETION_TOKENS": "2048",
                "USE_PLACEHOLDER_MODE": "false",
                "DEV_MODE": "false",
                "AUTO_OPEN_BROWSER": "false",
                "BIND_HOST": "127.0.0.1",
            },
            clear=False,
        )
        self._env.start()
        self.app = create_app()
        self.client = self.app.test_client()
        self.config = self.app.config["KL_CONFIG"]
        with proxy_handler._inflight_lock:
            proxy_handler._inflight_requests.clear()

    def tearDown(self):
        with proxy_handler._inflight_lock:
            proxy_handler._inflight_requests.clear()
        with proxy_handler._response_store_lock:
            proxy_handler._response_store.clear()
            proxy_handler._response_store_order.clear()
        self._env.stop()
        self._tmp.cleanup()

    def _headers(self):
        return {"Authorization": "Bearer proxy-test-token"}

    def test_models_endpoint_requires_auth_and_returns_env_options(self):
        unauth_response = self.client.get("/v1/models")
        self.assertEqual(unauth_response.status_code, 401)

        response = self.client.get("/v1/models", headers=self._headers())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["object"], "list")
        self.assertEqual([item["id"] for item in payload["data"]], ["codex-gpt", "codex-mini"])

    def test_chat_completion_maps_model_and_restores_response_model(self):
        fake_payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": "internal-gpt",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
        fake_client = _FakeOpenAIClient(_FakeCompletion(fake_payload))

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["model"], "codex-gpt")
        self.assertEqual(fake_client.calls[0]["model"], "internal-gpt")
        self.assertEqual(fake_client.calls[0]["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", fake_client.calls[0])
        self.assertTrue(fake_client.closed)

        usage = self.app.config["LOG_MANAGER"].get_usage_stats()
        self.assertEqual(usage["total_input_tokens"], 7)
        self.assertEqual(usage["total_output_tokens"], 3)
        self.assertEqual(usage["total_cost_usd"], 0.013)

    def test_model_pricing_table_and_cost_calculation(self):
        self.assertEqual(
            self.config.model_pricing,
            {
                "codex-gpt": {"input": 1.0, "output": 2.0},
                "codex-mini": {"input": 0.1, "output": 0.2},
            },
        )
        self.assertEqual(self.config.calculate_cost("codex-gpt", 1000, 2000), 5.0)

        table = self.config.get_model_pricing_table()
        self.assertEqual(
            table[0],
            {
                "model": "codex-gpt",
                "target_model": "internal-gpt",
                "input_cost_per_1k": 1.0,
                "output_cost_per_1k": 2.0,
                "configured": True,
            },
        )

    def test_legacy_per_million_pricing_is_converted_to_per_1k(self):
        with mock.patch.dict(
            os.environ,
            {
                "MODEL_OPTIONS": "codex-gpt",
                "MODEL_MAPPING": "codex-gpt=internal-gpt",
                "MODEL_PRICING_USD_PER_1K": "",
                "MODEL_PRICING_USD_PER_MILLION": "codex-gpt=1000/2000",
            },
            clear=False,
        ):
            config = Config()

        self.assertEqual(config.model_pricing, {"codex-gpt": {"input": 1.0, "output": 2.0}})
        self.assertAlmostEqual(config.calculate_cost("codex-gpt", 7, 3), 0.013)

    def test_codex_chat_max_tokens_is_converted_to_max_completion_tokens(self):
        fake_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "internal-gpt",
                    "choices": [],
                    "usage": {},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8192,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_client.calls[0]["max_completion_tokens"], 8192)
        self.assertNotIn("max_tokens", fake_client.calls[0])

    def test_unknown_openai_fields_are_forwarded_with_extra_body(self):
        fake_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "internal-gpt",
                    "choices": [],
                    "usage": {},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "messages": [{"role": "user", "content": "hello"}],
                    "internal_hint": {"route": "fast"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_client.calls[0]["extra_body"], {"internal_hint": {"route": "fast"}})

    def test_strict_model_allowlist_rejects_unconfigured_models(self):
        with mock.patch.dict(
            os.environ,
            {
                "MODEL_OPTIONS": "codex-gpt",
                "MODEL_MAPPING": "codex-gpt=internal-gpt",
                "STRICT_MODEL_ALLOWLIST": "true",
            },
            clear=False,
        ):
            config = Config()

        with self.assertRaises(ValueError):
            config.resolve_target_model("not-configured")

    def test_default_model_options_include_requested_openai_models(self):
        with mock.patch.dict(
            os.environ,
            {
                "MODEL_OPTIONS": "",
                "OPENAI_MODEL_OPTIONS": "",
                "CODEX_MODEL_OPTIONS": "",
                "MODEL_MAPPING": "",
                "MODEL_PRICING_USD_PER_1K": "",
                "MODEL_PRICING_USD_PER_MILLION": "",
                "DEFAULT_MODEL": "",
            },
            clear=False,
        ):
            config = Config()

        self.assertEqual(
            config.get_public_model_names(),
            [
                "gpt-5.5",
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.4-nano",
            ],
        )

    def test_duplicate_guard_blocks_identical_inflight_requests(self):
        body = {
            "model": "codex-gpt",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }

        with self.app.test_request_context(
            "/v1/chat/completions",
            method="POST",
            headers=self._headers(),
            json=body,
        ):
            fp1, ok1 = proxy_handler._register_inflight_request(self.config, body)
            fp2, ok2 = proxy_handler._register_inflight_request(self.config, body)
            proxy_handler._release_inflight_request(fp1)
            fp3, ok3 = proxy_handler._register_inflight_request(self.config, body)
            proxy_handler._release_inflight_request(fp3)

        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertTrue(ok3)
        self.assertEqual(fp1, fp2)
        self.assertEqual(fp1, fp3)

    def test_streaming_response_is_openai_sse_and_restores_model(self):
        stream = _FakeStream(
            [
                _FakeCompletion(
                    {
                        "id": "chunk-1",
                        "object": "chat.completion.chunk",
                        "created": 123,
                        "model": "internal-gpt",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                _FakeCompletion(
                    {
                        "id": "chunk-2",
                        "object": "chat.completion.chunk",
                        "created": 124,
                        "model": "internal-gpt",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    }
                ),
            ]
        )
        fake_client = _FakeOpenAIClient(stream)

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
                buffered=True,
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("data: [DONE]", body)
        lines = [line for line in body.splitlines() if line.startswith("data: {")]
        first_payload = json.loads(lines[0][6:])
        self.assertEqual(first_payload["model"], "codex-gpt")
        self.assertTrue(stream.closed)
        self.assertTrue(fake_client.closed)

        usage = self.app.config["LOG_MANAGER"].get_usage_stats()
        self.assertEqual(usage["total_input_tokens"], 5)
        self.assertEqual(usage["total_output_tokens"], 1)

    def test_responses_endpoint_translates_non_streaming_to_chat_completions(self):
        fake_payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": "internal-gpt",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
        fake_client = _FakeOpenAIClient(_FakeCompletion(fake_payload))

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "instructions": "You are terse.",
                    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
                    "max_output_tokens": 1024,
                    "stream": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["model"], "codex-gpt")
        self.assertEqual(payload["output"][0]["content"][0]["text"], "ok")
        self.assertEqual(payload["usage"]["input_tokens"], 7)
        self.assertEqual(fake_client.calls[0]["model"], "internal-gpt")
        self.assertEqual(fake_client.calls[0]["messages"][0], {"role": "system", "content": "You are terse."})
        self.assertEqual(fake_client.calls[0]["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(fake_client.calls[0]["max_completion_tokens"], 1024)

    def test_responses_streaming_translates_chat_chunks_to_responses_sse(self):
        stream = _FakeStream(
            [
                _FakeCompletion(
                    {
                        "id": "chunk-1",
                        "object": "chat.completion.chunk",
                        "created": 123,
                        "model": "internal-gpt",
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": "he"}, "finish_reason": None}
                        ],
                    }
                ),
                _FakeCompletion(
                    {
                        "id": "chunk-2",
                        "object": "chat.completion.chunk",
                        "created": 124,
                        "model": "internal-gpt",
                        "choices": [
                            {"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    }
                ),
            ]
        )
        fake_client = _FakeOpenAIClient(stream)

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={"model": "codex-gpt", "input": "hello", "stream": True},
                buffered=True,
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("event: response.created", body)
        self.assertIn("event: response.output_text.delta", body)
        self.assertIn("event: response.completed", body)
        self.assertIn('"text":"hello"', body)

    def test_responses_function_call_and_previous_response_tool_output(self):
        first_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-tool",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "internal-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {"name": "run_command", "arguments": "{\"cmd\":\"pwd\"}"},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=first_client):
            first = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "input": "run pwd",
                    "tools": [
                        {
                            "type": "function",
                            "name": "run_command",
                            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                        }
                    ],
                },
            )

        self.assertEqual(first.status_code, 200)
        first_payload = first.get_json()
        self.assertEqual(first_payload["output"][0]["type"], "function_call")
        response_id = first_payload["id"]

        second_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-final",
                    "object": "chat.completion",
                    "created": 124,
                    "model": "internal-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "/tmp/project"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=second_client):
            second = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "previous_response_id": response_id,
                    "input": [{"type": "function_call_output", "call_id": "call_123", "output": "/tmp/project"}],
                },
            )

        self.assertEqual(second.status_code, 200)
        messages = second_client.calls[0]["messages"]
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "call_123", "content": "/tmp/project"})

    def test_responses_chat_adapter_rejects_hosted_tools(self):
        response = self.client.post(
            "/v1/responses",
            headers=self._headers(),
            json={
                "model": "codex-gpt",
                "input": "search",
                "tools": [{"type": "web_search_preview"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "unsupported_tool")

    def test_responses_chat_adapter_accepts_codex_tool_declarations(self):
        fake_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-tools",
                    "object": "chat.completion",
                    "created": 124,
                    "model": "internal-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "input": "use tools if needed",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "code_exec",
                            "description": "Run code as freeform text.",
                        },
                        {
                            "type": "namespace",
                            "name": "workspace",
                            "description": "Workspace tools.",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "read_file",
                                    "description": "Read a file.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"path": {"type": "string"}},
                                        "required": ["path"],
                                        "additionalProperties": False,
                                    },
                                },
                                {
                                    "type": "function",
                                    "name": "write_file",
                                    "description": "Write a file.",
                                    "defer_loading": True,
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                        "required": ["path", "content"],
                                        "additionalProperties": False,
                                    },
                                },
                            ],
                        },
                        {"type": "tool_search"},
                        {"type": "web_search"},
                    ],
                    "tool_choice": {"type": "custom", "name": "code_exec"},
                },
            )

        self.assertEqual(response.status_code, 200)
        chat_tools = fake_client.calls[0]["tools"]
        self.assertEqual([tool["function"]["name"] for tool in chat_tools], ["code_exec", "read_file", "write_file"])
        self.assertEqual(
            fake_client.calls[0]["tool_choice"],
            {"type": "function", "function": {"name": "code_exec"}},
        )
        custom_parameters = chat_tools[0]["function"]["parameters"]
        self.assertEqual(custom_parameters["properties"]["input"]["type"], "string")
        self.assertTrue(chat_tools[1]["function"]["description"].startswith("workspace namespace:"))

    def test_responses_chat_adapter_preserves_custom_and_namespace_tool_calls(self):
        fake_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-tool-call",
                    "object": "chat.completion",
                    "created": 124,
                    "model": "internal-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_custom",
                                        "type": "function",
                                        "function": {
                                            "name": "code_exec",
                                            "arguments": "{\"input\":\"print(1)\"}",
                                        },
                                    },
                                    {
                                        "id": "call_namespace",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": "{\"path\":\"README.md\"}",
                                        },
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=fake_client):
            response = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "input": "use tools",
                    "tools": [
                        {"type": "custom", "name": "code_exec"},
                        {
                            "type": "namespace",
                            "name": "workspace",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "read_file",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"path": {"type": "string"}},
                                    },
                                }
                            ],
                        },
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        output = response.get_json()["output"]
        self.assertEqual(output[0]["type"], "custom_tool_call")
        self.assertEqual(output[0]["name"], "code_exec")
        self.assertEqual(output[0]["input"], "print(1)")
        self.assertEqual(output[1]["type"], "function_call")
        self.assertEqual(output[1]["name"], "read_file")
        self.assertEqual(output[1]["namespace"], "workspace")

    def test_responses_chat_adapter_rejects_file_and_image_inputs(self):
        response = self.client.post(
            "/v1/responses",
            headers=self._headers(),
            json={
                "model": "codex-gpt",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "summarize this"},
                            {"type": "input_file", "file_id": "file_123"},
                            {"type": "input_image", "image_url": "https://example.test/image.png"},
                        ],
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "unsupported_input")

    def test_previous_response_id_does_not_carry_prior_instructions(self):
        first_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-first",
                    "object": "chat.completion",
                    "created": 123,
                    "model": "internal-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "first"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=first_client):
            first = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "instructions": "Old instructions",
                    "input": "hello",
                },
            )

        self.assertEqual(first.status_code, 200)
        response_id = first.get_json()["id"]

        second_client = _FakeOpenAIClient(
            _FakeCompletion(
                {
                    "id": "chatcmpl-second",
                    "object": "chat.completion",
                    "created": 124,
                    "model": "internal-gpt",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "second"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
                }
            )
        )

        with mock.patch.object(proxy_handler, "OpenAI", return_value=second_client):
            second = self.client.post(
                "/v1/responses",
                headers=self._headers(),
                json={
                    "model": "codex-gpt",
                    "previous_response_id": response_id,
                    "instructions": "New instructions",
                    "input": "continue",
                },
            )

        self.assertEqual(second.status_code, 200)
        messages = second_client.calls[0]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "New instructions"})
        self.assertNotIn({"role": "system", "content": "Old instructions"}, messages)

    def test_logger_tracks_usage_even_for_non_2xx_status(self):
        manager = LoggerManager()
        manager.log_api_call(
            method="POST",
            path="/v1/chat/completions",
            status=499,
            duration_ms=1000,
            input_tokens=10,
            output_tokens=4,
            cost_usd=0.1234,
        )
        stats = manager.get_usage_stats()
        self.assertEqual(stats["failed_requests"], 1)
        self.assertEqual(stats["total_input_tokens"], 10)
        self.assertEqual(stats["total_output_tokens"], 4)
        self.assertEqual(stats["total_tokens"], 14)
        self.assertEqual(stats["total_cost_usd"], 0.1234)

    def test_logger_extracts_dashboard_error_message(self):
        manager = LoggerManager()
        manager.log_api_call(
            method="POST",
            path="/v1/chat/completions",
            status=502,
            duration_ms=1000,
            response_data={
                "error": {
                    "message": "Upstream rejected the request",
                    "type": "upstream_error",
                }
            },
        )

        calls = manager.get_api_calls()
        self.assertEqual(calls[0]["error_message"], "Upstream rejected the request")

    def test_oauth_refresh_scheduler_enforces_minimum_delay(self):
        timer_intervals = []

        class _FakeTimer:
            def __init__(self, interval, callback):
                timer_intervals.append(interval)
                self.interval = interval
                self.callback = callback
                self.daemon = False
                self.started = False

            def start(self):
                self.started = True

            def cancel(self):
                return None

        with mock.patch.object(
            sys.modules["oauth_manager"].requests,
            "post",
            return_value=_FakeOAuthTokenResponse(expires_in=1),
        ):
            with mock.patch.object(sys.modules["oauth_manager"].threading, "Timer", _FakeTimer):
                manager = OAuthManager(
                    token_endpoint="https://example.invalid/oauth/token",
                    client_id="test-client",
                    client_secret="test-secret",
                    refresh_buffer_minutes=5,
                )
                token = manager.get_token()
                self.assertEqual(token, "oauth-test-token")
                self.assertTrue(timer_intervals)
                self.assertGreaterEqual(timer_intervals[-1], manager.min_refresh_delay_seconds)
                manager.destroy()

    def test_dashboard_root_auto_authenticates_session_without_token_prompt(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        html = root.get_data(as_text=True)
        self.assertNotIn(self.config.proxy_access_token, html)
        self.assertNotIn(self.config.dashboard_access_token, html)

        models_response = self.client.get("/api/models")
        self.assertEqual(models_response.status_code, 200)

        config_response = self.client.get("/api/config")
        self.assertEqual(config_response.status_code, 200)
        config_payload = config_response.get_json()
        self.assertIn("model_pricing_table", config_payload)
        self.assertEqual(config_payload["model_pricing_table"][0]["model"], "codex-gpt")
        self.assertIn("codexConfig", config_payload)
        self.assertIn("model_providers", config_payload["codexConfig"])

    def test_codex_config_apply_and_restore_preserves_unrelated_settings(self):
        config_path = Path(os.environ["CODEX_CONFIG_PATH"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            """
model = "original-model"
model_provider = "openai"
approval_policy = "on-request"

[model_providers.existing]
name = "Existing"
base_url = "https://existing.example/v1"

[model_providers.codex-local-proxy]
name = "Old Proxy"
base_url = "http://old.example/v1"
wire_api = "responses"
""".lstrip(),
            encoding="utf-8",
        )

        status = apply_codex_config(self.config)
        self.assertTrue(status["is_applied"])
        applied = config_path.read_text(encoding="utf-8")
        self.assertIn('model = "codex-gpt"', applied)
        self.assertIn('model_provider = "codex-local-proxy"', applied)
        self.assertIn('approval_policy = "on-request"', applied)
        self.assertIn('[model_providers.existing]', applied)
        self.assertIn('args = ["' + str(self.config.proxy_token_file) + '"]', applied)

        restored = restore_codex_config(self.config)
        self.assertFalse(restored["is_applied"])
        restored_text = config_path.read_text(encoding="utf-8")
        self.assertIn('model = "original-model"', restored_text)
        self.assertIn('model_provider = "openai"', restored_text)
        self.assertIn('[model_providers.existing]', restored_text)
        self.assertIn('name = "Old Proxy"', restored_text)


if __name__ == "__main__":
    unittest.main()
