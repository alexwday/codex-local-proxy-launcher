import json
import os
import sys
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
        self._env = mock.patch.dict(
            os.environ,
            {
                "PROXY_ACCESS_TOKEN": "proxy-test-token",
                "DASHBOARD_ACCESS_TOKEN": "dashboard-test-token",
                "TARGET_API_KEY": "upstream-test-key",
                "TARGET_ENDPOINT": "https://internal.example.test/v1",
                "MODEL_OPTIONS": "kilo-gpt,kilo-mini",
                "MODEL_MAPPING": "kilo-gpt=internal-gpt,kilo-mini=internal-mini",
                "MODEL_PRICING_USD_PER_MILLION": "kilo-gpt=1000/2000,kilo-mini=100/200",
                "DEFAULT_MODEL": "kilo-gpt",
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
        self._env.stop()

    def _headers(self):
        return {"Authorization": "Bearer proxy-test-token"}

    def test_models_endpoint_requires_auth_and_returns_env_options(self):
        unauth_response = self.client.get("/v1/models")
        self.assertEqual(unauth_response.status_code, 401)

        response = self.client.get("/v1/models", headers=self._headers())
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["object"], "list")
        self.assertEqual([item["id"] for item in payload["data"]], ["kilo-gpt", "kilo-mini"])

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
                    "model": "kilo-gpt",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["model"], "kilo-gpt")
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
                "kilo-gpt": {"input": 1000.0, "output": 2000.0},
                "kilo-mini": {"input": 100.0, "output": 200.0},
            },
        )
        self.assertEqual(self.config.calculate_cost("kilo-gpt", 1000, 2000), 5.0)

        table = self.config.get_model_pricing_table()
        self.assertEqual(
            table[0],
            {
                "model": "kilo-gpt",
                "target_model": "internal-gpt",
                "input_cost_per_million": 1000.0,
                "output_cost_per_million": 2000.0,
                "configured": True,
            },
        )

    def test_kilo_max_tokens_is_converted_to_max_completion_tokens(self):
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
                    "model": "kilo-gpt",
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
                    "model": "kilo-gpt",
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
                "MODEL_OPTIONS": "kilo-gpt",
                "MODEL_MAPPING": "kilo-gpt=internal-gpt",
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
                "KILO_MODEL_OPTIONS": "",
                "MODEL_MAPPING": "",
                "MODEL_PRICING_USD_PER_MILLION": "",
                "DEFAULT_MODEL": "",
            },
            clear=False,
        ):
            config = Config()

        self.assertEqual(
            config.get_public_model_names(),
            [
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.4-nano",
                "gpt-5.2",
                "gpt-5.1",
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
            ],
        )
        self.assertTrue(config.model_supports_reasoning)
        self.assertFalse(config.model_supports_temperature)

    def test_duplicate_guard_blocks_identical_inflight_requests(self):
        body = {
            "model": "kilo-gpt",
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
                    "model": "kilo-gpt",
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
        self.assertEqual(first_payload["model"], "kilo-gpt")
        self.assertTrue(stream.closed)
        self.assertTrue(fake_client.closed)

        usage = self.app.config["LOG_MANAGER"].get_usage_stats()
        self.assertEqual(usage["total_input_tokens"], 5)
        self.assertEqual(usage["total_output_tokens"], 1)

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
        self.assertEqual(config_payload["model_pricing_table"][0]["model"], "kilo-gpt")
        self.assertNotIn("kiloConfigJson", config_payload)


if __name__ == "__main__":
    unittest.main()
