# Kilo/OpenAI Compatibility Check

Checked June 12, 2026.

## Kilo Code Expectations

- Kilo supports custom providers backed by OpenAI-compatible APIs.
- The custom provider dialog asks for provider ID, display name, base URL, API key, optional headers, and models.
- When a valid OpenAI-compatible base URL is entered, Kilo can fetch models from the endpoint.
- Kilo model references use `provider_id/model_id`.
- Custom provider config supports provider-level `options.apiKey`, `options.baseURL`, and `options.timeout`.
- Per-model config should set `tool_call` and `limit.context` / `limit.output`.
- Kilo documents that OpenAI-compatible providers may send `max_tokens` from `limit.output`; GPT-5-style endpoints may expect `max_completion_tokens`.

Implementation coverage:

- `GET /v1/models` returns OpenAI-shaped model objects from `MODEL_OPTIONS`.
- The dashboard shows and copies the generated Kilo spec for provider/model setup.
- The dashboard shows Kilo-facing models, upstream mappings, and configured per-1K-token pricing.
- Recent dashboard calls expand to show sanitized request/response details and inline error messages.
- The proxy accepts either `model` or `provider/model` and maps to upstream with `MODEL_MAPPING`.
- `max_tokens` is converted to `max_completion_tokens` by default.
- Missing token limits are filled with `DEFAULT_MAX_COMPLETION_TOKENS`.

## OpenAI-Compatible Endpoint Expectations

- Chat completions are sent to `/v1/chat/completions`.
- Streaming uses server-sent events and terminates with `data: [DONE]`.
- Model listing uses `/v1/models`.
- `max_completion_tokens` is the modern completion limit field; `max_tokens` is deprecated for newer models.

Implementation coverage:

- The proxy uses the OpenAI Python SDK with `base_url=TARGET_ENDPOINT`.
- Upstream auth is the same bearer-token model as `cc-launcher`: OAuth client credentials first, then `TARGET_API_KEY` / `OPENAI_API_KEY`.
- SSL handling keeps the same optional enterprise certificate setup and `SKIP_SSL_VERIFY` behavior.
- Non-streaming and streaming responses stay OpenAI-shaped when returned to Kilo.

## Internal Endpoint Setup

Copy these shared values from `cc-launcher/src/.env` when setting up work:

- `TARGET_ENDPOINT`
- `TARGET_API_KEY` or the `OAUTH_*` values
- `PROXY_PORT` if you want the same local port
- `SKIP_SSL_VERIFY` / certificate settings if needed

Then fill in:

- `MODEL_OPTIONS`
- `MODEL_MAPPING`
- `MODEL_PRICING_USD_PER_1K`
- `DEFAULT_MODEL`
- `DEFAULT_MAX_COMPLETION_TOKENS`
