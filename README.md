# Kilo-Launcher

Local OpenAI-compatible proxy + dashboard for routing Kilo Code traffic to an
internal OpenAI-compatible endpoint.

## What It Does

- Exposes `GET /v1/models` using model names from `MODEL_OPTIONS`.
- Exposes `POST /v1/chat/completions` for Kilo Code's OpenAI-compatible provider.
- Maps Kilo-facing model names to upstream/internal model names via `MODEL_MAPPING`.
- Converts Kilo's `max_tokens` request field to upstream `max_completion_tokens` by default.
- Uses the OpenAI Python SDK against `TARGET_ENDPOINT`.
- Supports non-streaming and streaming Chat Completions responses.
- Supports upstream OAuth client-credentials auth or static API-key auth.
- Provides a local dashboard for setup values, logs, usage, model mapping, and per-model pricing.

This intentionally does not include the old Anthropic-to-OpenAI translation
layer. Requests and responses stay OpenAI-compatible end to end, with only the
model field changed before the upstream call and restored before returning to
Kilo.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r src/requirements.txt`
3. Create config:
   - `cp src/.env.example src/.env`
4. Edit `src/.env` with your internal endpoint/auth/model settings.
5. Start:
   - `python src/app.py`
   - or `./run.sh`

## Kilo Code Settings

Use Kilo Code's OpenAI-compatible custom provider settings:

- Provider: `OpenAI Compatible`
- Base URL: `http://localhost:<PROXY_PORT>/v1`
- API key: the `PROXY_ACCESS_TOKEN` printed on startup, or the value you set in `src/.env`
- Models: values from `MODEL_OPTIONS`

The proxy also serves `GET /v1/models`, so Kilo can discover the model names
defined in `MODEL_OPTIONS` when it calls the base URL with the proxy API key.

## Model Mapping

`MODEL_OPTIONS` controls what Kilo sees and selects:

```env
MODEL_OPTIONS=gpt-5.4,gpt-5.4-mini,gpt-5.4-nano,gpt-5.2,gpt-5.1,gpt-5,gpt-5-mini,gpt-5-nano
```

`MODEL_MAPPING` controls what the internal endpoint receives:

```env
MODEL_MAPPING=gpt-5.4=internal-openai-gpt-5.4,gpt-5.4-mini=internal-openai-gpt-5.4-mini,gpt-5.4-nano=internal-openai-gpt-5.4-nano,gpt-5.2=internal-openai-gpt-5.2,gpt-5.1=internal-openai-gpt-5.1,gpt-5=internal-openai-gpt-5,gpt-5-mini=internal-openai-gpt-5-mini,gpt-5-nano=internal-openai-gpt-5-nano
```

If a selected model is not present in `MODEL_MAPPING`, the proxy passes it
through unchanged. Set `STRICT_MODEL_ALLOWLIST=true` to reject requests for
models not listed in `MODEL_OPTIONS`.

## Model Pricing

`MODEL_PRICING_USD_PER_MILLION` controls dashboard pricing display and session
cost tracking. Costs are USD per million tokens:

```env
MODEL_PRICING_USD_PER_MILLION=gpt-5.4=0/0,gpt-5.4-mini=0/0,gpt-5.4-nano=0/0,gpt-5.2=0/0,gpt-5.1=0/0,gpt-5=0/0,gpt-5-mini=0/0,gpt-5-nano=0/0
```

Each value is `input_cost/output_cost`. For example, `gpt-5.4=2.50/10.00`
means $2.50 per million input tokens and $10.00 per million output tokens.

## Token Limit Handling

Kilo Code's OpenAI-compatible provider uses model `limit.output` to set an
output cap, and its docs note that this may be sent as `max_tokens`. The proxy
defaults to converting `max_tokens` into `max_completion_tokens` before sending
requests upstream:

```env
COMPLETION_TOKEN_LIMIT_FIELD=max_completion_tokens
CONVERT_MAX_TOKENS_TO_MAX_COMPLETION_TOKENS=true
INJECT_DEFAULT_MAX_COMPLETION_TOKENS=true
DEFAULT_MAX_COMPLETION_TOKENS=16384
```

If Kilo omits a token limit entirely, the proxy injects
`DEFAULT_MAX_COMPLETION_TOKENS`. This prevents internal GPT-5-style endpoints
from falling back to low defaults such as 400 output tokens.

## Key Env Vars

- `PROXY_PORT`, `BIND_HOST`
- `PROXY_ACCESS_TOKEN`, `DASHBOARD_ACCESS_TOKEN`
- `TARGET_ENDPOINT`
- `OAUTH_TOKEN_ENDPOINT`, `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_SCOPE`
- `TARGET_API_KEY` or `OPENAI_API_KEY`
- `MODEL_OPTIONS`, `MODEL_MAPPING`, `MODEL_PRICING_USD_PER_MILLION`
- `DEFAULT_MODEL`, `STRICT_MODEL_ALLOWLIST`
- `DEFAULT_MAX_COMPLETION_TOKENS`, `INJECT_DEFAULT_MAX_COMPLETION_TOKENS`
- `COMPLETION_TOKEN_LIMIT_FIELD`, `CONVERT_MAX_TOKENS_TO_MAX_COMPLETION_TOKENS`
- `OPENAI_REQUEST_TIMEOUT_SECONDS`, `OPENAI_STREAMING_TIMEOUT_SECONDS`
- `SKIP_SSL_VERIFY`, `DEV_MODE`, `AUTO_OPEN_BROWSER`

## Work Machine Setup

1. Clone this repo.
2. Run `cp src/.env.example src/.env`.
3. Copy over the shared endpoint/auth values from `cc-launcher/src/.env`.
4. Fill in the right-hand side of each `MODEL_MAPPING` entry with the internal model names.
5. Fill in `MODEL_PRICING_USD_PER_MILLION` with input/output costs for dashboard tracking.
6. Run `./run.sh`.
7. In Kilo Code, configure an OpenAI-compatible custom provider:
   - Base URL: `http://localhost:<PROXY_PORT>/v1`
   - API key: `PROXY_ACCESS_TOKEN`
   - Models: load from the proxy or use the `MODEL_OPTIONS` names.

## Validation

Run:

```bash
python3 -m unittest discover -s tests -v
```

The test suite covers local auth, `/v1/models`, model mapping, model pricing,
`max_tokens` to `max_completion_tokens` conversion, default token injection,
non-streaming response rewriting, streaming response rewriting, and duplicate
in-flight request detection.
