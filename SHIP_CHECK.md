# Codex Local Proxy Compatibility Check

Checked June 17, 2026.

## Codex Desktop Expectations

- User-level `~/.codex/config.toml` is the source of truth for custom provider and auth configuration.
- Custom providers must use `wire_api = "responses"` in current Codex Desktop; `wire_api = "chat"` is rejected.
- Provider auth can be supplied by a command, which lets Codex read a local token without relying on shell environment variables.
- Codex Desktop should call the configured provider at `<base_url>/responses` and discover models from `<base_url>/models`.

Implementation coverage:

- `GET /v1/models` returns OpenAI-shaped model objects from `CODEX_MODEL_OPTIONS`.
- `POST /v1/responses` is the primary Codex-facing generation endpoint.
- The managed config writes `model`, `model_provider`, `[model_providers.<id>]`, and `[model_providers.<id>.auth]`.
- The proxy token is stored at `~/.codex/codex-launcher/proxy_token` with `0600` permissions.
- Applying config writes a timestamped full backup and a state file for restore.
- Restoring config preserves unrelated `config.toml` content and reinstates the previous model/provider/provider table.

## OpenAI-Compatible Endpoint Expectations

- Responses requests can be passed through to `/v1/responses` when the upstream supports it.
- Responses requests can be translated to `/v1/chat/completions` for upstreams that only support Chat Completions.
- Streaming uses server-sent events.
- Model listing uses `/v1/models`.
- Chat Completions upstreams may require `max_completion_tokens` rather than legacy `max_tokens`.

Implementation coverage:

- `CODEX_UPSTREAM_WIRE_API=chat_completions` translates Responses requests into Chat Completions requests.
- `CODEX_UPSTREAM_WIRE_API=responses` forwards Responses requests with local auth stripped and upstream auth applied.
- Text input, `instructions`, function tools, function-call outputs, `tool_choice`, `max_output_tokens`, usage, and streaming deltas are covered by tests.
- Hosted built-in Responses tools are rejected with an OpenAI-compatible error in Chat Completions adapter mode.
- The debug `/v1/chat/completions` endpoint preserves model mapping, token-limit normalization, streaming, and duplicate-request guard behavior.

## Internal Endpoint Setup

Configure these values in `src/.env`:

- `CODEX_TARGET_ENDPOINT`
- `CODEX_TARGET_API_KEY` or the `OAUTH_*` values
- `CODEX_UPSTREAM_WIRE_API`
- `CODEX_MODEL_OPTIONS`
- `MODEL_MAPPING`
- `MODEL_PRICING_USD_PER_1K`
- `CODEX_DEFAULT_MODEL`
- `DEFAULT_MAX_COMPLETION_TOKENS`

To chain through an existing local proxy, set `CODEX_TARGET_ENDPOINT` to that proxy's `/v1` base URL, keep `CODEX_UPSTREAM_WIRE_API=chat_completions`, and set `CODEX_TARGET_API_KEY` to the upstream proxy token.

## Validation

- `python3 -m unittest discover -s tests -v`
- `./run-dev.sh` with placeholder mode
- `codex doctor --strict-config`
- `codex exec --model gpt-5.5 "reply with ok"`
- Launch Codex Desktop and confirm dashboard logs show `POST /v1/responses`
