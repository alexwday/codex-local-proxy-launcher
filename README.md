# Codex Local Proxy Launcher

Local proxy, Codex Desktop config manager, and dashboard for routing Codex Desktop through a local OpenAI-compatible endpoint.

Work-machine setup is:

1. Copy the existing launcher env that already works to `src/.env`.
2. Append the Codex-only block in [Work Machine Env](#work-machine-env).
3. Run:

```bash
./run.sh
```

On startup the launcher creates a virtualenv, installs dependencies, starts the Flask proxy, health-checks it, writes a reversible user-level Codex config override, launches or restarts `/Applications/Codex.app`, and opens the dashboard when `AUTO_OPEN_BROWSER=true`.

## What It Does

- Exposes `GET /v1/models` and `POST /v1/responses` for Codex Desktop.
- Keeps `POST /v1/chat/completions` as a debug/backward-compatible endpoint.
- Writes `~/.codex/config.toml` with a managed provider using `wire_api = "responses"`.
- Stores the local Codex auth token at `~/.codex/codex-launcher/proxy_token` with `0600` permissions.
- Backs up the full config and saves the original model/provider/provider-table state before applying changes.
- Restores the previous Codex config with `./restore-codex.sh`.
- Shows proxy health, active Codex config, upstream mode, model mapping, pricing, logs, and restore/restart/smoke-test controls in the dashboard.

Codex Desktop currently expects custom providers to use the Responses API. The launcher therefore exposes `/v1/responses` to Codex even when the configured upstream only supports Chat Completions.

## Codex Config

The launcher manages user-level `~/.codex/config.toml`, not project-local `.codex/config.toml`, because provider and auth keys belong in the user config.

Managed config shape:

```toml
model = "gpt-5.5"
model_provider = "codex-local-proxy"

[model_providers.codex-local-proxy]
name = "Codex Local Proxy"
base_url = "http://127.0.0.1:5051/v1"
wire_api = "responses"
supports_websockets = false
stream_idle_timeout_ms = 900000

[model_providers.codex-local-proxy.auth]
command = "/bin/cat"
args = ["/Users/alexwday/.codex/codex-launcher/proxy_token"]
refresh_interval_ms = 0
```

## Upstream Mode

Default mode translates Codex Desktop's Responses requests to the same direct Chat Completions upstream used by the existing launcher:

```env
CODEX_UPSTREAM_WIRE_API=chat_completions
TARGET_ENDPOINT=<existing direct upstream /v1 endpoint>
OAUTH_TOKEN_ENDPOINT=<existing OAuth token endpoint>
OAUTH_CLIENT_ID=<existing OAuth client id>
OAUTH_CLIENT_SECRET=<existing OAuth client secret>
OAUTH_SCOPE=<existing OAuth scope>
SKIP_SSL_VERIFY=false
```

`run.sh` installs `rbc_security` best-effort and `src/config.py` enables it when available, so the direct upstream path uses the same enterprise certificate behavior as the existing launcher. If OAuth is configured, OAuth is used before any static `TARGET_API_KEY`.

Native mode is only for an upstream that already supports `/v1/responses`:

```env
CODEX_UPSTREAM_WIRE_API=responses
TARGET_ENDPOINT=<native Responses upstream /v1 endpoint>
```

The Chat Completions adapter supports text, instructions, function tools, function-call outputs, `max_output_tokens`, model mapping, usage extraction, non-streaming responses, streaming text deltas, and streaming function-call deltas. Hosted built-in Responses tools such as web/file search are rejected with an OpenAI-shaped error unless native Responses mode is enabled.

## Work Machine Env

Start with the existing launcher `.env` that already works on the work machine. Copy that file to this repo as `src/.env`; do not rewrite the upstream auth section. Keep the existing direct endpoint, OAuth, SSL, timeout, token-limit, model mapping, pricing, and rbc_security-related behavior.

Then append this Codex-only block at the bottom:

```env
# =============================================================================
# CODEX LOCAL PROXY LAUNCHER OVERRIDES
# =============================================================================

# Keep Codex on its own local port so it does not collide with the other launcher.
CODEX_PROXY_PORT=5051

# Codex Desktop custom providers currently use the Responses API. This launcher
# exposes /v1/responses to Codex and translates to Chat Completions upstream.
CODEX_UPSTREAM_WIRE_API=chat_completions

# Codex-facing provider settings.
CODEX_PROVIDER_ID=codex-local-proxy
CODEX_PROVIDER_NAME=Codex Local Proxy

# Codex Desktop app/config behavior.
CODEX_APP_PATH=/Applications/Codex.app
AUTO_APPLY_CODEX_CONFIG=true
AUTO_RESTART_CODEX_DESKTOP=true
AUTO_OPEN_BROWSER=true

# Leave blank unless you intentionally want fixed local tokens. When blank, the
# launcher creates ~/.codex/codex-launcher/proxy_token for Codex Desktop auth.
CODEX_PROXY_ACCESS_TOKEN=
CODEX_DASHBOARD_ACCESS_TOKEN=
```

Do not add `CODEX_TARGET_ENDPOINT` or `CODEX_TARGET_API_KEY` for the normal work-machine setup. Leaving those unset makes the Codex launcher inherit the existing direct `TARGET_ENDPOINT`, `TARGET_API_KEY` if present, `OAUTH_*`, `SKIP_SSL_VERIFY`, timeout, SSL, and rbc_security behavior from the copied env.

The Codex launcher also inherits the existing `MODEL_OPTIONS`, `OPENAI_MODEL_OPTIONS`, `DEFAULT_MODEL`, `MODEL_MAPPING`, and pricing values. Only set `CODEX_MODEL_OPTIONS` or `CODEX_DEFAULT_MODEL` if you intentionally want Codex Desktop to see a different model list/default than the existing launcher.

Make sure `MODEL_MAPPING` contains entries for every model Codex will expose. If the existing mapping does not include your chosen default model, either add its internal mapping or set `DEFAULT_MODEL` to a model that is already mapped.

Only use this alternative if you intentionally want to chain through another local proxy:

```env
CODEX_TARGET_ENDPOINT=http://127.0.0.1:5050/v1
CODEX_TARGET_API_KEY=<other-local-proxy-token>
```

## Key Env Vars

- `CODEX_PROXY_PORT`, `BIND_HOST`
- `CODEX_PROXY_ACCESS_TOKEN`, `CODEX_DASHBOARD_ACCESS_TOKEN`
- `CODEX_TARGET_ENDPOINT`, `CODEX_TARGET_API_KEY`, `CODEX_UPSTREAM_WIRE_API`
- `MODEL_OPTIONS`, `OPENAI_MODEL_OPTIONS`, `DEFAULT_MODEL`
- Optional overrides: `CODEX_MODEL_OPTIONS`, `CODEX_DEFAULT_MODEL`
- `MODEL_MAPPING`, `MODEL_PRICING_USD_PER_1K`
- `CODEX_HOME`, `CODEX_CONFIG_PATH`, `CODEX_PROXY_TOKEN_FILE`
- `CODEX_PROVIDER_ID`, `CODEX_PROVIDER_NAME`, `CODEX_APP_PATH`
- `AUTO_APPLY_CODEX_CONFIG`, `AUTO_RESTART_CODEX_DESKTOP`, `AUTO_OPEN_BROWSER`
- `USE_PLACEHOLDER_MODE`, `DEV_MODE`
- `OAUTH_TOKEN_ENDPOINT`, `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`, `OAUTH_SCOPE`

## Dashboard APIs

- `GET /api/codex/status`
- `POST /api/codex/apply-config`
- `POST /api/codex/restore-config`
- `POST /api/codex/restart-desktop`
- `POST /api/codex/smoke-test`

Sensitive values are redacted in API responses, dashboard views, and logs.

## Restore

Restore the saved Codex config:

```bash
./restore-codex.sh
```

Restore and restart Codex Desktop:

```bash
./restore-codex.sh --restart
```

Backups and state live under `~/.codex/codex-launcher/`.

## Development

Run with placeholder responses:

```bash
./run-dev.sh
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Optional integration checks on a machine with Codex CLI/Desktop installed:

```bash
codex doctor --strict-config
codex exec --model gpt-5.5 "reply with ok"
```
