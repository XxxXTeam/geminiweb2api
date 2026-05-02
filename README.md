# Gemini Web 2 API

[English](./README.md) | [中文](./ZH_CN.md)

`Gemini Web 2 API` is a lightweight Go proxy that exposes OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints while forwarding requests to the Gemini web experience. It supports streaming responses, session continuity, token refresh, and a small telemetry endpoint for the built-in WebUI.

### Highlights

- OpenAI-compatible chat and model endpoints
- Streaming SSE responses
- Session reuse with `X-Session-ID` or `conversation_id`
- Optional proxy and cookie-based token refresh
- Telemetry endpoint at `/api/telemetry`
- Embedded dashboard at `/` and manual page at `/help`
- WebUI defaults to English and can switch to Chinese

### Quick Start

1. Enter the project directory:

```bash
cd e:/Project/AI/All2API/geminiweb2api
```

2. Install dependencies and run:

```bash
go mod tidy
go run ./cmd/geminiweb2api
```

Project layout now uses folders by responsibility:

- `cmd/geminiweb2api`: application entrypoint
- `internal/server`: HTTP server assembly and routing
- `internal/gemini`: Gemini request/response logic
- `internal/config`, `internal/logging`, `internal/httpclient`, `internal/token`, `internal/metrics`: infrastructure modules
- `internal/web`: embedded dashboard and help pages

3. Default listen address:

- `http://127.0.0.1:8080`

4. Verify the service:

```bash
curl -s "http://127.0.0.1:8080/api/telemetry"
curl -s "http://127.0.0.1:8080/v1/models" \
  -H "Authorization: Bearer your-api-key-here"
```

### Docker

Build the image:

```bash
docker build -t geminiweb2api:latest .
```

Run with a local `config.json` mounted into the container:

```bash
docker run -d \
  --name geminiweb2api \
  -p 8080:8080 \
  -v $(pwd)/config.json:/app/config.json \
  geminiweb2api:latest
```

Windows PowerShell example:

```powershell
docker run -d `
  --name geminiweb2api `
  -p 8080:8080 `
  -v ${PWD}\config.json:/app/config.json `
  geminiweb2api:latest
```

If you need outbound proxy support inside the container, pass the standard proxy environment variables:

```bash
docker run -d \
  --name geminiweb2api \
  -p 8080:8080 \
  -v $(pwd)/config.json:/app/config.json \
  -e HTTP_PROXY=http://host.docker.internal:7890 \
  -e HTTPS_PROXY=http://host.docker.internal:7890 \
  geminiweb2api:latest
```

Health check inside the container:

```bash
curl -s http://127.0.0.1:8080/api/telemetry
```

The image prefers running as a non-root user and includes a Docker `HEALTHCHECK` based on `/api/telemetry`.

If the mounted `/app` volume is not writable by the non-root user in your Docker environment, the entrypoint automatically falls back to root so the container can still start instead of crashing on permissions.

Use `GET /healthz` for upstream business health checks. It returns `503` when the service is up but no account is currently healthy.

Docker Compose:

```bash
cp config.json.example config.json
docker compose up -d --build
```

Optional environment overrides in `docker compose`:

- `GEMINIWEB2API_API_KEY`
- `GEMINIWEB2API_PUBLIC_ACCOUNT_STATUS`

### Configuration

Use `config.json` in the project root. You can start from `config.json.example`:

```json
{
  "api_key": "your-api-key-here",
  "token": "",
  "cookies": "",
  "tokens": null,
  "proxy": "",
  "gemini_url": "",
  "gemini_home_url": "",
  "port": 8080,
  "log_file": "",
  "log_level": "info",
  "note": [
    "Auto-generated config"
  ]
}
```

### Config Fields

- `api_key`
  Protects this local service. Clients must send `Authorization: Bearer <api_key>`.
- `token`
  Manual `SNlM0e` fallback token for Gemini web when automatic fetch fails.
- `cookies`
  Gemini web cookie string. Recommended when anonymous access is unstable or the environment requires sign-in state.
- `tokens`
  Reserved field. Currently unused.
- `accounts`
  Optional multi-account pool. When present, requests are assigned by session binding plus round-robin selection across healthy accounts. Each account supports `id`, `email`, `cookies`, `token`, `proxy`, `enabled`, and `weight`.
- `proxy`
  Explicit proxy such as `http://127.0.0.1:7890`. The app also respects `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`.
- `models`
  Optional model ID list returned by `GET /v1/models`. If empty, the built-in default Gemini model list is used.
- `model_aliases`
  Optional request model alias map. Example: map `gpt-4.1` to `gemini-3-pro` for upstream panels such as NewAPI.
- `gemini_url`
  Override for the Gemini generation endpoint in reverse-proxy setups.
- `gemini_home_url`
  Override for the Gemini homepage used for token discovery.
- `port`
  Local listen port, default `8080`.
- `log_file`
  Log output path. Empty means stdout.
- `log_level`
  `debug`, `info`, `warn`, or `error`.
- `public_account_status`
  Defaults to `false`. When `false`, `GET /api/accounts` and `GET /api/accounts/bindings` require `Authorization: Bearer <api_key>`. Set to `true` only for trusted local deployments where unauthenticated read-only status is acceptable.
- `note`
  Free-form note strings surfaced by `/api/telemetry` and the WebUI.

### Environment Variables

Production deployments can override selected `config.json` values with environment variables:

- `GEMINIWEB2API_API_KEY`
- `GEMINIWEB2API_PROXY`
- `GEMINIWEB2API_PORT`
- `GEMINIWEB2API_LOG_LEVEL`
- `GEMINIWEB2API_PUBLIC_ACCOUNT_STATUS`

Environment values take precedence over `config.json` at load time.

### Security Notes

- Do not commit `config.json`; it can contain API keys, Google cookies, tokens, and proxies.
- Keep `public_account_status` disabled for public or production deployments.
- Management APIs that mutate accounts always require `Authorization: Bearer <api_key>`.
- The authenticated account details endpoint can return full cookies and tokens; only expose the service behind trusted networks or authentication layers.

### Hot Reload

The process checks `config.json` every 5 seconds and reloads it automatically when the file changes. You do not need to restart the service after editing the config.

### Token and Cookies

#### Automatic mode

- The service tries to fetch the anonymous Gemini token (`SNlM0e`) from the Gemini homepage by default.
- You can often run without `cookies`, but adding them improves stability.

#### Manual token mode

When automatic fetch fails:

1. Open `https://gemini.google.com/` or your configured `gemini_home_url`.
2. Open browser devtools.
3. Search for `SNlM0e` in `Elements`, `Sources`, or `Network`.
4. Copy the value after `"SNlM0e":"..."`.
5. Put it into `config.json`:

```json
{
  "token": "SNlM0e-example-value"
}
```

#### Cookies

If your environment needs a logged-in browser session or you see protection pages:

1. Open Gemini in a working browser session.
2. Open devtools and find the cookie store.
3. Copy the full cookie string:

```text
SID=...; APISID=...; SAPISID=...; ...
```

4. Put it into `config.json`:

```json
{
  "cookies": "SID=...; APISID=...; SAPISID=...; ..."
}
```

Do not commit real cookies to a public repository.

### Multi-Account Pool

You can now run the proxy in multi-account mode by filling `accounts` in `config.json`.

Example:

```json
{
  "api_key": "your-api-key-here",
  "accounts": [
    {
      "id": "acc-1",
      "email": "first@example.com",
      "cookies": "SID=...; APISID=...",
      "token": "",
      "proxy": "",
      "enabled": true,
      "weight": 1
    },
    {
      "id": "acc-2",
      "email": "second@example.com",
      "cookies": "SID=...; APISID=...",
      "token": "",
      "proxy": "http://user:pass@proxy-host:port",
      "enabled": true,
      "weight": 1
    }
  ]
}
```

Behavior:

- The same `X-Session-ID` stays bound to the same account while that account is healthy.
- New sessions are assigned by round-robin across healthy accounts.
- Failed accounts enter exponential backoff starting at 30 seconds, doubling up to 30 minutes.
- If an account has `proxy`, token refresh and Gemini requests for that account use that proxy.
- If account `proxy` is empty, the service falls back to the global `proxy` setting or the machine's proxy environment.
- If `accounts` is empty, the service falls back to the legacy single-account `cookies` and `token` fields.

### Session Binding Persistence

Session-to-account bindings are persisted in `state.json` beside `config.json`.

- Persisted: session/account binding, bind time, last used time
- Not persisted: short-lived runtime page tokens like `SNlM0e`, `BL`, `f.sid`

On restart, bindings are restored when the referenced account still exists.

### Account Pool APIs

- `GET /api/accounts`
  Returns configured accounts and runtime state.
- `POST /api/accounts`
  Creates or updates an account.
- `GET /api/accounts/bindings`
  Returns current session-to-account bindings.
- `POST /api/accounts/{id}/enable`
  Enables an account.
- `POST /api/accounts/{id}/disable`
  Disables an account.
- `POST /api/accounts/{id}/refresh`
  Refreshes token state for one account immediately.

All account APIs require `Authorization: Bearer <api_key>`.

### Google Account Manager v1.8 Compatibility

The legacy Google account manager can keep using its existing Gemini session callback:

```http
POST /api/session/cookies
Authorization: Bearer <api_key>
Content-Type: application/json
```

Body:

```json
{
  "email": "account@gmail.com",
  "cookies": "SID=...; __Secure-1PSID=...; ...",
  "proxy": "http://user:pass@proxy-host:port",
  "persist": true
}
```

When `email` is present, this endpoint now upserts the cookie into the multi-account pool instead of only updating the legacy single-account `cookies` field. The generated account ID uses the email directly, for example:

```text
account@gmail.com
```

In the Google account manager settings, set:

- `GEMINIWEB2API_URL` to this service, for example `http://127.0.0.1:8080`
- `GEMINIWEB2API_KEY` to this service's `api_key`
- `GEMINIWEB2API_PERSIST` to `true` if you want updates written to `config.json`
- Optional `GEMINIWEB2API_ACCOUNT_PROXY` if all callbacks from that manager should use the same outbound proxy in this service

Then use its existing `抓 Session` / `批量抓 Session` action. Successful callbacks should show the imported account in this service's account pool.

### Usage Examples

#### Health check

```bash
curl -s "http://127.0.0.1:8080/api/telemetry"
```

#### Model list

```bash
curl -s "http://127.0.0.1:8080/v1/models" \
  -H "Authorization: Bearer your-api-key-here"
```

#### Non-stream chat

```bash
curl -s "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: demo-session" \
  -d '{
    "model": "gemini-3-flash",
    "stream": false,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Please introduce yourself."}
    ]
  }'
```

#### Streaming SSE

```bash
curl -N "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: demo-session" \
  -d '{
    "model": "gemini-3-flash",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Introduce yourself in three sentences."}
    ]
  }'
```

### Use Behind NewAPI

If you run a NewAPI panel or any OpenAI-compatible gateway, the recommended topology is:

1. Google cookie -> `geminiweb2api`
2. NewAPI upstream -> `geminiweb2api`
3. End users -> NewAPI

Recommended upstream settings in NewAPI:

- Base URL: `http://your-geminiweb2api-host:8080/v1`
- API Key: the `api_key` from `config.json`
- Model discovery: `GET /v1/models`
- Chat endpoint: `POST /v1/chat/completions`
- Responses endpoint: `POST /v1/responses`
- Health check: `GET /healthz`

Notes:

- `GET /v1/models` also requires `Authorization: Bearer <api_key>`.
- `POST /v1/responses` is supported as a minimal compatibility layer and is internally translated into `/v1/chat/completions` for text input.
- Streaming is supported with SSE and ends with `data: [DONE]`. The current implementation streams incremental chunks from the final Gemini content instead of a true token-by-token upstream stream.
- `stream_options.include_usage` is supported.
- `model_aliases` can be used to align NewAPI/OpenAI-style model names with Gemini model IDs.
- Common OpenAI/NewAPI fields such as `max_completion_tokens`, `top_p`, `presence_penalty`, `frequency_penalty`, `response_format`, and `user` are accepted for compatibility. Some are pass-through compatibility fields and may not materially change Gemini Web behavior.

Recommended model names for upstream mapping:

- `gemini-3-flash`
- `gemini-3`
- `gemini-3-pro`
- `gemini-2.5-flash`
- `gemini-2.5-pro`

Suggested first-choice default:

- `gemini-3-flash`

Example NewAPI health probe:

```bash
curl -s "http://127.0.0.1:8080/v1/models" \
  -H "Authorization: Bearer your-api-key-here"
```

### Session Continuity

- Keep `X-Session-ID` stable for the same user or conversation.
- The service will reuse Gemini-side session context for better multi-turn continuity.

### Troubleshooting

- Gemini access fails or times out:
  Check `proxy` first, then system proxy environment variables.
- HTML, captcha, or protection pages appear:
  Try cookies, change IP/proxy, and enable `debug` logging.
- Conversation continuity is broken:
  Reuse the same `X-Session-ID`.
- Automatic token fetch fails:
  Manually set `token`, or verify that `cookies` are valid and not expired.

### WebUI

- `GET /` renders the dashboard
- `GET /help` renders the manual page
- Theme and language preferences are stored in local storage
- Default language is English, with a Chinese switch available in the UI
