# Gemini Web API

`geminiweb2api` 是一个用 Go 编写的轻量级代理服务，它将 OpenAI 兼容的 `/v1/chat/completions` 请求转发到非官方的 Gemini 流式后端，并在保留对话上下文、简单日志与监控的同时给出一致的接口。

## 核心能力

- 提供与 OpenAI 相同的 REST 路径（`/v1/chat/completions` 支持流式与非流式、`/v1/models`）。
- 通过 `X-Session-ID` 维护 Gemini 会话 ID，保持连续对话。
- 支持代理配置，并监控 `config.json`（每 5 秒自动重载）。
- 提供 `/` 端点供监控使用：展示运行状态、每分钟请求数 (RPM)、token 统计与自定义备注。
- 自定义日志器支持 `debug`/`info`/`warn`/`error` 级别，也可选写入文件。

## 配置说明 (`config.json`)

```json
{
  "api_key": "<your-secret-key>",
  "token": "<gemini-token>",
  "proxy": "",
  "port": 8080,
  "log_level": "debug",
  "log_file": "",
  "note": [
    "自定义监控信息"
  ]
}
```

- `api_key`：客户端调用时需携带的 Bearer 令牌。
- `token`：从已登录的 Gemini 网页提取的 session token。
- `proxy`：可选的出口代理地址。
- `port`：HTTP 服务端口，默认 `8080`。
- `log_level`：控制日志输出粒度。
- `log_file`：为空时输出到控制台，否则写入文件。
- `note`：可通过 `/` 端点查看的文本备注，用于宣传/状态说明。

> ⚠️ 配置文件中不要提交真凭据，请使用 `.gitignore` 或其它保密手段处理 `config.json`。

## 启动

```bash
go mod tidy
go run .
```

或可编译为二进制运行：

```bash
go build -o gemini-web .
./gemini-web
```

确保 `config.json` 与可执行文件处于同一目录；程序会监控该文件并在变更后自动重载网络/日志配置。

## API 说明

### GET `/`

返回服务的遥测数据，包含运行时间、RPM、token 统计及 `note` 内容，便于健康检查。

### GET `/v1/models`

返回一个模拟的 `gemini-3-flash` 模型，用于通过 OpenAI 风格的模型发现流程。

### POST `/v1/chat/completions`

- 必须包含 `Authorization: Bearer <api_key>`。
- 请求体示例：

  ```json
  {
    "model": "gemini-3-flash",
    "stream": true,
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "你好"}
    ]
  }
  ```

- 通过 `X-Session-ID` 锁定会话（缺失时默认 `default-<client_ip>`），以便 Gemini 保持上下文。
- `stream=true` 时返回 SSE 块，模拟 OpenAI 的 `chat.completion.chunk`；否则返回完整回答与 token 用量。

## 日志与指标

- 自定义日志器遵循配置的 `log_level`，调试级别会打印请求参数与 Gemini 响应片段。
- 指标结构记录总请求/成功/失败数量、输入输出 token 以及用于计算 RPM 的最近请求时间戳。


```bash
git add README.md
git commit -m "fact fix: document Gemini web proxy"
git push
```
