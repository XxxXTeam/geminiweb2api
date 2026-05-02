## Gemini Web 2 API

[English](./README.md) | [中文](./ZH_CN.md)

`geminiweb2api` 是一个用 Go 编写的轻量代理服务：对外提供 OpenAI 兼容的 `/v1/chat/completions` 与 `/v1/models` 接口，后端转发到 Gemini 网页端，并支持流式返回、会话保持、token 刷新以及用于内置 WebUI 的遥测接口。

### 功能概览

- OpenAI 兼容聊天与模型接口
- 支持流式 SSE 返回
- 通过 `X-Session-ID` 或 `conversation_id` 复用会话
- 支持代理与基于 Cookie 的 token 刷新
- 内置 `/api/telemetry` 遥测接口
- 提供 `/` 大屏与 `/help` 手册页
- WebUI 默认英文，可切换中文

### 快速开始

1. 进入项目目录：

```bash
cd e:/Project/AI/All2API/geminiweb2api
```

2. 安装依赖并运行：

```bash
go mod tidy
go run ./cmd/geminiweb2api
```

当前目录结构已按职责拆分：

- `cmd/geminiweb2api`：程序入口
- `internal/server`：HTTP 服务装配与路由
- `internal/gemini`：Gemini 请求与响应处理
- `internal/config`、`internal/logging`、`internal/httpclient`、`internal/token`、`internal/metrics`：基础设施模块
- `internal/web`：内嵌 WebUI 与帮助页

3. 默认监听地址：

- `http://127.0.0.1:8080`

4. 验证服务：

```bash
curl -s "http://127.0.0.1:8080/api/telemetry"
curl -s "http://127.0.0.1:8080/v1/models" \
  -H "Authorization: Bearer your-api-key-here"
```

### Docker

构建镜像：

```bash
docker build -t geminiweb2api:latest .
```

挂载本地 `config.json` 运行容器：

```bash
docker run -d \
  --name geminiweb2api \
  -p 8080:8080 \
  -v $(pwd)/config.json:/app/config.json \
  geminiweb2api:latest
```

Windows PowerShell 示例：

```powershell
docker run -d `
  --name geminiweb2api `
  -p 8080:8080 `
  -v ${PWD}\config.json:/app/config.json `
  geminiweb2api:latest
```

如果容器内需要代理访问外网，可以传入标准代理环境变量：

```bash
docker run -d \
  --name geminiweb2api \
  -p 8080:8080 \
  -v $(pwd)/config.json:/app/config.json \
  -e HTTP_PROXY=http://host.docker.internal:7890 \
  -e HTTPS_PROXY=http://host.docker.internal:7890 \
  geminiweb2api:latest
```

Docker Compose：

```bash
cp config.json.example config.json
docker compose up -d --build
```

### 配置文件

项目根目录使用 `config.json`，可以先从 `config.json.example` 复制：

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

### 字段说明

- `api_key`
  本地服务访问密钥，请求头中使用 `Authorization: Bearer <api_key>`。
- `token`
  手动填入 Gemini 的 `SNlM0e` 备用 token。
- `cookies`
  Gemini 网页 Cookie。匿名抓取不稳定或环境需要登录态时建议配置。
- `tokens`
  预留字段，目前未使用。
- `proxy`
  显式代理，例如 `http://127.0.0.1:7890`。
- `gemini_url`
  覆盖 Gemini 生成接口地址，适合反代场景。
- `gemini_home_url`
  覆盖 Gemini 首页地址，用于 token 获取。
- `port`
  本地监听端口，默认 `8080`。
- `log_file`
  日志输出路径，留空则输出到 stdout。
- `log_level`
  `debug`、`info`、`warn`、`error`。
- `note`
  任意备注字符串，会显示在 `/api/telemetry` 与 WebUI 中。

### 配置热加载

程序每 5 秒检查一次 `config.json`，文件变更后会自动重载，无需重启。

### token 与 cookies

#### 自动获取

- 程序默认会从 Gemini 首页抓取匿名 `SNlM0e` token。
- 不填 `cookies` 也会尝试匿名获取，但填写后通常更稳定。

#### 手动设置 token

当自动获取失败时：

1. 打开 `https://gemini.google.com/` 或你配置的 `gemini_home_url`
2. 打开浏览器开发者工具
3. 搜索 `SNlM0e`
4. 复制 `"SNlM0e":"..."` 中的值
5. 写入 `config.json`

#### 设置 cookies

如果环境需要登录态或存在防护页：

1. 在浏览器中正常打开 Gemini
2. 打开开发者工具，找到 Cookie 存储
3. 复制完整 Cookie 字符串
4. 写入 `config.json`

不要把真实 Cookie 提交到公共仓库。

### 调用示例

#### 健康检查

```bash
curl -s "http://127.0.0.1:8080/api/telemetry"
```

#### 获取模型列表

```bash
curl -s "http://127.0.0.1:8080/v1/models" \
  -H "Authorization: Bearer your-api-key-here"
```

#### 非流式对话

```bash
curl -s "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: demo-session" \
  -d '{
    "model": "gemini-3-flash",
    "stream": false,
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "请介绍一下自己。"}
    ]
  }'
```

#### 流式 SSE

```bash
curl -N "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: demo-session" \
  -d '{
    "model": "gemini-3-flash",
    "stream": true,
    "messages": [
      {"role": "user", "content": "用三句话介绍你自己。"}
    ]
  }'
```

### 会话保持

- 对同一用户或同一会话保持固定 `X-Session-ID`
- 服务会复用 Gemini 侧上下文以获得更好的多轮连续性

### 排障建议

- Gemini 访问失败或超时：
  先检查 `proxy`，再检查系统代理环境变量。
- 返回 HTML、验证码或防护页：
  尝试补充 `cookies`、更换代理或开启 `debug` 日志。
- 对话不连续：
  确认 `X-Session-ID` 没有每次变化。
- 自动 token 获取失败：
  手动设置 `token`，或确认 `cookies` 没有过期。

### WebUI

- `GET /` 为首页大屏
- `GET /help` 为用户手册页
- 主题与语言偏好保存在本地存储中
- 默认语言为英文，页面内可切换到中文
