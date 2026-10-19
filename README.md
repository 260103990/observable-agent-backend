# Observable Agent Backend

一个用于学习 Python 后端与 AI Agent 工程的 FastAPI 项目。

当前版本提供普通聊天接口、可测试的单步 Tool Calling Agent，以及请求级日志、Agent Trace、Trace 查询、配置管理和错误转换。

> 当前项目仍处于学习和迭代阶段，不是生产级 Agent 平台。
> 已支持模型决策和工具执行的 Trace 与 Span，执行记录保存在当前进程的内存中。

## 当前功能

### HTTP API

- `GET /health`：检查后端服务是否正常。
- `POST /chat`：调用 Ollama `/api/generate` 生成文本回答。
- `POST /agent/run`：让 Ollama 选择一个已注册工具并执行。
- `GET /traces/{trace_id}`：按 Trace ID 查询执行记录；记录不存在时返回 `404`。

四个接口分别在 OpenAPI 文档中归类为 `health`、`chat`、`agent` 和 `traces`。

### 单步 Agent

当前 Agent 执行过程：

1. 将 Tool Registry 中的工具导出为 JSON Schema；
2. 把用户 prompt 和工具 Schema 发送给 Ollama `/api/chat`；
3. 要求模型返回且只返回一个 Tool Call；
4. 解析工具名称和参数；
5. 检查工具是否位于注册表白名单中；
6. 使用 Pydantic 校验工具参数；
7. 执行工具并直接返回结果。

当前只注册了一个工具：

```text
add_numbers(a: int, b: int) -> int
```

示例流程：

```text
用户请求：“请计算 17 + 25”
        ↓
Ollama 选择 add_numbers
        ↓
参数 {"a": 17, "b": 25}
        ↓
Tool Registry 校验并执行
        ↓
返回 {"result": 42}
```

这里的“单步”意味着：

- 每次请求只接受并执行一个 Tool Call；
- 工具结果直接返回给客户端；
- 当前不会把工具结果再次发送给模型生成自然语言总结；
- 当前没有多步推理循环。

### 后端工程能力

- 使用 Pydantic 校验 HTTP 请求和工具参数；
- 使用 `pydantic-settings` 读取默认值、环境变量和 `.env`；
- 使用 FastAPI `Depends` 组装 `AgentRunner`；
- 使用 HTTPX 异步调用 Ollama；
- 使用自定义异常隔离网络层、Agent 层和 API 层；
- 使用 pytest、依赖覆盖和 `httpx.MockTransport` 进行离线测试。

### 请求级可观测性

- 自动生成或保留 `X-Request-ID`；
- 使用 `X-Process-Time-Ms` 返回服务端处理耗时；
- 使用结构化日志记录请求方法、路径、状态码、耗时和异常类型；
- 未处理异常会记录 traceback，并继续交由 Starlette 返回 HTTP `500`。

### Agent 级可观测性

- 使用共享 `TraceStore` 保存和查询执行记录。
- 一次成功执行包含 `agent.run`、`model.decision` 和 `tool.execute` 三个 Span。
- 记录执行状态、总耗时、各阶段耗时和失败阶段的异常类型。
- 使用 `request_id` 将 Agent Trace 与 HTTP 请求关联。
- 成功执行后，通过响应头 `X-Trace-ID` 返回本次执行的 Trace ID。
- 使用 `GET /traces/{trace_id}` 查询完整记录，包括嵌套的 Span 数据。
- 记录不存在时返回 `404`，响应正文为 `{"detail": "Trace not found"}`。

## 架构

```text
Client
  |
  v
Request Context Middleware
  | request_id / timing / structured logs
  v
FastAPI
  |
  +---------------- POST /chat ----------------+
  |                                             |
  | ChatRequest                                 |
  |      ↓                                      |
  | Ollama /api/generate                        |
  |      ↓                                      |
  | ChatResponse                                |
  |                                             |
  +-------------- POST /agent/run --------------+
                                                |
                                         AgentRunRequest
                                                |
                                                v
                                           AgentRunner
                                                |
                              +-----------------+-----------------+
                              |                                   |
                              v                                   v
                        Tool Registry                    OllamaDecisionMaker
                     导出工具 JSON Schema                 POST /api/chat
                              |                                   |
                              +---------- ToolCall <---------------+
                                                |
                                                v
                                  白名单检查 + Pydantic 参数校验
                                                |
                                                v
                                         add_numbers
                                                |
                                                v
                                      AgentRunResponse
```

## 项目结构

```text
observable-agent-backend/
├── app/
│   ├── agent/
│   │   ├── ollama_decision_maker.py   # 解析 Ollama 的工具决策
│   │   └── runner.py                  # 单步 Agent 编排
│   ├── api/
│   │   ├── dependencies.py            # AgentRunner 依赖组装
│   │   └── routes/
│   │       ├── agent.py               # POST /agent/run
│   │       ├── chat.py                # POST /chat
│   │       └── health.py              # GET /health
│   ├── schemas/
│   │   ├── agent.py                   # Agent 请求、响应和 ToolCall
│   │   └── chat.py                    # Chat 请求与响应
│   ├── services/
│   │   └── ollama_service.py          # Ollama HTTP 客户端
│   ├── tools/
│   │   ├── calculator.py              # add_numbers 工具
│   │   └── registry.py                # 工具注册、Schema 和执行
│   ├── config.py                      # 环境配置
│   └── main.py                        # FastAPI 应用与请求中间件
├── tests/
│   ├── test_agent_route.py
│   ├── test_agent_runner.py
│   ├── test_calculator.py
│   ├── test_chat.py
│   ├── test_chat_schemas.py
│   ├── test_config.py
│   ├── test_health.py
│   ├── test_ollama_decision_maker.py
│   ├── test_ollama_service.py
│   ├── test_request_context.py
│   └── test_tool_registry.py
├── .env.example
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.11 或更高版本
- Ollama
- 本地模型 `qwen2.5:1.5b`

检查已有模型：

```powershell
ollama list
```

如果模型尚未安装：

```powershell
ollama pull qwen2.5:1.5b
```

## 安装

```powershell
cd observable-agent-backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## 配置

`.env.example` 包含：

```dotenv
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_GENERATE_URL=http://127.0.0.1:11434/api/generate
OLLAMA_CHAT_URL=http://127.0.0.1:11434/api/chat
OLLAMA_TIMEOUT_SECONDS=120
```

- `OLLAMA_MODEL`：聊天和 Agent 使用的本地模型；
- `OLLAMA_GENERATE_URL`：`/chat` 使用的 Ollama 接口；
- `OLLAMA_CHAT_URL`：`/agent/run` 使用的 Tool Calling 接口；
- `OLLAMA_TIMEOUT_SECONDS`：等待 Ollama 响应的超时时间。

## 运行测试

```powershell
python -m pytest -q
```

测试覆盖：

- `/health`、`/chat` 和 `/agent/run`；
- 请求数据清理和空字符串拒绝；
- FastAPI 依赖覆盖；
- Ollama 请求构造与响应解析；
- Tool Call 解析和数量约束；
- Tool Registry 注册、重复注册和未知工具；
- Pydantic 工具参数校验；
- Ollama 网络错误和 HTTP 错误转换；
- Agent 路由的 `502`、`503` 和 `504` 映射；
- 请求 ID、处理耗时和结构化日志；
- OpenAPI 路由标签。

自动化测试使用假的 Agent、依赖覆盖和 `httpx.MockTransport`，因此测试时不需要启动 Ollama，也不会加载真实模型。

## 启动服务

确认 Ollama 正在运行，然后启动 FastAPI：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

交互式 API 文档：

```text
http://127.0.0.1:8000/docs
```

## API

### Health Check

```http
GET /health
```

响应：

```json
{
  "status": "ok"
}
```

### Chat

```http
POST /chat
Content-Type: application/json
```

请求：

```json
{
  "message": "请用一句中文介绍你自己"
}
```

响应：

```json
{
  "answer": "模型生成的回答"
}
```

PowerShell 示例：

```powershell
$body = @{
    message = "请用一句中文介绍你自己"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/chat" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    -TimeoutSec 120
```

### Agent Run

```http
POST /agent/run
Content-Type: application/json
```

请求：

```json
{
  "prompt": "你必须调用 add_numbers 工具计算 17 + 25，不要直接回答。"
}
```

成功响应：

```json
{
  "result": 42
}
```

PowerShell 示例：

```powershell
$body = @{
    prompt = "你必须调用 add_numbers 工具计算 17 + 25，不要直接回答。"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/agent/run" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    -TimeoutSec 120
```

真实运行结果取决于本地模型能否正确返回 Tool Call。自动化测试不依赖这种非确定行为。

## HTTP 状态码

通用状态码：

- `200 OK`：请求处理成功；
- `404 Not Found`：请求路径不存在；
- `405 Method Not Allowed`：路径存在，但不支持该 HTTP 方法；
- `422 Unprocessable Content`：请求 JSON 不符合 Pydantic 模型；
- `500 Internal Server Error`：发生尚未处理的服务器异常。

Ollama 相关状态码：

- `502 Bad Gateway`：Ollama 返回错误状态；
- `503 Service Unavailable`：无法连接 Ollama；
- `504 Gateway Timeout`：等待 Ollama 响应超时。

`/agent/run` 还会在以下情况返回 `502`：

- Ollama 返回无法解析的工具决策；
- Ollama 选择未注册工具；
- Ollama 提供不符合工具 Schema 的参数。

API 不会把底层 HTTPX、Pydantic 或 Registry 异常直接暴露给客户端。

## 请求可观测性

每个正常 HTTP 响应都会包含：

```http
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Process-Time-Ms: 3.14
```

客户端可以主动传入 `X-Request-ID`。如果没有传入，服务端会自动生成 UUID。

正常请求日志示例：

```text
INFO app.request request_completed request_id=live-log-001 method=POST path=/agent/run status_code=200 process_time_ms=3.14 exception_type=-
```

未处理异常使用 `request_failed` 事件和 `ERROR` 级别，并记录异常类型与 traceback。

HTTP 请求日志用于查看请求状态和处理耗时；Agent 内部的模型决策与工具执行步骤可以通过 Trace 查询接口查看。

## 当前限制

- Agent 只支持单步、单工具调用；
- 当前只注册了 `add_numbers`；
- Agent 直接返回工具结果，没有第二轮模型总结；
- 没有多步 Agent Loop、最大步数和循环检测；
- 没有对话历史和长期记忆；
- 没有数据库、RAG 或向量检索；
- 只支持 Ollama，不支持其他模型提供商；
- Trace 仅保存在当前进程内存中，重启后丢失，多个进程之间不共享；
- 尚未记录模型 Token 用量；
- 没有重试、限流、认证、权限控制和生产部署配置；
- 当前响应模型将 Agent 结果限定为整数，尚未支持通用工具结果。

## 后续计划

1. 支持将工具结果返回模型，生成最终自然语言回答；
2. 增加最大步骤数受控的多步 Agent Loop；
3. 扩展更多安全、可测试的工具；
4. 使用 PostgreSQL 保存会话、消息和执行记录；
5. 增加 RAG、评估数据集和自动评估；
6. 根据需要接入 Opik 等外部可观测性与评估平台。
