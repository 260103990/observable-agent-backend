# Observable Agent Backend

一个用于学习 Python 后端与 AI Agent 工程的 FastAPI 项目。

当前版本提供普通聊天接口、可测试的单步 Tool Calling Agent，以及请求级日志、Agent Trace、Trace 查询、配置管理和错误转换。Agent 会先让模型选择工具，再将工具执行结果交给模型，生成最终自然语言回答。

> 当前项目仍处于学习和迭代阶段，不是生产级 Agent 平台。
> 一次成功的 Agent 请求包含两次模型调用和一次工具执行。Trace 默认保存在内存中；配置 `DATABASE_URL` 后使用 PostgreSQL，服务重启后仍可查询。

## 当前功能

### HTTP API

- `GET /health`：检查后端服务是否正常。
- `POST /chat`：调用 Ollama `/api/generate` 生成文本回答。
- `POST /agent/run`：让 Ollama 选择一个已注册工具并执行，再根据工具结果生成最终回答。
- `GET /traces/{trace_id}`：按 Trace ID 查询执行记录；记录不存在时返回 `404`。

四个接口分别在 OpenAPI 文档中归类为 `health`、`chat`、`agent` 和 `traces`。

### 单步 Agent 与最终回答

当前 Agent 执行过程：

1. 将 Tool Registry 中的工具导出为 JSON Schema；
2. 第一次调用 Ollama `/api/chat`，发送用户 prompt 和工具 Schema；
3. 要求模型返回且只返回一个 Tool Call，解析工具名称和参数，并保留完整的 assistant 消息；
4. 检查工具是否位于注册表白名单中，使用 Pydantic 校验工具参数；
5. 执行工具，得到原始结果 `result`；
6. 第二次调用 Ollama `/api/chat`，发送原始用户消息、第一次的完整 assistant 消息，以及包含工具结果的 tool 消息；
7. 校验模型返回的最终回答 `answer`，然后同时返回 `result` 和 `answer`。

第二次请求不再提供工具 Schema，工具结果会先序列化为 JSON 字符串。最终回答必须是非空白字符串；如果模型再次返回非空的 `tool_calls`，本次请求会失败。

当前只注册了一个工具：

```text
add_numbers(a: int, b: int) -> int
```

示例流程：

```text
用户请求：“请使用工具计算 17 + 25，并给出最终回答”
        ↓
第一次模型调用：选择 add_numbers，参数为 {"a": 17, "b": 25}
        ↓
Tool Registry 校验并执行：得到整数 42
        ↓
第二次模型调用：根据原始请求、工具决策消息和工具结果生成回答
        ↓
返回 {"result": 42, "answer": "17 + 25 的结果是 42。"}
```

`result` 是工具计算得到的原始结果；`answer` 是模型根据工具结果生成的用户可读回答，具体措辞取决于模型。

这里的“单步”指每次请求只执行一个工具。成功路径固定为“模型决策 → 工具执行 → 模型总结”，目前还没有自主决定是否继续调用工具的多步 Agent Loop，也没有跨请求的对话历史。

### 后端工程能力

- 使用 Pydantic 校验 HTTP 请求和工具参数；
- 使用 `pydantic-settings` 读取默认值、环境变量和 `.env`；
- 使用 FastAPI `Depends` 组装工具注册表、模型决策组件、总结组件和共享 `TraceStore`；
- 使用 HTTPX 异步调用 Ollama；
- 使用 asyncpg 连接池、参数化 SQL 和事务保存 Trace 与 Span；
- 使用自定义异常隔离网络层、Agent 层和 API 层；
- 使用 pytest、依赖覆盖和 `httpx.MockTransport` 进行离线测试。

### 请求级可观测性

- 自动生成或保留 `X-Request-ID`；
- 使用 `X-Process-Time-Ms` 返回服务端处理耗时；
- 使用结构化日志记录请求方法、路径、状态码、耗时和异常类型；
- 未处理异常会记录 traceback，并继续交由 Starlette 返回 HTTP `500`。

### Agent 级可观测性

- 使用共享 `TraceStore` 连接真实 `AgentRunner` 与 Trace 查询接口；
- 一次成功执行包含 `agent.run`、`model.decision`、`tool.execute` 和 `model.summary` 四个 Span；
- 三个阶段 Span 都以 `agent.run` 为父 Span，总耗时包含最终回答生成阶段；
- 记录执行状态、各阶段耗时和失败阶段的异常类型；
- 使用 `request_id` 将 Agent Trace 与 HTTP 请求关联，成功响应通过 `X-Trace-ID` 返回 Trace ID；
- Trace 保存失败时记录 `trace_save_failed` 警告，保留原本的业务结果或业务异常。

失败时会保留已经执行的阶段，不会继续执行后续阶段：

| 执行结果 | `agent.run` | `model.decision` | `tool.execute` | `model.summary` |
| --- | --- | --- | --- | --- |
| 全部成功 | `ok` | `ok` | `ok` | `ok` |
| 模型决策失败 | `error` | `error` | 未执行 | 未执行 |
| 工具执行失败 | `error` | `ok` | `error` | 未执行 |
| 最终回答生成失败 | `error` | `ok` | `ok` | `error` |

`ok` 表示该阶段没有抛出异常并通过了现有校验，不代表模型回答的内容一定正确。

## 架构

```text
Client
  ↓
Request Context Middleware
  request_id / timing / structured logs
  ↓
FastAPI
  ├── POST /chat
  │     ChatRequest → Ollama /api/generate → ChatResponse
  │
  ├── POST /agent/run
  │     AgentRunRequest
  │       ↓
  │     AgentRunner
  │       ├── Tool Registry 导出工具 Schema
  │       ├── OllamaDecisionMaker → /api/chat（第一次模型调用）
  │       │     返回 ToolDecision：ToolCall + 完整 assistant 消息
  │       ├── Tool Registry 白名单检查、参数校验和工具执行
  │       │     得到 result
  │       └── OllamaSummarizer → /api/chat（第二次模型调用）
  │             得到 answer
  │       ↓
  │     AgentRunResult → AgentRunResponse {result, answer}
  │
  └── GET /traces/{trace_id}
        通过共享 TraceStore 查询 TraceRecord

AgentRunner → 共享 TraceStore ← Trace 查询接口
                  ├── 未配置 DATABASE_URL：InMemoryTraceStore
                  └── 已配置 DATABASE_URL：PostgresTraceStore → PostgreSQL

FastAPI lifespan：启动时创建连接池，关闭时释放
请求依赖：从 app.state.trace_store 取得当前应用的 Store
```

## 项目结构

省略各包的 `__init__.py`：

```text
observable-agent-backend/
├── app/
│   ├── agent/
│   │   ├── ollama_decision_maker.py   # 解析工具决策并保留 assistant 消息
│   │   ├── ollama_summarizer.py       # 根据工具结果生成并校验最终回答
│   │   └── runner.py                  # 编排模型决策、工具执行、总结和 Trace
│   ├── api/
│   │   ├── dependencies.py            # AgentRunner 与共享 TraceStore 的依赖组装
│   │   └── routes/
│   │       ├── agent.py               # POST /agent/run
│   │       ├── chat.py                # POST /chat
│   │       ├── health.py              # GET /health
│   │       └── traces.py              # GET /traces/{trace_id}
│   ├── observability/
│   │   ├── models.py                  # TraceRecord 与 SpanRecord
│   │   └── store.py                   # TraceStore 协议、内存和 PostgreSQL 实现
│   ├── schemas/
│   │   ├── agent.py                   # Agent 请求、响应、内部结果与工具决策
│   │   └── chat.py                    # Chat 请求与响应
│   ├── services/
│   │   └── ollama_service.py          # Ollama HTTP 调用与请求体构造
│   ├── tools/
│   │   ├── calculator.py              # add_numbers 工具
│   │   └── registry.py                # 工具注册、Schema 和执行
│   ├── config.py                      # 环境配置
│   └── main.py                        # FastAPI、lifespan 与请求中间件
├── tests/
│   ├── integration/
│   │   ├── test_postgres_trace_store.py
│   │   └── test_postgres_trace_api.py
│   ├── test_app_lifespan.py
│   ├── test_agent_route.py
│   ├── test_agent_runner.py
│   ├── test_agent_tracing.py
│   ├── test_calculator.py
│   ├── test_chat.py
│   ├── test_chat_schemas.py
│   ├── test_config.py
│   ├── test_health.py
│   ├── test_ollama_decision_maker.py
│   ├── test_ollama_service.py
│   ├── test_ollama_summarizer.py
│   ├── test_ollama_summary.py
│   ├── test_request_context.py
│   ├── test_tool_registry.py
│   ├── test_trace_dependencies.py
│   ├── test_trace_route.py
│   └── test_trace_store.py
├── docs/
│   ├── learning/                      # 项目路线与当前学习断点
│   └── superpowers/                   # 功能设计与实现计划
├── .env.example
├── requirements.txt
├── pytest.ini                         # 默认排除真实数据库测试
└── README.md
```

## 环境要求

- Python 3.11 或更高版本
- Ollama
- 本地模型 `qwen2.5:1.5b`
- PostgreSQL 模式另需 PostgreSQL 17；下面提供 Docker 本地示例

检查已有模型：

```powershell
ollama list
```

如果模型尚未安装：

```powershell
ollama pull qwen2.5:1.5b
```

## 安装

进入项目主目录后执行：

```powershell
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

# 可选：默认保持注释，使用内存 TraceStore
# DATABASE_URL=postgresql://agent:agent_dev_password@127.0.0.1:55432/observable_agent
```

- `OLLAMA_MODEL`：聊天和 Agent 使用的本地模型；
- `OLLAMA_GENERATE_URL`：`/chat` 使用的 Ollama 接口；
- `OLLAMA_CHAT_URL`：Agent 的工具决策和最终回答生成使用的 Ollama 接口；
- `OLLAMA_TIMEOUT_SECONDS`：每次 Ollama HTTP 调用使用的超时配置，单位为秒；不是整个 Agent 流程的总超时。
- `DATABASE_URL`：可选的应用数据库地址；未配置时使用内存，配置后使用 PostgreSQL。
- `TEST_DATABASE_URL`：真实数据库测试使用的系统环境变量，测试代码不从 `.env` 读取它。

推荐只在启动服务的专用 PowerShell 中设置 `DATABASE_URL`，测试终端使用独立的 `TEST_DATABASE_URL`。默认离线测试环境应保持未配置应用数据库地址；`.env.example` 的数据库示例因此保持注释。

## PostgreSQL 本地准备

默认内存模式可跳过本节。PostgreSQL 模式需要先准备数据库和表；应用启动时只创建连接池，不自动建库或建表。

首次创建本地容器可执行：

```powershell
docker run --name observable-agent-postgres `
    --detach `
    --publish 127.0.0.1:55432:5432 `
    --env POSTGRES_USER=agent `
    --env POSTGRES_PASSWORD=agent_dev_password `
    --env POSTGRES_DB=observable_agent `
    --volume observable-agent-postgres-data:/var/lib/postgresql/data `
    postgres:17
```

已有同名容器时复用它；容器已停止可执行 `docker start observable-agent-postgres`。上面的命名卷用于保存数据库数据。

首次初始化需要等待 PostgreSQL 就绪，再执行后续创建测试库和建表命令：

```powershell
docker exec observable-agent-postgres pg_isready -h 127.0.0.1 -U agent -d observable_agent
```

输出 `accepting connections` 后继续；尚未就绪时稍后重试这条检查。

应用库为 `observable_agent`。另建独立测试库（已经存在则跳过）：

```powershell
docker exec observable-agent-postgres createdb -U agent observable_agent_test
```

进入应用库：

```powershell
docker exec -it observable-agent-postgres psql -U agent -d observable_agent
```

执行初始化 SQL：

```sql
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
    span_index INTEGER NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    attributes JSONB NOT NULL,
    error_type TEXT,
    PRIMARY KEY (trace_id, span_index)
);
```

输入 `\q` 退出。再进入 `observable_agent_test`，执行相同的初始化 SQL，给真实数据库测试准备相同结构：

```powershell
docker exec -it observable-agent-postgres psql -U agent -d observable_agent_test
```

这些语句用于初始化新环境；已有表的后续结构变更需要另行迁移。

保存同一个 `trace_id` 时，主体与完整 Span 列表在一个事务中覆盖。`span_index` 记录列表位置；按它排序读取。删除 Trace 会级联删除其 Span。`request_id` 不要求唯一，`span_id` 也没有额外的唯一约束。

## 运行测试

默认运行离线测试：

```powershell
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m pytest -q
```

同时保持 `.env` 未启用 `DATABASE_URL`。`pytest.ini` 默认排除 `postgres` 标签，离线测试不需要 PostgreSQL 或 Ollama；连接池生命周期使用替身验证。

真实数据库测试使用前面准备好的独立测试库，显式选择 `postgres` 标签：

```powershell
$env:TEST_DATABASE_URL = "postgresql://agent:agent_dev_password@127.0.0.1:55432/observable_agent_test"
.\.venv\Scripts\python.exe -m pytest tests/integration -m postgres -q
```

数据库测试会访问真实 PostgreSQL；HTTP 整合测试仍使用预设模型响应，不需要启动 Ollama。测试使用随机 ID，并清理本次写入的记录。

测试覆盖：

- `/health`、`/chat`、`/agent/run` 和 `/traces/{trace_id}`；
- 请求数据清理、空字符串拒绝和 OpenAPI 路由标签；
- FastAPI 依赖覆盖、共享 Store 和真实 Runner 的依赖组装；
- Ollama 请求构造、Tool Call 数量约束与完整 assistant 消息保留；
- Tool Registry 注册、重复注册、白名单检查和 Pydantic 参数校验；
- 工具结果回传、最终回答解析，以及空回答、错误格式和再次调用工具的拒绝；
- Ollama 网络错误、HTTP 错误和 Agent 路由的 `502`、`503`、`504` 映射；
- 请求 ID、处理耗时和结构化日志；
- 四个 Span 的父子关系、阶段状态、异常类型与耗时；
- Trace 存取、保存失败处理，以及 Agent 执行到 HTTP 查询的集成验证；
- PostgreSQL Span 顺序、整体覆盖、事务回滚、空列表覆盖、对象独立性与 ID 契约；
- 连接池生命周期、请求级 Store 选择，以及真实数据库的 HTTP 查询和 Agent 写库。

默认测试使用模型组件替身、依赖覆盖和 `httpx.MockTransport`，不会加载真实模型。真实 Ollama 调用在启动服务后手动验收，见下面的 Agent 示例。

## 启动服务

确认 Ollama 正在运行。进入项目根目录，在服务专用终端选择一种存储模式。

### 内存模式

保持 `.env` 未启用 `DATABASE_URL`，执行：

```powershell
Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### PostgreSQL 模式

先完成数据库与表初始化，然后执行：

```powershell
$env:DATABASE_URL = "postgresql://agent:agent_dev_password@127.0.0.1:55432/observable_agent"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

环境变量仅影响该终端及其启动的服务。连接池由 `lifespan` 在应用启动时创建，关闭时释放；每次请求取用应用共享的 Store。按 Ctrl+C 可正常关闭服务。

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

响应示例：

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
  "prompt": "请使用 add_numbers 工具计算 17 + 25，并根据工具结果给出最终回答。"
}
```

成功响应示例：

```json
{
  "result": 42,
  "answer": "17 + 25 的结果是 42。"
}
```

其中 `result` 为整数，`answer` 为字符串。最终回答的具体措辞可能不同。

响应头示例：

```http
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Trace-ID: 80e6dddd-b1a3-4799-bc7c-1f628db0844c
X-Process-Time-Ms: 1500.00
```

PowerShell 示例：

```powershell
$body = @{
    prompt = "请使用 add_numbers 工具计算 17 + 25，并根据工具结果给出最终回答。"
} | ConvertTo-Json

$response = Invoke-WebRequest `
    -UseBasicParsing `
    -Method Post `
    -Uri "http://127.0.0.1:8000/agent/run" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    -TimeoutSec 300

# 从原始响应体按 UTF-8 解码，兼容响应字符串可能显示乱码的 PowerShell 环境
$agentResult = [System.Text.Encoding]::UTF8.GetString(
    $response.RawContentStream.ToArray()
) | ConvertFrom-Json
$agentResult

$traceId = [string]$response.Headers["X-Trace-ID"]
$traceId
```

Agent 成功时需要两次模型调用，因此客户端示例使用更长的等待时间。服务端的 `OLLAMA_TIMEOUT_SECONDS` 分别用于每次模型请求。

真实运行还要求本地模型正确返回工具决策和有效的最终回答。自动化测试使用预设响应验证流程，不依赖模型生成内容的确定性。

### Trace Query

读取 Agent 响应头中的 `X-Trace-ID`，查询执行记录。PostgreSQL 模式下，使用同一个数据库重启服务后仍可查询旧 ID：

```http
GET /traces/80e6dddd-b1a3-4799-bc7c-1f628db0844c
```

PowerShell 示例：

```powershell
# 沿用上一个示例取得的 $traceId；新终端中请先设置为本次响应头中的实际值
Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/traces/$traceId" |
    ConvertTo-Json -Depth 10
```

成功时返回 `TraceRecord`，包含 `trace_id`、`request_id`、`status`、`duration_ms` 和 `spans`。每个 Span 包含 ID、父 Span ID、名称、状态、耗时、属性和异常类型。

记录不存在时返回 HTTP `404`：

```json
{
  "detail": "Trace not found"
}
```

内存模式下，服务重启后旧记录丢失；PostgreSQL 模式下，只要仍连接同一数据库并保留记录，重启后仍可按旧 ID 查询。如果 Trace 保存失败，业务响应仍会保留，但对应记录可能查不到。

持久化验收：先保存本次 `$traceId`，在服务终端按 Ctrl+C，再以相同 `DATABASE_URL` 启动服务；请求终端查询原 ID，应读回同一 Trace 和四个成功 Span。

## HTTP 状态码

通用状态码：

- `200 OK`：请求处理成功；
- `404 Not Found`：请求路径不存在，或指定的 Trace 不存在；
- `405 Method Not Allowed`：路径存在，但不支持该 HTTP 方法；
- `422 Unprocessable Content`：请求 JSON 不符合 Pydantic 模型；
- `500 Internal Server Error`：发生尚未处理的服务器异常。

Ollama 相关状态码：

- `502 Bad Gateway`：Ollama 返回错误状态；
- `503 Service Unavailable`：无法连接 Ollama；
- `504 Gateway Timeout`：等待 Ollama 响应超时。

`/agent/run` 还会在以下情况返回 `502`：

- Ollama 返回无法解析或数量不符合要求的工具决策；
- Ollama 选择未注册工具；
- Ollama 提供不符合工具 Schema 的参数；
- 最终回答响应不是有效 JSON、格式不符合要求、内容不是字符串、为空或仅包含空白；
- 最终回答阶段再次返回非空的 `tool_calls`。

最终回答格式错误时，响应正文为 `{"detail": "Ollama returned an invalid summary"}`。该阶段发生连接失败或超时时，仍分别返回 `503` 或 `504`。

API 不会把底层 HTTPX、Pydantic 或 Registry 异常直接暴露给客户端。

## 请求可观测性

每个正常 HTTP 响应都会包含：

```http
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
X-Process-Time-Ms: 3.14
```

客户端可以主动传入 `X-Request-ID`。如果没有传入，服务端会自动生成 UUID。Agent 请求还会生成独立的 `trace_id`，用于关联内部阶段记录。

正常请求日志示例：

```text
INFO app.request request_completed request_id=live-log-001 method=POST path=/agent/run status_code=200 process_time_ms=1500.00 exception_type=-
```

未处理异常使用 `request_failed` 事件和 `ERROR` 级别，并记录异常类型与 traceback。

HTTP 请求日志用于查看请求状态和处理耗时；Agent 内部的模型决策、工具执行和最终回答生成阶段可以通过 Trace 查询接口查看。

## 当前限制

- Agent 只支持固定的“决策 → 单工具执行 → 总结”流程；
- 当前只注册了 `add_numbers`；
- 没有多步 Agent Loop、最大步数和循环检测；
- 没有对话历史和长期记忆；
- 尚未实现 Redis、RAG 或向量检索；
- 只支持 Ollama，不支持其他模型提供商；
- 默认内存模式的 Trace 在重启后丢失；需要持久化时启用 PostgreSQL 模式；
- PostgreSQL 当前保存 Trace 和 Span，尚未保存会话历史或通用任务结果；
- 最终回答校验只检查格式，不验证模型是否准确复述了工具结果；
- 尚未记录模型 Token 用量；
- 没有重试、限流、认证、权限控制和生产部署配置；
- HTTP 响应中的 `result` 字段目前限定为整数，尚未支持通用工具结果。

## 后续学习与开发方向

SQL 与 PostgreSQL TraceStore 已完成当前阶段的学习和本地验收，包括事务、应用连接池、真实模型请求和服务重启后查询。

后续按需推进，每阶段开始前单独确定范围：

1. **受控的 Agent Loop**：让模型根据已有结果决定继续调用工具或结束，增加最大步数限制，并扩展安全、可测试的工具；
2. **Redis**：结合实际查询需求练习缓存和 TTL，再按需求增加限流；
3. **会话与检索**：保存会话、消息和上下文，逐步增加 RAG、评估数据集和自动评估；
4. **实验任务后台**：在执行循环和存储基础上，练习后台任务、实验记录与报告生成；
5. 根据需要接入外部可观测性与评估平台。

项目路线和学习断点见 [PROJECT_PLAN.md](docs/learning/PROJECT_PLAN.md) 与 [CURRENT_TASK.md](docs/learning/CURRENT_TASK.md)。
