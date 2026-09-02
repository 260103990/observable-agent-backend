# Observable Agent Backend

这是 `mdx maas` 下的独立后端学习项目。项目将现有的 Ollama 流量实验逐步扩展为一个具有 RAG、Tool Calling、评估与可观测性的 LLM Agent Backend。

当前版本使用 FastAPI 提供 HTTP API，并通过异步 HTTP 请求调用本地 Ollama 中的 Qwen2.5 1.5B 模型。

## 当前功能

- `GET /health`：检查后端服务是否正常运行。
- `POST /chat`：接收用户消息并调用本地 Ollama 生成回答。
- Pydantic 请求与响应校验。
- 使用 HTTPX 异步调用 Ollama API。
- 使用 pytest 验证路由、请求数据和 Ollama service。
- 使用假 HTTP 响应测试 LLM service，不要求测试时启动真实模型。

## Architecture

```text
Client
  |
  | HTTP request
  v
FastAPI
  |
  | validated ChatRequest
  v
Ollama Service
  |
  | POST /api/generate
  v
Ollama + Qwen2.5 1.5B
  |
  | generated answer
  v
ChatResponse -> Client
```

## 项目结构

```text
observable-agent-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── services/
│       ├── __init__.py
│       └── ollama_service.py
├── tests/
│   ├── __init__.py
│   ├── test_chat.py
│   ├── test_health.py
│   └── test_ollama_service.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.11 或更高版本
- Ollama
- 本地模型 `qwen2.5:1.5b`

检查本地模型：

```powershell
ollama list
```

如果模型尚未安装：

```powershell
ollama pull qwen2.5:1.5b
```

## 安装

```powershell
cd "C:\Users\26010\Desktop\mdx maas\observable-agent-backend"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行测试

```powershell
python -m pytest -v
```

测试范围包括：

- `/health` 是否返回 HTTP 200 和正确的 JSON。
- `/chat` 是否调用 LLM service 并返回模型答案。
- Ollama 请求是否使用正确的模型、prompt、URL 和 POST 方法。
- Ollama 返回的 JSON 是否被正确解析。

自动测试使用假的 LLM 和 HTTP 响应，因此不依赖真实 Ollama，也不会实际加载模型。

## 启动服务

确认 Ollama 正在运行，然后启动 FastAPI：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务地址：

```text
http://127.0.0.1:8000
```

交互式 API 文档：

```text
http://127.0.0.1:8000/docs
```

## API

### Health check

```http
GET /health
```

响应：

```json
{
  "status": "ok"
}
```

PowerShell 调用：

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health" -TimeoutSec 10
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

PowerShell 调用：

```powershell
$body = @{message="请用一句中文介绍你自己"} | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 30
```

## HTTP 状态码

- `200 OK`：请求处理成功。
- `404 Not Found`：请求的路径不存在。
- `405 Method Not Allowed`：路径存在，但不支持该 HTTP 方法。
- `422 Unprocessable Content`：请求 JSON 不符合 Pydantic 模型。
- `500 Internal Server Error`：后端或外部模型调用发生未处理的异常。

## 当前限制

- 模型名称目前固定为 `qwen2.5:1.5b`。
- `/chat` 尚未保存聊天历史。
- Ollama 连接失败和超时尚未转换为清晰的 API 错误响应。
- 尚未接入 PostgreSQL、RAG、Tool Calling 和评估系统。
- 当前模型较小，指令遵循和回答质量可能不稳定。

## 后续计划

1. 增加 Ollama 超时和连接错误处理。
2. 记录 `request_id`、token 数量和请求延迟。
3. 使用 PostgreSQL 保存 conversation 和 message。
4. 使用 PostgreSQL 与 pgvector 实现 RAG。
5. 增加结构化 Tool Calling 和 Agent Loop。
6. 建立 Evaluation 数据集和自动评估脚本。
