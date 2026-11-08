# 快速启动

## 1. 基础服务与离线测试

按README创建并激活Python3.11虚拟环境：

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

默认无数据库即可启动，访问`/health`和`/docs`。配置通过环境变量或本地`.env`读取；可以从`.env.example`复制，修改后重启服务。`.env`不提交。

`/chat`和旧`/agent/run`使用Ollama的`qwen2.5:1.5b`。新`/agent/loop`与论文回答使用`DEEPSEEK_API_KEY`和默认`deepseek-flash`配置标签；真实模型/Search调用可能计费，仅在显式配置时执行。远端模型标签不是锁定权重版本。

## 2. 最小多轮Agent演示

在启动进程本地配置`DEEPSEEK_API_KEY`，请求：

```bash
curl -X POST http://127.0.0.1:8000/agent/loop \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"使用计算工具验证 (17 + 25) + 42"}'
```

返回包含`status`、`rounds`、`tool_calls`、`answer`、`trace_id`。用`GET /traces/{trace_id}`查看阶段记录。默认Trace在内存中；配置数据库后持久化。`limit_reached`为正常业务终止状态，不应当作完整答案。

## 3. 全新论文演示库

需要Docker的Linux容器能力，以及本地Ollama：

```bash
ollama pull qwen3-embedding:0.6b
docker compose -f compose.pgvector.yml up -d
```

Compose只启动开发用PostgreSQL/pgvector，绑定`127.0.0.1:55433`。下面是**全新演示库的显式初始化**，应用不会自动迁移既有数据库。

Linux/macOS：

```bash
docker compose -f compose.pgvector.yml exec -T postgres psql -U agent -d observable_agent -v ON_ERROR_STOP=1 < app/observability/schema.sql
docker compose -f compose.pgvector.yml exec -T postgres psql -U agent -d observable_agent -v ON_ERROR_STOP=1 < app/rag/schema.sql
```

PowerShell：

```powershell
Get-Content -Raw -Encoding UTF8 app/observability/schema.sql | docker compose -f compose.pgvector.yml exec -T postgres psql -U agent -d observable_agent -v ON_ERROR_STOP=1
Get-Content -Raw -Encoding UTF8 app/rag/schema.sql | docker compose -f compose.pgvector.yml exec -T postgres psql -U agent -d observable_agent -v ON_ERROR_STOP=1
```

在本地环境配置以下值，并配置自己的DeepSeek密钥后启动服务：

```dotenv
DATABASE_URL=postgresql://agent:agent_dev_password@127.0.0.1:55433/observable_agent
RAG_ENABLED=true
```

示例账号是Compose的本地开发配置。已有库保持显式管理，不把初始化命令当作任意生产库迁移。

上传自己有权使用的文本型PDF，默认最多20MiB/100页：

```bash
curl -F 'file=@paper.pdf' http://127.0.0.1:8000/papers
curl -X POST http://127.0.0.1:8000/papers/PAPER_ID/questions \
  -H 'Content-Type: application/json' \
  -d '{"question":"论文的实验使用了哪些模型？"}'
```

把`PAPER_ID`换成导入响应中的实际ID。返回的页码是PDF物理页序号，原文/规范化文本/字符范围用于回溯，不等于视觉排版完全正确。索引会记录Embedding模型、digest、维度和查询表示；模型身份变化会拒绝混用。

## 4. Web Search与缓存

```dotenv
WEB_SEARCH_ENABLED=true
WEB_SEARCH_CACHE_ENABLED=true
```

重启服务，再显式请求Web工具：

```bash
curl -X POST http://127.0.0.1:8000/agent/loop \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"查找DeepSeek官方Tool Calls文档，说明函数由谁执行","enable_web_search":true}'
```

服务端和请求双开关控制Web工具。默认TTL300秒，LRU128项，序列化总预算8MiB、单项256KiB；这不是Python RSS上限。相同冷key并发由一个请求搜索，其余等待；不是全局限流，也不跨worker共享。该工具取得提供方摘要，不是网页正文。

`RAG_RETRIEVAL_STRATEGY=hybrid`为可选全文/向量融合，默认vector。`ANSWER_SCOPE_POLICY_ENABLED=true`为实验提示，默认false；实验没有证明稳定语义收益。

## 5. 分层验证与复现

默认测试不需要外部服务：

```bash
python -m pytest -q
```

真实数据库测试需显式配置独立的`TEST_DATABASE_URL`与`TEST_PGVECTOR_DATABASE_URL`，数据库名以`_test`结尾，并预先准备相应schema。再执行：

```bash
python -m pytest -m postgres -q
```

真实模型测试另有`RUN_*_LIVE_TESTS`开关；未运行的用例显示skip，不算通过。论文相关实测需本地`PAPER_PDF_PATH`，其中Cortex基准/重启脚本要求相应固定PDF，不是任意文件；仓库不附带论文。

固定实验需要本地密钥与全新输出目录：

```bash
python scripts/evaluate_search_cache.py --live --repeats 2 --output artifacts/search-cache-run
python scripts/evaluate_search_coalescing.py --live --repeats 2 --callers 4 --output artifacts/search-coalescing-run
python scripts/evaluate_agent_cache.py --live --repeats 2 --output artifacts/agent-cache-run
```

完整原六题还需`EVAL_DATABASE_URL`、已保存的`EVAL_PAPER_ID`及匹配工作负载PDF hash；评估只读现有论文，不导入/迁移主库：

```bash
python scripts/evaluate_agent.py --live --repeat 1 --output artifacts/agent-eval-run
```

输出含manifest、JSONL、Trace、失败和待审信息，保留在本地`artifacts/`。具体参数可用各脚本`--help`核对。


## V4 本地容器方式

见[Docker Compose部署](deployment.md)。它创建独立demo库/文件卷，不引用原55432/55433开发卷，显式初始化后启动。最新读快照/批量写代码需重新build镜像，原论文与真实密钥不随仓库发布。

部分真实数据库测试会新建自己的随机测试库并清理，需要CREATEDB；所有管理连接都应是专用测试库。Trace性能入口和范围见[实验](evaluation.md#trace-批量写入对照)。
