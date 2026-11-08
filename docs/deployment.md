# 本地演示部署：Docker Compose

这套配置单独创建应用、PostgreSQL/pgvector、数据库卷和论文文件卷，不连接原 55432/55433 开发数据库。默认 API 是 `http://127.0.0.1:18080`，数据库没有宿主机端口。Ollama 复用宿主机已有实例；DeepSeek 密钥在运行时提供。

当前实际验收环境是 Windows Docker Desktop + Linux amd64 容器、Python 3.11。它是本地演示部署，不包含公网认证/TLS、高可用或自动备份。

## 首次启动

在项目根目录操作。先创建专用配置：

```powershell
Copy-Item .env.demo.example .env.demo
```

本地编辑 `.env.demo`：将 `DEMO_DB_PASSWORD` 换成自己的字母/数字/下划线/短横线密码，避免 URL 分隔符；如需真实问答，再填写 `DEEPSEEK_API_KEY`。未填密钥仍能启动、导入论文和检索，生成回答的相关接口会返回原有的 503。密钥无需发送到聊天，不要写进 Dockerfile 或示例文件。

`.env.demo` 与原 `.env` 分开，已加入 Git 忽略，构建上下文也排除它。`--env-file` 用于 Compose 的变量替换，只有 compose 文件显式声明的字段传入应用。Docker 管理员仍可读取容器环境变量，本方案不是专用密钥管理服务。

```powershell
docker compose --env-file .env.demo -f compose.demo.yml config --quiet
docker compose --env-file .env.demo -f compose.demo.yml build app
docker compose --env-file .env.demo -f compose.demo.yml up -d postgres --wait
docker compose --env-file .env.demo -f compose.demo.yml run --rm init-db
docker compose --env-file .env.demo -f compose.demo.yml up -d app --wait
```

这几个步骤分别是校验配置、构建镜像、等待数据库、显式建初始表、启动应用。`init-db` 不会在普通应用启动时自动执行。

```powershell
Invoke-RestMethod http://127.0.0.1:18080/health
Invoke-RestMethod http://127.0.0.1:18080/ready
Start-Process http://127.0.0.1:18080/docs
```

`/ready` 的 `database=ok`、`rag=initialized` 只证明本地初始化/数据库检查，没有向 DeepSeek 或 Ollama 发健康检查请求。

## 初始化为什么是单独一步

`init-db` 使用同一镜像里的 `scripts/init_demo_database.py --confirm-demo`，只接受 `postgres:5432/observable_agent_demo`，拒绝查询参数或其他目标，不读本地 `.env`。

新空库中的 Trace 和 RAG 五张表在一个事务里创建。事务级 advisory lock 防止两个初始化进程同时建表；`SET LOCAL search_path = public` 确保建表位置与应用的启动检查一致。任一脚本失败都会回滚整批。

再次执行时，恰好存在这五张 public 表就返回 `already_initialized`，不重建/清空数据。部分表或额外 public 表会返回 `schema_requires_review`，应先检查状态。这里仅检查表集合，不验证所有列/约束/历史版本，也不会自动迁移不兼容 schema；未来结构变更需单独制定迁移方案。

## Ollama 与真实导入

Docker Desktop 下默认 `http://host.docker.internal:11434` 指向宿主机。容器内的 `127.0.0.1` 指容器自己，不能沿用原宿主机地址。宿主机需已有 `qwen3-embedding:0.6b`；本轮不会自动下载或更换模型。

Linux 宿主机可能还需通过 Compose override 为 app 增加 `extra_hosts: ["host.docker.internal:host-gateway"]`，并让 Ollama 在容器可达的地址监听；这条宿主机连接路径尚未实测。本项目没有自动修改宿主机监听范围。

在 `/docs` 上传 PDF：原文件写入 `/app/data/papers` 对应的卷，页面、Chunk、1024 维向量与 Trace 存入 demo 数据库。当前依然是单篇检索与原来的证据/引用契约。

本批已用合成一页 PDF 实际验证导入、宿主机 Ollama Embedding、重建后的检索与 Trace 查询；没有调用 DeepSeek，因此不把它写成新的真实问答质量验收。

## 停止、更新与数据保留

```powershell
docker compose --env-file .env.demo -f compose.demo.yml stop
docker compose --env-file .env.demo -f compose.demo.yml up -d postgres --wait
docker compose --env-file .env.demo -f compose.demo.yml up -d app --wait
```

`stop` 保留容器和卷；`down` 删除本项目容器和网络，但保留命名卷。`down -v` 会额外删除本项目数据卷，不是日常停止命令。数据库和 PDF 需要一起保留，只有向量库而没有 PDF 并不等于保留完整来源。

默认项目名是 `observable-agent-demo`，卷名随项目名生成。使用 `-p` 更换项目名会使用另一套卷；不要把它误认为原数据被删除。旧开发卷没有被引用。已有卷的数据库密码不会因编辑 `POSTGRES_PASSWORD` 环境变量自动更改，应保持原配置；密码轮换需另行处理。

应用代码或运行配置更新后：

```powershell
docker compose --env-file .env.demo -f compose.demo.yml build app
docker compose --env-file .env.demo -f compose.demo.yml up -d --force-recreate app --wait
```

单纯 `restart app` 不会重新读取改变后的容器环境变量。应用使用 exec 形式启动 Python/Uvicorn，收到停止信号后先等待进行中的请求；Uvicorn 的宽限期为 270 秒、容器为 300 秒。超长导入或失去响应的依赖仍可能超过期限，不承诺任何情况下均无损退出。

## 故障定位

```powershell
docker compose --env-file .env.demo -f compose.demo.yml ps
docker compose --env-file .env.demo -f compose.demo.yml logs --tail 80 app
docker compose --env-file .env.demo -f compose.demo.yml logs --tail 80 postgres
```

- 启动缺表：确认执行了 init-db，并阅读其固定错误码；不要改指向旧开发库。
- `/health` 200、`/ready` 503：检查数据库和连接资源；进程能响应不等于数据库可用。
- ready 200、导入失败：另查 Ollama 网络、模型身份、PDF 和文件卷权限；ready 不验证这些业务步骤。
- 容器 healthy 是健康检查信号，不会自动把流量切到别处；`restart: unless-stopped` 也不会因为单纯 unhealthy 自动重启。
- 应用非 root（UID/GID 10001）、根文件系统只读；论文卷及 `/tmp` 可写。首建卷会继承论文目录权限。复用外部卷时需核对归属。

## 构建与验证边界

Dockerfile 固定 Python 基础镜像摘要；数据库镜像也固定摘要。`requirements.container.lock` 固定这次 Linux 解析得到的直接/间接运行依赖，正式镜像不安装 pytest。版本锁未附 wheel 哈希，也没有离线包仓库，不声称任意平台逐字节相同构建。依赖更新后需重新构建和验证，不直接拿 Windows freeze 覆盖 Linux 锁文件。

`.dockerignore` 默认拒绝，仅允许应用 `.py`/`.sql`、初始化脚本与锁文件；新增运行资源时要明确加入白名单。已有 app、tests、scripts 业务逻辑与 API 不因容器化而重构。

实际测试和限制见 [发布验证记录](validation.md)。官方依据：[Compose 启动顺序](https://docs.docker.com/compose/how-tos/startup-order/)、[Docker 构建最佳实践](https://docs.docker.com/build/building/best-practices/)。
