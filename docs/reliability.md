# 可靠性与本地部署

## 存活与就绪

`/health` 保持 200/status=ok，仅证明进程响应。`/ready` 检查 lifespan 初始化完成、已配置的业务池能 acquire 并 SELECT 1，以及启用 RAG 的服务对象已初始化。失败返回安全 503、no-store，不泄露 DSN 或异常详情。

默认 READINESS_TIMEOUT_SECONDS=1 约束获取/查询，连接归还可能另需有界清理。取消继续传播。未配置数据库的原内存模式可就绪。它不调用模型、不验证完整 schema/磁盘/语义，不自动重启或分流。

## Trace 一致性与写入

保存仍在一个事务里 UPSERT Trace、DELETE 旧 Span、批量 INSERT。多个 Span 用等长参数数组 unnest，一条 SQL 写入，span_index 保留顺序；重复 Span ID 与空列表合法。约束失败回滚整份覆盖。

读取概要与明细位于同一短只读 Repeatable Read 事务，避免两个 SELECT 间覆盖/删除导致旧概要配新/空明细。真实两个连接的事件交错测试已验证；读者可返回一致旧版本，不承诺响应时最新，也不阻塞写者提交。

## 显式初始化

demo 的 init-db 是独立命令，仅接受 postgres:5432/observable_agent_demo。两份既有 schema 在一个事务创建，事务级 advisory lock 串行化合作的初始化器，SET LOCAL search_path=public 固定建表位置。空库初始化，恰好五张 public 表时只返回 already_initialized，部分/其他状态拒绝。

这只检查表集合，不是迁移器或任意历史结构兼容性验证。原应用不自动建表，原开发库/卷不被引用。

## 容器与数据

正式镜像固定基础摘要和 33 个 Linux 运行包；UID/GID 10001、只读根文件系统、PDF 卷与临时目录可写。构建上下文默认拒绝，明确 COPY 源码/SQL/锁文件和初始化脚本，不含密钥、论文、Git、测试或私人日志。运行环境变量对 Docker 管理员仍可见。

Compose 只发布本机 API，数据库不发布端口，Ollama 复用宿主机。PG 与 PDF 使用两个命名卷，重建容器不删除卷。unless-stopped 不等于 unhealthy 自动重启，depends_on 只处理启动等待；运行故障由超时/错误/就绪机制处理。

最新镜像已真实验证合成 PDF→Ollama 1024维→数据库与 Trace；PG 停启后原 app 恢复；两容器重建后 PDF hash/向量/Trace 保留且可再次检索。没有调用 DeepSeek，历史真实问答与当前语义限制另记。

部署命令见[指南](deployment.md)，成绩与未验证范围见[验证](validation.md)。
