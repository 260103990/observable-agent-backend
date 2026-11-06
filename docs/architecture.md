# 架构与工程取舍

## 请求与模块

FastAPI路由负责HTTP校验，依赖组装服务与请求所有资源，业务服务不直接读取HTTP请求。原单步`AgentRunner`保留；新`AgentLoopRunner`处理受控多轮工具协议。

| 模块 | 职责 |
|---|---|
| `app/api/routes/` | chat、legacy agent、loop、papers、Trace接口 |
| `app/agent/loop_runner.py` | 模型轮次、严格call ID、预算、顺序工具执行、最终校验/有界修复 |
| `app/tools/async_registry.py` | 参数模型、执行超时、结果预算、只读显式瞬时错误有限重试 |
| `app/rag/service.py` | 导入/问答/纯检索、并发准入、模型身份核对、阶段Trace |
| `app/rag/store.py` | 原子导入、来源恢复、单篇余弦与可选混合检索 |
| `app/tools/web_search.py` | 有界HTTP Search、校验、缓存及并发合并接入 |
| `app/observability/` | Trace/Span模型、内存/PG Store |
| `app/evaluation/` | 隔离评估、显式观察完整性和分层比较 |

## RAG数据流与表关系

`PDF → 页面原始提取文本 → 保守清洗 → 同页Chunk → Ollama文档向量 → 原子保存`。

`问题 → 问题向量 → 核对索引身份 → 单篇候选 → 有界证据 → DeepSeek → 引用成员校验 → 回答/不足状态`。

- `rag_papers`：论文主键、来源hash、处理版本、Embedding配置。
- `rag_pages`：`(paper_id,page_number)`主键，原始提取文本与规范化文本。
- `rag_chunks`：`(paper_id,page_number,chunk_index)`主键，文本/字符范围/vector(1024)，复合外键指向页面。
- `traces`与`spans`：执行记录，业务数据和调用观测分开存储。

元数据、页面和全部Chunk/向量在同一事务保存，失败回滚；论文ID重复拒绝覆盖。存储前验证规范化和切分能从声明版本重现，不能仅靠字段长度一致伪造来源。

原检索用`WHERE paper_id=$1`限定范围，`<=>`做余弦距离排序并以页/块稳定打破平局。profile验证与查询使用同一只读repeatable_read快照。可选hybrid各取有界向量/PG全文候选，以RRF融合；`RetrievedChunk.score`仍为余弦，不是融合分数。

## Agent执行与证据

默认最多5轮、8工具、每轮4调用；模型120秒、工具60秒、业务240秒；历史64000字符、单工具结果32000字符。字符不是token，也不是内存字节。

工具可以报告安全错误，模型在剩余预算内修正参数。工具失败不伪造成功；有副作用的工具不能自动重试。重复call ID整批拒绝。最终格式/引用错误最多一次修复，使用原轮数和时间预算，修复阶段不能调用工具。

论文E、Web W证据ID按请求范围管理，跨轮稳定。PreparedToolResult在执行器确认结果预算后才提交本地账本，避免授权模型未实际看到的引用。模型历史中的完全相同证据可以引用较晚完整副本；原始来源与最终校验不变，不做有损摘要或URL级合并。

## 资源、并发和失败

FastAPI lifespan拥有asyncpg pool；RAG借用pool与共享HTTP客户端，Loop拥有自己的请求客户端。关闭/失败/取消路径释放资源。CPU解析与快照工作放在线程；取消不抢先关闭尚在使用的文件句柄。

导入设置上传/页数/字符/Chunk/并发预算。保存尝试后的不确定提交可能保留自有PDF，避免误删有效来源，Trace记录该状态；不是宣称已实现后台孤立文件回收。

Trace写入有独立清理预算，失败不覆盖原业务结果。`ok`是执行/结构状态，不是答案正确性的标签。

## Search缓存的职责

缓存key覆盖提供方/接口、规范化后的完整请求、凭据摘要、响应预算和契约版本。只去首尾空格，不擅自大小写/语义合并。缓存存序列化校验快照，不存HTTP客户端或明文密钥。

TTL非滑动、LRU和序列化容量预算控制寿命/空间；失败与空结果不长期缓存。命中时当前调用保有当前Trace，原获取时间与来源Trace可回溯，当前网络成本记0。过期后刷新失败不返回过期内容。

并发合并采用请求内owner与受保护通知：取消一个waiter不取消其他请求；owner失败/取消通知等待者，不隐式重试接管。队列有key/waiter上限，超限旁路，是缓存优化而非外部请求全局限流。

缓存本体和合并范围是单事件循环/单进程，尚无Redis或跨进程一致性。下一步基础设施应由真实并发/部署需求驱动。
