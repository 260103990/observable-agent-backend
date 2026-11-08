# 验证层次与发布边界

2026-11-08，V4 首版发布副本的实际验证：

| 层次 | 结果 | 范围 |
|---|---|---|
| Windows/Python3.11.7 默认 | 755 passed, 5 skipped, 58 deselected, 1 warning in 28.42s | 单元与受控ASGI/HTTP，不是外部模型验证 |
| 专用真实测试数据库 | 52 passed, 6 skipped, 760 deselected, 1 warning in 12.19s | PostgreSQL/pgvector、事务、快照、故障恢复；模型可为替身 |
| 最新Docker/Linux镜像 | 本地真实通过 | 合成1页PDF、宿主机Ollama1024维、Trace7Span、重建持久化、PG停启恢复 |
| GitHub CI | Linux/Windows Python3.11 | 实际状态见仓库Actions；不执行收费模型/数据库测试 |

skip/deselected不算通过。既有TestClient弃用warning保留，没有为消除warning改变HTTP库契约。公开源码与本地开发代码逐文件核对Python AST，导出只规范化换行/EOF；真实部署所用镜像文件又与该发布源码逐字节核对。

## 当前镜像实测

- 镜像ID `sha256:f53791d6befd6a5b9fe463d52bdb103cecfcee823158dee3565809db13efa0b4`，Python 3.11.18，UID10001、33项运行锁版本匹配、无pytest或运行密钥打包。
- 独立随机Compose project，API只绑定本机；init-db首次/重复执行符合约定。
- 真实Ollama导入合成1页/1Chunk，1024维；新批量Trace保存和快照读取在真实API链路执行。
- 停止本测试PG时ready503/health200，恢复PG后原app无需重启即ready200。
- app/PG两容器重建后旧Trace内容一致、PDFhash与索引匹配；再次真实Embedding检索命中原页。
- app/PG正常退出码均0，测试容器与网络移除、临时端口关闭，新测试卷保留；旧开发容器/数据未迁移。

本轮DeepSeek调用为0；早期RAG问答、Agent/Search和缓存真实样本仍按历史范围记录，不冒充本轮新外部生成验证。

## 研究与性能

Search冷/热和Agent样本见[实验](evaluation.md)，Trace写入ABBA3200次全通过及控制组结果也在该文档。它们不是生产容量、端到端通用加速或问答准确率。

## 尚未宣称完成

任意PDF布局/OCR/图表理解、所有答案语义正确、独立质量盲评、Redis共享/跨进程合并/语义缓存、网页Fetch、公网认证/TLS、高可用/自动备份、生产压测。回答聚焦实验默认关闭，真实结果未证明稳定质量收益。

V4首版范围是可复现本地部署、基础故障恢复、并发一致性与有证据的Store优化，后续需求不追溯增加为该首版的完成门槛。
