# 项目总路线与教学约定

## 项目
C:\Users\26010\Desktop\agent-study\observable-agent-backend

## 教学约定
- 中文指导，用户亲自修改项目源码、运行测试、操作数据库、提交和推送；学习记录由助手直接更新。
- 助手可以只读检查项目源码和 Git，并可直接更新 docs/learning 下的学习记录；记录维护无需用户手动粘贴。
- 每步说明目标、操作位置、新语法、实际作用和预期结果。
- 替换代码时解释具体差异和原因。
- 严格 TDD：测试红灯 → 用户确认 → 提供实现 → 用户验证绿灯。
- “1”或“OK”通常表示结果符合预期。
- 合并相关步骤，只解释新增知识，避免重复检查和讲解。

## 上下文规则
- 新聊天先读本文件和 CURRENT_TASK.md，再按需读当前相关源码。
- 不主动扫描整个项目；用户仅回复“1”时不重复读取刚检查的文件。
- 出现异常或跨模块改动时，再扩大检查范围。
- 结束学习、换聊天或模块完成时，由助手直接更新 CURRENT_TASK.md；教学约定或总路线变化时同步更新 PROJECT_PLAN.md。

## 总路线
- 已完成：FastAPI 接口、单步 Tool Calling Agent、最终回答、内存 TraceStore、SQL 基础练习、PostgreSQL TraceStore 功能与真实模型本地验收。
- 当前：PostgreSQL 模块说明文档、配置示例和重复测试清理已完成；待用户验证并自行提交当前分支。
- 已走完：数据库存取 → Span 顺序及整体覆盖 → 事务 → FastAPI 配置和连接池生命周期 → 真实 Agent 调用与服务重启持久化验收。
- 文档与 Git 收尾后，再单独确定受控 Agent Loop 的范围；Redis 为后续阶段，不同时展开。

## 必须保留的契约
- async save(trace)、async get(trace_id)；不存在返回 None，API 对应404。
- ID 是普通字符串，不要求 UUID；request_id 不要求唯一。
- Span 列表有顺序，空列表合法；同 trace_id 保存时整份覆盖。
- 保存和读取的对象相互独立。
- Trace 保存失败不覆盖业务结果或原异常。
- 默认离线测试与真实数据库测试分开。
- 连接池由应用 lifespan 管理，创建、使用、关闭应在同一事件循环。

## 已确认的最终设计
- traces 保存 Trace 主体；spans 保存 Span，并增加 trace_id、span_index。
- traces 主键为 trace_id；spans 主键为 (trace_id, span_index)。
- 按 span_index 还原列表顺序，attributes 使用 JSONB。
- span_id 不额外要求全局唯一，parent_span_id 不增加自引用外键。
- 不额外收紧现有模型允许的数据。
- 整体覆盖放在同一个事务中。

## 本地验收完成（2026-10-31）
- 用户确认数据库测试 11 passed，默认离线测试 116 passed / 11 deselected，保留既有 warning；助手未代跑测试。
- 真实 Ollama 使用 add_numbers 计算 17 + 25，返回 42 和最终回答；四个执行 Span 均为 ok。
- 用户确认服务重启后仍可查询旧 Trace e088324c-f4fa-4fab-b053-452875c4df75，PostgreSQL 持久化验收通过。
- 用户确认原始 HTTP 响应按 UTF-8 重解码后中文正常，本次属于客户端解码差异。
- Settings.database_url 可选，默认内存模式；数据库模式由 FastAPI lifespan 管理连接池，并通过请求依赖取用应用 Store。
- 当前实际运行使用应用终端的 DATABASE_URL；独立测试终端使用 TEST_DATABASE_URL。默认离线测试环境保持未配置应用数据库地址。
- 当前每次请求执行一个工具，只注册 add_numbers；多步执行和会话记忆尚未实现。
- 本次用户授权助手代完成 README.md / .env.example 和测试重复段清理；文档已同步、静态检查通过。提交与推送由用户自行执行，尚未完成。
