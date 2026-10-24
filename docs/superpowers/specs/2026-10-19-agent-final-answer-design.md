# 工具结果后的模型总结

## 目标

一次请求固定执行：模型选择工具 → 执行工具 → 模型生成最终回答。

## HTTP 响应

- 保留 result 字段，类型仍为整数，表示工具原始结果。
- 新增 answer 字段，类型为字符串，表示模型最终回答。
- 保留已有 request_id、trace_id 和响应头的关联规则。

## 组件分工

- ToolDecision 保存 ToolCall 和第一轮模型返回的完整 assistant 消息。
- 总结组件接收用户问题、ToolDecision 和工具结果，异步生成回答。
- Runner 负责编排执行，返回包含 result 和 answer 的 AgentRunResult。
- 路由负责将执行结果转换为 HTTP 响应。

## 第二轮模型请求

- messages 顺序为 user、第一轮完整 assistant 消息、tool。
- tool 消息包含 tool_name，content 为工具结果的 JSON 字符串。
- 保留原始 assistant 消息中的字段。
- 使用 stream=false，第二轮不提供工具列表。
- 第二轮再次返回工具调用时，视为无效总结。

## Trace 与异常

- 成功执行包含 agent.run、model.decision、tool.execute、model.summary。
- 三个阶段 Span 都属于 agent.run。
- 总结失败时，之前成功的阶段仍为 ok，总结阶段和根 Span 为 error。
- 网络异常沿用已有的 502、503、504 映射。
- 缺失、非字符串或空白答案，以及再次工具调用，返回 502。
- Trace 保存失败只记录警告，不覆盖业务结果或原始异常。

## 测试验收

- 第二轮请求保留完整消息上下文。
- 总结组件能解析有效回答，并拒绝无效回答。
- Runner 将真实工具结果交给总结组件。
- HTTP 响应同时包含 result 和 answer。
- 成功与失败 Trace 正确记录总结阶段。
- 所有自动化测试离线运行，使用测试替身和 MockTransport。
