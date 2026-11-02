# Mem0：本轮对话记忆提取

首版在每个 Run 正常结束或进入等待用户回答时，将这一轮的用户输入和助手文本交给 Mem0 Platform 异步提取。记忆的检索、注入和管理另行实现。

## 接入位置

外层业务图由 `StateGraph` 编排；一次 Run 可以经历问诊 `interrupt`、纯规则回答，或多个专家交接。LangChain `after_agent` 是单个 `create_agent` 的循环结束钩子，不能代表外层 Run 一定结束。LangGraph 的中断节点在恢复时还会从节点开头重新执行。

因此，`runtime/runs.py` 提供 Run 生命周期钩子，`memory/middleware.py` 实现记忆中间件。它在 Run 接收文本事件时收集本轮回答，Run 成功终态写入并完成收尾后，提交一个后台任务：

```text
Run 接收最新用户输入
  → LangGraph 执行并流式输出本轮助手文本
  → success / waiting_input
  → MemoryMiddleware.after_run
  → 后台调用 Mem0 SDK add
```

`waiting_input` 是当前 Run 的终态；用户回答后会创建新 Run。取消、超时和失败不会提交不完整的助手输出。未产生助手文本的 Run 会跳过。用户在同一轮切换专家时，以最终输出的专家作为 `agent_id`。

这个钩子属于项目的 Run 中间件，使用 LangGraph 的流式事件和执行终态；它不是把每个内部 Agent 的 `after_agent` 当作整个父图的结束。参考 [LangChain 中间件执行顺序](https://docs.langchain.com/oss/python/langchain/middleware/custom) 和 [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)。

## 请求参数和时间

调用 Mem0 Platform 的 `MemoryClient.add`：

- `messages`：最新一轮的用户输入及用户实际收到的助手文本，不上传旧对话、工具消息或原始图片。
- `user_id`：`MEM0_USER_ID`，当前默认 `smart_health_user`；接入账号后应替换为服务端认证用户标识。
- `agent_id`：最终响应专家的项目标识，例如 `clinic_agent`、`insurance_agent`。
- `custom_instructions`：`memory/prompts.py` 中的医疗提取提示词，末尾追加本轮对话的中国时间。
- `timezone` 和 `observation_datetime`：使用 `Asia/Shanghai` 和 Run 开始时的带时区时间，后台排队不会改变对话日期。

当前不额外传入 `app_id`、Mem0 的 `run_id` 或 metadata。文本、语音和图片的入口都使用同一捕获链路；图片模式发送本轮文字及解读答复，不发送图像字节或 base64。视觉识别结果不是用户亲口陈述。

提示词按单次请求传入，不修改 Mem0 项目级配置。Mem0 的文档说明：若项目另行设置了 `agent_custom_instructions`，同时传入 `user_id` 与 `agent_id` 时，助手归属的记忆可能使用那套项目级规则；当前代码只按要求传入 `custom_instructions`。[Mem0 自定义提示词](https://docs.mem0.ai/platform/features/custom-instructions)、[Add API 参数](https://docs.mem0.ai/api-reference/memory/add-memories)

## 配置

```dotenv
MEM0_API_KEY=your_mem0_api_key
MEM0_USER_ID=smart_health_user
MEM0_TELEMETRY=false
```

在 `backend/.env` 中配置；`.env.example` 保持空密钥。空密钥表示不启用记忆提交。SDK 初始化文件默认放在 `RUNTIME_DATA_DIR/mem0`；启用 `memory.middleware` 的 INFO 日志后可记录 Run ID 和有效的 Mem0 事件 ID，失败时记录异常类型，不输出对话、密钥或服务端错误正文。

当前没有用户登录，不同使用者共用固定 `user_id`；因此该配置只适用于当前单用户/演示场景。每轮文字会发送到 Mem0 Platform。

## 异步边界

后台任务不会阻塞当前回答。服务正常关闭时会有限等待任务；进程突然退出可能丢失尚未发送的任务。为避免在请求超时但服务端已接收时重复新增，首版不自动重试 `add`。

Mem0 V3 的成功响应通常只表示服务端已接受异步提取，返回 `PENDING` 和事件 ID，不等于记忆已经完成写入。首版只提交，不主动检索或轮询结果。[Mem0 Add API](https://docs.mem0.ai/api-reference/memory/add-memories)

验证：

```powershell
cd backend
uv sync
uv run python -m unittest test_memory.py
```

测试使用模拟传输，不发送真实健康数据；覆盖官方 SDK 请求参数、中国时间、成功/等待/取消、后台失败隔离、问诊切换恢复和图片输入。
