# Agent Runtime：执行控制与异常处理

三个入口（文字、图片、语音）共用 `backend/runtime/`。Agent 负责业务决策，
LangChain 中间件管理模型和工具的执行策略，Runtime 负责一次 Run 的准入、
总时限、归属和收尾，Checkpointer 保存对话状态。

## 多 Worker 部署

首版支持**同一主机、本地磁盘**上的多个 Uvicorn Worker，不需要粘性会话。
所有 Worker 必须使用同一份 `RUNTIME_DATA_DIR` 和 checkpoint 文件。

```powershell
cd backend
uv sync
# 已有 .env 中的模型配置继续生效；API 默认使用共享 SQLite checkpoint。
$env:CHECKPOINTER = "sqlite"
$env:RUNTIME_DATA_DIR = "D:/JavaProject/Smart-Health-Assistant/backend/.runtime"
uv run uvicorn main:app --workers 2 --host 127.0.0.1 --port 8000
```

开发时可以继续使用 `--reload`，它与 `--workers` 分开使用。
API 启动时拒绝内存 Checkpointer；直接运行图的单元测试、评测脚本仍可使用 MemorySaver。
首次切换后，原有内存会话不会自动迁移；如已有 SQLite 会话，请设置
`CHECKPOINT_DB_PATH` 指向原文件。

默认目录包含 `runs.sqlite3`、`checkpoints.sqlite3` 和 `locks/`，已经加入 gitignore。
运行数据库只记录随机标识、请求摘要、枚举状态和时间，不保存原始消息或图片；
checkpoint 仍包含正常对话状态，应按应用数据管理。

这套实现不支持把 SQLite / 文件锁放到 NFS、SMB 等网络文件系统，也不提供多主机 HA。
多主机需要替换 RunStore、checkpoint 和执行隔离机制。当前 SSE 由执行所属 Worker
提供，尚不提供跨进程 token 回放；断开文字流会取消执行，重新提交同一 request_id
会收到原 Run 标识及状态，不会自动重跑。语音断线维持原有后台完成策略。

## 目录职责

| 模块 | 职责 |
|---|---|
| `runtime/config.py` | 校验环境配置、执行预算 |
| `runtime/store.py` | SQLite 事务、跨进程唯一准入、归属、取消请求、文件锁 |
| `runtime/runs.py` | 后台任务、事件队列、心跳、取消、收尾、失联恢复 |
| `runtime/context.py` | 单次 Run 的临时计数、结果、deadline，隔离于业务 State |
| `runtime/errors.py` | 错误码、安全提示、可重试分类 |
| `runtime/events.py` | LangGraph 事件转换，保持文本/卡片/步骤协议 |
| `agents/factory.py` | 使用原生 `create_agent` 统一组装专家与模型调用 |
| `agents/middlewares/__init__.py` | 中间件链的唯一组装入口 |
| `agents/middlewares/execution_context.py` | 通过 `Runtime.context` 注入本轮执行依赖 |
| `agents/middlewares/llm_error_handling.py` | 模型尝试时限、异常分类、退避、部分流保护 |
| `agents/middlewares/tool_error_handling.py` | 工具时限、安全的错误 ToolMessage |
| `agents/middlewares/tool_deduplication.py` | 只读工具 single-flight、成功复用、失败记忆 |
| `agents/middlewares/loop_detection.py` | 请求次数统计、重复告警、条件终止 |
| `agents/tool_policy.py` | 业务只读工具声明与技能身份规范化 |

## 原生中间件组装

代码组织借鉴 DeerFlow 的 `agents/middlewares/` 和 Lead Agent 组装入口。
每个策略继承 LangChain `AgentMiddleware`，在公开钩子中处理自己的边界：

```text
API → RunManager → 现有业务图（路由 / 问诊 / 专家交接）
                         ↓
                 agents/factory.py → create_agent
                         ├─ awrap_model_call → 模型时限 / 重试
                         ├─ awrap_tool_call  → 次数 → 去重 → 时限 / 错误 → 工具
                         └─ abefore_model    → 限制触发后 jump_to="end"
```

五个专家统一使用 `create_specialist_agent`；路由、问诊信息提取、结论摘要、
图片识别通过 `invoke_model` 进入只含模型中间件的原生 Agent。结构化调用显式
使用 `ProviderStrategy`，保持提供商 JSON Schema 输出，由业务节点处理语义校验。

```python
from agents.factory import create_specialist_agent, get_agent_context

agent = create_specialist_agent(llm, tools=tools, system_prompt=system)
result = await agent.ainvoke({"messages": messages}, context=get_agent_context())
```

工厂声明 `context_schema=RunContext`。一次托管 Run 内，不同专家共享同一个
`RunContext` 的预算和缓存；独立调用使用新 context。任务、缓存、deadline 和
资源关闭回调通过 `Runtime.context` 传递，不放入消息、State 或 checkpoint。
已编译 Agent 可以跨轮复用，每轮调用传入新的 context。

中间件只控制调用边界。准入、跨进程取消、总时限和异常收尾仍由外层 Runtime
负责，因为图失败或取消后不能依靠 `after_agent` 保证执行清理。
模型使用标准 `ChatOpenAI`，没有重写私有模型方法或额外包一层 ToolNode。
流式进度由本次尝试的公开 Callback 观察，模型副本共享客户端而独立持有回调。

API 用法参见 [LangChain 自定义中间件](https://docs.langchain.com/oss/python/langchain/middleware/custom)。
实现兼容锁文件中的 LangChain 1.2.13；中间件顺序显式保存在 `build_middlewares`。

## Run 生命周期与并发

```text
pending → running → success / waiting_input / error / timeout / cancelled
```

SQLite `BEGIN IMMEDIATE` 和 active-thread 唯一索引完成准入；总并发上限也由共享
数据库计算。同一会话的图片识别、checkpoint 输入解析、图执行和取消后的修复
都在同一个执行保留范围中。不同会话可以并行。

每个 Run 同时持有该会话的 OS 文件锁，直到图流关闭、工具任务和 checkpoint 清理
完成。**租约过期并不证明旧进程停止**，失联恢复必须拿到文件锁并再次在事务中
检查租约。旧进程仍持锁时拒绝接管；进程崩溃由操作系统释放锁，其他 Worker 在
启动、准入、心跳或状态查询时回收过期记录，写入 `orphan_recovered`。
锁文件运行中不删除，防止出现两份同名但不同 inode 的锁。

心跳只续租仍有效、归属匹配的活跃行；续租失败至租约截止时取消本地任务。
终态写入再次检查归属、租约和取消标记，已接受的取消不能被随后到达的成功覆盖。
取消按协作方式执行；无法及时退出的本地执行继续持锁，保障后续请求不能并发写入。

`waiting_input` 是本次 Run 的终态，保留 LangGraph 的真实 interrupt，释放执行名额；
用户回答会产生新 Run，通过 `Command(resume=...)` 恢复。急症闸门、追问上限和
handoff 上限继续由现有业务图负责。

开始写图状态之前会持久标记 thread dirty。成功或正常 interrupt 清除标记；异常
结束先修复 pending 节点、保留问答，然后才释放锁。清理失败则保留 dirty，下一次
运行必须修复成功才能执行。checkpoint 读取失败明确报错，不回灌整段历史。
关闭服务时先取消并等待 Run；尚有未收尾任务时不提前关闭 checkpoint 连接。

## 请求与工具重复保护

文字 JSON 与图片表单支持 `request_id`，前端为每次逻辑提交生成 UUID，并在首次
发送前确定 thread_id。旧客户端可以省略 request_id，但没有重复提交保护。

- 同一 request_id、同一规范化请求：HTTP 409 `duplicate_request`，包含原 run_id、thread_id、status。
- 同一 request_id、不同输入：HTTP 409 `request_id_conflict`。
- 会话已被其他 Run 占用：HTTP 409 `thread_busy`。
- 全局活跃 Run 已达上限：HTTP 503 `runtime_busy`。

请求摘要覆盖模式、用户信息、消息、会话及图片摘要，不保存原图。运行记录保留
默认 24 小时，过期清理后该 request_id 可以再次使用。保护范围是保留期内的准入，
不构成任意外部操作的“恰好执行一次”保证。

同一 Run 内按规范化工具名和参数识别重复；`load_skill` 与直接技能工具共享身份。
只读工具显式列在 `agents/tool_policy.py`，或设置 `metadata.runtime_readonly=true`。
并行相同调用共享任务，成功结果可复用，复用时重建当前 tool_call_id；卡片只产生一次。
失败结果保留 error 状态，同参数再次请求返回先前错误，尤其避免超时的同步任务
尚未停止时重新启动。新一轮 Run 使用新预算和新结果。

第二次相同调用提示模型利用已有结果收尾，第三次触发 `loop_detected`；总工具请求
上限独立控制，包括命中和被拦截的请求。硬停止先生成配对的错误 ToolMessage，
随后在模型前置钩子停止，避免继续调用模型或留下未配对 tool_call。
模型包装钩子也检查 Run 的硬停止标记，阻止业务节点随后启动摘要等纯模型调用。
未知新工具默认不复用，工具不自动重试。

## 超时、重试与事件

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `RUNTIME_RUN_TIMEOUT` | 120 | 普通文字 Run 总秒数 |
| `RUNTIME_VISION_TIMEOUT` | 180 | 图片识别与推理合计秒数 |
| `RUNTIME_MODEL_TIMEOUT` | 30 | 每次模型尝试的秒数 |
| `RUNTIME_TOOL_TIMEOUT` | 10 | 每次工具执行的秒数 |
| `RUNTIME_RETRIEVAL_TIMEOUT` | 10 | 检索整体秒数，包含等待初始化 |
| `RUNTIME_CLEANUP_TIMEOUT` | 10 | 状态修复与停机等待预算 |
| `RUNTIME_MODEL_ATTEMPTS` | 3 | 模型最大尝试次数，包含第一次 |
| `RUNTIME_RETRY_BASE` / `RUNTIME_RETRY_CAP` | 0.5 / 4 | 退避起点与上限秒数 |
| `RUNTIME_LEASE_SECONDS` / `RUNTIME_HEARTBEAT_SECONDS` | 15 / 3 | 租约秒数与心跳间隔 |
| `RUNTIME_MAX_CONCURRENT_RUNS` | 8 | 所有 Worker 合计活跃 Run 上限 |
| `RUNTIME_RECURSION_LIMIT` / `RUNTIME_AGENT_RECURSION_LIMIT` | 64 / 24 | 主图与内层 ReAct 的 super-step 上限 |
| `RUNTIME_MAX_TOOL_CALLS` / `RUNTIME_REPEAT_LIMIT` | 12 / 3 | Run 工具请求总数与同参数次数阈值 |
| `RUNTIME_RETENTION_SECONDS` | 86400 | Run 和请求去重记录保留秒数 |
| `RUNTIME_EVENT_BUFFER` | 256 | 单 Run 有界实时事件缓冲 |

语音仍采用 VoiceTimings 的 20 秒无业务进展监控，图执行硬时限交给公共 Runtime，
保留默认 120 秒上限。子操作预算裁剪到 Run 剩余时间，退避也不能越过总期限。
RAG 初始化任务由每个进程共享，调用方超时只结束等待，避免重复加载同一模型。

仅对网络错误、超时、408、临时 429 和 5xx 重试。鉴权、配额耗尽、参数错误直接
失败。底层 SDK 的 max_retries 显式为 0，避免重试相乘。单次流输出任何 chunk
（含 tool-call chunk）后均不重试；节点、工具链和整张图不自动重放。
取消和 GraphBubbleUp 控制信号直接传播。托管 Run 创建的聊天与视觉模型客户端
在收尾阶段统一关闭；独立视觉调用在调用结束后关闭客户端。

普通工具异常返回 `ToolMessage(status="error")` 供模型调整。最终运行错误使用
`error{run_id,code,content,retryable}`；正常完成使用 `finish{run_id,status}`。
前端仅在 finish 后调用 onDone，error 与提前 EOF 仅进入错误终态，并保留已显示内容。
运行状态、心跳和日志不包含原始医疗输入。

状态与取消接口均以 thread_id 和 run_id 定位：

```text
GET  /api/threads/{thread_id}/runs/{run_id}
POST /api/threads/{thread_id}/runs/{run_id}/cancel
```

取消可以发给任意 Worker，所属 Worker 在下次心跳观察取消请求。状态接口不暴露
租约 owner、请求摘要或模型凭据。当前应用沿用既有会话标识访问方式。

## 确定性验证

```powershell
cd backend
uv run python -m unittest test_middlewares.py test_runtime.py test_llm.py test_vision.py test_main_config.py test_observability.py test_evals.py test_graph_build.py test_clinic_graph.py test_clinic_persistence.py test_handoff.py test_cards.py test_skill_tools.py test_voice_runs.py test_voice_clinic.py test_voice_gateway.py
cd ../frontend
node test_chat_stream.cjs
npm run lint
npm run build
node test_runtime_ui.cjs
```

测试覆盖独立进程持锁、租约过期且旧进程存活、旧进程终止后的恢复、跨 Worker
准入/去重/取消、超时和反复取消、修复失败、模型部分流、并行工具复用与消息配对。
中间件测试直接运行真实 `create_agent` 图，验证重试上限、部分流不重放、异常
ToolMessage 配对、原生条件终止以及 checkpoint 不保存执行控制对象。
这些测试不调用真实模型，不验证真实提供商可用性或生产负载容量。
