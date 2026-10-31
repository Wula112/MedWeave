from contextlib import aclosing, asynccontextmanager
import asyncio
import inspect
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Scope, Receive, Send
from anyio import CancelScope
from pydantic import BaseModel, Field
import hashlib
import json
import os
from typing import Any
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END
from agents.handoff import HANDOFF_TOOL_LABELS
from agents.llm import LLMConfigurationError, resolve_model_settings
from agents.vision import (
    SCAN_TYPE_TO_AGENT,
    VisionInputError,
    compose_agent_message,
    normalize_scan_type,
    recognize_image,
    validate_image_upload,
)
from observability import configure_observability
from voice.gateway import GraphBridge, StarletteTransport, serve_voice_call
from voice.protocol import error_frame
from voice.providers import SpeechConfigurationError, load_speech_providers
from runtime.config import get_settings
from runtime.context import RunContext, current_run
from runtime.errors import RuntimeFailure
from runtime.events import iter_graph_events
from runtime.events import split_stream_chunk as _split_stream_chunk, interrupt_question as _interrupt_question
from runtime.runs import RunManager, request_fingerprint

load_dotenv()

_observability_runtime = configure_observability()
if _observability_runtime.provider != "none":
    print(
        f"--- [Observability] {_observability_runtime.provider} tracing enabled ---",
        flush=True,
    )


async def _close_checkpointer() -> None:
    """Release the checkpointer's backing connection (sqlite) on shutdown."""
    connection = getattr(getattr(_master_app, "checkpointer", None), "conn", None)
    close = getattr(connection, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # pragma: no cover - shutdown best effort
        print(f"--- [API] Checkpointer close failed: {type(exc).__name__} ---", flush=True)


async def _warm_knowledge_base() -> None:
    """Load the RAG index at startup so the first clinic conclusion doesn't pay for it."""
    import time

    started = time.perf_counter()
    try:
        from rag.knowledge_base import get_knowledge_base

        await get_knowledge_base().warm()
    except Exception as exc:  # never block serving on RAG
        print(f"--- [RAG] Warm-up failed ({type(exc).__name__}); will retry lazily ---", flush=True)
        return
    print(f"--- [RAG] Knowledge base warmed in {time.perf_counter() - started:.1f}s ---", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _runtime, _master_app
    # Shared graph state is mandatory for the production multi-worker runtime.
    if os.getenv("CHECKPOINTER", "sqlite").strip().lower() not in {"sqlite", "aiosqlite", ""}:
        raise RuntimeError("The API runtime requires CHECKPOINTER=sqlite for shared worker state")
    manager = get_runtime()
    await manager.recover_expired()
    await get_master_app().checkpointer.setup()
    # In the background: serving starts immediately; a request that needs RAG
    # before warm-up finishes simply waits on the same init lock.
    warmup = asyncio.create_task(_warm_knowledge_base())
    try:
        yield
    finally:
        warmup.cancel()
        await asyncio.gather(warmup, return_exceptions=True)
        if await manager.shutdown():
            await _close_checkpointer()
            _runtime = None
            _master_app = None
        _observability_runtime.shutdown()


app = FastAPI(title="大健康 AI 后端", version="0.5.0", lifespan=lifespan)

# Vite dev servers. Shared by CORS and the /api/voice WebSocket Origin check
# (CORS does not apply to WebSocket handshakes).
ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    # Comma-separated extras, e.g. a LAN address when testing on a phone.
    *(o.strip() for o in os.getenv("EXTRA_ALLOWED_ORIGINS", "").split(",") if o.strip()),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the graph to speed up startup
_master_app = None
_runtime: RunManager | None = None


def get_runtime() -> RunManager:
    global _runtime
    if _runtime is None:
        _runtime = RunManager()
    return _runtime

def get_master_app():
    global _master_app
    if _master_app is None:
        from agents.graph import build_graph
        from agents.checkpointing import create_checkpointer
        data_dir = get_settings().data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        settings = {**os.environ, "CHECKPOINTER": "sqlite",
                    "CHECKPOINT_DB_PATH": os.getenv("CHECKPOINT_DB_PATH") or str(data_dir / "checkpoints.sqlite3")}
        _master_app = build_graph(checkpointer=create_checkpointer(settings))
    return _master_app


@app.exception_handler(RuntimeFailure)
async def runtime_failure_handler(_request, exc: RuntimeFailure):
    return JSONResponse(status_code=exc.http_status, content={"detail": {
        "code": exc.code, "message": exc.message, "retryable": exc.retryable, **exc.details,
    }})


# --- 工具名称映射 (Tool / Node display names) ---
_NODE_LABELS = {
    "router": "分析您的需求",
    "clinic_node": "进入预问诊模块",
    "classify_reply": "正在理解您的咨询需求…",
    "insurance_node": "进入医保咨询模块",
    "report_node": "进入报告解读模块",
    "advisor_node": "进入健康顾问模块",
    "pharmacy_node": "进入用药咨询模块",
    # clinic subgraph internals
    "emergency_gate": "正在进行急症安全筛查…",
    "extract_symptoms": "正在整理症状要点…",
    "check_sufficiency": "正在检查信息是否完整…",
    "ask_followup": "正在准备追问…",
    "conclude": "正在生成分诊建议…",
}

# --- 技能工具 → 前端展示标签 ---
_SKILL_LABELS: dict[str, str] = {
    # insurance tools
    "get_insurance_balance":    "正在查询医保余额…",
    "get_consumption_records":  "正在查询消费明细…",
    "get_payment_records":      "正在查询缴费记录…",
    "get_cross_region_info":    "正在查询异地就医信息…",
    "search_insurance_policy":  "正在检索医保政策知识库…",
    # pharmacy tools
    "search_drug_info":         "正在查询药品信息…",
    "check_drug_interaction":   "正在检查药物相互作用…",
    "find_nearby_pharmacy":     "正在查找附近药店…",
    "get_otc_recommendation":   "正在检索用药建议…",
    # skills
    "load_skill":               "正在加载技能模块…",
    "emergency_triage":         "正在进行急症安全评估…",
    "symptom_scorer":           "正在评估症状严重程度…",
    "health_calculator":        "正在计算健康指标…",
    "lab_interpreter":          "正在解读化验指标…",
    "risk_assessor":            "正在评估慢性病风险…",
    "medication_calculator":    "正在计算用药剂量…",
    # agent-to-agent handoffs
    **HANDOFF_TOOL_LABELS,
}

_MODE_TO_AGENT = {
    "clinic": "clinic_agent",
    "insurance": "insurance_agent",
    "report": "report_agent",
    "pharmacy": "pharmacy_agent",
    "general": "advisor_agent",
    "dashboard": "advisor_agent",
}


class ChatMessage(BaseModel):
    role: str
    content: str


class UserInfo(BaseModel):
    name: str = "用户"
    age: int | None = None
    medical_history: str = "无"
    elder_mode: bool = False
    region: str = ""


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_info: UserInfo | None = None
    chat_mode: str = "general"
    # Server-side conversation id. When supplied, the checkpointer holds the
    # history and only the newest user message is sent to the graph.
    thread_id: str | None = None
    request_id: str | None = Field(default=None, min_length=1, max_length=128)


def _sse_payload(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_lc_messages(messages: list[ChatMessage]) -> list[HumanMessage | AIMessage]:
    return [
        HumanMessage(content=msg.content)
        if msg.role == "user"
        else AIMessage(content=msg.content)
        for msg in messages
        if msg.role in ("user", "assistant") and msg.content.strip()
    ]


def _build_initial_state(
    messages: list[ChatMessage],
    user_info: UserInfo | None,
    active_agent: str,
    channel: str = "text",
) -> dict:
    user_info_dict = user_info.model_dump() if user_info else {}
    return {
        "messages": _build_lc_messages(messages)[-10:],
        "user_info": user_info_dict,
        "next_agent": "",
        "active_agent": active_agent,
        "handoff_count": 0,
        "channel": channel,
    }


def _parse_user_info_json(raw: str) -> UserInfo:
    try:
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError("user_info must be an object")
        return UserInfo(**data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="user_info 格式错误，请检查输入。") from None


def _parse_messages_json(raw: str) -> list[ChatMessage]:
    try:
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            raise ValueError("messages must be a list")
        return [ChatMessage(**item) for item in data if isinstance(item, dict)]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="messages 格式错误，请检查输入。") from None


# Custom-stream payloads an agent may push straight through to the client via
# agents.streaming. They must reuse the SSE event types the frontend already
# understands — this whitelist is the only place event types are gated.
_STREAMABLE_CUSTOM_TYPES = {"card", "text"}


def _latest_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content
    return ""


async def _resolve_graph_input(
    config: dict,
    turn_state: dict,
    full_state: dict,
    user_text: str,
) -> Any:
    """
    Decide what to feed the graph for this request.

    - pending interrupt on the thread -> resume; clinic decides answer / handoff / cancel
    - known thread                    -> only the newest message
    - unknown thread                  -> the full history from the request
    """
    master_app = get_master_app()
    try:
        snapshot = await master_app.aget_state(config)
    except Exception as exc:  # pragma: no cover - checkpointer backend dependent
        print(f"--- [API] Unable to read thread state: {type(exc).__name__} ---", flush=True)
        raise RuntimeFailure("checkpoint_unavailable", "会话状态读取失败，请稍后重试。", retryable=True) from None

    pending = [
        interrupt
        for task in getattr(snapshot, "tasks", ())
        for interrupt in getattr(task, "interrupts", ())
    ]
    if pending:
        print("--- [API] Resuming a pending interrupt ---", flush=True)
        return Command(resume=user_text)

    if getattr(snapshot, "values", None):
        return turn_state
    return full_state


async def _iter_agent_events(
    graph_input: Any,
    config: dict | None = None,
    *,
    custom_types: frozenset[str] | set[str] | None = None,
):
    """Observe the graph; errors propagate to the owning runtime worker."""
    async with aclosing(iter_graph_events(
        get_master_app(), graph_input, config, node_labels=_NODE_LABELS,
        tool_labels=_SKILL_LABELS,
        custom_types=_STREAMABLE_CUSTOM_TYPES if custom_types is None else custom_types,
    )) as events:
        async for event in events:
            yield event


async def _run_graph(graph_input: Any, config: dict, *, custom_types=None):
    context = current_run.get()
    config = {**config, "recursion_limit": get_settings().recursion_limit}
    if context:
        await context.begin_graph()
        config["metadata"] = {"run_id": context.run_id, "thread_id": context.thread_id,
                              "channel": context.channel}
    async with aclosing(_iter_agent_events(graph_input, config, custom_types=custom_types)) as events:
        async for event in events:
            yield event


async def _cleanup_thread(thread_id: str) -> None:
    await _abandon_pending_run({"configurable": {"thread_id": thread_id}})


async def _run_sse(handle):
    async with aclosing(handle.events()) as events:
        async for event in events:
            if event.get("type") == "interrupt":
                event = {key: value for key, value in event.items() if key != "kind"}
            yield _sse_payload(event)


class RunStreamingResponse(StreamingResponse):
    """Close the run even when sending headers/body fails before iteration."""
    def __init__(self, handle) -> None:
        self.handle = handle
        super().__init__(_run_sse(handle), media_type="text/event-stream",
                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette's background callback is skipped on ASGI 2.4 send errors.
            with CancelScope(shield=True):
                await self.handle.cancel("client_disconnected")
                await self.body_iterator.aclose()


def _run_response(handle) -> StreamingResponse:
    return RunStreamingResponse(handle)


async def _stream_agent_events(graph_input: Any, config: dict | None = None):
    """Unmanaged SSE adapter for graph unit tests and local diagnostic scripts."""
    async for event in _iter_agent_events(graph_input, config):
        if event.get("type") == "interrupt" and "kind" in event:
            event = {key: value for key, value in event.items() if key != "kind"}
        yield _sse_payload(event)


@app.get("/")
async def root():
    return {"message": "大健康 AI 后端 v0.5 (LangGraph)"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        resolve_model_settings()
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=f"大模型配置错误：{exc}") from exc
    if not _latest_user_text(request.messages):
        raise HTTPException(status_code=422, detail="请提供非空用户消息")
    active_agent = _MODE_TO_AGENT.get(request.chat_mode, "advisor_agent")
    thread_id = (request.thread_id or "").strip() or uuid4().hex
    full_state = _build_initial_state(request.messages, request.user_info, active_agent)
    newest_user = next(message for message in reversed(request.messages) if message.role == "user" and message.content.strip())
    turn_state = _build_initial_state([newest_user], request.user_info, active_agent)
    user_text = _latest_user_text(request.messages)
    manager = get_runtime()

    async def source(context: RunContext):
        config = {"configurable": {"thread_id": context.thread_id, "channel": "text"}}
        graph_input = await _resolve_graph_input(config, turn_state, full_state, user_text)
        if graph_input is None:
            await manager.move_thread(context, uuid4().hex)
            config["configurable"]["thread_id"] = context.thread_id
            graph_input = full_state
        yield {"type": "session", "thread_id": context.thread_id}
        async with aclosing(_run_graph(graph_input, config)) as events:
            async for event in events:
                yield event

    handle = await manager.start(thread_id=thread_id, request_id=request.request_id,
                                 fingerprint=request_fingerprint(request.model_dump()),
                                 channel="text", source=source, cleanup=_cleanup_thread)
    return _run_response(handle)


@app.post("/api/vision-chat")
async def vision_chat(
    file: UploadFile = File(...),
    scan_type: str = Form(...),
    user_info: str = Form("{}"),
    messages: str = Form("[]"),
    thread_id: str = Form(""),
    request_id: str = Form("", max_length=128),
):
    try:
        normalized_scan_type = normalize_scan_type(scan_type)
    except VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    parsed_user_info = _parse_user_info_json(user_info)
    parsed_messages = _parse_messages_json(messages)
    try:
        image_bytes = await file.read(8 * 1024 * 1024 + 1)
        validate_image_upload(file.content_type, len(image_bytes))
    except VisionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
    try:
        resolve_model_settings()
        resolve_model_settings(purpose="vision")
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=f"大模型配置错误：{exc}") from exc
    resolved_thread_id = thread_id.strip() or uuid4().hex

    async def source(context: RunContext):
        config = {"configurable": {"thread_id": context.thread_id, "channel": "text"}}
        yield {"type": "session", "thread_id": context.thread_id}
        yield {"type": "tool_start", "tool": "vision_model", "content": "正在识别图片内容…"}
        vision_text = await recognize_image(image_bytes, file.content_type or "", normalized_scan_type)
        yield {"type": "tool_end", "tool": "vision_model"}
        injected_message = compose_agent_message(normalized_scan_type, vision_text)
        latest_message = ChatMessage(role="user", content=injected_message)
        vision_messages = [*parsed_messages[-9:], latest_message]
        active_agent = SCAN_TYPE_TO_AGENT[normalized_scan_type]
        full_state = _build_initial_state(vision_messages, parsed_user_info, active_agent)
        turn_state = _build_initial_state([latest_message], parsed_user_info, active_agent)
        graph_input = await _resolve_graph_input(config, turn_state, full_state, injected_message)
        async with aclosing(_run_graph(graph_input or full_state, config)) as events:
            async for event in events:
                yield event

    handle = await get_runtime().start(
        thread_id=resolved_thread_id, request_id=request_id or None,
        fingerprint=request_fingerprint({"thread_id": thread_id, "scan_type": normalized_scan_type,
            "user_info": parsed_user_info.model_dump(), "messages": [m.model_dump() for m in parsed_messages],
            "image_digest": hashlib.sha256(image_bytes).hexdigest(), "content_type": file.content_type}),
        channel="vision", source=source, cleanup=_cleanup_thread, timeout=get_settings().vision_timeout)
    return _run_response(handle)


@app.get("/api/threads/{thread_id}/runs/{run_id}")
async def run_status(thread_id: str, run_id: str):
    await get_runtime().recover_expired()
    row = await asyncio.to_thread(get_runtime().store.get, run_id)
    if row is None or row["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="未找到该次运行")
    return {key: row[key] for key in ("run_id", "thread_id", "status", "channel", "stop_reason", "cancel_requested")}


@app.post("/api/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str):
    await get_runtime().recover_expired()
    return await get_runtime().cancel(run_id, thread_id)


# ─── /api/voice: real-time voice clinic ──────────────────────────────────────

_speech_providers = None


def _get_speech_providers():
    """Resolve ASR/TTS once (cue audio is cached per provider instance)."""
    global _speech_providers
    if _speech_providers is None:
        _speech_providers = load_speech_providers()
    return _speech_providers


def _build_voice_state(text: str, user_info: dict, channel: str) -> dict:
    try:
        parsed = UserInfo(**(user_info or {}))
    except Exception:
        parsed = UserInfo()
    return _build_initial_state(
        [ChatMessage(role="user", content=text)], parsed, "clinic_agent", channel=channel
    )


async def _get_thread_state(config: dict):
    thread_id = config["configurable"]["thread_id"]
    return await get_runtime().read_thread(
        thread_id, lambda: get_master_app().aget_state(config, subgraphs=True), _cleanup_thread)


async def _voice_run_events(text: str, thread_id: str, user_info: dict, timeout: float):
    from voice.runs import VOICE_CUSTOM_TYPES

    async def source(context: RunContext):
        config = {"configurable": {"thread_id": thread_id, "channel": "voice"}}
        state = _build_voice_state(text, user_info, "voice")
        graph_input = await _resolve_graph_input(config, state, state, text)
        async with aclosing(_run_graph(graph_input if graph_input is not None else state,
                                      config, custom_types=VOICE_CUSTOM_TYPES)) as events:
            async for event in events:
                yield event

    handle = await get_runtime().start(
        thread_id=thread_id, request_id=uuid4().hex,
        fingerprint=request_fingerprint({"text": text, "thread_id": thread_id, "user_info": user_info}),
        channel="voice", source=source, cleanup=_cleanup_thread, timeout=timeout)
    async with aclosing(handle.events()) as events:
        async for event in events:
            # Voice's versioned wire frames retain their existing field contract.
            yield {key: value for key, value in event.items() if key not in {"run_id", "status", "code", "retryable"}}


async def _abandon_pending_run(config: dict) -> None:
    """
    Close out a run the voice gateway had to cancel (stall / hard cap).

    A cancelled run leaves its step pending in the checkpoint — typically the
    clinic subgraph still "waiting" on the interrupt it was resuming, with the
    old answer already bound to it. Resuming that would silently drop the
    user's next words, so the pending node is marked finished. What the
    subgraph had already produced this turn (its ``turn_messages``: follow-up
    questions, answers, the read-back) is written to the parent history with
    it, so the next message starts a normal turn over the full transcript and
    the interview continues where it stopped.
    """
    app = get_master_app()
    snapshot = await app.aget_state(config, subgraphs=True)
    for task in getattr(snapshot, "tasks", ()) or ():
        sub_state = getattr(task, "state", None)
        sub_values = getattr(sub_state, "values", None)
        kept = []
        parent_id = (getattr(snapshot, "config", None) or {}).get("configurable", {}).get("checkpoint_id")
        child_parent_id = (getattr(sub_state, "metadata", None) or {}).get("parents", {}).get("")
        stale_child = bool(parent_id and child_parent_id and parent_id != child_parent_id)
        # A stateful child may still expose the previous invocation, or have
        # received the full parent history without running its entry node yet.
        before_entry = "enter_interview" in (getattr(sub_state, "next", ()) or ())
        if isinstance(sub_values, dict) and not stale_child and not before_entry:
            kept = list(sub_values.get("turn_messages") or [])
            if not kept and sub_values.get("interview_status") == "completed":
                # Completion may have reset the private buffer before the
                # parent consumed the subgraph's final output.
                kept = list(sub_values.get("messages") or [])
        await app.aupdate_state(config, {"messages": kept}, as_node=task.name)
    # Completing a router task can schedule its conditional successor. Explicitly
    # clear pending work after preserving Q&A so a later input starts a fresh turn.
    if getattr(snapshot, "tasks", ()):
        await app.aupdate_state(config, None, as_node=END)


def _voice_bridge() -> GraphBridge:
    # Late-bound lambdas so tests can patch the module-level functions.
    return GraphBridge(
        resolve_input=lambda *args: _resolve_graph_input(*args),
        iter_events=lambda *args, **kwargs: _iter_agent_events(*args, **kwargs),
        build_state=lambda *args: _build_voice_state(*args),
        get_state=lambda config: _get_thread_state(config),
        check_config=lambda: resolve_model_settings(),
        abandon_run=lambda config: _abandon_pending_run(config),
        run_stream=_voice_run_events,
    )


def _origin_allowed(origin: str | None) -> bool:
    return origin in ALLOWED_ORIGINS


@app.websocket("/api/voice")
async def voice(websocket: WebSocket):
    # CORS does not cover WebSocket handshakes: check Origin ourselves.
    # Closing before accept() makes the server reject the handshake (HTTP 403).
    origin = websocket.headers.get("origin")
    if not _origin_allowed(origin):
        print(f"--- [Voice] Rejected WebSocket origin: {origin!r} ---", flush=True)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        asr, tts = _get_speech_providers()
    except SpeechConfigurationError as exc:
        await websocket.send_text(
            json.dumps(error_frame("fatal", f"语音服务配置错误：{exc}"), ensure_ascii=False)
        )
        await websocket.close(code=1011)
        return

    await serve_voice_call(
        StarletteTransport(websocket),
        bridge=_voice_bridge(),
        asr=asr,
        tts=tts,
    )
