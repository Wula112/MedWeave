import type { ChatMessage, AgentStep, ChatMode, ChatCardPayload, ThreadId } from '../types';
import { dispatchStreamEvent, type StreamEvent } from './streamEvents';

export interface ChatServiceOptions {
    /** Reuse only when resubmitting the exact same logical request. */
    requestId?: string;
    signal?: AbortSignal;
    onRun?: (runId: string) => void;
    onChunk: (text: string) => void;
    onStep: (step: AgentStep) => void;
    onStepFinish: (nodeOrTool: string) => void;
    /** Called when LangGraph enters a specialized agent node, or advisor_node (general) to exit a mode */
    onModeChange?: (mode: ChatMode) => void;
    /** Called when a backend tool yields a structured UI card payload */
    onCard?: (card: ChatCardPayload) => void;
    /** Called with the server-side thread id so it can be reused on the next turn */
    onSession?: (threadId: ThreadId) => void;
    /**
     * Called when the graph suspended on an `interrupt()` (a clinic follow-up
     * question). The question itself already arrived as `text` chunks, so the
     * UI only needs this to know the turn ended with a pending question.
     */
    onInterrupt?: (question: string) => void;
    onDone: () => void;
    onError: (error: Error) => void;
}

export interface UserInfoPayload {
    name?: string;
    age?: number;
    medical_history?: string;
    elder_mode?: boolean;
    region?: string;
}

export type VisionScanType = 'report' | 'drug_box' | 'trace_code';

// getRandomValues also works on HTTP LAN origins where randomUUID is absent.
const newRequestId = (): string => typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');

const consumeSseResponse = async (response: Response, options: ChatServiceOptions) => {
    if (!response.ok) {
        const body = await response.json().catch(() => null);
        const detail = body?.detail;
        throw new ChatStreamError(
            typeof detail === 'string' ? detail : detail?.message ?? `请求失败（${response.status}）`,
            detail?.code, detail?.run_id, detail?.retryable,
        );
    }

    if (!response.body) {
        throw new Error("ReadableStream not yet supported in this browser.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    try {
        while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value, { stream: !done });
            const lines = buffer.split('\n');
            buffer = done ? '' : lines.pop() ?? '';
            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const jsonStr = line.slice(5).trim();
                if (!jsonStr) continue;
                let data: StreamEvent;
                try { data = JSON.parse(jsonStr) as StreamEvent; }
                catch { throw new ChatStreamError('响应数据异常，请重新发起请求。', 'invalid_stream'); }
                if (data.run_id) options.onRun?.(data.run_id);
                if (data.type === 'finish') return;
                if (data.type === 'error') {
                    throw new ChatStreamError(data.content ?? '处理失败，请稍后重试。', data.code, data.run_id, data.retryable);
                }
                dispatchStreamEvent(data, options);
            }
            if (done) throw new ChatStreamError('连接提前结束，本次回答可能不完整。', 'stream_interrupted');
        }
    } finally {
        await reader.cancel().catch(() => undefined);
        reader.releaseLock();
    }
};

export class ChatStreamError extends Error {
    code?: string;
    runId?: string;
    retryable?: boolean;

    constructor(message: string, code?: string, runId?: string, retryable?: boolean) {
        super(message);
        this.name = 'ChatStreamError';
        this.code = code;
        this.runId = runId;
        this.retryable = retryable;
    }
}

export const streamChat = async (
    messages: ChatMessage[],
    options: ChatServiceOptions,
    userInfo?: UserInfoPayload,
    chatMode: ChatMode = 'general',
    threadId?: ThreadId | null
) => {
    try {
        // With a thread_id the backend checkpointer owns the history and only
        // needs the newest user message; without one it replays everything.
        const outbound = threadId ? messages.slice(-1) : messages;
        const resolvedThreadId = threadId ?? newRequestId();
        options.onSession?.(resolvedThreadId);
        const payload = {
            messages: outbound.map(msg => ({
                role: msg.role,
                content: msg.text
            })),
            user_info: userInfo ?? {},
            chat_mode: chatMode,
            thread_id: resolvedThreadId,
            request_id: options.requestId ?? newRequestId(),
        };

        const response = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: options.signal,
        });

        await consumeSseResponse(response, options);

        options.onDone();

    } catch (err) {
        options.onError(err instanceof Error ? err : new Error(String(err)));
    }
};

export const streamVisionChat = async (
    file: File,
    scanType: VisionScanType,
    messages: ChatMessage[],
    options: ChatServiceOptions,
    userInfo?: UserInfoPayload,
    threadId?: ThreadId | null
) => {
    try {
        const outbound = threadId ? messages.slice(-1) : messages;
        const resolvedThreadId = threadId ?? newRequestId();
        options.onSession?.(resolvedThreadId);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('scan_type', scanType);
        formData.append('user_info', JSON.stringify(userInfo ?? {}));
        formData.append('thread_id', resolvedThreadId);
        formData.append('request_id', options.requestId ?? newRequestId());
        formData.append('messages', JSON.stringify(outbound.map(msg => ({
            role: msg.role,
            content: msg.text
        }))));

        const response = await fetch('http://localhost:8000/api/vision-chat', {
            method: 'POST',
            body: formData,
            signal: options.signal,
        });

        await consumeSseResponse(response, options);
        options.onDone();

    } catch (err) {
        options.onError(err instanceof Error ? err : new Error(String(err)));
    }
};
