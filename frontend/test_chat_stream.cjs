// Exercise the real TS client with in-memory HTTP/SSE responses; no API needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const ts = require('typescript');

const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'health-stream-'));
for (const name of ['chatService', 'streamEvents']) {
    const source = fs.readFileSync(path.join(__dirname, 'src', 'services', `${name}.ts`), 'utf8');
    const compiled = ts.transpileModule(source, { compilerOptions: {
        module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
    } }).outputText;
    fs.writeFileSync(path.join(directory, `${name}.js`), compiled);
}
const { streamChat, streamVisionChat } = require(path.join(directory, 'chatService.js'));
const encoder = new TextEncoder();
const message = [{ role: 'user', text: 'test' }];

function observer() {
    const events = [];
    return { events, options: {
        requestId: 'stable-request',
        onChunk: text => events.push(['text', text]),
        onStep() {}, onStepFinish() {},
        onDone: () => events.push(['done']),
        onError: error => events.push(['error', error.code, error.message]),
    } };
}

function sse(chunks) {
    return new Response(new ReadableStream({ start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
    } }), { headers: { 'Content-Type': 'text/event-stream' } });
}

(async () => {
    let outbound;
    global.fetch = async (_url, options) => {
        outbound = JSON.parse(options.body);
        return sse(['data: {"type":"text","content":"一', '段"}\n\ndata: {"type":"finish"}\n\n']);
    };
    let test = observer();
    await streamChat(message, test.options, {}, 'general', 'thread');
    assert.deepEqual(test.events, [['text', '一段'], ['done']]);
    assert.equal(outbound.request_id, 'stable-request');

    global.fetch = async () => sse(['data: {"type":"text","content":"partial"}\n\n',
        'data: {"type":"error","code":"model_timeout","content":"超时"}\n\n']);
    test = observer();
    await streamChat(message, test.options);
    assert.deepEqual(test.events, [['text', 'partial'], ['error', 'model_timeout', '超时']]);

    global.fetch = async () => sse(['data: {"type":"text","content":"partial"}\n\n']);
    test = observer();
    await streamChat(message, test.options);
    assert.equal(test.events.at(-1)[1], 'stream_interrupted');
    assert.equal(test.events.filter(event => event[0] === 'done').length, 0);

    global.fetch = async () => new Response(JSON.stringify({ detail: {
        code: 'thread_busy', message: '会话处理中', run_id: 'original',
    } }), { status: 409 });
    test = observer();
    await streamChat(message, test.options);
    assert.deepEqual(test.events, [['error', 'thread_busy', '会话处理中']]);

    global.fetch = async (_url, options) => {
        assert.equal(options.body.get('request_id'), 'stable-request');
        assert.ok(options.body.get('thread_id'));
        return sse(['data: {"type":"error","code":"model_timeout","content":"超时"}\n\n']);
    };
    test = observer();
    await streamVisionChat(new File(['image'], 'report.png', { type: 'image/png' }), 'report', message, test.options);
    assert.deepEqual(test.events, [['error', 'model_timeout', '超时']]);
    console.log('Chat stream tests passed (success, error, premature EOF, conflict, vision).');
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => {
    // Only this test's verified temporary directory is removed.
    assert.equal(path.dirname(directory), os.tmpdir());
    fs.rmSync(directory, { recursive: true, force: true });
});
