// Exercise the voice transcript and its return to chat with a mocked WebSocket.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
    const browser = await chromium.launch({
        headless: true,
        args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
            '--unsafely-treat-insecure-origin-as-secure=http://health.test'],
        ...(fs.existsSync(chromium.executablePath()) ? {} : { channel: 'chrome' }),
    });
    try {
        const context = await browser.newContext({ viewport: { width: 390, height: 844 }, permissions: ['microphone'] });
        const page = await context.newPage();
        const failures = [];
        page.on('pageerror', error => failures.push(error.message));
        const dist = path.resolve(__dirname, 'dist');
        await page.route('http://health.test/**', async route => {
            const pathname = new URL(route.request().url()).pathname;
            const file = path.resolve(dist, pathname === '/' ? 'index.html' : `.${pathname}`);
            assert.ok(file.startsWith(dist + path.sep));
            if (!fs.existsSync(file)) return route.fulfill({ status: 404, body: '' });
            const contentType = file.endsWith('.js') ? 'application/javascript' :
                file.endsWith('.css') ? 'text/css' : file.endsWith('.html') ? 'text/html' : 'image/svg+xml';
            await route.fulfill({ body: fs.readFileSync(file), contentType });
        });
        await page.routeWebSocket('ws://localhost:8000/api/voice', socket => {
            socket.onMessage(raw => {
                if (typeof raw !== 'string') return;
                const message = JSON.parse(raw);
                if (message.type === 'hello') {
                    for (const frame of [
                        { type: 'session', thread_id: 'voice-handoff' },
                        { type: 'state', value: 'thinking' },
                        { type: 'stt.final', turn_id: 1, text: '医保怎么报销？' },
                        { type: 'node_start', node: 'clinic_node' },
                        { type: 'text', content: '这个问题我帮您转给医保助手。' },
                        { type: 'node_start', node: 'insurance_node' },
                        { type: 'text', content: '请先告诉我您的参保地。' },
                    ]) socket.send(JSON.stringify(frame));
                } else if (message.type === 'bye') socket.close({ code: 1000 });
            });
        });
        await page.goto('http://health.test/');
        await page.getByRole('button', { name: '语音问诊通话' }).click();
        await page.getByRole('button', { name: '同意并开始' }).click();
        await page.getByText('已连接').waitFor({ timeout: 15000 });
        await page.getByTestId('open-transcript').click();
        const sheet = page.getByTestId('transcript-sheet');
        const transition = sheet.locator('[data-caption-role="assistant"]').filter({ hasText: '这个问题我帮您转给医保助手。' });
        const answer = sheet.locator('[data-caption-role="assistant"]').filter({ hasText: '请先告诉我您的参保地。' });
        await answer.waitFor();
        assert.equal(await transition.count(), 1);
        assert.equal(await transition.getAttribute('data-caption-final'), 'true');
        assert.equal(await transition.getByText('请先告诉我您的参保地。', { exact: true }).count(), 0);
        assert.equal(await answer.count(), 1);

        await page.getByRole('button', { name: '收起通话记录' }).click();
        await page.getByRole('button', { name: '挂断' }).click();
        await page.getByText('请先告诉我您的参保地。', { exact: true }).waitFor();
        const chatTransition = page.getByText('这个问题我帮您转给医保助手。', { exact: true }).locator('xpath=../../..');
        assert.equal(await chatTransition.getByText('请先告诉我您的参保地。', { exact: true }).count(), 0);
        assert.deepEqual(failures, []);
        await context.close();
        console.log('Voice handoff transcript and chat regression passed.');
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
