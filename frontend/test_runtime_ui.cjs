// Mobile regression against built assets and an intercepted API; no live model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
    const browser = await chromium.launch({ headless: true,
        ...(fs.existsSync(chromium.executablePath()) ? {} : { channel: 'chrome' }) });
    try {
        const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
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
        let release, requestSeen;
        const gate = new Promise(resolve => { release = resolve; });
        const entered = new Promise(resolve => { requestSeen = resolve; });
        const requests = [];
        await page.route('http://localhost:8000/api/chat', async route => {
            if (route.request().method() === 'OPTIONS') {
                await route.fulfill({ status: 204, headers: {
                    'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'content-type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                } });
                return;
            }
            requests.push(route.request().postDataJSON());
            requestSeen();
            if (requests.length === 1) await gate;
            const terminal = requests.length === 1
                ? { type: 'error', code: 'model_timeout', content: '模型响应超时，请稍后重试。' }
                : { type: 'finish', status: 'success' };
            const text = requests.length === 1 ? '部分内容' : requests.length === 2 ? '已恢复' : '继续刚才的问诊';
            const modeEvent = requests.length === 2
                ? { type: 'node_start', node: 'advisor_node' }
                : requests.length === 3 ? { type: 'node_start', node: 'clinic_node' } : null;
            await route.fulfill({ status: 200, headers: { 'Access-Control-Allow-Origin': '*' },
                contentType: 'text/event-stream', body:
                    (modeEvent ? `data: ${JSON.stringify(modeEvent)}\n\n` : '') +
                    `data: ${JSON.stringify({ type: 'text', content: text })}\n\n` +
                    `data: ${JSON.stringify(terminal)}\n\n` });
        });
        await page.goto('http://health.test/');
        const input = page.getByPlaceholder('描述症状、问医保、查报告…');
        const send = page.getByRole('button', { name: '发送', exact: true });
        await input.fill('测试问题');
        await send.click();
        await Promise.race([entered, new Promise((_, reject) => {
            setTimeout(() => reject(new Error('No chat request within five seconds')), 5000).unref();
        })]);
        await input.fill('下一个问题');
        assert.equal(await send.isDisabled(), true);
        await input.press('Enter');
        assert.equal(requests.length, 1);
        release();
        await page.getByText(/部分内容/).waitFor();
        await page.getByText(/模型响应超时/).waitFor();
        assert.equal(await page.getByLabel('正在输入').count(), 0);
        assert.equal(await send.isEnabled(), true);
        await send.click();
        await page.getByText('已恢复', { exact: true }).waitFor();
        assert.equal(requests.length, 2);
        assert.ok(requests[0].request_id && requests[0].thread_id);
        assert.notEqual(requests[0].request_id, requests[1].request_id);
        assert.equal(requests[0].thread_id, requests[1].thread_id);
        await input.fill('回到问诊');
        await send.click();
        await page.getByText('继续刚才的问诊', { exact: true }).waitFor();
        assert.equal(requests.length, 3);
        assert.equal(requests[2].thread_id, requests[0].thread_id);
        assert.notEqual(requests[2].request_id, requests[1].request_id);
        assert.equal(requests[2].messages.length, 1);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        assert.deepEqual(failures, []);
        const shot = path.resolve(__dirname, '../backend/.runtime/mobile-runtime.png');
        fs.mkdirSync(path.dirname(shot), { recursive: true });
        await page.screenshot({ path: shot, animations: 'disabled' });
        console.log(`Mobile runtime regression passed; screenshot: ${shot}`);
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
