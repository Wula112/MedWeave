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
        const sse = events => events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
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
            const node = (name, content) => ({ type: 'node_start', node: name, content });
            const text = content => ({ type: 'text', content });
            const finish = { type: 'finish', status: 'success' };
            const cases = [
                [text('部分内容'), { type: 'error', code: 'model_timeout', content: '模型响应超时，请稍后重试。' }],
                [node('advisor_node'), text('已恢复'), finish],
                [node('clinic_node'), text('继续刚才的问诊'), finish],
                [node('insurance_node', '进入医保咨询模块'), text('这属于身体不适的症状咨询，'),
                    { type: 'tool_start', tool: 'transfer_to_clinic', content: '正在转接预问诊助手…' },
                    { type: 'tool_end', tool: 'transfer_to_clinic' }, text('已为你转接预问诊助手，请稍候。'),
                    { type: 'node_end', node: 'insurance_node' }, node('clinic_node', '进入预问诊模块'),
                    { type: 'card', payload: { type: 'clinic_recommendation', data: {
                        departments: ['消化内科'], urgency: 'routine', summary: '建议就诊咨询',
                    } } }, text('胃胀让你不舒服多久了？'),
                    { type: 'interrupt', content: '胃胀让你不舒服多久了？' }, finish],
                [node('clinic_node'), text('追问的前半句，'), node('ask_followup'), text('后半句。'), finish],
                [node('insurance_node', '无过渡语的医保节点'), { type: 'node_end', node: 'insurance_node' },
                    node('clinic_node', '无过渡语的问诊节点'), text('请说说症状。'), finish],
                [node('insurance_node'), text('这就帮您转接。'), node('clinic_node'),
                    { type: 'error', code: 'model_timeout', content: '接手的助手暂时不可用。' }],
            ];
            await route.fulfill({ status: 200, headers: { 'Access-Control-Allow-Origin': '*' },
                contentType: 'text/event-stream', body: sse(cases[requests.length - 1]) });
        });
        let visionRequested = false;
        await page.route('http://localhost:8000/api/vision-chat', async route => {
            visionRequested = true;
            await route.fulfill({ status: 200, headers: { 'Access-Control-Allow-Origin': '*' },
                contentType: 'text/event-stream', body: sse([
                    { type: 'node_start', node: 'report_node' },
                    { type: 'text', content: '这张图片涉及用药，我帮您转接。' },
                    { type: 'node_start', node: 'pharmacy_node' },
                    { type: 'text', content: '请告诉我药品名称。' },
                    { type: 'finish', status: 'success' },
                ]) });
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

        const messageContaining = value => page.getByText(value, { exact: true }).locator('xpath=../../..');
        await page.getByRole('button', { name: '查医保', exact: true }).click();
        await input.fill('我胃胀');
        await send.click();
        const transition = '这属于身体不适的症状咨询，已为你转接预问诊助手，请稍候。';
        await page.getByText('胃胀让你不舒服多久了？', { exact: true }).waitFor();
        const beforeHandoff = messageContaining(transition);
        const afterHandoff = messageContaining('胃胀让你不舒服多久了？');
        assert.equal(await beforeHandoff.getByText('胃胀让你不舒服多久了？', { exact: true }).count(), 0);
        assert.equal(await beforeHandoff.getByText('进入医保咨询模块', { exact: true }).count(), 1);
        assert.equal(await beforeHandoff.getByText('进入预问诊模块', { exact: true }).count(), 0);
        assert.equal(await afterHandoff.getByText('进入预问诊模块', { exact: true }).count(), 1);
        assert.equal(await afterHandoff.locator('[data-card="clinic_recommendation"]').count(), 1);
        assert.equal(await afterHandoff.getByText('您好！我是您的专属线上医生助手。请告诉我您主要哪里不舒服？大概持续多久了？', { exact: true }).count(), 1);
        assert.equal(await page.getByLabel('正在输入').count(), 0);
        await page.getByText('胃胀让你不舒服多久了？', { exact: true }).scrollIntoViewIfNeeded();
        const handoffShot = path.resolve(__dirname, '../backend/.runtime/mobile-handoff.png');
        fs.mkdirSync(path.dirname(handoffShot), { recursive: true });
        await page.screenshot({ path: handoffShot, animations: 'disabled' });

        await input.fill('三天了');
        await send.click();
        await page.getByText('追问的前半句，后半句。', { exact: true }).waitFor();
        assert.equal(requests[4].messages.length, 1);
        assert.equal(requests[4].thread_id, requests[0].thread_id);

        await page.getByRole('button', { name: '查医保', exact: true }).click();
        const completedBefore = await page.getByText('已完成', { exact: true }).count();
        await input.fill('另一种转接');
        await send.click();
        await page.getByText('请说说症状。', { exact: true }).waitFor();
        const withoutTransition = messageContaining('请说说症状。');
        assert.equal(await withoutTransition.getByText('无过渡语的医保节点', { exact: true }).count(), 1);
        assert.equal(await withoutTransition.getByText('无过渡语的问诊节点', { exact: true }).count(), 1);
        assert.equal(await page.getByText('已完成', { exact: true }).count(), completedBefore + 1);

        await page.getByRole('button', { name: '查医保', exact: true }).click();
        await input.fill('转接后失败');
        await send.click();
        await page.getByText('[接手的助手暂时不可用。]', { exact: true }).waitFor();
        assert.equal(await messageContaining('这就帮您转接。').getByText('[接手的助手暂时不可用。]', { exact: true }).count(), 0);
        assert.equal(await page.getByLabel('正在输入').count(), 0);

        await page.locator('input[type="file"]').setInputFiles({ name: 'report.png', mimeType: 'image/png', buffer: Buffer.from('test') });
        await page.getByText('请告诉我药品名称。', { exact: true }).waitFor();
        assert.equal(visionRequested, true);
        assert.equal(await messageContaining('这张图片涉及用药，我帮您转接。').getByText('请告诉我药品名称。', { exact: true }).count(), 0);

        // Deliver the handoff in separate network chunks to check the in-flight UI too.
        await page.evaluate(() => {
            let controller;
            window.fetch = async () => new Response(new ReadableStream({ start(stream) { controller = stream; } }),
                { headers: { 'Content-Type': 'text/event-stream' } });
            window.sendTestEvents = events => controller.enqueue(new TextEncoder().encode(
                events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('')));
        });
        await input.fill('流式转接');
        await send.click();
        await page.evaluate(() => window.sendTestEvents([
            { type: 'node_start', node: 'pharmacy_node' },
            { type: 'text', content: '先转给问诊助手。' },
        ]));
        await page.getByText('先转给问诊助手。', { exact: true }).waitFor();
        await page.evaluate(() => window.sendTestEvents([{ type: 'node_start', node: 'clinic_node' }]));
        await page.getByLabel('正在输入').waitFor();
        assert.equal(await messageContaining('先转给问诊助手。').getByLabel('正在输入').count(), 0);
        await input.fill('下一条消息');
        assert.equal(await send.isDisabled(), true);
        await page.evaluate(() => window.sendTestEvents([
            { type: 'text', content: '具体哪里不舒服？' }, { type: 'finish', status: 'success' },
        ]));
        await page.getByText('具体哪里不舒服？', { exact: true }).waitFor();
        assert.equal(await messageContaining('先转给问诊助手。').getByText('具体哪里不舒服？', { exact: true }).count(), 0);
        assert.equal(await page.getByLabel('正在输入').count(), 0);
        assert.equal(await send.isEnabled(), true);
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
