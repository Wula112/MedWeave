[English](README.en.md) | **简体中文**

# 🏥 MedWeave——多智能体医疗助手

MedWeave 是基于 **LangGraph、FastAPI 和 React** 的多智能体医疗助手，支持流式对话、工具调用和结构化结果卡片。

## ✨ 核心功能

- **多智能体协作**：全局路由根据用户意图调度预问诊、医保、健康咨询、药品和报告解读等智能体。
- **智能预问诊**：围绕症状部位、持续时间和伴随症状进行多轮追问，生成包含紧急程度和推荐科室的就诊建议。
- **会话暂停与恢复**：问诊过程中可咨询其他智能体，随后返回原对话继续问诊，并保留完整历史记录。
- **智能体转接**：不同专科智能体可直接转接当前对话，无需用户手动退出或重新描述问题。
- **多模态识别**：支持检验报告解读、拍照问药和药品追溯码识别。
- **RAG 知识库**：结合 BM25 与向量检索，覆盖常见疾病、医保政策、检验参考范围和用药指南。详见 [RAG 文档](docs/rag.md)。
- **可扩展技能系统**：智能体可动态加载领域技能，现有技能涵盖急症预检、症状评分、健康指标计算、化验单解读、慢病风险评估和药物剂量计算。详见 [Skills 文档](docs/skills.md)。
- **结构化界面**：后端通过 SSE 推送结果卡片，前端统一展示医保信息、报告解读、药品信息和就诊建议。
- **长辈模式**：一键放大字体、按钮和关键操作入口。
- **多模型配置**：聊天模型与图片模型可独立配置，支持 ARK、OpenAI、DeepSeek、通义千问、智谱及其他 OpenAI-compatible 服务。
- **可观测与评估**：支持 OpenTelemetry、OpenInference、Jaeger 和 DeepEval。详见 [可观测与评估文档](docs/observability-evals.md)。

## 📸 界面预览

项目采用移动端优先设计，建议使用浏览器的手机模拟模式查看。

<table>
  <tr>
    <td align="center"><img src="docs/images/preview-home.png" width="250" alt="首页：健康概览与对话"><br/><b>首页：健康概览与对话</b></td>
    <td align="center"><img src="docs/images/preview-report.png" width="250" alt="检验报告解读"><br/><b>检验报告解读</b></td>
    <td align="center"><img src="docs/images/preview-dashboard.png" width="250" alt="健康小目标"><br/><b>健康小目标</b></td>
  </tr>
</table>

截图中的数据均为模拟数据。

## 🛠️ 技术栈

- **后端**：Python、FastAPI、LangChain、LangGraph、Uvicorn
- **前端**：React、TypeScript、Tailwind CSS、Vite

## 🚀 快速开始

```bash
git clone https://github.com/Wula112/MedWeave.git
cd MedWeave
```

### 启动后端

```bash
cd backend
uv sync
cp .env.example .env
```

在 `.env` 中配置模型服务商、API Key 和模型名称，然后构建 RAG 索引并启动服务：

```bash
uv run python -m rag.ingest
uv run uvicorn main:app --reload --port 8000
```

首次构建索引时，默认的本地 HuggingFace 嵌入模型会自动下载，大小约 90 MB。

### 启动前端

另开一个终端：

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173，并切换至手机视图。

## 📄 开源协议

本项目采用 [MIT License](LICENSE)。
