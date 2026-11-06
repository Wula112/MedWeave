# MedWeave — Multi-Agent Medical Assistant

[![Upstream release](https://img.shields.io/github/v/release/wananing/Smart-Health-Assistant)](https://github.com/wananing/Smart-Health-Assistant/releases/latest)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](backend/pyproject.toml)
[![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=111)](frontend/package.json)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**English** | [简体中文](README.md)

MedWeave is a privacy-aware, mobile-first reference application for building multimodal health AI experiences. Inspired by products such as Ant Afu, it combines LangGraph orchestration, a FastAPI streaming backend, and a React client with structured, server-driven UI cards.

The project demonstrates multi-agent routing, hybrid RAG, pluggable Agent Skills, direct vision-model input for medical reports and medicine packaging, OpenTelemetry tracing, and regression evaluation.

> [!IMPORTANT]
> This repository is a technical reference, not a medical device. It must not be used for diagnosis, treatment decisions, emergencies, or medication changes without review by a qualified healthcare professional.

## Screenshots

The interface is designed for mobile viewports. The current product copy and bundled knowledge base are Chinese-first.

<table>
  <tr>
    <td align="center"><img src="docs/images/preview-home.png" width="240" alt="Home: health summary and chat"><br><strong>Home: health summary + chat</strong></td>
    <td align="center"><img src="docs/images/preview-report.png" width="240" alt="Lab report analysis"><br><strong>Lab report analysis</strong></td>
    <td align="center"><img src="docs/images/preview-dashboard.png" width="240" alt="Daily health goals"><br><strong>Daily health goals</strong></td>
  </tr>
</table>

All data in the screenshots is synthetic.

## What It Includes

- **Multi-agent orchestration:** a global router dispatches requests to advisor, clinic, insurance, report, and pharmacy agents while preserving the active specialist context. Specialists can hand a turn to each other with LangGraph `Command` (swarm-style handoff), so a symptom described in insurance mode is answered by the clinic agent in the same turn.
- **Stateful sessions and a real clinic state machine:** the graph is compiled with a checkpointer (memory or SQLite) keyed by `thread_id`. The clinic agent is a compiled subgraph with a code-level emergency gate that runs before any model call and again on every follow-up answer, one structured call that extracts symptoms and drafts the next question, `interrupt()`-based follow-up questions, and a structured triage card. See [`docs/langgraph-runtime.md`](docs/langgraph-runtime.md).
- **Pause and resume consultations:** switch specialists during a follow-up, then return to the clinic within the same conversation. Completed consultations clear active interview state while preserving conversation context.
- **Multimodal health workflows:** report, medicine-box, and trace-code images are sent directly to a configured vision-capable model; no separate OCR service is required.
- **Privacy-aware image handling:** report previews are blurred by default, press-and-hold reveal is temporary, and common identity fields are redacted from model output on a best-effort basis.
- **Streaming, server-driven UI:** FastAPI emits text, node status, tool status, and structured card events over SSE; React renders cards in the conversation context.
- **Hybrid RAG:** BM25 and dense retrieval are combined with reciprocal rank fusion. Embedding and vector-store backends are configurable.
- **Context compression:** summary budgets adapt to model capacity and prompt overhead. See [context management](backend/docs/context-management.md).
- **Pluggable Agent Skills:** domain tools are discovered from `backend/skills/<skill_name>/` without a central registration list.
- **Open observability and evals:** OpenInference spans are exported through OpenTelemetry to Jaeger or another OTLP backend. Anonymous regression cases support strict local scoring and optional DeepEval.
- **Design system and elder mode:** semantic colour, type, radius and shadow tokens with shared result-card components ([`frontend/docs/design-system.md`](frontend/docs/design-system.md)); elder mode moves every text role one size up.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Agent runtime | LangGraph, LangChain, OpenAI-compatible model APIs |
| API | Python 3.13, FastAPI, Uvicorn, SSE |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Retrieval | BM25, dense embeddings, reciprocal rank fusion |
| Vector stores | Chroma, FAISS, Qdrant, pgvector |
| Observability | OpenInference, OpenTelemetry, Jaeger, optional LangSmith |
| Evaluation | Repository-managed JSONL cases, local scoring, optional DeepEval |

## Quick Start

### Prerequisites

- Python `3.13`
- [`uv`](https://docs.astral.sh/uv/)
- Node.js `20.19+` or `22.12+`
- An API key for a supported chat model
- Docker with Compose, only if you want local Jaeger tracing

### 1. Clone and configure the backend

```bash
git clone https://github.com/Wula112/MedWeave.git
cd MedWeave/backend

uv sync
cp .env.example .env
```

Edit `backend/.env`. ARK is the backward-compatible default:

```dotenv
LLM_PROVIDER=ark
ARK_API_KEY=your_api_key
ARK_MODEL_ID=your_model_or_endpoint_id
```

Build the local RAG index. The default Hugging Face embedding model is downloaded on first use.

```bash
uv run python -m rag.ingest
uv run uvicorn main:app --reload --port 8000
```

For multiple workers on one host, use `uv run uvicorn main:app --workers 2 --port 8000`.
The API uses shared SQLite checkpoints and run admission; all workers must share
the same local data directory. See [execution control and deployment](docs/agent-runtime.md).

The API is available at `http://localhost:8000`; health checks use `GET /health`.

### 2. Start the frontend

In a second terminal:

```bash
cd MedWeave/frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Use a mobile browser viewport for the intended layout.

## Model Configuration

All chat integrations use OpenAI-compatible APIs behind one configuration layer.

| Provider | `LLM_PROVIDER` | Main environment variables |
| --- | --- | --- |
| Volcengine ARK / Doubao | `ark` | `ARK_API_KEY`, `ARK_MODEL_ID` |
| OpenAI | `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` |
| Alibaba Cloud Model Studio / Qwen | `qwen` | `DASHSCOPE_API_KEY`, `QWEN_MODEL` |
| Zhipu / GLM | `zhipu` | `ZAI_API_KEY`, `ZHIPU_MODEL` |
| Any compatible endpoint | `custom` | `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` |

Vision inherits the chat provider by default. If the selected chat model cannot accept images, configure a separate provider and vision-capable model:

```dotenv
VISION_PROVIDER=custom
VISION_API_KEY=your_api_key
VISION_MODEL=your_vision_model
VISION_BASE_URL=https://provider.example.com/v1
```

See [`backend/.env.example`](backend/.env.example) for all provider, embedding, vector-store, tracing, and evaluation options.

## Project Layout

```text
backend/
  agents/          LangGraph router, specialist agents, clinic subgraph, handoffs
  evals/           Regression cases and scoring adapters
  rag/             Retrieval, ingestion, and knowledge documents
  skills/          Auto-discovered domain skills
  main.py          FastAPI and SSE endpoints
  observability.py OpenTelemetry and LangSmith setup
frontend/src/
  components/      Reusable UI, structured chat cards
  design/          Design tokens and the elder-mode text scale
  screens/         Home chat, health dashboard
  services/        SSE and vision clients
  store/           Shared client state
  types/           Frontend event and card contracts
docs/              Design notes and screenshots
compose.observability.yml
```

## Development Checks

Run deterministic backend tests:

```bash
cd backend
uv run python -m unittest \
  test_llm.py \
  test_vision.py \
  test_main_config.py \
  test_observability.py \
  test_evals.py \
  test_clinic_graph.py \
  test_clinic_persistence.py \
  test_graph_build.py \
  test_handoff.py \
  test_cards.py \
  test_skill_tools.py
uv lock --check
```

Lint and build the frontend:

```bash
cd frontend
npm run lint
npm run build
```

With both development servers running, execute the Playwright smoke flow with `node test_pw.cjs` from `frontend/`.

## Privacy, Security, and Limitations

- Uploaded images are sent to the configured third-party model provider. Review that provider's data handling terms before using sensitive material.
- UI blur protects against casual on-screen exposure; it does not anonymize the original image sent to the model.
- PII filtering is best effort and is not a substitute for access control, encryption, retention policies, consent, or compliance review.
- Only JPEG, PNG, and WebP uploads up to 8 MB are accepted.
- Bundled insurance services and user records are mock data. The application does not include production authentication, tenancy, billing, or clinical governance.
- Drug trace-code recognition extracts visible content only and does not verify authenticity against a regulatory database.
- Use synthetic or de-identified data for development, testing, screenshots, traces, and evaluation cases.

## Contributing

Issues and focused pull requests are welcome. Describe the behavior change, include verification commands, and attach screenshots for mobile UI changes. New agent behavior should include deterministic tests and at least one normal-path and one safety-boundary evaluation case.

See [`AGENTS.md`](AGENTS.md) for repository conventions and contributor commands.

## License

This project is available under the [MIT License](LICENSE).
