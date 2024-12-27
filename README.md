# 🌍 Atlas AI — Cross-Border Market Entry Intelligence

**Multi-agent RAG swarm for cross-regional market entry analysis.**

9 specialized AI agents research regulatory, corporate, cultural, competitive, talent, and economic dimensions across Europe, MENA, and Asia — in parallel. Each agent retrieves from uploaded documents (RAG) and live web search (DuckDuckGo) before calling the LLM. LangGraph orchestrates. DeepSeek powers the LLM. Everything else is open-source and local.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/demo-live-green.svg)](https://atlas-ai.helixos.pro/)

**🌐 Live at [https://atlas-ai.helixos.pro/](https://atlas-ai.helixos.pro/)** — Deployed on Oracle Cloud with GitHub Actions CI/CD.

---

## The Problem

Expanding a digital product into a new country means answering questions across 6 dimensions per market:

- **Regulatory:** Can a foreign company own 100%? What license is needed?
- **Corporate:** What entity type? What's the tax rate? Free zone vs mainland?
- **Cultural:** How do you negotiate? What offends local partners?
- **Competitive:** Who already dominates? What do they charge?
- **Talent:** What are salary benchmarks? Contractor vs employee rules?
- **Economic:** Is the currency stable? Any geopolitical risks?

**3 markets × 6 dimensions = 18 deep-dive research tasks.** A consulting firm takes weeks and charges $10K–$50K. A single LLM prompt gives generic, unsourced answers — one generalist model can't think deeply about UAE corporate law AND German cultural norms AND Singapore salary data in the same response.

Atlas AI solves this by deploying **18 specialized agents in parallel** — each researches one dimension in one country with its own domain knowledge, retrieved documents, and live web search. What took weeks now takes ~6 minutes, with every score backed by real sources.

---

## Architecture

```
Streamlit UI (:8501) ──► FastAPI (:9734) ──► LangGraph StateGraph
       │                        │                      │
       │  file upload           │  POST /upload        │
       │  (optional)            │  Docling → embed     │
       │                        ▼                      │
       │                  Milvus (6 domain             │
       │                  collections)                 │
       │                        ▲                      │
       │                        │  retrieve (market-   │
       │                        │  filtered, COSINE)   │
       │                        │                      │
       │              ┌─────────┼───────────┐          │
       │              ▼         ▼           ▼          │
       │        Market A   Market B   Market C         │
       │        (6 agents) (6 agents) (6 agents)       │
       │              │         │           │          │
       │     Each agent runs:                        │
       │     ┌─ Milvus RAG  ─┐                       │
       │     └─ DuckDuckGo ──┘  (parallel)           │
       │              │         │           │          │
       │              └─────────┼───────────┘          │
       │                        ▼                      │
       │                 Synthesis Agent               │
       │                        │                      │
       │                 Devil's Advocate              │
       │                        │                      │
       └────────────────── Final Report                │
```

### How Each Agent Works

Before the LLM call, every research agent runs two retrievals in parallel:

```
agent.run()
    │
    ├── Milvus RAG ──► similarity_search("trade_laws", query)
    │   (market-filtered: market == "UAE" or market == "global")
    │
    └── DuckDuckGo ──► web search for current data
    │
    ▼
Both injected into the YAML prompt as {rag_context} and {web_search_context}
    │
    ▼
LLM call → PydanticOutputParser → structured result with scores + sources
```

Documents uploaded via the UI are tagged with a market (or "global" for market-agnostic docs).
The Milvus retriever filters by market at query time — a UAE trade law PDF won't surface when
researching Germany unless explicitly tagged global.

### Agent Swarm (9 types, 21 invocations per run)

| Agent | Domain | RAG Source |
|-------|--------|-----------|
| Regulatory Navigator | Business laws, licensing, data protection | Trade law documents, government gazettes |
| Corporate Structuring | Entity types, tax optimization, DTAA | Tax codes, treaties, free zone rules |
| Cultural Intelligence | Business etiquette, negotiation, localization | Hofstede, case studies, etiquette guides |
| Competitive Intelligence | Local competitors, pricing, market share | Web search, news, company data |
| Talent & Workforce | Salary benchmarks, visas, labor laws | Salary surveys, visa databases |
| Economic & Political Risk | Currency, inflation, sovereign ratings | World Bank API, IMF data |
| **Synthesis** | Cross-market comparison, ranked recommendation | Results from all 6 agents |
| **Devil's Advocate** | Challenges recommendation, finds gaps | Synthesis output |
| **Supervisor** | Orchestrates workflow, handles errors | All state |

### Tech Stack (All Open-Source Except LLM)

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | DeepSeek (swappable → Ollama) | LLM factory pattern — swap via `.env`, no code changes |
| **Orchestration** | LangGraph | StateGraph with `Send()` parallel fan-out: 18 agents run concurrently across 3 markets |
| **RAG Framework** | LangChain + langchain-milvus | Pre-LLM retrieval: Milvus RAG + DuckDuckGo web search run in parallel via `asyncio.gather()`, injected into prompt. Market-filtered scalar expressions prevent cross-market contamination |
| **Embeddings** | BGE-M3 (1024d, multilingual) | 100+ languages, runs locally on CPU, zero API cost. Handles German, Arabic, Chinese legal docs natively. Queries and documents embedded into same 1024d space for COSINE similarity |
| **Vector DB** | Milvus 2.5 (Docker) | Self-hosted standalone, embedded etcd, local storage. 6 domain collections with COSINE similarity. Market metadata stored per chunk for filtered retrieval |
| **Doc Parsing** | Docling (IBM, MIT) | PDF/DOCX/PPTX/HTML with table extraction. UI upload → parse → chunk → embed → Milvus in a single request. CPU-bound ops via `asyncio.to_thread()` |
| **Web Search** | DuckDuckGo | Free, zero API key. Runs in parallel with RAG retrieval — agent always has live + curated sources. 1h Redis cache to avoid rate limiting |
| **Economic Data** | World Bank API | Free GDP, inflation, trade indicators per country |
| **API** | FastAPI | Async job management, background `asyncio.create_task()` for swarm execution |
| **UI** | Streamlit | Real-time progress polling, per-market expandable results, scored comparison tables |
| **Cache** | Redis | Targeted: web pages (24h), embeddings (permanent), search results (1h) |
| **PDF Reports** | ReportLab | Downloadable market entry reports |
| **CI/CD** | GitHub Actions | Self-hosted Oracle Cloud runner, auto-deploy on push to main |

---

## Quick Start

### Option A: Docker (Recommended)

```bash
git clone https://github.com/KunalScriptz/atlas-ai.git
cd atlas-ai

# Copy env template and add your DeepSeek API key
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY=sk-your-key

# Start everything (Milvus, Redis, API, UI)
docker compose up -d

# View logs with container names
docker compose logs -f
```

Open http://localhost:8501, fill the form, optionally upload market research PDFs/DOCX, click **Run Analysis**.

### Option B: Manual (Development)

```bash
git clone https://github.com/KunalScriptz/atlas-ai.git
cd atlas-ai

# Copy env template and add your DeepSeek API key
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY=sk-your-key

# Start infrastructure only
docker compose up -d milvus redis

# Install dependencies
pip install -e .
# Or faster: uv sync --no-dev (requires uv: pip install uv)

# (Optional) Bulk ingest RAG documents via CLI
# Place PDFs/DOCX in data/{trade_laws,tax_corporate,cultural,talent,economic,competitive}/
python -m src.rag.ingest data/

# Terminal 1: API
uvicorn src.main:app --port 9734 --reload

# Terminal 2: UI
streamlit run src/ui/app.py --server.port 8501
```

Open http://localhost:8501, fill the form, optionally upload documents via the file uploader, click **Run Analysis**.

> **Uploading documents:** Use the file uploader in the UI to add trade laws, salary surveys, or market reports. Files are parsed on the spot with Docling, embedded with BGE-M3, and stored in Milvus. Select which market the document applies to (or "All Markets" for global docs). For bulk ingestion, use the CLI command above.

---

## Example: Finnish SaaS → Dubai, Berlin, Singapore

**Input:**
- Product: "AI-powered HR analytics platform for SMEs"
- Home: Helsinki, Finland
- Markets: UAE (Dubai), Germany (Berlin), Singapore
- Budget: €100K–€500K
- Priorities: Speed to market, Cost, Talent access

**Output (~6 minutes):**
- 6-dimensional scored comparison table with cited sources
- Ranked recommendation with confidence score (adjusted by devil's advocate)
- Per-market expandable agent detail cards (Milvus docs + web search results used)
- Phased entry roadmap
- Devil's advocate risk flags + missing data gaps
- PDF download

---

## LLM Swap: DeepSeek → Ollama (Fully Local)

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:14b
OLLAMA_BASE_URL=http://localhost:11434
```

Zero API keys. Everything runs locally including embeddings, vector DB, and search.

---

## Project Structure

```
atlas-ai/
├── src/
│   ├── agents/          # 9 agents (base + 8 specialized)
│   ├── graph/           # LangGraph: state, nodes, edges, workflow
│   ├── prompts/         # YAML prompts (zero hardcoded strings)
│   ├── rag/             # Embeddings, retrievers, vector store, loaders
│   ├── tools/           # DuckDuckGo search, World Bank API
│   ├── llm/             # LLM factory (DeepSeek | Ollama)
│   ├── api/             # FastAPI routes + schemas
│   ├── ui/              # Streamlit app + components
│   └── utils/           # Logging, Redis cache, PDF generation
├── data/                # RAG source documents (gitignored)
├── docker-compose.yml   # Milvus + Redis + API + UI (all-in-one)
├── Dockerfile
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE)

---

🤖 Built with LangGraph, LangChain, DeepSeek, BGE-M3, Milvus, Docling, and DuckDuckGo.
