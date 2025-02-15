# 🌍 Atlas AI — Cross-Border Market Entry Intelligence

**Multi-agent RAG swarm for cross-regional market entry analysis.**

9 specialized AI agents research regulatory, corporate, cultural, competitive, talent, and economic dimensions across Europe, MENA, and Asia — in parallel. LangGraph orchestrates. DeepSeek powers the LLM. Everything else is open-source and local.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Architecture

```
Streamlit UI (:8501) ──► FastAPI (:9734) ──► LangGraph StateGraph
                            │                      │
                ┌───────────┼───────────┐          │
                ▼           ▼           ▼          │
          Market A     Market B     Market C       │
          (6 agents)   (6 agents)   (6 agents)     │
                │           │           │          │
                └───────────┼───────────┘          │
                            ▼                      │
                     Synthesis Agent               │
                            │                      │
                     Devil's Advocate              │
                            │                      │
                      Final Report                 │
```

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

| Layer | Technology | API Key? |
|-------|-----------|----------|
| LLM | DeepSeek (swappable → Ollama) | Yes |
| Orchestration | LangGraph | No |
| RAG Framework | LangChain + langchain-milvus | No |
| Embeddings | BGE-M3 (1024d, multilingual, local) | No |
| Vector DB | Milvus 2.5 (Docker, hybrid BM25) | No |
| Doc Parsing | Docling (IBM, MIT) | No |
| Web Search | DuckDuckGo | No |
| API | FastAPI | No |
| UI | Streamlit | No |
| Cache | Redis | No |

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

Open http://localhost:8501, fill the form, click **Run Analysis**.

### Option B: Manual (Development)

```bash
git clone https://github.com/KunalScriptz/atlas-ai.git
cd atlas-ai

# Copy env template and add your DeepSeek API key
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY=sk-your-key

# Start infrastructure only
docker compose up -d milvus redis

# Install dependencies (first run downloads BGE-M3 ~2GB)
pip install -e .

# (Optional) Ingest RAG documents
# Place PDFs/DOCX in data/{trade_laws,tax_corporate,cultural,talent,economic,competitive}/
python -m src.rag.ingest data/

# Terminal 1: API
uvicorn src.main:app --port 9734 --reload

# Terminal 2: UI
streamlit run src/ui/app.py --server.port 8501
```

Open http://localhost:8501, fill the form, click **Run Analysis**.

---

## Example: Finnish SaaS → Dubai, Berlin, Singapore

**Input:**
- Product: "AI-powered HR analytics platform for SMEs"
- Home: Helsinki, Finland
- Markets: UAE (Dubai), Germany (Berlin), Singapore
- Budget: €100K–€500K
- Priorities: Speed to market, Cost, Talent access

**Output (~6 minutes):**
- 6-dimensional scored comparison table
- Ranked recommendation with confidence score
- Phased entry roadmap
- Devil's advocate risk flags
- PDF report

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
├── docker-compose.yml   # Milvus + Redis
├── Dockerfile
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE)

---

🤖 Built with LangGraph, LangChain, DeepSeek, BGE-M3, Milvus, Docling, and DuckDuckGo.
