# Cikgu AI Kimia 🧪

**SPM Chemistry AI Tutor** — Deterministic solver + Multilingual RAG + Telegram Bot

---

## Quick Start (5 minutes)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — add your GROQ_API_KEY and TELEGRAM_BOT_TOKEN

# 3. Build indexes
python scripts/build_index_v2.py --kb-dir knowledge_base --validate

# 4. Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 5. Run Telegram bot (new terminal)
python bot/telegram_bot.py
```

---

## Project Structure

```
cikgu-ai-kimia/
│
├── knowledge_base/              ← Your SPM content
│   ├── theory/
│   │   ├── form4/               ← BAB 1-5 Form 4 notes
│   │   └── form5/               ← BAB 1-8 Form 5 notes
│   ├── calculations/            ← Calculation worked examples
│   ├── questions/past_years/    ← SPM past year papers (add later)
│   └── images/                  ← Diagram images (optional)
│
├── rag/                         ← RAG pipeline
│   ├── chunker.py               ← Markdown → chunks
│   ├── metadata_tagger.py       ← Tag chapters, topics, keywords
│   ├── embedder.py              ← Multilingual sentence transformer
│   ├── indexer.py               ← Build 3 FAISS indexes
│   ├── retriever.py             ← Query → relevant chunks
│   ├── diagram_processor.py     ← Diagram descriptions (60 built-in)
│   └── past_year_questions.py   ← Built-in SPM question bank
│
├── solver/                      ← Deterministic Python solver
│   ├── solver_engine.py         ← All calculation solvers
│   ├── extractor.py             ← Extract numbers/formulas from text
│   ├── router.py                ← Route question to correct solver
│   ├── formula_parser.py        ← Parse chemical formulas
│   ├── equation_parser.py       ← Parse chemical equations
│   └── units.py                 ← Unit conversions
│
├── api/
│   └── main.py                  ← FastAPI application (7 endpoints)
│
├── bot/
│   └── telegram_bot.py          ← Telegram bot
│
├── scripts/
│   └── build_index_v2.py        ← Build all FAISS indexes
│
├── faiss_indexes/               ← Built indexes (commit to git!)
├── render.yaml                  ← Render deployment config
├── requirements.txt
└── .env.example
```

---

## Architecture

```
Student Question
       │
       ▼
   router.py  ──────────────────────────────────────┐
       │                                             │
       │ Calculation detected                        │ Theory/concept
       ▼                                             ▼
 solver_engine.py                            retriever.py
 (Pure Python, deterministic)                (FAISS search)
       │                                             │
       ▼                                             ▼
 SPM Format Answer                          RAG Context
 Diberi/Formula/                            (Top-k chunks)
 Pengiraan/Jawapan                               │
       │                                          ▼
       └──────────────────────────────────► Groq LLM
                                           (Explanation only)
                                                │
                                                ▼
                                          Final Answer (BM)
```

**Critical rule: LLM never computes. Python solver handles all maths.**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health + index stats |
| POST | `/api/chat` | Main Q&A (auto-routes) |
| POST | `/api/solve` | Calculation only (no LLM) |
| POST | `/api/retrieve` | Raw retrieval (debug) |
| POST | `/api/quiz` | Generate quiz questions |
| GET | `/api/index/stats` | FAISS statistics |
| POST | `/api/index/rebuild` | Rebuild indexes (background) |

### Test immediately after startup:

```bash
# Health check
curl http://localhost:8000/api/health

# Calculation (no LLM needed)
curl -X POST http://localhost:8000/api/solve \
  -H "Content-Type: application/json" \
  -d '{"question": "Hitungkan bilangan mol dalam 4.7g K2O"}'

# Theory question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Terangkan tindak balas eksotermik"}'
```

---

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome screen |
| `/help` | Show all commands |
| `/solve [question]` | Calculation only |
| `/quiz [topic]` | Generate quiz |
| `/chapter [n]` | Filter by chapter |
| `/clear` | Reset session |

---

## Deploy to Render

```bash
# 1. Build indexes locally first
python scripts/build_index_v2.py --kb-dir knowledge_base

# 2. Push to GitHub (includes faiss_indexes/)
git init
git add .
git commit -m "Initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/cikgu-ai-kimia.git
git push -u origin main

# 3. Connect GitHub repo to Render
# render.yaml is already configured — Render detects it automatically

# 4. Add environment variables in Render dashboard:
#    GROQ_API_KEY = gsk_...
#    TELEGRAM_BOT_TOKEN = your_token
#    API_BASE_URL = https://cikgu-ai-kimia-api.onrender.com
```

---

## Adding Past Year Questions

Create `knowledge_base/questions/past_years/SPM_2023_Kimia.md`:

```markdown
# SPM 2023 Kimia Tingkatan 5

---

## Soalan 1 — Bab 7 Kadar Tindak Balas [3 markah]

Hitungkan kadar tindak balas purata bagi minit pertama jika...

Keywords: kadar tindak balas, graf, isipadu gas

---

### Jawapan

Diberi:
...

Formula:
Kadar = perubahan isipadu ÷ masa

Pengiraan:
...

Jawapan:
30 cm³ min⁻¹
```

Then rebuild: `python scripts/build_index_v2.py --kb-dir knowledge_base`

---

## FAISS Indexes

| Index | Content | Handles |
|-------|---------|---------|
| `index_theory` | BAB notes, definitions, concepts | Theory questions |
| `index_calculations` | Worked examples, formulas | Calc questions |
| `index_qa` | Past year questions, schemes | Exam prep |

Embedding model: `paraphrase-multilingual-MiniLM-L12-v2` (BM + EN)

---

## Groq Models

| Model | Use case |
|-------|----------|
| `llama-3.1-70b-versatile` | Best quality, recommended |
| `llama-3.1-8b-instant` | Fastest, good for bot |
| `mixtral-8x7b-32768` | Long context questions |
