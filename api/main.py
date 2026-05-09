"""
main.py — Cikgu AI Kimia FastAPI Application
=============================================
Production-ready REST API with:
  - /api/chat        — main Q&A endpoint (RAG + solver)
  - /api/solve       — calculation-only endpoint
  - /api/quiz        — quiz generation
  - /api/retrieve    — raw retrieval (for debugging)
  - /api/health      — health check
  - /api/index/stats — FAISS index statistics

Architecture:
  Question → Router → Calculation Solver (deterministic)
                   → RAG Retrieval → LLM (theory/explanation)

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Adjust Python path so rag/ and solver/ modules are importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR / "solver"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cikgu_ai_kimia")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", str(BASE_DIR / "faiss_indexes"))
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
SCORE_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.30"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3000"))

# ---------------------------------------------------------------------------
# APP LIFESPAN — load models once at startup
# ---------------------------------------------------------------------------

_app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG components at startup, free at shutdown."""
    logger.info("Starting Cikgu AI Kimia...")

    try:
        from embedder import get_embedder
        from retriever import get_retriever
        from router import route
        from solver_engine import solve_by_task

        _app_state["embedder"] = get_embedder()
        _app_state["retriever"] = get_retriever(
            index_dir=INDEX_DIR,
            score_threshold=SCORE_THRESHOLD,
        )
        _app_state["route_fn"] = route
        _app_state["solve_fn"] = solve_by_task
        logger.info("All components loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load components: {e}")
        # Allow app to start but flag degraded mode
        _app_state["error"] = str(e)

    yield

    logger.info("Shutting down Cikgu AI Kimia...")
    _app_state.clear()


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cikgu AI Kimia",
    description="SPM Chemistry AI Tutor — RAG + Deterministic Solver",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          example="Hitungkan bilangan mol dalam 4.7 g K2O")
    language: str = Field(default="BM", example="BM",
                          description="Response language: BM or EN")
    chapter_filter: Optional[int] = Field(default=None, example=3)
    tingkatan_filter: Optional[int] = Field(default=None, example=4)
    top_k: int = Field(default=5, ge=1, le=10)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    answer_type: str             # "calculation" | "theory" | "fallback"
    sources: List[Dict[str, Any]]
    retrieval_scores: List[float]
    solver_used: bool
    context_found: bool
    processing_time_ms: float
    session_id: Optional[str]


class SolveRequest(BaseModel):
    question: str = Field(..., min_length=3)


class SolveResponse(BaseModel):
    question: str
    task: str
    answer: str
    success: bool
    error: Optional[str] = None


class RetrieveRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    chapter_filter: Optional[int] = None
    tingkatan_filter: Optional[int] = None
    score_threshold: float = Field(default=0.30)
    index_names: Optional[List[str]] = None


class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    count: int


class QuizRequest(BaseModel):
    topic: str = Field(..., example="Konsep Mol")
    chapter: Optional[int] = None
    tingkatan: Optional[int] = None
    question_type: str = Field(default="mcq", example="mcq")
    num_questions: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="BM")


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    index_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# LLM CALL (supports OpenAI and Anthropic)
# ---------------------------------------------------------------------------

async def call_llm(prompt: str, max_tokens: int = 800) -> str:
    """
    Call Groq LLM — fast, free, multilingual.
    Falls back to OpenAI/Anthropic if GROQ_API_KEY not set.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")

    # Primary: Groq
    if groq_key:
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=groq_key)
            resp = await client.chat.completions.create(
                model=groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return f"[Ralat Groq: {e}]"

    # Fallback: OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            resp = await client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[Ralat OpenAI: {e}]"

    return "[LLM tidak dikonfigurasi. Tambah GROQ_API_KEY dalam .env]"


# ---------------------------------------------------------------------------
# CORE ANSWER PIPELINE
# ---------------------------------------------------------------------------

async def answer_question(req: ChatRequest) -> ChatResponse:
    t0 = time.time()

    route_fn = _app_state.get("route_fn")
    solve_fn = _app_state.get("solve_fn")
    retriever = _app_state.get("retriever")

    if not all([route_fn, solve_fn, retriever]):
        raise HTTPException(503, detail="Components not loaded. Check server logs.")

    # ── Step 1: Route the question ─────────────────────────────────────────
    task, data = route_fn(req.question)
    if task == "jmr" and data is None:
        import re
        formulas = re.findall(r"[A-Z][A-Za-z0-9()]+", req.question)
        bad = {"Hitungkan","Tentukan","Berapakah","Jisim","Molar","Formula"}
        formulas = [f for f in formulas if f not in bad]
        if formulas:
            data = {"task": "jmr", "formula": formulas[0], "formulas": formulas}
    solver_used = False
    context_found = False
    sources = []
    retrieval_scores = []

    # ── Step 2A: Calculation path ──────────────────────────────────────────
    if task != "unknown" and data is not None:
        try:
            solver_answer = solve_fn(task, data)
            solver_used = True

            # Optionally augment with retrieved explanation
            rag_results = retriever.retrieve_for_calculation(req.question, k=2)

            if rag_results and retriever.is_sufficient_context(rag_results, min_score=0.35):
                context_found = True
                sources = [
                    {
                        "chunk_id": r.chunk_id,
                        "topic": r.topic,
                        "content_type": r.content_type,
                        "score": round(r.score, 4),
                    }
                    for r in rag_results
                ]
                retrieval_scores = [r.score for r in rag_results]

                # Build explanation prompt
                explanation_prompt = f"""Kamu adalah Cikgu AI Kimia, tutor kimia SPM yang pakar.

Pengiraan berikut telah diselesaikan:

{solver_answer}

Tugas kamu: Terangkan SETIAP LANGKAH pengiraan di atas kepada pelajar SPM dalam Bahasa Malaysia.

Format jawapan:
**Penjelasan Langkah demi Langkah:**

Langkah 1: [nama langkah]
→ [terangkan APA yang dilakukan dan MENGAPA]

Langkah 2: [nama langkah]  
→ [terangkan APA yang dilakukan dan MENGAPA]

[teruskan untuk semua langkah]

**Konsep Penting:**
→ [terangkan konsep utama yang digunakan dalam 2-3 ayat]

**Tip SPM:**
→ [berikan tip atau perkara penting untuk diingat]

PERATURAN:
- Gunakan Bahasa Malaysia SPM yang betul
- Terangkan MENGAPA setiap langkah dilakukan, bukan sekadar APA
- Gunakan bahasa yang mudah difahami pelajar tingkatan 4 dan 5
- Jangan ulang semula pengiraan — TERANGKAN sahaja"""

                explanation = await call_llm(explanation_prompt, max_tokens=600)
                final_answer = solver_answer + "\n\n---\n" + explanation
            else:
                final_answer = solver_answer

            elapsed = (time.time() - t0) * 1000
            return ChatResponse(
                question=req.question,
                answer=final_answer,
                answer_type="calculation",
                sources=sources,
                retrieval_scores=retrieval_scores,
                solver_used=True,
                context_found=context_found,
                processing_time_ms=round(elapsed, 1),
                session_id=req.session_id,
            )

        except Exception as e:
            logger.warning(f"Solver failed for task '{task}': {e}. Falling back to RAG.")

    # ── Step 2B: Theory / RAG path ────────────────────────────────────────
    rag_results = retriever.retrieve(
        query=req.question,
        k=req.top_k,
        chapter_filter=req.chapter_filter,
        tingkatan_filter=req.tingkatan_filter,
    )

    sources = [
        {
            "chunk_id": r.chunk_id,
            "topic": r.topic,
            "subtopic": r.subtopic,
            "content_type": r.content_type,
            "chapter": r.chapter,
            "score": round(r.score, 4),
        }
        for r in rag_results
    ]
    retrieval_scores = [r.score for r in rag_results]

    if retriever.is_sufficient_context(rag_results, min_score=0.25):
        context_found = True
        from retriever import build_rag_prompt
        prompt = build_rag_prompt(req.question, rag_results, max_context_chars=MAX_CONTEXT_CHARS)
        answer = await call_llm(prompt, max_tokens=700)
        answer_type = "theory"
    else:
        # No sufficient context found — anti-hallucination response
        answer = (
            "Maaf, soalan ini tidak terdapat dalam nota kimia saya. "
            "Sila rujuk buku teks SPM atau tanya guru kamu untuk maklumat yang lebih tepat."
        )
        answer_type = "fallback"

    elapsed = (time.time() - t0) * 1000
    return ChatResponse(
        question=req.question,
        answer=answer,
        answer_type=answer_type,
        sources=sources,
        retrieval_scores=retrieval_scores,
        solver_used=False,
        context_found=context_found,
        processing_time_ms=round(elapsed, 1),
        session_id=req.session_id,
    )


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Cikgu AI Kimia API", "version": "1.0.0", "status": "ok"}


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    components = {}
    components["embedder"] = "ok" if "embedder" in _app_state else "not loaded"
    components["retriever"] = "ok" if "retriever" in _app_state else "not loaded"
    components["solver"] = "ok" if "solve_fn" in _app_state else "not loaded"
    components["router"] = "ok" if "route_fn" in _app_state else "not loaded"

    index_stats = {}
    try:
        retriever = _app_state.get("retriever")
        if retriever:
            index_stats = retriever.manager.stats()
    except Exception:
        index_stats = {"error": "could not read index stats"}

    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components, index_stats=index_stats)


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Main Q&A endpoint.
    Routes to deterministic solver for calculations,
    or RAG + LLM for theory / explanation questions.
    """
    return await answer_question(req)


@app.post("/api/solve", response_model=SolveResponse, tags=["Solver"])
async def solve(req: SolveRequest):
    """
    Calculation-only endpoint.
    Uses deterministic Python solver — no LLM.
    Returns SPM-formatted answer or error.
    """
    route_fn = _app_state.get("route_fn")
    solve_fn = _app_state.get("solve_fn")
    if not route_fn or not solve_fn:
        raise HTTPException(503, "Solver not loaded")

    task, data = route_fn(req.question)

    # Fix: handle jmr when data is None
    if task == "jmr" and data is None:
        import re
        formulas = re.findall(r"[A-Z][A-Za-z0-9()]+", req.question)
        bad = {"Hitungkan","Tentukan","Berapakah","Jisim","Molar","Formula","Hitung"}
        formulas = [f for f in formulas if f not in bad]
        if formulas:
            data = {"task": "jmr", "formula": formulas[0], "formulas": formulas}

    if task == "unknown" or data is None:
        return SolveResponse(
            question=req.question,
            task=task,
            answer="",
            success=False,
            error=f"Tidak dapat mengenalpasti jenis pengiraan. Task: {task}",
        )
    try:
        answer = solve_fn(task, data)
        return SolveResponse(
            question=req.question,
            task=task,
            answer=answer,
            success=True,
        )
    except Exception as e:
        return SolveResponse(
            question=req.question,
            task=task,
            answer="",
            success=False,
            error=str(e),
        )


@app.post("/api/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def retrieve(req: RetrieveRequest):
    """
    Raw retrieval endpoint for debugging and testing.
    Returns top-k chunks with scores and metadata.
    """
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")

    results = retriever.retrieve(
        query=req.query,
        k=req.k,
        chapter_filter=req.chapter_filter,
        tingkatan_filter=req.tingkatan_filter,
        score_threshold=req.score_threshold,
        index_names=req.index_names,
    )

    return RetrieveResponse(
        query=req.query,
        results=[
            {
                "rank": r.rank,
                "score": round(r.score, 4),
                "chunk_id": r.chunk_id,
                "topic": r.topic,
                "subtopic": r.subtopic,
                "content_type": r.content_type,
                "chapter": r.chapter,
                "tingkatan": r.tingkatan,
                "has_worked_example": r.has_worked_example,
                "formulas": r.formulas,
                "diagrams": r.diagrams,
                "content_preview": r.content[:300],
            }
            for r in results
        ],
        count=len(results),
    )


@app.post("/api/quiz", tags=["Quiz"])
async def generate_quiz(req: QuizRequest):
    """
    Generate quiz questions from the knowledge base.
    Uses RAG to retrieve relevant content then LLM to form questions.
    """
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")

    # Retrieve relevant content for this topic
    results = retriever.retrieve(
        query=req.topic,
        k=8,
        chapter_filter=req.chapter,
        tingkatan_filter=req.tingkatan,
    )

    if not results:
        raise HTTPException(404, f"Tiada kandungan ditemui untuk topik: {req.topic}")

    context = retriever.build_context(results, max_chars=2500)

    quiz_prompt = f"""Kamu adalah Cikgu AI Kimia. Berdasarkan nota berikut, buat {req.num_questions} soalan {req.question_type.upper()} dalam Bahasa Malaysia.

NOTA:
{context}

FORMAT SOALAN (JSON):
{{
  "questions": [
    {{
      "soalan": "...",
      "pilihan": ["A. ...", "B. ...", "C. ...", "D. ..."],  // hanya untuk MCQ
      "jawapan": "A",
      "penjelasan": "..."
    }}
  ]
}}

Hasilkan HANYA JSON yang sah. Tiada teks lain."""

    raw = await call_llm(quiz_prompt, max_tokens=1200)

    # Parse JSON safely
    import json, re
    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            quiz_data = json.loads(json_match.group())
        else:
            quiz_data = {"raw": raw, "parse_error": True}
    except Exception as e:
        quiz_data = {"raw": raw, "parse_error": str(e)}

    return {
        "topic": req.topic,
        "question_type": req.question_type,
        "num_questions": req.num_questions,
        "quiz": quiz_data,
        "sources_used": len(results),
    }


@app.get("/api/index/stats", tags=["Admin"])
async def index_stats():
    """Return FAISS index statistics."""
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")
    return retriever.manager.stats()


@app.post("/api/index/rebuild", tags=["Admin"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Trigger index rebuild from knowledge base.
    Runs in background — returns immediately.
    """
    kb_dir = os.environ.get("KB_DIR", str(BASE_DIR / "knowledge_base"))

    async def _rebuild():
        try:
            from chunker import chunk_all_files
            from metadata_tagger import tag_chunks
            from indexer import build_indexes_from_chunks

            embedder = _app_state.get("embedder")
            if not embedder:
                logger.error("Embedder not loaded — cannot rebuild")
                return

            logger.info(f"Rebuilding index from {kb_dir}...")
            raw = chunk_all_files(kb_dir)
            tagged = tag_chunks(raw)
            build_indexes_from_chunks(tagged, embedder, index_dir=INDEX_DIR)

            # Reload retriever
            from retriever import ChemistryRetriever
            _app_state["retriever"] = ChemistryRetriever(
                index_dir=INDEX_DIR,
                embedder=embedder,
                score_threshold=SCORE_THRESHOLD,
            )
            logger.info("Index rebuild complete.")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")

    background_tasks.add_task(_rebuild)
    return {"message": "Index rebuild started in background.", "kb_dir": kb_dir}


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
