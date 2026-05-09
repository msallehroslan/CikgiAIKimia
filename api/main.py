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
  - /webhook         — Telegram webhook endpoint

Fixes v2:
  - Always show step-by-step explanation for calculations
  - Fixed RAG source routing per task type
  - Auto language detection (BM/EN) for responses

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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR / "solver"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cikgu_ai_kimia")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", str(BASE_DIR / "faiss_indexes"))
SCORE_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.30"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3000"))
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")

# ---------------------------------------------------------------------------
# APP STATE
# ---------------------------------------------------------------------------

_app_state: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# FIX 3: LANGUAGE DETECTION
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect if question is in English or Bahasa Malaysia.
    Returns 'EN' or 'BM'.
    """
    en_keywords = [
        'what', 'why', 'how', 'explain', 'calculate', 'find', 'determine',
        'define', 'describe', 'compare', 'difference', 'between', 'state',
        'list', 'give', 'write', 'draw', 'show', 'the', 'is', 'are', 'of',
        'in', 'and', 'with', 'from', 'number', 'moles', 'mass', 'volume',
        'concentration', 'reaction', 'acid', 'base', 'salt', 'bond',
    ]
    text_lower = text.lower()
    words = text_lower.split()
    en_count = sum(1 for w in words if w in en_keywords)
    # If more than 2 English keywords found, treat as English
    return 'EN' if en_count >= 2 else 'BM'


# ---------------------------------------------------------------------------
# FIX 2: TASK TO INDEX MAPPING
# ---------------------------------------------------------------------------

TASK_INDEX_MAP = {
    # Mol calculations → index_calculations
    "moles_from_mass":         ["index_calculations"],
    "moles_from_volume":       ["index_calculations"],
    "mass_from_moles":         ["index_calculations"],
    "volume_from_moles":       ["index_calculations"],
    "mass_from_volume":        ["index_calculations"],
    "volume_from_mass":        ["index_calculations"],
    "particles_from_moles":    ["index_calculations"],
    "particles_from_mass":     ["index_calculations"],
    "particles_from_volume":   ["index_calculations"],
    "molarity_from_mass":      ["index_calculations"],
    "concentration_g_dm3":     ["index_calculations"],
    "dilution":                ["index_calculations"],
    "stoichiometry_mass_to_mass": ["index_calculations"],
    "empirical_formula":       ["index_calculations"],
    "jmr":                     ["index_calculations"],
    # Acid/Base → index_calculations + index_theory
    "ph_from_h":               ["index_calculations", "index_theory"],
    "h_from_ph":               ["index_calculations", "index_theory"],
    "poh_from_oh":             ["index_calculations", "index_theory"],
    "ph_from_poh":             ["index_calculations", "index_theory"],
    "titration_find_volume":   ["index_calculations", "index_theory"],
    # Thermochemistry → index_calculations
    "calorimetry":             ["index_calculations"],
    "delta_h_from_calorimetry": ["index_calculations"],
    # Redox → index_calculations
    "oxidation_number":        ["index_calculations"],
    # Rate → index_calculations
    "rate_average":            ["index_calculations"],
    # Atomic structure → index_calculations
    "ar_from_abundance":       ["index_calculations"],
    "subatomic":               ["index_calculations"],
}

def get_indexes_for_task(task: str) -> List[str]:
    return TASK_INDEX_MAP.get(task, ["index_calculations", "index_theory"])


# ---------------------------------------------------------------------------
# TELEGRAM BOT SETUP
# ---------------------------------------------------------------------------

async def setup_telegram(app_instance):
    """Initialize telegram bot and set webhook."""
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
        return

    try:
        from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.ext import (
            Application,
            CommandHandler,
            MessageHandler,
            CallbackQueryHandler,
            ContextTypes,
            filters,
        )
        from telegram.constants import ChatAction, ParseMode
        import httpx

        def format_answer(answer: str, answer_type: str) -> str:
            emoji = {"calculation": "🧮", "theory": "📚", "fallback": "ℹ️"}.get(answer_type, "💬")
            lines = []
            for line in answer.split('\n'):
                if line.strip().startswith(('Diberi:', 'Formula:', 'Pengiraan:', 'Jawapan:',
                                            'Given:', 'Formula:', 'Calculation:', 'Answer:')):
                    lines.append(f"*{line.strip()}*")
                else:
                    lines.append(line)
            return f"{emoji} *Jawapan Cikgu AI Kimia*\n\n" + '\n'.join(lines)

        async def call_api(question: str, session_id: str) -> dict:
            lang = detect_language(question)
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"http://localhost:{os.environ.get('PORT', 10000)}/api/chat",
                    json={
                        "question": question,
                        "session_id": session_id,
                        "language": lang,
                        "top_k": 5,
                    },
                )
                resp.raise_for_status()
                return resp.json()

        async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            keyboard = [
                [
                    InlineKeyboardButton("📚 Teori", callback_data="mode_theory"),
                    InlineKeyboardButton("🧮 Pengiraan", callback_data="mode_calc"),
                ],
                [
                    InlineKeyboardButton("📝 Kuiz", callback_data="mode_quiz"),
                    InlineKeyboardButton("❓ Bantuan", callback_data="mode_help"),
                ],
            ]
            await update.message.reply_text(
                "👋 *Selamat datang ke Cikgu AI Kimia!*\n\n"
                "Saya boleh membantu anda dalam:\n"
                "• Pengiraan kimia SPM (langkah demi langkah)\n"
                "• Teori dan konsep kimia\n"
                "• Soalan latihan dan kuiz\n\n"
                "Taip soalan dalam *Bahasa Malaysia atau English*.\n\n"
                "Contoh:\n"
                "_Hitungkan bilangan mol dalam 4.7 g K₂O_\n"
                "_Calculate the pH if H+ concentration is 0.01 mol/dm3_",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(
                "📚 *Cikgu AI Kimia — Arahan*\n\n"
                "/start — Halaman utama\n"
                "/help — Arahan ini\n"
                "/quiz [topik] — Jana soalan kuiz\n"
                "/solve [soalan] — Pengiraan sahaja\n"
                "/clear — Kosongkan tetapan sesi\n\n"
                "*Bahasa:* BM dan English disokong\n\n"
                "*Format jawapan pengiraan:*\n"
                "Diberi: → Formula: → Pengiraan: → Jawapan:",
                parse_mode=ParseMode.MARKDOWN,
            )

        async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
            topic = " ".join(context.args) if context.args else "Konsep Mol"
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"http://localhost:{os.environ.get('PORT', 10000)}/api/quiz",
                        json={"topic": topic, "num_questions": 3, "language": "BM"},
                    )
                    result = resp.json()
                questions = result.get("quiz", {}).get("questions", [])
                if not questions:
                    await update.message.reply_text(f"Tiada soalan untuk topik: {topic}")
                    return
                msg = f"📝 *Kuiz: {topic}*\n\n"
                for i, q in enumerate(questions, 1):
                    msg += f"*{i}. {q.get('soalan', '')}*\n"
                    for opt in q.get('pilihan', []):
                        msg += f"   {opt}\n"
                    msg += f"✅ {q.get('jawapan', '')}\n"
                    if q.get('penjelasan'):
                        msg += f"💡 _{q['penjelasan']}_\n"
                    msg += "\n"
                if len(msg) > 4000:
                    msg = msg[:4000] + "\n_[dipendekkan]_"
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                await update.message.reply_text(f"Ralat: {e}")

        async def cmd_solve(update: Update, context: ContextTypes.DEFAULT_TYPE):
            question = " ".join(context.args) if context.args else ""
            if not question:
                await update.message.reply_text("Contoh: /solve Hitung mol 2g H2O")
                return
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"http://localhost:{os.environ.get('PORT', 10000)}/api/solve",
                        json={"question": question},
                    )
                    result = resp.json()
                if result.get("success"):
                    await update.message.reply_text(
                        f"🧮 *Pengiraan*\n\n{result['answer']}",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                else:
                    await update.message.reply_text(f"❌ {result.get('error', 'Gagal.')}")
            except Exception as e:
                await update.message.reply_text(f"Ralat: {e}")

        async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
            context.user_data.clear()
            await update.message.reply_text("✅ Tetapan sesi dikosongkan.")

        async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            query = update.callback_query
            await query.answer()
            data = query.data
            if data == "mode_help":
                await cmd_help(update, context)
            elif data == "mode_quiz":
                await query.edit_message_text("Taip: /quiz [topik]\nContoh: /quiz Konsep Mol")
            elif data == "mode_theory":
                await query.edit_message_text(
                    "📚 *Mod Teori*\nTaip soalan teori anda.\nContoh: _Apakah maksud pH?_",
                    parse_mode=ParseMode.MARKDOWN,
                )
            elif data == "mode_calc":
                await query.edit_message_text(
                    "🧮 *Mod Pengiraan*\nTaip soalan pengiraan anda.\n"
                    "Contoh: _Hitungkan bilangan mol dalam 4.7 g K₂O_",
                    parse_mode=ParseMode.MARKDOWN,
                )

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            question = update.message.text.strip()
            if not question or len(question) < 3:
                return
            user_id = update.effective_user.id
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                result = await call_api(question, session_id=f"tg_{user_id}")
                answer = result.get("answer", "Maaf, tiada jawapan.")
                answer_type = result.get("answer_type", "fallback")
                sources = result.get("sources", [])
                ms = result.get("processing_time_ms", 0)

                formatted = format_answer(answer, answer_type)

                src_lines = [f"• {s.get('topic','')}" for s in sources[:2] if s.get('topic')]
                if src_lines:
                    formatted += "\n\n📖 _Sumber: " + ", ".join(src_lines) + "_"
                formatted += f"\n\n_⏱ {ms:.0f}ms_"

                if len(formatted) > 4096:
                    for chunk in [formatted[i:i+4000] for i in range(0, len(formatted), 4000)]:
                        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                else:
                    await update.message.reply_text(formatted, parse_mode=ParseMode.MARKDOWN)

            except Exception as e:
                logger.error(f"Bot message error: {e}")
                await update.message.reply_text("Maaf, ralat berlaku. Sila cuba lagi.")

        telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", cmd_start))
        telegram_app.add_handler(CommandHandler("help", cmd_help))
        telegram_app.add_handler(CommandHandler("quiz", cmd_quiz))
        telegram_app.add_handler(CommandHandler("solve", cmd_solve))
        telegram_app.add_handler(CommandHandler("clear", cmd_clear))
        telegram_app.add_handler(CallbackQueryHandler(button_handler))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        await telegram_app.initialize()
        _app_state["telegram_app"] = telegram_app

        webhook_url = f"{API_BASE_URL}/webhook"
        await telegram_app.bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=True,
        )
        logger.info(f"Telegram webhook set: {webhook_url}")

    except Exception as e:
        logger.error(f"Telegram setup failed: {e}")


# ---------------------------------------------------------------------------
# APP LIFESPAN
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        _app_state["error"] = str(e)

    await setup_telegram(app)

    yield

    if "telegram_app" in _app_state:
        await _app_state["telegram_app"].shutdown()

    logger.info("Shutting down Cikgu AI Kimia...")
    _app_state.clear()


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cikgu AI Kimia",
    description="SPM Chemistry AI Tutor — RAG + Deterministic Solver",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    language: str = Field(default="auto")  # auto, BM, EN
    chapter_filter: Optional[int] = None
    tingkatan_filter: Optional[int] = None
    top_k: int = Field(default=5, ge=1, le=10)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    answer_type: str
    sources: List[Dict[str, Any]]
    retrieval_scores: List[float]
    solver_used: bool
    context_found: bool
    processing_time_ms: float
    session_id: Optional[str]
    language: str


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
    question_type: str = Field(default="mcq")
    num_questions: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="BM")


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    index_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# LLM CALL
# ---------------------------------------------------------------------------

async def call_llm(prompt: str, max_tokens: int = 800) -> str:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")

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

    return "[LLM tidak dikonfigurasi. Tambah GROQ_API_KEY]"


# ---------------------------------------------------------------------------
# FIX 3: LANGUAGE-AWARE EXPLANATION PROMPT
# ---------------------------------------------------------------------------

def build_explanation_prompt(solver_answer: str, lang: str) -> str:
    if lang == 'EN':
        return f"""You are Cikgu AI Kimia, an SPM Chemistry tutor.

The following calculation has been solved:

{solver_answer}

Your task: Explain EVERY STEP of the calculation above to an SPM student in ENGLISH.

Format:
**Step-by-Step Explanation:**

Step 1: [step name]
→ [explain WHAT was done and WHY]

Step 2: [step name]
→ [explain WHAT was done and WHY]

**Key Concept:**
→ [explain the main concept in 2-3 sentences]

**SPM Tip:**
→ [important tip to remember]

RULES:
- Use clear English suitable for SPM students
- Explain WHY each step is done, not just WHAT
- Do NOT repeat the calculation — EXPLAIN only"""
    else:
        return f"""Kamu adalah Cikgu AI Kimia, tutor kimia SPM yang pakar.

Pengiraan berikut telah diselesaikan:

{solver_answer}

Tugas kamu: Terangkan SETIAP LANGKAH pengiraan di atas kepada pelajar SPM dalam Bahasa Malaysia.

Format jawapan:
**Penjelasan Langkah demi Langkah:**

Langkah 1: [nama langkah]
→ [terangkan APA yang dilakukan dan MENGAPA]

Langkah 2: [nama langkah]
→ [terangkan APA yang dilakukan dan MENGAPA]

**Konsep Penting:**
→ [terangkan konsep utama dalam 2-3 ayat]

**Tip SPM:**
→ [tip penting untuk diingat]

PERATURAN:
- Gunakan Bahasa Malaysia SPM yang betul
- Terangkan MENGAPA setiap langkah dilakukan
- Jangan ulang semula pengiraan — TERANGKAN sahaja"""


def build_theory_prompt(context: str, question: str, lang: str) -> str:
    if lang == 'EN':
        return f"""You are Cikgu AI Kimia, an expert SPM Chemistry tutor.

IMPORTANT RULES:
1. Answer ONLY based on the reference notes below.
2. Answer in clear, complete English sentences.
3. Use proper SPM Chemistry terminology.
4. Include relevant examples from the notes.
5. Do NOT fabricate facts or formulas not in the notes.

REFERENCE NOTES:
{context}

STUDENT QUESTION:
{question}"""
    else:
        return f"""Kamu adalah Cikgu AI Kimia, tutor kimia SPM yang pakar.

PERATURAN PENTING:
1. Jawab HANYA berdasarkan petikan nota yang diberikan di bawah.
2. Jawab dalam bentuk ayat yang jelas dan lengkap dalam Bahasa Malaysia.
3. Kekalkan terminologi Bahasa Malaysia SPM.
4. Sertakan contoh yang relevan daripada nota.
5. JANGAN reka fakta atau formula yang tidak ada dalam nota.

NOTA RUJUKAN:
{context}

SOALAN PELAJAR:
{question}"""


# ---------------------------------------------------------------------------
# CORE ANSWER PIPELINE
# ---------------------------------------------------------------------------

async def answer_question(req: ChatRequest) -> ChatResponse:
    t0 = time.time()

    route_fn = _app_state.get("route_fn")
    solve_fn = _app_state.get("solve_fn")
    retriever = _app_state.get("retriever")

    if not all([route_fn, solve_fn, retriever]):
        raise HTTPException(503, detail="Components not loaded.")

    # FIX 3: Detect language
    lang = req.language
    if lang == "auto" or lang not in ("BM", "EN"):
        lang = detect_language(req.question)

    task, data = route_fn(req.question)
    if task == "jmr" and data is None:
        import re
        formulas = re.findall(r"[A-Z][A-Za-z0-9()]+", req.question)
        bad = {"Hitungkan", "Tentukan", "Berapakah", "Jisim", "Molar", "Formula",
               "Calculate", "Find", "Determine", "The", "What"}
        formulas = [f for f in formulas if f not in bad]
        if formulas:
            data = {"task": "jmr", "formula": formulas[0], "formulas": formulas}

    solver_used = False
    context_found = False
    sources = []
    retrieval_scores = []

    # ── Calculation path ───────────────────────────────────────────────────
    if task != "unknown" and data is not None:
        try:
            solver_answer = solve_fn(task, data)
            solver_used = True

            # FIX 2: Use correct index based on task type
            target_indexes = get_indexes_for_task(task)
            rag_results = retriever.retrieve(
                query=req.question,
                k=2,
                index_names=target_indexes,
            )

            if rag_results:
                context_found = True
                sources = [
                    {"chunk_id": r.chunk_id, "topic": r.topic,
                     "content_type": r.content_type, "score": round(r.score, 4)}
                    for r in rag_results
                ]
                retrieval_scores = [r.score for r in rag_results]

            # FIX 1: ALWAYS explain — no score threshold check
            # FIX 3: Use language-aware prompt
            explanation_prompt = build_explanation_prompt(solver_answer, lang)
            explanation = await call_llm(explanation_prompt, max_tokens=600)
            final_answer = solver_answer + "\n\n---\n" + explanation

            elapsed = (time.time() - t0) * 1000
            return ChatResponse(
                question=req.question, answer=final_answer,
                answer_type="calculation", sources=sources,
                retrieval_scores=retrieval_scores, solver_used=True,
                context_found=context_found,
                processing_time_ms=round(elapsed, 1),
                session_id=req.session_id,
                language=lang,
            )

        except Exception as e:
            logger.warning(f"Solver failed for task '{task}': {e}. Falling back to RAG.")

    # ── Theory / RAG path ──────────────────────────────────────────────────
    rag_results = retriever.retrieve(
        query=req.question, k=req.top_k,
        chapter_filter=req.chapter_filter,
        tingkatan_filter=req.tingkatan_filter,
    )

    sources = [
        {"chunk_id": r.chunk_id, "topic": r.topic, "subtopic": r.subtopic,
         "content_type": r.content_type, "chapter": r.chapter, "score": round(r.score, 4)}
        for r in rag_results
    ]
    retrieval_scores = [r.score for r in rag_results]

    if retriever.is_sufficient_context(rag_results, min_score=0.25):
        context_found = True
        # Build context string
        context = ""
        chars_used = 0
        for i, r in enumerate(rag_results, 1):
            block = r.context_block
            if chars_used + len(block) > MAX_CONTEXT_CHARS:
                break
            context += f"\n--- Petikan {i} (Skor: {r.score:.2f}) ---\n{block}\n"
            chars_used += len(block)

        # FIX 3: Language-aware theory prompt
        prompt = build_theory_prompt(context.strip(), req.question, lang)
        answer = await call_llm(prompt, max_tokens=700)
        answer_type = "theory"
    else:
        # FIX 3: Language-aware fallback message
        if lang == 'EN':
            answer = (
                "Sorry, this question is not found in my chemistry notes. "
                "Please refer to your SPM textbook or ask your teacher for more accurate information."
            )
        else:
            answer = (
                "Maaf, soalan ini tidak terdapat dalam nota kimia saya. "
                "Sila rujuk buku teks SPM atau tanya guru kamu untuk maklumat yang lebih tepat."
            )
        answer_type = "fallback"

    elapsed = (time.time() - t0) * 1000
    return ChatResponse(
        question=req.question, answer=answer, answer_type=answer_type,
        sources=sources, retrieval_scores=retrieval_scores,
        solver_used=False, context_found=context_found,
        processing_time_ms=round(elapsed, 1),
        session_id=req.session_id,
        language=lang,
    )


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    return {"message": "Cikgu AI Kimia API", "version": "2.0.0", "status": "ok"}


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    components = {
        "embedder": "ok" if "embedder" in _app_state else "not loaded",
        "retriever": "ok" if "retriever" in _app_state else "not loaded",
        "solver": "ok" if "solve_fn" in _app_state else "not loaded",
        "router": "ok" if "route_fn" in _app_state else "not loaded",
        "telegram": "ok" if "telegram_app" in _app_state else "not loaded",
    }
    index_stats = {}
    try:
        retriever = _app_state.get("retriever")
        if retriever:
            index_stats = retriever.manager.stats()
    except Exception:
        index_stats = {"error": "could not read index stats"}

    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components, index_stats=index_stats)


@app.post("/webhook", tags=["Telegram"])
async def telegram_webhook(request: Request):
    """Receive Telegram updates via webhook."""
    telegram_app = _app_state.get("telegram_app")
    if not telegram_app:
        raise HTTPException(503, "Telegram bot not initialized")

    try:
        from telegram import Update
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    return await answer_question(req)


@app.post("/api/solve", response_model=SolveResponse, tags=["Solver"])
async def solve(req: SolveRequest):
    route_fn = _app_state.get("route_fn")
    solve_fn = _app_state.get("solve_fn")
    if not route_fn or not solve_fn:
        raise HTTPException(503, "Solver not loaded")

    task, data = route_fn(req.question)
    if task == "jmr" and data is None:
        import re
        formulas = re.findall(r"[A-Z][A-Za-z0-9()]+", req.question)
        bad = {"Hitungkan", "Tentukan", "Berapakah", "Jisim", "Molar", "Formula", "Hitung"}
        formulas = [f for f in formulas if f not in bad]
        if formulas:
            data = {"task": "jmr", "formula": formulas[0], "formulas": formulas}

    if task == "unknown" or data is None:
        return SolveResponse(
            question=req.question, task=task, answer="", success=False,
            error=f"Tidak dapat mengenalpasti jenis pengiraan. Task: {task}",
        )
    try:
        answer = solve_fn(task, data)
        return SolveResponse(question=req.question, task=task, answer=answer, success=True)
    except Exception as e:
        return SolveResponse(question=req.question, task=task, answer="", success=False, error=str(e))


@app.post("/api/retrieve", response_model=RetrieveResponse, tags=["RAG"])
async def retrieve(req: RetrieveRequest):
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")

    results = retriever.retrieve(
        query=req.query, k=req.k,
        chapter_filter=req.chapter_filter,
        tingkatan_filter=req.tingkatan_filter,
        score_threshold=req.score_threshold,
        index_names=req.index_names,
    )
    return RetrieveResponse(
        query=req.query,
        results=[
            {"rank": r.rank, "score": round(r.score, 4), "chunk_id": r.chunk_id,
             "topic": r.topic, "subtopic": r.subtopic, "content_type": r.content_type,
             "chapter": r.chapter, "tingkatan": r.tingkatan,
             "has_worked_example": r.has_worked_example,
             "formulas": r.formulas, "content_preview": r.content[:300]}
            for r in results
        ],
        count=len(results),
    )


@app.post("/api/quiz", tags=["Quiz"])
async def generate_quiz(req: QuizRequest):
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")

    results = retriever.retrieve(query=req.topic, k=8, chapter_filter=req.chapter)
    if not results:
        raise HTTPException(404, f"Tiada kandungan untuk topik: {req.topic}")

    context = ""
    chars = 0
    for i, r in enumerate(results, 1):
        block = r.context_block
        if chars + len(block) > 2500:
            break
        context += f"\n--- Petikan {i} ---\n{block}\n"
        chars += len(block)

    lang_instruction = "in English" if req.language == "EN" else "dalam Bahasa Malaysia"
    quiz_prompt = f"""Kamu adalah Cikgu AI Kimia. Berdasarkan nota berikut, buat {req.num_questions} soalan {req.question_type.upper()} {lang_instruction}.

NOTA:
{context}

FORMAT (JSON sahaja, tiada teks lain):
{{"questions": [{{"soalan": "...", "pilihan": ["A. ...", "B. ...", "C. ...", "D. ..."], "jawapan": "A", "penjelasan": "..."}}]}}"""

    raw = await call_llm(quiz_prompt, max_tokens=1200)

    import json, re
    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        quiz_data = json.loads(json_match.group()) if json_match else {"raw": raw}
    except Exception as e:
        quiz_data = {"raw": raw, "parse_error": str(e)}

    return {"topic": req.topic, "question_type": req.question_type,
            "num_questions": req.num_questions, "quiz": quiz_data,
            "sources_used": len(results), "language": req.language}


@app.get("/api/index/stats", tags=["Admin"])
async def index_stats():
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")
    return retriever.manager.stats()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
