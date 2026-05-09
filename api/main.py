"""
main.py — Cikgu AI Kimia FastAPI Application
=============================================
Version 3.1.0 — LLM Discipline + Bug Fixes

Changes in v3.1.0:
  - LLM hanya untuk: (1) explain solver output, (2) teori dengan RAG context
  - LLM TIDAK BOLEH jawab pengiraan tanpa solver
  - LLM TIDAK BOLEH jawab teori tanpa RAG context dari nota
  - Jika solver fail + RAG tiada context → bagi mesej jelas, bukan hallucinate
  - Fix Bug #9: jisim larutan betul untuk entalpi
  - Fix Bug #10: mass_from_molarity task baru
  - Fix Bug #7b: ion charge parsing betul
"""

from __future__ import annotations

import os
import sys
import time
import logging
import re
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR / "solver"))
sys.path.insert(0, str(BASE_DIR / "api"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cikgu_ai_kimia")

INDEX_DIR       = os.environ.get("FAISS_INDEX_DIR", str(BASE_DIR / "faiss_indexes"))
SCORE_THRESHOLD = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.30"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "3000"))
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE_URL    = os.environ.get("API_BASE_URL", "")

_app_state: Dict[str, Any] = {}


def detect_language(text: str) -> str:
    en_keywords = [
        'what','why','how','explain','calculate','find','determine','define',
        'describe','compare','difference','between','state','list','give',
        'write','draw','show','the','is','are','of','in','and','with','from',
        'number','moles','mass','volume','concentration','reaction','acid',
        'base','salt','bond',
    ]
    words = text.lower().split()
    en_count = sum(1 for w in words if w in en_keywords)
    return 'EN' if en_count >= 2 else 'BM'


TASK_INDEX_MAP = {
    "moles_from_mass":            ["index_calculations"],
    "moles_from_volume":          ["index_calculations"],
    "moles_multi":                ["index_calculations"],
    "mass_from_moles":            ["index_calculations"],
    "volume_from_moles":          ["index_calculations"],
    "mass_from_volume":           ["index_calculations"],
    "volume_from_mass":           ["index_calculations"],
    "particles_from_moles":       ["index_calculations"],
    "particles_from_mass":        ["index_calculations"],
    "particles_from_volume":      ["index_calculations"],
    "molarity_from_mass":         ["index_calculations"],
    "mass_from_molarity":         ["index_calculations"],
    "concentration_g_dm3":        ["index_calculations"],
    "dilution":                   ["index_calculations"],
    "stoichiometry_mass_to_mass": ["index_calculations"],
    "empirical_formula":          ["index_calculations"],
    "jmr":                        ["index_calculations"],
    "ph_from_h":                  ["index_calculations","index_theory"],
    "h_from_ph":                  ["index_calculations","index_theory"],
    "poh_from_oh":                ["index_calculations","index_theory"],
    "ph_from_poh":                ["index_calculations","index_theory"],
    "titration_find_volume":      ["index_calculations","index_theory"],
    "titration_find_molarity":    ["index_calculations","index_theory"],
    "calorimetry":                ["index_calculations"],
    "delta_h_from_calorimetry":   ["index_calculations"],
    "oxidation_number":           ["index_calculations"],
    "rate_average":               ["index_calculations"],
    "rate_from_points":           ["index_calculations"],
    "ar_from_abundance":          ["index_calculations"],
    "subatomic":                  ["index_calculations"],
    "redox_change":               ["index_calculations"],
}

def get_indexes_for_task(task: str) -> List[str]:
    return TASK_INDEX_MAP.get(task, ["index_calculations","index_theory"])


def translate_solver_output(text: str, lang: str) -> str:
    if lang != 'EN':
        return text
    return (
        text
        .replace("Diberi:",     "Given:")
        .replace("Pengiraan:",  "Calculation:")
        .replace("Jawapan:",    "Answer:")
        .replace("Bilangan mol","Number of moles")
        .replace("Isipadu",     "Volume")
        .replace("Jisim molar", "Molar mass")
        .replace("Jisim",       "Mass")
        .replace("Kemolaran",   "Molarity")
        .replace("Nombor pengoksidaan","Oxidation number")
    )


def build_explanation_prompt(solver_answer: str, lang: str, history: str = "") -> str:
    """LLM role: explain solver output ONLY. No recalculation."""
    if lang == 'EN':
        return f"""You are Cikgu AI Kimia, SPM Chemistry tutor.
{history}
The deterministic solver has produced this answer:
{solver_answer}

Explain in 3-4 sentences ONLY:
1. WHY this formula is used
2. What the key numbers mean
3. One SPM exam tip

STRICT RULES:
- Do NOT recalculate anything
- Do NOT change or question the answer above
- Do NOT add new steps or values
- Be concise"""
    else:
        return f"""Kamu adalah Cikgu AI Kimia, tutor kimia SPM.
{history}
Solver telah menghasilkan jawapan ini:
{solver_answer}

Terangkan dalam 3-4 ayat SAHAJA:
1. KENAPA formula ini digunakan
2. Apa maksud angka-angka penting
3. Satu tip peperiksaan SPM

PERATURAN KETAT:
- JANGAN kira semula
- JANGAN ubah jawapan di atas
- JANGAN tambah langkah atau nilai baru
- Ringkas dan padat"""


def build_theory_prompt(context: str, question: str, lang: str, history: str = "") -> str:
    """LLM role: answer theory using ONLY provided RAG context from notes."""
    if lang == 'EN':
        return f"""You are Cikgu AI Kimia, SPM Chemistry tutor.
{history}
Answer using ONLY the SPM notes below. Do NOT use outside knowledge.
Maximum 5 sentences. Use SPM terminology.

SPM NOTES:
{context}

QUESTION: {question}

If the notes lack sufficient information, say exactly:
"This topic is not fully covered in my notes. Please refer to your SPM textbook."
Do NOT guess."""
    else:
        return f"""Kamu adalah Cikgu AI Kimia, tutor kimia SPM.
{history}
Jawab berdasarkan nota SPM di bawah SAHAJA. JANGAN guna pengetahuan luar.
Maksimum 5 ayat. Guna istilah SPM.

NOTA SPM:
{context}

SOALAN: {question}

Jika nota tidak mencukupi, katakan tepat-tepat:
"Topik ini tidak diliputi sepenuhnya dalam nota saya. Sila rujuk buku teks SPM."
JANGAN teka."""


def fallback_message(lang: str, reason: str = "no_context") -> str:
    """Honest fallback — no LLM hallucination."""
    if lang == 'EN':
        if reason == "solver_fail":
            return (
                "I could not identify the calculation type. "
                "Please include the formula, values, and units clearly."
            )
        return (
            "Sorry, this topic is not in my SPM chemistry notes. "
            "Please refer to your textbook or ask your teacher."
        )
    else:
        if reason == "solver_fail":
            return (
                "Saya tidak dapat mengenalpasti jenis pengiraan ini. "
                "Sila nyatakan formula, nilai, dan unit dengan jelas."
            )
        return (
            "Maaf, topik ini tidak terdapat dalam nota kimia SPM saya. "
            "Sila rujuk buku teks SPM atau tanya guru kamu."
        )


async def setup_telegram(app_instance):
    if not TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
        return
    try:
        from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.ext import (
            Application, CommandHandler, MessageHandler,
            CallbackQueryHandler, ContextTypes, filters,
        )
        from telegram.constants import ChatAction, ParseMode
        import httpx

        def format_answer(answer: str, answer_type: str) -> str:
            emoji = {"calculation":"🧮","theory":"📚","fallback":"ℹ️"}.get(answer_type,"💬")
            lines = []
            for line in answer.split('\n'):
                if line.strip().startswith((
                    'Diberi:','Formula:','Pengiraan:','Jawapan:',
                    'Given:','Calculation:','Answer:'
                )):
                    lines.append(f"*{line.strip()}*")
                else:
                    lines.append(line)
            return f"{emoji} *Jawapan Cikgu AI Kimia*\n\n" + '\n'.join(lines)

        async def call_api(question: str, session_id: str) -> dict:
            lang = detect_language(question)
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"http://localhost:{os.environ.get('PORT', 10000)}/api/chat",
                    json={"question": question, "session_id": session_id, "language": lang, "top_k": 5},
                )
                resp.raise_for_status()
                return resp.json()

        async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            keyboard = [[
                InlineKeyboardButton("📚 Teori", callback_data="mode_theory"),
                InlineKeyboardButton("🧮 Pengiraan", callback_data="mode_calc"),
            ],[
                InlineKeyboardButton("📝 Kuiz", callback_data="mode_quiz"),
                InlineKeyboardButton("❓ Bantuan", callback_data="mode_help"),
            ]]
            await update.message.reply_text(
                "👋 *Selamat datang ke Cikgu AI Kimia!*\n\n"
                "Saya boleh membantu:\n• Pengiraan kimia SPM\n• Teori dan konsep\n"
                "• Kuiz dan latihan\n\nTaip soalan dalam *BM atau English*.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(
                "📚 *Arahan*\n\n/start /help /quiz /solve /clear /stats",
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
                    await update.message.reply_text(f"Tiada soalan untuk: {topic}")
                    return
                msg = f"📝 *Kuiz: {topic}*\n\n"
                for i, q in enumerate(questions, 1):
                    msg += f"*{i}. {q.get('soalan','')}*\n"
                    for opt in q.get('pilihan', []):
                        msg += f"   {opt}\n"
                    msg += f"✅ {q.get('jawapan','')}\n"
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
                        f"🧮 *Pengiraan*\n\n{result['answer']}", parse_mode=ParseMode.MARKDOWN,
                    )
                else:
                    await update.message.reply_text(f"❌ {result.get('error','Gagal.')}")
            except Exception as e:
                await update.message.reply_text(f"Ralat: {e}")

        async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = update.effective_user.id
            try:
                from memory import clear_session
                clear_session(f"tg_{user_id}")
            except Exception:
                pass
            context.user_data.clear()
            await update.message.reply_text("✅ Sesi dikosongkan.")

        async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"http://localhost:{os.environ.get('PORT', 10000)}/api/memory/stats"
                    )
                    stats = resp.json()
                cache = stats.get("cache", {})
                msg = (
                    f"📊 *Statistik*\n\n"
                    f"👥 Pelajar: {stats.get('total_students',0)}\n"
                    f"💾 Cache: {cache.get('total_cached',0)}\n"
                    f"⚡ Hits: {cache.get('total_cache_hits',0)}\n"
                )
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                await update.message.reply_text(f"Ralat: {e}")

        async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            query = update.callback_query
            await query.answer()
            if query.data == "mode_help":
                await cmd_help(update, context)
            elif query.data == "mode_quiz":
                await query.edit_message_text("Taip: /quiz [topik]")
            elif query.data == "mode_theory":
                await query.edit_message_text("📚 *Mod Teori*\nTaip soalan.", parse_mode=ParseMode.MARKDOWN)
            elif query.data == "mode_calc":
                await query.edit_message_text("🧮 *Mod Pengiraan*\nContoh: Hitung mol 4g NaOH", parse_mode=ParseMode.MARKDOWN)

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
                from_cache = result.get("from_cache", False)

                formatted = format_answer(answer, answer_type)
                src_lines = [f"• {s.get('topic','')}" for s in sources[:2] if s.get('topic')]
                if src_lines:
                    formatted += "\n\n📖 _" + ", ".join(src_lines) + "_"
                formatted += f"\n_{'⚡ Cache' if from_cache else '⏱'} {ms:.0f}ms_"

                if len(formatted) > 4096:
                    for chunk in [formatted[i:i+4000] for i in range(0, len(formatted), 4000)]:
                        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                else:
                    await update.message.reply_text(formatted, parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                logger.error(f"Bot error: {e}")
                await update.message.reply_text("Maaf, ralat berlaku. Sila cuba lagi.")

        telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", cmd_start))
        telegram_app.add_handler(CommandHandler("help", cmd_help))
        telegram_app.add_handler(CommandHandler("quiz", cmd_quiz))
        telegram_app.add_handler(CommandHandler("solve", cmd_solve))
        telegram_app.add_handler(CommandHandler("clear", cmd_clear))
        telegram_app.add_handler(CommandHandler("stats", cmd_stats))
        telegram_app.add_handler(CallbackQueryHandler(button_handler))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        await telegram_app.initialize()
        _app_state["telegram_app"] = telegram_app
        webhook_url = f"{API_BASE_URL}/webhook"
        await telegram_app.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
        logger.info(f"Webhook set: {webhook_url}")

    except Exception as e:
        logger.error(f"Telegram setup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cikgu AI Kimia v3.1.0...")
    try:
        from memory import init_db
        init_db()
        _app_state["memory_ok"] = True
    except Exception as e:
        logger.error(f"Memory init failed: {e}")
        _app_state["memory_ok"] = False

    try:
        from embedder import get_embedder
        from retriever import get_retriever
        from router import route
        from solver_engine import solve_by_task
        _app_state["embedder"]  = get_embedder()
        _app_state["retriever"] = get_retriever(index_dir=INDEX_DIR, score_threshold=SCORE_THRESHOLD)
        _app_state["route_fn"]  = route
        _app_state["solve_fn"]  = solve_by_task
        logger.info("All components loaded.")
    except Exception as e:
        logger.error(f"Component load failed: {e}")
        _app_state["error"] = str(e)

    await setup_telegram(app)
    yield
    if "telegram_app" in _app_state:
        await _app_state["telegram_app"].shutdown()
    _app_state.clear()


app = FastAPI(title="Cikgu AI Kimia", version="3.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    language: str = Field(default="auto")
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
    from_cache: bool
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
    question_type: str = Field(default="mcq")
    num_questions: int = Field(default=5, ge=1, le=20)
    language: str = Field(default="BM")

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    index_stats: Dict[str, Any]


async def call_llm(prompt: str, max_tokens: int = 250) -> str:
    groq_key   = os.environ.get("GROQ_API_KEY", "")
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
    return ""


async def answer_question(req: ChatRequest) -> ChatResponse:
    t0 = time.time()
    route_fn  = _app_state.get("route_fn")
    solve_fn  = _app_state.get("solve_fn")
    retriever = _app_state.get("retriever")
    if not all([route_fn, solve_fn, retriever]):
        raise HTTPException(503, detail="Components not loaded.")

    lang = req.language
    if lang == "auto" or lang not in ("BM","EN"):
        lang = detect_language(req.question)
    session_id = req.session_id or "anonymous"

    # Memory lookup
    cached_result = None
    history_context = ""
    if _app_state.get("memory_ok"):
        try:
            from memory import smart_lookup, save_message, save_to_cache, should_cache
            cached_result, history_context = smart_lookup(req.question, session_id, lang)
        except Exception as e:
            logger.warning(f"Memory lookup failed: {e}")

    if cached_result:
        try:
            save_message(session_id, "user", req.question)
            save_message(session_id, "assistant", cached_result["answer"])
        except Exception:
            pass
        elapsed = (time.time() - t0) * 1000
        return ChatResponse(
            question=req.question, answer=cached_result["answer"],
            answer_type=cached_result.get("answer_type","calculation"),
            sources=[], retrieval_scores=[], solver_used=False,
            context_found=True, from_cache=True,
            processing_time_ms=round(elapsed,1), session_id=session_id, language=lang,
        )

    # Route
    task, data = route_fn(req.question)
    if task == "jmr" and data is None:
        formulas = re.findall(r"[A-Z][A-Za-z0-9()]+", req.question)
        bad = {"Hitungkan","Tentukan","Berapakah","Jisim","Molar","Formula","Calculate","Find","Determine","The","What"}
        formulas = [f for f in formulas if f not in bad]
        if formulas:
            data = {"task":"jmr","formula":formulas[0],"formulas":formulas}

    solver_used = False
    context_found = False
    sources = []
    retrieval_scores = []
    final_answer = ""
    answer_type = "fallback"

    # ── PATH 1: CALCULATION ───────────────────────────────────────────────
    if task != "unknown" and data is not None:
        try:
            solver_answer = solve_fn(task, data)
            solver_used   = True

            rag_results = retriever.retrieve(
                query=req.question, k=2,
                index_names=get_indexes_for_task(task),
            )
            if rag_results:
                context_found = True
                sources = [{"chunk_id":r.chunk_id,"topic":r.topic,
                            "content_type":r.content_type,"score":round(r.score,4)}
                           for r in rag_results]
                retrieval_scores = [r.score for r in rag_results]

            # LLM: explain only, no recalculation
            explanation = await call_llm(
                build_explanation_prompt(solver_answer, lang, history_context),
                max_tokens=200,
            )
            translated = translate_solver_output(solver_answer, lang)
            final_answer = translated + "\n\n---\n" + explanation if explanation else translated
            answer_type  = "calculation"

        except Exception as e:
            logger.warning(f"Solver failed task='{task}': {e}")
            final_answer = fallback_message(lang, "solver_fail")
            answer_type  = "fallback"

    # ── PATH 2: THEORY — RAG context required ────────────────────────────
    if not final_answer or answer_type == "fallback":
        rag_results = retriever.retrieve(
            query=req.question, k=req.top_k,
            chapter_filter=req.chapter_filter,
            tingkatan_filter=req.tingkatan_filter,
        )
        sources = [{"chunk_id":r.chunk_id,"topic":r.topic,"subtopic":r.subtopic,
                    "content_type":r.content_type,"chapter":r.chapter,"score":round(r.score,4)}
                   for r in rag_results]
        retrieval_scores = [r.score for r in rag_results]

        if retriever.is_sufficient_context(rag_results, min_score=0.25):
            context_found = True
            context = ""
            chars_used = 0
            for i, r in enumerate(rag_results, 1):
                block = r.context_block
                if chars_used + len(block) > MAX_CONTEXT_CHARS:
                    break
                context    += f"\n--- Petikan {i} ---\n{block}\n"
                chars_used += len(block)

            llm_answer = await call_llm(
                build_theory_prompt(context.strip(), req.question, lang, history_context),
                max_tokens=300,
            )
            final_answer = llm_answer if llm_answer else fallback_message(lang)
            answer_type  = "theory" if llm_answer else "fallback"
        else:
            # No RAG context → honest fallback, no LLM guessing
            final_answer = fallback_message(lang, "no_context")
            answer_type  = "fallback"

    # Save to memory
    if _app_state.get("memory_ok"):
        try:
            save_message(session_id, "user", req.question)
            save_message(session_id, "assistant", final_answer)
            if should_cache(answer_type, task):
                save_to_cache(question=req.question, answer=final_answer,
                              answer_type=answer_type, task=task, language=lang)
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    elapsed = (time.time() - t0) * 1000
    return ChatResponse(
        question=req.question, answer=final_answer, answer_type=answer_type,
        sources=sources, retrieval_scores=retrieval_scores,
        solver_used=solver_used, context_found=context_found, from_cache=False,
        processing_time_ms=round(elapsed,1), session_id=session_id, language=lang,
    )


@app.get("/", tags=["Root"])
async def root():
    return {"message":"Cikgu AI Kimia API","version":"3.1.0","status":"ok"}

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health():
    components = {
        "embedder":  "ok" if "embedder"     in _app_state else "not loaded",
        "retriever": "ok" if "retriever"    in _app_state else "not loaded",
        "solver":    "ok" if "solve_fn"     in _app_state else "not loaded",
        "router":    "ok" if "route_fn"     in _app_state else "not loaded",
        "telegram":  "ok" if "telegram_app" in _app_state else "not loaded",
        "memory":    "ok" if _app_state.get("memory_ok") else "not loaded",
    }
    index_stats = {}
    try:
        retriever = _app_state.get("retriever")
        if retriever:
            index_stats = retriever.manager.stats()
    except Exception:
        index_stats = {"error":"could not read stats"}
    overall = "ok" if all(v=="ok" for v in components.values()) else "degraded"
    return HealthResponse(status=overall, components=components, index_stats=index_stats)

@app.get("/api/memory/stats", tags=["Memory"])
async def memory_stats():
    try:
        from memory import get_all_stats
        return get_all_stats()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/api/memory/session/{session_id}", tags=["Memory"])
async def delete_session(session_id: str):
    try:
        from memory import clear_session
        clear_session(session_id)
        return {"message": f"Session {session_id} cleared"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/webhook", tags=["Telegram"])
async def telegram_webhook(request: Request):
    telegram_app = _app_state.get("telegram_app")
    if not telegram_app:
        raise HTTPException(503, "Telegram bot not initialized")
    try:
        from telegram import Update
        data   = await request.json()
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
        formulas = re.findall(r"[A-Z][A-Za-z0-9()]+", req.question)
        bad = {"Hitungkan","Tentukan","Berapakah","Jisim","Molar","Formula","Hitung"}
        formulas = [f for f in formulas if f not in bad]
        if formulas:
            data = {"task":"jmr","formula":formulas[0],"formulas":formulas}
    if task == "unknown" or data is None:
        return SolveResponse(question=req.question, task=task, answer="", success=False,
                             error=f"Task tidak dikenali: {task}")
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
    results = retriever.retrieve(query=req.query, k=req.k,
                                  chapter_filter=req.chapter_filter,
                                  tingkatan_filter=req.tingkatan_filter,
                                  score_threshold=req.score_threshold,
                                  index_names=req.index_names)
    return RetrieveResponse(
        query=req.query,
        results=[{"rank":r.rank,"score":round(r.score,4),"chunk_id":r.chunk_id,
                  "topic":r.topic,"subtopic":r.subtopic,"content_type":r.content_type,
                  "chapter":r.chapter,"tingkatan":r.tingkatan,
                  "has_worked_example":r.has_worked_example,
                  "formulas":r.formulas,"content_preview":r.content[:300]}
                 for r in results],
        count=len(results),
    )

@app.post("/api/quiz", tags=["Quiz"])
async def generate_quiz(req: QuizRequest):
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")
    results = retriever.retrieve(query=req.topic, k=8, chapter_filter=req.chapter)
    if not results:
        raise HTTPException(404, f"Tiada kandungan: {req.topic}")
    context = ""
    chars = 0
    for i, r in enumerate(results, 1):
        block = r.context_block
        if chars + len(block) > 2500:
            break
        context += f"\n--- Petikan {i} ---\n{block}\n"
        chars   += len(block)
    lang_instr = "in English" if req.language == "EN" else "dalam Bahasa Malaysia"
    quiz_prompt = (
        f"Buat {req.num_questions} soalan {req.question_type.upper()} {lang_instr} "
        f"berdasarkan nota berikut SAHAJA.\n\nNOTA:\n{context}\n\n"
        f'FORMAT JSON sahaja: {{"questions": [{{"soalan":"...","pilihan":["A...","B...","C...","D..."],"jawapan":"A","penjelasan":"..."}}]}}'
    )
    raw = await call_llm(quiz_prompt, max_tokens=1200)
    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        quiz_data  = json.loads(json_match.group()) if json_match else {"raw": raw}
    except Exception as e:
        quiz_data = {"raw": raw, "parse_error": str(e)}
    return {"topic":req.topic,"question_type":req.question_type,
            "num_questions":req.num_questions,"quiz":quiz_data,
            "sources_used":len(results),"language":req.language}

@app.get("/api/index/stats", tags=["Admin"])
async def index_stats():
    retriever = _app_state.get("retriever")
    if not retriever:
        raise HTTPException(503, "Retriever not loaded")
    return retriever.manager.stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
