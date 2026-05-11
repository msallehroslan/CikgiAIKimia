"""
telegram_bot.py — Cikgu AI Kimia Telegram Bot
===============================================
Production Telegram bot (PATCHED v3.3.0):
  - ParseMode.HTML throughout
  - format_answer_telegram uses HTML (safe for chemistry unicode)
  - Local OCR → vision API routing
  - Pending vision confirmation flow

Original description:
  - Text question handling
  - Photo/image handling (OCR for diagram questions)
  - /start, /help, /quiz, /chapter commands
  - Session tracking per user
  - Typing indicator while processing
  - SPM-formatted responses
  - Rate limiting per user

Calls the FastAPI backend at API_BASE_URL.

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

import httpx
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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("cikgu_bot")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30.0

# Rate limiting: max requests per minute per user
RATE_LIMIT_RPM = 20
_user_timestamps: Dict[int, list] = {}


# ---------------------------------------------------------------------------
# RATE LIMITING
# ---------------------------------------------------------------------------

def is_rate_limited(user_id: int) -> bool:
    now = time.time()
    ts = _user_timestamps.setdefault(user_id, [])
    _user_timestamps[user_id] = [t for t in ts if now - t < 60]
    if len(_user_timestamps[user_id]) >= RATE_LIMIT_RPM:
        return True
    _user_timestamps[user_id].append(now)
    return False


# ---------------------------------------------------------------------------
# API CLIENT
# ---------------------------------------------------------------------------

async def call_chat_api(
    question: str,
    session_id: str,
    language: str = "BM",
    chapter_filter: Optional[int] = None,
) -> dict:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        payload = {
            "question": question,
            "language": language,
            "session_id": session_id,
            "top_k": 5,
        }
        if chapter_filter:
            payload["chapter_filter"] = chapter_filter

        resp = await client.post(f"{API_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


async def call_solve_api(question: str) -> dict:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{API_BASE_URL}/api/solve", json={"question": question})
        resp.raise_for_status()
        return resp.json()


async def call_quiz_api(topic: str, question_type: str = "mcq", n: int = 3) -> dict:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        payload = {
            "topic": topic,
            "question_type": question_type,
            "num_questions": n,
            "language": "BM",
        }
        resp = await client.post(f"{API_BASE_URL}/api/quiz", json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# MESSAGE FORMATTERS
# ---------------------------------------------------------------------------

def format_answer_telegram(answer: str, answer_type: str) -> str:
    """PATCHED v3.3.0 — HTML mode. Safe for chemistry unicode (ΔH, ⁻¹, °C, →)."""
    import html as _html
    TYPE_EMOJI = {
        "calculation": "🧮",
        "theory": "📚",
        "fallback": "ℹ️",
    }
    emoji = TYPE_EMOJI.get(answer_type, "💬")

    headers = ('Diberi:', 'Formula:', 'Pengiraan:', 'Jawapan:',
               'Given:', 'Calculation:', 'Answer:')
    lines = []
    for line in answer.split('\n'):
        if line.strip().startswith(headers):
            lines.append(f"<b>{_html.escape(line.strip())}</b>")
        else:
            lines.append(_html.escape(line))

    formatted = '\n'.join(lines)
    return f"{emoji} <b>Jawapan Cikgu AI Kimia</b>\n\n{formatted}"


def format_sources(sources: list) -> str:
    """PATCHED v3.3.0 — HTML mode."""
    import html as _html
    if not sources:
        return ""
    source_lines = []
    for s in sources[:3]:
        topic = s.get('topic', '')
        ch    = s.get('chapter', '')
        if topic:
            source_lines.append(f"• {_html.escape(topic)}" + (f" (Bab {ch})" if ch else ""))
    if source_lines:
        return "\n\n<i>📖 Sumber: " + ", ".join(source_lines) + "</i>"
    return ""


# ---------------------------------------------------------------------------
# OCR FOR DIAGRAM QUESTIONS
# ---------------------------------------------------------------------------

async def ocr_image(file_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang='eng+msa')
        return text.strip()
    except ImportError:
        return "[pytesseract tidak dipasang — gambar tidak dapat dibaca]"
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# COMMAND HANDLERS
# ---------------------------------------------------------------------------

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
    reply_markup = InlineKeyboardMarkup(keyboard)

    welcome = (
        "👋 <b>Selamat datang ke Cikgu AI Kimia!</b>\n\n"
        "Saya boleh membantu anda dalam:\n"
        "• Pengiraan kimia SPM (langkah demi langkah)\n"
        "• Teori dan konsep kimia\n"
        "• Soalan latihan dan kuiz\n\n"
        "Taip soalan anda dalam Bahasa Malaysia atau English.\n\n"
        "Contoh:\n"
        "<i>Hitungkan bilangan mol dalam 4.7 g K₂O</i>\n"
        "<i>Terangkan perbezaan eksotermik dan endotermik</i>"
    )
    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📚 <b>Cikgu AI Kimia — Arahan</b>\n\n"
        "/start — Halaman utama\n"
        "/help — Arahan ini\n"
        "/quiz [topik] — Jana soalan kuiz\n"
        "/solve [soalan] — Pengiraan sahaja\n"
        "/chapter [nombor] — Tetapkan penapis bab\n"
        "/clear — Kosongkan tetapan sesi\n\n"
        "<b>Format jawapan pengiraan:</b>\n"
        "Diberi: → Formula: → Pengiraan: → Jawapan:"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = " ".join(context.args) if context.args else "Konsep Mol"
    user_id = update.effective_user.id

    if is_rate_limited(user_id):
        await update.message.reply_text("⏳ Sila tunggu sebentar sebelum hantar soalan baru.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        result = await call_quiz_api(topic, n=3)
        quiz_data = result.get("quiz", {})
        questions = quiz_data.get("questions", [])

        if not questions:
            await update.message.reply_text(f"Tiada soalan ditemui untuk topik: {topic}")
            return

        msg = f"📝 <b>Kuiz: {topic}</b>\n\n"
        for i, q in enumerate(questions, 1):
            msg += f"<b>{i}. {q.get('soalan', '')}</b>\n"
            for opt in q.get('pilihan', []):
                msg += f"   {opt}\n"
            msg += f"✅ Jawapan: {q.get('jawapan', '')}\n"
            if q.get('penjelasan'):
                msg += f"💡 <i>{q['penjelasan']}</i>\n"
            msg += "\n"

        if len(msg) > 4000:
            msg = msg[:4000] + "\n\n_[Soalan dipendekkan]_"

        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Quiz error: {e}")
        await update.message.reply_text(f"Ralat menjana kuiz: {e}")


async def cmd_solve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = " ".join(context.args) if context.args else ""
    if not question:
        await update.message.reply_text("Sila masukkan soalan. Contoh: /solve Hitung mol 2g H2O")
        return

    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("⏳ Sila tunggu sebentar.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        result = await call_solve_api(question)
        if result.get("success"):
            await update.message.reply_text(
                f"🧮 <b>Pengiraan</b>\n\n{result['answer']}",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                f"❌ {result.get('error', 'Pengiraan gagal.')}"
            )
    except Exception as e:
        await update.message.reply_text(f"Ralat: {e}")


async def cmd_chapter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        try:
            ch = int(context.args[0])
            context.user_data["chapter_filter"] = ch
            await update.message.reply_text(f"✅ Penapis bab ditetapkan ke Bab {ch}.")
        except ValueError:
            await update.message.reply_text("Sila masukkan nombor bab. Contoh: /chapter 3")
    else:
        context.user_data.pop("chapter_filter", None)
        await update.message.reply_text("✅ Penapis bab dibuang.")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("✅ Tetapan sesi dikosongkan.")


# ---------------------------------------------------------------------------
# INLINE BUTTON HANDLER
# ---------------------------------------------------------------------------

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "mode_help":
        await cmd_help(update, context)
    elif data == "mode_quiz":
        await query.edit_message_text("Taip topik kuiz anda. Contoh:\n/quiz Konsep Mol")
    elif data == "mode_theory":
        await query.edit_message_text(
            "📚 <b>Mod Teori</b>\nTaip soalan teori anda.\nContoh: <i>Apakah maksud pH?</i>",
            parse_mode=ParseMode.HTML,
        )
    elif data == "mode_calc":
        await query.edit_message_text(
            "🧮 <b>Mod Pengiraan</b>\nTaip soalan pengiraan anda.\n"
            "Contoh: _Hitungkan bilangan mol dalam 4.7 g K₂O_",
            parse_mode=ParseMode.HTML,
        )


# ---------------------------------------------------------------------------
# MAIN MESSAGE HANDLER
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id  = update.effective_user.id
    question = update.message.text.strip()

    if not question or len(question) < 3:
        return

    # PATCHED v3.3.0 — pending vision confirmation flow
    if question.lower() in ("ya", "yes", "betul", "correct", "ok"):
        pending = context.user_data.pop("pending_vision_question", None)
        if pending:
            question = pending

    if is_rate_limited(user_id):
        await update.message.reply_text("⏳ Terlalu banyak permintaan. Sila cuba sebentar lagi.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    session_id = f"tg_{user_id}"
    chapter_filter = context.user_data.get("chapter_filter")

    try:
        result = await call_chat_api(
            question=question,
            session_id=session_id,
            chapter_filter=chapter_filter,
        )

        answer = result.get("answer", "Maaf, tiada jawapan ditemui.")
        answer_type = result.get("answer_type", "fallback")
        sources = result.get("sources", [])
        processing_ms = result.get("processing_time_ms", 0)

        formatted = format_answer_telegram(answer, answer_type)
        formatted += format_sources(sources)
        formatted += f"\n\n<i>⏱ {processing_ms:.0f}ms</i>"

        if len(formatted) > 4096:
            chunks = [formatted[i:i+4000] for i in range(0, len(formatted), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(formatted, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Message handler error: {e}")
        await update.message.reply_text(
            "Maaf, ralat berlaku. Sila cuba lagi sebentar.",
        )


# ---------------------------------------------------------------------------
# PHOTO / DIAGRAM HANDLER
# ---------------------------------------------------------------------------

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if is_rate_limited(user_id):
        await update.message.reply_text("⏳ Sila tunggu sebentar.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text("📷 Memproses gambar... Sila tunggu.")

    photo = update.message.photo[-1]
    photo_file = await photo.get_file()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await photo_file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    extracted = await ocr_image(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    caption = update.message.caption or ""
    question = (caption + " " + extracted).strip()

    if len(question) < 5:
        await update.message.reply_text(
            "Tidak dapat membaca teks daripada gambar. "
            "Sila taip soalan anda sebagai teks."
        )
        return

    await update.message.reply_text(
        import html as _html_bot
        f"🔍 Teks dijumpai dalam gambar:\n<code>{_html_bot.escape(question[:200])}</code>",
        parse_mode=ParseMode.HTML,
    )

    try:
        result = await call_chat_api(
            question=question[:500],
            session_id=f"tg_{user_id}",
        )
        answer = result.get("answer", "Maaf, tiada jawapan.")
        formatted = format_answer_telegram(answer, result.get("answer_type", "theory"))
        await update.message.reply_text(formatted, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"Ralat memproses gambar: {e}")


# ---------------------------------------------------------------------------
# MAIN — POLLING MODE (no webhook needed)
# ---------------------------------------------------------------------------

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

    logger.info(f"Connecting to API: {API_BASE_URL}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("solve", cmd_solve))
    app.add_handler(CommandHandler("chapter", cmd_chapter))
    app.add_handler(CommandHandler("clear", cmd_clear))

    # Buttons
    app.add_handler(CallbackQueryHandler(button_handler))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Cikgu AI Kimia Bot started (polling mode)")

    # Drop pending updates so old messages dont flood
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
