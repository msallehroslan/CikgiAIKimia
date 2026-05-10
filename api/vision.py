"""
vision.py — Cikgu AI Kimia Vision Module
=========================================
Version: 1.0.0
Date   : 10 May 2026

Handles photo/image input from Telegram.
Extracts chemistry question text from image using vision AI.

SUPPORTED PROVIDERS (set via env var VISION_PROVIDER):
  groq     → Llama 4 Scout (default, free, already in your Groq account)
  gemini   → Google Gemini Flash (free 1500 RPD, more generous)
  tesseract→ Local OCR (free, no API, requires tesseract binary — local dev only)
  none     → Disabled (bot will ask user to type instead)

ENVIRONMENT VARIABLES:
  VISION_PROVIDER          = groq | gemini | tesseract | none  (default: groq)
  GROQ_API_KEY             = your_groq_key  (already set)
  GEMINI_API_KEY           = your_gemini_key (only if using gemini)

GROQ VISION MODEL:
  meta-llama/llama-4-scout-17b-16e-instruct
  - Free tier: 30 RPM, 1K RPD, 30K TPM
  - Multimodal: supports image + text input
  - Available in your account (confirmed from rate limits page)

GEMINI VISION MODEL:
  gemini-1.5-flash
  - Free tier: 1500 RPD, 1M TPM (much more generous)
  - Get API key: https://aistudio.google.com/app/apikey
  - pip install google-generativeai

USAGE:
  from vision import extract_question_from_image

  # bytes of image from Telegram
  question_text = await extract_question_from_image(image_bytes)
  if question_text:
      # pass to normal solver/RAG pipeline
      result = await call_api(question_text, session_id)
"""

from __future__ import annotations

import os
import base64
import logging
from typing import Optional

logger = logging.getLogger("cikgu_ai_kimia.vision")

# ── CONFIG ────────────────────────────────────────────────────────────────────
VISION_PROVIDER   = os.environ.get("VISION_PROVIDER", "groq").lower()
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GEMINI_MODEL      = "gemini-1.5-flash"

# Prompt untuk extract soalan kimia dari gambar
# KRITIKAL: No LaTeX, no unicode subscript, plain text formula only
VISION_PROMPT_BM = """Kamu adalah pembantu kimia SPM Malaysia.
Lihat gambar ini dan ekstrak SEMUA teks soalan kimia.

PERATURAN WAJIB untuk formula kimia:
- Tulis dalam format BIASA: H2O, NaOH, K4Fe(CN)6, H2SO4, CO2
- JANGAN guna LaTeX: JANGAN tulis $\\rm K_4Fe(CN)_6$ atau \\text{}
- JANGAN guna subscript unicode: JANGAN tulis H₂O atau SO₄²⁻
- GUNAKAN nombor biasa: H2O bukan H₂O, SO4 bukan SO₄
- Formula dengan titik air kristal: tulis K4Fe(CN)6.3H2O
- Ion berkas: tulis SO4 2-, MnO4-, Cr2O7 2-

PERATURAN WAJIB untuk kotak dan rajah:
- Jika ada formula atau teks dalam KOTAK (box) — MESTI baca dan extract kandungannya
- Jika ada formula dalam rajah berlabel "Rajah 1" atau "Diagram 1" — MESTI extract formula tersebut
- Jangan tulis [RAJAH] jika kotak mengandungi formula kimia — baca dan tulis formulanya
- Hanya tulis [RAJAH] jika kotak mengandungi gambar/struktur yang benar-benar tidak boleh dibaca sebagai teks

PERATURAN untuk soalan MCQ:
- Sertakan soalan DAN semua pilihan jawapan (A, B, C, D)
- Sertakan data yang diberi (Jisim atom relatif, dll)
- Jika ada jadual, tulis dalam format teks biasa

Balas dengan teks soalan SAHAJA. Tiada penjelasan. Tiada markdown header."""

VISION_PROMPT_EN = """You are an SPM chemistry tutor assistant.
Extract ALL chemistry question text from this image.

CRITICAL RULES for chemical formulas:
- Use PLAIN TEXT format only: H2O, NaOH, K4Fe(CN)6, H2SO4, CO2
- NO LaTeX: do NOT write $\\rm K_4Fe(CN)_6$ or \\text{}
- NO unicode subscripts: do NOT write H₂O or SO₄²⁻
- Use regular numbers: H2O not H₂O, SO4 not SO₄
- Water of crystallisation: write K4Fe(CN)6.3H2O
- Ionic charges: write SO4 2-, MnO4-, Cr2O7 2-

CRITICAL RULES for boxes and diagrams:
- If a formula or text is inside a BOX — you MUST read and extract its content
- If a formula is in a diagram labelled "Diagram 1" or "Rajah 1" — MUST extract the formula
- Do NOT write [DIAGRAM] if the box contains a chemical formula — read and write it
- Only write [DIAGRAM] if the box contains an actual image/structure that truly cannot be read as text

RULES for MCQ questions:
- Include question AND all answer options (A, B, C, D)
- Include given data (Relative atomic mass, etc.)
- If there is a table, write it in plain text

Reply with question text ONLY. No explanation. No markdown headers."""


# ── GROQ VISION ───────────────────────────────────────────────────────────────
async def _extract_groq(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    Extract text from image using Groq Llama 4 Scout (multimodal).
    Free tier: 30 RPM, 1K RPD, 30K TPM.
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set")
        return None

    try:
        from groq import AsyncGroq

        # Convert image to base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect image type from bytes magic header
        if image_bytes[:3] == b'\xff\xd8\xff':
            media_type = "image/jpeg"
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            media_type = "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"  # default

        prompt = VISION_PROMPT_BM if lang == "BM" else VISION_PROMPT_EN

        client = AsyncGroq(api_key=GROQ_API_KEY)
        response = await client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            max_tokens=1000,
            temperature=0.1,
        )

        extracted = response.choices[0].message.content.strip()
        logger.info(f"Groq Vision extracted {len(extracted)} chars")
        return extracted if extracted else None

    except Exception as e:
        logger.error(f"Groq Vision error: {e}")
        return None


# ── GEMINI VISION ─────────────────────────────────────────────────────────────
async def _extract_gemini(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    Extract text from image using Google Gemini Flash.
    Free tier: 1500 RPD, 1M TPM — more generous than Groq.
    Get API key: https://aistudio.google.com/app/apikey
    Install: pip install google-generativeai
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return None

    try:
        import google.generativeai as genai
        from PIL import Image
        import io

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        prompt = VISION_PROMPT_BM if lang == "BM" else VISION_PROMPT_EN

        # Run in thread pool (Gemini SDK is sync)
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content([image, prompt])
        )

        extracted = response.text.strip()
        logger.info(f"Gemini Vision extracted {len(extracted)} chars")
        return extracted if extracted else None

    except ImportError:
        logger.error("google-generativeai not installed. Run: pip install google-generativeai Pillow")
        return None
    except Exception as e:
        logger.error(f"Gemini Vision error: {e}")
        return None


# ── TESSERACT OCR (LOCAL ONLY) ────────────────────────────────────────────────
async def _extract_tesseract(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    Extract text using Tesseract OCR — local development only.
    Requires: apt-get install tesseract-ocr tesseract-ocr-msa
    NOT suitable for Render free tier (no apt-get).
    """
    try:
        import pytesseract
        from PIL import Image
        import io
        import asyncio

        image = Image.open(io.BytesIO(image_bytes))

        # Use both English and Malay language packs
        tess_lang = "eng+msa" if lang == "BM" else "eng"

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: pytesseract.image_to_string(image, lang=tess_lang)
        )

        extracted = text.strip()
        logger.info(f"Tesseract extracted {len(extracted)} chars")
        return extracted if len(extracted) > 10 else None

    except ImportError:
        logger.error("pytesseract or Pillow not installed")
        return None
    except Exception as e:
        logger.error(f"Tesseract error: {e}")
        return None


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
async def extract_question_from_image(
    image_bytes: bytes,
    lang: str = "BM",
) -> Optional[str]:
    """
    Main entry point — extract chemistry question from image.

    Args:
        image_bytes : raw bytes of image from Telegram
        lang        : "BM" or "EN" for prompt language

    Returns:
        Extracted question text, or None if failed

    Provider selection via VISION_PROVIDER env var:
        groq      → Llama 4 Scout (default)
        gemini    → Google Gemini Flash
        tesseract → Local OCR (dev only)
        none      → Disabled
    """
    if VISION_PROVIDER == "none":
        return None

    logger.info(f"Vision provider: {VISION_PROVIDER}, image size: {len(image_bytes)} bytes")

    raw_text = None
    if VISION_PROVIDER == "groq":
        raw_text = await _extract_groq(image_bytes, lang)
    elif VISION_PROVIDER == "gemini":
        raw_text = await _extract_gemini(image_bytes, lang)
    elif VISION_PROVIDER == "tesseract":
        raw_text = await _extract_tesseract(image_bytes, lang)
    else:
        logger.error(f"Unknown VISION_PROVIDER: {VISION_PROVIDER}")
        return None

    if raw_text:
        # Always clean extracted text — remove LaTeX, unicode subscripts, etc.
        cleaned = clean_extracted_text(raw_text)
        logger.info(f"Cleaned text ({len(raw_text)} → {len(cleaned)} chars)")
        return cleaned
    return None


def clean_extracted_text(text: str) -> str:
    """
    Post-process extracted text from vision AI.
    Remove LaTeX formatting, unicode subscripts, and other artifacts
    that would break the chemistry solver/extractor.
    """
    import re

    # Remove LaTeX math delimiters
    text = re.sub(r'\$\\rm\s*', '', text)
    text = re.sub(r'\$', '', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\rm\s*', '', text)

    # Remove LaTeX subscripts/superscripts → plain numbers
    # e.g. K_4 → K4, H_2O → H2O, Fe(CN)_6 → Fe(CN)6
    text = re.sub(r'_\{(\d+)\}', r'\1', text)
    text = re.sub(r'\^\{([^}]+)\}', r'\1', text)
    text = re.sub(r'_(\d+)', r'\1', text)
    text = re.sub(r'\^(\d+)', r'\1', text)

    # Unicode subscript digits → regular digits
    subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    text = text.translate(subscript_map)

    # Unicode superscripts → regular
    superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')
    text = text.translate(superscript_map)

    # Unicode arrows → plain text
    text = text.replace('→', '->').replace('⟶', '->')
    text = text.replace('⇌', '<->').replace('⇆', '<->')

    # Unicode minus/dash variants → plain hyphen
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')

    # Unicode multiply → x
    text = text.replace('×', 'x')

    # Remove excessive whitespace
    import re as _re
    text = _re.sub(r'\n{3,}', '\n\n', text)
    text = _re.sub(r' {2,}', ' ', text)

    return text.strip()


async def interpret_question(
    raw_text: str,
    lang: str = "BM",
    groq_api_key: str = "",
    explain_model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Step 2 of vision pipeline — LLM interpret extracted text.

    Takes messy OCR/vision output and converts to clean,
    solver-ready question text.

    Examples:
      IN:  "## Soalan Kimia SPM\n5\n$\\rm K_4Fe(CN)_6.3H_2O$\nApakah jisim relatif?"
      OUT: "Apakah jisim relatif bagi K4Fe(CN)6.3H2O? [H=1, C=12, O=16, K=39, Fe=56, N=14]"

      IN:  "A 141\nB 256\nC 389\nD 422"  (MCQ with no question)
      OUT: "Hitungkan JMR bagi K4Fe(CN)6.3H2O [H=1,C=12,O=16,K=39,Fe=56,N=14]"

    Args:
        raw_text     : cleaned extracted text from vision
        lang         : "BM" or "EN"
        groq_api_key : Groq API key
        explain_model: model to use (8b-instant, fast and sufficient)

    Returns:
        Clean question text ready for solver/RAG pipeline
        Falls back to raw_text if LLM fails
    """
    if not groq_api_key or not raw_text:
        return raw_text

    if lang == "BM":
        prompt = f"""Kamu adalah pembantu kimia SPM. Tugas kamu adalah menginterpret teks yang diextract dari gambar soalan kimia SPM.

TEKS DARI GAMBAR:
{raw_text}

TUGASAN:
1. Kenal pasti soalan kimia UTAMA yang perlu dijawab
2. Jika soalan MCQ — tulis soalan dan pilihan jawapan (A, B, C, D) dengan jelas
3. Jika ada data (Jisim atom relatif, kemolaran, dll) — sertakan dalam soalan
4. Jika ada formula kimia — pastikan dalam format biasa (H2O, NaOH, K4Fe(CN)6)
5. Buang teks tidak relevan (nombor halaman, header, footer, teks berulang)

PENTING — DERIVE FORMULA DARI NAMA KIMIA:
Jika nama kimia IUPAC ada dalam soalan tapi formula tidak kelihatan
(mungkin dalam kotak/rajah yang tidak dapat dibaca), DERIVE formula dari nama:
- "kalium heksasianoferat(III) terhidrat" → K4Fe(CN)6.3H2O
- "kuprum(II) sulfat pentahidrat" → CuSO4.5H2O
- "natrium tiosulfat" → Na2S2O3
- "ferum(III) oksida" → Fe2O3
- Gunakan pengetahuan kimia untuk derive formula lain yang serupa

PENTING — RAJAH/KOTAK:
Jika ada sebutan "Rajah X" atau "Diagram X" yang mengandungi formula/struktur
tapi tidak dapat dibaca dalam teks — gunakan nama kimia dalam soalan untuk
derive formula tersebut dan sertakan dalam output.

OUTPUT: Tulis semula soalan dengan LENGKAP dan JELAS.
Sertakan formula yang betul dan semua data yang diperlukan.
JANGAN jawab soalan — hanya tulis semula soalan dengan jelas."""

    else:
        prompt = f"""You are an SPM chemistry assistant. Interpret text extracted from an SPM chemistry question image.

EXTRACTED TEXT:
{raw_text}

TASK:
1. Identify the MAIN chemistry question to be answered
2. If MCQ — write the question and all answer options (A, B, C, D) clearly
3. If there is data (Relative atomic mass, molarity, etc.) — include it
4. If chemical formulas present — ensure plain text format (H2O, NaOH, K4Fe(CN)6)
5. Remove irrelevant text (page numbers, headers, footers, repeated text)

IMPORTANT — DERIVE FORMULA FROM IUPAC NAME:
If a chemical IUPAC name exists but formula is missing
(possibly in a box/diagram that was not extracted), DERIVE the formula:
- "potassium hexacyanoferrate(III) trihydrate" → K4Fe(CN)6.3H2O
- "copper(II) sulphate pentahydrate" → CuSO4.5H2O
- "sodium thiosulphate" → Na2S2O3
- "iron(III) oxide" → Fe2O3
- Use chemistry knowledge to derive other similar formulas

IMPORTANT — DIAGRAMS/BOXES:
If "Diagram X" or "Rajah X" contains a formula/structure
that was not extracted — use the chemical name to derive the formula
and include it in your output.

OUTPUT: Rewrite the question COMPLETELY and CLEARLY.
Include the correct formula and all required data.
DO NOT answer — only rewrite the question clearly."""

    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=groq_api_key)
        resp = await client.chat.completions.create(
            model=explain_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        interpreted = resp.choices[0].message.content.strip()

        # Apply same cleaning to interpreted text
        interpreted = clean_extracted_text(interpreted)

        logger.info(f"LLM interpreted: {len(raw_text)} → {len(interpreted)} chars")
        return interpreted if interpreted else raw_text

    except Exception as e:
        logger.warning(f"interpret_question failed: {e} — using raw text")
        return raw_text


def vision_is_enabled() -> bool:
    """Check if vision is configured and ready."""
    if VISION_PROVIDER == "none":
        return False
    if VISION_PROVIDER == "groq" and not GROQ_API_KEY:
        return False
    if VISION_PROVIDER == "gemini" and not GEMINI_API_KEY:
        return False
    return True


def vision_provider_info() -> dict:
    """Return info about current vision configuration."""
    return {
        "provider": VISION_PROVIDER,
        "enabled": vision_is_enabled(),
        "model": {
            "groq": GROQ_VISION_MODEL,
            "gemini": GEMINI_MODEL,
            "tesseract": "tesseract-ocr local",
            "none": "disabled",
        }.get(VISION_PROVIDER, "unknown"),
    }
