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

PERATURAN untuk soalan MCQ:
- Sertakan soalan DAN semua pilihan jawapan (A, B, C, D)
- Sertakan data yang diberi (Jisim atom relatif, dll)
- Jika ada jadual, tulis dalam format teks biasa

PERATURAN untuk rajah/gambar:
- Jika ada rajah yang tidak boleh dibaca dalam teks, tulis [RAJAH]
- Jika ada graf, tulis [GRAF] dan huraikan paksi jika boleh dibaca
- Jika ada formula struktur kimia organik, huraikan dalam teks

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

RULES for MCQ questions:
- Include question AND all answer options (A, B, C, D)
- Include given data (Relative atomic mass, etc.)
- If there is a table, write it in plain text

RULES for diagrams:
- If diagram cannot be read as text, write [DIAGRAM]
- If graph, write [GRAPH] and describe axes if readable
- If organic structural formula, describe it in text

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
