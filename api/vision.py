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
VISION_PROMPT_BM = """Kamu adalah pembantu kimia SPM. 
Lihat gambar ini dan ekstrak SEMUA teks soalan kimia yang ada.
Jika ada soalan dengan nombor (1, 2, 3...), senaraikan semua.
Jika ada formula kimia, tulis dengan betul (contoh: H2O, NaOH, CO2).
Jika ada rajah atau gambar yang tidak boleh dibaca, tulis [RAJAH].
Balas dengan teks soalan sahaja, tiada penjelasan tambahan."""

VISION_PROMPT_EN = """You are an SPM chemistry assistant.
Look at this image and extract ALL chemistry question text.
If there are numbered questions, list all of them.
Write chemical formulas correctly (e.g. H2O, NaOH, CO2).
If there is a diagram that cannot be read, write [DIAGRAM].
Reply with question text only, no extra explanation."""


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

    if VISION_PROVIDER == "groq":
        return await _extract_groq(image_bytes, lang)
    elif VISION_PROVIDER == "gemini":
        return await _extract_gemini(image_bytes, lang)
    elif VISION_PROVIDER == "tesseract":
        return await _extract_tesseract(image_bytes, lang)
    else:
        logger.error(f"Unknown VISION_PROVIDER: {VISION_PROVIDER}")
        return None


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
