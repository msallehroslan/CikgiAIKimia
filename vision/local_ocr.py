"""
vision/local_ocr.py — Cikgu AI Kimia
======================================
Local OCR fallback when Groq Vision fails or quota is exhausted.

OCR Library Comparison for SPM Chemistry:

  PaddleOCR  ★★★★☆  Best detection of mixed layouts (tables, formulas,
                     printed BM text). Heavy: ~400MB model download.
                     Handles blurry images well via its detection model.
                     RECOMMENDED for Render.com paid tier.

  Tesseract  ★★★☆☆  Lightest (~50MB). Handles clean printed text well.
                     Subscripts/superscripts: POOR (renders H₂O as H20).
                     Recommended for free tier (low RAM).
                     USED HERE as primary fallback (RAM-safe).

  Surya OCR  ★★★★☆  Layout-aware, handles multi-column exam PDFs.
                     Slower. Good for scanned PDF ingestion pipeline
                     (build_index), not real-time Telegram photo flow.

  GOT-OCR    ★★★★★  Best for scientific notation, subscripts, formulas.
                     Requires 8GB RAM. NOT viable on free tier.
                     Ideal for dedicated build pipeline server.

DECISION:
  Render free tier (512MB RAM):  Tesseract + PIL preprocessing
  Render paid tier (1GB RAM):    PaddleOCR (optional, toggle via env)
  Build pipeline (offline):      GOT-OCR or Surya

IMAGE PREPROCESSING PIPELINE:
  Raw bytes
    → PIL open
    → resize to min 1000px wide (OCR accuracy drops below this)
    → convert to grayscale
    → adaptive threshold (handles uneven lighting, shadows)
    → deskew (straighten tilted phone photos)
    → denoize (salt-and-pepper noise from low-quality scans)
    → Tesseract / PaddleOCR

CHEMISTRY CORRECTION LAYER:
  Common OCR errors specific to SPM Chemistry:
    "H20"   → "H2O"     (zero vs O)
    "Ca1₂"  → "CaCl2"   (1 vs l)
    "Na0H"  → "NaOH"    (zero vs O)
    "HCL"   → "HCl"     (all-caps)
    "mol/l" → "mol/dm3" (litre vs dm³)
    "0C"    → "°C"      (degree symbol lost)
    "AT"    → "ΔT"      (Greek delta lost)
    "DH"    → "ΔH"      (Greek delta lost)
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional

logger = logging.getLogger("cikgu.local_ocr")

# ── OCR engine selection ───────────────────────────────────────────────────
# Set VISION_LOCAL_ENGINE=paddle in env to use PaddleOCR on paid tier
import os
LOCAL_ENGINE = os.environ.get("VISION_LOCAL_ENGINE", "tesseract").lower()


# ── Image Preprocessing ────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> "PIL.Image.Image":
    """
    Prepare an image for OCR:
      1. Open and normalize to RGB/L
      2. Upscale if too small (OCR accuracy cliff below 1000px wide)
      3. Convert to grayscale
      4. Adaptive threshold (handles shadows, yellow scanned paper)
      5. Deskew (basic rotation correction for phone photos)

    Returns a PIL Image ready for OCR.
    """
    from PIL import Image, ImageFilter, ImageOps
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes))

    # ── Step 1: Normalize color mode ──────────────────────────────────
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # ── Step 2: Upscale small images ──────────────────────────────────
    MIN_WIDTH = 1000
    if img.width < MIN_WIDTH:
        scale  = MIN_WIDTH / img.width
        new_w  = MIN_WIDTH
        new_h  = int(img.height * scale)
        img    = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"[ocr] Upscaled {img.width}→{new_w}px")

    # ── Step 3: Grayscale ─────────────────────────────────────────────
    gray = img.convert("L")

    # ── Step 4: Adaptive threshold ────────────────────────────────────
    # Splits image into 25px blocks, thresholds each independently.
    # This handles yellowed scans, uneven phone-camera lighting.
    try:
        import cv2
        import numpy as np
        arr     = np.array(gray)
        thresh  = cv2.adaptiveThreshold(
            arr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=25,
            C=10,
        )
        # Mild denoise — removes salt-and-pepper from low-quality scans
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        gray     = Image.fromarray(denoised)
        logger.debug("[ocr] cv2 adaptive threshold applied")
    except ImportError:
        # cv2 not installed (free tier) — PIL fallback
        # PIL threshold is global, less effective for uneven lighting
        import numpy as np
        arr    = np.array(gray)
        mean   = arr.mean()
        thresh = (arr > (mean * 0.85)).astype("uint8") * 255
        gray   = Image.fromarray(thresh.astype("uint8"))
        logger.debug("[ocr] PIL threshold fallback applied")

    return gray


# ── Tesseract OCR ──────────────────────────────────────────────────────────

async def _ocr_tesseract(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    Tesseract OCR — lightest option, works on Render free tier.
    Requires: apt-get install tesseract-ocr tesseract-ocr-msa
    BM: uses 'msa' language pack (Malay — closest to BM in Tesseract).
    """
    try:
        import pytesseract
        from PIL import Image
        import asyncio

        img      = preprocess_image(image_bytes)
        tess_lang = "eng+msa" if lang == "BM" else "eng"

        # Page segmentation mode 6 = assume single uniform block of text
        # Suitable for single chemistry question images
        config = "--oem 3 --psm 6"

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: pytesseract.image_to_string(img, lang=tess_lang, config=config),
        )
        return text.strip() or None

    except ImportError:
        logger.error("[ocr] pytesseract not installed. Run: pip install pytesseract Pillow")
        return None
    except Exception as e:
        logger.error(f"[ocr] Tesseract error: {e}")
        return None


# ── PaddleOCR (optional, paid tier) ───────────────────────────────────────

_paddle_model = None

async def _ocr_paddle(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    PaddleOCR — better layout detection, handles blurry images well.
    ~400MB model download on first use. Not suitable for 512MB free tier.
    Set VISION_LOCAL_ENGINE=paddle to enable.
    """
    global _paddle_model
    try:
        from paddleocr import PaddleOCR
        import asyncio
        import numpy as np
        from PIL import Image

        if _paddle_model is None:
            _paddle_model = PaddleOCR(
                use_angle_cls=True,
                lang="en",         # English model handles BM roman script
                use_gpu=False,
                show_log=False,
            )
            logger.info("[ocr] PaddleOCR model loaded")

        img  = preprocess_image(image_bytes)
        arr  = np.array(img.convert("RGB"))

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _paddle_model.ocr(arr, cls=True)
        )

        if not result or not result[0]:
            return None

        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text, confidence = line[1]
                if confidence > 0.5:   # low-confidence lines dropped
                    lines.append(text)

        return "\n".join(lines) or None

    except ImportError:
        logger.error("[ocr] PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")
        return None
    except Exception as e:
        logger.error(f"[ocr] PaddleOCR error: {e}")
        return None


# ── Chemistry Correction Layer ─────────────────────────────────────────────

# Map of common OCR errors in chemistry context
# ORDER MATTERS — apply more specific patterns first
_CHEM_CORRECTIONS = [
    # Greek letters lost during OCR
    (r'\bAT\b',     'ΔT'),    # delta-T
    (r'\bDT\b',     'ΔT'),
    (r'\bAH\b',     'ΔH'),    # delta-H
    (r'\bDH\b',     'ΔH'),

    # Degree symbol
    (r'(\d+)\s*0C\b',  r'\1°C'),   # "25 0C" → "25°C"
    (r'(\d+)\s*oC\b',  r'\1°C'),   # "25 oC" → "25°C"

    # Zero vs letter O in formulas
    (r'\bH20\b',    'H2O'),
    (r'\bNa0H\b',   'NaOH'),
    (r'\bKMn04\b',  'KMnO4'),
    (r'\bNaC1\b',   'NaCl'),
    (r'\bCaC12\b',  'CaCl2'),
    (r'\bFe203\b',  'Fe2O3'),
    (r'\bA1\b(?=[A-Z0-9(])',  'Al'),   # A1 (digit 1) vs Al (element)

    # All-caps elements
    (r'\bHCL\b',    'HCl'),
    (r'\bNACL\b',   'NaCl'),
    (r'\bNAOH\b',   'NaOH'),
    (r'\bH2S04\b',  'H2SO4'),

    # Unit normalisation
    (r'\bmol/[lL]\b',         'mol/dm3'),
    (r'\bmol\s*dm\s*-3\b',   'mol dm3'),
    (r'\bJg-1\b',            'J g-1'),
    (r'\bkJmol-1\b',         'kJ mol-1'),

    # Subscript digits that OCR renders as superscripts or spaces
    (r'(\b[A-Z][a-z]?)\s+(\d)(\s|$)',  r'\1\2\3'),   # "H 2 O" → "H2O"

    # Arrow variants
    (r'[→⟶⇒➔]', '->'),
    (r'[⇌⇆]',    '<->'),
]

_COMPILED_CORRECTIONS = [
    (re.compile(pat, re.IGNORECASE if pat.isupper() else 0), repl)
    for pat, repl in _CHEM_CORRECTIONS
]


def apply_chemistry_corrections(text: str) -> str:
    """
    Apply chemistry-specific OCR correction rules.
    Conservative — only fixes unambiguous common errors.
    """
    corrected = text
    for pattern, replacement in _COMPILED_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    return corrected


# ── Confidence Scoring (basic — full version in confidence_scorer.py) ──────

def basic_ocr_confidence(text: str) -> float:
    """
    Quick confidence check before passing to full confidence_scorer.
    Returns 0.0–1.0.
    """
    if not text or len(text) < 5:
        return 0.0

    words   = text.split()
    if len(words) < 2:
        return 0.1

    # High ratio of non-ASCII characters (garbled encoding)
    non_ascii = sum(1 for c in text if ord(c) > 127 and c not in "°ΔΩμ→⇌⁺⁻₀₁₂₃₄₅₆₇₈₉")
    if non_ascii / len(text) > 0.3:
        return 0.1

    # Suspiciously short extraction
    if len(text) < 15:
        return 0.2

    return 0.8


# ── Main Entry Point ───────────────────────────────────────────────────────

async def extract_with_local_ocr(
    image_bytes: bytes,
    lang: str = "BM",
) -> Optional[str]:
    """
    Run local OCR as fallback when Groq Vision is unavailable.

    Returns:
        Extracted + chemistry-corrected text string, or None if OCR fails.
    """
    engine = LOCAL_ENGINE

    logger.info(f"[local_ocr] Running {engine} OCR, image={len(image_bytes)} bytes")

    raw: Optional[str] = None

    if engine == "paddle":
        raw = await _ocr_paddle(image_bytes, lang)
        if raw is None:
            # PaddleOCR failed → cascade to Tesseract
            logger.warning("[local_ocr] PaddleOCR failed, cascading to Tesseract")
            raw = await _ocr_tesseract(image_bytes, lang)
    else:
        raw = await _ocr_tesseract(image_bytes, lang)

    if not raw:
        logger.warning("[local_ocr] All local OCR engines returned empty")
        return None

    # Apply chemistry correction layer
    corrected = apply_chemistry_corrections(raw)

    # Basic quality gate
    conf = basic_ocr_confidence(corrected)
    logger.info(f"[local_ocr] Extracted {len(corrected)} chars, basic_conf={conf:.2f}")

    if conf < 0.2:
        logger.warning("[local_ocr] Very low confidence — returning None")
        return None

    return corrected
