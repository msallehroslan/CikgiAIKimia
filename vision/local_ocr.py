"""
vision/local_ocr.py — Cikgu AI Kimia  [PRODUCTION HARDENING v4.0]
==================================================================
REPLACES: existing vision/local_ocr.py  (adds full preprocessing + corrections)

FALLBACK CHAIN (Groq-only architecture):
  Groq Vision (llama-4-scout)
    ↓ fails / quota exhausted / circuit open
  Local OCR  ← THIS FILE
    ├── Tesseract (default, free tier, 512MB RAM)
    └── PaddleOCR (optional, set VISION_LOCAL_ENGINE=paddle)
    ↓ both fail
  Ask user for clearer image / type manually

OCR LIBRARY COMPARISON — SPM Chemistry:
  ┌────────────┬────────┬──────────┬─────────────┬──────────────┐
  │ Library    │ RAM    │ Subscript│ Blurry Image│ Render Free? │
  ├────────────┼────────┼──────────┼─────────────┼──────────────┤
  │ Tesseract  │  ~50MB │   POOR   │    OK        │   YES ✓      │
  │ PaddleOCR  │ ~400MB │   GOOD   │   GREAT      │   NO  ✗      │
  │ Surya OCR  │ ~600MB │   GOOD   │   GOOD       │   NO  ✗      │
  │ GOT-OCR    │  ~8GB  │  BEST    │   BEST       │   NO  ✗      │
  └────────────┴────────┴──────────┴─────────────┴──────────────┘

  DECISION:
    Render free (512MB)  → Tesseract + PIL preprocessing
    Render paid (1GB+)   → PaddleOCR (set VISION_LOCAL_ENGINE=paddle)
    Offline build server → GOT-OCR (not implemented here — for indexing only)

IMAGE PREPROCESSING PIPELINE:
  Raw bytes
    → PIL open → normalize to RGB/grayscale
    → upscale if < 1000px wide  (OCR accuracy cliff)
    → convert to grayscale
    → adaptive threshold         (handles uneven lighting, yellow paper)
    → mild denoise               (median filter, salt-and-pepper noise)
    → deskew                     (straighten tilted phone photos ±15°)
    → Tesseract / Paddle

CHEMISTRY CORRECTION LAYER:
  Common OCR errors for SPM Chemistry (applied AFTER extraction):
    "H20"  → "H2O"     (digit zero vs letter O)
    "NaC1" → "NaCl"    (digit one vs letter l)
    "HCL"  → "HCl"     (all-caps element symbol)
    "DH"   → "ΔH"      (Greek delta lost in OCR)
    "AT"   → "ΔT"
    "25 0C"→ "25°C"    (degree symbol lost)
    "mol/l"→ "mol/dm3" (litre shorthand)

ENV VARS:
  VISION_LOCAL_ENGINE = tesseract | paddle   (default: tesseract)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from typing import Optional

logger = logging.getLogger("cikgu.local_ocr")

LOCAL_ENGINE = os.environ.get("VISION_LOCAL_ENGINE", "tesseract").lower()


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> "PIL.Image.Image":
    """
    Full preprocessing pipeline for OCR accuracy.

    Steps:
      1. Open and normalise to consistent mode
      2. Upscale if width < 1000px
      3. Convert to grayscale (L)
      4. Adaptive threshold → binary image
      5. Mild median denoise
      6. Deskew (rotate to correct tilt ±15°)

    Returns:
      PIL Image in mode "L" (grayscale), ready for Tesseract or PaddleOCR.
      Raises ImportError if Pillow not installed.
    """
    from PIL import Image, ImageFilter, ImageOps
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes))

    # ── Step 1: Normalise mode ──────────────────────────────────────────────
    if img.mode == "RGBA":
        # Flatten transparency onto white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # ── Step 2: Upscale if too small ───────────────────────────────────────
    MIN_WIDTH = 1000
    if img.width < MIN_WIDTH:
        scale  = MIN_WIDTH / img.width
        new_w  = int(img.width * scale)
        new_h  = int(img.height * scale)
        img    = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"[ocr] upscaled {img.width}x{img.height}")

    # ── Step 3: Grayscale ──────────────────────────────────────────────────
    img = img.convert("L")

    # ── Step 4: Adaptive threshold ─────────────────────────────────────────
    # Binarises the image, handles uneven lighting / yellow paper well.
    # Uses PIL's autocontrast as a simple approximation.
    # For full adaptive threshold you need opencv-python (heavy dep).
    # PIL-only approach: CLAHE equivalent via autocontrast then posterize.
    img = ImageOps.autocontrast(img, cutoff=2)

    # ── Step 5: Denoise (mild median filter) ──────────────────────────────
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # ── Step 6: Deskew ────────────────────────────────────────────────────
    try:
        img = _deskew(img)
    except Exception as e:
        logger.debug(f"[ocr] deskew skipped: {e}")

    return img


def _deskew(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """
    Detect and correct image tilt using projection profile method.
    Handles tilts in range ±15°.
    Falls back to original image if numpy not available.
    """
    import numpy as np
    from PIL import Image

    arr = np.array(img)

    # Binarise: pixels < 128 → 1 (text), else → 0
    binary = (arr < 128).astype(np.float32)

    best_angle = 0.0
    best_score = -1.0

    # Scan angles in small steps around 0°
    for angle in range(-15, 16, 1):
        from PIL import Image as _Image
        rotated = img.rotate(angle, expand=False, fillcolor=255)
        arr_r   = np.array(rotated)
        binary_r = (arr_r < 128).astype(np.float32)
        row_sums = binary_r.sum(axis=1)
        # High variance = well-aligned text lines
        score = float(row_sums.var())
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) > 0.5:
        logger.debug(f"[ocr] deskew angle={best_angle}°")
        img = img.rotate(best_angle, expand=False, fillcolor=255)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
# TESSERACT OCR
# ═══════════════════════════════════════════════════════════════════════════════

async def _ocr_tesseract(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    Tesseract OCR with preprocessing.

    Tesseract config:
      --oem 3   → LSTM neural mode (best accuracy)
      --psm 6   → Assume uniform block of text
      --psm 4   → Assume single column (better for exam question layout)

    Language:
      BM → "eng+msa"  (malay script is Latin; msa pack handles vocab)
      EN → "eng"

    Requires:
      pip install pytesseract Pillow
      apt-get install tesseract-ocr tesseract-ocr-msa
    """
    try:
        import pytesseract

        img       = preprocess_image(image_bytes)
        tess_lang = "eng+msa" if lang == "BM" else "eng"

        # PSM 4: single column text — better for exam question blocks
        config = "--oem 3 --psm 4"

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            lambda: pytesseract.image_to_string(img, lang=tess_lang, config=config),
        )
        result = text.strip()
        if not result:
            # Retry with PSM 6 (uniform text block) if PSM 4 returns empty
            config2 = "--oem 3 --psm 6"
            text2 = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(img, lang=tess_lang, config=config2),
            )
            result = text2.strip()

        if result:
            logger.info(f"[ocr] tesseract extracted chars={len(result)}")
        return result or None

    except ImportError:
        logger.error(
            "[ocr] pytesseract not installed. "
            "Run: apt-get install tesseract-ocr tesseract-ocr-msa && pip install pytesseract Pillow"
        )
        return None
    except Exception as e:
        logger.error(f"[ocr] tesseract error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PADDLEOCR (OPTIONAL — paid tier only)
# ═══════════════════════════════════════════════════════════════════════════════

_paddle_model = None   # Lazy-loaded singleton


async def _ocr_paddle(image_bytes: bytes, lang: str = "BM") -> Optional[str]:
    """
    PaddleOCR — higher accuracy for blurry/low-quality images.
    ~400MB model. NOT suitable for Render free tier (512MB RAM).

    Enable: set VISION_LOCAL_ENGINE=paddle in env.

    If PaddleOCR fails → cascades back to Tesseract automatically.
    """
    global _paddle_model
    try:
        from paddleocr import PaddleOCR
        import numpy as np

        if _paddle_model is None:
            logger.info("[ocr] loading PaddleOCR model (first use)...")
            _paddle_model = PaddleOCR(
                use_angle_cls=True,
                lang="en",       # English model handles BM Latin script
                use_gpu=False,
                show_log=False,
            )
            logger.info("[ocr] PaddleOCR model loaded")

        img = preprocess_image(image_bytes)
        arr = np.array(img.convert("RGB"))

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _paddle_model.ocr(arr, cls=True)
        )

        if not result or not result[0]:
            return None

        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text, confidence = line[1]
                if confidence > 0.5:    # drop low-confidence tokens
                    lines.append(text)

        extracted = "\n".join(lines)
        if extracted:
            logger.info(f"[ocr] paddle extracted chars={len(extracted)}")
        return extracted or None

    except ImportError:
        logger.error(
            "[ocr] PaddleOCR not installed. "
            "Run: pip install paddlepaddle paddleocr"
        )
        return None
    except Exception as e:
        logger.error(f"[ocr] paddle error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHEMISTRY CORRECTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# ORDER MATTERS — more specific patterns first.
# re.compile flags: use re.IGNORECASE only for ALL-CAPS checks.
_CHEM_CORRECTIONS = [
    # ── Greek letters lost in OCR ─────────────────────────────────────────
    (r'\bAT\b',                    'ΔT'),     # delta-T
    (r'\bDT\b',                    'ΔT'),
    (r'\bAH\b',                    'ΔH'),     # delta-H
    (r'\bDH\b',                    'ΔH'),

    # ── Degree symbol ─────────────────────────────────────────────────────
    (r'(\d+)\s*0C\b',              r'\1°C'),  # "25 0C" → "25°C"
    (r'(\d+)\s*oC\b',              r'\1°C'),  # "25 oC" → "25°C"

    # ── Digit zero vs letter O in formulas ────────────────────────────────
    (r'\bH20\b',                   'H2O'),
    (r'\bNa0H\b',                  'NaOH'),
    (r'\bKMn04\b',                 'KMnO4'),
    (r'\bNaC1\b',                  'NaCl'),
    (r'\bCaC12\b',                 'CaCl2'),
    (r'\bFe203\b',                 'Fe2O3'),
    (r'\bCu0\b',                   'CuO'),
    (r'\bMg0\b',                   'MgO'),
    (r'\bZn0\b',                   'ZnO'),

    # ── Digit 1 vs letter l ───────────────────────────────────────────────
    (r'\bA1\b(?=[A-Z0-9(])',       'Al'),     # "A1Cl3" → "AlCl3"
    (r'\bA1C13\b',                 'AlCl3'),
    (r'\bA12O3\b',                 'Al2O3'),
    (r'\bNaC1\b',                  'NaCl'),

    # ── All-caps element symbols ───────────────────────────────────────────
    (r'\bHCL\b',                   'HCl'),
    (r'\bNACL\b',                  'NaCl'),
    (r'\bNAOH\b',                  'NaOH'),
    (r'\bH2S04\b',                 'H2SO4'),  # zero vs O
    (r'\bKOH(?![a-z])\b',         'KOH'),     # keep KOH, it's correct

    # ── Spaced subscripts ("H 2 O" → "H2O") ─────────────────────────────
    # Conservative: only for known formulas
    (r'\bH\s+2\s*O\b',            'H2O'),
    (r'\bC\s*O\s*2\b',            'CO2'),
    (r'\bN\s*H\s*3\b',            'NH3'),
    (r'\bH\s*C\s*l\b',            'HCl'),
    (r'\bN\s*a\s*O\s*H\b',        'NaOH'),

    # ── Unit normalisation ─────────────────────────────────────────────────
    (r'\bmol\s*/\s*[lL]\b',       'mol/dm3'),
    (r'\bmol\s*dm\s*-3\b',        'mol/dm3'),
    (r'\bJg-1\b',                  'J g-1'),
    (r'\bkJmol-1\b',               'kJ mol-1'),

    # ── Arrow variants → plain "->" ────────────────────────────────────────
    (r'[→⟶⇒➔]',                  '->'),
    (r'[⇌⇆]',                    '<->'),
]

_COMPILED_CORRECTIONS = [
    (re.compile(pat), repl)
    for pat, repl in _CHEM_CORRECTIONS
]


def apply_chemistry_corrections(text: str) -> str:
    """
    Apply chemistry-specific OCR correction rules.
    Conservative: only fixes unambiguous, high-confidence corrections.
    Never introduces new information — only corrects known OCR failure modes.
    """
    corrected = text
    for pattern, replacement in _COMPILED_CORRECTIONS:
        corrected = pattern.sub(replacement, corrected)
    return corrected


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC CONFIDENCE CHECK (fast pre-filter before full confidence_scorer.py)
# ═══════════════════════════════════════════════════════════════════════════════

def basic_ocr_confidence(text: str) -> float:
    """
    Quick 0.0–1.0 confidence score.
    Full scoring is done by vision/confidence_scorer.py.
    This is a fast pre-filter to reject obviously garbage output.
    """
    if not text or len(text) < 5:
        return 0.0

    # Too short to be a chemistry question
    if len(text) < 15:
        return 0.15

    words = text.split()
    if len(words) < 3:
        return 0.20

    # High non-ASCII ratio (garbled encoding / binary garbage)
    allowed_unicode = set("°ΔΩμ→⇌⁺⁻₀₁₂₃₄₅₆₇₈₉αβγδ")
    non_ascii = sum(1 for c in text if ord(c) > 127 and c not in allowed_unicode)
    non_ascii_ratio = non_ascii / max(len(text), 1)
    if non_ascii_ratio > 0.30:
        return 0.10

    # Repeated character runs (OCR table borders misread as text)
    if re.search(r'(.)\1{5,}', text):
        return 0.20

    # Very long all-caps word (random noise)
    if re.search(r'\b[A-Z]{10,}\b', text):
        return 0.25

    return 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_with_local_ocr(
    image_bytes: bytes,
    lang: str = "BM",
) -> Optional[str]:
    """
    Run local OCR as fallback when Groq Vision fails.

    Flow:
      1. Preprocess image (upscale, grayscale, threshold, deskew)
      2. Run OCR engine (Tesseract or PaddleOCR)
         - If PaddleOCR fails → cascade to Tesseract
      3. Apply chemistry correction layer
      4. Basic confidence check
      5. Return corrected text, or None if quality is too low

    Returns:
        str  — extracted and chemistry-corrected text
        None — OCR failed or quality too low to be useful
    """
    engine = LOCAL_ENGINE
    logger.info(f"[local_ocr] engine={engine} image_size={len(image_bytes)}")

    raw: Optional[str] = None

    if engine == "paddle":
        raw = await _ocr_paddle(image_bytes, lang)
        if raw is None:
            logger.warning("[local_ocr] PaddleOCR failed → cascading to Tesseract")
            raw = await _ocr_tesseract(image_bytes, lang)
    else:
        raw = await _ocr_tesseract(image_bytes, lang)

    if not raw:
        logger.warning("[local_ocr] all engines returned empty")
        return None

    # Chemistry corrections
    corrected = apply_chemistry_corrections(raw)

    # Basic quality gate
    conf = basic_ocr_confidence(corrected)
    logger.info(f"[local_ocr] chars={len(corrected)} basic_conf={conf:.2f}")

    if conf < 0.2:
        logger.warning("[local_ocr] very low confidence — returning None")
        return None

    return corrected
