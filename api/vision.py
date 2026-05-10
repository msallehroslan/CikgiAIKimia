"""
vision.py — Cikgu AI Kimia Vision Module
=========================================
Version: 2.0.0
Date   : 10 May 2026

Handles photo/image input from Telegram.
Extracts chemistry question text from image using vision AI.

CHANGES v2.0.0:
  - Updated VISION_PROMPT_BM/EN → structured output format
  - Added preprocess_vision_question() → parse structured output
  - Backward compatible: extract_question_from_image() still returns string
  - New: extract_question_structured() returns dict for router

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

GEMINI VISION MODEL:
  gemini-1.5-flash
  - Free tier: 1500 RPD, 1M TPM (much more generous)
  - pip install google-generativeai

USAGE:
  from vision import extract_question_from_image, extract_question_structured

  # Returns string (backward compatible)
  question_text = await extract_question_from_image(image_bytes)

  # Returns dict with structured info for router
  result = await extract_question_structured(image_bytes)
  # result = {
  #   "clean_question": "...",
  #   "soalan_type": "thermochemistry",
  #   "pilihan": {"A": "...", "B": "...", "C": "...", "D": "..."},
  #   "data": {"delta_T": 10, "volume_cm3": 100, ...},
  #   "formula_kimia": ["NaOH", "HCl"],
  #   "persamaan": "NaOH + HCl -> NaCl + H2O",
  # }
"""

from __future__ import annotations

import os
import re
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

# ─────────────────────────────────────────────────────────────────────────────
# VISION PROMPT v2.0 — Structured output untuk semua jenis soalan SPM
# Cover: Johor 2021, Terengganu 2021, Selangor 2024, semua negeri lain
# ─────────────────────────────────────────────────────────────────────────────

VISION_PROMPT_BM = """Kamu adalah tutor kimia SPM Malaysia yang membaca gambar soalan.
Buat DUA perkara dalam SATU respons:

LANGKAH 1 — BACA gambar dan extract semua teks
LANGKAH 2 — OUTPUT soalan dalam format berstruktur

FORMAT OUTPUT (ikut TEPAT-TEPAT):
SOALAN: [soalan PENUH dalam BM dan/atau EN — termasuk persamaan kimia]
PILIHAN: A.[teks] B.[teks] C.[teks] D.[teks]  ← tulis TIADA jika bukan MCQ
DATA_NOMBOR: [SEMUA nilai numerik — V=50cm³, M=2.0mol/dm³, delta_T=10°C, Q=2100J, E0=-0.76V, Ar_Cu=64, dsb]
FORMULA_KIMIA: [semua formula kimia — NaOH, H2SO4, Cu(NO3)2, K4Fe(CN)6.3H2O, dsb]
PERSAMAAN_KIMIA: [persamaan lengkap dengan koeffisien — 2H2+O2->2H2O — atau TIADA]
JENIS_PENGIRAAN: [pilih satu: stoich_mass / stoich_vol / stoich_from_molarity / thermochem_forward / thermochem_reverse / ph_from_H / ph_from_OH / titration / molarmass / empirical / voltaic / rate / TEORI]

PERATURAN formula kimia (WAJIB):
- Format biasa SAHAJA: H2O, NaOH, K4Fe(CN)6.3H2O, SO4 2-, MnO4-
- JANGAN LaTeX: tiada $\\rm K_4Fe$ atau _4 atau ^2-
- JANGAN unicode: tiada H₂O, SO₄²⁻ — guna H2O, SO4 2-
- Formula MESTI dalam bahagian SOALAN dan FORMULA_KIMIA

PERATURAN kotak/rajah:
- Formula dalam KOTAK → MESTI tulis dalam SOALAN
- Nama kimia ada tapi formula tiada → DERIVE formula:
  "kalium heksasianoferat(III) terhidrat" → K4Fe(CN)6.3H2O
  "kuprum(II) sulfat pentahidrat" → CuSO4.5H2O
  "natrium tiosulfat" → Na2S2O3

PERATURAN JENIS_PENGIRAAN:
- Soalan beri Q(Joule), cari ΔT → thermochem_reverse
- Soalan beri kemolaran → jisim produk → stoich_from_molarity
- Soalan isipadu gas → isipadu gas → stoich_vol
- Soalan OH⁻ → pH → ph_from_OH
- Soalan beri ΔT, cari ΔH → thermochem_forward
- Soalan teori sahaja → TEORI

CONTOH OUTPUT:

Untuk soalan thermochem_reverse (beri Q, cari ΔT):
SOALAN: Tindak balas 25cm³ HCl dengan 25cm³ NaOH membebaskan 2100J. Berapakah perubahan suhu?
PILIHAN: A.1.0°C B.2.0°C C.10.0°C D.20.0°C
DATA_NOMBOR: V_HCl=25cm³, V_NaOH=25cm³, Q=2100J, c=4.2Jg-1C-1, density=1.0gcm-3
FORMULA_KIMIA: HCl, NaOH, NaCl
PERSAMAAN_KIMIA: HCl + NaOH -> NaCl + H2O
JENIS_PENGIRAAN: thermochem_reverse

Untuk soalan stoich_from_molarity:
SOALAN: Hitung jisim CaSO4 apabila 25cm³ Na2SO4 0.5 mol dm-3 bertindak balas dengan Ca(NO3)2 berlebihan.
PILIHAN: A.0.85g B.1.70g C.2.20g D.3.40g
DATA_NOMBOR: V_Na2SO4=25cm³, M_Na2SO4=0.5, Ar_Ca=40, Ar_S=32, Ar_O=16
FORMULA_KIMIA: Na2SO4, Ca(NO3)2, CaSO4
PERSAMAAN_KIMIA: Na2SO4 + Ca(NO3)2 -> CaSO4 + 2NaNO3
JENIS_PENGIRAAN: stoich_from_molarity

Balas dengan FORMAT OUTPUT sahaja. Tiada penjelasan lain."""


VISION_PROMPT_EN = """You are an SPM chemistry tutor reading a question image.
Do TWO things in ONE response:

STEP 1 — READ image and extract all text
STEP 2 — OUTPUT question in structured format

OUTPUT FORMAT (follow EXACTLY):
QUESTION: [full question in BM and/or EN — include chemical equations]
OPTIONS: A.[text] B.[text] C.[text] D.[text]  ← write NONE if not MCQ
DATA_NUMBERS: [ALL numeric values — V=50cm³, M=2.0mol/dm³, delta_T=10°C, Q=2100J, E0=-0.76V, Ar_Cu=64, etc]
FORMULA_CHEM: [all chemical formulas — NaOH, H2SO4, Cu(NO3)2, K4Fe(CN)6.3H2O, etc]
EQUATION_CHEM: [full equation with coefficients — 2H2+O2->2H2O — or NONE]
CALCULATION_TYPE: [one of: stoich_mass / stoich_vol / stoich_from_molarity / thermochem_forward / thermochem_reverse / ph_from_H / ph_from_OH / titration / molarmass / empirical / voltaic / rate / THEORY]

FORMULA RULES (MANDATORY):
- Plain text ONLY: H2O, NaOH, K4Fe(CN)6.3H2O, SO4 2-, MnO4-
- NO LaTeX: no $\\rm K_4Fe$ or _4 or ^2-
- NO unicode: no H₂O, SO₄²⁻ — use H2O, SO4 2-
- Formula MUST be in QUESTION and FORMULA_CHEM sections

CALCULATION_TYPE RULES:
- Given Q(Joules), find ΔT → thermochem_reverse
- Given molarity → find mass of product → stoich_from_molarity
- Gas volume → gas volume → stoich_vol
- Given OH⁻ → find pH → ph_from_OH
- Given ΔT, find ΔH → thermochem_forward
- Theory question only → THEORY

Reply with OUTPUT FORMAT only. No extra explanation."""


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

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        if image_bytes[:3] == b'\xff\xd8\xff':
            media_type = "image/jpeg"
        elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            media_type = "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"

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
            max_tokens=1200,
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
    Free tier: 1500 RPD, 1M TPM.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        return None

    try:
        import google.generativeai as genai
        from PIL import Image
        import io
        import asyncio

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        image = Image.open(io.BytesIO(image_bytes))
        prompt = VISION_PROMPT_BM if lang == "BM" else VISION_PROMPT_EN

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
    NOT suitable for Render free tier.
    """
    try:
        import pytesseract
        from PIL import Image
        import io
        import asyncio

        image = Image.open(io.BytesIO(image_bytes))
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


# ── STRUCTURED OUTPUT PARSER ──────────────────────────────────────────────────

def preprocess_vision_question(raw_text: str) -> dict:
    """
    Parse structured output dari vision model v2.0.
    Returns dict siap untuk router/solver.

    Input (contoh):
        SOALAN: 25cm³ HCl + 25cm³ NaOH → 2100J. ΔT=?
        PILIHAN: A.1°C B.2°C C.10°C D.20°C
        DATA_NOMBOR: V=50cm³, Q=2100J, c=4.2
        FORMULA_KIMIA: HCl, NaOH
        PERSAMAAN_KIMIA: HCl + NaOH -> NaCl + H2O
        JENIS_PENGIRAAN: thermochem_reverse

    Output:
        {
            "soalan": "25cm³ HCl + 25cm³ NaOH → 2100J. ΔT=?",
            "pilihan": {"A": "1°C", "B": "2°C", "C": "10°C", "D": "20°C"},
            "data": {"V": 50, "Q": 2100, "c": 4.2},
            "formula_kimia": ["HCl", "NaOH"],
            "persamaan": "HCl + NaOH -> NaCl + H2O",
            "soalan_type": "thermochemistry_reverse",
            "clean_question": "...",
        }
    """
    result = {
        "soalan": "",
        "pilihan": {},
        "data": {},
        "formula_kimia": [],
        "persamaan": "",
        "soalan_type": "theory",
        "clean_question": "",
        "raw": raw_text,
    }

    # ── Parse tiap field ──────────────────────────────────────────────────
    # Support BM dan EN field names
    field_map = {
        "SOALAN": "soalan",
        "QUESTION": "soalan",
        "PILIHAN": "_pilihan_raw",
        "OPTIONS": "_pilihan_raw",
        "DATA_NOMBOR": "_data_raw",
        "DATA_NUMBERS": "_data_raw",
        "DATA": "_data_raw",
        "FORMULA_KIMIA": "_formula_raw",
        "FORMULA_CHEM": "_formula_raw",
        "PERSAMAAN_KIMIA": "persamaan",
        "EQUATION_CHEM": "persamaan",
        "JENIS_PENGIRAAN": "_jenis_raw",
        "CALCULATION_TYPE": "_jenis_raw",
    }

    current_key = None
    current_lines = []

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        matched = False
        for field, target in field_map.items():
            if line.upper().startswith(field + ":"):
                # Save previous
                if current_key and current_lines:
                    result[current_key] = " ".join(current_lines).strip()
                current_key = target
                current_lines = [line[len(field)+1:].strip()]
                matched = True
                break

        if not matched and current_key:
            current_lines.append(line)

    # Save last field
    if current_key and current_lines:
        result[current_key] = " ".join(current_lines).strip()

    # ── Parse pilihan MCQ ─────────────────────────────────────────────────
    pilihan_raw = result.pop("_pilihan_raw", "")
    if pilihan_raw and pilihan_raw.upper() != "TIADA" and pilihan_raw.upper() != "NONE":
        for m in re.finditer(r'([A-D])[.\)]\s*([^A-D]*?)(?=\s*[A-D][.\)]|$)', pilihan_raw):
            result["pilihan"][m.group(1)] = m.group(2).strip()

    # ── Parse data numerik ────────────────────────────────────────────────
    data_raw = result.pop("_data_raw", "")
    result["data"] = _parse_data_numbers(data_raw)

    # ── Parse formula kimia ───────────────────────────────────────────────
    formula_raw = result.pop("_formula_raw", "")
    if formula_raw and formula_raw.upper() != "TIADA":
        for token in re.split(r'[,;\s]+', formula_raw):
            token = token.strip()
            if token and re.match(r'^[A-Z][a-zA-Z0-9()._·•]+$', token):
                result["formula_kimia"].append(token)

    # ── Map JENIS_PENGIRAAN → soalan_type ────────────────────────────────
    jenis_raw = result.pop("_jenis_raw", "").lower().strip()
    result["soalan_type"] = _map_jenis_to_type(jenis_raw, result["soalan"])

    # ── Clean persamaan ───────────────────────────────────────────────────
    if result["persamaan"].upper() in ("TIADA", "NONE", ""):
        result["persamaan"] = ""

    # ── Build clean_question untuk solver ────────────────────────────────
    result["clean_question"] = _build_clean_question(result)

    return result


def _parse_data_numbers(data_str: str) -> dict:
    """Extract nilai numerik dari DATA_NOMBOR field."""
    result = {}
    if not data_str or data_str.upper() in ("TIADA", "NONE"):
        return result

    # Pattern: key=value (dengan pelbagai format)
    patterns = [
        (r'V(?:_\w+)?\s*=\s*([\d.]+)\s*cm', 'volume_cm3'),
        (r'V(?:_\w+)?\s*=\s*([\d.]+)\s*dm', 'volume_dm3'),
        (r'M(?:_\w+|olarity)?\s*=\s*([\d.]+)', 'molarity'),
        (r'delta[_\s]?[Tt]\s*=\s*([+-]?[\d.]+)', 'delta_T'),
        (r'[ΔδD][Tt]\s*=\s*([+-]?[\d.]+)', 'delta_T'),
        (r'Q\s*=\s*([\d.]+)\s*[Jj]', 'Q_joules'),
        (r'delta[_\s]?[Hh]\s*=\s*([+-]?[\d.]+)', 'delta_H'),
        (r'[ΔδD][Hh]\s*=\s*([+-]?[\d.]+)', 'delta_H'),
        (r'[Ee]0?[_\s]?\w*\s*=\s*([+-]?[\d.]+)\s*[Vv]', 'E0'),
        (r'[Aa]r[_\s]?(\w+)\s*=\s*([\d.]+)', 'Ar'),   # Ar_Cu=64
        (r'[Cc]\s*=\s*([\d.]+)', 'c_specific_heat'),
        (r'density\s*=\s*([\d.]+)', 'density'),
        (r'ketumpatan\s*=\s*([\d.]+)', 'density'),
        (r'[Vv]m\s*=\s*([\d.]+)', 'Vm'),
        (r'[Pp][Hh]\s*=\s*([\d.]+)', 'pH'),
        (r'[Mm][Oo][Ll]\s*=\s*([\d.]+)', 'mol'),
        (r'jisim\s*=\s*([\d.]+)\s*g', 'jisim_g'),
        (r'm\s*=\s*([\d.]+)\s*g', 'jisim_g'),
        (r'isipadu\s*=\s*([\d.]+)\s*dm', 'volume_dm3'),
        (r'isipadu\s*=\s*([\d.]+)\s*cm', 'volume_cm3'),
    ]

    for pattern, key in patterns:
        if key == 'Ar':
            # Special: Ar_Cu=64 → ar_override dict
            for m in re.finditer(pattern, data_str, re.IGNORECASE):
                elem = m.group(1)
                val = float(m.group(2))
                if 'ar_override' not in result:
                    result['ar_override'] = {}
                result['ar_override'][elem] = val
        else:
            m = re.search(pattern, data_str, re.IGNORECASE)
            if m and key not in result:
                try:
                    result[key] = float(m.group(1))
                except (ValueError, IndexError):
                    pass

    # Generic key=value fallback
    for m in re.finditer(r'(\w+)\s*=\s*([+-]?[\d.]+)', data_str):
        k = m.group(1).lower()
        if k not in result and k not in ('ar',):
            try:
                result[k] = float(m.group(2))
            except ValueError:
                pass

    return result


def _map_jenis_to_type(jenis: str, soalan: str = "") -> str:
    """Map JENIS_PENGIRAAN dari vision → soalan_type untuk router."""

    # Direct mapping dari vision output
    direct_map = {
        "stoich_mass":           "stoichiometry",
        "stoich_mass_to_mass":   "stoichiometry",
        "stoich_vol":            "stoichiometry_vol",
        "stoich_mass_to_vol":    "stoichiometry_vol",
        "stoich_vol_to_mass":    "stoichiometry_vol_to_mass",
        "stoich_vol_to_vol":     "stoichiometry_vol",
        "stoich_from_molarity":  "stoichiometry_from_molarity",
        "thermochem_forward":    "thermochemistry",
        "thermochem":            "thermochemistry",
        "thermochem_reverse":    "thermochemistry_reverse",
        "ph_from_h":             "ph_calculation",
        "ph_from_oh":            "ph_from_oh",
        "ph_calculation":        "ph_calculation",
        "titration":             "titration",
        "molarmass":             "molar_mass",
        "molar_mass":            "molar_mass",
        "empirical":             "empirical_formula",
        "voltaic":               "voltaic_cell",
        "voltaic_cell":          "voltaic_cell",
        "rate":                  "rate_of_reaction",
        "rate_of_reaction":      "rate_of_reaction",
        "teori":                 "theory",
        "theory":                "theory",
    }

    if jenis in direct_map:
        return direct_map[jenis]

    # Fallback: detect dari teks soalan
    return _detect_type_from_text(soalan)


def _detect_type_from_text(soalan: str) -> str:
    """Detect solver type dari teks soalan (fallback)."""
    q = soalan.lower()

    # Priority order — specific dulu
    if any(x in q for x in ['keupayaan sel', 'cell potential', 'e0', 'e⁰', 'voltaic', 'sel voltaik']):
        return "voltaic_cell"
    if any(x in q for x in ['entalpi', 'enthalpy', 'haba tindak', 'heat of', 'suhu naik', 'suhu turun',
                              'temperature rise', 'temperature drop', 'q = mc', 'muatan haba']):
        return "thermochemistry"
    if any(x in q for x in ['meneutralkan', 'neutralis', 'titrat', 'buret']):
        return "titration"
    if any(x in q for x in ['oh-', 'oh⁻', 'hidroksida', 'hydroxide']) and 'ph' in q:
        return "ph_from_oh"
    if 'ph' in q and any(x in q for x in ['hitung', 'nilai', 'calculate']):
        return "ph_calculation"
    if any(x in q for x in ['formula empirik', 'empirical formula', 'komposisi', 'composition']):
        return "empirical_formula"
    if any(x in q for x in ['jisim molar', 'relative molecular mass', 'jisim atom relatif', 'jmr']):
        return "molar_mass"
    if any(x in q for x in ['kadar tindak balas', 'rate of reaction']):
        return "rate_of_reaction"
    if any(x in q for x in ['→', '->', 'dihasilkan', 'produced', 'terbakar', 'burnt']):
        return "stoichiometry"
    if any(x in q for x in ['kepekatan', 'kemolaran', 'molarity', 'concentration']):
        return "concentration"

    return "theory"


def _build_clean_question(result: dict) -> str:
    """Bina soalan bersih untuk dihantar ke solver/extractor."""
    parts = []

    if result.get("soalan"):
        parts.append(result["soalan"])

    if result.get("persamaan"):
        parts.append(f"Persamaan: {result['persamaan']}")

    # Tambah data penting
    data = result.get("data", {})
    data_parts = []
    if data.get("volume_cm3"):
        data_parts.append(f"V={data['volume_cm3']}cm³")
    if data.get("volume_dm3"):
        data_parts.append(f"V={data['volume_dm3']}dm³")
    if data.get("molarity"):
        data_parts.append(f"M={data['molarity']}mol/dm³")
    if data.get("delta_T"):
        data_parts.append(f"ΔT={data['delta_T']}°C")
    if data.get("Q_joules"):
        data_parts.append(f"Q={data['Q_joules']}J")
    if data.get("delta_H"):
        data_parts.append(f"ΔH={data['delta_H']}kJ/mol")
    if data.get("jisim_g"):
        data_parts.append(f"m={data['jisim_g']}g")

    if data_parts:
        parts.append("Data: " + ", ".join(data_parts))

    return " | ".join(parts)


# ── MAIN FUNCTIONS ─────────────────────────────────────────────────────────────

async def extract_question_from_image(
    image_bytes: bytes,
    lang: str = "BM",
) -> Optional[str]:
    """
    Main entry point — backward compatible.
    Returns extracted question as string.
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
        cleaned = clean_extracted_text(raw_text)
        logger.info(f"Cleaned text ({len(raw_text)} → {len(cleaned)} chars)")
        return cleaned
    return None


async def extract_question_structured(
    image_bytes: bytes,
    lang: str = "BM",
) -> Optional[dict]:
    """
    NEW v2.0 — Returns structured dict for router/solver.

    Returns:
        {
            "soalan": str,
            "pilihan": dict,
            "data": dict,
            "formula_kimia": list,
            "persamaan": str,
            "soalan_type": str,
            "clean_question": str,
            "raw": str,
        }
    """
    raw_text = await extract_question_from_image(image_bytes, lang)
    if not raw_text:
        return None

    structured = preprocess_vision_question(raw_text)
    logger.info(f"Structured: type={structured['soalan_type']}, "
                f"formulas={structured['formula_kimia'][:3]}")
    return structured


def clean_extracted_text(text: str) -> str:
    """
    Post-process extracted text from vision AI.
    Remove LaTeX formatting, unicode subscripts, and other artifacts.
    """
    # Remove LaTeX math delimiters
    text = re.sub(r'\$\\rm\s*', '', text)
    text = re.sub(r'\$', '', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\rm\s*', '', text)

    # Remove LaTeX subscripts/superscripts → plain numbers
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

    # Unicode multiply
    text = text.replace('×', 'x')

    # Clean excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


async def interpret_question(
    raw_text: str,
    lang: str = "BM",
    groq_api_key: str = "",
    explain_model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Passthrough for Groq/Gemini (combined prompt).
    Only runs LLM interpretation for Tesseract OCR output.
    """
    if VISION_PROVIDER in ("groq", "gemini"):
        logger.info("interpret_question: skipped (groq/gemini combined prompt)")
        return raw_text

    if not groq_api_key or not raw_text:
        return raw_text

    prompt_bm = f"""Kamu adalah pembantu kimia SPM. Tugas kamu adalah menginterpret teks yang diextract dari gambar soalan kimia SPM.

TEKS DARI GAMBAR:
{raw_text}

TUGASAN:
1. Kenal pasti soalan kimia UTAMA yang perlu dijawab
2. Jika soalan MCQ — tulis soalan dan pilihan jawapan (A, B, C, D) dengan jelas
3. Jika ada data (Jisim atom relatif, kemolaran, dll) — sertakan dalam soalan
4. Jika ada formula kimia — pastikan dalam format biasa (H2O, NaOH, K4Fe(CN)6)
5. Buang teks tidak relevan (nombor halaman, header, footer)

PENTING — DERIVE FORMULA DARI NAMA KIMIA jika formula tidak kelihatan:
- "kalium heksasianoferat(III) terhidrat" → K4Fe(CN)6.3H2O
- "kuprum(II) sulfat pentahidrat" → CuSO4.5H2O
- "natrium tiosulfat" → Na2S2O3
- "ferum(III) oksida" → Fe2O3

OUTPUT: Tulis semula soalan dengan LENGKAP dan JELAS. JANGAN jawab soalan."""

    prompt_en = f"""You are an SPM chemistry assistant. Interpret extracted text from chemistry question image.

EXTRACTED TEXT:
{raw_text}

TASK: Rewrite the question completely and clearly with correct formulas.
DO NOT answer the question — only rewrite it clearly."""

    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=groq_api_key)
        resp = await client.chat.completions.create(
            model=explain_model,
            messages=[{"role": "user", "content": prompt_bm if lang == "BM" else prompt_en}],
            max_tokens=400,
            temperature=0.1,
        )
        interpreted = resp.choices[0].message.content.strip()
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
