"""
vision/confidence_scorer.py — Cikgu AI Kimia  [PRODUCTION HARDENING v4.0]
==========================================================================
REPLACES: existing vision/confidence_scorer.py  (adds Groq-aware scoring)

Chemistry-aware OCR confidence scoring.
Zero LLM calls — pure deterministic Python.

SCORING ALGORITHM (0.0 – 1.0):

  Score starts at BASE_SCORE (0.50).
  Positive signals add to score (up to +0.50 total).
  Negative signals subtract from score (down to 0.0).

  POSITIVE SIGNALS (+):
    +0.15  Has chemistry keyword (mol, jisim, pH, etc.)
    +0.10  Has valid chemical formula (H2O, NaOH, etc.)
    +0.10  Has numeric value with unit (50 cm³, 2.0 mol/dm³)
    +0.08  Has chemical equation (arrow present: A -> B)
    +0.05  Has calculation type keyword (hitungkan, calculate)
    +0.05  Text length is reasonable (15–500 chars for a question)
    +0.03  Has Groq structured format markers (SOALAN:, DATA_NOMBOR:)

  NEGATIVE SIGNALS (−):
    −0.30  Garbage pattern detected (repeated chars, high non-ASCII)
    −0.20  Contains malformed formula (unknown elements, broken bracket)
    −0.15  Very short text (< 10 chars)
    −0.10  No chemistry keyword AND no numeric value
    −0.08  Malformed equation (unbalanced brackets, invalid arrow)
    −0.05  Too long (> 1500 chars — OCR over-extracted junk)

THRESHOLDS:
  ≥ 0.70  HIGH    → proceed to solver pipeline normally
  0.45–0.69  MEDIUM → show preview to user, ask to confirm
  < 0.45  LOW     → ask user to retype or send clearer image

SOURCE ADJUSTMENT:
  groq_vision:  no penalty  (already an LLM output, structurally correct)
  local_ocr:   −0.10 penalty (raw OCR output, higher noise floor)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── Thresholds ──────────────────────────────────────────────────────────────
CONF_HIGH   = 0.70
CONF_MEDIUM = 0.45

# ── Known valid SPM chemistry elements ──────────────────────────────────────
_VALID_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo",
    "Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","U",
}

# SPM-relevant formulas that commonly appear in questions
_COMMON_SPM_FORMULAS = {
    "H2O","HCl","NaOH","KOH","H2SO4","HNO3","Na2SO4","CaCO3",
    "CuSO4","FeCl3","NH3","CO2","SO2","NO2","CH4","C2H5OH",
    "NaCl","MgO","CaO","Fe2O3","Al2O3","KMnO4","Na2CO3",
    "NaHCO3","Ca(OH)2","NH4Cl","CaCl2","MgCl2","ZnSO4",
    "K2Cr2O7","Na2S2O3","K4Fe(CN)6",
}

# ── Compiled patterns ────────────────────────────────────────────────────────
_FORMULA_PATTERN  = re.compile(
    r'\b[A-Z][a-z]?\d*(?:\([A-Za-z0-9]+\)\d*)*[A-Za-z0-9]*\b'
)
_EQUATION_PATTERN = re.compile(r'.{2,}\s*->\s*.{2,}')
_NUMBER_WITH_UNIT = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:g|mol|dm3|cm3|kJ|J|°C|M|V|%|dm|cm|L|l)\b',
    re.IGNORECASE
)
_GARBAGE_PATTERNS = [
    re.compile(r'(.)\1{5,}'),          # aaaaa — repeated chars
    re.compile(r'[^\x00-\x7F]{6,}'),   # 6+ consecutive non-ASCII
    re.compile(r'[|\\]{4,}'),           # table borders
    re.compile(r'\b[A-Z]{10,}\b'),      # random all-caps ≥10 chars
    re.compile(r'[\x00-\x08\x0b-\x1f\x7f]'),  # control chars
]

_CHEM_KEYWORDS_BM = [
    "mol","jisim","isipadu","kepekatan","kemolaran",
    "hitungkan","tentukan","kira","berapakah","nyatakan",
    "ph","poh","entalpi","enthalpi","termokimia",
    "titrasi","stoikiometri","formula","persamaan",
    "asid","bes","garam","larutan","tindak balas",
    "kadar","pengoksidaan","penurunan","elektrod",
    "unsur","sebatian","ion","elektron","proton",
    "nombor oxidasi","jmr","ar",
]
_CHEM_KEYWORDS_EN = [
    "mol","mass","volume","concentration","molarity",
    "calculate","determine","find","what is","state",
    "ph","poh","enthalpy","thermochem",
    "titration","stoichiometry","formula","equation",
    "acid","base","salt","solution","reaction",
    "rate","oxidation","reduction","electrode",
    "element","compound","ion","electron","proton",
    "relative","molar","empirical",
]

# Groq structured format markers (strong positive signal)
_GROQ_MARKERS = ["SOALAN:", "PILIHAN:", "DATA_NOMBOR:", "FORMULA_KIMIA:",
                  "PERSAMAAN_KIMIA:", "JENIS_PENGIRAAN:", "QUESTION:", "OPTIONS:"]


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfidenceResult:
    score:        float        # 0.0 – 1.0
    tier:         str          # "high" | "medium" | "low"
    signals:      List[str]    # human-readable scoring log
    proceed:      bool         # safe to pass to solver
    ask_confirm:  bool         # show preview, ask user to confirm
    ask_retype:   bool         # image too bad, ask user to retype
    source:       str = ""     # "groq_vision" | "local_ocr"


# ═══════════════════════════════════════════════════════════════════════════════
# FORMULA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _is_valid_formula(formula: str) -> bool:
    """
    Check if a string is a plausible chemical formula.
    Validates that all element symbols are in the known SPM element set.
    Does not validate stoichiometric correctness (that's the solver's job).
    """
    # Strip state symbols: (aq), (s), (l), (g)
    clean = re.sub(r'\((aq|s|l|g)\)', '', formula, flags=re.IGNORECASE)
    # Strip charge notation: 2-, +, 3+
    clean = re.sub(r'[\^]?[0-9]*[+-]$', '', clean.strip())

    # Extract all element symbols
    elements = re.findall(r'[A-Z][a-z]?', clean)
    if not elements:
        return False

    for elem in elements:
        if elem not in _VALID_ELEMENTS:
            return False

    return True


def _detect_formulas(text: str) -> List[str]:
    """Extract all plausible chemical formula tokens from text."""
    candidates = _FORMULA_PATTERN.findall(text)
    valid = []
    for c in candidates:
        if c in _COMMON_SPM_FORMULAS or _is_valid_formula(c):
            valid.append(c)
    return valid


def _detect_malformed_formula(text: str) -> Optional[str]:
    """
    Detect obviously malformed chemistry notation.
    Returns a description of the problem, or None if no issue detected.
    """
    # Unbalanced brackets
    if text.count("(") != text.count(")"):
        return "unbalanced_parentheses"
    if text.count("[") != text.count("]"):
        return "unbalanced_brackets"

    # Known OCR corruption patterns
    ocr_corruptions = [
        (r'\b[A-Z]\d[A-Z](?!\w)', "possible_digit_for_letter"),   # H2C → H2Cl?
        (r'\b[0-9][A-Z][0-9]\b',   "number_letter_number"),         # likely garbage
        (r'\b[a-z]{4,}\b',         "all_lowercase_long_word"),       # OCR lost caps
    ]
    for pat, reason in ocr_corruptions:
        if re.search(pat, text):
            return reason

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def score_ocr_confidence(text: str, source: str = "unknown") -> ConfidenceResult:
    """
    Score extracted OCR text for chemistry relevance and quality.

    Args:
        text:   The extracted text from Groq Vision or local OCR.
        source: "groq_vision" | "local_ocr" | "unknown"

    Returns:
        ConfidenceResult with score, tier, and action flags.
    """
    signals: List[str] = []
    score = 0.50   # base

    if not text or not text.strip():
        return ConfidenceResult(
            score=0.0, tier="low", signals=["empty_text"],
            proceed=False, ask_confirm=False, ask_retype=True,
            source=source,
        )

    text_lower = text.lower()
    text_len   = len(text)

    # ── SOURCE PENALTY ──────────────────────────────────────────────────────
    if source == "local_ocr":
        score -= 0.10
        signals.append("source_penalty=local_ocr(-0.10)")
    elif source == "groq_vision":
        # No penalty — LLM output is structurally cleaner
        signals.append("source=groq_vision(no_penalty)")

    # ── GARBAGE DETECTION ───────────────────────────────────────────────────
    garbage_hits = 0
    for gp in _GARBAGE_PATTERNS:
        if gp.search(text):
            garbage_hits += 1

    if garbage_hits >= 2:
        score -= 0.30
        signals.append(f"garbage_patterns={garbage_hits}(-0.30)")
    elif garbage_hits == 1:
        score -= 0.15
        signals.append(f"garbage_pattern=1(-0.15)")

    # ── TEXT LENGTH ─────────────────────────────────────────────────────────
    if text_len < 10:
        score -= 0.15
        signals.append(f"very_short_text(len={text_len})(-0.15)")
    elif text_len < 20:
        score -= 0.05
        signals.append(f"short_text(len={text_len})(-0.05)")
    elif 20 <= text_len <= 500:
        score += 0.05
        signals.append(f"good_length(len={text_len})(+0.05)")
    elif text_len > 1500:
        score -= 0.05
        signals.append(f"too_long(len={text_len})(-0.05)")

    # ── GROQ STRUCTURED FORMAT MARKERS ─────────────────────────────────────
    groq_markers_found = [m for m in _GROQ_MARKERS if m in text.upper()]
    if len(groq_markers_found) >= 3:
        score += 0.10   # strong positive: structured Groq output
        signals.append(f"groq_markers={len(groq_markers_found)}(+0.10)")
    elif len(groq_markers_found) >= 1:
        score += 0.05
        signals.append(f"groq_marker=1(+0.05)")

    # ── CHEMISTRY KEYWORDS ──────────────────────────────────────────────────
    bm_hits = sum(1 for kw in _CHEM_KEYWORDS_BM if kw in text_lower)
    en_hits = sum(1 for kw in _CHEM_KEYWORDS_EN if kw in text_lower)
    chem_hits = max(bm_hits, en_hits)

    if chem_hits >= 3:
        score += 0.15
        signals.append(f"chem_keywords={chem_hits}(+0.15)")
    elif chem_hits >= 1:
        score += 0.08
        signals.append(f"chem_keyword=1(+0.08)")
    else:
        score -= 0.10
        signals.append("no_chem_keywords(-0.10)")

    # ── CHEMICAL FORMULAS ────────────────────────────────────────────────────
    formulas = _detect_formulas(text)
    if len(formulas) >= 2:
        score += 0.10
        signals.append(f"valid_formulas={formulas[:3]}(+0.10)")
    elif len(formulas) == 1:
        score += 0.05
        signals.append(f"valid_formula=1(+0.05)")

    # Malformed formula check
    malformed = _detect_malformed_formula(text)
    if malformed:
        score -= 0.20
        signals.append(f"malformed_formula={malformed}(-0.20)")

    # ── NUMERIC VALUES WITH UNITS ────────────────────────────────────────────
    unit_matches = _NUMBER_WITH_UNIT.findall(text)
    if len(unit_matches) >= 2:
        score += 0.10
        signals.append(f"numeric_with_units={len(unit_matches)}(+0.10)")
    elif len(unit_matches) == 1:
        score += 0.05
        signals.append(f"numeric_with_unit=1(+0.05)")

    # ── EQUATION DETECTION ───────────────────────────────────────────────────
    if _EQUATION_PATTERN.search(text):
        score += 0.08
        signals.append("equation_detected(+0.08)")

    # ── CALCULATION KEYWORDS ─────────────────────────────────────────────────
    calc_keywords = [
        "hitungkan","calculate","berapakah","find","hitung","tentukan",
        "determine","nilai","value",
    ]
    if any(kw in text_lower for kw in calc_keywords):
        score += 0.05
        signals.append("calculation_keyword(+0.05)")

    # ── CLAMP ────────────────────────────────────────────────────────────────
    score = max(0.0, min(1.0, score))

    # ── DETERMINE TIER ───────────────────────────────────────────────────────
    if score >= CONF_HIGH:
        tier         = "high"
        proceed      = True
        ask_confirm  = False
        ask_retype   = False
    elif score >= CONF_MEDIUM:
        tier         = "medium"
        proceed      = False
        ask_confirm  = True
        ask_retype   = False
    else:
        tier         = "low"
        proceed      = False
        ask_confirm  = False
        ask_retype   = True

    return ConfidenceResult(
        score=round(score, 3),
        tier=tier,
        signals=signals,
        proceed=proceed,
        ask_confirm=ask_confirm,
        ask_retype=ask_retype,
        source=source,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# USER MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════

def medium_confidence_message(extracted_text: str, lang: str = "BM") -> str:
    """
    Message shown to user when OCR confidence is MEDIUM.
    Shows a preview of what was extracted and asks for confirmation.
    """
    preview = extracted_text[:200] + ("..." if len(extracted_text) > 200 else "")
    if lang == "BM":
        return (
            f"📷 Cikgu AI telah baca gambar ini:\n\n"
            f"<code>{preview}</code>\n\n"
            f"Adakah ini betul? Jawab:\n"
            f"  ✅ <b>Ya</b> — teruskan penyelesaian\n"
            f"  ✏️ <b>Tidak</b> — taip soalan dalam teks"
        )
    else:
        return (
            f"📷 I extracted this from your image:\n\n"
            f"<code>{preview}</code>\n\n"
            f"Is this correct?\n"
            f"  ✅ <b>Yes</b> — proceed with solution\n"
            f"  ✏️ <b>No</b> — please type the question as text"
        )


def low_confidence_message(lang: str = "BM") -> str:
    """Message shown when OCR confidence is LOW."""
    if lang == "BM":
        return (
            "📷 Maaf, gambar kurang jelas untuk dibaca dengan tepat.\n\n"
            "Sila cuba:\n"
            "  1️⃣ Hantar gambar yang lebih jelas\n"
            "  2️⃣ Taip soalan terus dalam teks\n\n"
            "Tip: cahaya yang baik + kamera tegak = gambar yang lebih jelas 📸"
        )
    else:
        return (
            "📷 Sorry, the image is unclear for accurate reading.\n\n"
            "Please try:\n"
            "  1️⃣ Send a clearer image\n"
            "  2️⃣ Type the question as text\n\n"
            "Tip: good lighting + straight camera angle = clearer image 📸"
        )
