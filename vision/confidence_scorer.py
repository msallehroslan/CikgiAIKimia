"""
vision/confidence_scorer.py — Cikgu AI Kimia
==============================================
Chemistry-aware OCR confidence scoring.

Scores extracted text on 0.0–1.0 scale by checking:
  1. Structural completeness  — does text look like a question?
  2. Formula integrity        — are detected formulas chemically valid?
  3. Numeric integrity        — are numbers sensible (not garbled)?
  4. Equation integrity       — if equation present, is it parseable?
  5. OCR garbage signals      — high non-ASCII, repeated chars, etc.

Thresholds:
  ≥ 0.70  → HIGH   — proceed to solver pipeline normally
  0.45–0.69 → MEDIUM — show extracted text to user, ask to confirm
  < 0.45  → LOW    — ask user to retype or send clearer image

This module does NOT call any LLM. Pure deterministic scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# ── Threshold constants ────────────────────────────────────────────────────
CONF_HIGH   = 0.70
CONF_MEDIUM = 0.45


@dataclass
class ConfidenceResult:
    score: float                   # 0.0 – 1.0
    tier: str                      # "high" | "medium" | "low"
    signals: List[str]             # human-readable reasons
    proceed: bool                  # True = safe to run solver
    ask_confirm: bool              # True = show preview, ask user to confirm
    ask_retype: bool               # True = image too bad, ask user to retype


# ── Known valid chemistry elements ────────────────────────────────────────
_VALID_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Ag","Sn","I","Ba","Pb","Hg","Au","Pt",
}

# Common SPM chemistry formula patterns (positive indicators)
_FORMULA_PATTERN   = re.compile(r'\b[A-Z][a-z]?\d*(?:\([A-Za-z0-9]+\)\d*)*[A-Za-z0-9]*\b')
_EQUATION_PATTERN  = re.compile(r'.{2,}\s*->\s*.{2,}')
_NUMBER_PATTERN    = re.compile(r'\b\d+(?:\.\d+)?\s*(?:g|mol|dm3|cm3|kJ|J|°C|M|V|%)\b', re.IGNORECASE)

# OCR garbage signals
_GARBAGE_PATTERNS = [
    re.compile(r'(.)\1{4,}'),          # "aaaaa" — repeated chars
    re.compile(r'[^\x00-\x7F]{5,}'),   # 5+ consecutive non-ASCII
    re.compile(r'[|\\]{3,}'),           # table borders misread as chars
    re.compile(r'\b[A-Z]{8,}\b'),       # random all-caps word ≥8 chars
]

# Expected chemistry keywords (at least one should appear for chemistry question)
_CHEM_KEYWORDS_BM = [
    "mol", "jisim", "isipadu", "kepekatan", "kemolaran",
    "hitungkan", "tentukan", "kira", "berapakah", "nyatakan",
    "ph", "poh", "entalpi", "enthalpi", "termokimia",
    "titrasi", "stoikiometri", "formula", "persamaan",
    "asid", "bes", "garam", "larutan", "tindak balas",
    "kadar", "pengoksidaan", "penurunan", "elektrod",
]
_CHEM_KEYWORDS_EN = [
    "mol", "mass", "volume", "concentration", "molarity",
    "calculate", "determine", "find", "what is",
    "ph", "poh", "enthalpy", "thermochem",
    "titration", "stoichiometry", "formula", "equation",
    "acid", "base", "salt", "solution", "reaction",
    "rate", "oxidation", "reduction", "electrode",
]


def _detect_garbled(text: str) -> List[str]:
    """Return list of garbage signal descriptions found in text."""
    found = []
    for pat in _GARBAGE_PATTERNS:
        m = pat.search(text)
        if m:
            found.append(f"garbage_pattern:'{m.group()[:20]}'")
    return found


def _score_formulas(text: str) -> tuple[float, List[str]]:
    """
    Extract apparent chemistry formulas and check element validity.
    Returns (score_component, signals).
    """
    candidates = _FORMULA_PATTERN.findall(text)
    if not candidates:
        return 0.5, []   # neutral — not all questions have formulas

    valid_count   = 0
    invalid_found = []

    for cand in candidates[:10]:   # check first 10 candidates
        # Extract first element symbol
        elem_match = re.match(r'([A-Z][a-z]?)', cand)
        if not elem_match:
            continue
        elem = elem_match.group(1)
        if elem in _VALID_ELEMENTS:
            valid_count += 1
        else:
            # Could be OCR artifact, BM word start, or genuinely unknown
            if len(cand) > 1 and cand[0].isupper():
                invalid_found.append(cand)

    # Ratio of valid vs total detected
    total = max(len(candidates), 1)
    ratio = valid_count / total

    signals = []
    if invalid_found:
        signals.append(f"suspect_formula:{invalid_found[:3]}")

    if ratio >= 0.7:
        return 0.9, signals
    elif ratio >= 0.4:
        return 0.6, signals
    else:
        return 0.3, signals + ["low_formula_validity"]


def _score_numbers(text: str) -> tuple[float, List[str]]:
    """Check numeric values have sensible units and ranges."""
    matches = _NUMBER_PATTERN.findall(text)
    signals = []

    if not matches:
        # No numeric values — may be theory question, not suspicious
        return 0.6, []

    # Check for obviously garbled numbers (e.g., "1.2.3g", "0.0.1mol")
    bad_numbers = re.findall(r'\d+\.\d+\.\d+', text)
    if bad_numbers:
        signals.append(f"malformed_number:{bad_numbers[:2]}")
        return 0.3, signals

    return 0.85, signals


def _score_question_structure(text: str) -> tuple[float, List[str]]:
    """
    Does the text look like a complete chemistry question?
    Awards score for: question words, reasonable length, chemistry keywords.
    """
    signals = []
    tl      = text.lower()

    # Minimum length
    if len(text) < 10:
        return 0.1, ["text_too_short"]
    if len(text) < 25:
        signals.append("text_very_short")
        return 0.3, signals

    # Chemistry keyword presence
    all_kw    = _CHEM_KEYWORDS_BM + _CHEM_KEYWORDS_EN
    kw_hits   = sum(1 for kw in all_kw if kw in tl)

    if kw_hits >= 3:
        kw_score = 1.0
    elif kw_hits == 2:
        kw_score = 0.8
    elif kw_hits == 1:
        kw_score = 0.6
        signals.append("few_chemistry_keywords")
    else:
        kw_score = 0.3
        signals.append("no_chemistry_keywords")

    # Question structure — has a verb (asking something)
    question_verbs = ["hitungkan","kira","tentukan","berapakah","calculate",
                      "find","determine","what","which","why","explain"]
    has_verb = any(v in tl for v in question_verbs)
    if not has_verb:
        signals.append("no_question_verb")
        kw_score *= 0.8

    return min(kw_score, 1.0), signals


def _score_non_ascii_ratio(text: str) -> tuple[float, List[str]]:
    """
    High ratio of unexpected non-ASCII = OCR garbling.
    Chemistry allows: ΔHΩ°→⇌ and subscript/superscript digits.
    """
    allowed_special = set("°ΔΩμ→⇌⁺⁻₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹")
    bad_chars = [c for c in text
                 if ord(c) > 127 and c not in allowed_special]
    ratio = len(bad_chars) / max(len(text), 1)

    if ratio > 0.25:
        return 0.1, [f"high_non_ascii_ratio:{ratio:.2f}"]
    if ratio > 0.10:
        return 0.5, [f"moderate_non_ascii_ratio:{ratio:.2f}"]
    return 1.0, []


def _score_equation(text: str) -> tuple[float, List[str]]:
    """If an equation is present, verify it has identifiable LHS and RHS."""
    if '->' not in text and '→' not in text and '<->' not in text:
        return 0.7, []   # neutral — not all questions have equations

    parts = re.split(r'->|→|<->', text)
    if len(parts) < 2:
        return 0.4, ["equation_missing_rhs"]

    lhs = parts[0].strip().split()[-3:]   # last 3 words of LHS
    rhs = parts[1].strip().split()[:3]    # first 3 words of RHS

    lhs_str = " ".join(lhs)
    rhs_str = " ".join(rhs)

    # Both sides should have at least one capital letter (formula)
    has_caps_lhs = any(c.isupper() for c in lhs_str)
    has_caps_rhs = any(c.isupper() for c in rhs_str)

    if has_caps_lhs and has_caps_rhs:
        return 0.9, []
    else:
        return 0.5, [f"equation_suspect_lhs='{lhs_str}' rhs='{rhs_str}'"]


# ── Composite Scorer ───────────────────────────────────────────────────────

def score_ocr_confidence(
    text: str,
    source: str = "unknown",   # "groq_vision" | "tesseract" | "paddle"
) -> ConfidenceResult:
    """
    Compute overall OCR confidence for extracted chemistry question text.

    Weights:
      - Structure:    25%
      - Formula:      25%
      - Numbers:      15%
      - Non-ASCII:    20%
      - Equation:     10%
      - Garbage:       5% (penalty only)

    source="groq_vision" gets a small bonus (model understands context).
    """
    signals: List[str] = []

    # ── Individual component scores ────────────────────────────────────
    s_struct,   sig1 = _score_question_structure(text)
    s_formula,  sig2 = _score_formulas(text)
    s_numbers,  sig3 = _score_numbers(text)
    s_nonascii, sig4 = _score_non_ascii_ratio(text)
    s_equation, sig5 = _score_equation(text)

    signals += sig1 + sig2 + sig3 + sig4 + sig5

    # ── Garbage penalty ────────────────────────────────────────────────
    garbage_sigs = _detect_garbled(text)
    signals += garbage_sigs
    garbage_penalty = 0.4 * len(garbage_sigs)   # each garbage signal costs 0.4

    # ── Weighted composite ─────────────────────────────────────────────
    raw_score = (
        0.25 * s_struct   +
        0.25 * s_formula  +
        0.15 * s_numbers  +
        0.20 * s_nonascii +
        0.15 * s_equation
    )

    # Source bonus: Groq Vision generally more reliable than Tesseract
    if source == "groq_vision":
        raw_score = min(raw_score + 0.05, 1.0)
    elif source == "paddle":
        raw_score = min(raw_score + 0.02, 1.0)

    final_score = max(0.0, raw_score - garbage_penalty)
    final_score = round(final_score, 3)

    # ── Tier classification ────────────────────────────────────────────
    if final_score >= CONF_HIGH:
        tier, proceed, ask_confirm, ask_retype = "high",   True,  False, False
    elif final_score >= CONF_MEDIUM:
        tier, proceed, ask_confirm, ask_retype = "medium", True,  True,  False
    else:
        tier, proceed, ask_confirm, ask_retype = "low",    False, False, True

    return ConfidenceResult(
        score=final_score,
        tier=tier,
        signals=signals,
        proceed=proceed,
        ask_confirm=ask_confirm,
        ask_retype=ask_retype,
    )


def format_confidence_warning(result: ConfidenceResult, lang: str = "BM") -> Optional[str]:
    """
    Return a user-facing warning message if confidence is not HIGH.
    Returns None if confidence is high (no message needed).
    """
    if result.tier == "high":
        return None

    if lang == "BM":
        if result.ask_retype:
            return (
                "⚠️ _Maaf, Cikgu AI tidak dapat membaca soalan dengan jelas._\n\n"
                "Sila cuba:\n"
                "• 📸 Hantar gambar yang lebih jelas dan terang\n"
                "• ✍️ Taip soalan dalam teks terus\n"
                "• 🔍 Pastikan soalan tidak kabur atau miring"
            )
        else:  # medium
            return (
                "📋 _Soalan yang Cikgu AI baca:_\n\n"
                "{preview}\n\n"
                "_Adakah soalan ini betul? Jawab 'ya' untuk teruskan atau "
                "taip semula soalan anda._"
            )
    else:
        if result.ask_retype:
            return (
                "⚠️ _Sorry, I couldn't read the question clearly._\n\n"
                "Please try:\n"
                "• 📸 Send a clearer, well-lit photo\n"
                "• ✍️ Type the question as text\n"
                "• 🔍 Make sure the image is sharp and upright"
            )
        else:
            return (
                "📋 _Question I detected:_\n\n"
                "{preview}\n\n"
                "_Is this correct? Reply 'yes' to continue or retype your question._"
            )
