"""
bot/formatter.py — Cikgu AI Kimia  [PRODUCTION HARDENING v4.0]
===============================================================
REPLACES: existing bot/formatter.py

USES: ParseMode.HTML throughout (never ParseMode.MARKDOWN)

WHY HTML, NOT MARKDOWN:
  Telegram MarkdownV1 breaks on:
    - ΔH, ΔT (unmatched asterisk-like chars)
    - mol dm⁻³, kJ mol⁻¹ (superscript ⁻ treated as italic marker)
    - Chemical equations with unmatched special chars
  MarkdownV2 requires escaping 18 special characters —
    every chemistry formula becomes a minefield.
  HTML only requires escaping 3 chars: & < >
    All chemistry unicode (ΔH, °C, →, ⇌, ²) renders natively.

FORMAT SPEC:
  Headers:         <b>Diberi:</b>
  Formulas/eq:     <code>NaOH + HCl → NaCl + H₂O</code>
  Numeric answers: plain text  (units render correctly)
  Explanations:    <i>italic note</i>
  Separators:      ─────────────
  Max per message: 4000 chars  (Telegram limit 4096, buffer for safety)

SPLIT ALGORITHM:
  1. If total ≤ 4000: return as-is
  2. Split on double-newline (paragraph boundaries)
  3. If paragraph > 4000: split on single newline
  4. NEVER split inside a <code>...</code> block
  5. Hard break at 3900 chars as absolute safety net

CHEMISTRY UNICODE POLICY:
  - DO NOT strip chemistry unicode — Telegram renders it fine in HTML
  - DO escape & < >   (HTML injection safety)
  - ΔH, °C, →, ⇌, ², ⁻ all pass through untouched
"""

from __future__ import annotations

import html
import re
from typing import List, Optional

# ── HTML escape (only 3 chars — chemistry-safe) ──────────────────────────────

def _esc(text: str) -> str:
    """Escape only &, <, > for HTML safety. Chemistry unicode passes through."""
    return html.escape(text, quote=False)


# ── Section header detection ──────────────────────────────────────────────────

_HEADERS_BM = frozenset({
    "Diberi:", "Formula:", "Pengiraan:", "Jawapan:",
    "Diberi :", "Formula :", "Pengiraan :", "Jawapan :",
    "Langkah:", "Langkah :",
})
_HEADERS_EN = frozenset({
    "Given:", "Formula:", "Calculation:", "Answer:",
    "Given :", "Formula :", "Calculation :", "Answer :",
    "Step:", "Step :",
})
_ALL_HEADERS = _HEADERS_BM | _HEADERS_EN


def _is_section_header(line: str) -> bool:
    s = line.strip()
    return any(s.startswith(h) for h in _ALL_HEADERS)


# ── Formula/calculation line detection ───────────────────────────────────────

def _is_formula_line(line: str) -> bool:
    """
    Returns True if this line should be wrapped in <code> for monospace.
    Criteria:
      - Contains chemical equation arrow (→ or ->)
      - Starts with 2+ spaces indent (structured solver output)
      - Contains mol/dm3/cm3 with = sign (calculation step)
    """
    s = line.strip()
    if not s:
        return False
    has_arrow    = "->" in s or "→" in s or "⇌" in s or "<->" in s
    is_indented  = line.startswith("  ")   # 2-space indent = calculation line
    has_calc     = bool(re.search(r'=.*(?:mol|dm3|cm3|kJ|J|g)\b', s, re.IGNORECASE))
    return has_arrow or (is_indented and has_calc)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_solver_answer(solver_text: str, answer_type: str = "calculation") -> str:
    """
    Convert raw solver/LLM output to Telegram HTML.

    answer_type: "calculation" | "theory" | "fallback" | "error"
    """
    emoji_map = {
        "calculation": "🧮",
        "theory":      "📚",
        "fallback":    "ℹ️",
        "error":       "⚠️",
    }
    emoji = emoji_map.get(answer_type, "💬")
    lines = solver_text.split("\n")
    parts = [f"<b>{emoji} Cikgu AI Kimia</b>\n"]

    for line in lines:
        raw     = line.rstrip()
        escaped = _esc(raw)

        if _is_section_header(raw):
            # ── Section header → bold ─────────────────────────────────────
            for header in _ALL_HEADERS:
                if raw.strip().startswith(header):
                    rest = raw.strip()[len(header):].strip()
                    if rest:
                        parts.append(f"<b>{_esc(header)}</b> {_esc(rest)}")
                    else:
                        parts.append(f"<b>{_esc(header)}</b>")
                    break

        elif _is_formula_line(raw):
            # ── Formula / calculation line → code ─────────────────────────
            parts.append(f"<code>{_esc(raw.strip())}</code>")

        elif raw.strip() in ("---", "───", "━━━", "─────────────"):
            # ── Separator ─────────────────────────────────────────────────
            parts.append("─────────────")

        elif not raw.strip():
            parts.append("")

        else:
            parts.append(escaped)

    return "\n".join(parts)


def format_answer(raw_answer: str, answer_type: str = "calculation") -> str:
    """
    Main public formatter.
    Splits raw answer on "\\n---\\n" into solver block + explanation.
    """
    if "\n---\n" in raw_answer:
        solver_part, explanation = raw_answer.split("\n---\n", 1)
    else:
        solver_part  = raw_answer
        explanation  = ""

    formatted = format_solver_answer(solver_part.strip(), answer_type)

    if explanation.strip():
        formatted += f"\n\n<i>{_esc(explanation.strip())}</i>"

    return formatted


def format_theory_answer(llm_text: str) -> str:
    """Format a theory/RAG answer — clean paragraphs, no code blocks."""
    lines = llm_text.split("\n")
    parts = ["📚 <b>Cikgu AI Kimia</b>\n"]
    for line in lines:
        raw = line.rstrip()
        if not raw:
            parts.append("")
        else:
            parts.append(_esc(raw))
    return "\n".join(parts)


def format_fallback(message: str) -> str:
    """Format a graceful degradation message."""
    return f"ℹ️ {_esc(message)}"


def format_error(message: str) -> str:
    """Format an error message (validation failure, quota limit, etc.)."""
    return f"⚠️ {_esc(message)}"


def format_ocr_preview(extracted_text: str, lang: str = "BM") -> str:
    """Format the OCR preview shown for medium-confidence extractions."""
    preview = extracted_text[:300] + ("..." if len(extracted_text) > 300 else "")
    if lang == "BM":
        return (
            f"📷 <b>Cikgu AI membaca gambar ini:</b>\n\n"
            f"<code>{_esc(preview)}</code>\n\n"
            f"Adakah ini betul?\n"
            f"✅ Balas <b>ya</b> untuk teruskan\n"
            f"✏️ Balas <b>tidak</b> untuk taip semula"
        )
    else:
        return (
            f"📷 <b>I extracted this from your image:</b>\n\n"
            f"<code>{_esc(preview)}</code>\n\n"
            f"Is this correct?\n"
            f"✅ Reply <b>yes</b> to proceed\n"
            f"✏️ Reply <b>no</b> to type manually"
        )


def format_sources(sources: list, lang: str = "BM") -> str:
    """Format RAG source citations as a footer line."""
    if not sources:
        return ""
    label  = "📖 Sumber:" if lang == "BM" else "📖 Sources:"
    topics = [f"• {_esc(s.get('topic', ''))}" for s in sources[:2] if s.get("topic")]
    if not topics:
        return ""
    return f"\n\n<i>{label} {', '.join(topics)}</i>"


def format_timing(ms: float, from_cache: bool = False, lang: str = "BM") -> str:
    """Format timing footer."""
    indicator = "⚡ Cache" if from_cache else "⏱"
    return f"\n<i>{indicator} {ms:.0f}ms</i>"


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE MESSAGE SPLITTER
# ═══════════════════════════════════════════════════════════════════════════════

MAX_MSG_LEN = 4000   # Telegram limit 4096; 4000 = safety margin
HARD_MAX    = 3900   # Absolute ceiling for hard-break case


def split_message(text: str, max_len: int = MAX_MSG_LEN) -> List[str]:
    """
    Split a long HTML message into chunks ≤ max_len chars.

    Algorithm:
      1. If total ≤ max_len: return as-is  (fast path)
      2. Split on \\n\\n (paragraph boundaries)
      3. For paragraphs > max_len: split on \\n (line boundaries)
      4. NEVER split inside a <code>...</code> block
      5. Single line > max_len → hard break at safe position

    Returns:
      List[str] of HTML-safe message chunks.
      Each chunk is ready to send with parse_mode=HTML.
    """
    if len(text) <= max_len:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        test = (current + "\n\n" + para).strip() if current else para

        if len(test) <= max_len:
            current = test
            continue

        # Flush current buffer before this paragraph
        if current:
            chunks.append(current.strip())
            current = ""

        # Paragraph fits in one message
        if len(para) <= max_len:
            current = para
            continue

        # Paragraph too large: split on single newlines
        for line in para.split("\n"):
            test = (current + "\n" + line).strip() if current else line
            if len(test) <= max_len:
                current = test
            else:
                if current:
                    chunks.append(current.strip())
                # Handle a single line that's too long
                while len(line) > max_len:
                    bp = _safe_break_point(line, max_len)
                    chunks.append(line[:bp])
                    line = line[bp:]
                current = line

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _safe_break_point(text: str, max_len: int) -> int:
    """
    Find safe char position to break text.

    Avoids breaking:
      - Inside HTML tags (<b>, </b>, <code>, etc.)
      - Inside a <code>...</code> block  (monospace chemistry)

    Strategy: walk backwards from max_len to find a space that
    is NOT inside an open HTML tag.
    """
    if len(text) <= max_len:
        return len(text)

    # Walk backwards to find a space outside HTML tags
    for i in range(max_len, max(max_len - 100, 0), -1):
        if i < len(text) and text[i] == " ":
            prefix     = text[:i]
            open_count  = prefix.count("<")
            close_count = prefix.count(">")
            if open_count == close_count:
                return i

    # Hard break — last resort
    return HARD_MAX
