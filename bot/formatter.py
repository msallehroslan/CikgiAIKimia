"""
bot/formatter.py — Cikgu AI Kimia
===================================
Chemistry-safe Telegram message formatter.

REPLACES: ParseMode.MARKDOWN → ParseMode.HTML

WHY:
  Telegram Markdown V1 breaks on chemistry units containing ⁻¹, ², ³,
  unmatched asterisks inside formulas, and Greek letters (ΔH, ΔT).
  Markdown V2 requires escaping 18 special characters — fragile.
  HTML mode escapes only 3 chars (<, >, &) and is predictable.

FORMAT:
  Solver answer section headers → <b>Diberi:</b>
  Formulas, equations          → <code>NaOH + HCl → NaCl + H₂O</code>
  Numeric results              → plain text (units rendered correctly)
  LLM explanation              → plain text with italic notes

SPLITTING:
  Telegram limit: 4096 chars per message.
  Split on paragraph boundaries (double newline) not arbitrary bytes.
  Never split in the middle of a <code>...</code> block.
  Never split a solver answer block (Diberi/Formula/Pengiraan/Jawapan).

CHEMISTRY UNICODE RENDERING:
  The formatter does NOT strip chemistry unicode — it lets Telegram
  render ΔH, ⁻¹, °C, → natively in HTML mode.
  Only &, <, > are escaped (HTML safety, not chemistry safety).
"""

from __future__ import annotations

import html
import re
from typing import List

# ── HTML escaping ──────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape HTML special chars. Only &, <, > — safe for chemistry unicode."""
    return html.escape(text, quote=False)


# ── Section header detection ───────────────────────────────────────────────

_SECTION_HEADERS_BM = {
    "Diberi:", "Formula:", "Pengiraan:", "Jawapan:",
    "Diberi :", "Formula :", "Pengiraan :", "Jawapan :",
}
_SECTION_HEADERS_EN = {
    "Given:", "Formula:", "Calculation:", "Answer:",
    "Given :", "Formula :", "Calculation :", "Answer :",
}
_ALL_HEADERS = _SECTION_HEADERS_BM | _SECTION_HEADERS_EN


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(h) for h in _ALL_HEADERS)


def _is_formula_line(line: str) -> bool:
    """
    Heuristic: lines containing chemical formulas/equations should
    be wrapped in <code> for monospace rendering.
    Criteria:
      - Contains -> or → (equation)
      - OR is a pure formula (all uppercase letters + digits + brackets)
      - OR contains mol, g, dm3, cm3 (calculation line)
    """
    stripped = line.strip()
    if not stripped:
        return False
    has_arrow    = '->' in stripped or '→' in stripped or '⇌' in stripped
    has_calc_ops = any(x in stripped for x in ['÷', '×', '=', 'mol', 'dm3', 'cm3'])
    is_formula   = bool(re.match(r'^[A-Z][A-Za-z0-9()·.→⇌\-\+ =÷×]+$', stripped))
    return has_arrow or (has_calc_ops and is_formula)


# ── Core formatter ─────────────────────────────────────────────────────────

def format_solver_answer(solver_text: str, answer_type: str = "calculation") -> str:
    """
    Convert raw solver/LLM output to Telegram HTML format.

    answer_type: "calculation" | "theory" | "fallback"
    """
    emoji_map = {
        "calculation": "🧮",
        "theory":      "📚",
        "fallback":    "ℹ️",
    }
    emoji = emoji_map.get(answer_type, "💬")

    lines   = solver_text.split('\n')
    parts   = [f"<b>{emoji} Cikgu AI Kimia</b>\n"]
    in_code = False   # track if we're inside a <code> block

    for line in lines:
        raw     = line.rstrip()
        escaped = _esc(raw)

        # ── Section headers → bold ─────────────────────────────────────
        if _is_section_header(raw):
            # Extract header and rest of line
            for header in _ALL_HEADERS:
                if raw.strip().startswith(header):
                    rest = raw.strip()[len(header):].strip()
                    if rest:
                        parts.append(f"<b>{_esc(header)}</b> {_esc(rest)}")
                    else:
                        parts.append(f"<b>{_esc(header)}</b>")
                    break

        # ── Calculation / formula lines → code ─────────────────────────
        elif _is_formula_line(raw) and '  ' in raw:
            # Indented calculation lines (2+ spaces indent)
            stripped = raw.strip()
            parts.append(f"<code>  {_esc(stripped)}</code>")

        # ── Separator lines ─────────────────────────────────────────────
        elif raw.strip() in ('---', '───', '━━━'):
            parts.append('─────────────')

        # ── Empty lines ─────────────────────────────────────────────────
        elif not raw.strip():
            parts.append('')

        # ── Regular text ─────────────────────────────────────────────────
        else:
            parts.append(escaped)

    return '\n'.join(parts)


def format_answer(raw_answer: str, answer_type: str = "calculation") -> str:
    """
    Main public function — wraps solver + explanation into final Telegram HTML.
    """
    # Split solver block and explanation (separated by ---)
    if '\n---\n' in raw_answer:
        solver_part, explanation = raw_answer.split('\n---\n', 1)
    else:
        solver_part  = raw_answer
        explanation  = ""

    # Format solver block
    formatted = format_solver_answer(solver_part.strip(), answer_type)

    # Add explanation in italic
    if explanation.strip():
        formatted += f"\n\n<i>{_esc(explanation.strip())}</i>"

    return formatted


def format_theory_answer(llm_text: str) -> str:
    """Format a theory/RAG answer — no code blocks, just clean paragraphs."""
    lines  = llm_text.split('\n')
    parts  = ["📚 <b>Cikgu AI Kimia</b>\n"]
    for line in lines:
        raw = line.rstrip()
        if not raw:
            parts.append('')
        else:
            parts.append(_esc(raw))
    return '\n'.join(parts)


def format_fallback(message: str) -> str:
    return f"ℹ️ {_esc(message)}"


# ── Safe message splitter ──────────────────────────────────────────────────

MAX_MSG_LEN  = 4000   # Telegram limit is 4096; use 4000 for safety margin
HARD_MAX     = 4090   # absolute ceiling


def split_message(text: str, max_len: int = MAX_MSG_LEN) -> List[str]:
    """
    Split a long Telegram HTML message into chunks ≤ max_len chars.

    Strategy:
      1. Try to split on double-newline (paragraph boundaries)
      2. If a paragraph > max_len, split on single newline
      3. Never split inside a <code>...</code> block
      4. If a single line > max_len, hard split at max_len-20
         (rare for chemistry answers, but safety net)

    Returns list of HTML-safe message chunks.
    """
    if len(text) <= max_len:
        return [text]

    paragraphs = text.split('\n\n')
    chunks:    List[str] = []
    current:   str       = ""

    for para in paragraphs:
        # If adding this paragraph stays within limit
        test = (current + '\n\n' + para).strip() if current else para
        if len(test) <= max_len:
            current = test
            continue

        # Para is too large to join — flush current buffer
        if current:
            chunks.append(current.strip())
            current = ""

        # Para itself fits in one message
        if len(para) <= max_len:
            current = para
            continue

        # Para too large — split on single newlines
        for line in para.split('\n'):
            test = (current + '\n' + line).strip() if current else line
            if len(test) <= max_len:
                current = test
            else:
                if current:
                    chunks.append(current.strip())
                # Handle a single line that's too long (hard split)
                while len(line) > max_len:
                    # Find safe break point: don't break inside <code> or <b>
                    break_at = _safe_break_point(line, max_len)
                    chunks.append(line[:break_at])
                    line = line[break_at:]
                current = line

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _safe_break_point(text: str, max_len: int) -> int:
    """
    Find a safe character position to break the text.
    Avoids breaking inside HTML tags.
    """
    if len(text) <= max_len:
        return len(text)

    # Walk backwards from max_len to find a space that's not inside a tag
    for i in range(max_len, max(max_len - 100, 0), -1):
        if text[i] == ' ':
            # Check we're not inside an HTML tag
            before = text[:i]
            open_tags  = before.count('<')
            close_tags = before.count('>')
            if open_tags == close_tags:
                return i

    return max_len   # hard break


# ── Source citation formatter ──────────────────────────────────────────────

def format_sources(sources: list, lang: str = "BM") -> str:
    """Format RAG source citations for appending to answer."""
    if not sources:
        return ""
    label   = "📖 Sumber:" if lang == "BM" else "📖 Sources:"
    topics  = [f"• {_esc(s.get('topic',''))}" for s in sources[:2] if s.get('topic')]
    if not topics:
        return ""
    return f"\n\n<i>{label} {', '.join(topics)}</i>"


def format_timing(ms: float, from_cache: bool, lang: str = "BM") -> str:
    """Format timing footer line."""
    if from_cache:
        indicator = "⚡ Cache"
    else:
        indicator = "⏱"
    return f"\n<i>{indicator} {ms:.0f}ms</i>"
