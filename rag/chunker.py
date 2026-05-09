"""
chunker.py — Cikgu AI Kimia RAG Pipeline
=========================================
Heading-hierarchy Markdown chunker with:
  - H1 / H2 / H3 split boundaries
  - Atomic worked-example preservation (never split Diberi/Formula/Pengiraan/Jawapan)
  - Formula block preservation
  - Diagram reference extraction
  - SPM answer structure detection
  - Table preservation
  - Keyword extraction from inline `Keywords:` markers

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class RawChunk:
    """A single content chunk before metadata tagging."""
    chunk_id: str
    source_file: str
    heading_h1: str
    heading_h2: str
    heading_h3: str
    content: str
    raw_keywords: List[str]
    formulas: List[str]
    diagrams: List[dict]       # [{"alt": str, "path": str}]
    equations: List[str]
    has_worked_example: bool
    has_table: bool
    char_count: int

    @property
    def full_heading(self) -> str:
        parts = [p for p in [self.heading_h1, self.heading_h2, self.heading_h3] if p]
        return " > ".join(parts)


# ---------------------------------------------------------------------------
# REGEX PATTERNS
# ---------------------------------------------------------------------------

_H1 = re.compile(r'^# (.+)$', re.MULTILINE)
_H2 = re.compile(r'^## (.+)$', re.MULTILINE)
_H3 = re.compile(r'^### (.+)$', re.MULTILINE)

# Inline chemical formulas: H2O, CaCO3, Al2(SO4)3, etc.
_FORMULA_INLINE = re.compile(
    r'\b(?:[A-Z][a-z]?)(?:\d+)?(?:\((?:[A-Z][a-z]?\d*)+\)\d*)*'
    r'(?:[A-Z][a-z]?\d*)*'
    r'(?:\^?[+-]?\d*)?'
    r'(?:\([aqlsg]+\))?\b'
)

# Chemical equations: contains → or → or ⇌ or ->
_EQUATION = re.compile(r'.{3,}(?:→|⇌|->|⟶).{3,}')

# Diagram references: ![alt text](path)
_DIAGRAM = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

# Keywords line: `Keywords: word1, word2`
_KEYWORDS = re.compile(r'Keywords?:\s*(.+)', re.IGNORECASE)

# SPM worked-example markers
_WORKED_EXAMPLE_MARKERS = re.compile(
    r'(?:^##+ (?:Soalan|Contoh|Pengiraan|Diberi|Formula|Jawapan|Langkah))',
    re.MULTILINE | re.IGNORECASE
)

# Section separator (--- horizontal rule)
_SEPARATOR = re.compile(r'^---+$', re.MULTILINE)

# Table detection
_TABLE = re.compile(r'^\|.+\|', re.MULTILINE)

# Calculation file header
_CALC_HEADER = re.compile(r'^Jenis:\s*calculation', re.MULTILINE | re.IGNORECASE)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _generate_id(source: str, heading: str, idx: int) -> str:
    stem = Path(source).stem.lower().replace(' ', '_')
    slug = re.sub(r'[^a-z0-9_]', '', heading.lower().replace(' ', '_'))[:40]
    return f"{stem}__{slug}__{idx:04d}"


def _extract_keywords(text: str) -> List[str]:
    kws = []
    for m in _KEYWORDS.finditer(text):
        raw = m.group(1).strip()
        kws.extend([k.strip() for k in raw.split(',') if k.strip()])
    return list(dict.fromkeys(kws))  # deduplicate, preserve order


def _extract_formulas(text: str) -> List[str]:
    """
    Extract chemical formulas from text.
    Uses a conservative filter: must contain at least one uppercase letter
    followed by a digit or lowercase letter to avoid grabbing plain words.
    """
    MIN_FORMULA_LEN = 2
    candidates = set()

    # Explicit equation formulas
    for eq in _EQUATION.findall(text):
        parts = re.split(r'[→⇌\->+]', eq)
        for p in parts:
            p = p.strip().split()[0] if p.strip().split() else ''
            if p and re.search(r'[A-Z][a-z0-9]', p):
                candidates.add(p.strip('(),. '))

    # Inline formulas in backtick-style lines
    for line in text.splitlines():
        stripped = line.strip()
        # Lines that look like pure formula lines (no spaces, capital letter start)
        if re.match(r'^[A-Z][A-Za-z0-9()·.^+\-=₂₃₄⁺⁻]+$', stripped):
            if len(stripped) >= MIN_FORMULA_LEN:
                candidates.add(stripped)

    return sorted(candidates)


def _extract_equations(text: str) -> List[str]:
    return [m.strip() for m in _EQUATION.findall(text)]


def _extract_diagrams(text: str) -> List[dict]:
    results = []
    for m in _DIAGRAM.finditer(text):
        alt = m.group(1).strip()
        path = m.group(2).strip()
        results.append({"alt": alt, "path": path})
    return results


def _is_worked_example(text: str) -> bool:
    """
    True if this chunk contains a worked example block.
    Worked examples have Diberi/Formula/Pengiraan/Jawapan structure.
    """
    markers = ['diberi', 'formula', 'pengiraan', 'jawapan', 'langkah', 'penyelesaian']
    text_lower = text.lower()
    count = sum(1 for m in markers if f'## {m}' in text_lower or f'# {m}' in text_lower or f'\n{m}:' in text_lower)
    return count >= 2


def _is_calculation_file(text: str) -> bool:
    return bool(_CALC_HEADER.search(text))


# ---------------------------------------------------------------------------
# WORKED-EXAMPLE ATOMIC BLOCK DETECTION
# ---------------------------------------------------------------------------

def _split_preserving_worked_examples(text: str) -> List[str]:
    """
    Split text at --- boundaries, BUT never split a worked-example block
    (Soalan/Diberi/Formula/Pengiraan/Jawapan).
    Returns list of text blocks that should not be further split.
    """
    # Split on --- separators first
    raw_sections = _SEPARATOR.split(text)

    if not raw_sections:
        return [text]

    result_blocks: List[str] = []
    pending = ""

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        section_lower = section.lower()

        # If this section starts or continues a worked example, accumulate it
        is_soalan = bool(re.search(r'^##? soalan', section_lower, re.MULTILINE))
        is_contoh = bool(re.search(r'^##? contoh', section_lower, re.MULTILINE))
        is_diberi = bool(re.search(r'^diberi', section_lower, re.MULTILINE))
        is_formula = bool(re.search(r'^formula', section_lower, re.MULTILINE))
        is_pengiraan = bool(re.search(r'^pengiraan', section_lower, re.MULTILINE))
        is_jawapan = bool(re.search(r'^jawapan', section_lower, re.MULTILINE))
        is_langkah = bool(re.search(r'^##? langkah', section_lower, re.MULTILINE))

        is_example_part = any([
            is_soalan, is_contoh, is_diberi, is_formula,
            is_pengiraan, is_jawapan, is_langkah
        ])

        if is_example_part:
            pending += "\n\n---\n\n" + section
        else:
            if pending:
                result_blocks.append(pending.strip())
                pending = ""
            result_blocks.append(section)

    if pending:
        result_blocks.append(pending.strip())

    return [b for b in result_blocks if b.strip()]


# ---------------------------------------------------------------------------
# CORE CHUNKER
# ---------------------------------------------------------------------------

class MarkdownChunker:
    """
    Chunks Markdown files using heading hierarchy with worked-example preservation.

    Strategy:
    1. Split on H1 headings → top-level sections
    2. Within each H1, split on H2 → sub-sections
    3. Within each H2, split on H3 OR --- separators → chunks
    4. Never split a worked-example block
    5. Merge very short chunks (< min_chars) with adjacent chunk
    6. Extract metadata from each chunk

    Parameters
    ----------
    min_chars : int
        Minimum characters per chunk. Smaller chunks are merged upward.
    max_chars : int
        Target maximum characters per chunk (not a hard limit — worked
        examples are never split even if they exceed this).
    """

    def __init__(self, min_chars: int = 150, max_chars: int = 1800):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk_file(self, filepath: str | Path) -> List[RawChunk]:
        filepath = Path(filepath)
        text = filepath.read_text(encoding='utf-8')
        source = filepath.name
        is_calc = _is_calculation_file(text)
        return self._chunk_text(text, source, is_calc)

    def chunk_text(self, text: str, source: str = "inline") -> List[RawChunk]:
        is_calc = _is_calculation_file(text)
        return self._chunk_text(text, source, is_calc)

    def _chunk_text(self, text: str, source: str, is_calc_file: bool) -> List[RawChunk]:
        chunks: List[RawChunk] = []
        idx = 0

        # ── Step 1: Split on H1 ──────────────────────────────────────────
        h1_sections = self._split_by_heading(text, level=1)

        for h1_text, h1_title in h1_sections:
            # ── Step 2: Split on H2 ──────────────────────────────────────
            h2_sections = self._split_by_heading(h1_text, level=2)

            for h2_text, h2_title in h2_sections:
                # ── Step 3: Split on H3 or separators ────────────────────
                h3_sections = self._split_by_heading(h2_text, level=3)

                for h3_text, h3_title in h3_sections:
                    # ── Step 4: Preserve worked examples ─────────────────
                    atomic_blocks = _split_preserving_worked_examples(h3_text)

                    pending_short = ""
                    for block in atomic_blocks:
                        block = block.strip()
                        if not block:
                            continue

                        # Merge very short blocks with next
                        if len(block) < self.min_chars and not _is_worked_example(block):
                            pending_short += "\n\n" + block
                            continue

                        if pending_short:
                            block = pending_short.strip() + "\n\n" + block
                            pending_short = ""

                        chunk = self._make_chunk(
                            block, source, h1_title, h2_title, h3_title, idx
                        )
                        if chunk:
                            chunks.append(chunk)
                            idx += 1

                    # flush leftover short content
                    if pending_short.strip():
                        chunk = self._make_chunk(
                            pending_short.strip(), source,
                            h1_title, h2_title, h3_title, idx
                        )
                        if chunk:
                            chunks.append(chunk)
                            idx += 1

        return chunks

    def _split_by_heading(self, text: str, level: int) -> List[tuple]:
        """
        Split text at headings of the given level.
        Returns list of (section_text, heading_title).
        """
        pattern = re.compile(r'^#{' + str(level) + r'} (.+)$', re.MULTILINE)
        matches = list(pattern.finditer(text))

        if not matches:
            return [(text, "")]

        sections = []
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            sections.append((section_text, title))

        # Prepend any content before the first heading
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.insert(0, (pre, ""))

        return sections

    def _make_chunk(
        self,
        content: str,
        source: str,
        h1: str,
        h2: str,
        h3: str,
        idx: int,
    ) -> Optional[RawChunk]:
        content = content.strip()
        if len(content) < 20:
            return None

        chunk_id = _generate_id(source, h2 or h1, idx)
        keywords = _extract_keywords(content)
        formulas = _extract_formulas(content)
        diagrams = _extract_diagrams(content)
        equations = _extract_equations(content)
        has_worked = _is_worked_example(content)
        has_table = bool(_TABLE.search(content))

        return RawChunk(
            chunk_id=chunk_id,
            source_file=source,
            heading_h1=h1,
            heading_h2=h2,
            heading_h3=h3,
            content=content,
            raw_keywords=keywords,
            formulas=formulas,
            diagrams=diagrams,
            equations=equations,
            has_worked_example=has_worked,
            has_table=has_table,
            char_count=len(content),
        )


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def chunk_all_files(
    knowledge_base_dir: str | Path,
    extensions: tuple = ('.md',),
    min_chars: int = 150,
    max_chars: int = 1800,
) -> List[RawChunk]:
    """
    Chunk all Markdown files in a directory (recursive).

    Parameters
    ----------
    knowledge_base_dir : path to knowledge_base/ folder
    extensions : file extensions to process
    min_chars, max_chars : chunk size limits

    Returns
    -------
    List of RawChunk objects ready for metadata tagging
    """
    kb_path = Path(knowledge_base_dir)
    chunker = MarkdownChunker(min_chars=min_chars, max_chars=max_chars)
    all_chunks: List[RawChunk] = []

    md_files = sorted(kb_path.rglob('*'))
    md_files = [f for f in md_files if f.suffix in extensions and f.is_file()]

    print(f"[chunker] Found {len(md_files)} files in {kb_path}")

    for fp in md_files:
        try:
            file_chunks = chunker.chunk_file(fp)
            print(f"[chunker]   {fp.name:55s} → {len(file_chunks):3d} chunks")
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"[chunker] ERROR processing {fp}: {e}")

    print(f"[chunker] Total chunks: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI TESTING
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_md_file_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        chunks = chunk_all_files(target)
    else:
        chunker = MarkdownChunker()
        chunks = chunker.chunk_file(target)

    for c in chunks[:5]:
        print(f"\n{'='*60}")
        print(f"ID      : {c.chunk_id}")
        print(f"Heading : {c.full_heading}")
        print(f"Chars   : {c.char_count}")
        print(f"Keywords: {c.raw_keywords[:5]}")
        print(f"Formulas: {c.formulas[:5]}")
        print(f"Diagrams: {c.diagrams}")
        print(f"Worked  : {c.has_worked_example}")
        print(f"Content preview:\n{c.content[:200]}...")
