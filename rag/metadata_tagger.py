"""
metadata_tagger.py — Cikgu AI Kimia RAG Pipeline
==================================================
Converts RawChunk → ChemistryChunk with full metadata:
  - chapter, tingkatan, topic, subtopic
  - content_type (theory / calculation / formula / qa_scheme / definition)
  - formulas, equations, diagrams
  - language (BM / EN / mixed)
  - SPM keywords + English synonyms
  - exam_year, question_type (if applicable)
  - bilingual keyword augmentation

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from chunker import RawChunk


# ---------------------------------------------------------------------------
# DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass
class ChemistryChunk:
    """A fully tagged chunk ready for FAISS indexing."""

    # Identity
    chunk_id: str
    source_file: str

    # Headings
    heading_h1: str
    heading_h2: str
    heading_h3: str
    full_heading: str

    # Content
    content: str
    char_count: int

    # SPM Classification
    chapter: Optional[int]
    tingkatan: Optional[int]          # 4 or 5
    topic: str
    subtopic: str

    # Content type — controls which FAISS index receives this chunk
    content_type: str                 # theory | calculation | formula | qa_scheme | definition

    # Chemistry metadata
    formulas: List[str]
    equations: List[str]
    keywords_bm: List[str]
    keywords_en: List[str]
    diagrams: List[dict]              # [{"alt": str, "path": str}]
    has_worked_example: bool
    has_table: bool
    has_diagram: bool

    # Language
    language: str                     # BM | EN | mixed

    # Exam metadata
    exam_year: Optional[int]
    question_type: Optional[str]      # mcq | struktur | esei | None

    # Retrieval text (what gets embedded)
    embed_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "heading_h1": self.heading_h1,
            "heading_h2": self.heading_h2,
            "heading_h3": self.heading_h3,
            "full_heading": self.full_heading,
            "content": self.content,
            "char_count": self.char_count,
            "chapter": self.chapter,
            "tingkatan": self.tingkatan,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "content_type": self.content_type,
            "formulas": self.formulas,
            "equations": self.equations,
            "keywords_bm": self.keywords_bm,
            "keywords_en": self.keywords_en,
            "diagrams": self.diagrams,
            "has_worked_example": self.has_worked_example,
            "has_table": self.has_table,
            "has_diagram": self.has_diagram,
            "language": self.language,
            "exam_year": self.exam_year,
            "question_type": self.question_type,
            "embed_text": self.embed_text,
        }


# ---------------------------------------------------------------------------
# CHAPTER / TINGKATAN DETECTION
# ---------------------------------------------------------------------------

# Map filename patterns to (chapter, tingkatan, topic)
_FILE_MAP: Dict[str, tuple] = {
    "BAB_1_PENGENALAN_KEPADA_KIMIA":    (1, 4, "Pengenalan Kepada Kimia"),
    "BAB_2_JIRIM_DAN_STRUKTUR_ATOM":    (2, 4, "Jirim dan Struktur Atom"),
    "BAB__3_KONSEP_MOL":                (3, 4, "Konsep Mol, Formula & Persamaan Kimia"),
    "BAB_3_KONSEP_MOL":                 (3, 4, "Konsep Mol, Formula & Persamaan Kimia"),
    "BAB_4_JADUAL_BERKALA_UNSUR":       (4, 4, "Jadual Berkala Unsur"),
    "BAB_5_IKATAN_KIMIA":               (5, 4, "Ikatan Kimia"),
    "BAB_1_KESEIMBANGAN_REDOKS":        (1, 5, "Keseimbangan dan Redoks"),
    "BAB_2_SEBATIAN_KARBON":            (2, 5, "Sebatian Karbon"),
    "BAB_3_TERMOKIMIA":                 (3, 5, "Termokimia"),
    "BAB_4_POLIMER":                    (4, 5, "Polimer"),
    "BAB_5_KIMIA_KONSUMER":             (5, 5, "Kimia Konsumer dan Industri"),
    "BAB_6_ASID_BES_GARAM":             (6, 5, "Asid, Bes & Garam"),
    "BAB_7_KADAR_TINDAK_BALAS":         (7, 5, "Kadar Tindak Balas"),
    "BAB_8_BAHAN_BUATAN":               (8, 5, "Bahan Buatan dalam Industri"),
    # Calculation files
    "mol_calculations":                 (3, 4, "Konsep Mol"),
    "acid_calculations":                (6, 5, "Asid Bes Garam"),
    "rate_calculations":                (7, 5, "Kadar Tindak Balas"),
    "termokimia_calculations":          (3, 5, "Termokimia"),
    "redox_calculations":               (1, 5, "Redoks"),
    "jisim_atom_relatif":               (2, 4, "Jirim dan Struktur Atom"),
    "master_calculation_reference":     (0, 0, "Rujukan Formula"),
}


def _detect_chapter_info(source_file: str) -> tuple:
    """Returns (chapter, tingkatan, topic) from filename."""
    stem = Path(source_file).stem.upper()
    for key, val in _FILE_MAP.items():
        if key.upper() in stem:
            return val
    # Fallback: try to extract chapter number from filename
    m = re.search(r'BAB_?(\d+)', stem)
    if m:
        ch = int(m.group(1))
        tingkatan = 4 if ch <= 5 else 5
        return (ch, tingkatan, "")
    return (None, None, "")


# ---------------------------------------------------------------------------
# CONTENT TYPE DETECTION
# ---------------------------------------------------------------------------

_CALC_MARKERS = re.compile(
    r'(?:pengiraan|hitungkan|kira|calculate|find|nilai|value'
    r'|diberi|formula|langkah|jawapan|Q\s*=|n\s*=|m\s*=|pH\s*=|ΔH)',
    re.IGNORECASE
)

_FORMULA_ONLY_MARKERS = re.compile(
    r'^(?:[A-Z][a-z]?[\d()]+|n\s*=|m\s*=|Q\s*=|pH\s*=|ΔH)',
    re.MULTILINE
)

_QA_MARKERS = re.compile(
    r'(?:soalan|contoh soalan|jawapan skema|marking scheme|'
    r'answer scheme|topikal|SPM \d{4})',
    re.IGNORECASE
)

_DEFINITION_MARKERS = re.compile(
    r'(?:ditakrifkan sebagai|adalah|ialah|defined as|refers to|merujuk kepada)',
    re.IGNORECASE
)


def _detect_content_type(chunk: RawChunk, source_file: str) -> str:
    stem = Path(source_file).stem.lower()

    # Calculation files are always calculation type
    if 'calculation' in stem or 'pengiraan' in stem:
        return "calculation"

    if 'master' in stem or 'reference' in stem or 'jisim_atom' in stem:
        return "formula"

    text = chunk.content
    has_calc = bool(_CALC_MARKERS.search(text))
    has_worked = chunk.has_worked_example
    has_qa = bool(_QA_MARKERS.search(text))
    has_def = bool(_DEFINITION_MARKERS.search(text))
    has_formula_lines = bool(_FORMULA_ONLY_MARKERS.search(text))

    if has_qa and has_worked:
        return "qa_scheme"
    if has_worked or (has_calc and len(chunk.equations) > 0):
        return "calculation"
    if has_formula_lines and len(chunk.formulas) >= 3 and not has_def:
        return "formula"
    if has_def and not has_worked:
        return "definition"
    return "theory"


# ---------------------------------------------------------------------------
# LANGUAGE DETECTION
# ---------------------------------------------------------------------------

_BM_MARKERS = re.compile(
    r'\b(?:ialah|adalah|daripada|kepada|dengan|untuk|yang|ini|dalam|'
    r'bilangan|jisim|isipadu|tindak balas|kimia|pengiraan|hitungkan|'
    r'tentukan|hitung|langkah|jawapan|diberi)\b',
    re.IGNORECASE
)

_EN_MARKERS = re.compile(
    r'\b(?:calculate|find|determine|given|solution|answer|formula|'
    r'reaction|volume|mass|number|step|molar|concentration)\b',
    re.IGNORECASE
)


def _detect_language(text: str) -> str:
    bm = len(_BM_MARKERS.findall(text))
    en = len(_EN_MARKERS.findall(text))
    if bm > en * 2:
        return "BM"
    if en > bm * 2:
        return "EN"
    return "mixed"


# ---------------------------------------------------------------------------
# BILINGUAL KEYWORD AUGMENTATION
# ---------------------------------------------------------------------------

# BM → EN synonym map for SPM Chemistry terms
_BM_EN_SYNONYMS: Dict[str, str] = {
    "mol": "mole",
    "jisim": "mass",
    "jisim molar": "molar mass",
    "jisim atom relatif": "relative atomic mass",
    "jisim molekul relatif": "relative molecular mass",
    "isipadu": "volume",
    "isi padu": "volume",
    "bilangan zarah": "number of particles",
    "bilangan mol": "number of moles",
    "nombor avogadro": "Avogadro's number",
    "kemolaran": "molarity",
    "kepekatan": "concentration",
    "asid": "acid",
    "bes": "base",
    "alkali": "alkali",
    "garam": "salt",
    "neutralisasi": "neutralisation",
    "pentitratan": "titration",
    "pengoksidaan": "oxidation",
    "penurunan": "reduction",
    "nombor pengoksidaan": "oxidation number",
    "kadar tindak balas": "rate of reaction",
    "tenaga pengaktifan": "activation energy",
    "mangkin": "catalyst",
    "haba": "heat",
    "entalpi": "enthalpy",
    "eksotermik": "exothermic",
    "endotermik": "endothermic",
    "termokimia": "thermochemistry",
    "polimer": "polymer",
    "monomer": "monomer",
    "pempolimeran": "polymerisation",
    "ikatan kimia": "chemical bond",
    "ikatan ion": "ionic bond",
    "ikatan kovalen": "covalent bond",
    "ikatan hidrogen": "hydrogen bond",
    "elektron valens": "valence electron",
    "susunan elektron": "electron configuration",
    "jadual berkala": "periodic table",
    "kumpulan": "group",
    "kala": "period",
    "unsur peralihan": "transition element",
    "gas adi": "noble gas",
    "formula empirik": "empirical formula",
    "formula molekul": "molecular formula",
    "persamaan kimia": "chemical equation",
    "stoikiometri": "stoichiometry",
    "alkana": "alkane",
    "alkena": "alkene",
    "alkuna": "alkyne",
    "alkohol": "alcohol",
    "asid karboksilik": "carboxylic acid",
    "ester": "ester",
    "getah": "rubber",
    "pemvulkanan": "vulcanisation",
    "pH": "pH",
    "pOH": "pOH",
    "ion hidrogen": "hydrogen ion",
    "ion hidroksida": "hydroxide ion",
    "isotop": "isotope",
    "nombor proton": "proton number",
    "nombor nukleon": "nucleon number",
    "elektrod": "electrode",
    "sel elektrolisis": "electrolysis cell",
    "sel galvani": "galvanic cell",
}


def _get_english_synonyms(bm_keywords: List[str]) -> List[str]:
    """Return English equivalents for BM keywords."""
    en_kws = []
    for kw in bm_keywords:
        kw_lower = kw.lower().strip()
        if kw_lower in _BM_EN_SYNONYMS:
            en_kws.append(_BM_EN_SYNONYMS[kw_lower])
        # Also do substring matching for compound terms
        for bm, en in _BM_EN_SYNONYMS.items():
            if bm in kw_lower and en not in en_kws:
                en_kws.append(en)
    return list(dict.fromkeys(en_kws))


# ---------------------------------------------------------------------------
# EMBED TEXT CONSTRUCTION
# ---------------------------------------------------------------------------

def _build_embed_text(chunk: RawChunk, meta: ChemistryChunk) -> str:
    """
    Construct the text string that will be embedded into FAISS.
    Combines heading context + keywords (BM + EN) + content.
    This maximises retrieval accuracy for both BM and EN queries.
    """
    parts = []

    # Heading context
    if meta.full_heading:
        parts.append(f"Tajuk: {meta.full_heading}")

    # Topic
    if meta.topic:
        parts.append(f"Topik: {meta.topic}")

    # All keywords (BM + EN)
    all_kws = meta.keywords_bm + meta.keywords_en
    if all_kws:
        parts.append(f"Kata kunci: {', '.join(all_kws[:20])}")

    # Formulas
    if meta.formulas:
        parts.append(f"Formula: {', '.join(meta.formulas[:10])}")

    # Main content
    parts.append(chunk.content)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# SUBTOPIC EXTRACTION
# ---------------------------------------------------------------------------

def _extract_subtopic(h2: str, h3: str) -> str:
    """Best available subtopic label."""
    if h3 and h3 not in ('', 'Konsep SPM'):
        return h3
    if h2 and h2:
        # Remove heading number prefix like "6.1 " etc.
        return re.sub(r'^\d+\.\d+\s*', '', h2).strip()
    return ""


# ---------------------------------------------------------------------------
# CORE TAGGER
# ---------------------------------------------------------------------------

class MetadataTagger:
    """
    Tags RawChunk objects with full ChemistryChunk metadata.
    """

    def tag(self, chunk: RawChunk) -> ChemistryChunk:
        chapter, tingkatan, topic = _detect_chapter_info(chunk.source_file)
        content_type = _detect_content_type(chunk, chunk.source_file)
        language = _detect_language(chunk.content)
        keywords_bm = chunk.raw_keywords
        keywords_en = _get_english_synonyms(keywords_bm)
        subtopic = _extract_subtopic(chunk.heading_h2, chunk.heading_h3)

        # If topic empty, derive from H1
        if not topic and chunk.heading_h1:
            topic = re.sub(r'^BAB \d+:\s*', '', chunk.heading_h1).strip()

        # Override topic from heading if it looks like a sub-section title
        if chunk.heading_h2 and not topic:
            topic = chunk.heading_h2

        # Exam year detection in content
        exam_year_m = re.search(r'\bSPM[_ ]?(\d{4})\b', chunk.content)
        exam_year = int(exam_year_m.group(1)) if exam_year_m else None

        # Question type detection
        question_type = None
        qtype_text = chunk.content.lower()
        if 'pilihan jawapan' in qtype_text or 'mcq' in qtype_text:
            question_type = 'mcq'
        elif 'esei' in qtype_text or 'essay' in qtype_text:
            question_type = 'esei'
        elif 'soalan struktur' in qtype_text or 'structured' in qtype_text:
            question_type = 'struktur'

        # Build preliminary meta object (embed_text computed after)
        meta = ChemistryChunk(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            heading_h1=chunk.heading_h1,
            heading_h2=chunk.heading_h2,
            heading_h3=chunk.heading_h3,
            full_heading=chunk.full_heading,
            content=chunk.content,
            char_count=chunk.char_count,
            chapter=chapter,
            tingkatan=tingkatan,
            topic=topic,
            subtopic=subtopic,
            content_type=content_type,
            formulas=chunk.formulas,
            equations=chunk.equations,
            keywords_bm=keywords_bm,
            keywords_en=keywords_en,
            diagrams=chunk.diagrams,
            has_worked_example=chunk.has_worked_example,
            has_table=chunk.has_table,
            has_diagram=len(chunk.diagrams) > 0,
            language=language,
            exam_year=exam_year,
            question_type=question_type,
            embed_text="",  # placeholder
        )

        # Now build embed_text with full context
        meta.embed_text = _build_embed_text(chunk, meta)

        return meta

    def tag_all(self, chunks: List[RawChunk]) -> List[ChemistryChunk]:
        tagged = []
        for chunk in chunks:
            try:
                tagged.append(self.tag(chunk))
            except Exception as e:
                print(f"[tagger] ERROR tagging {chunk.chunk_id}: {e}")
        return tagged


# ---------------------------------------------------------------------------
# CONVENIENCE
# ---------------------------------------------------------------------------

def tag_chunks(raw_chunks: List[RawChunk]) -> List[ChemistryChunk]:
    tagger = MetadataTagger()
    return tagger.tag_all(raw_chunks)


def print_chunk_summary(chunks: List[ChemistryChunk]) -> None:
    """Print a summary table of tagged chunks."""
    from collections import Counter
    type_counts = Counter(c.content_type for c in chunks)
    lang_counts = Counter(c.language for c in chunks)
    chap_counts = Counter(c.chapter for c in chunks)

    print(f"\n{'='*60}")
    print(f"TAGGED CHUNKS SUMMARY: {len(chunks)} total")
    print(f"{'='*60}")
    print("Content types:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t:20s} {n:4d}")
    print("Languages:")
    for l, n in sorted(lang_counts.items()):
        print(f"  {l:20s} {n:4d}")
    print("Chapters (None = cross-chapter):")
    for ch, n in sorted(chap_counts.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"  Chapter {str(ch):6s} {n:4d}")


# ---------------------------------------------------------------------------
# CLI TESTING
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from chunker import chunk_all_files

    if len(sys.argv) < 2:
        print("Usage: python metadata_tagger.py <knowledge_base_dir>")
        sys.exit(1)

    raw_chunks = chunk_all_files(sys.argv[1])
    tagged = tag_chunks(raw_chunks)
    print_chunk_summary(tagged)

    # Show 3 examples
    for tc in tagged[:3]:
        print(f"\n{'─'*60}")
        print(f"ID         : {tc.chunk_id}")
        print(f"Type       : {tc.content_type}")
        print(f"Chapter    : {tc.chapter}  Tingkatan: {tc.tingkatan}")
        print(f"Topic      : {tc.topic}")
        print(f"Subtopic   : {tc.subtopic}")
        print(f"Keywords BM: {tc.keywords_bm[:5]}")
        print(f"Keywords EN: {tc.keywords_en[:5]}")
        print(f"Formulas   : {tc.formulas[:3]}")
        print(f"Diagrams   : {tc.diagrams}")
        print(f"Worked eg  : {tc.has_worked_example}")
        print(f"Embed text preview:\n{tc.embed_text[:300]}")
