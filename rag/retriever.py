"""
retriever.py — Cikgu AI Kimia RAG Pipeline
===========================================
Production retrieval engine with:
  - Multi-index search routing (theory / calculation / qa / all)
  - Metadata pre-filtering (chapter, tingkatan, content_type)
  - Score thresholding (anti-hallucination)
  - Query augmentation (BM + EN synonym expansion)
  - Deduplicated results
  - SPM-formatted context building for LLM prompt injection
  - Retrieval diagnostics

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from indexer import FAISSIndexManager, get_index_manager
from embedder import ChemistryEmbedder, get_embedder


# ---------------------------------------------------------------------------
# RESULT DATA CLASS
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    rank: int
    score: float
    chunk_id: str
    source_file: str
    content_type: str
    chapter: Optional[int]
    tingkatan: Optional[int]
    topic: str
    subtopic: str
    content: str
    formulas: List[str]
    equations: List[str]
    keywords_bm: List[str]
    diagrams: List[dict]
    has_worked_example: bool
    index_name: str

    @property
    def context_block(self) -> str:
        """Formatted content block for LLM prompt injection."""
        lines = []
        if self.topic:
            lines.append(f"[Topik: {self.topic}]")
        if self.subtopic:
            lines.append(f"[Subtopik: {self.subtopic}]")
        if self.diagrams:
            diag_list = ', '.join(d.get('alt', d.get('path', '')) for d in self.diagrams)
            lines.append(f"[Rajah: {diag_list}]")
        lines.append(self.content)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QUERY AUGMENTATION
# ---------------------------------------------------------------------------

# Expand short BM queries with EN equivalents for better recall
_QUERY_AUGMENTATION_MAP = {
    "mol": "mol mole bilangan mol",
    "jisim": "jisim mass jisim molar molar mass",
    "pH": "pH keasidan acid hydrogen ion",
    "pOH": "pOH alkalinity hydroxide ion",
    "kadar": "kadar tindak balas rate of reaction laju",
    "haba": "haba heat thermochemistry termokimia",
    "entalpi": "entalpi enthalpy ΔH perubahan haba",
    "redoks": "redoks redox pengoksidaan penurunan oxidation reduction",
    "titration": "pentitratan titration peneutralan",
    "polimer": "polimer polymer monomer pempolimeran polymerisation",
    "ikatan": "ikatan kimia chemical bond ion kovalen ionic covalent",
    "isotop": "isotop isotope jisim atom relatif",
}


def augment_query(query: str) -> str:
    """
    Expand query with synonyms to improve multilingual recall.
    Only augments if keywords are found; doesn't bloat clean queries.
    """
    q_lower = query.lower()
    additions = []
    for key, expansion in _QUERY_AUGMENTATION_MAP.items():
        if key in q_lower and expansion not in q_lower:
            # Add only the expansion words not already in query
            for word in expansion.split():
                if word.lower() not in q_lower:
                    additions.append(word)

    if additions:
        return query + " " + " ".join(additions[:10])
    return query


# ---------------------------------------------------------------------------
# INDEX ROUTING
# ---------------------------------------------------------------------------

def route_to_indexes(
    query: str,
    force_index: Optional[str] = None,
) -> List[str]:
    """
    Determine which FAISS indexes to search based on query content.

    Rules:
    - Calculation keywords → search calculation index first
    - Definition/concept keywords → theory index
    - SPM exam keywords → qa index
    - Unknown / broad → all indexes
    """
    if force_index:
        return [force_index]

    q = query.lower()

    # Calculation signals
    calc_signals = [
        'hitung', 'kira', 'calculate', 'find', 'nilai', 'berapa',
        'mol ', 'mole', 'jisim', 'mass', 'isipadu', 'volume',
        'ph =', 'poh', 'delta h', 'ΔH', 'q =', 'kalorimetri',
        'titration', 'pentitratan', 'stoikiometri', 'nisbah mol',
        'nombor pengoksidaan', 'oxidation number', 'kadar tindak balas',
        'rate =', 'pencairan', 'dilution', 'kemolaran',
    ]

    # Theory / definition signals
    theory_signals = [
        'terangkan', 'explain', 'takrifkan', 'define', 'apakah',
        'what is', 'jelaskan', 'describe', 'bandingkan', 'compare',
        'bezakan', 'perbezaan', 'difference', 'mengapa', 'why',
        'bagaimana', 'how does', 'senaraikan', 'list',
        'kesan', 'effect', 'faktor', 'factor', 'ciri', 'property',
    ]

    # QA signals
    qa_signals = [
        'soalan', 'question', 'jawab', 'answer', 'spm', 'exam',
        'peperiksaan', 'skema', 'scheme', 'marking',
    ]

    has_calc = any(s in q for s in calc_signals)
    has_theory = any(s in q for s in theory_signals)
    has_qa = any(s in q for s in qa_signals)

    # Theory questions with "terangkan", "jelaskan", "apakah" etc
    # always search theory index FIRST regardless of topic
    if has_theory:
        return ["index_theory", "index_qa"]

    if has_calc and not has_theory and not has_qa:
        return ["index_calculations", "index_theory"]
    if has_qa and not has_calc:
        return ["index_qa", "index_theory"]

    # Default: theory first, then calculations
    return ["index_theory", "index_calculations", "index_qa"]


# ---------------------------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------------------------

class ChemistryRetriever:
    """
    Production retriever for Cikgu AI Kimia.

    Usage:
        retriever = ChemistryRetriever(index_dir="faiss_indexes")
        results = retriever.retrieve("terangkan tindak balas eksotermik", k=5)
        context = retriever.build_context(results)
    """

    def __init__(
        self,
        index_dir: str = "faiss_indexes",
        embedder: Optional[ChemistryEmbedder] = None,
        score_threshold: float = 0.30,
        augment_queries: bool = True,
    ):
        self.index_dir = index_dir
        self.embedder = embedder or get_embedder()
        self.score_threshold = score_threshold
        self.augment_queries = augment_queries
        self._manager: Optional[FAISSIndexManager] = None

    @property
    def manager(self) -> FAISSIndexManager:
        if self._manager is None:
            self._manager = FAISSIndexManager(
                index_dir=self.index_dir,
                embed_dim=384,
            )
        return self._manager

    def retrieve(
        self,
        query: str,
        k: int = 5,
        index_names: Optional[List[str]] = None,
        chapter_filter: Optional[int] = None,
        tingkatan_filter: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query : student's question
        k : number of results per index
        index_names : override auto-routing
        chapter_filter : filter by chapter number
        tingkatan_filter : filter by form (4 or 5)
        content_type_filter : filter by content type
        score_threshold : override default threshold

        Returns
        -------
        List of RetrievalResult, sorted by relevance
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Query augmentation
        search_query = augment_query(query) if self.augment_queries else query

        # Route to indexes
        target_indexes = index_names or route_to_indexes(query)

        # Embed query
        query_vec = self.embedder.embed_query(search_query)

        # Build metadata filter
        meta_filter: Dict[str, Any] = {}
        if chapter_filter is not None:
            meta_filter["chapter"] = chapter_filter
        if tingkatan_filter is not None:
            meta_filter["tingkatan"] = tingkatan_filter
        if content_type_filter is not None:
            meta_filter["content_type"] = content_type_filter

        # Search
        raw_results = self.manager.search(
            query_vec=query_vec,
            index_names=target_indexes,
            k=k,
            score_threshold=threshold,
            metadata_filter=meta_filter if meta_filter else None,
        )

        # Deduplicate by content (same chunk may appear in multiple searches)
        seen_ids = set()
        deduped = []
        for r in raw_results:
            cid = r["metadata"].get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append(r)

        # Convert to RetrievalResult
        results = []
        for rank, r in enumerate(deduped[:k], start=1):
            meta = r["metadata"]
            results.append(RetrievalResult(
                rank=rank,
                score=r["score"],
                chunk_id=meta.get("chunk_id", ""),
                source_file=meta.get("source_file", ""),
                content_type=meta.get("content_type", ""),
                chapter=meta.get("chapter"),
                tingkatan=meta.get("tingkatan"),
                topic=meta.get("topic", ""),
                subtopic=meta.get("subtopic", ""),
                content=meta.get("content", ""),
                formulas=meta.get("formulas", []),
                equations=meta.get("equations", []),
                keywords_bm=meta.get("keywords_bm", []),
                diagrams=meta.get("diagrams", []),
                has_worked_example=meta.get("has_worked_example", False),
                index_name=r["index"],
            ))

        return results

    def retrieve_for_calculation(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """Optimised retrieval for calculation-type queries (fewer, denser chunks)."""
        return self.retrieve(
            query=query,
            k=k,
            index_names=["index_calculations"],
            content_type_filter=None,
        )

    def retrieve_for_theory(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Optimised retrieval for theory/concept queries."""
        return self.retrieve(
            query=query,
            k=k,
            index_names=["index_theory"],
        )

    def build_context(
        self,
        results: List[RetrievalResult],
        max_chars: int = 3000,
        include_metadata_header: bool = True,
    ) -> str:
        """
        Build a LLM-ready context string from retrieval results.

        Parameters
        ----------
        results : list of RetrievalResult
        max_chars : total character budget for context
        include_metadata_header : prepend topic/source info

        Returns
        -------
        Formatted context string for injection into LLM prompt
        """
        if not results:
            return ""

        context_parts = []
        total_chars = 0

        for i, r in enumerate(results, start=1):
            block = r.context_block
            if total_chars + len(block) > max_chars:
                break
            context_parts.append(f"--- Petikan {i} (Relevansi: {r.score:.2f}) ---\n{block}")
            total_chars += len(block)

        return "\n\n".join(context_parts)

    def is_sufficient_context(
        self,
        results: List[RetrievalResult],
        min_score: float = 0.45,
        min_results: int = 1,
    ) -> bool:
        """
        Returns True if retrieval results are good enough to answer from.
        Used to decide: answer from RAG or say "soalan tidak dijumpai".
        """
        if len(results) < min_results:
            return False
        return any(r.score >= min_score for r in results)

    def diagnose(self, query: str, k: int = 10) -> None:
        """Print diagnostic information about retrieval for a query."""
        print(f"\n{'='*70}")
        print(f"RETRIEVAL DIAGNOSTICS")
        print(f"Query: {query}")
        print(f"Augmented: {augment_query(query)}")
        print(f"Routed to: {route_to_indexes(query)}")
        print(f"{'='*70}")

        results = self.retrieve(query, k=k, score_threshold=0.0)
        if not results:
            print("  No results found.")
            return

        for r in results:
            diag_flag = "✓" if r.score >= self.score_threshold else "✗"
            print(f"  {diag_flag} [{r.score:.4f}] {r.content_type:15s} "
                  f"Ch{r.chapter or '?'} | {r.topic[:30]:30s} | {r.subtopic[:25]:25s}")
            if r.formulas:
                print(f"       Formulas: {r.formulas[:3]}")
            print(f"       Preview: {r.content[:100].replace(chr(10), ' ')}")


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """Kamu adalah Cikgu AI Kimia, tutor kimia SPM yang pakar.

PERATURAN PENTING:
1. Jawab HANYA berdasarkan petikan nota yang diberikan di bawah.
2. Jika soalan adalah TEORI atau KONSEP: jawab dalam bentuk ayat yang jelas dan lengkap. JANGAN guna format Diberi/Formula/Pengiraan.
3. Jika soalan memerlukan PENGIRAAN sahaja: gunakan format SPM:
   Diberi:
   Formula:
   Pengiraan:
   Jawapan:
4. Kekalkan terminologi Bahasa Malaysia SPM.
5. Sertakan contoh yang relevan daripada nota.
6. Jika nota mengandungi maklumat SEPARA, gunakan apa yang ada dan tambah penjelasan ringkas.
7. JANGAN reka fakta atau formula yang tidak ada dalam nota.
8. Jawapan mestilah lengkap - jangan tinggalkan bahagian penting.

NOTA RUJUKAN:
{context}

SOALAN PELAJAR:
{question}
"""


def build_rag_prompt(
    question: str,
    results: List[RetrievalResult],
    max_context_chars: int = 3000,
) -> str:
    """
    Build a complete RAG prompt for the LLM.

    Parameters
    ----------
    question : student's original question
    results : retrieval results
    max_context_chars : context budget

    Returns
    -------
    Formatted prompt string
    """
    if not results:
        context = "Tiada nota relevan dijumpai untuk soalan ini."
    else:
        context = ""
        chars_used = 0
        for i, r in enumerate(results, 1):
            block = r.context_block
            if chars_used + len(block) > max_context_chars:
                break
            context += f"\n--- Petikan {i} (Skor: {r.score:.2f}) ---\n{block}\n"
            chars_used += len(block)

    return SYSTEM_PROMPT_TEMPLATE.format(
        context=context.strip(),
        question=question,
    )


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def get_retriever(
    index_dir: str = "faiss_indexes",
    score_threshold: float = 0.30,
) -> ChemistryRetriever:
    return ChemistryRetriever(
        index_dir=index_dir,
        score_threshold=score_threshold,
    )


# ---------------------------------------------------------------------------
# CLI TESTING
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    index_dir = sys.argv[1] if len(sys.argv) > 1 else "faiss_indexes"

    retriever = ChemistryRetriever(
        index_dir=index_dir,
        score_threshold=0.30,
    )

    test_queries = [
        "Hitungkan bilangan mol dalam 4.7 g kalium oksida K2O",
        "Terangkan perbezaan antara tindak balas eksotermik dan endotermik",
        "Apakah maksud pH dan bagaimana cara mengiranya",
        "Faktor yang mempengaruhi kadar tindak balas kimia",
        "Bagaimana pempolimeran penambahan berlaku",
        "Calculate the volume of 0.5 mol oxygen gas at room temperature",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"SOALAN: {q}")
        results = retriever.retrieve(q, k=3)
        if results:
            for r in results:
                flag = "✓" if r.score >= 0.45 else "~"
                print(f"  {flag} [{r.score:.4f}] {r.content_type:14s} | {r.topic[:35]}")
            context = retriever.build_context(results)
            print(f"\nContext ({len(context)} chars):")
            print(context[:400] + "..." if len(context) > 400 else context)
        else:
            print("  Tiada hasil ditemui.")
