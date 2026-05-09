"""
rag_pipeline.py — Cikgu AI Kimia Unified Pipeline
===================================================
Single entry point that wires together:
  chunker → metadata_tagger → embedder → indexer → retriever

Also provides the top-level answer() function used by FastAPI and the bot.

This file is the integration layer — it imports from all submodules
and coordinates the full question → answer flow.

Usage (from Python):
    from rag_pipeline import CikguAIPipeline
    pipeline = CikguAIPipeline(index_dir="faiss_indexes")
    answer = await pipeline.answer("Hitungkan mol dalam 4.7g K2O")

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("rag_pipeline")

# Path setup
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "rag"))
sys.path.insert(0, str(_HERE / "solver"))


class CikguAIPipeline:
    """
    Unified pipeline for Cikgu AI Kimia.

    Wires together:
      - Deterministic Python solver (for calculations)
      - RAG retriever (for theory)
      - LLM (for explanation + formatting)

    Thread-safe singleton — use get_pipeline() factory.
    """

    def __init__(
        self,
        index_dir: str = "faiss_indexes",
        score_threshold: float = 0.30,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        self.index_dir = index_dir
        self.score_threshold = score_threshold
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.llm_model = llm_model

        # Lazy-load components
        self._embedder = None
        self._retriever = None
        self._router = None
        self._solver = None

    # ── Component accessors (lazy load) ──────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            from embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    @property
    def retriever(self):
        if self._retriever is None:
            from retriever import ChemistryRetriever
            self._retriever = ChemistryRetriever(
                index_dir=self.index_dir,
                embedder=self.embedder,
                score_threshold=self.score_threshold,
            )
        return self._retriever

    @property
    def router(self):
        if self._router is None:
            from router import route
            self._router = route
        return self._router

    @property
    def solver(self):
        if self._solver is None:
            from solver_engine import solve_by_task
            self._solver = solve_by_task
        return self._solver

    # ── Main answer method ───────────────────────────────────────────────

    async def answer(
        self,
        question: str,
        chapter_filter: Optional[int] = None,
        tingkatan_filter: Optional[int] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer a student's chemistry question.

        Returns dict with:
            answer: str
            answer_type: "calculation" | "theory" | "fallback"
            solver_used: bool
            context_found: bool
            sources: list
        """
        # Step 1: Route
        task, data = self.router(question)

        # Step 2A: Calculation path
        if task != "unknown" and data is not None:
            try:
                solver_answer = self.solver(task, data)
                rag_results = self.retriever.retrieve_for_calculation(question, k=2)

                return {
                    "answer": solver_answer,
                    "answer_type": "calculation",
                    "solver_used": True,
                    "context_found": len(rag_results) > 0,
                    "sources": [
                        {"topic": r.topic, "score": r.score}
                        for r in rag_results
                    ],
                }
            except Exception as e:
                logger.warning(f"Solver failed ({task}): {e}. Falling back to RAG.")

        # Step 2B: Theory / RAG path
        rag_results = self.retriever.retrieve(
            query=question,
            k=top_k,
            chapter_filter=chapter_filter,
            tingkatan_filter=tingkatan_filter,
        )

        if self.retriever.is_sufficient_context(rag_results, min_score=0.35):
            from retriever import build_rag_prompt
            prompt = build_rag_prompt(question, rag_results)
            llm_answer = await self._call_llm(prompt)
            return {
                "answer": llm_answer,
                "answer_type": "theory",
                "solver_used": False,
                "context_found": True,
                "sources": [
                    {"topic": r.topic, "score": r.score, "chunk_id": r.chunk_id}
                    for r in rag_results
                ],
            }

        return {
            "answer": (
                "Maaf, soalan ini tidak terdapat dalam nota kimia saya. "
                "Sila rujuk buku teks SPM."
            ),
            "answer_type": "fallback",
            "solver_used": False,
            "context_found": False,
            "sources": [],
        }

    async def _call_llm(self, prompt: str, max_tokens: int = 700) -> str:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
        if groq_key:
            try:
                from groq import AsyncGroq
                client = AsyncGroq(api_key=groq_key)
                resp = await client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"[Ralat Groq: {e}]"
        return "[GROQ_API_KEY tidak dikonfigurasi]"

    # ── Build indexes ─────────────────────────────────────────────────────

    def build_indexes(
        self,
        kb_dir: str,
        use_cache: bool = True,
    ) -> None:
        """Build or rebuild FAISS indexes from knowledge base directory."""
        from chunker import chunk_all_files
        from metadata_tagger import tag_chunks
        from indexer import build_indexes_from_chunks

        raw = chunk_all_files(kb_dir)
        tagged = tag_chunks(raw)
        build_indexes_from_chunks(
            chunks=tagged,
            embedder=self.embedder,
            index_dir=self.index_dir,
            use_cache=use_cache,
        )
        # Invalidate cached retriever so it reloads
        self._retriever = None
        logger.info("Indexes rebuilt successfully.")


# ── Singleton factory ──────────────────────────────────────────────────────

_pipeline_instance: Optional[CikguAIPipeline] = None


def get_pipeline(
    index_dir: str = "faiss_indexes",
    score_threshold: float = 0.30,
) -> CikguAIPipeline:
    """Return the global CikguAIPipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CikguAIPipeline(
            index_dir=index_dir,
            score_threshold=score_threshold,
        )
    return _pipeline_instance


# ── CLI quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    async def test():
        pipeline = CikguAIPipeline(index_dir="faiss_indexes")
        questions = [
            "Hitungkan bilangan mol dalam 4.7 g K2O",
            "Terangkan tindak balas eksotermik",
            "Apakah pH?",
        ]
        for q in questions:
            print(f"\nQ: {q}")
            result = await pipeline.answer(q)
            print(f"Type: {result['answer_type']}")
            print(f"Answer:\n{result['answer'][:300]}")

    asyncio.run(test())
