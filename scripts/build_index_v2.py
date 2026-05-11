"""
build_index_v2.py — Cikgu AI Kimia (Updated)
=============================================
Full pipeline including:
  ✓ Theory notes chunking
  ✓ Calculation file chunking
  ✓ Diagram text injection (built-in descriptions)
  ✓ Past year questions (built-in bank + .md files)

Usage:
    python scripts/build_index_v2.py
    python scripts/build_index_v2.py --kb-dir knowledge_base --validate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "rag"))
sys.path.insert(0, str(BASE_DIR / "solver"))


def main():
    parser = argparse.ArgumentParser(description="Build Cikgu AI Kimia FAISS indexes (v2)")
    parser.add_argument("--kb-dir", default=str(BASE_DIR / "knowledge_base"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "faiss_indexes"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-diagrams", action="store_true",
                        help="Skip diagram chunk injection")
    parser.add_argument("--skip-questions", action="store_true",
                        help="Skip past year question indexing")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("  CIKGU AI KIMIA — INDEX BUILDER v2")
    print("  (Theory + Calculations + Diagrams + Past Year Questions)")
    print("=" * 70)

    t_total = time.time()
    all_chunks = []

    # ── Step 1: Theory + Calculation chunks ─────────────────────────────
    print("\n[1/5] Chunking Markdown notes...")
    from chunker import chunk_all_files
    from metadata_tagger import tag_chunks, print_chunk_summary

    raw_chunks = chunk_all_files(kb_dir)
    tagged_chunks = tag_chunks(raw_chunks)
    print_chunk_summary(tagged_chunks)

    # Inject diagram descriptions into theory chunks
    if not args.skip_diagrams:
        print("\n[2/5] Injecting diagram descriptions into chunks...")
        from diagram_processor import DiagramProcessor

        processor = DiagramProcessor(
            images_dir=str(kb_dir / "images"),
            use_ocr=True,
        )

        injected_count = 0
        for chunk in tagged_chunks:
            if chunk.has_diagram:
                chunk.content = processor.inject_descriptions_into_chunk(chunk.content)
                chunk.embed_text = processor.inject_descriptions_into_chunk(chunk.embed_text)
                injected_count += 1

        print(f"  Injected descriptions into {injected_count} chunks with diagrams")

        # Add dedicated diagram chunks (one per unique diagram)
        print("\n[3/5] Building dedicated diagram index chunks...")
        diagram_chunks = processor.build_diagram_chunks(str(kb_dir))
        print(f"  Created {len(diagram_chunks)} diagram chunks")
        all_chunks.extend(diagram_chunks)
    else:
        print("\n[2/5] Skipping diagram injection (--skip-diagrams)")
        print("\n[3/5] Skipping diagram chunks (--skip-diagrams)")

    all_chunks.extend([c.to_dict() for c in tagged_chunks])

    # ── Step 4: Past year questions ──────────────────────────────────────
    if not args.skip_questions:
        print("\n[4/5] Loading past year questions...")
        from past_year_questions import questions_to_chunks, load_from_markdown_files

        # Built-in question bank
        builtin_qa = questions_to_chunks()
        print(f"  Built-in question bank: {len(builtin_qa)} questions")

        # Questions from Markdown files (if any)
        md_qa_dir = kb_dir / "questions" / "past_years"
        md_qa = load_from_markdown_files(str(md_qa_dir)) if md_qa_dir.exists() else []
        if md_qa:
            print(f"  Markdown question files: {len(md_qa)} questions")

        all_qa = builtin_qa + md_qa
        print(f"  Total QA chunks: {len(all_qa)}")
        all_chunks.extend(all_qa)
    else:
        print("\n[4/5] Skipping past year questions (--skip-questions)")

    # ── Step 5: Embed + Index ────────────────────────────────────────────
    print(f"\n[5/5] Embedding {len(all_chunks)} total chunks and building indexes...")
    from embedder import get_embedder
    from indexer import FAISSIndexManager
    import numpy as np

    embedder = get_embedder()
    cache_path = str(BASE_DIR / ".embedding_cache.pkl")

    # Extract embed texts
    texts = [c.get("embed_text", c.get("content", "")) if isinstance(c, dict)
             else (c.embed_text or c.content)
             for c in all_chunks]

    # Use cache if available
    if not args.no_cache:
        from embedder import EmbeddingCache

        class _ChunkWrapper:
            def __init__(self, d):
                self._d = d
            def get(self, k, default=None):
                return self._d.get(k, default) if isinstance(self._d, dict) else getattr(self._d, k, default)
            @property
            def content_type(self):
                return self._d.get("content_type", "theory") if isinstance(self._d, dict) else self._d.content_type
            def to_dict(self):
                return self._d if isinstance(self._d, dict) else self._d.to_dict()

        cache = EmbeddingCache(cache_path)
        embeddings = cache.embed_with_cache(texts, embedder)
    else:
        embeddings = embedder.embed_texts(texts)

    # Build indexes
    manager = FAISSIndexManager(index_dir=str(out_dir), embed_dim=embeddings.shape[1])

    # Wrap raw dicts in a compatible object for add_chunks
    class _DictChunk:
        def __init__(self, d):
            self._d = d
            self.content_type = d.get("content_type", "theory")
        def to_dict(self):
            return self._d

    wrapped = [_DictChunk(c) if isinstance(c, dict) else c for c in all_chunks]
    manager.add_chunks(wrapped, embeddings)
    manager.save_all()
    manager.print_stats()

    # ── Validate ─────────────────────────────────────────────────────────
    if args.validate:
        print("\n[VALIDATION] Running test queries...")
        _validate(out_dir, embedder)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  BUILD COMPLETE in {elapsed:.1f}s")
    print(f"  Total chunks indexed: {len(all_chunks)}")
    print(f"  Indexes saved to: {out_dir}")
    print(f"{'='*70}")


def _validate(index_dir, embedder):
    from retriever import ChemistryRetriever

    retriever = ChemistryRetriever(
        index_dir=str(index_dir),
        embedder=embedder,
        score_threshold=0.0,
    )

    tests = [
        ("Hitungkan bilangan mol 11.2g Fe",           ["index_calculations"]),
        ("Terangkan tindak balas eksotermik",          ["index_theory"]),
        ("Rajah kalorimeter menunjukkan apa",          ["index_theory"]),
        ("Soalan SPM kadar tindak balas",              ["index_qa"]),
        ("Perbezaan pempolimeran penambahan kondensasi", ["index_theory", "index_qa"]),
        ("Nombor pengoksidaan KMnO4",                  ["index_qa", "index_calculations"]),
    ]

    print(f"\n{'─'*70}")
    all_pass = True
    for query, indexes in tests:
        results = retriever.retrieve(query, k=3, index_names=indexes)
        top = results[0] if results else None
        score = top.score if top else 0.0
        status = "✓" if score >= 0.30 else "✗"
        if score < 0.30:
            all_pass = False
        ctype = top.content_type if top else "none"
        print(f"  {status} [{score:.4f}] {ctype:15s} | {query[:50]}")

    print(f"  {'ALL PASS ✓' if all_pass else 'SOME BELOW THRESHOLD ✗'}")


if __name__ == "__main__":
    main()
