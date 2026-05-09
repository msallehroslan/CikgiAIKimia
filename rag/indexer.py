"""
indexer.py — Cikgu AI Kimia RAG Pipeline
=========================================
Builds and manages 3 separate FAISS indexes:
  - Index A: Theory       (concepts, definitions, explanations)
  - Index B: Calculations (worked examples, step-by-step, formulas)
  - Index C: QA / Exam    (questions, answer schemes, exam items)

Features:
  - Separate FAISS IndexFlatIP (inner product = cosine on normalised vecs)
  - Full metadata sidecar stored as JSON
  - Incremental update support (add without rebuild)
  - Index health check and statistics
  - Save/load from disk

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# INDEX DEFINITIONS
# ---------------------------------------------------------------------------

INDEX_NAMES = {
    "theory":       "index_theory",
    "calculation":  "index_calculations",
    "qa_scheme":    "index_qa",
    # Aliases
    "formula":      "index_calculations",  # formulas go into calculation index
    "definition":   "index_theory",        # definitions go into theory index
}

CONTENT_TYPE_TO_INDEX = {
    "theory":      "theory",
    "definition":  "theory",
    "calculation": "calculation",
    "formula":     "calculation",
    "qa_scheme":   "qa_scheme",
}


# ---------------------------------------------------------------------------
# FAISS INDEX MANAGER
# ---------------------------------------------------------------------------

class FAISSIndexManager:
    """
    Manages 3 FAISS indexes for Cikgu AI Kimia.

    Directory layout:
        faiss_indexes/
            index_theory.faiss      — FAISS binary
            index_theory_meta.json  — metadata sidecar
            index_calculations.faiss
            index_calculations_meta.json
            index_qa.faiss
            index_qa_meta.json
    """

    def __init__(
        self,
        index_dir: str | Path = "faiss_indexes",
        embed_dim: int = 384,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim

        # Lazy-loaded FAISS indexes
        self._indexes: Dict[str, Any] = {}
        # Metadata lists (parallel to FAISS vectors)
        self._metadata: Dict[str, List[Dict]] = {}

    # -----------------------------------------------------------------------
    # FAISS IMPORT HELPER
    # -----------------------------------------------------------------------

    @staticmethod
    def _faiss():
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

    # -----------------------------------------------------------------------
    # INDEX CREATION / LOADING
    # -----------------------------------------------------------------------

    def _get_or_create_index(self, index_name: str):
        """Return the FAISS index for this name, loading from disk or creating new."""
        if index_name in self._indexes:
            return self._indexes[index_name]

        faiss = self._faiss()
        faiss_path = self.index_dir / f"{index_name}.faiss"
        meta_path = self.index_dir / f"{index_name}_meta.json"

        if faiss_path.exists() and meta_path.exists():
            # Load existing
            index = faiss.read_index(str(faiss_path))
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            print(f"[indexer] Loaded {index_name}: {index.ntotal} vectors")
        else:
            # Create new inner-product index (cosine on normalised vectors)
            index = faiss.IndexFlatIP(self.embed_dim)
            meta = []
            print(f"[indexer] Created new index: {index_name}")

        self._indexes[index_name] = index
        self._metadata[index_name] = meta
        return index

    def _resolve_index_name(self, content_type: str) -> str:
        """Map content_type to index name."""
        category = CONTENT_TYPE_TO_INDEX.get(content_type, "theory")
        return INDEX_NAMES.get(category, "index_theory")

    # -----------------------------------------------------------------------
    # ADD VECTORS
    # -----------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Any],      # List[ChemistryChunk]
        embeddings: np.ndarray,
    ) -> Dict[str, int]:
        """
        Add chunks and their embeddings to the appropriate indexes.

        Parameters
        ----------
        chunks : List of ChemistryChunk objects
        embeddings : np.ndarray shape (N, embed_dim)

        Returns
        -------
        dict mapping index_name → number of vectors added
        """
        assert len(chunks) == len(embeddings), "chunks and embeddings must have same length"

        counts: Dict[str, int] = {}

        # Group by target index
        groups: Dict[str, Tuple[list, list]] = {}
        for chunk, vec in zip(chunks, embeddings):
            ct = getattr(chunk, 'content_type', None) or chunk.get('content_type', 'theory')
            idx_name = self._resolve_index_name(ct)
            if idx_name not in groups:
                groups[idx_name] = ([], [])
            meta_dict = chunk.to_dict() if hasattr(chunk, 'to_dict') else dict(chunk)
            groups[idx_name][0].append(meta_dict)
            groups[idx_name][1].append(vec)

        for idx_name, (metas, vecs) in groups.items():
            index = self._get_or_create_index(idx_name)
            arr = np.stack(vecs).astype(np.float32)
            index.add(arr)
            self._metadata[idx_name].extend(metas)
            counts[idx_name] = len(metas)
            print(f"[indexer] Added {len(metas):4d} vectors to {idx_name} (total: {index.ntotal})")

        return counts

    # -----------------------------------------------------------------------
    # SAVE TO DISK
    # -----------------------------------------------------------------------

    def save_all(self) -> None:
        """Persist all loaded indexes and metadata to disk."""
        faiss = self._faiss()
        for idx_name, index in self._indexes.items():
            faiss_path = self.index_dir / f"{idx_name}.faiss"
            meta_path = self.index_dir / f"{idx_name}_meta.json"

            faiss.write_index(index, str(faiss_path))
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self._metadata[idx_name], f, ensure_ascii=False, indent=2)

            print(f"[indexer] Saved {idx_name}: {index.ntotal} vectors → {faiss_path}")

    # -----------------------------------------------------------------------
    # SEARCH
    # -----------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        index_names: Optional[List[str]] = None,
        k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search across specified indexes (or all if None).

        Parameters
        ----------
        query_vec : np.ndarray shape (1, embed_dim)
        index_names : list of index names to search, or None for all
        k : number of results per index
        score_threshold : minimum cosine similarity (0.0 = no filter)
        metadata_filter : dict of {field: value} to filter results

        Returns
        -------
        List of result dicts sorted by score desc, each containing:
            - score: float cosine similarity
            - metadata: ChemistryChunk dict
        """
        if index_names is None:
            index_names = ["index_theory", "index_calculations", "index_qa"]

        results = []

        for idx_name in index_names:
            faiss_path = self.index_dir / f"{idx_name}.faiss"
            if not faiss_path.exists():
                continue

            index = self._get_or_create_index(idx_name)
            if index.ntotal == 0:
                continue

            # Search
            k_actual = min(k, index.ntotal)
            scores, indices = index.search(query_vec, k_actual)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score < score_threshold:
                    continue

                meta = self._metadata[idx_name][idx]

                # Apply metadata filter
                if metadata_filter:
                    match = all(
                        meta.get(field) == val
                        for field, val in metadata_filter.items()
                        if val is not None
                    )
                    if not match:
                        continue

                results.append({
                    "score": float(score),
                    "index": idx_name,
                    "metadata": meta,
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # -----------------------------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return statistics about all indexes."""
        stats = {}
        for idx_name in ["index_theory", "index_calculations", "index_qa"]:
            faiss_path = self.index_dir / f"{idx_name}.faiss"
            meta_path = self.index_dir / f"{idx_name}_meta.json"

            if faiss_path.exists():
                if idx_name in self._indexes:
                    total = self._indexes[idx_name].ntotal
                else:
                    faiss = self._faiss()
                    idx = faiss.read_index(str(faiss_path))
                    total = idx.ntotal
                stats[idx_name] = {"vectors": total, "on_disk": True}
            else:
                stats[idx_name] = {"vectors": 0, "on_disk": False}

        return stats

    def print_stats(self) -> None:
        s = self.stats()
        print("\n" + "="*50)
        print("FAISS INDEX STATISTICS")
        print("="*50)
        for name, info in s.items():
            status = "✓" if info["on_disk"] else "✗"
            print(f"  {status} {name:35s}: {info['vectors']:5d} vectors")
        total = sum(v["vectors"] for v in s.values())
        print(f"  {'TOTAL':36s}: {total:5d} vectors")


# ---------------------------------------------------------------------------
# GLOBAL INSTANCE
# ---------------------------------------------------------------------------

_manager_instance: Optional[FAISSIndexManager] = None


def get_index_manager(
    index_dir: str = "faiss_indexes",
    embed_dim: int = 384,
) -> FAISSIndexManager:
    """Return the singleton FAISSIndexManager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = FAISSIndexManager(index_dir=index_dir, embed_dim=embed_dim)
    return _manager_instance


# ---------------------------------------------------------------------------
# BUILD PIPELINE ENTRYPOINT
# ---------------------------------------------------------------------------

def build_indexes_from_chunks(
    chunks: List[Any],              # List[ChemistryChunk]
    embedder,                       # ChemistryEmbedder instance
    index_dir: str = "faiss_indexes",
    use_cache: bool = True,
    cache_path: str = ".embedding_cache.pkl",
) -> FAISSIndexManager:
    """
    Full pipeline: chunks → embeddings → FAISS indexes.

    Parameters
    ----------
    chunks : tagged ChemistryChunk list
    embedder : ChemistryEmbedder instance
    index_dir : where to save FAISS files
    use_cache : use embedding cache to avoid re-embedding unchanged text
    cache_path : path to embedding cache file

    Returns
    -------
    FAISSIndexManager (all indexes built and saved)
    """
    from embedder import EmbeddingCache

    print(f"\n[indexer] Building indexes from {len(chunks)} chunks...")
    t0 = time.time()

    texts = [c.embed_text if hasattr(c, 'embed_text') else c.get('embed_text', '') for c in chunks]

    if use_cache:
        cache = EmbeddingCache(cache_path)
        embeddings = cache.embed_with_cache(texts, embedder)
    else:
        embeddings = embedder.embed_chunks(chunks)

    manager = FAISSIndexManager(index_dir=index_dir, embed_dim=embeddings.shape[1])
    manager.add_chunks(chunks, embeddings)
    manager.save_all()

    elapsed = time.time() - t0
    print(f"[indexer] Done in {elapsed:.1f}s")
    manager.print_stats()

    return manager


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from chunker import chunk_all_files
    from metadata_tagger import tag_chunks
    from embedder import get_embedder

    if len(sys.argv) < 2:
        print("Usage: python indexer.py <knowledge_base_dir> [index_output_dir]")
        sys.exit(1)

    kb_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "faiss_indexes"

    print(f"[indexer] Knowledge base: {kb_dir}")
    print(f"[indexer] Index output:   {out_dir}")

    raw_chunks = chunk_all_files(kb_dir)
    tagged = tag_chunks(raw_chunks)
    embedder = get_embedder()
    manager = build_indexes_from_chunks(tagged, embedder, index_dir=out_dir)
    manager.print_stats()
