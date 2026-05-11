"""
embedder.py — Cikgu AI Kimia RAG Pipeline
==========================================
Multilingual sentence embedding using:
  fastembed (ONNX runtime — NO torch, NO GPU required)
  Model: BAAI/bge-small-en-v1.5 (multilingual capable)

RAM Usage: ~80MB vs ~1GB with sentence-transformers + torch
Platform:  Render.com free tier (512MB RAM) compatible

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Dimensionality for this model
EMBED_DIM = 384

# Batch size for encoding
DEFAULT_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# SINGLETON MODEL LOADER
# ---------------------------------------------------------------------------

_model_instance = None


def get_model():
    """Load the fastembed model once; return cached instance on subsequent calls."""
    global _model_instance
    if _model_instance is None:
        from fastembed import TextEmbedding
        print(f"[embedder] Loading fastembed model: {MODEL_NAME}")
        _model_instance = TextEmbedding(model_name=MODEL_NAME)
        print(f"[embedder] Model loaded. Embedding dim: {EMBED_DIM}")
    return _model_instance


# ---------------------------------------------------------------------------
# CORE EMBEDDER CLASS
# ---------------------------------------------------------------------------

class ChemistryEmbedder:
    """
    Wrapper around fastembed TextEmbedding for Cikgu AI Kimia.

    Uses ONNX runtime instead of PyTorch — dramatically lower RAM usage.
    Compatible with Render.com free tier (512MB RAM limit).

    The BAAI/bge-small-en-v1.5 model handles Bahasa Malaysia
    and English in the same vector space via cross-lingual transfer.
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_model()
        return self._model

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Parameters
        ----------
        texts : list of strings to embed
        show_progress : show progress (fastembed handles internally)
        normalize : L2-normalize embeddings (required for cosine similarity)

        Returns
        -------
        np.ndarray of shape (len(texts), EMBED_DIM)
        """
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        # fastembed returns a generator — convert to list then stack
        embeddings_gen = self.model.embed(texts, batch_size=self.batch_size)
        embeddings = np.array(list(embeddings_gen), dtype=np.float32)

        if normalize:
            # L2 normalize each row
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid division by zero
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns shape (1, EMBED_DIM) — ready for FAISS search.
        """
        embeddings_gen = self.model.embed([query], batch_size=1)
        vec = np.array(list(embeddings_gen), dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.astype(np.float32)

    def embed_chunks(
        self,
        chunks: list,
        text_field: str = "embed_text",
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of ChemistryChunk (or dict) objects.
        Reads the `embed_text` field by default.
        """
        if not chunks:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        texts = []
        for c in chunks:
            if isinstance(c, dict):
                t = c.get(text_field, "")
            else:
                t = getattr(c, text_field, "")
            texts.append(t or "")

        return self.embed_texts(texts, show_progress=show_progress)


# ---------------------------------------------------------------------------
# EMBEDDING CACHE
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Simple disk cache for embeddings.
    Key = SHA256 of the text — avoids re-embedding unchanged chunks.
    """

    def __init__(self, cache_path: str | Path = ".embedding_cache.pkl"):
        self.cache_path = Path(cache_path)
        self._cache: dict = {}
        self._load()

    def _load(self):
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
            print(f"[cache] Loaded {len(self._cache)} cached embeddings")

    def save(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)
        print(f"[cache] Saved {len(self._cache)} embeddings to {self.cache_path}")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._cache.get(self._hash(text))

    def set(self, text: str, vec: np.ndarray):
        self._cache[self._hash(text)] = vec

    def embed_with_cache(
        self,
        texts: List[str],
        embedder: ChemistryEmbedder,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed texts, using cache for texts already seen.
        Only calls the model for new texts.
        """
        results = [None] * len(texts)
        to_embed: List[tuple] = []  # (original_idx, text)

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                to_embed.append((i, text))

        if to_embed:
            print(f"[cache] Embedding {len(to_embed)} new texts ({len(texts) - len(to_embed)} cached)")
            new_texts = [t for _, t in to_embed]
            new_vecs = embedder.embed_texts(new_texts, show_progress=show_progress)
            for (orig_idx, text), vec in zip(to_embed, new_vecs):
                results[orig_idx] = vec
                self.set(text, vec)
            self.save()
        else:
            print(f"[cache] All {len(texts)} texts served from cache")

        return np.stack(results).astype(np.float32)


# ---------------------------------------------------------------------------
# CONVENIENCE
# ---------------------------------------------------------------------------

def get_embedder(batch_size: int = DEFAULT_BATCH_SIZE) -> ChemistryEmbedder:
    return ChemistryEmbedder(batch_size=batch_size)


# ---------------------------------------------------------------------------
# CLI TESTING
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embedder = get_embedder()

    test_texts = [
        "Hitungkan bilangan mol dalam 4.7 g kalium oksida (K₂O)",
        "Calculate the number of moles in 4.7 g of potassium oxide",
        "pH = -log [H+]",
        "Alkali ditakrifkan sebagai bahan yang mengion dalam air untuk menghasilkan ion OH-",
        "Tindak balas eksotermik membebaskan haba ke persekitaran",
    ]

    print("[embedder] Embedding test texts...")
    vecs = embedder.embed_texts(test_texts)
    print(f"Shape: {vecs.shape}")

    # Cosine similarities (already normalised)
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = float(np.dot(vecs[i], vecs[j]))
            print(f"  sim({i},{j}): {sim:.4f}  |  '{test_texts[i][:40]}' ↔ '{test_texts[j][:40]}'")

    # Query test
    query_vec = embedder.embed_query("how many moles is 4.7 grams of K2O")
    sims = vecs @ query_vec.T
    print(f"\nQuery: 'how many moles is 4.7 grams of K2O'")
    for i, s in enumerate(sims.flatten()):
        print(f"  [{s:.4f}] {test_texts[i][:60]}")
