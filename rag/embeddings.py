"""
Embeddings.

The embedder is defined behind a small Protocol so the rest of the pipeline
depends on the interface, not on sentence-transformers directly. That gives us
two things:
  * the real system uses a local sentence-transformers model (no API key), and
  * tests can inject a deterministic hashing embedder that needs no model
    download or network, so the whole retrieval pipeline is testable offline.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """Real embedder backed by a local sentence-transformers model."""

    def __init__(self, model_name: str):
        # Imported lazily so the package is only required when actually used.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


class HashingEmbedder:
    """
    Deterministic, dependency-free embedder for offline testing.

    It is a bag-of-hashed-tokens vector: not semantically strong, but stable and
    good enough to prove the ingest -> store -> retrieve wiring end to end
    without downloading any model. NOT for production use.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            # L2 normalise so cosine/IP behaves sensibly.
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def build_embedder(model_name: str, *, offline: bool = False) -> Embedder:
    """Factory: real embedder normally, hashing embedder when offline=True."""
    if offline:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
