"""
Retriever = vector search  (+ optional cross-encoder reranker).

Stretch goal #2 (reranker): a bi-encoder vector search is fast but coarse — it
compares independent embeddings of the query and each chunk. A cross-encoder
reads the (query, chunk) pair together and scores true relevance far more
accurately. We over-fetch top_k from the vector store, then rerank down to
top_n with the cross-encoder. The reranker is optional and degrades gracefully:
if the model can't load (e.g. offline), we fall back to vector order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import settings
from .vectorstore import Retrieved, VectorStore


@dataclass
class Reranker:
    model_name: str

    def __post_init__(self):
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception:
            # Kept None -> caller falls back to vector ordering.
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, hits: Sequence[Retrieved]) -> list[Retrieved]:
        if not self._model or not hits:
            return list(hits)
        pairs = [(query, h.text) for h in hits]
        raw = self._model.predict(pairs)
        # Map cross-encoder logits to (0,1) so they can drive the confidence gate.
        import math

        rescored: list[Retrieved] = []
        for h, s in zip(hits, raw):
            prob = 1.0 / (1.0 + math.exp(-float(s)))
            rescored.append(
                Retrieved(
                    id=h.id, text=h.text, metadata=h.metadata,
                    distance=h.distance, score=prob,
                )
            )
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored


class Retriever:
    def __init__(self, store: VectorStore, reranker: Reranker | None = None):
        self.store = store
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int | None = None,
                 top_n: int | None = None) -> list[Retrieved]:
        top_k = top_k or settings.top_k_vector
        top_n = top_n or settings.top_n_final

        hits = self.store.query(query, top_k=top_k)
        if self.reranker and self.reranker.available:
            hits = self.reranker.rerank(query, hits)
        return hits[:top_n]
