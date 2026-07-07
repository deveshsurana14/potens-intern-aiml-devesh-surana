"""
Vector store wrapper around ChromaDB (persistent, local).

We manage embeddings ourselves (via the injected Embedder) rather than using
Chroma's built-in embedding function, so that the same embedder is used at
ingest and query time and can be swapped/faked for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb

from .chunking import Chunk
from .embeddings import Embedder


@dataclass
class Retrieved:
    id: str
    text: str
    metadata: dict
    distance: float          # raw Chroma distance (lower = closer)
    score: float             # normalised similarity in [0, 1] (higher = better)


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder

    # -- ingest -------------------------------------------------------------
    def reset(self) -> None:
        """Drop and recreate the collection (used for a clean re-ingest)."""
        name = self._collection.name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=self._embedder.embed([c.text for c in batch]),
            )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    # -- query --------------------------------------------------------------
    def query(self, text: str, top_k: int) -> list[Retrieved]:
        q_emb = self._embedder.embed([text])[0]
        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Retrieved] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Cosine distance in [0, 2]; convert to a [0,1]-ish similarity.
            score = max(0.0, 1.0 - dist / 2.0)
            hits.append(
                Retrieved(id=_id, text=doc, metadata=meta, distance=dist, score=score)
            )
        return hits

    def get_by_source(self, source_file: str) -> list[Retrieved]:
        """All chunks belonging to one document (used by /contradict)."""
        res = self._collection.get(
            where={"source_file": source_file},
            include=["documents", "metadatas"],
        )
        out: list[Retrieved] = []
        for _id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append(Retrieved(id=_id, text=doc, metadata=meta, distance=0.0, score=1.0))
        return out

    def list_sources(self) -> list[str]:
        res = self._collection.get(include=["metadatas"])
        seen: list[str] = []
        for meta in res["metadatas"]:
            sf = meta.get("source_file")
            if sf and sf not in seen:
                seen.append(sf)
        return sorted(seen)
