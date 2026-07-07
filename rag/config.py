"""
Central configuration for the RAG service.

All tunable knobs live here so the retrieval/answering behaviour can be reasoned
about in one place (and so reviewers can see the thresholds that drive the
no-hallucination gate without reading through the whole codebase).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "agri_micro_irrigation"


@dataclass
class Settings:
    # -- Models -------------------------------------------------------------
    # Embeddings run locally via sentence-transformers (no API key, offline).
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Cross-encoder used for the reranker stretch goal.
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # LLM served by Groq (fast + free tier). Swappable via env.
    llm_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))

    # -- Chunking -----------------------------------------------------------
    # Structure-aware chunking: we split on markdown headers first, then pack
    # sentences up to `chunk_target_tokens` with `chunk_overlap_tokens` overlap.
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 60
    # ~4 chars per token is a good rough estimate for English policy prose.
    chars_per_token: int = 4

    # -- Retrieval ----------------------------------------------------------
    # Pull this many candidates from the vector store, then rerank down to top_n.
    top_k_vector: int = 12
    top_n_final: int = 4
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"

    # -- No-hallucination / confidence gates --------------------------------
    # If the best (reranked) relevance score is below this, we refuse to answer
    # from the docs and say so explicitly. This is the retrieval-side guard.
    min_relevance_to_answer: float = float(os.getenv("MIN_RELEVANCE", "0.15"))
    # Answers whose computed confidence is below this get a human-review flag.
    low_confidence_threshold: float = float(os.getenv("LOW_CONFIDENCE", "0.45"))

    @property
    def chunk_target_chars(self) -> int:
        return self.chunk_target_tokens * self.chars_per_token

    @property
    def chunk_overlap_chars(self) -> int:
        return self.chunk_overlap_tokens * self.chars_per_token


settings = Settings()
