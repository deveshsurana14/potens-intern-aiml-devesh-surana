"""
Assembly / factory.

One place that wires the components together so the API, the Streamlit app, and
the tests all build the system the same way. `offline=True` swaps in the hashing
embedder and skips the cross-encoder download, which lets the retrieval pipeline
run with no network and no model weights (used by the test suite).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CHROMA_DIR, COLLECTION_NAME, settings
from .contradict import ContradictionEngine
from .embeddings import build_embedder
from .llm import GroqLLM
from .qa import QAEngine
from .retriever import Reranker, Retriever
from .translate import Translator
from .vectorstore import VectorStore


@dataclass
class RAGSystem:
    store: VectorStore
    retriever: Retriever
    qa: QAEngine
    contradict: ContradictionEngine
    llm: GroqLLM


def build_system(*, offline: bool = False) -> RAGSystem:
    embedder = build_embedder(settings.embedding_model, offline=offline)
    store = VectorStore(CHROMA_DIR, COLLECTION_NAME, embedder)

    reranker = None
    if settings.use_reranker and not offline:
        reranker = Reranker(settings.reranker_model)

    retriever = Retriever(store, reranker)
    llm = GroqLLM()
    translator = Translator(llm)
    qa = QAEngine(retriever, llm, translator)
    contradict = ContradictionEngine(store, llm, retriever)
    return RAGSystem(store=store, retriever=retriever, qa=qa,
                     contradict=contradict, llm=llm)
