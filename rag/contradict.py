"""
The /contradict flow.

Given two document IDs (source filenames) and an optional topic, decide whether
the two documents genuinely conflict.

Approach:
  * If a topic is given, retrieve the most relevant chunks about that topic FROM
    EACH document separately (a filtered vector search per document). This keeps
    the comparison focused instead of dumping whole documents at the model.
  * If no topic is given, fall back to each document's most information-dense
    sections (we just take a spread of chunks).
  * Send both excerpt sets to the LLM with a strict-JSON contradiction prompt and
    parse the structured verdict.

The agri corpus has real conflicts (e.g. Maharashtra 80/75% effective subsidy vs
Gujarat 70/60%; 5 ha vs 10 ha area ceiling; DBT-to-farmer vs supplier-credit), so
this endpoint returns meaningful results rather than a toy demo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import GroqLLM
from .prompts import CONTRADICT_SYSTEM, build_contradict_user_prompt
from .retriever import Retriever
from .vectorstore import Retrieved, VectorStore


@dataclass
class ContradictionResult:
    conflict: bool
    topic: str
    document_a: str
    document_b: str
    document_a_position: str
    document_b_position: str
    reasoning: str
    excerpts_a: list[str]
    excerpts_b: list[str]


def _topic_chunks(store: VectorStore, source_file: str, topic: str,
                  k: int = 4) -> list[Retrieved]:
    """Most relevant chunks about `topic` within a single document."""
    q_emb = store._embedder.embed([topic])[0]  # noqa: SLF001 (intentional)
    res = store._collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where={"source_file": source_file},
        include=["documents", "metadatas", "distances"],
    )
    out: list[Retrieved] = []
    for _id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(Retrieved(id=_id, text=doc, metadata=meta,
                             distance=dist, score=max(0.0, 1.0 - dist / 2.0)))
    return out


def _extract_json(text: str) -> dict:
    """Robustly pull the JSON object out of the model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


class ContradictionEngine:
    def __init__(self, store: VectorStore, llm: GroqLLM,
                 retriever: Retriever | None = None):
        self.store = store
        self.llm = llm

    def compare(self, source_a: str, source_b: str, topic: str = "") -> ContradictionResult:
        if topic:
            chunks_a = _topic_chunks(self.store, source_a, topic)
            chunks_b = _topic_chunks(self.store, source_b, topic)
        else:
            chunks_a = self.store.get_by_source(source_a)[:5]
            chunks_b = self.store.get_by_source(source_b)[:5]

        if not chunks_a or not chunks_b:
            missing = source_a if not chunks_a else source_b
            raise ValueError(f"No chunks found for document '{missing}'. "
                             f"Check the source filename.")

        user_prompt = build_contradict_user_prompt(
            topic, source_a, chunks_a, source_b, chunks_b
        )
        raw = self.llm.complete(CONTRADICT_SYSTEM, user_prompt, max_tokens=700)
        data = _extract_json(raw)

        return ContradictionResult(
            conflict=bool(data.get("conflict", False)),
            topic=data.get("topic", topic or "(unspecified)"),
            document_a=source_a,
            document_b=source_b,
            document_a_position=data.get("document_a_position", ""),
            document_b_position=data.get("document_b_position", ""),
            reasoning=data.get("reasoning", ""),
            excerpts_a=[c.text for c in chunks_a],
            excerpts_b=[c.text for c in chunks_b],
        )
