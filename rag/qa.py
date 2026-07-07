"""
The /ask flow.

Pipeline:
  1. Detect the query language; translate to English for retrieval if needed.
  2. Retrieve (vector search + rerank) the top-N chunks.
  3. NO-HALLUCINATION GATE (retrieval side): if the best chunk's relevance is
     below `min_relevance_to_answer`, refuse — say the docs don't cover it.
  4. Ask the LLM with a grounding-first prompt. If it returns the INSUFFICIENT
     sentinel (model side of the guard), refuse too.
  5. Build precise citations (file + section + snippet) for the chunks that were
     actually referenced.
  6. Score confidence and set the human-review flag.
  7. Translate the answer back to the query language if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import settings
from .confidence import compute_confidence
from .llm import GroqLLM
from .prompts import INSUFFICIENT, QA_SYSTEM, build_qa_user_prompt
from .retriever import Retriever
from .translate import Translator
from .vectorstore import Retrieved

# Message shown (and translated) when we deliberately refuse to answer.
_REFUSAL = ("The provided documents do not contain enough information to answer "
            "this question.")


@dataclass
class Citation:
    marker: int
    source_file: str
    section: str
    doc_id: str
    chunk_id: str
    snippet: str


@dataclass
class Answer:
    question: str
    answer: str
    answered: bool                 # False when we refused (no hallucination)
    citations: list[Citation]
    confidence: float
    needs_human_review: bool
    language: str
    signals: dict = field(default_factory=dict)
    retrieved: list[dict] = field(default_factory=list)


def _referenced_markers(text: str) -> set[int]:
    import re
    return {int(m) for m in re.findall(r"\[(\d+)\]", text)}


def _snippet(text: str, limit: int = 320) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


class QAEngine:
    def __init__(self, retriever: Retriever, llm: GroqLLM,
                 translator: Translator | None = None):
        self.retriever = retriever
        self.llm = llm
        self.translator = translator or Translator(llm)

    def ask(self, question: str) -> Answer:
        # 1. language handling
        language = self.translator.detect_language(question)
        query_en = self.translator.to_english(question, language)

        # 2. retrieve
        hits = self.retriever.retrieve(query_en)
        retrieved_dump = [
            {"chunk_id": h.id, "source_file": h.metadata.get("source_file"),
             "section": h.metadata.get("section"), "score": round(h.score, 3)}
            for h in hits
        ]

        # 3. retrieval-side no-hallucination gate
        best = hits[0].score if hits else 0.0
        if best < settings.min_relevance_to_answer:
            conf = compute_confidence("", hits, answered=False)
            return Answer(
                question=question, answer=self.translator.from_english(_REFUSAL, language),
                answered=False, citations=[], confidence=conf.score,
                needs_human_review=True, language=language,
                signals=conf.signals, retrieved=retrieved_dump,
            )

        # 4. grounded generation
        user_prompt = build_qa_user_prompt(query_en, hits)
        raw = self.llm.complete(QA_SYSTEM, user_prompt)

        # 4b. model-side no-hallucination gate
        if raw.strip().upper().startswith(INSUFFICIENT):
            conf = compute_confidence("", hits, answered=False)
            return Answer(
                question=question, answer=self.translator.from_english(_REFUSAL, language),
                answered=False, citations=[], confidence=conf.score,
                needs_human_review=True, language=language,
                signals=conf.signals, retrieved=retrieved_dump,
            )

        # 5. build citations for markers the model actually used
        used = _referenced_markers(raw)
        citations: list[Citation] = []
        for marker in sorted(used):
            if 1 <= marker <= len(hits):
                h = hits[marker - 1]
                citations.append(Citation(
                    marker=marker,
                    source_file=h.metadata.get("source_file", "?"),
                    section=h.metadata.get("section", "?"),
                    doc_id=h.metadata.get("doc_id", "?"),
                    chunk_id=h.id,
                    snippet=_snippet(h.text),
                ))

        # 6. confidence
        conf = compute_confidence(raw, hits, answered=True)

        # 7. translate answer back
        answer_text = self.translator.from_english(raw, language)

        return Answer(
            question=question, answer=answer_text, answered=True,
            citations=citations, confidence=conf.score,
            needs_human_review=conf.needs_human_review, language=language,
            signals=conf.signals, retrieved=retrieved_dump,
        )
