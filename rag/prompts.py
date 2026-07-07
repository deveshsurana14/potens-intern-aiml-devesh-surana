"""
Prompt templates.

The QA prompt is deliberately strict about grounding: the model must answer ONLY
from the supplied context and must emit a sentinel (`INSUFFICIENT_CONTEXT`) when
the context does not cover the question. This sentinel is the model-side half of
the no-hallucination guard (the retrieval-side half is the relevance threshold).

Each context block is numbered [1], [2], ... and the model is told to reference
those numbers, which we later map back to concrete citations.
"""
from __future__ import annotations

from .vectorstore import Retrieved

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

QA_SYSTEM = f"""You are a careful assistant answering questions about Indian \
micro-irrigation (drip/sprinkler) subsidy policy and drip system engineering.

Rules you MUST follow:
1. Answer ONLY using facts stated in the provided CONTEXT blocks. Do not use \
outside knowledge and do not guess.
2. If the context does not contain enough information to answer, reply with \
exactly this token and nothing else: {INSUFFICIENT}
3. When you use a fact, cite the block it came from using its bracket number, \
e.g. "Small farmers receive 55% [1]." Cite every factual sentence.
4. Be concise and precise. Do not add caveats that are not in the context.
5. If different blocks give different figures for different States or schemes, \
keep them distinct and attribute each to its source. Never average or merge them.
"""


def build_qa_user_prompt(question: str, contexts: list[Retrieved]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        src = c.metadata.get("source_file", "?")
        sec = c.metadata.get("section", "?")
        blocks.append(f"[{i}] (source: {src} — section: {sec})\n{c.text}")
    context_text = "\n\n".join(blocks)
    return (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the context above, with bracket citations."
    )


CONTRADICT_SYSTEM = """You compare two sets of excerpts from two different policy \
documents and decide whether they CONFLICT on a specific topic.

Definitions:
- CONFLICT = the two documents make claims that cannot both be true at once for \
the same subject (e.g. different mandatory subsidy percentages for the same \
farmer category, different area ceilings, mutually exclusive procedures).
- NOT A CONFLICT = they simply cover different points, add detail, or describe \
different-but-compatible things (e.g. one is central policy and the other is an \
optional State top-up that stacks on it).

Respond in STRICT JSON with exactly these keys:
{
  "conflict": true or false,
  "topic": "<short topic string>",
  "document_a_position": "<what doc A says, with the specific figures/rules>",
  "document_b_position": "<what doc B says, with the specific figures/rules>",
  "reasoning": "<why this is or is not a genuine conflict>"
}
Base every statement only on the provided excerpts. Do not invent figures. \
Output JSON only — no prose before or after."""


def build_contradict_user_prompt(topic: str, name_a: str, chunks_a: list[Retrieved],
                                 name_b: str, chunks_b: list[Retrieved]) -> str:
    def fmt(chunks: list[Retrieved]) -> str:
        return "\n".join(f"- ({c.metadata.get('section','?')}) {c.text}" for c in chunks)

    topic_line = f"TOPIC TO COMPARE: {topic}\n\n" if topic else ""
    return (
        f"{topic_line}"
        f"DOCUMENT A = {name_a}\n{fmt(chunks_a)}\n\n"
        f"DOCUMENT B = {name_b}\n{fmt(chunks_b)}\n\n"
        f"Do these two documents conflict? Respond in the required JSON format."
    )
