"""
Confidence scoring + human-in-the-loop gate (stretch goal #1).

We do not have model logprobs from the Groq chat API, so confidence is computed
from signals we DO have and can defend:

  * retrieval_strength : the top reranked relevance score. If the best evidence
    is weak, we should not be confident regardless of how fluent the answer is.
  * evidence_margin    : how much better the top hit is than the runner-up. A big
    margin means one chunk clearly answers the question; a flat distribution
    across many weak chunks is a warning sign.
  * citation_coverage  : did the model actually cite the context? An answer with
    no citations, or that hit the INSUFFICIENT sentinel, is treated as unsupported.

These are combined into a single [0,1] confidence. Below `low_confidence_threshold`
the answer is flagged `needs_human_review = True` so a person can gate it before
it is trusted — which is the right posture for a product used in audited, high-
responsibility settings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .vectorstore import Retrieved

_CITE_RE = re.compile(r"\[\d+\]")


@dataclass
class Confidence:
    score: float
    needs_human_review: bool
    signals: dict


def compute_confidence(answer: str, contexts: list[Retrieved],
                       *, answered: bool) -> Confidence:
    scores = [c.score for c in contexts] or [0.0]
    retrieval_strength = scores[0]
    evidence_margin = (scores[0] - scores[1]) if len(scores) > 1 else scores[0]
    evidence_margin = max(0.0, min(evidence_margin, 1.0))

    citations = len(set(_CITE_RE.findall(answer)))
    citation_coverage = 1.0 if citations >= 1 else 0.0
    if not answered:
        citation_coverage = 0.0

    # Weighted blend; retrieval strength dominates because a fluent answer over
    # weak evidence is exactly the failure mode we want to catch.
    score = (
        0.55 * retrieval_strength
        + 0.20 * evidence_margin
        + 0.25 * citation_coverage
    )
    score = round(max(0.0, min(score, 1.0)), 3)

    needs_review = (not answered) or score < settings.low_confidence_threshold
    return Confidence(
        score=score,
        needs_human_review=needs_review,
        signals={
            "retrieval_strength": round(retrieval_strength, 3),
            "evidence_margin": round(evidence_margin, 3),
            "citation_coverage": citation_coverage,
            "num_citations": citations,
            "threshold": settings.low_confidence_threshold,
        },
    )
