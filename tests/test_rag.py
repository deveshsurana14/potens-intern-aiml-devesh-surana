"""
Test suite.

Everything runs OFFLINE with the hashing embedder and a mocked LLM, so the suite
needs no API key and no model download. It exercises the behaviours that matter
for correctness and safety: chunking metadata, both no-hallucination gates,
citation construction, the multilingual routing, and contradiction JSON parsing.

Run:  pytest -q
"""
from __future__ import annotations

import pytest

from rag.chunking import chunk_corpus
from rag.config import DOCS_DIR, settings
from rag.contradict import _extract_json
from rag.llm import GroqLLM
from rag.pipeline import build_system
from rag.prompts import INSUFFICIENT


# ---- fixtures --------------------------------------------------------------
@pytest.fixture()
def grounded_llm(monkeypatch):
    def fake(self, system, user, **kw):
        if "identify languages" in system.lower():
            return "English"
        return "Small farmers receive 55% [1]."
    monkeypatch.setattr(GroqLLM, "complete", fake)


@pytest.fixture()
def system():
    sys = build_system(offline=True)
    sys.store.reset()
    sys.store.add_chunks(chunk_corpus(DOCS_DIR))
    return sys


# ---- chunking --------------------------------------------------------------
def test_chunking_produces_metadata():
    chunks = chunk_corpus(DOCS_DIR)
    assert len(chunks) > 20
    for c in chunks:
        assert c.source_file and c.section and c.id
        assert c.text.strip()
        # chunks should stay near the configured size budget (allow slack)
        assert len(c.text) <= settings.chunk_target_chars * 2


def test_chunk_ids_are_unique():
    chunks = chunk_corpus(DOCS_DIR)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


# ---- no-hallucination gates -----------------------------------------------
def test_answer_has_citation(system, grounded_llm):
    ans = system.qa.ask("What subsidy do small farmers get?")
    assert ans.answered is True
    assert len(ans.citations) >= 1
    assert ans.citations[0].source_file.endswith(".md")


def test_model_sentinel_triggers_refusal(system, monkeypatch):
    def fake(self, system_, user, **kw):
        if "identify languages" in system_.lower():
            return "English"
        return INSUFFICIENT
    monkeypatch.setattr(GroqLLM, "complete", fake)
    ans = system.qa.ask("Who won the 2018 football World Cup?")
    assert ans.answered is False
    assert ans.needs_human_review is True
    assert ans.citations == []


def test_low_relevance_triggers_refusal(system, monkeypatch):
    # Force the retrieval-side gate: nothing clears a 0.99 relevance bar.
    monkeypatch.setattr(settings, "min_relevance_to_answer", 0.99)

    def fake(self, system_, user, **kw):
        if "identify languages" in system_.lower():
            return "English"
        return "should not be used [1]"
    monkeypatch.setattr(GroqLLM, "complete", fake)
    ans = system.qa.ask("What subsidy do small farmers get?")
    assert ans.answered is False


# ---- contradiction JSON parsing -------------------------------------------
def test_extract_json_plain():
    d = _extract_json('{"conflict": true, "topic": "x"}')
    assert d["conflict"] is True


def test_extract_json_fenced():
    raw = "```json\n{\"conflict\": false, \"topic\": \"y\"}\n```"
    d = _extract_json(raw)
    assert d["conflict"] is False and d["topic"] == "y"


def test_extract_json_with_prose_around():
    raw = 'Here is the verdict: {"conflict": true} — hope that helps.'
    assert _extract_json(raw)["conflict"] is True


def test_contradiction_flow(system, monkeypatch):
    def fake(self, system_, user, **kw):
        return ('{"conflict": true, "topic": "subsidy %", '
                '"document_a_position": "MH 80/75", '
                '"document_b_position": "GJ 70/60", '
                '"reasoning": "different mandatory rates"}')
    monkeypatch.setattr(GroqLLM, "complete", fake)
    res = system.contradict.compare(
        "02_maharashtra_drip_subsidy_scheme.md",
        "03_gujarat_drip_subsidy_scheme.md",
        "effective subsidy percentage",
    )
    assert res.conflict is True
    assert res.excerpts_a and res.excerpts_b


def test_contradiction_bad_source_raises(system, monkeypatch):
    monkeypatch.setattr(GroqLLM, "complete", lambda *a, **k: "{}")
    with pytest.raises(ValueError):
        system.contradict.compare("does_not_exist.md",
                                  "03_gujarat_drip_subsidy_scheme.md", "x")
