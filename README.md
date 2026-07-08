# Micro-Irrigation Policy — Document Q&A with Citations

A citation-first RAG system over six real Indian **drip-irrigation subsidy and
technical documents** (PMKSY–Per Drop More Crop, the Maharashtra and Gujarat
state schemes, a NABARD adoption study, a drip engineering guide, and a DBT /
grievance circular). It answers questions with precise citations, refuses to
answer when the documents don't cover the question, detects genuine
contradictions between two documents, works across English / Hindi / Marathi,
and flags low-confidence answers for human review.

> Built for the Potens AI/ML take-home — **Q1: Document Q&A with Citations.**

## Why this domain

The corpus is deliberately chosen so the hard parts of the brief are *real*, not
decorative. Indian micro-irrigation policy has genuine, checkable conflicts
across jurisdictions — Maharashtra's effective subsidy (80% / 75%) versus
Gujarat's (70% / 60%), a 5 ha versus 10 ha area ceiling, Direct-Benefit-Transfer
versus supplier-credit disbursement, a 7-year versus 5-year cooling-off period.
That gives the `/contradict` endpoint substance and gives the no-hallucination
guard something to actually guard against.

## Architecture

```
                 query (EN / HI / MR)
                        │
              ┌─────────▼──────────┐
              │  language boundary │  detect → translate to EN for retrieval
              └─────────┬──────────┘
                        │
        ┌───────────────▼────────────────┐
        │  Retriever                      │
        │   ChromaDB vector search (k=12) │  all-MiniLM-L6-v2 embeddings
        │        → cross-encoder rerank   │  ms-marco-MiniLM-L-6-v2  (→ top 4)
        └───────────────┬────────────────┘
                        │
              ┌─────────▼──────────┐
              │ no-hallucination   │  (1) refuse if best relevance < threshold
              │      gates         │  (2) refuse if model emits INSUFFICIENT
              └─────────┬──────────┘
                        │
        ┌───────────────▼────────────────┐
        │  Groq · Llama 3.3 70B           │  grounding-first prompt, bracket cites
        └───────────────┬────────────────┘
                        │
      ┌─────────────────▼──────────────────┐
      │ citations (file + section + snippet)│
      │ confidence score → human-review flag│  translate answer back to query lang
      └─────────────────────────────────────┘
```

Everything is wired through one factory (`rag/pipeline.py`) so the API, the
Streamlit UI, and the tests build the identical system.

## Quick start

> **Windows note:** the native dependencies (Chroma, onnxruntime, PyTorch) need
> the Microsoft Visual C++ Redistributable (x64). If you hit a `DLL load failed`
> or an access-violation on import, install it from
> https://aka.ms/vs/17/release/vc_redist.x64.exe and reopen the terminal.

```bash
# 1. install
pip install -r requirements.txt          # or: make install

# 2. add your (free) Groq key
cp .env.example .env                      # then edit .env, paste your key
#    get one at https://console.groq.com/keys

# 3. ingest the corpus (downloads the embedding model once, ~90 MB)
python -m scripts.ingest --reset          # or: make ingest

# 4a. try it in the browser
streamlit run app.py                      # or: make ui   → http://localhost:8501

# 4b. …or run the API
uvicorn api.main:app --reload             # or: make api  → http://localhost:8000/docs
```

Example API calls:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What subsidy do small farmers get in Maharashtra?"}' | jq

curl -s localhost:8000/contradict -H 'content-type: application/json' \
  -d '{"document_a":"02_maharashtra_drip_subsidy_scheme.md",
       "document_b":"03_gujarat_drip_subsidy_scheme.md",
       "topic":"effective subsidy percentage"}' | jq
```

## Tests & evaluation (no API key needed)

```bash
pytest -q                 # 10 tests, fully offline (mocked LLM, hashing embedder)
python -m eval.run_eval   # retrieval Hit@k + MRR over 10 ground-truth Q&A pairs
```

The test suite and the offline modes use a dependency-free **hashing embedder** so
the whole retrieval pipeline runs with no network and no model weights. That's
also why `--offline` eval numbers are only a wiring check — real
`all-MiniLM-L6-v2` embeddings score materially higher.

Retrieval eval with the real `all-MiniLM-L6-v2` embeddings + cross-encoder
reranker, over the 10 ground-truth Q&A pairs in `eval/eval_set.json`:

| Metric | Score |
|--------|-------|
| Hit@1  | 90% (9/10) |
| Hit@3  | 100% (10/10) |
| Hit@5  | 100% (10/10) |
| MRR    | 0.933 |

The one question that misses Hit@1 still lands the correct document at rank 3, so
the gold source is always inside the top-k the answer is grounded on.

## Design decisions

**Structure-aware chunking (`rag/chunking.py`).** Fixed-size windows routinely cut
a subsidy figure away from the category it belongs to, which is fatal for
citations. Instead I split on markdown headers first (so every chunk inherits a
human-readable *section*), then pack whole sentences up to ~320 tokens with a
60-token sentence overlap so a fact that straddles a boundary still survives
intact in at least one chunk. Every chunk carries `source_file · section ·
chunk_id`, which is exactly what a precise citation needs.

**Two-layer no-hallucination guard (`rag/qa.py`).** (1) A *retrieval-side* gate: if
the best reranked relevance score is below a threshold, the system refuses and
says the documents don't cover the question — before the LLM is even called.
(2) A *model-side* gate: the prompt instructs the model to emit a sentinel
(`INSUFFICIENT_CONTEXT`) when the context is inadequate, which we detect and turn
into the same honest refusal. Off-topic questions return "not covered," not a
confident fabrication.

**Citations that point somewhere (`rag/prompts.py`, `rag/qa.py`).** Context blocks are
numbered `[1]…[n]`; the model cites those numbers; we map each used marker back to
its `source_file`, `section`, `chunk_id`, and the exact snippet. Only chunks the
model actually referenced become citations.

**Cross-encoder reranker (stretch — `rag/retriever.py`).** Bi-encoder vector search is
fast but coarse. We over-fetch 12 candidates and rerank with
`ms-marco-MiniLM-L-6-v2`, which reads each (query, chunk) pair jointly, then keep
the top 4. Degrades gracefully to vector order if the model can't load.

**Confidence + human-in-the-loop (stretch — `rag/confidence.py`).** The Groq chat API
gives no logprobs, so confidence is built from signals we *can* defend: retrieval
strength (top score), evidence margin (gap to the runner-up), and citation
coverage. Below the threshold, `needs_human_review` is set and surfaced in both
the API and UI — the right posture for a product used where a careless answer
costs someone their day.

**Multilingual at the boundary (stretch/required — `rag/translate.py`).** A query in
Hindi or Marathi is detected, translated to English for retrieval and reasoning,
answered in English with citations, then the answer is translated back to the
query language. Retrieval stays single-language (simpler, more reliable for a
small corpus), and the same Groq model does the translation, so no extra
dependency.

## Requirements checklist

Required:
- [x] Ingest, chunk, embed, store — `scripts/ingest.py`, `rag/chunking.py`, `rag/vectorstore.py`; chunking strategy documented above.
- [x] `/ask` returns answers **with citations** (source file + section/chunk ref + snippet) — `api/main.py`, `rag/qa.py`.
- [x] `/contradict` takes two document IDs and returns whether they conflict, with reasoning — `rag/contradict.py`.
- [x] Multilingual flow — query language in, same language out — `rag/translate.py`.
- [x] Streamlit UI — `app.py`.
- [x] **No silent hallucination** — two-layer guard, explicit refusal.
- [x] Chroma vector store + Groq (free tier) LLM.

Stretch (all three attempted):
- [x] Confidence score per answer + human-in-the-loop gate below threshold.
- [x] Reranker layered on vector retrieval (cross-encoder).
- [x] Eval set of 10 Q&A pairs with ground truth, scored on retrieval@k (+ MRR).

## What's unfinished / would do next

- **Answer-faithfulness check.** The guard prevents *answering without evidence*,
  but doesn't yet verify every generated sentence is entailed by its cited chunk.
  Next: a lightweight NLI / self-check pass that flags any sentence not supported
  by its citation.
- **True multilingual retrieval.** The boundary-translation approach is fine at
  this scale; for larger corpora I'd index with a multilingual embedding model and
  retrieve natively to avoid a translation hop.
- **Contradiction across the whole corpus.** `/contradict` compares a chosen pair;
  a nice extension is scanning all pairs for a topic and surfacing every conflict.
- **Streaming responses** in the UI, and caching of repeated queries.
- **Larger eval** with answer-quality (not just retrieval) scoring.

## Project layout

```
docs/                     6 source documents (the corpus)
rag/
  chunking.py             structure-aware chunking
  embeddings.py           embedder interface + ST impl + offline hashing fake
  vectorstore.py          ChromaDB wrapper
  retriever.py            vector search + cross-encoder reranker
  llm.py / prompts.py     Groq client + grounding-first prompts
  qa.py                   /ask flow (gates, citations, confidence, multilingual)
  contradict.py           /contradict flow
  confidence.py           confidence scoring + human-review gate
  translate.py            multilingual boundary
  pipeline.py             one factory that assembles everything
api/main.py               FastAPI service
app.py                    Streamlit UI
scripts/ingest.py         corpus ingestion CLI
eval/                     10-item ground-truth set + retrieval@k / MRR runner
tests/                    offline pytest suite (10 tests)
```

---

## AI USE LOG

Per the brief, an honest account of the AI assistance I used.

I built this the way I think a working engineer builds in 2026: I used Claude as
a coding partner and drove it with a small number of focused, high-context
prompts rather than writing every line by hand. What I own is the direction and
the judgment — picking Q1, choosing the micro-irrigation domain precisely because
it has real, checkable contradictions to test the no-hallucination guard, and
then reviewing, running, and validating everything that came back.

| Tool | Approx. usage | What it was used for |
|------|---------------|----------------------|
| Claude (Anthropic, Opus 4.8) | ~7 messages (large code-generation replies, roughly ~20–30k tokens total, rough) | Generated the project from focused prompts: architecture, the structure-aware chunking, the two-layer no-hallucination guard, the reranker / confidence / multilingual modules, the FastAPI service, the Streamlit UI, the tests, and the eval harness. I set the framing and constraints in each prompt, then read, ran, and adjusted the output. |
| Claude Code | a short session | Organized the git commit history and helped finalize this README. |

I did not use any other AI tool. The engineering decisions I stand behind —
domain and corpus choice, the contradiction pairs, and the confidence signals —
were mine; every generated file was run and reviewed before it went in.
