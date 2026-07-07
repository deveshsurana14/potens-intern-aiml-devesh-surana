"""
FastAPI service.

Endpoints:
  GET  /health                 -> liveness + whether the store is populated
  GET  /sources                -> list of ingested document filenames
  POST /ask        {question}  -> grounded answer + citations + confidence
  POST /contradict {a,b,topic} -> conflict verdict between two documents

The RAG system is built once at startup. Set OFFLINE=1 to run with the hashing
embedder (smoke tests / no network); otherwise real embeddings + Groq are used.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.llm import LLMNotConfigured
from rag.pipeline import build_system

OFFLINE = os.getenv("OFFLINE", "0") == "1"
_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _system
    _system = build_system(offline=OFFLINE)
    yield


app = FastAPI(title="Agri Micro-Irrigation RAG", version="1.0.0", lifespan=lifespan)


# ---- request/response models ----------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=[
        "What subsidy do small farmers get in Maharashtra?"])


class ContradictRequest(BaseModel):
    document_a: str = Field(..., examples=["02_maharashtra_drip_subsidy_scheme.md"])
    document_b: str = Field(..., examples=["03_gujarat_drip_subsidy_scheme.md"])
    topic: str = Field("", examples=["effective subsidy percentage"])


# ---- endpoints -------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "offline_mode": OFFLINE,
            "vectors": _system.store.count()}


@app.get("/sources")
def sources():
    return {"sources": _system.store.list_sources()}


@app.post("/ask")
def ask(req: AskRequest):
    if _system.store.count() == 0:
        raise HTTPException(503, "No documents ingested. Run: python -m scripts.ingest")
    try:
        ans = _system.qa.ask(req.question)
    except LLMNotConfigured as e:
        raise HTTPException(503, str(e))
    return {
        "question": ans.question,
        "answer": ans.answer,
        "answered": ans.answered,
        "language": ans.language,
        "confidence": ans.confidence,
        "needs_human_review": ans.needs_human_review,
        "citations": [c.__dict__ for c in ans.citations],
        "confidence_signals": ans.signals,
        "retrieved": ans.retrieved,
    }


@app.post("/contradict")
def contradict(req: ContradictRequest):
    if _system.store.count() == 0:
        raise HTTPException(503, "No documents ingested. Run: python -m scripts.ingest")
    try:
        res = _system.contradict.compare(req.document_a, req.document_b, req.topic)
    except LLMNotConfigured as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return res.__dict__


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
