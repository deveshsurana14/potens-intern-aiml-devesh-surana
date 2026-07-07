"""
Retrieval evaluation (stretch goal #3).

For each eval question we retrieve chunks and check whether a GOLD source document
appears in the top-k. We report:
  * Hit@k for k in {1, 3, 5} : fraction of questions whose gold doc was retrieved
                               within the top-k chunks.
  * MRR                      : mean reciprocal rank of the first gold-source chunk.

Run:
    python -m eval.run_eval              # real embeddings (recommended)
    python -m eval.run_eval --offline    # hashing embedder (wiring check only;
                                         #   scores will be poor by design)

Note: this evaluates RETRIEVAL only and needs no LLM/API key — retrieval quality
is the thing that most determines answer quality, and it is cheap to measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import DOCS_DIR, settings
from rag.pipeline import build_system

EVAL_PATH = Path(__file__).parent / "eval_set.json"


def rank_of_first_gold(hits, gold_sources: set[str]) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h.metadata.get("source_file") in gold_sources:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--k", type=int, default=10, help="how many chunks to retrieve")
    args = ap.parse_args()

    data = json.loads(EVAL_PATH.read_text())
    items = data["items"]

    system = build_system(offline=args.offline)
    if system.store.count() == 0:
        raise SystemExit("Vector store is empty. Run: python -m scripts.ingest"
                         + (" --offline" if args.offline else ""))

    ks = [1, 3, 5]
    hits_at = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    rows = []

    for it in items:
        gold = set(it["gold_sources"])
        # Retrieve top-k chunks (rerank included if enabled).
        retrieved = system.retriever.retrieve(it["question"], top_k=args.k, top_n=args.k)
        rank = rank_of_first_gold(retrieved, gold)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        top_src = retrieved[0].metadata.get("source_file") if retrieved else "-"
        rows.append((it["id"], "✓" if rank and rank <= 3 else "✗",
                     rank if rank else "miss", top_src))

    n = len(items)
    print(f"\nEvaluated {n} questions "
          f"({'hashing/offline' if args.offline else settings.embedding_model}, "
          f"reranker={'on' if settings.use_reranker and not args.offline else 'off'})\n")
    print(f"{'id':<5}{'hit@3':<7}{'rank':<7}{'top source retrieved'}")
    print("-" * 60)
    for rid, ok, rank, src in rows:
        print(f"{rid:<5}{ok:<7}{str(rank):<7}{src}")
    print("-" * 60)
    for k in ks:
        print(f"Hit@{k}: {hits_at[k]}/{n} = {hits_at[k]/n:.0%}")
    print(f"MRR : {sum(reciprocal_ranks)/n:.3f}")


if __name__ == "__main__":
    main()
