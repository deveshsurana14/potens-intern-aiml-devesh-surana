"""
Ingest the document corpus into ChromaDB.

Usage:
    python -m scripts.ingest            # real embeddings (downloads model once)
    python -m scripts.ingest --offline  # hashing embedder, no network (for smoke tests)
    python -m scripts.ingest --reset    # drop and rebuild the collection
"""
from __future__ import annotations

import argparse
import time

from rag.chunking import chunk_corpus
from rag.config import DOCS_DIR
from rag.pipeline import build_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store.")
    parser.add_argument("--offline", action="store_true",
                        help="Use the dependency-free hashing embedder.")
    parser.add_argument("--reset", action="store_true",
                        help="Reset the collection before ingesting.")
    args = parser.parse_args()

    print(f"Chunking corpus in {DOCS_DIR} ...")
    chunks = chunk_corpus(DOCS_DIR)
    print(f"  -> {len(chunks)} chunks from "
          f"{len({c.source_file for c in chunks})} documents")

    system = build_system(offline=args.offline)
    if args.reset:
        print("Resetting collection ...")
        system.store.reset()

    t0 = time.time()
    print(f"Embedding + storing ({'hashing' if args.offline else 'sentence-transformers'}) ...")
    n = system.store.add_chunks(chunks)
    print(f"  -> stored {n} chunks in {time.time() - t0:.1f}s")
    print(f"Collection now holds {system.store.count()} vectors.")
    print("Sources:", ", ".join(system.store.list_sources()))
    print("Done.")


if __name__ == "__main__":
    main()
