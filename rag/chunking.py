"""
Structure-aware chunking.

Why not fixed-size chunking?
----------------------------
Naive fixed-window chunking splits blindly across the text and routinely cuts a
sentence — or a subsidy figure and the category it belongs to — in half. For a
citation-first RAG system that is a real problem: the retrieved chunk needs to be
a self-contained, quotable unit, and the citation needs to point at a meaningful
location ("Section 2. Subsidy pattern") not "characters 1400-1720".

Strategy (two passes):
  1. Split each document on its markdown headers so every chunk inherits the
     SECTION it came from. Policy documents are already written in numbered
     sections; we respect that structure.
  2. Within a section, pack whole SENTENCES into chunks up to a target size,
     carrying a sentence-level overlap between consecutive chunks so a fact that
     straddles a boundary still appears intact in at least one chunk.

Every chunk carries metadata: source_file, doc_id, section, chunk_index. This is
what makes precise citations (file + section + snippet) possible downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import settings


@dataclass
class Chunk:
    id: str            # stable unique id, e.g. "03_gujarat...::s2::c0"
    text: str
    source_file: str   # filename, shown in citations
    doc_id: str        # the "Document ID:" declared inside the file
    section: str       # human-readable section heading
    chunk_index: int   # position within the document

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_DOC_ID_RE = re.compile(r"Document ID:\s*(.+)", re.IGNORECASE)
# Sentence splitter that keeps the terminator and is tolerant of "Rs." / "e.g.".
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _extract_doc_id(text: str, fallback: str) -> str:
    m = _DOC_ID_RE.search(text)
    return m.group(1).strip() if m else fallback


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return a list of (section_heading, section_body) tuples."""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [("Document", text.strip())]

    # Any preamble before the first header.
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        if pre:
            sections.append(("Preamble", pre))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def _sentences(text: str) -> list[str]:
    # Collapse whitespace so chunks read cleanly, then split on sentence ends.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _pack_sentences(sentences: list[str]) -> list[str]:
    """Greedily pack sentences into chunks with a trailing-sentence overlap."""
    target = settings.chunk_target_chars
    overlap = settings.chunk_overlap_chars
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        # A single very long sentence becomes its own chunk.
        if current and current_len + len(sent) + 1 > target:
            chunks.append(" ".join(current))
            # Build overlap from the tail of the chunk we just closed.
            tail: list[str] = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s)
            current = tail
            current_len = tail_len
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(path: Path) -> list[Chunk]:
    """Chunk a single document file into a list of Chunk objects."""
    raw = path.read_text(encoding="utf-8")
    doc_id = _extract_doc_id(raw, fallback=path.stem)
    chunks: list[Chunk] = []
    running_index = 0

    for sec_idx, (heading, body) in enumerate(_split_into_sections(raw)):
        for piece_idx, piece in enumerate(_pack_sentences(_sentences(body))):
            chunk_id = f"{path.stem}::s{sec_idx}::c{piece_idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source_file=path.name,
                    doc_id=doc_id,
                    section=heading,
                    chunk_index=running_index,
                )
            )
            running_index += 1
    return chunks


def chunk_corpus(docs_dir: Path) -> list[Chunk]:
    """Chunk every .md / .txt file in the docs directory."""
    files = sorted(
        p for p in docs_dir.iterdir() if p.suffix.lower() in {".md", ".txt"}
    )
    all_chunks: list[Chunk] = []
    for f in files:
        all_chunks.extend(chunk_document(f))
    return all_chunks
