"""
Multilingual boundary.

Requirement: a query in one language must return an answer in the same language.
The brief allows a translation step at the boundary for the 24-hour version, so
that is exactly what we do:

    query (any language)
        -> detect language
        -> translate query to English   (retrieval + reasoning happen in English)
        -> answer in English with citations
        -> translate the answer back to the original language
        (citations/snippets stay in the source language of the documents)

Keeping retrieval in English means we index and search one language, which is
simpler and more reliable than multilingual embeddings for a small corpus. The
same LLM does the translation, so no extra dependency is needed.
"""
from __future__ import annotations

import re

from .llm import GroqLLM

_LATIN_RE = re.compile(r"[A-Za-z]")


def looks_english(text: str) -> bool:
    """Cheap heuristic: mostly-Latin script and no Devanagari => treat as English.

    Avoids a network/LLM call for the common English case. Non-Latin scripts
    (Devanagari, etc.) are routed through the LLM detector/translator.
    """
    if re.search(r"[\u0900-\u097F]", text):  # Devanagari (Hindi/Marathi)
        return False
    latin = len(_LATIN_RE.findall(text))
    return latin >= max(3, int(0.5 * len(text.replace(" ", ""))))


class Translator:
    def __init__(self, llm: GroqLLM):
        self.llm = llm

    def detect_language(self, text: str) -> str:
        if looks_english(text):
            return "English"
        out = self.llm.complete(
            system="You identify languages. Reply with only the English name of "
                   "the language of the user's text (e.g. 'Hindi', 'Marathi', "
                   "'English'). One word only.",
            user=text,
            max_tokens=5,
        )
        return out.strip().splitlines()[0].strip(" .")

    def to_english(self, text: str, source_language: str) -> str:
        if source_language.lower() == "english":
            return text
        return self.llm.complete(
            system=f"Translate the user's {source_language} text into English. "
                   f"Return only the translation, no notes.",
            user=text,
        )

    def from_english(self, text: str, target_language: str) -> str:
        if target_language.lower() == "english":
            return text
        return self.llm.complete(
            system=f"Translate the user's English text into {target_language}. "
                   f"Preserve any bracketed citation markers like [1], [2] exactly. "
                   f"Return only the translation.",
            user=text,
        )
