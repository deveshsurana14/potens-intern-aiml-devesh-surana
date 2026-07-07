"""
LLM client (Groq).

Groq serves open models (Llama 3.3 70B here) very fast on a free tier, which
makes the Streamlit demo feel instant. The client is a thin wrapper: it takes a
system + user prompt and returns text. If no API key is configured it raises a
clear, actionable error rather than failing deep inside a request handler.
"""
from __future__ import annotations

from .config import settings


class LLMNotConfigured(RuntimeError):
    pass


class GroqLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.groq_api_key
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self.api_key:
            raise LLMNotConfigured(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://console.groq.com/keys"
            )
        from groq import Groq

        self._client = Groq(api_key=self.api_key)

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 max_tokens: int = 900) -> str:
        self._ensure_client()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
