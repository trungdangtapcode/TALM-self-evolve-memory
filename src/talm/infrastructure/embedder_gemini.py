"""Gemini embedding adapter implementing IEmbedder.

Uses the new google-genai SDK. Requires GOOGLE_API_KEY environment variable.
"""

from __future__ import annotations

import logging

from google import genai

from talm.core.interfaces import IEmbedder

logger = logging.getLogger(__name__)

_GEMINI_EMBED_DIM = 768


class GeminiEmbedder(IEmbedder):
    """Embedding via Google Gemini's text-embedding-004 model."""

    def __init__(self, model_name: str = "text-embedding-004") -> None:
        self._model_name = model_name
        self._client = genai.Client()

    def get_dimension(self) -> int:
        return _GEMINI_EMBED_DIM

    async def embed(self, text: str) -> list[float]:
        result = await self._client.aio.models.embed_content(
            model=self._model_name,
            contents=text,
        )
        return list(result.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            results.append(await self.embed(text))
        return results
