"""Gemini embedding adapter implementing IEmbedder.

Uses Google's text-embedding-004 model via google-generativeai SDK.
Requires GOOGLE_API_KEY environment variable.
"""

from __future__ import annotations

import logging

import google.generativeai as genai

from talm.core.interfaces import IEmbedder

logger = logging.getLogger(__name__)

# Gemini text-embedding-004 produces 768-dim vectors
_GEMINI_EMBED_DIM = 768


class GeminiEmbedder(IEmbedder):
    """Embedding via Google Gemini's text-embedding-004 model."""

    def __init__(self, model_name: str = "models/text-embedding-004") -> None:
        self._model_name = model_name

    def get_dimension(self) -> int:
        return _GEMINI_EMBED_DIM

    async def embed(self, text: str) -> list[float]:
        result = await genai.embed_content_async(
            model=self._model_name,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            results.append(await self.embed(text))
        return results
