"""Sentence-Transformers embedding adapter implementing IEmbedder.

Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).
Can be swapped to any sentence-transformers compatible model via config.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from sentence_transformers import SentenceTransformer

from talm.core.interfaces import IEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(IEmbedder):
    """Local embedding model via sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading sentence-transformers model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self._dimension)

    def get_dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string. Runs in executor to avoid blocking."""
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(
            None, partial(self._model.encode, text, normalize_embeddings=True)
        )
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single batch call."""
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            partial(self._model.encode, texts, normalize_embeddings=True),
        )
        return vectors.tolist()
