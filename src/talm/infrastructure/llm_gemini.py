"""Google Gemini adapter implementing ILLMClient.

Uses the new google-genai SDK (replaces deprecated google-generativeai).
Reads GOOGLE_API_KEY from environment automatically.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

from talm.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)


class GeminiLLMClient(ILLMClient):
    """Concrete LLM client backed by Google Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name
        self._client = genai.Client()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> str:
        """Call Gemini to generate a completion."""
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=config,
        )
        return response.text
