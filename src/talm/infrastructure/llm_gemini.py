"""Google Gemini adapter implementing ILLMClient.

Uses the google-generativeai SDK.  Reads credentials from environment
variables: GOOGLE_API_KEY (or GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION
for Vertex AI).
"""

from __future__ import annotations

import logging

import google.generativeai as genai

from talm.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)


class GeminiLLMClient(ILLMClient):
    """Concrete LLM client backed by Google Gemini."""

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self._model_name = model_name

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> str:
        """Call Gemini to generate a completion."""
        config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        model = genai.GenerativeModel(
            self._model_name,
            system_instruction=system_prompt,
            generation_config=config,
        )
        response = await model.generate_content_async(user_prompt)
        return response.text
