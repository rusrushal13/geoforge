"""Google Gemini LLM provider."""

import json

from google import genai
from google.genai import types

from geoforge.config import settings
from geoforge.llm.base import (
    GEOMETRY_SPEC_SCHEMA,
    PROMPT_ENHANCEMENT_SCHEMA,
    GeometrySpec,
    LLMProvider,
    RetryContext,
    _format_prompt_enhancement_retry_message,
    _format_retry_message,
)
from geoforge.prompts.gdsfactory import (
    GEOMETRY_SPEC_PROMPT,
    PROMPT_ENHANCER_PROMPT,
    build_code_prompt,
)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize Gemini provider.

        Args:
            model: Model name (defaults to settings.gemini_model)
            api_key: API key (defaults to settings.gemini_api_key)
        """
        self.model = model or settings.gemini_model
        self.api_key = api_key or settings.gemini_api_key

        if not self.api_key:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env")

        self.client = genai.Client(api_key=self.api_key)

    async def _enhance_prompt_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Rewrite raw user prompt into explicit geometry constraints."""
        system_prompt = PROMPT_ENHANCER_PROMPT + PROMPT_ENHANCEMENT_SCHEMA
        contents = f"{system_prompt}\n\nUser request: {prompt}"

        if retry_context:
            correction = _format_prompt_enhancement_retry_message(retry_context)
            contents += f"\n\n{correction}"

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=settings.llm_temperature,
                seed=settings.llm_seed,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned empty response for prompt enhancement")

        return json.loads(response.text)

    async def _generate_geometry_spec_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Convert natural language to geometry specification using Gemini."""
        system_prompt = GEOMETRY_SPEC_PROMPT + GEOMETRY_SPEC_SCHEMA

        contents = f"{system_prompt}\n\nUser request: {prompt}"

        # On retry, append error feedback
        if retry_context:
            correction = _format_retry_message(retry_context)
            contents += f"\n\n{correction}"

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=settings.llm_temperature,
                seed=settings.llm_seed,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned empty response")

        return json.loads(response.text)

    async def _generate_gdsfactory_code_impl(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
        retry_context: RetryContext | None = None,
    ) -> str:
        """Generate GDSFactory code from specification using Gemini."""
        prompt = build_code_prompt(
            {
                "component_type": spec.component_type,
                "description": spec.description,
                "parameters": spec.parameters,
                "layers": [layer.model_dump() for layer in spec.layers],
                "geometry_ranges": [r.model_dump() for r in spec.geometry_ranges],
                "primitives": [p.model_dump() for p in spec.primitives],
            },
            original_prompt=original_prompt,
        )

        contents = prompt

        # On retry, append error feedback
        if retry_context:
            correction = _format_retry_message(retry_context)
            contents += f"\n\n{correction}"

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=settings.llm_temperature,
                seed=settings.llm_seed,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned empty response")

        return response.text
