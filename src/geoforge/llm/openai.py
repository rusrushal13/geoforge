"""OpenAI LLM provider (GPT models)."""

import json
from typing import Any, cast

from openai import AsyncOpenAI

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


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider (ChatGPT, GPT-4, etc.)."""

    name = "openai"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize OpenAI provider.

        Args:
            model: Model name (defaults to settings.openai_model)
            api_key: API key (defaults to settings.openai_api_key)
        """
        self.model = model or settings.openai_model
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")

        self.client = AsyncOpenAI(api_key=self.api_key)

    async def _enhance_prompt_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Rewrite raw user prompt into explicit geometry constraints."""
        system_prompt = PROMPT_ENHANCER_PROMPT + PROMPT_ENHANCEMENT_SCHEMA
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if retry_context:
            if retry_context.previous_response_snippet:
                messages.append(
                    {"role": "assistant", "content": retry_context.previous_response_snippet}
                )
            messages.append(
                {
                    "role": "user",
                    "content": _format_prompt_enhancement_retry_message(retry_context),
                }
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast("Any", messages),
            response_format={"type": "json_object"},
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty response for prompt enhancement")

        return json.loads(content)

    async def _generate_geometry_spec_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Convert natural language to geometry specification using OpenAI."""
        system_prompt = GEOMETRY_SPEC_PROMPT + GEOMETRY_SPEC_SCHEMA

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # On retry, add the error feedback as a multi-turn conversation
        if retry_context:
            if retry_context.previous_response_snippet:
                messages.append(
                    {"role": "assistant", "content": retry_context.previous_response_snippet}
                )
            messages.append({"role": "user", "content": _format_retry_message(retry_context)})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast("Any", messages),
            response_format={"type": "json_object"},
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty response")

        return json.loads(content)

    async def _generate_gdsfactory_code_impl(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
        retry_context: RetryContext | None = None,
    ) -> str:
        """Generate GDSFactory code from specification using OpenAI."""
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

        messages = [{"role": "user", "content": prompt}]

        # On retry, add the error feedback
        if retry_context:
            if retry_context.previous_response_snippet:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"```python\n{retry_context.previous_response_snippet}\n```",
                    }
                )
            messages.append({"role": "user", "content": _format_retry_message(retry_context)})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast("Any", messages),
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty response")

        return content
