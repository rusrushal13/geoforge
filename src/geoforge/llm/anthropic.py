"""Anthropic LLM provider (Claude models)."""

import json
from typing import Any, cast

from anthropic import AsyncAnthropic

from geoforge.config import settings
from geoforge.llm.base import (
    GEOMETRY_SPEC_SCHEMA,
    GeometrySpec,
    LLMProvider,
    RetryContext,
    _format_retry_message,
)
from geoforge.prompts.gdsfactory import GEOMETRY_SPEC_PROMPT, build_code_prompt


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider (Claude)."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initialize Anthropic provider.

        Args:
            model: Model name (defaults to settings.anthropic_model)
            api_key: API key (defaults to settings.anthropic_api_key)
        """
        self.model = model or settings.anthropic_model
        self.api_key = api_key or settings.anthropic_api_key

        if not self.api_key:
            raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY in .env")

        self.client = AsyncAnthropic(api_key=self.api_key)

    async def _generate_geometry_spec_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Convert natural language to geometry specification using Claude."""
        system_prompt = GEOMETRY_SPEC_PROMPT + GEOMETRY_SPEC_SCHEMA

        messages = [{"role": "user", "content": prompt}]

        # On retry, add the error feedback as a multi-turn conversation
        if retry_context:
            if retry_context.previous_response_snippet:
                messages.append(
                    {"role": "assistant", "content": retry_context.previous_response_snippet}
                )
            messages.append({"role": "user", "content": _format_retry_message(retry_context)})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=settings.llm_temperature,
            system=system_prompt,
            messages=cast("Any", messages),
        )

        # Extract text from response (non-text blocks can be returned in strict typing mode)
        content = None
        if response.content:
            first_block = cast("Any", response.content[0])
            content = first_block.text if hasattr(first_block, "text") else None
        if not content:
            raise ValueError("Anthropic returned empty response")

        return json.loads(content)

    async def _generate_gdsfactory_code_impl(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
        retry_context: RetryContext | None = None,
    ) -> str:
        """Generate GDSFactory code from specification using Claude."""
        prompt = build_code_prompt(
            {
                "component_type": spec.component_type,
                "description": spec.description,
                "parameters": spec.parameters,
                "layers": [layer.model_dump() for layer in spec.layers],
                "geometry_ranges": [r.model_dump() for r in spec.geometry_ranges],
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

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=settings.llm_temperature,
            messages=cast("Any", messages),
        )

        content = None
        if response.content:
            first_block = cast("Any", response.content[0])
            content = first_block.text if hasattr(first_block, "text") else None
        if not content:
            raise ValueError("Anthropic returned empty response")

        return content
