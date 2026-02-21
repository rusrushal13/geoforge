"""Prompt templates for LLM providers."""

from geoforge.prompts.gdsfactory import (
    COMMON_MISTAKES,
    COMPONENT_EXAMPLES,
    GDSFACTORY_CODE_PROMPT,
    GDSFACTORY_CODE_PROMPT_TEMPLATE,
    GEOMETRY_SPEC_PROMPT,
    PROMPT_ENHANCER_PROMPT,
    build_code_prompt,
)

__all__ = [
    "COMMON_MISTAKES",
    "COMPONENT_EXAMPLES",
    "GDSFACTORY_CODE_PROMPT",
    "GDSFACTORY_CODE_PROMPT_TEMPLATE",
    "GEOMETRY_SPEC_PROMPT",
    "PROMPT_ENHANCER_PROMPT",
    "build_code_prompt",
]
