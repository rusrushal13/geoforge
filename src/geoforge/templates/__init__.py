"""Deterministic geometry code templates.

When a component type has a registered template, the pipeline uses it
instead of LLM stage-2 code generation. This gives pixel-exact,
reproducible geometry for well-known structure families.
"""

from geoforge.templates.registry import generate_from_template, has_template, list_templates

__all__ = ["generate_from_template", "has_template", "list_templates"]
