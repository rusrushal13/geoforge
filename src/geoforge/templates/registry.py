"""Template registry — maps component types to deterministic code generators."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from geoforge.llm.base import GeometrySpec

# Registry: component_type → generator function
_TEMPLATES: dict[str, Callable[[GeometrySpec], str]] = {}


def register_template(component_type: str):
    """Decorator to register a template generator for a component type."""

    def decorator(fn: Callable[[GeometrySpec], str]):
        _TEMPLATES[component_type] = fn
        return fn

    return decorator


def has_template(component_type: str) -> bool:
    """Check if a deterministic template exists for this component type."""
    _ensure_loaded()
    return component_type in _TEMPLATES


def list_templates() -> list[str]:
    """Return all registered template component types."""
    _ensure_loaded()
    return list(_TEMPLATES.keys())


def generate_from_template(spec: GeometrySpec) -> str:
    """Generate GDSFactory code from a deterministic template.

    Args:
        spec: GeometrySpec with component_type and parameters populated.

    Returns:
        Complete, runnable GDSFactory Python code string.

    Raises:
        KeyError: If no template is registered for the component type.
    """
    _ensure_loaded()
    if spec.component_type not in _TEMPLATES:
        raise KeyError(
            f"No template registered for component type '{spec.component_type}'. "
            f"Available: {list(_TEMPLATES.keys())}"
        )
    return _TEMPLATES[spec.component_type](spec)


_loaded = False


def _ensure_loaded():
    """Lazy-load template generators to avoid circular imports."""
    global _loaded  # noqa: PLW0603
    if not _loaded:
        import geoforge.templates.generators  # noqa: F401

        _loaded = True
