"""LLM provider registry and factory."""

from typing import TYPE_CHECKING

from geoforge.config import settings

if TYPE_CHECKING:
    from geoforge.llm.base import LLMProvider

# Registry of available providers
_PROVIDERS: dict[str, type["LLMProvider"]] = {}


def register_provider(name: str):
    """Decorator to register an LLM provider."""

    def decorator(cls: type["LLMProvider"]):
        _PROVIDERS[name] = cls
        return cls

    return decorator


def get_provider(name: str | None = None, **kwargs) -> "LLMProvider":
    """Get an LLM provider instance by name.

    Args:
        name: Provider name (defaults to settings.default_llm_provider)
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider not found or not configured
    """
    name = name or settings.default_llm_provider

    # Lazy import providers to avoid circular imports
    if not _PROVIDERS:
        _load_providers()

    if name not in _PROVIDERS:
        available = list(_PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    return _PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    """List all registered provider names."""
    if not _PROVIDERS:
        _load_providers()
    return list(_PROVIDERS.keys())


def _load_providers():
    """Load all provider implementations."""
    # Import providers to trigger registration
    from geoforge.llm.anthropic import AnthropicProvider
    from geoforge.llm.google import GeminiProvider
    from geoforge.llm.ollama import OllamaProvider
    from geoforge.llm.openai import OpenAIProvider

    _PROVIDERS["ollama"] = OllamaProvider
    _PROVIDERS["gemini"] = GeminiProvider
    _PROVIDERS["openai"] = OpenAIProvider
    _PROVIDERS["anthropic"] = AnthropicProvider
