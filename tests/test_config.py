"""Tests for configuration module."""

from geoforge.config import Settings


def test_settings_defaults():
    """Test default settings values."""
    s = Settings()
    assert s.default_llm_provider == "ollama"  # Ollama is default (free, offline)
    assert s.ollama_host == "http://localhost:11434"


def test_available_providers_includes_ollama():
    """Test that Ollama is always available."""
    s = Settings()
    providers = s.get_available_providers()
    assert "ollama" in providers
