"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider API Keys
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Default provider (ollama is default - free and works offline)
    default_llm_provider: Literal["gemini", "openai", "anthropic", "ollama"] = Field(
        default="ollama", alias="DEFAULT_LLM_PROVIDER"
    )

    # Model names per provider (from official docs - January 2026)
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    openai_model: str = Field(default="gpt-5.2", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    ollama_model: str = Field(default="qwen3-coder-next", alias="OLLAMA_MODEL")

    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")

    # Reproducibility: temperature=0 for deterministic output across runs
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_seed: int = Field(default=42, alias="LLM_SEED")

    # Output settings
    output_dir: Path = Field(default=Path("examples/outputs"), alias="OUTPUT_DIR")

    def get_available_providers(self) -> list[str]:
        """Return list of providers that have API keys configured."""
        available = ["ollama"]  # Ollama is always available (local)

        if self.gemini_api_key:
            available.append("gemini")
        if self.openai_api_key:
            available.append("openai")
        if self.anthropic_api_key:
            available.append("anthropic")

        return available

    def get_cloud_providers(self) -> list[str]:
        """Return list of cloud providers that have API keys configured."""
        cloud = []
        if self.gemini_api_key:
            cloud.append("gemini")
        if self.openai_api_key:
            cloud.append("openai")
        if self.anthropic_api_key:
            cloud.append("anthropic")
        return cloud

    def has_multiple_providers(self) -> bool:
        """Check if multiple providers are available (potential conflict)."""
        return len(self.get_cloud_providers()) > 0  # ollama + at least one cloud


# Global settings instance
settings = Settings()
