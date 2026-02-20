"""Ollama LLM provider for local models."""

import json

import ollama as ollama_client
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from geoforge.config import settings
from geoforge.llm.base import (
    GEOMETRY_SPEC_SCHEMA,
    GeometrySpec,
    LLMProvider,
    RetryContext,
    _format_retry_message,
)
from geoforge.prompts.gdsfactory import GEOMETRY_SPEC_PROMPT, build_code_prompt

console = Console()


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM models."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, auto_pull: bool = True):
        """Initialize Ollama provider.

        Args:
            model: Model name (defaults to settings.ollama_model)
            host: Ollama host URL (defaults to settings.ollama_host)
            auto_pull: Automatically download model if not present (default: True)
        """
        self.model = model or settings.ollama_model
        self.host = host or settings.ollama_host
        self.auto_pull = auto_pull
        self._sync_client = ollama_client.Client(host=self.host)
        self.client = ollama_client.AsyncClient(host=self.host)

        # Check and pull model if needed
        if self.auto_pull:
            self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Check if model is available, download if not."""
        try:
            # Check if Ollama is running
            models = self._sync_client.list()
            model_names: list[str] = (
                [m.model for m in models.models if m.model is not None] if models.models else []
            )

            # Check if our model is in the list (handle tags like llama3.2:latest)
            model_available = any(
                self.model in name or name.startswith(f"{self.model}:") for name in model_names
            )

            if not model_available:
                console.print(
                    f"[yellow]Model '{self.model}' not found locally. Downloading...[/yellow]"
                )
                self._pull_model()
                console.print(f"[green]Model '{self.model}' downloaded successfully![/green]")

        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.host}. "
                    "Please ensure Ollama is running:\n"
                    "  1. Install Ollama: https://ollama.ai\n"
                    "  2. Start Ollama: ollama serve"
                ) from e
            raise

    def _pull_model(self) -> None:
        """Pull/download the model with progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {self.model}...", total=None)

            # Stream the pull progress
            for status in self._sync_client.pull(self.model, stream=True):
                if hasattr(status, "status"):
                    progress.update(task, description=f"{self.model}: {status.status}")

    @classmethod
    def is_available(cls, host: str | None = None) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            client = ollama_client.Client(host=host or settings.ollama_host)
            client.list()
            return True
        except Exception:
            return False

    @classmethod
    def list_local_models(cls, host: str | None = None) -> list[str]:
        """List all locally available models."""
        try:
            client = ollama_client.Client(host=host or settings.ollama_host)
            models = client.list()
            return [m.model for m in models.models if m.model is not None] if models.models else []
        except Exception:
            return []

    async def _generate_geometry_spec_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Convert natural language to geometry specification using Ollama."""
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

        response = await self.client.chat(
            model=self.model,
            messages=messages,
            format="json",
            options={
                "temperature": settings.llm_temperature,
                "seed": settings.llm_seed,
            },
        )

        return json.loads(response["message"]["content"])

    async def _generate_gdsfactory_code_impl(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
        retry_context: RetryContext | None = None,
    ) -> str:
        """Generate GDSFactory code from specification using Ollama."""
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

        response = await self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": settings.llm_temperature,
                "seed": settings.llm_seed,
            },
        )

        return response["message"]["content"]
