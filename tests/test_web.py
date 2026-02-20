"""Tests for web UI module."""

import asyncio

import pytest

from geoforge.web import handlers
from geoforge.web.handlers import format_spec_markdown, get_provider_choices


class TestGetProviderChoices:
    """Tests for provider choice listing."""

    def test_returns_list(self):
        choices = get_provider_choices()
        assert isinstance(choices, list)
        assert len(choices) >= 1

    def test_ollama_always_present(self):
        choices = get_provider_choices()
        assert "ollama" in choices


class TestFormatSpecMarkdown:
    """Tests for spec markdown formatting."""

    def test_basic_formatting(self):
        from geoforge.llm.base import GeometrySpec, LayerSpec

        spec = GeometrySpec(
            component_type="via_array",
            description="test array",
            parameters={"rows": 3},
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        md = format_spec_markdown(spec)
        assert "via_array" in md
        assert "test array" in md
        assert "rows" in md
        assert "metal1" in md


class TestFormatValidationMarkdown:
    """Tests for validation result formatting."""

    def test_valid_result(self):
        from geoforge.core.validator import ValidationResult
        from geoforge.web.handlers import format_validation_markdown

        result = ValidationResult(
            is_valid=True,
            syntax_ok=True,
            safety_ok=True,
            executes_ok=True,
            spec_match_ok=True,
            gds_created=True,
            oas_created=True,
        )
        md = format_validation_markdown(result)
        assert "Valid" in md
        assert "Pass" in md

    def test_invalid_result_with_errors(self):
        from geoforge.core.validator import ValidationResult
        from geoforge.web.handlers import format_validation_markdown

        result = ValidationResult(
            is_valid=False,
            syntax_ok=True,
            safety_ok=True,
            executes_ok=False,
            errors=["Runtime error: division by zero"],
        )
        md = format_validation_markdown(result)
        assert "Invalid" in md
        assert "division by zero" in md


class TestCreateApp:
    """Tests for Gradio app creation."""

    def test_create_app_returns_blocks(self):
        import gradio as gr

        from geoforge.web.app import create_app

        app = create_app()
        assert isinstance(app, gr.Blocks)


@pytest.mark.asyncio
async def test_run_pipeline_dedupes_concurrent_identical_requests(monkeypatch):
    """Concurrent duplicate requests should only run pipeline once."""
    call_count = 0

    async def fake_run_once(prompt: str, provider_name: str):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return handlers.PipelineResult(success=True, code=f"{provider_name}:{prompt}")

    monkeypatch.setattr(handlers, "_run_pipeline_once", fake_run_once)
    handlers._INFLIGHT_PIPELINES.clear()
    handlers._RECENT_PIPELINE_RESULTS.clear()

    results = await asyncio.gather(
        handlers.run_pipeline("create via array", "gemini"),
        handlers.run_pipeline("create via array", "gemini"),
    )

    assert call_count == 1
    assert all(result.success for result in results)


@pytest.mark.asyncio
async def test_run_pipeline_dedupes_immediate_retries(monkeypatch):
    """Immediate retries should reuse recent result and avoid duplicate file writes."""
    call_count = 0

    async def fake_run_once(prompt: str, provider_name: str):
        nonlocal call_count
        call_count += 1
        return handlers.PipelineResult(
            success=True, code=f"run-{call_count}:{provider_name}:{prompt}"
        )

    monkeypatch.setattr(handlers, "_run_pipeline_once", fake_run_once)
    handlers._INFLIGHT_PIPELINES.clear()
    handlers._RECENT_PIPELINE_RESULTS.clear()

    first = await handlers.run_pipeline("create via array", "gemini")
    second = await handlers.run_pipeline("create via array", "gemini")

    assert call_count == 1
    assert second.code == first.code
