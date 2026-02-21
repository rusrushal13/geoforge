"""Tests for web UI module."""

import asyncio
from types import SimpleNamespace

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

    def test_request_key_normalizes_prompt_and_provider(self):
        key = handlers._request_key("  make   via   array  ", " GeMiNi ")
        assert key == ("gemini", "make via array")

    def test_prune_expired_recent_results(self):
        handlers._RECENT_PIPELINE_RESULTS.clear()
        handlers._RECENT_PIPELINE_RESULTS[("gemini", "new")] = (99.5, handlers.PipelineResult())
        handlers._RECENT_PIPELINE_RESULTS[("gemini", "old")] = (
            0.0,
            handlers.PipelineResult(),
        )
        handlers._prune_expired_recent_results(100.0)
        assert ("gemini", "old") not in handlers._RECENT_PIPELINE_RESULTS
        assert ("gemini", "new") in handlers._RECENT_PIPELINE_RESULTS


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

    def test_includes_geometry_ranges_and_primitives(self):
        from geoforge.llm.base import GeometryRange, GeometrySpec, PrimitiveSpec

        spec = GeometrySpec(
            component_type="test_pattern",
            description="custom art",
            parameters={"size": 200.0},
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=2,
                    shape="rectangle",
                    width_start=10.0,
                    width_end=20.0,
                    height_start=5.0,
                    height_end=7.0,
                    center_x=1.0,
                    center_y=2.0,
                )
            ],
            primitives=[
                PrimitiveSpec(
                    primitive_type="polygon",
                    layer_number=1,
                    points=[(-1.0, -1.0), (0.0, 2.0), (1.0, -1.0)],
                    rotation_deg=15.0,
                )
            ],
        )
        md = format_spec_markdown(spec)
        assert "Geometry Ranges" in md
        assert "Primitives" in md
        assert "rotation 15.0 deg" in md


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

    def test_includes_execution_time_and_warnings(self):
        from geoforge.core.validator import ValidationResult
        from geoforge.web.handlers import format_validation_markdown

        result = ValidationResult(
            is_valid=True,
            syntax_ok=True,
            safety_ok=True,
            executes_ok=True,
            spec_match_ok=True,
            gds_created=True,
            oas_created=False,
            execution_time_seconds=1.23,
            warnings=["missing oas"],
        )
        md = format_validation_markdown(result)
        assert "Execution time: 1.23s" in md
        assert "Warnings" in md
        assert "missing oas" in md


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


@pytest.mark.asyncio
async def test_run_pipeline_cache_expiry_triggers_rerun(monkeypatch):
    """Expired cached results should trigger a fresh pipeline run."""
    call_count = 0

    async def fake_run_once(prompt: str, provider_name: str):
        nonlocal call_count
        call_count += 1
        return handlers.PipelineResult(success=True, code=f"{call_count}:{provider_name}:{prompt}")

    monkeypatch.setattr(handlers, "_run_pipeline_once", fake_run_once)
    monkeypatch.setattr(handlers, "_PIPELINE_DEDUPE_TTL_SECONDS", 0.0)
    handlers._INFLIGHT_PIPELINES.clear()
    handlers._RECENT_PIPELINE_RESULTS.clear()

    first = await handlers.run_pipeline("create via", "gemini")
    second = await handlers.run_pipeline("create via", "gemini")

    assert call_count == 2
    assert first.code != second.code


@pytest.mark.asyncio
async def test_run_pipeline_once_provider_error(monkeypatch):
    from geoforge import llm as llm_module

    monkeypatch.setattr(
        llm_module, "get_provider", lambda _name: (_ for _ in ()).throw(ValueError("bad provider"))
    )
    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is False
    assert result.error is not None and "Provider error" in result.error


@pytest.mark.asyncio
async def test_run_pipeline_once_generation_error(monkeypatch):
    from geoforge import llm as llm_module

    class BadProvider:
        async def generate(self, prompt: str):
            del prompt
            raise RuntimeError("generation exploded")

    monkeypatch.setattr(llm_module, "get_provider", lambda _name: BadProvider())
    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is False
    assert result.error is not None and "Generation failed" in result.error


@pytest.mark.asyncio
async def test_run_pipeline_once_no_code_generated(monkeypatch):
    from geoforge import llm as llm_module
    from geoforge.llm.base import GeometrySpec

    class NoCodeProvider:
        async def generate(self, prompt: str):
            del prompt
            return GeometrySpec(component_type="test_pattern", description="no code")

    monkeypatch.setattr(llm_module, "get_provider", lambda _name: NoCodeProvider())
    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is False
    assert result.error == "No code was generated"


@pytest.mark.asyncio
async def test_run_pipeline_once_validation_failure_skips_export_preview(monkeypatch):
    from geoforge import llm as llm_module
    from geoforge.core import validator as validator_module
    from geoforge.core.validator import ValidationResult
    from geoforge.llm.base import GeometrySpec
    from geoforge.viz import renderer as renderer_module

    class Provider:
        async def generate(self, prompt: str):
            del prompt
            return GeometrySpec(
                component_type="test_pattern",
                description="invalid",
                gdsfactory_code="print('x')",
            )

    monkeypatch.setattr(llm_module, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(
        validator_module,
        "validate_generated_code",
        lambda _spec: ValidationResult(
            is_valid=False,
            syntax_ok=True,
            safety_ok=True,
            executes_ok=False,
            spec_match_ok=True,
            gds_created=False,
            oas_created=False,
        ),
    )
    monkeypatch.setattr(
        validator_module,
        "export_code_and_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("export should be skipped")),
    )
    monkeypatch.setattr(
        renderer_module,
        "render_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preview should be skipped")
        ),
    )

    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is False
    assert "Export skipped: validation failed" in result.logs


@pytest.mark.asyncio
async def test_run_pipeline_once_success_renders_preview_from_exported_gds(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    from geoforge import llm as llm_module
    from geoforge.core import validator as validator_module
    from geoforge.core.validator import ValidationResult
    from geoforge.llm.base import GeometrySpec
    from geoforge.viz import renderer as renderer_module

    gds_path = tmp_path / "unit.gds"
    oas_path = tmp_path / "unit.oas"
    py_path = tmp_path / "unit.py"
    gds_path.write_text("gds")
    oas_path.write_text("oas")
    py_path.write_text("py")

    class Provider:
        async def generate(self, prompt: str):
            del prompt
            return GeometrySpec(
                component_type="test_pattern",
                description="ok",
                gdsfactory_code="print('x')",
            )

    monkeypatch.setattr(llm_module, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(
        validator_module,
        "validate_generated_code",
        lambda _spec: ValidationResult(
            is_valid=True,
            syntax_ok=True,
            safety_ok=True,
            executes_ok=True,
            spec_match_ok=True,
            gds_created=True,
            oas_created=True,
        ),
    )
    monkeypatch.setattr(validator_module, "generate_output_name", lambda _ctype: "unit")
    monkeypatch.setattr(
        validator_module,
        "export_code_and_files",
        lambda *_args, **_kwargs: SimpleNamespace(
            py_path=py_path, gds_path=gds_path, oas_path=oas_path
        ),
    )
    monkeypatch.setattr(handlers.settings, "output_dir", tmp_path, raising=False)
    monkeypatch.setattr(renderer_module, "render_gds_file", lambda *_args, **_kwargs: plt.figure())

    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is True
    assert result.preview_path is not None
    assert result.gds_path == gds_path
    assert "Preview rendered" in result.logs


@pytest.mark.asyncio
async def test_run_pipeline_once_validation_error_logged(monkeypatch):
    from geoforge import llm as llm_module
    from geoforge.core import validator as validator_module
    from geoforge.llm.base import GeometrySpec
    from geoforge.viz import renderer as renderer_module

    class Provider:
        async def generate(self, prompt: str):
            del prompt
            return GeometrySpec(
                component_type="test_pattern",
                description="ok",
                gdsfactory_code="print('x')",
            )

    monkeypatch.setattr(llm_module, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(
        validator_module,
        "validate_generated_code",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("validator crash")),
    )
    monkeypatch.setattr(renderer_module, "render_code", lambda *_args, **_kwargs: None)

    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is False
    assert result.validation_markdown is not None
    assert "Validation error: validator crash" in result.validation_markdown
    assert any("Validation error" in log for log in result.logs)


@pytest.mark.asyncio
async def test_run_pipeline_once_preview_error_logged(monkeypatch, tmp_path):
    from geoforge import llm as llm_module
    from geoforge.core import validator as validator_module
    from geoforge.core.validator import ValidationResult
    from geoforge.llm.base import GeometrySpec
    from geoforge.viz import renderer as renderer_module

    class Provider:
        async def generate(self, prompt: str):
            del prompt
            return GeometrySpec(
                component_type="test_pattern",
                description="ok",
                gdsfactory_code="print('x')",
            )

    monkeypatch.setattr(llm_module, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(
        validator_module,
        "validate_generated_code",
        lambda _spec: ValidationResult(
            is_valid=True,
            syntax_ok=True,
            safety_ok=True,
            executes_ok=True,
            spec_match_ok=True,
            gds_created=True,
            oas_created=True,
        ),
    )
    monkeypatch.setattr(validator_module, "generate_output_name", lambda _ctype: "unit")
    monkeypatch.setattr(
        validator_module,
        "export_code_and_files",
        lambda *_args, **_kwargs: SimpleNamespace(py_path=None, gds_path=None, oas_path=None),
    )
    monkeypatch.setattr(handlers.settings, "output_dir", tmp_path, raising=False)
    monkeypatch.setattr(
        renderer_module,
        "render_code",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("render crash")),
    )

    result = await handlers._run_pipeline_once("prompt", "gemini")
    assert result.success is True
    assert any("Preview error: render crash" in log for log in result.logs)
