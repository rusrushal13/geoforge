"""Tests for CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from geoforge.cli import app

runner = CliRunner()


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_shows_version(self):
        """Version command should display version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "GeoForge" in result.stdout
        assert "v" in result.stdout or "0." in result.stdout


class TestProvidersCommand:
    """Tests for the providers command."""

    def test_providers_lists_providers(self):
        """Providers command should list available providers."""
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "ollama" in result.stdout.lower()
        # Should mention at least ollama


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_nonexistent_file(self):
        """Validate should fail for nonexistent file."""
        result = runner.invoke(app, ["validate", "nonexistent_file.py"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


class TestGenerateCommand:
    """Tests for the generate command."""

    def test_generate_no_args_shows_help(self):
        """Generate without args should show error."""
        result = runner.invoke(app, ["generate"])
        # Should fail because prompt is required
        assert result.exit_code != 0

    def test_help_flag(self):
        """Help flag should show usage."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.stdout
        assert "validate" in result.stdout
        assert "providers" in result.stdout
        assert "version" in result.stdout

    def test_generate_help(self):
        """Generate help should show options."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.stdout
        assert "--output" in result.stdout
        assert "--validate" in result.stdout
        assert "--preview" in result.stdout
        assert "--debug" in result.stdout

    def test_generate_success_with_preview_debug_and_execute(self, monkeypatch, tmp_path):
        """Generate should reuse one basename across export/preview/debug/execute."""
        from geoforge import cli
        from geoforge.core import logging as core_logging
        from geoforge.core import validator as core_validator
        from geoforge.core.validator import ExportResult, ValidationResult
        from geoforge.llm.base import GeometrySpec, LayerSpec
        from geoforge.viz import renderer

        calls: dict[str, object] = {}

        class FakeProvider:
            async def generate(self, prompt: str, logger=None):
                calls["prompt"] = prompt
                calls["logger_provided"] = logger is not None
                return GeometrySpec(
                    component_type="via_array",
                    description="test geometry",
                    parameters={"rows": 2, "cols": 2},
                    layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
                    gdsfactory_code=(
                        "import gdsfactory as gf\n"
                        "gf.gpdk.PDK.activate()\n"
                        "c = gf.Component('unit_test_cell')\n"
                        "c.write_gds('unit_test.gds')\n"
                        "c.write('unit_test.oas')\n"
                    ),
                )

        class DummyLogger:
            def __init__(self, provider: str):
                calls["logger_provider"] = provider

            def save(self, path: Path):
                calls["debug_path"] = path

        def fake_validate_generated_code(_spec):
            return ValidationResult(
                is_valid=True,
                syntax_ok=True,
                safety_ok=True,
                executes_ok=True,
                spec_match_ok=True,
                gds_created=True,
                oas_created=True,
                execution_time_seconds=0.12,
            )

        def fake_export_code_and_files(code: str, output_dir: Path, base_name: str):
            calls["export_base_name"] = base_name
            output_dir.mkdir(parents=True, exist_ok=True)
            py_path = output_dir / f"{base_name}.py"
            gds_path = output_dir / f"{base_name}.gds"
            oas_path = output_dir / f"{base_name}.oas"
            py_path.write_text(code)
            gds_path.write_text("gds")
            oas_path.write_text("oas")
            return ExportResult(
                success=True,
                py_path=py_path,
                gds_path=gds_path,
                oas_path=oas_path,
            )

        def fake_prepare_code_for_execution(code: str, output_dir: Path, base_name: str):
            calls["execute_base_name"] = base_name
            return "prepared_ok = True"

        def fake_render_gds_file(gds_path: Path, save_path: Path, dpi: int = 150):
            calls["preview_source"] = gds_path
            calls["preview_path"] = save_path
            return object()

        monkeypatch.setattr(cli, "get_provider", lambda _provider: FakeProvider())
        monkeypatch.setattr(core_logging, "PipelineLogger", DummyLogger)
        monkeypatch.setattr(core_validator, "validate_generated_code", fake_validate_generated_code)
        monkeypatch.setattr(core_validator, "export_code_and_files", fake_export_code_and_files)
        monkeypatch.setattr(
            core_validator, "_prepare_code_for_execution", fake_prepare_code_for_execution
        )
        monkeypatch.setattr(renderer, "render_gds_file", fake_render_gds_file)

        result = runner.invoke(
            app,
            [
                "generate",
                "create a test array",
                "--provider",
                "mock",
                "--output-dir",
                str(tmp_path),
                "--output",
                "unit_case",
                "--no-show-code",
                "--preview",
                "--debug",
                "--execute",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert calls["export_base_name"] == "unit_case"
        assert calls["execute_base_name"] == "unit_case"
        assert calls["debug_path"] == tmp_path / "unit_case_debug.json"
        assert calls["preview_source"] == tmp_path / "unit_case.gds"
        assert calls["preview_path"] == tmp_path / "unit_case.png"

    def test_generate_validation_failure_still_saves_code(self, monkeypatch, tmp_path):
        """When validation fails, code should still be saved and preview fallback used."""
        from geoforge import cli
        from geoforge.core import validator as core_validator
        from geoforge.core.validator import ValidationResult
        from geoforge.llm.base import GeometrySpec
        from geoforge.viz import renderer

        calls: dict[str, object] = {}

        class FakeProvider:
            async def generate(self, _prompt: str, logger=None):
                calls["logger_provided"] = logger is not None
                return GeometrySpec(
                    component_type="via_array",
                    description="invalid test",
                    gdsfactory_code="import gdsfactory as gf\nc = gf.Component('bad_cell')\n",
                )

        def fake_validate_generated_code(_spec):
            return ValidationResult(
                is_valid=False,
                syntax_ok=True,
                safety_ok=True,
                executes_ok=False,
                spec_match_ok=True,
                gds_created=False,
                oas_created=False,
                errors=["execution failed"],
            )

        def should_not_export(*_args, **_kwargs):
            msg = "Export should be skipped when validation fails"
            raise AssertionError(msg)

        def fake_render_code(_code: str, save_path: Path, dpi: int = 150):
            calls["preview_path"] = save_path

        monkeypatch.setattr(cli, "get_provider", lambda _provider: FakeProvider())
        monkeypatch.setattr(core_validator, "validate_generated_code", fake_validate_generated_code)
        monkeypatch.setattr(core_validator, "export_code_and_files", should_not_export)
        monkeypatch.setattr(renderer, "render_code", fake_render_code)
        monkeypatch.setattr(
            renderer,
            "render_gds_file",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected gds render")
            ),
        )

        result = runner.invoke(
            app,
            [
                "generate",
                "create invalid example",
                "--provider",
                "mock",
                "--output-dir",
                str(tmp_path),
                "--output",
                "invalid_case",
                "--no-show-code",
                "--preview",
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "invalid_case.py").exists()
        assert calls["preview_path"] == tmp_path / "invalid_case.png"


class TestWebCommand:
    """Tests for the web command."""

    def test_web_help(self):
        """Web help should show options."""
        result = runner.invoke(app, ["web", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--share" in result.stdout


class TestAppHelp:
    """Tests for app-level help."""

    def test_app_no_args_shows_help(self):
        """App without args should show help (exit code 0 or 2 for help)."""
        result = runner.invoke(app, [])
        # Typer with no_args_is_help=True returns exit code 0 or 2
        assert result.exit_code in (0, 2)
        assert "generate" in result.stdout.lower()

    def test_app_has_description(self):
        """App should have description."""
        result = runner.invoke(app, ["--help"])
        assert "semiconductor" in result.stdout.lower() or "geometry" in result.stdout.lower()


class TestCliHelpers:
    """Tests for helper functions inside the CLI module."""

    def test_select_provider_returns_explicit_choice(self):
        from geoforge import cli

        assert cli._select_provider("gemini") == "gemini"

    def test_select_provider_defaults_to_ollama_without_cloud(self, monkeypatch):
        from geoforge import cli

        monkeypatch.setattr(type(cli.settings), "get_cloud_providers", lambda _self: [])
        assert cli._select_provider(None) == "ollama"

    def test_select_provider_prompts_when_cloud_is_available(self, monkeypatch):
        from geoforge import cli

        monkeypatch.setattr(
            type(cli.settings),
            "get_cloud_providers",
            lambda _self: ["gemini", "openai"],
        )
        monkeypatch.setattr(cli.Prompt, "ask", lambda *_args, **_kwargs: "2")
        assert cli._select_provider(None) == "gemini"

    def test_validate_command_success_path(self, monkeypatch, tmp_path):
        from geoforge.core import validator as core_validator

        code_file = tmp_path / "sample.py"
        code_file.write_text("x = 1\n")

        monkeypatch.setattr(core_validator, "validate_syntax", lambda _code: (True, None))
        monkeypatch.setattr(
            core_validator,
            "validate_execution",
            lambda _code: (
                True,
                None,
                {"gds_files": [tmp_path / "one.gds"], "oas_files": [tmp_path / "one.oas"]},
            ),
        )

        result = runner.invoke(app, ["validate", str(code_file)])
        assert result.exit_code == 0
        assert "Execution OK" in result.stdout
