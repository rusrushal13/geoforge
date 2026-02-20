"""Integration tests for the full pipeline using mock provider."""

import tempfile
from pathlib import Path

import pytest

from tests.mock_provider import MOCK_GDSFACTORY_CODE, MockLLMProvider


class TestFullPipeline:
    """End-to-end pipeline tests with mock provider."""

    @pytest.mark.asyncio
    async def test_generate_returns_spec_with_code(self):
        """Full generate() should return spec with code populated."""
        provider = MockLLMProvider()
        spec = await provider.generate("Create a via array")
        assert spec.component_type == "via_array"
        assert spec.gdsfactory_code is not None
        assert "gdsfactory" in spec.gdsfactory_code or "gf" in spec.gdsfactory_code

    @pytest.mark.asyncio
    async def test_generate_with_logger(self):
        """Generate with PipelineLogger should record events."""
        from geoforge.core.logging import PipelineLogger

        provider = MockLLMProvider()
        logger = PipelineLogger(provider="mock")
        spec = await provider.generate("Create a via array", logger=logger)

        assert spec.gdsfactory_code is not None
        log_data = logger.to_dict()
        assert log_data["provider"] == "mock"
        assert log_data["event_count"] >= 4  # start + success for each stage
        stages = [e["stage"] for e in log_data["events"]]
        assert "spec_generation" in stages
        assert "code_generation" in stages

    @pytest.mark.asyncio
    async def test_generate_spec_failure_logged(self):
        """Failed spec generation should be logged."""
        from geoforge.core.logging import PipelineLogger

        provider = MockLLMProvider(fail_spec=True)
        provider.max_retries = 1
        logger = PipelineLogger(provider="mock")

        with pytest.raises(ValueError):
            await provider.generate("test", logger=logger)

        log_data = logger.to_dict()
        error_events = [e for e in log_data["events"] if e["status"] == "error"]
        assert len(error_events) >= 1

    @pytest.mark.slow
    def test_validate_mock_code(self):
        """Mock code should pass validation."""
        from geoforge.core.validator import validate_execution, validate_syntax

        ok, err = validate_syntax(MOCK_GDSFACTORY_CODE)
        assert ok, f"Syntax error: {err}"

        ok, err, info = validate_execution(MOCK_GDSFACTORY_CODE)
        assert ok, f"Execution error: {err}"
        assert info["gds_files"] or info["oas_files"]

    @pytest.mark.slow
    def test_export_mock_code(self):
        """Mock code should export files."""
        from geoforge.core.validator import export_code_and_files

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_code_and_files(MOCK_GDSFACTORY_CODE, Path(tmpdir), "test_mock")
            assert result.py_path is not None
            assert result.py_path.exists()
            assert result.success


class TestPipelineLogger:
    """Tests for PipelineLogger."""

    def test_start_and_success(self):
        from geoforge.core.logging import PipelineLogger

        logger = PipelineLogger(provider="test")
        logger.start("spec_generation")
        logger.success("spec_generation", component="via_array")

        data = logger.to_dict()
        assert len(data["events"]) == 2
        assert data["events"][0]["status"] == "start"
        assert data["events"][1]["status"] == "success"
        assert data["events"][1]["duration_seconds"] is not None

    def test_error_logging(self):
        from geoforge.core.logging import PipelineLogger

        logger = PipelineLogger(provider="test")
        logger.start("code_generation")
        logger.log_error("code_generation", "syntax error", attempt=1)

        data = logger.to_dict()
        error_event = data["events"][1]
        assert error_event["status"] == "error"
        assert error_event["error"] == "syntax error"
        assert error_event["attempt"] == 1

    def test_retry_logging(self):
        from geoforge.core.logging import PipelineLogger

        logger = PipelineLogger(provider="test")
        logger.retry("spec_generation", attempt=1, error="bad json")

        data = logger.to_dict()
        assert data["events"][0]["status"] == "retry"

    def test_save_to_file(self):
        from geoforge.core.logging import PipelineLogger

        logger = PipelineLogger(provider="test")
        logger.start("spec_generation")
        logger.success("spec_generation")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "debug.json"
            logger.save(path)
            assert path.exists()
            import json

            data = json.loads(path.read_text())
            assert data["provider"] == "test"
            assert len(data["events"]) == 2
