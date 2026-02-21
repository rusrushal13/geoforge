"""Tests for error-aware retry logic."""

import pytest

from geoforge.llm.base import (
    GeometrySpec,
    LLMProvider,
    RetryContext,
    _classify_error,
    _format_retry_message,
)


class TestRetryContext:
    """Tests for RetryContext model."""

    def test_basic_creation(self):
        ctx = RetryContext(
            attempt_number=1,
            previous_error="Invalid JSON",
            error_category="json_parse",
        )
        assert ctx.attempt_number == 1
        assert ctx.error_category == "json_parse"
        assert ctx.previous_response_snippet is None

    def test_with_snippet(self):
        ctx = RetryContext(
            attempt_number=2,
            previous_error="Schema validation failed",
            previous_response_snippet='{"component_type": 123}',
            error_category="schema_validation",
        )
        assert ctx.previous_response_snippet == '{"component_type": 123}'

    def test_default_category(self):
        ctx = RetryContext(attempt_number=1, previous_error="unknown")
        assert ctx.error_category == "other"


class TestClassifyError:
    """Tests for error classification."""

    def test_json_decode_error(self):
        import json

        try:
            json.loads("not json")
        except json.JSONDecodeError as e:
            assert _classify_error(e) == "json_parse"

    def test_validation_error(self):
        from pydantic import ValidationError

        from geoforge.llm.base import GeometrySpec

        try:
            GeometrySpec(component_type=123, description=456)
        except ValidationError as e:
            assert _classify_error(e) == "schema_validation"

    def test_too_short_error(self):
        assert _classify_error(ValueError("code is too short")) == "code_too_short"

    def test_empty_error(self):
        assert _classify_error(ValueError("response is empty")) == "code_too_short"

    def test_import_error(self):
        assert _classify_error(ValueError("missing import gdsfactory")) == "missing_import"

    def test_syntax_error(self):
        assert _classify_error(ValueError("syntax issue found")) == "syntax_error"

    def test_execution_error(self):
        assert _classify_error(RuntimeError("Execution failed: AttributeError: bad API")) == (
            "execution_error"
        )

    def test_generic_error(self):
        assert _classify_error(RuntimeError("something broke")) == "other"


class TestFormatRetryMessage:
    """Tests for retry message formatting."""

    def test_basic_format(self):
        ctx = RetryContext(
            attempt_number=1,
            previous_error="Invalid JSON",
            error_category="json_parse",
        )
        msg = _format_retry_message(ctx)
        assert "attempt 1" in msg
        assert "json_parse" in msg
        assert "Invalid JSON" in msg
        assert "fix" in msg.lower()

    def test_format_with_snippet(self):
        ctx = RetryContext(
            attempt_number=2,
            previous_error="bad schema",
            previous_response_snippet="some output",
            error_category="schema_validation",
        )
        msg = _format_retry_message(ctx)
        assert "some output" in msg
        assert "previous response started with" in msg

    def test_execution_retry_hints(self):
        ctx = RetryContext(
            attempt_number=1,
            previous_error="AttributeError: Path has no end_point",
            error_category="execution_error",
        )
        msg = _format_retry_message(ctx)
        assert "Execution fix hints" in msg
        assert "Path.end_point" in msg


class TestRetryBehavior:
    """Tests for the actual retry behavior in providers."""

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_success(self):
        """Provider should retry and succeed after initial failures."""
        from tests.mock_provider import MockLLMProvider

        provider = MockLLMProvider(fail_count=1)
        spec = await provider.generate_geometry_spec("test prompt")
        assert spec.component_type == "via_array"
        assert provider._spec_calls == 2  # First call fails, second succeeds

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Provider should raise after max retries."""
        from tests.mock_provider import MockLLMProvider

        provider = MockLLMProvider(fail_spec=True)
        provider.max_retries = 2
        with pytest.raises(ValueError, match="Failed to generate"):
            await provider.generate_geometry_spec("test prompt")
        assert provider._spec_calls == 2

    @pytest.mark.asyncio
    async def test_code_gen_retry(self):
        """Code generation should retry on failure."""
        from tests.mock_provider import MockLLMProvider

        provider = MockLLMProvider(fail_count=1)
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
        )
        code = await provider.generate_gdsfactory_code(spec)
        assert "gdsfactory" in code or "gf" in code
        assert provider._code_calls == 2


class _PromptSuccessProvider(LLMProvider):
    async def _generate_geometry_spec_impl(self, prompt: str, retry_context=None) -> dict:
        return {
            "component_type": "test_pattern",
            "description": "test",
            "parameters": {},
            "layers": [],
        }

    async def _generate_gdsfactory_code_impl(
        self, spec: GeometrySpec, original_prompt=None, retry_context=None
    ) -> str:
        return (
            "import gdsfactory as gf\nc = gf.Component('x')\nc.write_gds('x.gds')\nc.write('x.oas')"
        )

    async def _enhance_prompt_impl(self, prompt: str, retry_context=None) -> dict:
        return {
            "rewritten_prompt": f"Enhanced: {prompt}",
            "key_constraints": ["single layer", "centered at origin"],
        }


class _PromptFailureProvider(_PromptSuccessProvider):
    async def _enhance_prompt_impl(self, prompt: str, retry_context=None) -> dict:
        raise ValueError("enhancement failed")


@pytest.mark.asyncio
async def test_prompt_enhancement_success():
    provider = _PromptSuccessProvider()
    result = await provider.enhance_prompt("draw a logo")
    assert result.rewritten_prompt.startswith("Enhanced:")
    assert "centered at origin" in result.key_constraints


@pytest.mark.asyncio
async def test_prompt_enhancement_fallback_to_original():
    provider = _PromptFailureProvider()
    provider.max_retries = 1
    result = await provider.enhance_prompt("keep original text")
    assert result.rewritten_prompt == "keep original text"
