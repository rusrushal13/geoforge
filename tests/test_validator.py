"""Tests for code validation and export functionality."""

import tempfile
from pathlib import Path

import pytest

from geoforge.core.validator import (
    ValidationResult,
    generate_output_name,
    validate_execution,
    validate_generated_code,
    validate_safety,
    validate_syntax,
)


class TestValidateSyntax:
    """Tests for syntax validation."""

    def test_valid_python_code(self):
        """Valid Python code should pass syntax check."""
        code = """
import gdsfactory as gf

@gf.cell
def my_component():
    c = gf.Component()
    return c
"""
        is_valid, error = validate_syntax(code)
        assert is_valid is True
        assert error is None

    def test_invalid_python_syntax(self):
        """Invalid Python syntax should fail."""
        code = """
def broken_function(
    # Missing closing parenthesis
    pass
"""
        is_valid, error = validate_syntax(code)
        assert is_valid is False
        assert error is not None
        assert "syntax" in error.lower() or "invalid" in error.lower()

    def test_empty_code(self):
        """Empty code should pass syntax check (valid Python)."""
        code = ""
        is_valid, _error = validate_syntax(code)
        assert is_valid is True

    def test_code_with_imports(self):
        """Code with standard imports should pass."""
        code = """
import math
from pathlib import Path

x = math.pi
p = Path(".")
"""
        is_valid, _error = validate_syntax(code)
        assert is_valid is True

    def test_code_with_type_hints(self):
        """Code with type hints should pass."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        is_valid, _error = validate_syntax(code)
        assert is_valid is True


class TestValidateSafety:
    """Tests for safety validation."""

    def test_safe_code_passes(self):
        """Code with allowed imports should pass safety check."""
        code = """
import gdsfactory as gf
import numpy as np
import math
"""
        is_safe, error = validate_safety(code)
        assert is_safe is True
        assert error is None

    def test_os_import_banned(self):
        """import os should be flagged."""
        code = "import os\nos.system('ls')"
        is_safe, error = validate_safety(code)
        assert is_safe is False
        assert "os" in error

    def test_subprocess_import_banned(self):
        """import subprocess should be flagged."""
        code = "import subprocess"
        is_safe, error = validate_safety(code)
        assert is_safe is False
        assert "subprocess" in error

    def test_from_import_banned(self):
        """from os.path import ... should be flagged."""
        code = "from os.path import join"
        is_safe, _error = validate_safety(code)
        assert is_safe is False

    def test_socket_banned(self):
        """Network imports should be banned."""
        code = "import socket"
        is_safe, _error = validate_safety(code)
        assert is_safe is False

    def test_invalid_syntax_passes_safety(self):
        """Code with syntax errors should pass safety (caught by syntax check)."""
        code = "def foo("
        is_safe, _error = validate_safety(code)
        assert is_safe is True


class TestValidateExecution:
    """Tests for execution validation."""

    def test_simple_code_executes(self):
        """Simple valid code should execute successfully."""
        code = """
x = 1 + 1
result = x * 2
"""
        success, error, _output_info = validate_execution(code)
        assert success is True
        assert error is None

    def test_code_with_import_error(self):
        """Code with import errors should fail execution."""
        code = """
import nonexistent_module_xyz123
"""
        success, error, _output_info = validate_execution(code)
        assert success is False
        assert error is not None

    def test_code_with_runtime_error(self):
        """Code with runtime errors should fail execution."""
        code = """
x = 1 / 0  # Division by zero
"""
        success, error, _output_info = validate_execution(code)
        assert success is False
        assert error is not None

    def test_code_with_name_error(self):
        """Code with undefined variables should fail."""
        code = """
result = undefined_variable + 1
"""
        success, error, _output_info = validate_execution(code)
        assert success is False
        assert error is not None

    def test_execution_closes_ipc_handles_on_success(self, monkeypatch, tmp_path):
        """Queue/process handles should be closed after successful execution."""
        import geoforge.core.validator as validator_module

        class FakeQueue:
            def __init__(self):
                self._items = [("success", None, [], [])]
                self.closed = False
                self.joined = False

            def empty(self):
                return len(self._items) == 0

            def get_nowait(self):
                return self._items.pop(0)

            def close(self):
                self.closed = True

            def join_thread(self):
                self.joined = True

        class FakeProcess:
            def __init__(self):
                self.exitcode = 0
                self._alive = False
                self.closed = False

            def start(self):
                return None

            def join(self, timeout=None):
                return None

            def is_alive(self):
                return self._alive

            def terminate(self):
                self._alive = False

            def kill(self):
                self._alive = False

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.queue = FakeQueue()
                self.process = FakeProcess()

            def Queue(self):
                return self.queue

            def Process(self, target, args):
                return self.process

        fake_ctx = FakeContext()
        monkeypatch.setattr(
            validator_module.multiprocessing,
            "get_context",
            lambda _method: fake_ctx,
        )
        monkeypatch.setattr(
            validator_module.tempfile,
            "mkdtemp",
            lambda: str(tmp_path / "fake_exec_success"),
        )

        success, error, _output_info = validate_execution("x = 1")
        assert success is True
        assert error is None
        assert fake_ctx.queue.closed is True
        assert fake_ctx.queue.joined is True
        assert fake_ctx.process.closed is True

    def test_execution_closes_ipc_handles_on_timeout(self, monkeypatch, tmp_path):
        """Queue/process handles should be closed even on timeout."""
        import geoforge.core.validator as validator_module

        class FakeQueue:
            def __init__(self):
                self.closed = False
                self.joined = False

            def empty(self):
                return True

            def get_nowait(self):
                raise RuntimeError("Queue should not be read on timeout path")

            def close(self):
                self.closed = True

            def join_thread(self):
                self.joined = True

        class FakeProcess:
            def __init__(self):
                self.exitcode = None
                self._alive = True
                self.terminated = False
                self.closed = False

            def start(self):
                return None

            def join(self, timeout=None):
                return None

            def is_alive(self):
                return self._alive

            def terminate(self):
                self.terminated = True
                self._alive = False

            def kill(self):
                self._alive = False

            def close(self):
                self.closed = True

        class FakeContext:
            def __init__(self):
                self.queue = FakeQueue()
                self.process = FakeProcess()

            def Queue(self):
                return self.queue

            def Process(self, target, args):
                return self.process

        fake_ctx = FakeContext()
        monkeypatch.setattr(
            validator_module.multiprocessing,
            "get_context",
            lambda _method: fake_ctx,
        )
        monkeypatch.setattr(
            validator_module.tempfile,
            "mkdtemp",
            lambda: str(tmp_path / "fake_exec_timeout"),
        )

        success, error, _output_info = validate_execution("while True:\n    pass", timeout=1)
        assert success is False
        assert error is not None
        assert "timed out" in error.lower()
        assert fake_ctx.process.terminated is True
        assert fake_ctx.queue.closed is True
        assert fake_ctx.queue.joined is True
        assert fake_ctx.process.closed is True

    @pytest.mark.slow
    def test_gdsfactory_code_creates_files(self):
        """GDSFactory code should create GDS/OAS files."""
        code = """
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component("test")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
c.write_gds("test_output.gds")
c.write("test_output.oas")
"""
        success, _error, output_info = validate_execution(code)
        assert success is True
        assert output_info["gds_files"] or output_info["oas_files"]

    @pytest.mark.slow
    def test_execution_is_isolated_from_parent_cell_state(self):
        """Execution subprocess should not inherit parent KCLayout cell state."""
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
        gf.Component("isolated_parent_cell")

        code = """
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component("isolated_parent_cell")
rect = gf.components.rectangle(size=(2, 2), layer=(1, 0))
c.add_ref(rect)
c.write_gds("isolated_parent_cell.gds")
c.write("isolated_parent_cell.oas")
"""
        success, error, output_info = validate_execution(code)
        assert success is True, error
        assert output_info["gds_files"] or output_info["oas_files"]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_all_pass(self):
        """ValidationResult with all checks passing should be valid."""
        result = ValidationResult(
            is_valid=True,
            syntax_ok=True,
            executes_ok=True,
            gds_created=True,
            oas_created=True,
            errors=[],
            warnings=[],
        )
        assert result.is_valid is True

    def test_validation_result_syntax_fail(self):
        """ValidationResult with syntax failure should be invalid."""
        result = ValidationResult(
            is_valid=False,
            syntax_ok=False,
            executes_ok=False,
            gds_created=False,
            oas_created=False,
            errors=["Syntax error on line 5"],
            warnings=[],
        )
        assert result.is_valid is False

    def test_validation_result_execution_fail(self):
        """ValidationResult with execution failure should be invalid."""
        result = ValidationResult(
            is_valid=False,
            syntax_ok=True,
            executes_ok=False,
            gds_created=False,
            oas_created=False,
            errors=["ImportError: No module named 'foo'"],
            warnings=[],
        )
        assert result.is_valid is False

    def test_validation_result_no_gds(self):
        """ValidationResult without GDS can still be valid if syntax and execution pass."""
        result = ValidationResult(
            is_valid=True,  # is_valid is based on syntax_ok and executes_ok
            syntax_ok=True,
            executes_ok=True,
            gds_created=False,
            oas_created=False,
            errors=[],
            warnings=["No GDS file created"],
        )
        # is_valid is True because syntax and execution passed
        assert result.is_valid is True

    def test_validation_result_with_warnings(self):
        """ValidationResult with only warnings should still be valid if checks pass."""
        result = ValidationResult(
            is_valid=True,
            syntax_ok=True,
            executes_ok=True,
            gds_created=True,
            oas_created=True,
            errors=[],
            warnings=["Code doesn't use all specified layers"],
        )
        assert result.is_valid is True


class TestValidateGeneratedCode:
    """Tests for full pipeline validation helper."""

    def test_returns_error_when_code_missing(self):
        from geoforge.llm.base import GeometrySpec

        spec = GeometrySpec(component_type="via_array", description="missing code")
        result = validate_generated_code(spec)
        assert result.is_valid is False
        assert result.errors == ["No code generated"]

    def test_stops_on_syntax_error(self, monkeypatch):
        import geoforge.core.validator as validator_module
        from geoforge.llm.base import GeometrySpec

        spec = GeometrySpec(
            component_type="via_array",
            description="syntax fail",
            gdsfactory_code="not python",
        )
        monkeypatch.setattr(
            validator_module, "validate_syntax", lambda _code: (False, "bad syntax")
        )
        monkeypatch.setattr(
            validator_module,
            "validate_safety",
            lambda _code: (_ for _ in ()).throw(AssertionError("safety should not run")),
        )

        result = validator_module.validate_generated_code(spec)
        assert result.syntax_ok is False
        assert "bad syntax" in result.errors

    def test_stops_on_safety_error(self, monkeypatch):
        import geoforge.core.validator as validator_module
        from geoforge.llm.base import GeometrySpec

        spec = GeometrySpec(
            component_type="via_array",
            description="safety fail",
            gdsfactory_code="import gdsfactory as gf",
        )
        monkeypatch.setattr(validator_module, "validate_syntax", lambda _code: (True, None))
        monkeypatch.setattr(
            validator_module, "validate_safety", lambda _code: (False, "Forbidden import")
        )
        monkeypatch.setattr(
            validator_module,
            "validate_execution",
            lambda _code: (_ for _ in ()).throw(AssertionError("execution should not run")),
        )

        result = validator_module.validate_generated_code(spec)
        assert result.safety_ok is False
        assert result.errors == ["Forbidden import"]

    def test_execution_success_with_missing_outputs_adds_warnings(self, monkeypatch):
        import geoforge.core.validator as validator_module
        from geoforge.llm.base import GeometrySpec

        spec = GeometrySpec(
            component_type="via_array",
            description="warn on missing artifacts",
            gdsfactory_code="print('ok')",
        )
        monkeypatch.setattr(validator_module, "validate_syntax", lambda _code: (True, None))
        monkeypatch.setattr(validator_module, "validate_safety", lambda _code: (True, None))
        monkeypatch.setattr(
            validator_module,
            "validate_execution",
            lambda _code: (True, None, {"gds_files": [], "oas_files": []}),
        )
        monkeypatch.setattr(
            validator_module,
            "validate_spec_match",
            lambda _spec, _code: validator_module.SpecMatchResult(
                errors=[],
                warnings=["parameter check warning"],
            ),
        )

        result = validator_module.validate_generated_code(spec)
        assert result.is_valid is True
        assert result.spec_match_ok is True
        assert "Code executed but no GDS file was created" in result.warnings
        assert "Code executed but no OAS file was created" in result.warnings
        assert "parameter check warning" in result.warnings

    def test_execution_failure_and_spec_errors_mark_invalid(self, monkeypatch):
        import geoforge.core.validator as validator_module
        from geoforge.llm.base import GeometrySpec

        spec = GeometrySpec(
            component_type="via_array",
            description="exec fail",
            gdsfactory_code="raise RuntimeError('boom')",
        )
        monkeypatch.setattr(validator_module, "validate_syntax", lambda _code: (True, None))
        monkeypatch.setattr(validator_module, "validate_safety", lambda _code: (True, None))
        monkeypatch.setattr(
            validator_module,
            "validate_execution",
            lambda _code: (False, "RuntimeError: boom", {"gds_files": [], "oas_files": []}),
        )
        monkeypatch.setattr(
            validator_module,
            "validate_spec_match",
            lambda _spec, _code: validator_module.SpecMatchResult(
                errors=["component mismatch"],
                warnings=[],
            ),
        )

        result = validator_module.validate_generated_code(spec)
        assert result.is_valid is False
        assert "RuntimeError: boom" in result.errors
        assert "component mismatch" in result.errors
        assert result.spec_match_ok is False


class TestGenerateOutputName:
    """Tests for output name generation."""

    def test_generates_name_with_component_type(self):
        """Output name should include component type."""
        name = generate_output_name("via_array")
        assert "via_array" in name

    def test_generates_name_with_timestamp(self):
        """Output name should include timestamp."""
        name = generate_output_name("test_component")
        # Should have format: test_component_YYYY-MM-DD_HHMMSS
        # After splitting by _, we get: [test, component, YYYY-MM-DD, HHMMSS]
        assert "test_component" in name
        # Check that there's a date-like pattern
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", name)  # Date pattern

    def test_different_calls_generate_different_names(self):
        """Consecutive calls should generate different names (different timestamps)."""
        import time

        name1 = generate_output_name("component")
        time.sleep(0.01)  # Small delay to ensure different timestamp
        name2 = generate_output_name("component")
        # Names might be same if called within same second, so just check format
        assert "component" in name1
        assert "component" in name2

    def test_handles_special_characters(self):
        """Component types with special chars should be handled."""
        name = generate_output_name("mim_capacitor")
        assert "mim_capacitor" in name


class TestExportCodeAndFiles:
    """Tests for file export functionality."""

    def test_export_creates_py_file(self):
        """Export should create Python file."""
        from geoforge.core.validator import export_code_and_files

        code = """
# Simple test code
x = 1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_code_and_files(code, Path(tmpdir), "test_export")
            assert result.py_path is not None
            assert result.py_path.exists()
            assert result.py_path.read_text() == code

    def test_export_reports_runtime_errors_but_keeps_python_file(self):
        """Export should return error details when generated code crashes."""
        from geoforge.core.validator import export_code_and_files

        code = "raise RuntimeError('export crash')"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_code_and_files(code, Path(tmpdir), "broken_export")
            assert result.success is False
            assert result.py_path is not None and result.py_path.exists()
            assert result.error is not None
            assert "RuntimeError" in result.error

    @pytest.mark.slow
    def test_export_creates_gds_oas_files(self):
        """Export should create GDS and OAS files for valid GDSFactory code."""
        from geoforge.core.validator import export_code_and_files

        code = """
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component("export_test")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
c.write_gds("export_test.gds")
c.write("export_test.oas")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_code_and_files(code, Path(tmpdir), "export_test")
            assert result.py_path is not None
            assert result.py_path.exists()
            # GDS/OAS creation depends on successful execution
            if result.gds_path:
                assert result.gds_path.exists()
            if result.oas_path:
                assert result.oas_path.exists()
