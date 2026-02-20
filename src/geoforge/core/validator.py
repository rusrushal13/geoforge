"""Validation utilities for generated GDSFactory code."""

import ast
import multiprocessing
import re
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from geoforge.llm.base import GeometrySpec

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue


class ValidationResult(BaseModel):
    """Result of code validation."""

    is_valid: bool
    syntax_ok: bool = False
    executes_ok: bool = False
    safety_ok: bool = False
    spec_match_ok: bool = False
    gds_created: bool = False
    oas_created: bool = False
    component_name: str | None = None
    errors: list[str] = []
    warnings: list[str] = []
    execution_time_seconds: float | None = None


class SpecMatchResult(BaseModel):
    """Result of spec-to-code matching validation."""

    errors: list[str] = []
    warnings: list[str] = []


class ExportResult(BaseModel):
    """Result of code export."""

    success: bool
    py_path: Path | None = None
    gds_path: Path | None = None
    oas_path: Path | None = None
    error: str | None = None


# Imports that generated code is allowed to use
_ALLOWED_IMPORTS = frozenset(
    {
        "gdsfactory",
        "gf",
        "math",
        "numpy",
        "np",
        "functools",
        "itertools",
        "collections",
        "typing",
        "pathlib",
    }
)

# Imports that are explicitly banned for safety
_BANNED_IMPORTS = frozenset(
    {
        "os",
        "subprocess",
        "shutil",
        "sys",
        "socket",
        "http",
        "urllib",
        "requests",
        "ctypes",
        "signal",
        "multiprocessing",
        "threading",
        "importlib",
    }
)


def validate_syntax(code: str) -> tuple[bool, str | None]:
    """Check if code is valid Python syntax.

    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def validate_safety(code: str) -> tuple[bool, str | None]:
    """Check code for potentially dangerous patterns.

    Scans the AST for banned imports and dangerous function calls including
    dynamic import techniques (__import__, importlib, eval, exec, compile).

    Returns:
        (is_safe, error_message)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors are caught by validate_syntax; here we just skip
        return True, None

    # Dangerous builtins that can bypass static import checks
    _DANGEROUS_CALLS = frozenset({"__import__", "eval", "compile", "exec"})

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in _BANNED_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"

        if isinstance(node, ast.ImportFrom) and node.module:
            root_module = node.module.split(".")[0]
            if root_module in _BANNED_IMPORTS:
                return False, f"Forbidden import: from {node.module}"

        # Detect dangerous function calls: __import__(), eval(), compile(), exec()
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name and func_name in _DANGEROUS_CALLS:
                return False, f"Forbidden call: {func_name}() (dynamic code execution not allowed)"

    return True, None


def _prepare_code_for_execution(code: str, output_dir: Path, base_name: str) -> str:
    """Prepare code for execution by redirecting file paths.

    Args:
        code: Original GDSFactory code
        output_dir: Directory to save output files
        base_name: Base name for output files (without extension)

    Returns:
        Modified code with redirected file paths
    """
    # Add imports and path redirection
    prefix_code = f'''
import os as _os
_output_dir = "{output_dir}"
_base_name = "{base_name}"

def _redirect_path(original_path):
    """Redirect output path to use our naming convention."""
    ext = _os.path.splitext(str(original_path))[1]
    return _os.path.join(_output_dir, _base_name + ext)
'''
    modified_code = prefix_code + code

    # Redirect write_gds and write calls to use the wrapper
    modified_code = re.sub(
        r"\.write_gds\(([^)]+)\)",
        r".write_gds(_redirect_path(\1))",
        modified_code,
    )
    return re.sub(
        r"\.write\(([^)]+)\)",
        r".write(_redirect_path(\1))",
        modified_code,
    )


def _run_code_in_process(code: str, tmpdir: str, result_queue: multiprocessing.Queue) -> None:
    """Execute code in a separate process for timeout isolation."""
    import gdsfactory as gf

    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()

    modified_code = _prepare_code_for_execution(code, Path(tmpdir), "test_output")
    modified_code = modified_code.replace(".show()", "# .show() disabled for validation")

    try:
        exec_globals = {
            "__name__": "__main__",
            "gf": gf,
        }
        exec(modified_code, exec_globals)

        # Collect output files
        tmppath = Path(tmpdir)
        gds_files = [str(p) for p in tmppath.glob("*.gds")]
        oas_files = [str(p) for p in tmppath.glob("*.oas")]

        result_queue.put(("success", None, gds_files, oas_files))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}", [], []))


def validate_execution(code: str, timeout: int = 30) -> tuple[bool, str | None, dict]:
    """Execute code in isolated environment with timeout protection.

    Runs code in a separate process so infinite loops or heavy operations
    can be killed after the timeout.

    Returns:
        (success, error_message, output_info)
    """
    output_info: dict = {
        "gds_files": [],
        "oas_files": [],
        "component_name": None,
    }

    # Create temp directory BEFORE forking so child can write and parent can read
    tmpdir = tempfile.mkdtemp()
    result_queue: Queue | None = None
    proc: BaseProcess | None = None

    try:
        # Use "spawn" for a clean interpreter state on each execution.
        # This avoids inheriting global GDSFactory/KLayout state via fork(),
        # which can cause flaky duplicate-cell errors across test runs.
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_run_code_in_process,
            args=(code, tmpdir, result_queue),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            return False, f"Execution timed out after {timeout}s", output_info

        if proc.exitcode != 0 and result_queue.empty():
            return False, f"Process exited with code {proc.exitcode}", output_info

        if result_queue.empty():
            return False, "Process completed but produced no result", output_info

        status, error, gds_files, oas_files = result_queue.get_nowait()
        output_info["gds_files"] = [Path(p) for p in gds_files]
        output_info["oas_files"] = [Path(p) for p in oas_files]

        if status == "success":
            return True, None, output_info
        return False, error, output_info

    except Exception as e:
        return False, f"Validation process error: {type(e).__name__}: {e}", output_info
    finally:
        # Explicitly close multiprocessing handles to avoid resource_tracker warnings
        # about leaked semaphores after many validation runs.
        if result_queue is not None:
            with suppress(Exception):
                result_queue.close()
            with suppress(Exception):
                result_queue.join_thread()

        if proc is not None:
            with suppress(Exception):
                proc.close()

        # Clean up temp directory
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def validate_spec_match(spec: GeometrySpec, code: str) -> SpecMatchResult:
    """Check if generated code matches the specification.

    Returns errors for critical mismatches and warnings for minor ones.

    Returns:
        SpecMatchResult with errors and warnings.
    """
    result = SpecMatchResult()

    # Check if component type is mentioned
    component_terms = spec.component_type.lower().replace("_", " ").split()
    if not any(term in code.lower() for term in component_terms):
        result.errors.append(f"Component type '{spec.component_type}' not referenced in code")

    # Check if ANY specified layers appear in code
    if spec.layers:
        # Detect dynamic layer generation patterns (loops, dicts, f-strings)
        # When code builds layers programmatically, literal tuples won't appear
        dynamic_layer_patterns = [
            "for",  # loop-based layer generation
            "range(",  # iterating over layer numbers
            "layer_num",  # common variable name for dynamic layers
            "layer_number",  # another common pattern
            "(i, 0)",  # generic loop variable tuple
            "(i,0)",  # without space
            "(layer_num, 0)",  # explicit loop variable
            'f"layer_{',  # f-string layer lookup
            "f'layer_{",  # single-quote f-string
        ]
        uses_dynamic_layers = any(pattern in code for pattern in dynamic_layer_patterns)

        found_any_layer = False
        missing_layers = []
        for layer in spec.layers:
            layer_tuple = f"({layer.layer_number}, {layer.datatype})"
            alt_tuple = f"({layer.layer_number},{layer.datatype})"
            if layer_tuple in code or alt_tuple in code:
                found_any_layer = True
            else:
                missing_layers.append(f"Layer {layer.name} {layer_tuple} not found in code")

        if found_any_layer or uses_dynamic_layers:
            # Some layers found literally or layers are generated dynamically
            if uses_dynamic_layers and not found_any_layer:
                result.warnings.append(
                    "Layers appear to be generated dynamically (via loops/dicts); "
                    "skipping individual layer literal checks"
                )
            else:
                result.warnings.extend(missing_layers)
        else:
            result.errors.append("Generated code does not use ANY of the specified layers")
            result.warnings.extend(missing_layers)

    # Check if parameters are used (soft warning only)
    for key, value in spec.parameters.items():
        if (
            isinstance(value, int | float)
            and str(value) not in code
            and str(float(value)) not in code
        ):
            result.warnings.append(f"Parameter '{key}={value}' may not be used in code")

    # Check geometry_ranges coverage and placement.
    # These are strict checks because missing range handling changes geometry intent.
    if spec.geometry_ranges:

        def _numeric_literal_present(value: float) -> bool:
            return str(value) in code or (float(value).is_integer() and str(int(value)) in code)

        has_data_driven_ranges = (
            "geometry_ranges" in code
            or ("start_layer" in code and "end_layer" in code)
            or ("for r in" in code and "range(" in code)
        )
        has_range_loop = any(
            pattern in code
            for pattern in ("for i in range(", "for layer in range(", "for layer_num in range(")
        )

        if len(spec.geometry_ranges) > 1 and not has_data_driven_ranges:
            referenced_starts = sum(
                1 for gr in spec.geometry_ranges if _numeric_literal_present(float(gr.start_layer))
            )
            if referenced_starts < len(spec.geometry_ranges):
                result.errors.append(
                    "Generated code does not appear to handle all geometry_ranges "
                    f"(found {referenced_starts} of {len(spec.geometry_ranges)} range starts)"
                )

        for gr in spec.geometry_ranges:
            has_range_ref = _numeric_literal_present(
                float(gr.start_layer)
            ) and _numeric_literal_present(float(gr.end_layer))

            if not has_range_ref and not has_data_driven_ranges:
                error_msg = (
                    f"Geometry range {gr.start_layer}-{gr.end_layer} "
                    f"may not be handled in generated code"
                )
                if has_range_loop:
                    error_msg = (
                        f"Geometry range {gr.start_layer}-{gr.end_layer} not reflected in code"
                    )
                result.errors.append(error_msg)

            if gr.center_x != 0.0 or gr.center_y != 0.0:
                has_any_placement = any(
                    pattern in code
                    for pattern in ("dmove(", ".move(", ".movex(", ".movey(", ".dcenter")
                )
                if not has_any_placement and not has_data_driven_ranges:
                    result.errors.append(
                        f"Range {gr.start_layer}-{gr.end_layer} has non-zero placement "
                        f"({gr.center_x}, {gr.center_y}) but code has no placement operations"
                    )
                    continue

                required_offsets: list[float] = []
                if gr.center_x != 0.0:
                    required_offsets.append(gr.center_x)
                if gr.center_y != 0.0:
                    required_offsets.append(gr.center_y)

                if (
                    required_offsets
                    and not has_data_driven_ranges
                    and not all(_numeric_literal_present(v) for v in required_offsets)
                ):
                    result.warnings.append(
                        f"Range {gr.start_layer}-{gr.end_layer} has non-zero placement "
                        f"({gr.center_x}, {gr.center_y}) but offset literals were not found in code"
                    )

    return result


def validate_generated_code(spec: GeometrySpec) -> ValidationResult:
    """Full validation of generated code.

    Args:
        spec: GeometrySpec with gdsfactory_code populated

    Returns:
        ValidationResult with all validation details
    """
    import time

    result = ValidationResult(is_valid=False)

    if not spec.gdsfactory_code:
        result.errors.append("No code generated")
        return result

    code = spec.gdsfactory_code

    # 1. Syntax validation
    syntax_ok, syntax_error = validate_syntax(code)
    result.syntax_ok = syntax_ok
    if not syntax_ok:
        result.errors.append(syntax_error or "Unknown syntax error")
        return result

    # 2. Safety validation
    safety_ok, safety_error = validate_safety(code)
    result.safety_ok = safety_ok
    if not safety_ok:
        result.errors.append(safety_error or "Unknown safety issue")
        return result

    # 3. Execution validation (with timeout)
    start_time = time.monotonic()
    exec_ok, exec_error, output_info = validate_execution(code)
    result.execution_time_seconds = round(time.monotonic() - start_time, 2)
    result.executes_ok = exec_ok
    if not exec_ok:
        result.errors.append(exec_error or "Unknown execution error")

    # 4. Output file validation
    result.gds_created = len(output_info.get("gds_files", [])) > 0
    result.oas_created = len(output_info.get("oas_files", [])) > 0

    if exec_ok and not result.gds_created:
        result.warnings.append("Code executed but no GDS file was created")
    if exec_ok and not result.oas_created:
        result.warnings.append("Code executed but no OAS file was created")

    # 5. Spec match validation (errors + warnings)
    spec_match = validate_spec_match(spec, code)
    result.spec_match_ok = len(spec_match.errors) == 0
    result.errors.extend(spec_match.errors)
    result.warnings.extend(spec_match.warnings)

    # Overall validity: syntax + safety + execution + spec match (no critical errors)
    result.is_valid = (
        result.syntax_ok and result.safety_ok and result.executes_ok and result.spec_match_ok
    )

    return result


def generate_output_name(component_type: str) -> str:
    """Generate a unique output name based on component type and timestamp.

    Args:
        component_type: Type of component (e.g., 'via_array')

    Returns:
        Base name like 'via_array_2026-01-29_095030'
    """
    # Sanitize component type for filename
    safe_name = re.sub(r"[^\w\-]", "_", component_type.lower())
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    return f"{safe_name}_{timestamp}"


def export_code_and_files(
    code: str,
    output_dir: Path,
    base_name: str,
) -> ExportResult:
    """Export generated code and GDS/OAS files to output directory.

    Args:
        code: GDSFactory Python code
        output_dir: Directory to save files
        base_name: Base name for files (without extension)

    Returns:
        ExportResult with paths to created files
    """
    import gdsfactory as gf

    # Activate generic PDK if no PDK is active
    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    result = ExportResult(success=False)

    # Save Python code
    py_path = output_dir / f"{base_name}.py"
    py_path.write_text(code)
    result.py_path = py_path

    # Prepare and execute code to generate GDS/OAS
    modified_code = _prepare_code_for_execution(code, output_dir, base_name)

    # Disable show() to prevent GUI popup
    modified_code = modified_code.replace(".show()", "# .show() disabled for export")

    try:
        exec_globals = {
            "__name__": "__main__",
            "gf": gf,
        }
        exec(modified_code, exec_globals)

        # Check for created files
        gds_path = output_dir / f"{base_name}.gds"
        oas_path = output_dir / f"{base_name}.oas"

        if gds_path.exists():
            result.gds_path = gds_path
        if oas_path.exists():
            result.oas_path = oas_path

        result.success = True

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        # Code file is still saved even if execution fails

    return result
