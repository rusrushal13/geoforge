"""Pipeline handler functions for the web UI."""

from __future__ import annotations

import asyncio
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from geoforge.config import settings

if TYPE_CHECKING:
    from geoforge.llm.base import GeometrySpec


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    success: bool = False
    spec: GeometrySpec | None = None
    code: str | None = None
    validation_markdown: str | None = None
    preview_path: Path | None = None
    gds_path: Path | None = None
    oas_path: Path | None = None
    py_path: Path | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)


_PIPELINE_DEDUPE_TTL_SECONDS = 2.0
_PIPELINE_REQUEST_LOCK = threading.Lock()
_INFLIGHT_PIPELINES: dict[tuple[str, str], asyncio.Task[PipelineResult]] = {}
_RECENT_PIPELINE_RESULTS: dict[tuple[str, str], tuple[float, PipelineResult]] = {}


def _request_key(prompt: str, provider_name: str) -> tuple[str, str]:
    """Build a stable key for deduplicating near-identical web requests."""
    normalized_prompt = " ".join(prompt.split())
    return provider_name.strip().lower(), normalized_prompt


def _prune_expired_recent_results(now: float) -> None:
    """Remove expired dedupe entries to keep memory bounded."""
    expired_keys = [
        key
        for key, (timestamp, _) in _RECENT_PIPELINE_RESULTS.items()
        if now - timestamp > _PIPELINE_DEDUPE_TTL_SECONDS
    ]
    for key in expired_keys:
        _RECENT_PIPELINE_RESULTS.pop(key, None)


def get_provider_choices() -> list[str]:
    """Get list of available (configured) provider names."""
    choices = []
    cloud = settings.get_cloud_providers()

    # Ollama always available as option
    choices.append("ollama")
    choices.extend(cloud)
    return choices


def format_validation_markdown(result) -> str:
    """Format a ValidationResult as markdown for the web UI."""
    lines = ["## Validation Results\n"]

    checks = [
        ("Syntax", result.syntax_ok),
        ("Safety", result.safety_ok),
        ("Execution", result.executes_ok),
        ("Spec Match", result.spec_match_ok),
        ("GDS Created", result.gds_created),
        ("OAS Created", result.oas_created),
    ]

    for name, passed in checks:
        icon = "+" if passed else "-"
        lines.append(f"  {icon} **{name}**: {'Pass' if passed else 'Fail'}")

    if result.execution_time_seconds is not None:
        lines.append(f"\nExecution time: {result.execution_time_seconds}s")

    if result.errors:
        lines.append("\n### Errors")
        lines.extend(f"- {err}" for err in result.errors)

    if result.warnings:
        lines.append("\n### Warnings")
        lines.extend(f"- {warn}" for warn in result.warnings)

    overall = "Valid" if result.is_valid else "Invalid"
    lines.append(f"\n**Overall: {overall}**")

    return "\n".join(lines)


def format_spec_markdown(spec: GeometrySpec) -> str:
    """Format a GeometrySpec as markdown."""
    lines = [f"**Component Type:** {spec.component_type}\n"]
    lines.append(f"**Description:** {spec.description}\n")

    if spec.parameters:
        lines.append("### Parameters")
        for k, v in spec.parameters.items():
            lines.append(f"- `{k}`: {v}")

    if spec.layers:
        lines.append("\n### Layers")
        for layer in spec.layers:
            detail = f"({layer.layer_number}, {layer.datatype}) {layer.name}"
            if layer.material:
                detail += f" - {layer.material}"
            if layer.thickness_nm:
                detail += f" ({layer.thickness_nm}nm)"
            lines.append(f"- {detail}")

    if spec.geometry_ranges:
        lines.append("\n### Geometry Ranges")
        for r in spec.geometry_ranges:
            detail = (
                f"Layers {r.start_layer}-{r.end_layer}: {r.shape}, "
                f"width {r.width_start}-{r.width_end} um, "
                f"height {r.height_start}-{r.height_end} um"
            )
            if r.center_x != 0.0 or r.center_y != 0.0:
                detail += f", center ({r.center_x}, {r.center_y})"
            lines.append(f"- {detail}")

    return "\n".join(lines)


async def _run_pipeline_once(prompt: str, provider_name: str) -> PipelineResult:
    """Run the full generation pipeline exactly once.

    Args:
        prompt: Natural language geometry description.
        provider_name: Name of the LLM provider to use.

    Returns:
        PipelineResult with all outputs.
    """
    from geoforge.core.validator import (
        export_code_and_files,
        generate_output_name,
        validate_generated_code,
    )
    from geoforge.llm import get_provider
    from geoforge.viz.renderer import render_code, render_gds_file

    result = PipelineResult()

    # 1. Get provider
    try:
        llm = get_provider(provider_name)
        result.logs.append(f"Using provider: {provider_name}")
    except Exception as e:
        result.error = f"Provider error: {e}"
        return result

    # 2. Generate spec + code
    try:
        spec = await llm.generate(prompt)
        result.spec = spec
        result.code = spec.gdsfactory_code
        result.logs.append(f"Generated spec: {spec.component_type}")
    except Exception as e:
        result.error = f"Generation failed: {e}"
        return result

    if not spec.gdsfactory_code:
        result.error = "No code was generated"
        return result

    # 3. Validate
    validation_passed = False
    try:
        validation = validate_generated_code(spec)
        validation_passed = validation.is_valid
        result.validation_markdown = format_validation_markdown(validation)
        result.logs.append(f"Validation: {'passed' if validation.is_valid else 'failed'}")
    except Exception as e:
        result.validation_markdown = f"Validation error: {e}"
        result.logs.append(f"Validation error: {e}")

    # 4. Export files (only if validation passed)
    if validation_passed:
        try:
            output_dir = settings.output_dir
            base_name = generate_output_name(spec.component_type)
            export = export_code_and_files(spec.gdsfactory_code, output_dir, base_name)

            result.py_path = export.py_path
            result.gds_path = export.gds_path
            result.oas_path = export.oas_path
            result.logs.append(f"Exported to: {output_dir / base_name}")
        except Exception as e:
            result.logs.append(f"Export error: {e}")
    else:
        result.logs.append("Export skipped: validation failed")

    # 5. Render preview (only if validation passed)
    if validation_passed:
        try:
            tmpdir = Path(tempfile.mkdtemp())
            png_path = tmpdir / "preview.png"

            # Reuse exported GDS when available to avoid another code execution pass.
            if result.gds_path and result.gds_path.exists():
                fig = render_gds_file(result.gds_path, save_path=png_path)
            else:
                fig = render_code(spec.gdsfactory_code, save_path=png_path)

            if fig:
                result.preview_path = png_path
                import matplotlib.pyplot as plt

                plt.close(fig)
                result.logs.append("Preview rendered")
        except Exception as e:
            result.logs.append(f"Preview error: {e}")

    result.success = validation_passed
    return result


async def run_pipeline(prompt: str, provider_name: str) -> PipelineResult:
    """Run the generation pipeline with short-window request deduplication."""
    key = _request_key(prompt, provider_name)
    now = monotonic()
    _prune_expired_recent_results(now)

    cached = _RECENT_PIPELINE_RESULTS.get(key)
    if cached and now - cached[0] <= _PIPELINE_DEDUPE_TTL_SECONDS:
        return cached[1]

    is_owner = False

    with _PIPELINE_REQUEST_LOCK:
        task = _INFLIGHT_PIPELINES.get(key)
        if task is None:
            task = asyncio.create_task(_run_pipeline_once(prompt, provider_name))
            _INFLIGHT_PIPELINES[key] = task
            is_owner = True

    try:
        result = await task
    finally:
        if is_owner:
            with _PIPELINE_REQUEST_LOCK:
                _INFLIGHT_PIPELINES.pop(key, None)

    _RECENT_PIPELINE_RESULTS[key] = (monotonic(), result)
    return result
