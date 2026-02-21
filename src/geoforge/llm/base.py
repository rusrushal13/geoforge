"""Abstract base class for LLM providers."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from geoforge.core.logging import PipelineLogger

from pydantic import BaseModel, Field, ValidationError, model_validator
from rich.console import Console

console = Console()

# Default retry settings
DEFAULT_MAX_RETRIES = 3


class LayerSpec(BaseModel):
    """Specification for a single layer in the geometry."""

    layer_number: int = Field(description="GDS layer number")
    datatype: int = Field(default=0, description="GDS datatype")
    name: str = Field(description="Layer name (e.g., 'metal1', 'via1')")
    thickness_nm: float | None = Field(default=None, description="Layer thickness in nanometers")
    material: str | None = Field(
        default=None, description="Material name (e.g., 'copper', 'oxide')"
    )


class GeometryRange(BaseModel):
    """Specification for a contiguous range of layers sharing geometry rules.

    Used by multi_layer_rectangles to express per-range sizing and placement.
    Linear interpolation is applied from start_layer to end_layer.
    """

    start_layer: int = Field(ge=1, description="First layer number in this range (inclusive)")
    end_layer: int = Field(ge=1, description="Last layer number in this range (inclusive)")
    shape: Literal["rectangle", "square"] = Field(
        default="rectangle",
        description="Shape type: 'rectangle' or 'square'",
    )
    width_start: float = Field(gt=0, description="Width (um) at start_layer")
    width_end: float = Field(gt=0, description="Width (um) at end_layer")
    height_start: float = Field(gt=0, description="Height (um) at start_layer")
    height_end: float = Field(gt=0, description="Height (um) at end_layer")
    center_x: float = Field(default=0.0, description="X center offset (um)")
    center_y: float = Field(default=0.0, description="Y center offset (um)")

    @model_validator(mode="after")
    def _validate_geometry_range(self) -> "GeometryRange":
        """Apply cross-field checks for geometry range consistency."""
        if self.end_layer < self.start_layer:
            raise ValueError("end_layer must be greater than or equal to start_layer")

        if self.shape == "square":
            if abs(self.width_start - self.height_start) > 1e-9:
                raise ValueError("Square geometry_range requires width_start == height_start")
            if abs(self.width_end - self.height_end) > 1e-9:
                raise ValueError("Square geometry_range requires width_end == height_end")

        return self


PrimitiveKind = Literal["rectangle", "circle", "polygon"]


class PrimitiveSpec(BaseModel):
    """Explicit primitive instruction used for custom logos and art-like geometry."""

    primitive_type: PrimitiveKind = Field(
        description="Primitive type: rectangle, circle, or polygon"
    )
    layer_number: int = Field(ge=1, description="GDS layer number for this primitive")
    datatype: int = Field(default=0, ge=0, description="GDS datatype for this primitive")
    center_x: float = Field(default=0.0, description="Primitive center X coordinate (um)")
    center_y: float = Field(default=0.0, description="Primitive center Y coordinate (um)")
    width: float | None = Field(default=None, gt=0, description="Width (um) for rectangles")
    height: float | None = Field(default=None, gt=0, description="Height (um) for rectangles")
    radius: float | None = Field(default=None, gt=0, description="Radius (um) for circles")
    points: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Polygon points as [x, y] coordinate pairs in um",
    )
    rotation_deg: float = Field(default=0.0, description="Rotation angle in degrees")
    name: str | None = Field(default=None, description="Optional primitive label")

    @model_validator(mode="after")
    def _validate_shape_fields(self) -> "PrimitiveSpec":
        """Ensure required per-shape fields are present."""
        if self.primitive_type == "rectangle":
            if self.width is None or self.height is None:
                raise ValueError("Rectangle primitive requires both width and height")
        elif self.primitive_type == "circle":
            if self.radius is None:
                raise ValueError("Circle primitive requires radius")
        elif self.primitive_type == "polygon" and len(self.points) < 3:
            raise ValueError("Polygon primitive requires at least 3 points")
        return self


class GeometrySpec(BaseModel):
    """Structured specification for geometry generation."""

    component_type: str = Field(description="Type of component (e.g., 'via_array', 'interconnect')")
    description: str = Field(description="Human-readable description of the geometry")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Component-specific parameters"
    )
    layers: list[LayerSpec] = Field(default_factory=list, description="Layer definitions")
    geometry_ranges: list[GeometryRange] = Field(
        default_factory=list,
        description="Per-range geometry specs for multi-layer components (optional)",
    )
    primitives: list[PrimitiveSpec] = Field(
        default_factory=list,
        description=(
            "Optional explicit primitives with coordinates for custom logos and "
            "art-like geometry requests"
        ),
    )
    gdsfactory_code: str | None = Field(
        default=None, description="Generated GDSFactory Python code"
    )


ErrorCategory = Literal[
    "json_parse",
    "schema_validation",
    "code_too_short",
    "missing_import",
    "syntax_error",
    "execution_error",
    "other",
]


class RetryContext(BaseModel):
    """Context about a previous failed attempt, fed back to the LLM on retry."""

    attempt_number: int
    previous_error: str
    previous_response_snippet: str | None = None
    error_category: ErrorCategory = "other"


class PromptEnhancement(BaseModel):
    """Normalized prompt text produced by the prompt-enhancer stage."""

    rewritten_prompt: str = Field(min_length=1, description="Constraint-focused rewritten prompt")
    key_constraints: list[str] = Field(
        default_factory=list,
        description="Most important geometry constraints extracted from the user intent",
    )


# Common schema prompt for all providers
GEOMETRY_SPEC_SCHEMA = """
IMPORTANT: Respond ONLY with valid JSON matching this EXACT schema:
{
    "component_type": "string (e.g., 'via_array', 'interconnect')",
    "description": "string describing the geometry",
    "parameters": {"key": "value pairs with numeric values as numbers, not strings"},
    "layers": [
        {
            "layer_number": 1,
            "datatype": 0,
            "name": "metal1",
            "thickness_nm": 100.0,
            "material": "copper"
        }
    ],
    "geometry_ranges": [
        {
            "start_layer": 1,
            "end_layer": 100,
            "shape": "rectangle",
            "width_start": 1000.0,
            "width_end": 100000.0,
            "height_start": 1000.0,
            "height_end": 100000.0,
            "center_x": 0.0,
            "center_y": 0.0
        }
    ],
    "primitives": [
        {
            "primitive_type": "polygon",
            "layer_number": 1,
            "datatype": 0,
            "center_x": 0.0,
            "center_y": 0.0,
            "width": null,
            "height": null,
            "radius": null,
            "points": [[-20.0, -10.0], [0.0, 30.0], [20.0, -10.0]],
            "rotation_deg": 0.0,
            "name": "outline"
        }
    ]
}

CRITICAL RULES:
- layer_number MUST be an integer (e.g., 1, not "1")
- datatype MUST be an integer (e.g., 0, not "0")
- name is REQUIRED for each layer
- All numeric values must be numbers, not strings
- geometry_ranges is OPTIONAL — use it ONLY for multi_layer_rectangles when layers
  have different sizing rules or positions per range. Omit it for other component types.
- Each range uses linear interpolation from width_start/height_start at start_layer
  to width_end/height_end at end_layer.
- For squares, set width_start == height_start and width_end == height_end.
- primitives is OPTIONAL — use it for custom logos, silhouettes, symbols, and when
  the user asks for explicit shape-level control.
- primitive_type can be rectangle, circle, or polygon.
- rectangle requires width and height. circle requires radius.
- polygon requires at least 3 points in [[x, y], ...] format (all in um).
- Keep primitive layer_number/datatype consistent with the layers list.
"""


PROMPT_ENHANCEMENT_SCHEMA = """
IMPORTANT: Respond ONLY with valid JSON matching this EXACT schema:
{
    "rewritten_prompt": "string with explicit geometry constraints, units, and layer intent",
    "key_constraints": [
        "short bullet-like constraint strings"
    ]
}

CRITICAL RULES:
- rewritten_prompt MUST preserve user intent and required dimensions.
- Keep all geometry units in um unless explicitly requested otherwise.
- Do not remove constraints like centering, layer count, or export requirements.
- key_constraints should be concise and concrete (3-8 items preferred).
"""


def _classify_error(error: Exception, raw_response: str | None = None) -> ErrorCategory:  # noqa: ARG001, PLR0911
    """Classify an error for the RetryContext."""
    if isinstance(error, json.JSONDecodeError):
        return "json_parse"
    if isinstance(error, ValidationError):
        return "schema_validation"
    msg = str(error).lower()
    if "too short" in msg or "empty" in msg:
        return "code_too_short"
    if "import" in msg or "gdsfactory" in msg:
        return "missing_import"
    if "syntax" in msg:
        return "syntax_error"
    if any(
        token in msg
        for token in (
            "execution",
            "runtime",
            "attributeerror",
            "typeerror",
            "nameerror",
            "keyerror",
            "indexerror",
            "zerodivisionerror",
            "timed out",
        )
    ):
        return "execution_error"
    return "other"


def _format_retry_message(retry_context: RetryContext) -> str:
    """Format a retry context into a correction message for the LLM."""
    msg = (
        f"Your previous response (attempt {retry_context.attempt_number}) had an error.\n"
        f"Error type: {retry_context.error_category}\n"
        f"Error details: {retry_context.previous_error}\n"
    )
    if retry_context.previous_response_snippet:
        msg += (
            f"\nYour previous response started with:\n{retry_context.previous_response_snippet}\n"
        )
    if retry_context.error_category == "execution_error":
        msg += (
            "\nExecution fix hints:\n"
            "- Use stable gdsfactory APIs that exist across versions.\n"
            "- Do NOT use Path.end_point or Path.end_orientation.\n"
            "- For path endpoints/orientation, use ports from the extruded component "
            "reference (for example, ref.ports['o2']).\n"
        )
    msg += "\nPlease fix the issue and try again."
    return msg


def _format_prompt_enhancement_retry_message(retry_context: RetryContext) -> str:
    """Format retry feedback specifically for prompt-enhancement failures."""
    msg = (
        f"Your previous prompt-enhancement response (attempt {retry_context.attempt_number}) "
        "had an error.\n"
        f"Error type: {retry_context.error_category}\n"
        f"Error details: {retry_context.previous_error}\n"
    )
    if retry_context.previous_response_snippet:
        msg += (
            f"\nYour previous response started with:\n{retry_context.previous_response_snippet}\n"
        )
    msg += (
        "\nReturn ONLY valid JSON with rewritten_prompt and key_constraints."
        "\nKeep the same user intent and constraints."
    )
    return msg


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Includes built-in retry logic with error feedback for handling malformed LLM responses.
    Subclasses implement _generate_geometry_spec_impl and _generate_gdsfactory_code_impl.
    """

    name: str = "base"
    max_retries: int = DEFAULT_MAX_RETRIES

    @abstractmethod
    async def _generate_geometry_spec_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        """Implementation: Convert prompt to raw JSON dict.

        Args:
            prompt: Natural language description of desired geometry.
            retry_context: Context about previous failed attempt (if retrying).

        Returns:
            Raw dictionary from LLM response (will be validated by base class).
        """

    @abstractmethod
    async def _generate_gdsfactory_code_impl(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
        retry_context: RetryContext | None = None,
    ) -> str:
        """Implementation: Generate GDSFactory code from specification.

        Args:
            spec: Structured geometry specification.
            original_prompt: The original natural-language user request (optional).
            retry_context: Context about previous failed attempt (if retrying).

        Returns:
            Raw code string from LLM (will be validated by base class).
        """

    async def _enhance_prompt_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,  # noqa: ARG002
    ) -> dict:
        """Implementation hook for prompt-enhancement.

        Providers can override this to call their native model API. The default
        implementation is a no-op passthrough that preserves existing behavior.
        """
        return {"rewritten_prompt": prompt, "key_constraints": []}

    async def enhance_prompt(self, prompt: str) -> PromptEnhancement:
        """Rewrite user text into a constraint-focused geometry prompt.

        This stage is fail-open: if enhancement repeatedly fails, the original
        prompt is returned so the pipeline can still proceed.
        """
        last_error = None
        retry_context: RetryContext | None = None
        last_raw_response: str | None = None

        for attempt in range(self.max_retries):
            try:
                data = await self._enhance_prompt_impl(prompt, retry_context)

                if isinstance(data, str):
                    last_raw_response = data
                    stripped = data.strip()
                    data = (
                        json.loads(stripped)
                        if stripped.startswith("{")
                        else {"rewritten_prompt": stripped, "key_constraints": []}
                    )
                else:
                    last_raw_response = json.dumps(data)

                if not isinstance(data, dict):
                    raise TypeError("Prompt enhancement response must be a JSON object")

                enhancement = PromptEnhancement(**cast("dict[str, Any]", data))
                if not enhancement.rewritten_prompt.strip():
                    raise ValueError("rewritten_prompt is empty")
                return enhancement

            except Exception as e:
                last_error = str(e)
                retry_context = RetryContext(
                    attempt_number=attempt + 1,
                    previous_error=last_error,
                    previous_response_snippet=last_raw_response[:500]
                    if last_raw_response
                    else None,
                    error_category=_classify_error(e, last_raw_response),
                )
                console.print(
                    f"[yellow]Attempt {attempt + 1}/{self.max_retries}: "
                    "Prompt enhancement issue, retrying...[/yellow]"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(min(2**attempt, 8))

        console.print(
            "[yellow]Prompt enhancement failed after retries; "
            "continuing with original prompt.[/yellow]"
        )
        if last_error:
            console.print(f"[dim]Enhancement error: {last_error}[/dim]")
        return PromptEnhancement(rewritten_prompt=prompt, key_constraints=[])

    async def generate_geometry_spec(self, prompt: str) -> GeometrySpec:
        """Convert natural language prompt to structured geometry specification.

        Includes error-aware retry logic that feeds failure context back to the LLM.

        Args:
            prompt: Natural language description of desired geometry.

        Returns:
            Validated GeometrySpec object.
        """
        last_error = None
        retry_context: RetryContext | None = None
        last_raw_response: str | None = None

        for attempt in range(self.max_retries):
            try:
                # Get raw response from provider (with retry context on subsequent attempts)
                data = await self._generate_geometry_spec_impl(prompt, retry_context)

                # Handle string response (parse JSON)
                if isinstance(data, str):
                    last_raw_response = data
                    data = json.loads(data)
                else:
                    last_raw_response = json.dumps(data)

                # Validate with Pydantic
                return GeometrySpec(**data)

            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                retry_context = RetryContext(
                    attempt_number=attempt + 1,
                    previous_error=last_error,
                    previous_response_snippet=last_raw_response[:500]
                    if last_raw_response
                    else None,
                    error_category="json_parse",
                )
                console.print(
                    f"[yellow]Attempt {attempt + 1}/{self.max_retries}: "
                    f"Malformed JSON, retrying with error feedback...[/yellow]"
                )

            except ValidationError as e:
                error_details = "; ".join(f"{err['loc']}: {err['msg']}" for err in e.errors()[:3])
                last_error = f"Schema validation failed: {error_details}"
                retry_context = RetryContext(
                    attempt_number=attempt + 1,
                    previous_error=last_error,
                    previous_response_snippet=last_raw_response[:500]
                    if last_raw_response
                    else None,
                    error_category="schema_validation",
                )
                console.print(
                    f"[yellow]Attempt {attempt + 1}/{self.max_retries}: "
                    f"Invalid schema, retrying with error feedback...[/yellow]"
                )

            except Exception as e:
                last_error = str(e)
                retry_context = RetryContext(
                    attempt_number=attempt + 1,
                    previous_error=last_error,
                    previous_response_snippet=last_raw_response[:500]
                    if last_raw_response
                    else None,
                    error_category=_classify_error(e, last_raw_response),
                )
                console.print(
                    f"[yellow]Attempt {attempt + 1}/{self.max_retries}: "
                    f"Error: {e}, retrying with error feedback...[/yellow]"
                )

            # Exponential backoff for API rate limits
            if attempt < self.max_retries - 1:
                await asyncio.sleep(min(2**attempt, 8))

        # All retries failed
        raise ValueError(
            f"Failed to generate valid geometry spec after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def generate_gdsfactory_code(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
    ) -> str:
        """Generate GDSFactory Python code from a geometry specification.

        Includes error-aware retry logic with syntax and execution validation in-the-loop.

        Args:
            spec: Structured geometry specification.
            original_prompt: The original natural-language user request (optional).

        Returns:
            Valid GDSFactory Python code as a string.
        """
        from geoforge.core.validator import validate_execution, validate_syntax

        last_error = None
        retry_context: RetryContext | None = None
        last_code: str | None = None

        for attempt in range(self.max_retries):
            try:
                # Get raw code from provider (with retry context on subsequent attempts)
                code = await self._generate_gdsfactory_code_impl(
                    spec,
                    original_prompt,
                    retry_context,
                )

                # Extract from markdown if needed
                code = self._extract_code_from_markdown(code)
                last_code = code

                # Basic validation
                if not code or len(code) < 50:
                    raise ValueError("Generated code is too short or empty")

                if "gdsfactory" not in code and "gf" not in code:
                    raise ValueError("Generated code doesn't import gdsfactory")

                # Syntax validation (catch errors before returning)
                syntax_ok, syntax_error = validate_syntax(code)
                if not syntax_ok:
                    raise SyntaxError(f"Generated code has syntax error: {syntax_error}")

                # Execution validation (catch runtime/API errors before returning)
                exec_ok, exec_error, output_info = validate_execution(code)
                if not exec_ok:
                    raise RuntimeError(f"Generated code failed execution validation: {exec_error}")

                # Require artifact generation so downstream export is reliable
                if not output_info.get("gds_files") and not output_info.get("oas_files"):
                    raise RuntimeError(
                        "Generated code executed but did not create GDS/OAS output files"
                    )

                return code

            except Exception as e:
                last_error = str(e)
                retry_context = RetryContext(
                    attempt_number=attempt + 1,
                    previous_error=last_error,
                    previous_response_snippet=last_code[:500] if last_code else None,
                    error_category=_classify_error(e, last_code),
                )
                console.print(
                    f"[yellow]Attempt {attempt + 1}/{self.max_retries}: "
                    f"Code generation issue: {last_error[:80]}..., "
                    f"retrying with error feedback...[/yellow]"
                )

            # Exponential backoff for API rate limits
            if attempt < self.max_retries - 1:
                await asyncio.sleep(min(2**attempt, 8))

        # All retries failed
        raise ValueError(
            f"Failed to generate valid code after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _extract_code_from_markdown(text: str) -> str:
        """Extract Python code from markdown code blocks."""
        if "```python" in text:
            text = text.split("```python")[1].split("```", maxsplit=1)[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    async def generate(
        self,
        prompt: str,
        logger: "PipelineLogger | None" = None,
    ) -> GeometrySpec:
        """Full pipeline: prompt -> spec -> code.

        Args:
            prompt: Natural language description of desired geometry.
            logger: Optional PipelineLogger for structured debug logging.

        Returns:
            GeometrySpec with gdsfactory_code populated.
        """
        enhancement = await self.enhance_prompt(prompt)
        enhanced_prompt = enhancement.rewritten_prompt.strip()
        if enhanced_prompt and enhanced_prompt != prompt.strip():
            console.print("[dim]Prompt enhanced for better geometry extraction.[/dim]")

        if logger:
            logger.start("spec_generation")
        try:
            spec = await self.generate_geometry_spec(enhanced_prompt or prompt)
            if logger:
                logger.success(
                    "spec_generation",
                    component_type=spec.component_type,
                    prompt_enhanced=bool(enhanced_prompt and enhanced_prompt != prompt.strip()),
                    enhancement_constraints=enhancement.key_constraints,
                )
        except Exception as e:
            if logger:
                logger.log_error("spec_generation", str(e))
            raise

        # Check if a deterministic template exists for this component type.
        # If so, skip LLM stage-2 entirely for reproducible, fast generation.
        from geoforge.templates import generate_from_template, has_template

        if has_template(spec.component_type):
            if logger:
                logger.start("code_generation")
            try:
                code = generate_from_template(spec)
                spec.gdsfactory_code = code
                if logger:
                    logger.success(
                        "code_generation",
                        code_length=len(code),
                        method="template",
                    )
                console.print(
                    f"[green]Using deterministic template for '{spec.component_type}'[/green]"
                )
            except Exception as e:
                if logger:
                    logger.log_error("code_generation", str(e))
                # Fall through to LLM generation on template failure
                console.print(f"[yellow]Template failed ({e}), falling back to LLM...[/yellow]")
            else:
                return spec

        if logger:
            logger.start("code_generation")
        try:
            code = await self.generate_gdsfactory_code(spec, original_prompt=prompt)
            spec.gdsfactory_code = code
            if logger:
                logger.success("code_generation", code_length=len(code), method="llm")
        except Exception as e:
            if logger:
                logger.log_error("code_generation", str(e))
            raise

        return spec
