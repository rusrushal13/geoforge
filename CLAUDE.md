# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoForge is a natural language semiconductor geometry generator. It converts plain English descriptions of semiconductor structures into validated GDS/OASIS CAD files using LLMs and [GDSFactory](https://gdsfactory.github.io/gdsfactory/). The tool supports multiple LLM providers (Ollama for local/free, plus Anthropic, Google Gemini, and OpenAI cloud providers) and includes a CLI, a Gradio web interface, and a visualization/rendering pipeline.

**Full Pipeline**:

```text
User prompt (natural language)
  → LLM Stage 1: Extract structured GeometrySpec (JSON with Pydantic validation)
  → LLM Stage 2: Generate GDSFactory Python code from spec
  → Validate: Syntax (AST) → Safety (banned imports) → Execution (sandbox, 30s timeout)
  → Export: .py source + .gds + .oas files to examples/outputs/
  → Optional: Render PNG preview, launch in KLayout
```

Each LLM stage has built-in retry logic (3 attempts with exponential backoff) that feeds error context back to the LLM for self-correction.

## Tech Stack

- **Python**: 3.13+ (required)
- **Build system**: hatchling
- **Package manager**: uv
- **CLI framework**: Typer + Rich
- **Data validation**: Pydantic v2 + Pydantic Settings
- **CAD engine**: GDSFactory 9.32+
- **Web UI**: Gradio 6.5+
- **Visualization**: matplotlib (Agg backend for headless rendering)
- **LLM clients**: ollama, anthropic, google-genai, openai
- **Testing**: pytest + pytest-asyncio + pytest-cov + pytest-timeout
- **Linting/Formatting**: ruff, bandit, ty (type checker), codespell, pymarkdown
- **Git hooks**: pre-commit

## Commands

### Running the CLI

```bash
# Generate geometry (validates + saves + exports by default)
geoforge generate "Create a 5x5 via array with 1um pitch"

# With options
geoforge generate "..." --provider gemini --output my_design --no-validate
geoforge generate "..." --preview      # Also render PNG preview
geoforge generate "..." --execute      # Open result in KLayout
geoforge generate "..." --debug        # Save pipeline log JSON
geoforge generate "..." --show-code    # Display generated code

# Validate existing generated code
geoforge validate examples/outputs/via_array.py

# List available LLM providers (shows Ollama status + configured cloud providers)
geoforge providers

# Launch Gradio web interface
geoforge web --host 0.0.0.0 --port 7860 --share

# Show version
geoforge version
```

### Testing

```bash
uv run pytest                              # All tests (~135 tests)
uv run pytest tests/test_llm.py            # Single file
uv run pytest -m "not slow"                # Skip slow tests (GDSFactory execution)
uv run pytest -m integration               # Integration tests only
uv run pytest --cov=src/geoforge           # With coverage
uv run pytest --timeout=60                 # With timeout per test
```

### Code Quality (all run via pre-commit on commit)

```bash
uv run ruff format src/ tests/             # Format
uv run ruff check src/ tests/ --fix        # Lint with auto-fix
uv run ty check src/                       # Type check
uv run bandit -r src/ -c pyproject.toml    # Security scan
uv run pre-commit run --all-files          # Run all hooks at once
```

## Project Structure

```text
src/geoforge/
├── __init__.py              # Package init, exports __version__ = "0.1.0"
├── cli.py                   # Typer CLI app (generate, validate, providers, web, version)
├── config.py                # Pydantic Settings (env vars, API keys, model names)
├── core/
│   ├── __init__.py
│   ├── logging.py           # PipelineLogger: structured event logging with timing
│   └── validator.py         # Syntax, safety, execution validation + export
├── llm/
│   ├── __init__.py          # Public API: GeometrySpec, LLMProvider, get_provider
│   ├── base.py              # Abstract LLMProvider, GeometrySpec/LayerSpec models, retry logic
│   ├── registry.py          # Factory pattern with lazy-loading provider registry
│   ├── anthropic.py         # Claude provider (claude-sonnet-4-5)
│   ├── google.py            # Gemini provider (gemini-2.5-flash)
│   ├── ollama.py            # Local Ollama provider (qwen2.5-coder:14b, default)
│   └── openai.py            # OpenAI provider (gpt-5.2)
├── prompts/
│   ├── __init__.py          # Public API for prompt templates
│   └── gdsfactory.py        # GEOMETRY_SPEC_PROMPT + COMPONENT_EXAMPLES + build_code_prompt()
├── viz/
│   ├── __init__.py          # Public API: render_code, render_component, render_gds_file
│   └── renderer.py          # Render GDS/components to PNG (matplotlib Agg backend)
└── web/
    ├── __init__.py          # Public API: create_app, launch
    ├── app.py               # Gradio Blocks UI (prompt input, tabs, file downloads)
    └── handlers.py          # run_pipeline(), format helpers, PipelineResult dataclass

tests/
├── conftest.py              # Shared fixtures: sample_prompt, sample_spec (GeometrySpec)
├── mock_provider.py         # MockLLMProvider with configurable failures + call counting
├── test_cli.py              # 10 tests: CLI commands, help text, argument validation
├── test_config.py           # 2 tests: Settings defaults, provider availability
├── test_geometry.py         # 39 tests: Multi-layer specs, layer relationships, geometry ranges
├── test_integration.py      # 15 tests: Full pipeline, PipelineLogger, export (async)
├── test_llm.py              # 5 tests: Provider registry, GeometrySpec/LayerSpec models
├── test_retry.py            # 14 tests: Error classification, RetryContext, retry behavior
├── test_spec_match.py       # 13 tests: Spec-to-code validation (layers, component type, ranges)
├── test_validator.py        # 35 tests: Syntax, safety, execution, output name, export
├── test_viz.py              # 9 tests: render_component, render_code, render_gds_file
└── test_web.py              # 9 tests: Provider choices, markdown formatting, Gradio app
```

## Architecture Deep Dive

### LLM Provider System (`src/geoforge/llm/`)

**Base class** (`base.py`):
- `LLMProvider` is an abstract base class. Subclasses implement two async methods:
  - `_generate_geometry_spec_impl(prompt, retry_context)` → dict (JSON)
  - `_generate_gdsfactory_code_impl(spec, original_prompt, retry_context)` → str (Python code)
- The base class wraps these with retry logic in the public methods `generate_geometry_spec()` and `generate_gdsfactory_code()`, plus a combined `generate()` method that runs both stages.
- **Retry logic**: Up to 3 attempts per stage. On failure, errors are classified into categories (`json_parse`, `schema_validation`, `code_too_short`, `missing_import`, `syntax_error`, `execution_error`, `other`) and fed back to the LLM via `RetryContext` so it can self-correct. Exponential backoff: `min(2^attempt, 8)` seconds.
- **Code validation in the retry loop**: Generated code is checked for syntax (AST parse), minimum length (50+ chars), and required imports (`gdsfactory` or `gf`) before being accepted.

**Data models** (`base.py`):
- `GeometrySpec`: component_type, description, parameters (dict), layers (list[LayerSpec]), geometry_ranges (list[GeometryRange], optional), gdsfactory_code (populated after stage 2)
- `GeometryRange`: start_layer, end_layer, shape, width_start, width_end, height_start, height_end, center_x, center_y — used by multi_layer_rectangles for per-range sizing and placement
- `LayerSpec`: layer_number (int), datatype (int, default 0), name (str), thickness_nm (optional), material (optional)
- `RetryContext`: attempt_number, previous_error, previous_response_snippet, error_category

**Registry** (`registry.py`):
- Factory pattern: `get_provider(name)` lazily imports and instantiates the requested provider.
- `register_provider(name)` decorator for registration. Providers are loaded on first use to avoid circular imports and unnecessary dependencies.
- `list_providers()` returns all registered names.

**Provider implementations**:
- `ollama.py`: Default provider. Auto-pulls models if missing. Uses `format="json"` for spec generation. Multi-turn retry (appends previous response + error as new messages). Checks Ollama server availability with helpful error messages.
- `anthropic.py`: Uses `system` parameter for system prompt. max_tokens: 4096 (spec) / 8192 (code).
- `google.py`: Uses `response_mime_type="application/json"` for spec. Async via `aio.models.generate_content()`.
- `openai.py`: Uses `response_format={"type": "json_object"}` for spec. Standard ChatCompletions API.

### Validation & Security (`src/geoforge/core/validator.py`)

Five-stage validation pipeline via `validate_generated_code(spec)`:

1. **`validate_syntax(code)`**: AST parsing. Returns (is_valid, error_message).
2. **`validate_safety(code)`**: AST-based import and call scanning.
   - Allowed: `gdsfactory`, `gf`, `math`, `numpy`, `np`, `functools`, `itertools`, `collections`, `typing`, `pathlib`
   - Banned imports: `os`, `subprocess`, `shutil`, `sys`, `socket`, `http`, `urllib`, `requests`, `ctypes`, `signal`, `multiprocessing`, `threading`, `importlib`
   - Banned calls: `__import__()`, `eval()`, `compile()`, `exec()` (prevents dynamic import/code execution bypass)
3. **`validate_execution(code, timeout=30)`**: Runs code in a `multiprocessing.Process` with a temp directory. Captures generated .gds and .oas files. Graceful termination (SIGTERM, then SIGKILL after 5s). Returns (success, error, output_info with file paths).
4. **Output file check**: Verifies GDS/OAS files were actually created.
5. **`validate_spec_match(spec, code)`**: Semantic check that the generated code references the expected component type, layer tuples, and parameters. Returns `SpecMatchResult` with errors and warnings.

**Export** (`export_code_and_files()`): Saves .py file, executes code to generate .gds/.oas, returns `ExportResult` with all paths.

**Code preparation**: Before execution, `_prepare_code_for_execution()` injects output directory variables and redirects `.write_gds()`/`.write()` calls to the target directory using regex substitution.

### CLI (`src/geoforge/cli.py`, 455 lines)

Typer app with five commands:
- **`generate`**: Full pipeline. Options: `--provider`, `--output-dir`, `--output`, `--show-code`, `--execute`, `--validate/--no-validate`, `--save/--no-save`, `--preview`, `--debug`. Interactive provider selection when multiple are available.
- **`validate`**: Standalone file validation (syntax + execution).
- **`providers`**: Lists available providers with status (Ollama running/not, cloud keys configured/not, available local models).
- **`web`**: Launches Gradio web interface.
- **`version`**: Displays version.

Uses `asyncio.run()` to bridge sync CLI with async LLM calls. Rich console for colored output, panels, tables, syntax highlighting, progress spinners, and interactive prompts.

### Configuration (`src/geoforge/config.py`)

`Settings(BaseSettings)` loads from `.env` file:
- **API keys**: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- **Model names**: `GEMINI_MODEL` (gemini-2.5-flash), `OPENAI_MODEL` (gpt-5.2), `ANTHROPIC_MODEL` (claude-sonnet-4-5), `OLLAMA_MODEL` (qwen2.5-coder:14b)
- **Ollama**: `OLLAMA_HOST` (http://localhost:11434)
- **Reproducibility**: `LLM_TEMPERATURE` (0.0), `LLM_SEED` (42)
- **Output**: `OUTPUT_DIR` (examples/outputs)
- **Default provider**: `DEFAULT_LLM_PROVIDER` (ollama) - Literal["gemini", "openai", "anthropic", "ollama"]

Helper methods: `get_available_providers()`, `get_cloud_providers()`, `has_multiple_providers()`.

Global singleton: `settings = Settings()`.

### Prompts (`src/geoforge/prompts/gdsfactory.py`, 600+ lines)

**`GEOMETRY_SPEC_PROMPT`** (~236 lines): Guides the LLM through a 5-step reasoning process to extract a structured GeometrySpec from natural language. Includes:
- A catalog of 15 supported component types with layers, parameters, and constraints
- Layer numbering convention: metal1=(1,0), via1=(2,0), metal2=(3,0), via2=(4,0), metal3=(5,0)
- Units: nm for thickness metadata, um for geometry dimensions
- 6 few-shot examples covering via arrays, MIM capacitors, guard rings, inductors, interconnects, and bond pads

**`COMPONENT_EXAMPLES`** (dict): 10 complete, runnable GDSFactory code examples for: via_stack, via_array, interconnect, mim_capacitor, guard_ring, bond_pad, inductor, layer_stack, test_pattern, mom_capacitor. Each example uses `@gf.cell` decorator, proper layer definitions, `dmove()` with tuple args, correct `gf.components.array()` usage, and `.write_gds()`/`.write()` calls.

**`build_code_prompt(spec_dict, original_prompt)`**: Constructs the code generation prompt by combining the template with spec details (component_type, description, parameters, layers) and the matching example if available.

### Visualization (`src/geoforge/viz/renderer.py`)

Three rendering functions:
- `render_component(component, save_path, dpi)`: Renders a GDSFactory Component to matplotlib Figure/PNG.
- `render_gds_file(gds_path, save_path, dpi)`: Imports a .gds file and renders it.
- `render_code(code, save_path, dpi)`: Executes code in sandbox, finds the Component in the namespace, and renders it. Falls back to loading generated .gds files.

All use matplotlib Agg backend for headless operation. PDK activation before rendering. Returns Figure or None on failure.

### Web Interface (`src/geoforge/web/`)

**`app.py`**: Gradio Blocks app with:
- Input row: prompt textbox, provider dropdown, generate button
- Status display
- Tabbed output: Generated Code, Layout Preview (image), Validation (markdown), Geometry Spec (markdown)
- Download row: Python file, GDS file, OAS file

**`handlers.py`**: `run_pipeline()` orchestrates the full async pipeline for the web UI:
1. Get provider → 2. Generate spec+code → 3. Validate → 4. Export files (only if validation passed) → 5. Render PNG preview (only if validation passed)
Returns `PipelineResult` dataclass with all paths and metadata. Export and preview are gated on validation success to prevent invalid GDS files from being written.

### Pipeline Logging (`src/geoforge/core/logging.py`)

`PipelineLogger` records structured events (`PipelineEvent` dataclass) with:
- UTC timestamps, stage names, status (start/success/error/retry), duration tracking
- Saves to JSON file when `--debug` flag is used
- Tracks provider name, event count, and all events in order

## Supported Multi-Layer Components

| Component | Layers | Key Parameters |
|-----------|--------|----------------|
| `via_stack` | M1 + Via + M2 | metal_size, via_size |
| `via_array` | M1 + Via + M2 | rows, cols, pitch, metal_size, via_size |
| `mim_capacitor` | M1 + M2 | plate_width, plate_height, extension |
| `mom_capacitor` | M1 + M2 | num_fingers, finger_width, finger_gap |
| `guard_ring` | M1 + Via + M2 | inner_size, ring_width, via_pitch |
| `interconnect` | M1 + Via + M2 | trace_width, num_traces, spacing |
| `bond_pad` | M1 + Via + M2 | pad_size, via_array params |
| `inductor` | M1 + Via + M2 | turns, inner_radius, trace_width |
| `layer_stack` | Configurable | layer_count, sizes per layer |
| `test_pattern` | Configurable | shape types, sizes |
| `differential_pair` | M1 + M2 | paired traces with ground |
| `seal_ring` | M1 + Via + M2 | chip edge seal structure |
| `alignment_mark` | Configurable | lithography alignment marks |
| `transmission_line` | M1 + M2 | coplanar waveguide params |
| `meander` | Configurable | delayed trace, spacing |

**Key insight**: GDS is a 2D format. "Multi-layer" means different shapes on different GDS layer numbers (tuples like `(1, 0)`), not physical 3D stacking.

**Layer numbering convention**: metal1=(1,0), via1=(2,0), metal2=(3,0), via2=(4,0), metal3=(5,0)

## Key Patterns

- **Async-first**: All LLM calls are async (`await provider.generate(prompt)`). CLI uses `asyncio.run()` to bridge.
- **Pydantic everywhere**: `GeometrySpec`, `LayerSpec`, `GeometryRange`, `Settings` all use Pydantic for validation. LLM JSON output is validated through Pydantic models with retry on validation errors.
- **Two-stage LLM pipeline**: Stage 1 extracts structured JSON spec (including optional `geometry_ranges` for multi-layer components), Stage 2 generates code from that spec via template or LLM. Each stage validates independently and retries separately.
- **Per-range geometry**: `GeometryRange` lets the spec express different sizing and placement per layer range. The `multi_layer_rectangles` template generates per-range for-loops with linear interpolation and `dmove()` for non-origin placement.
- **Retry with error feedback**: Errors are classified by category and fed back to the LLM as `RetryContext` so it can self-correct. Exponential backoff between attempts.
- **Sandbox execution**: Generated code runs in an isolated `multiprocessing.Process` with temp directory, 30-second timeout, and import allowlisting. Code is prepared by injecting output paths and disabling `.show()` calls.
- **Provider factory**: Lazy-loaded registry avoids importing unused provider dependencies. `get_provider(name)` returns an instantiated provider.
- **File naming**: Outputs use `{component_type}_{YYYY-MM-DD_HHMMSS}.{ext}` with regex sanitization for special characters.
- **Error recovery in pipeline**: Web handlers catch errors per-step and continue, so partial results are still returned (e.g., spec generated but validation failed).

## Adding a New LLM Provider

1. Create `src/geoforge/llm/newprovider.py` inheriting from `LLMProvider`
2. Implement two async methods:
   - `_generate_geometry_spec_impl(self, prompt: str, retry_context: RetryContext | None) -> dict`
   - `_generate_gdsfactory_code_impl(self, spec: GeometrySpec, original_prompt: str | None, retry_context: RetryContext | None) -> str`
3. Use the `@register_provider("name")` decorator on the class
4. Add API key and model name settings in `config.py` (as `SecretStr` / `str` fields)
5. Import the new module in `registry.py`'s `_load_providers()` function
6. Update `.env.example` with the new env vars

The base class handles all retry logic, code validation, and error classification automatically.

## GDSFactory API Quick Reference

```python
import gdsfactory as gf

# Component decorator (required for all component functions)
@gf.cell
def my_component() -> gf.Component:
    c = gf.Component()
    # ... build component ...
    return c

# Basic shapes
gf.components.rectangle(size=(w, h), layer=(L, D), centered=True)
gf.components.circle(radius=r, layer=(L, D))
gf.components.ring(radius=R, width=W, layer=(L, D))

# Arrays - use gf.components.array, NOT c.add_array!
arr = gf.components.array(component, columns=N, rows=M, column_pitch=X, row_pitch=Y)
c.add_ref(arr)

# Positioning - dmove takes a TUPLE, not keyword args
ref = c.add_ref(shape)
ref.dmove((dx, dy))  # Correct
# ref.dmove(x=dx, y=dy)  # WRONG - will error!

# Boolean operations
gf.boolean(A=shape1, B=shape2, operation="not", layer=(L, D))  # subtract

# Export
c.write_gds("output.gds")
c.write("output.oas")       # OASIS format

# Show (disabled in sandbox, but used in standalone scripts)
c.show()
```

## Testing Details

**~135 tests** across 10 test files. Key patterns:

- **Fixtures**: `conftest.py` provides `sample_prompt` and `sample_spec`. `test_geometry.py` has 8 component-specific fixtures (via_stack, via_array, mim_capacitor, guard_ring, interconnect, bond_pad, inductor, five_metal_stack).
- **Mock provider**: `mock_provider.py` contains `MockLLMProvider` with configurable failures (`fail_count`, `fail_spec`, `fail_code`), call counting (`_spec_calls`, `_code_calls`), and predetermined responses. Avoids needing real LLM calls in tests.
- **Markers**: `@pytest.mark.slow` for tests requiring GDSFactory execution or file I/O. `@pytest.mark.asyncio` for async tests. `@pytest.mark.integration` for end-to-end tests. Tests use `pytest.mark.parametrize` for prompt-to-component mapping.
- **Async testing**: `pytest-asyncio` with `asyncio_mode = "auto"` in pyproject.toml.
- **Temp directories**: File I/O tests use `tempfile.TemporaryDirectory()` for isolation.

| Test File | Count | Focus |
|-----------|-------|-------|
| test_validator.py | 35 | Syntax, safety, execution, output naming, export |
| test_geometry.py | 39 | Multi-layer specs, layer relationships, geometry ranges |
| test_integration.py | 15 | Full pipeline, PipelineLogger, async export |
| test_retry.py | 14 | Error classification, RetryContext, retry behavior |
| test_cli.py | 10 | CLI commands, help text, argument validation |
| test_spec_match.py | 13 | Spec-to-code validation (layers, component type, ranges) |
| test_web.py | 9 | Provider choices, markdown formatting, Gradio app |
| test_viz.py | 9 | render_component, render_code, render_gds_file |
| test_llm.py | 5 | Provider registry, GeometrySpec/LayerSpec models |
| test_config.py | 2 | Settings defaults, provider availability |

## Environment Variables

```bash
# LLM Provider API Keys (all optional - Ollama works without any keys)
GEMINI_API_KEY=           # Google Gemini
OPENAI_API_KEY=           # OpenAI
ANTHROPIC_API_KEY=        # Anthropic Claude

# Default LLM provider
DEFAULT_LLM_PROVIDER=ollama   # Options: ollama, gemini, openai, anthropic

# Model names (defaults shown)
GEMINI_MODEL=gemini-2.5-flash
OPENAI_MODEL=gpt-5.2
ANTHROPIC_MODEL=claude-sonnet-4-5
OLLAMA_MODEL=qwen2.5-coder:14b

# Reproducibility (temperature=0 for deterministic output)
LLM_TEMPERATURE=0.0
LLM_SEED=42

# Ollama connection
OLLAMA_HOST=http://localhost:11434

# Output directory for generated files
OUTPUT_DIR=examples/outputs
```

## Linting & Code Quality Configuration

- **Line length**: 100 characters
- **Ruff rules**: Extensive rule set including E, F, I, UP, B, SIM, PTH, C4, DTZ, T10, EXE, ISC, ICN, PIE, Q, RSE, RET, SLOT, TID, TCH, ARG, PGH, PL, TRY, FLY, PERF, FURB, RUF
- **Bandit skips**: B101 (assert), B102 (exec) - intentional for sandbox code validation
- **Test file ignores**: ARG001/ARG002 (unused fixtures), PLR2004 (magic values)
- **Type checker**: ty targeting Python 3.13
- **Pre-commit hooks**: trailing-whitespace, end-of-file-fixer, check-yaml/toml, check-added-large-files (1MB), check-merge-conflict, detect-private-key, mixed-line-ending (LF), ruff lint+format, bandit, codespell, ty, pymarkdown

## Error Handling Patterns

| Layer | Error Type | Handling |
|-------|-----------|----------|
| CLI | ConnectionError (Ollama not running) | Exit code 1, helpful tip to run `ollama serve` |
| CLI | ValueError (missing config) | Exit code 1, descriptive error message |
| Config | Missing API key | Loads as None, provider excluded from cloud providers list |
| LLM Base | JSON parse error | Retry with `json_parse` category in RetryContext |
| LLM Base | Pydantic validation error | Retry with `schema_validation` category |
| LLM Base | Syntax error in generated code | Retry with `syntax_error` category |
| LLM Base | Code too short (<50 chars) | Retry with `code_too_short` category |
| LLM Base | Missing import (no gdsfactory) | Retry with `missing_import` category |
| Ollama | Connection refused | ConnectionError with install instructions URL |
| Validator | Execution timeout (>30s) | Returns "timed out after 30s", process killed |
| Validator | Banned import detected | Returns safety violation with module name |
| Validator | Dynamic import/exec bypass | Returns forbidden call error (__import__, eval, etc.) |
| Validator | Runtime error in generated code | Returns error message, execution marked failed |
| Web | Validation failure | Export and preview skipped, result.success = False |
| Web | Other pipeline step failure | Logged, partial results returned |
