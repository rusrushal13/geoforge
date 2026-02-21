"""GeoForge CLI - Natural language semiconductor geometry generator."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from geoforge import __version__
from geoforge.config import settings
from geoforge.llm import get_provider, list_providers

app = typer.Typer(
    name="geoforge",
    help="Natural language semiconductor geometry generator using GDSFactory",
    no_args_is_help=True,
)
console = Console()


def _select_provider(explicit_provider: str | None) -> str:
    """Select provider, prompting user if multiple are available."""
    if explicit_provider:
        return explicit_provider

    cloud_providers = settings.get_cloud_providers()
    has_cloud = len(cloud_providers) > 0

    if not has_cloud:
        return "ollama"

    console.print("[yellow]Multiple LLM providers available:[/yellow]")
    console.print()

    options = ["ollama (local, free)"]
    options.extend(f"{p} (cloud)" for p in cloud_providers)

    for i, opt in enumerate(options, 1):
        console.print(f"  {i}. {opt}")

    console.print()

    all_options = ["ollama", *cloud_providers]

    choice = Prompt.ask(
        "Select provider",
        choices=[str(i) for i in range(1, len(options) + 1)],
        default="1",
    )

    selected = all_options[int(choice) - 1]
    console.print()
    return selected


def _display_validation_result(result) -> None:
    """Display validation results in a nice table."""
    table = Table(title="Validation Results", show_header=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Syntax check
    table.add_row(
        "Syntax",
        "[green]✓ Pass[/green]" if result.syntax_ok else "[red]✗ Fail[/red]",
        "",
    )

    # Safety check
    table.add_row(
        "Safety",
        "[green]✓ Pass[/green]" if result.safety_ok else "[red]✗ Fail[/red]",
        "",
    )

    # Execution check
    exec_details = ""
    if result.execution_time_seconds is not None:
        exec_details = f"{result.execution_time_seconds}s"
    table.add_row(
        "Execution",
        "[green]✓ Pass[/green]" if result.executes_ok else "[red]✗ Fail[/red]",
        exec_details,
    )

    # Spec match
    table.add_row(
        "Spec Match",
        "[green]✓ Pass[/green]" if result.spec_match_ok else "[red]✗ Fail[/red]",
        "",
    )

    # GDS output
    table.add_row(
        "GDS Created",
        "[green]✓ Yes[/green]" if result.gds_created else "[yellow]○ No[/yellow]",
        "",
    )

    # OAS output
    table.add_row(
        "OAS Created",
        "[green]✓ Yes[/green]" if result.oas_created else "[yellow]○ No[/yellow]",
        "",
    )

    console.print(table)

    # Show errors
    if result.errors:
        console.print("\n[red]Errors:[/red]")
        for error in result.errors:
            console.print(f"  [red]•[/red] {error}")

    # Show warnings
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")

    # Overall result
    if result.is_valid:
        console.print("\n[green]✓ Code is valid and executes correctly[/green]")
    else:
        console.print("\n[red]✗ Code validation failed[/red]")


def _display_export_summary(export_result, validation_skipped: bool = False) -> None:
    """Display summary of exported files."""
    console.print("\n[bold]Output Files:[/bold]")

    if export_result.py_path:
        console.print(f"  [green]✓[/green] Code:  {export_result.py_path}")

    if export_result.gds_path:
        console.print(f"  [green]✓[/green] GDS:   {export_result.gds_path}")
    elif not validation_skipped:
        console.print("  [yellow]○[/yellow] GDS:   not created")

    if export_result.oas_path:
        console.print(f"  [green]✓[/green] OASIS: {export_result.oas_path}")
    elif not validation_skipped:
        console.print("  [yellow]○[/yellow] OASIS: not created")

    if export_result.error:
        console.print(f"\n[red]Export error:[/red] {export_result.error}")


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Natural language description of geometry")],
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="LLM provider to use", show_default="ollama"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", "-d", help="Output directory", show_default="examples/outputs"
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Base name for output files",
            show_default="{component}_{timestamp}",
        ),
    ] = None,
    show_code: Annotated[
        bool, typer.Option("--show-code/--no-show-code", "-c/-C", help="Display generated code")
    ] = True,
    execute: Annotated[
        bool, typer.Option("--execute", "-x", help="Open geometry in KLayout after generation")
    ] = False,
    validate: Annotated[
        bool,
        typer.Option("--validate/--no-validate", "-v/-V", help="Validate generated code"),
    ] = True,
    save: Annotated[
        bool,
        typer.Option("--save/--no-save", "-s/-S", help="Save code and export GDS/OAS files"),
    ] = True,
    preview: Annotated[
        bool,
        typer.Option("--preview", help="Save a PNG preview of the layout"),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Save debug log JSON alongside outputs"),
    ] = False,
):
    """Generate semiconductor geometry from natural language description.

    By default, GeoForge will:
    - Validate the generated code (syntax + execution)
    - Save the Python code to examples/outputs/
    - Export GDS and OASIS files to examples/outputs/

    Use --no-validate to skip validation, --no-save to skip saving files.
    Use --execute to also open the geometry in KLayout.
    Use --preview to save a PNG preview alongside other outputs.
    """
    from geoforge.core.validator import (
        ExportResult,
        export_code_and_files,
        generate_output_name,
        validate_generated_code,
    )

    provider_name = _select_provider(provider)

    console.print(f"[blue]Using provider:[/blue] {provider_name}")
    console.print(f"[blue]Prompt:[/blue] {prompt}\n")

    # Create pipeline logger if --debug
    pipeline_logger = None
    if debug:
        from geoforge.core.logging import PipelineLogger

        pipeline_logger = PipelineLogger(provider=provider_name)

    try:
        llm = get_provider(provider_name)
    except ConnectionError as e:
        console.print(f"[red]Connection Error:[/red] {e}")
        console.print("\n[dim]Tip: Start Ollama with 'ollama serve' or use --provider gemini[/dim]")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    # Generate geometry
    with console.status("[bold green]Generating geometry specification..."):
        try:
            spec = asyncio.run(llm.generate(prompt, logger=pipeline_logger))
        except Exception as e:
            console.print(f"[red]Generation failed:[/red] {e}")
            raise typer.Exit(1) from e

    console.print(Panel(f"[green]Component Type:[/green] {spec.component_type}"))
    console.print(f"[dim]Description:[/dim] {spec.description}\n")

    # Show parameters extracted by LLM
    if spec.parameters:
        console.print("[bold]Extracted Parameters:[/bold]")
        for key, value in spec.parameters.items():
            console.print(f"  {key}: {value}")
        console.print()

    # Show layers
    if spec.layers:
        console.print("[bold]Layers:[/bold]")
        for layer in spec.layers:
            console.print(f"  ({layer.layer_number}, {layer.datatype}) {layer.name}", end="")
            if layer.material:
                console.print(f" - {layer.material}", end="")
            if layer.thickness_nm:
                console.print(f" ({layer.thickness_nm}nm)", end="")
            console.print()
        console.print()

    if spec.primitives:
        console.print("[bold]Primitive Geometry:[/bold]")
        for primitive in spec.primitives:
            detail = (
                f"  {primitive.primitive_type} on ({primitive.layer_number}, {primitive.datatype})"
            )
            if primitive.primitive_type == "rectangle":
                detail += f" size=({primitive.width}, {primitive.height})"
            elif primitive.primitive_type == "circle":
                detail += f" radius={primitive.radius}"
            elif primitive.primitive_type == "polygon":
                detail += f" points={len(primitive.points)}"
            detail += f" center=({primitive.center_x}, {primitive.center_y})"
            if primitive.rotation_deg:
                detail += f" rot={primitive.rotation_deg}"
            console.print(detail)
        console.print()

    # Show generated code
    if show_code and spec.gdsfactory_code:
        console.print("[bold]Generated GDSFactory Code:[/bold]")
        syntax = Syntax(spec.gdsfactory_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print()

    out_dir = output_dir or settings.output_dir
    output_base_name = output or generate_output_name(spec.component_type)

    # Validate code (default: enabled)
    validation_result = None
    if validate and spec.gdsfactory_code:
        console.print("[bold]Validating generated code...[/bold]\n")
        with console.status("[bold green]Running validation..."):
            validation_result = validate_generated_code(spec)
        _display_validation_result(validation_result)
        console.print()

    # Save code and export files (default: enabled)
    export_result = None
    if save and spec.gdsfactory_code:
        # Only export GDS/OAS if validation passed or was skipped
        should_export = not validate or (validation_result and validation_result.is_valid)

        if should_export:
            console.print("[bold]Exporting files...[/bold]")
            with console.status("[bold green]Saving code and generating outputs..."):
                export_result = export_code_and_files(
                    spec.gdsfactory_code,
                    out_dir,
                    output_base_name,
                )
            _display_export_summary(export_result, validation_skipped=not validate)
        else:
            # Validation failed - still save the code for debugging
            console.print("[bold]Saving code for debugging...[/bold]")
            out_dir.mkdir(parents=True, exist_ok=True)
            py_path = out_dir / f"{output_base_name}.py"
            py_path.write_text(spec.gdsfactory_code)
            export_result = ExportResult(success=False, py_path=py_path)
            console.print(f"\n[yellow]Code saved to:[/yellow] {py_path}")
            console.print("[dim]GDS/OAS not exported due to validation failure[/dim]")

    # Render preview PNG (only with --preview flag)
    if preview and spec.gdsfactory_code:
        from geoforge.viz.renderer import render_code, render_gds_file

        png_path = out_dir / f"{output_base_name}.png"
        out_dir.mkdir(parents=True, exist_ok=True)

        with console.status("[bold green]Rendering preview..."):
            try:
                if export_result and export_result.gds_path and export_result.gds_path.exists():
                    fig = render_gds_file(export_result.gds_path, save_path=png_path)
                else:
                    fig = render_code(spec.gdsfactory_code, save_path=png_path)

                if fig:
                    console.print(f"\n[green]Preview saved:[/green] {png_path}")
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                else:
                    console.print("\n[yellow]Preview: No component found to render[/yellow]")
            except Exception as e:
                console.print(f"\n[yellow]Preview failed:[/yellow] {e}")

    # Save debug log (only with --debug flag)
    if debug and pipeline_logger:
        debug_path = out_dir / f"{output_base_name}_debug.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline_logger.save(debug_path)
        console.print(f"\n[dim]Debug log saved:[/dim] {debug_path}")

    # Open in KLayout (only with --execute flag)
    if execute and spec.gdsfactory_code:
        from geoforge.core.validator import _prepare_code_for_execution

        console.print("\n[yellow]Opening in KLayout...[/yellow]")
        out_dir.mkdir(parents=True, exist_ok=True)

        modified_code = _prepare_code_for_execution(spec.gdsfactory_code, out_dir, output_base_name)
        exec_globals = {"__name__": "__main__"}
        try:
            exec(modified_code, exec_globals)
            console.print("[green]Code executed successfully![/green]")
        except Exception as e:
            console.print(f"[red]Execution error:[/red] {e}")


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="Python file containing GDSFactory code")],
):
    """Validate a GDSFactory code file."""
    from geoforge.core.validator import validate_execution, validate_syntax

    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    code = file.read_text()
    console.print(f"[blue]Validating:[/blue] {file}\n")

    # Syntax check
    syntax_ok, syntax_error = validate_syntax(code)
    if syntax_ok:
        console.print("[green]✓ Syntax OK[/green]")
    else:
        console.print(f"[red]✗ Syntax Error:[/red] {syntax_error}")
        raise typer.Exit(1)

    # Execution check
    console.print("[dim]Running code...[/dim]")
    exec_ok, exec_error, output_info = validate_execution(code)

    if exec_ok:
        console.print("[green]✓ Execution OK[/green]")
        if output_info["gds_files"]:
            console.print(f"[green]✓ GDS files created:[/green] {len(output_info['gds_files'])}")
        if output_info["oas_files"]:
            console.print(f"[green]✓ OAS files created:[/green] {len(output_info['oas_files'])}")
    else:
        console.print(f"[red]✗ Execution Error:[/red] {exec_error}")
        raise typer.Exit(1)


@app.command()
def providers():
    """List available LLM providers."""
    from geoforge.llm.ollama import OllamaProvider

    console.print("[bold]Available LLM Providers:[/bold]\n")

    cloud_providers = settings.get_cloud_providers()
    all_provider_names = list_providers()

    ollama_running = OllamaProvider.is_available()
    local_models = OllamaProvider.list_local_models() if ollama_running else []

    for name in all_provider_names:
        if name == "ollama":
            if ollama_running:
                status = "[green]✓ running[/green]"
                if local_models:
                    models_str = ", ".join(local_models[:3])
                    if len(local_models) > 3:
                        models_str += f" (+{len(local_models) - 3} more)"
                    status += f" [dim]({models_str})[/dim]"
            else:
                status = "[yellow]○ not running[/yellow] [dim](start with 'ollama serve')[/dim]"
        else:
            status = (
                "[green]✓ configured[/green]"
                if name in cloud_providers
                else "[dim]○ not configured[/dim]"
            )

        default = " [yellow](default)[/yellow]" if name == settings.default_llm_provider else ""
        console.print(f"  {name}: {status}{default}")

    console.print("\n[dim]Configure API keys in .env file[/dim]")
    console.print("[dim]Install Ollama: https://ollama.ai[/dim]")


@app.command()
def web(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "0.0.0.0",  # nosec B104
    port: Annotated[int, typer.Option("--port", help="Port to listen on")] = 7860,
    share: Annotated[bool, typer.Option("--share", help="Create public Gradio share link")] = False,
):
    """Launch the GeoForge web UI."""
    from geoforge.web.app import launch

    console.print(f"[bold green]Starting GeoForge Web UI on {host}:{port}[/bold green]")
    launch(host=host, port=port, share=share)


@app.command()
def version():
    """Show version information."""
    console.print(f"GeoForge v{__version__}")


if __name__ == "__main__":
    app()
