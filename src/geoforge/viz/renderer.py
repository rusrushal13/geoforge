"""Visualization utilities for rendering GDS layouts as images."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _setup_matplotlib() -> None:
    """Configure matplotlib for headless rendering."""
    import matplotlib as mpl

    mpl.use("Agg")


def render_component(
    component,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Render a GDSFactory component to a matplotlib Figure.

    Args:
        component: A gdsfactory Component object.
        save_path: Optional path to save the figure as PNG.
        dpi: Resolution for saved image.

    Returns:
        matplotlib Figure.
    """
    _setup_matplotlib()
    fig = component.plot(return_fig=True)
    if fig is None:
        msg = "Component.plot(return_fig=True) returned None"
        raise RuntimeError(msg)
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def render_gds_file(
    gds_path: Path | str,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Render a GDS file to a matplotlib Figure.

    Args:
        gds_path: Path to a .gds file.
        save_path: Optional path to save the figure as PNG.
        dpi: Resolution for saved image.

    Returns:
        matplotlib Figure.
    """
    import gdsfactory as gf

    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()

    component = gf.import_gds(str(gds_path))
    return render_component(component, save_path=save_path, dpi=dpi)


def render_code(
    code: str,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure | None:
    """Execute GDSFactory code and render the resulting component.

    Runs the code in a sandbox, finds the first gf.Component in the exec
    namespace, and renders it.

    Args:
        code: GDSFactory Python code string.
        save_path: Optional path to save the figure as PNG.
        dpi: Resolution for saved image.

    Returns:
        matplotlib Figure, or None if no component was found.
    """
    import tempfile

    import gdsfactory as gf

    from geoforge.core.validator import _prepare_code_for_execution

    _setup_matplotlib()

    try:
        gf.get_active_pdk()
    except ValueError:
        gf.gpdk.PDK.activate()

    # Execute in temp directory — redirect all write_gds/write calls there
    tmpdir = tempfile.mkdtemp()

    modified = _prepare_code_for_execution(code, Path(tmpdir), "render_output")
    modified = modified.replace(".show()", "# .show() disabled for rendering")

    exec_globals: dict = {
        "__name__": "__main__",
        "gf": gf,
    }

    try:
        exec(modified, exec_globals)
    except Exception:
        # If code execution fails, try rendering from GDS files it may have created
        gds_files = list(Path(tmpdir).glob("*.gds"))
        if gds_files:
            return render_gds_file(gds_files[0], save_path=save_path, dpi=dpi)
        return None

    # Find the first Component in the exec namespace
    for value in exec_globals.values():
        if isinstance(value, gf.Component):
            return render_component(value, save_path=save_path, dpi=dpi)

    # No component variable found - check if any GDS files were written
    gds_files = list(Path(tmpdir).glob("*.gds"))
    if gds_files:
        return render_gds_file(gds_files[0], save_path=save_path, dpi=dpi)

    return None
