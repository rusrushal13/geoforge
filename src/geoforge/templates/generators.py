"""Deterministic code generators for specific geometry families.

Each generator receives a GeometrySpec (with parameters extracted by the LLM
in stage 1) and returns a complete, runnable GDSFactory Python script.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geoforge.templates.registry import register_template

if TYPE_CHECKING:
    from geoforge.llm.base import GeometrySpec


# ---------------------------------------------------------------------------
# curved_trace_bundle
# ---------------------------------------------------------------------------


@register_template("curved_trace_bundle")
def _gen_curved_trace_bundle(spec: GeometrySpec) -> str:
    """Parallel curved bundle with concentric euler/arc bends."""
    p = spec.parameters
    num_traces = int(p.get("num_traces", 10))
    trace_width = float(p.get("trace_width", 8.0))
    trace_spacing = float(p.get("trace_spacing", trace_width))
    lead_length = float(p.get("lead_length", 300.0))
    bend_radius_start = float(p.get("bend_radius_start", 140.0))
    bend_angle = float(p.get("bend_angle", 200.0))
    tail_length = float(p.get("tail_length", 50.0))

    layer = _first_layer_tuple(spec)

    return f'''\
import gdsfactory as gf

gf.gpdk.PDK.activate()

LAYER = {layer}

@gf.cell
def curved_trace_bundle() -> gf.Component:
    """Parallel curved trace bundle with concentric bends."""
    c = gf.Component()

    num_traces = {num_traces}
    trace_width = {trace_width}
    trace_spacing = {trace_spacing}
    pitch = trace_width + trace_spacing
    lead_length = {lead_length}
    bend_radius_start = {bend_radius_start}
    bend_angle = {bend_angle}
    tail_length = {tail_length}

    bundle_height = (num_traces - 1) * pitch + trace_width

    # Left bus bar
    bus = gf.components.rectangle(
        size=(trace_width * 2, bundle_height), layer=LAYER, centered=True,
    )
    c.add_ref(bus)

    for i in range(num_traces):
        y_offset = -bundle_height / 2 + trace_width / 2 + i * pitch
        radius = bend_radius_start + i * pitch

        path = gf.path.Path()
        path.append(gf.path.straight(length=lead_length))
        path.append(gf.path.euler(radius=radius, angle=bend_angle))
        path.append(gf.path.straight(length=tail_length))

        route = gf.path.extrude(path, width=trace_width, layer=LAYER)
        ref = c.add_ref(route)
        ref.dmove((trace_width, y_offset))

    return c

if __name__ == "__main__":
    c = curved_trace_bundle()
    c.write_gds("curved_trace_bundle.gds")
    c.write("curved_trace_bundle.oas")
'''


# ---------------------------------------------------------------------------
# comb_serpentine
# ---------------------------------------------------------------------------


@register_template("comb_serpentine")
def _gen_comb_serpentine(spec: GeometrySpec) -> str:
    """Horizontal comb fanout from center pad into bottom rail."""
    p = spec.parameters
    trace_width = float(p.get("trace_width", 6.0))
    center_pad_size = float(p.get("center_pad_size", 60.0))
    num_fingers = int(p.get("num_fingers", 30))
    finger_pitch = float(p.get("finger_pitch", trace_width + 2.0))
    rail_y = float(p.get("rail_y", 0.0))
    pad_center_y = float(p.get("pad_center_y", 55.0))
    min_finger_length = float(p.get("min_finger_length", 80.0))
    max_finger_length = float(p.get("max_finger_length", 420.0))
    bend_radius = float(p.get("bend_radius", 8.0))

    layer = _first_layer_tuple(spec)

    return f'''\
import numpy as np
import gdsfactory as gf

gf.gpdk.PDK.activate()

LAYER = {layer}

@gf.cell
def comb_serpentine() -> gf.Component:
    """Horizontal comb fanout from center pad into bottom rail."""
    c = gf.Component()

    trace_width = {trace_width}
    center_pad_size = {center_pad_size}
    num_fingers = {num_fingers}
    finger_pitch = {finger_pitch}
    rail_y = {rail_y}
    pad_center_y = {pad_center_y}
    min_finger_length = {min_finger_length}
    max_finger_length = {max_finger_length}
    bend_radius = {bend_radius}

    # Center pad
    pad = gf.components.rectangle(
        size=(center_pad_size, center_pad_size), layer=LAYER, centered=True,
    )
    pad_ref = c.add_ref(pad)
    pad_ref.dmove((0, pad_center_y))

    # Bottom rail
    rail_length = 2 * (max_finger_length + center_pad_size / 2 + 80.0)
    rail = gf.components.rectangle(
        size=(rail_length, trace_width), layer=LAYER, centered=True,
    )
    rail_ref = c.add_ref(rail)
    rail_ref.dmove((0, rail_y))

    # Fingers
    per_side = max(1, num_fingers // 2)
    y_start = pad_center_y - ((per_side - 1) * finger_pitch) / 2
    y_coords = [y_start + i * finger_pitch for i in range(per_side)]
    lengths = np.linspace(min_finger_length, max_finger_length, per_side)

    for y_local, finger_length in zip(y_coords, lengths, strict=False):
        vertical_drop = max(10.0, y_local - rail_y - trace_width)

        p = gf.path.Path()
        p.append(gf.path.straight(finger_length))
        p.append(gf.path.euler(radius=bend_radius, angle=-90))
        p.append(gf.path.straight(vertical_drop))

        # Right finger
        right = gf.path.extrude(p, width=trace_width, layer=LAYER)
        right_ref = c.add_ref(right)
        right_ref.dmove((center_pad_size / 2, y_local))

        # Left finger (mirrored)
        left = gf.path.extrude(p, width=trace_width, layer=LAYER)
        left_ref = c.add_ref(left)
        left_ref.dmove((-center_pad_size / 2, y_local))
        left_ref.mirror_y()

    return c

if __name__ == "__main__":
    c = comb_serpentine()
    c.write_gds("comb_serpentine.gds")
    c.write("comb_serpentine.oas")
'''


# ---------------------------------------------------------------------------
# multi_layer_rectangles
# ---------------------------------------------------------------------------


@register_template("multi_layer_rectangles")
def _gen_multi_layer_rectangles(spec: GeometrySpec) -> str:
    """N layers, one rectangle per layer with parameterized sizing.

    When spec.geometry_ranges is populated, generates per-range for-loops
    with independent sizing and placement. Otherwise uses the legacy
    single-formula path.
    """
    p = spec.parameters
    cell_name = str(p.get("cell_name", "multi_layer_rectangles"))

    if spec.geometry_ranges:
        return _gen_multi_layer_with_ranges(spec.geometry_ranges, cell_name)

    # --- Legacy single-formula path (unchanged) ---
    num_layers = int(p.get("num_layers", p.get("total_layers", 10)))
    base_width = float(p.get("base_width", p.get("width_start", 1000.0)))
    base_height = float(p.get("base_height", p.get("height_start", 1000.0)))
    width_step = float(p.get("width_step", base_width))
    height_step = float(p.get("height_step", base_height))

    return f'''\
import gdsfactory as gf

gf.gpdk.PDK.activate()

@gf.cell
def multi_layer_rectangles() -> gf.Component:
    """One rectangle per layer with linearly growing dimensions."""
    c = gf.Component("{cell_name}")

    num_layers = {num_layers}
    base_width = {base_width}
    base_height = {base_height}
    width_step = {width_step}
    height_step = {height_step}

    for i in range(1, num_layers + 1):
        w = base_width + (i - 1) * width_step
        h = base_height + (i - 1) * height_step
        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
        c.add_ref(rect)

    return c

if __name__ == "__main__":
    c = multi_layer_rectangles()
    c.write_gds("multi_layer_rectangles.gds")
    c.write("multi_layer_rectangles.oas")
'''


def _gen_multi_layer_with_ranges(
    ranges: list,
    cell_name: str,
) -> str:
    """Generate multi-layer code with per-range sizing and placement."""
    range_blocks: list[str] = []
    for r in ranges:
        num_in_range = r.end_layer - r.start_layer + 1
        # Build the for-loop block for this range
        lines = [
            f"    # Layers {r.start_layer}-{r.end_layer}: {r.shape}",
            f"    for i in range({r.start_layer}, {r.end_layer} + 1):",
            f"        t = (i - {r.start_layer}) / max(1, {num_in_range - 1})",
            f"        w = {r.width_start} + t * ({r.width_end} - {r.width_start})",
            f"        h = {r.height_start} + t * ({r.height_end} - {r.height_start})",
            "        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)",
        ]
        if r.center_x != 0.0 or r.center_y != 0.0:
            lines.append("        ref = c.add_ref(rect)")
            lines.append(f"        ref.dmove(({r.center_x}, {r.center_y}))")
        else:
            lines.append("        c.add_ref(rect)")

        range_blocks.append("\n".join(lines))

    all_range_code = "\n\n".join(range_blocks)

    return f'''\
import gdsfactory as gf

gf.gpdk.PDK.activate()

@gf.cell
def multi_layer_rectangles() -> gf.Component:
    """Multi-range rectangles with per-range sizing and placement."""
    c = gf.Component("{cell_name}")

{all_range_code}

    return c

if __name__ == "__main__":
    c = multi_layer_rectangles()
    c.write_gds("multi_layer_rectangles.gds")
    c.write("multi_layer_rectangles.oas")
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_layer_tuple(spec: GeometrySpec) -> tuple[int, int]:
    """Extract the first layer tuple from the spec, defaulting to (1, 0)."""
    if spec.layers:
        return (spec.layers[0].layer_number, spec.layers[0].datatype)
    return (1, 0)
