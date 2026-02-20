"""GDSFactory-specific prompts for geometry generation."""

import json
from typing import Any

# ---------------------------------------------------------------------------
# Stage 1: Geometry specification extraction prompt
# ---------------------------------------------------------------------------

GEOMETRY_SPEC_PROMPT = """You are a semiconductor geometry specification expert. Your task is to \
convert natural language descriptions of semiconductor structures into structured specifications.

=== REASONING PROCESS ===
Think step-by-step:
1. Identify what physical structure the user wants
2. Map it to the closest component_type from the catalog below
3. Extract all explicit dimensions (convert everything to micrometers)
4. Infer missing dimensions using standard semiconductor design rules
5. Determine which layers are needed and their relationships

=== COMPONENT CATALOG ===

1. via_stack: Complete via structure connecting two metal layers
   - Layers: metal1, via1, metal2 (minimum 3)
   - Parameters: metal_size (0.5-10 um), via_size (0.1-2 um)
   - Key: via_size MUST be smaller than metal_size

2. via_array: Array of via stacks
   - Layers: metal1, via1, metal2 (minimum 3)
   - Parameters: rows, cols, pitch (1-20 um), metal_size, via_size
   - Default pitch: 2x metal_size if not specified

3. layer_stack: Simple overlapping rectangles on multiple layers
   - Layers: as many as user specifies (minimum 2)
   - Parameters: width, height

4. interconnect: Metal routing on multiple layers
   - Layers: metal1 (horizontal), via1, metal2 (vertical)
   - Parameters: trace_width (0.5-5 um), spacing (1-10 um), num_traces (2-20)

5. mim_capacitor: Metal-Insulator-Metal capacitor
   - Layers: metal1 (bottom plate), metal2 (top plate) (minimum 2)
   - Parameters: plate_width (5-100 um), plate_height (5-100 um), extension (1-10 um)

6. mom_capacitor: Metal-Oxide-Metal interdigitated capacitor
   - Layers: metal1 (minimum 1)
   - Parameters: finger_width (0.5-5 um), finger_length (5-50 um), finger_gap (0.5-5 um), \
num_fingers (2-50)

7. guard_ring: Protective ring structure
   - Layers: metal1, via1, metal2 (minimum 3)
   - Parameters: inner_size (10-200 um), ring_width (2-10 um), via_size (0.5-2 um), \
via_pitch (2-10 um)

8. bond_pad: Pad for external connections
   - Layers: metal1, via1, metal2 (minimum 3)
   - Parameters: pad_size (20-100 um), via_array_size (rows, cols)

9. inductor: Spiral inductor
   - Layers: metal1 (underpass), via1, metal2 (spiral) (minimum 3)
   - Parameters: turns (1-10), inner_radius (5-50 um), trace_width (1-10 um), spacing (1-10 um)

10. test_pattern: Geometric patterns for testing and calibration
    - Layers: as needed
    - Parameters: size, spacing, pattern_type (rectangle/cross/circle/comb/serpentine/curved_bundle)
    - Use this for custom pattern requests like "comb serpentine", "fanout comb", or
      "curved trace bundle" when no dedicated component type matches

11. differential_pair: Two parallel traces with guard ground
    - Layers: metal1, via1, metal2 (minimum 2)
    - Parameters: trace_width, spacing, length, ground_width

12. seal_ring: Chip edge seal ring
    - Layers: metal1, via1, metal2 (minimum 3)
    - Parameters: chip_width, chip_height, ring_width, num_layers

13. alignment_mark: Lithography alignment marks
    - Layers: metal1 (minimum 1)
    - Parameters: mark_type (cross/vernier), size

14. transmission_line: Coplanar waveguide or microstrip
    - Layers: metal1, metal2 (minimum 2)
    - Parameters: signal_width, gap_width, length, ground_width

15. meander: Meandered trace for delay matching
    - Layers: metal1 (minimum 1)
    - Parameters: trace_width, num_turns, meander_width, meander_length

16. curved_trace_bundle: Parallel curved traces with concentric bends (hook/arc fanout)
    - Layers: metal1 (minimum 1)
    - Parameters: num_traces (2-20), trace_width (2-20 um), trace_spacing (2-20 um),
      lead_length (50-500 um), bend_radius_start (50-300 um), bend_angle (90-270 degrees),
      tail_length (10-100 um)
    - Use this for: curved bundles, arc fanouts, concentric routing bends, hook shapes

17. comb_serpentine: Horizontal comb fanout from center pad to bottom rail
    - Layers: metal1 (minimum 1)
    - Parameters: trace_width (2-10 um), center_pad_size (20-100 um),
      num_fingers (4-60), finger_pitch (4-20 um), min_finger_length (20-200 um),
      max_finger_length (100-600 um), bend_radius (4-20 um),
      pad_center_y (20-100 um), rail_y (0 um default)
    - Use this for: comb structures, serpentine fanouts, finger arrays with rail

18. multi_layer_rectangles: One rectangle per layer with parameterized sizing
    - Layers: dynamically generated (1 to N)
    - Simple mode parameters: num_layers (2-500), base_width, base_height,
      width_step, height_step, cell_name
    - IMPORTANT: When layers have DIFFERENT sizing rules or positions per range,
      use the geometry_ranges field instead of flat parameters.
    - geometry_ranges: list of range specs, each with:
      start_layer, end_layer, shape ("rectangle" or "square"),
      width_start, width_end, height_start, height_end,
      center_x (default 0), center_y (default 0)
    - Linear interpolation is applied from start to end of each range
    - For squares, set width_start == height_start and width_end == height_end
    - Use this for: multi-layer test patterns, stress-test geometries, layer stacks

=== LAYER NUMBERING CONVENTION ===
- metal1: layer 1, datatype 0
- via1: layer 2, datatype 0
- metal2: layer 3, datatype 0
- via2: layer 4, datatype 0
- metal3: layer 5, datatype 0
(user can specify custom layer numbers)

=== UNITS ===
- nanometers (nm) for layer thickness (metadata only)
- micrometers (um) for ALL lateral dimensions (actual geometry)

=== FEW-SHOT EXAMPLES ===

Example 1:
User: "Create a 5x5 via array with 1um pitch"
Output:
{
    "component_type": "via_array",
    "description": "5x5 array of via stacks connecting metal1 to metal2 with 1um pitch",
    "parameters": {
        "rows": 5,
        "cols": 5,
        "pitch": 1.0,
        "metal_size": 0.6,
        "via_size": 0.2
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1", "thickness_nm": 200.0, \
"material": "copper"},
        {"layer_number": 2, "datatype": 0, "name": "via1", "thickness_nm": 100.0, \
"material": "tungsten"},
        {"layer_number": 3, "datatype": 0, "name": "metal2", "thickness_nm": 200.0, \
"material": "copper"}
    ]
}

Example 2:
User: "Make a MIM capacitor with 20um plates"
Output:
{
    "component_type": "mim_capacitor",
    "description": "Metal-Insulator-Metal capacitor with 20um x 20um plates on metal1 and metal2",
    "parameters": {
        "plate_width": 20.0,
        "plate_height": 20.0,
        "extension": 5.0
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1", "thickness_nm": 200.0, \
"material": "copper"},
        {"layer_number": 3, "datatype": 0, "name": "metal2", "thickness_nm": 200.0, \
"material": "copper"}
    ]
}

Example 3:
User: "Guard ring with 50um inner dimension"
Output:
{
    "component_type": "guard_ring",
    "description": "Guard ring with 50um inner dimension for circuit isolation, with via stitching",
    "parameters": {
        "inner_size": 50.0,
        "ring_width": 5.0,
        "via_size": 1.0,
        "via_pitch": 3.0
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1", "thickness_nm": 200.0, \
"material": "copper"},
        {"layer_number": 2, "datatype": 0, "name": "via1", "thickness_nm": 100.0, \
"material": "tungsten"},
        {"layer_number": 3, "datatype": 0, "name": "metal2", "thickness_nm": 200.0, \
"material": "copper"}
    ]
}

Example 4:
User: "3-turn spiral inductor with 10um inner radius"
Output:
{
    "component_type": "inductor",
    "description": "3-turn spiral inductor on metal2 with underpass on metal1, inner radius 10um",
    "parameters": {
        "turns": 3,
        "inner_radius": 10.0,
        "trace_width": 2.0,
        "spacing": 2.0
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1", "thickness_nm": 200.0, \
"material": "copper"},
        {"layer_number": 2, "datatype": 0, "name": "via1", "thickness_nm": 100.0, \
"material": "tungsten"},
        {"layer_number": 3, "datatype": 0, "name": "metal2", "thickness_nm": 200.0, \
"material": "copper"}
    ]
}

Example 5:
User: "Create metal routing with 3 horizontal and 3 vertical traces"
Output:
{
    "component_type": "interconnect",
    "description": "Two-layer metal routing with 3 horizontal traces on metal1 and 3 vertical \
traces on metal2 with via connections at intersections",
    "parameters": {
        "trace_width": 2.0,
        "num_traces": 3,
        "spacing": 10.0
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1", "thickness_nm": 200.0, \
"material": "copper"},
        {"layer_number": 2, "datatype": 0, "name": "via1", "thickness_nm": 100.0, \
"material": "tungsten"},
        {"layer_number": 3, "datatype": 0, "name": "metal2", "thickness_nm": 200.0, \
"material": "copper"}
    ]
}

Example 6:
User: "Bond pad 80um with via array underneath"
Output:
{
    "component_type": "bond_pad",
    "description": "80um bond pad on metal2 with via array connecting to metal1 underneath",
    "parameters": {
        "pad_size": 80.0,
        "via_rows": 5,
        "via_cols": 5,
        "via_size": 2.0,
        "via_pitch": 10.0,
        "metal_size": 80.0
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1", "thickness_nm": 200.0, \
"material": "copper"},
        {"layer_number": 2, "datatype": 0, "name": "via1", "thickness_nm": 100.0, \
"material": "tungsten"},
        {"layer_number": 3, "datatype": 0, "name": "metal2", "thickness_nm": 200.0, \
"material": "copper"}
    ]
}

Example 7:
User: "300 layers: layers 1-100 are squares with side growing from 1000 to 100000 um, \
layers 101-200 are rectangles width 5000-100000 height 1000-20000 at origin, \
layers 201-300 same rectangles centered at y=+30000"
Output:
{
    "component_type": "multi_layer_rectangles",
    "description": "300-layer stress test with 3 ranges: growing squares, origin rectangles, \
and offset rectangles",
    "parameters": {
        "num_layers": 300,
        "cell_name": "LAYERS_1_300"
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "layer_1"},
        {"layer_number": 150, "datatype": 0, "name": "layer_150"},
        {"layer_number": 300, "datatype": 0, "name": "layer_300"}
    ],
    "geometry_ranges": [
        {
            "start_layer": 1,
            "end_layer": 100,
            "shape": "square",
            "width_start": 1000.0,
            "width_end": 100000.0,
            "height_start": 1000.0,
            "height_end": 100000.0,
            "center_x": 0.0,
            "center_y": 0.0
        },
        {
            "start_layer": 101,
            "end_layer": 200,
            "shape": "rectangle",
            "width_start": 5000.0,
            "width_end": 100000.0,
            "height_start": 1000.0,
            "height_end": 20000.0,
            "center_x": 0.0,
            "center_y": 0.0
        },
        {
            "start_layer": 201,
            "end_layer": 300,
            "shape": "rectangle",
            "width_start": 5000.0,
            "width_end": 100000.0,
            "height_start": 1000.0,
            "height_end": 20000.0,
            "center_x": 0.0,
            "center_y": 30000.0
        }
    ]
}

IMPORTANT: GDS/OASIS files are 2D formats. Layers are tags on polygons, not physical 3D layers.
The thickness/material info is metadata - the actual file stores 2D shapes with layer numbers.

Respond with a valid JSON object matching the GeometrySpec schema."""


# ---------------------------------------------------------------------------
# Stage 2: GDSFactory code generation - component-specific examples
# ---------------------------------------------------------------------------

COMPONENT_EXAMPLES: dict[str, str] = {
    "via_stack": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def via_stack(metal_size: float = 2.0, via_size: float = 0.5) -> gf.Component:
    """Single via stack: metal1 pad + via + metal2 pad."""
    c = gf.Component()
    m1 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL1, centered=True)
    c.add_ref(m1)
    via = gf.components.rectangle(size=(via_size, via_size), layer=VIA1, centered=True)
    c.add_ref(via)
    m2 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL2, centered=True)
    c.add_ref(m2)
    return c

if __name__ == "__main__":
    c = via_stack()
    c.write_gds("via_stack.gds")
    c.write("via_stack.oas")
''',
    "via_array": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def single_via(metal_size: float = 2.0, via_size: float = 0.5) -> gf.Component:
    """Single via element with metal pads on both layers."""
    c = gf.Component()
    m1 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL1, centered=True)
    c.add_ref(m1)
    via = gf.components.rectangle(size=(via_size, via_size), layer=VIA1, centered=True)
    c.add_ref(via)
    m2 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL2, centered=True)
    c.add_ref(m2)
    return c

@gf.cell
def via_array(rows: int = 3, cols: int = 3, pitch: float = 5.0,
              metal_size: float = 2.0, via_size: float = 0.5) -> gf.Component:
    """Array of via stacks connecting metal1 to metal2."""
    c = gf.Component()
    single = single_via(metal_size=metal_size, via_size=via_size)
    arr = gf.components.array(
        component=single, columns=cols, rows=rows,
        column_pitch=pitch, row_pitch=pitch,
    )
    c.add_ref(arr)
    return c

if __name__ == "__main__":
    c = via_array()
    c.write_gds("via_array.gds")
    c.write("via_array.oas")
''',
    "interconnect": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def interconnect(num_traces: int = 3, trace_width: float = 2.0,
                 spacing: float = 10.0) -> gf.Component:
    """Two-layer routing: horizontal on metal1, vertical on metal2, vias at intersections."""
    c = gf.Component()
    total_span = (num_traces - 1) * spacing
    width = total_span + 20
    height = total_span + 10

    # Metal1: horizontal traces
    for i in range(num_traces):
        y = -total_span / 2 + i * spacing
        trace = gf.components.rectangle(size=(width, trace_width), layer=METAL1, centered=True)
        ref = c.add_ref(trace)
        ref.dmove((0, y))

    # Metal2: vertical traces
    for i in range(num_traces):
        x = -total_span / 2 + i * spacing
        trace = gf.components.rectangle(size=(trace_width, height), layer=METAL2, centered=True)
        ref = c.add_ref(trace)
        ref.dmove((x, 0))

    # Vias at intersections
    for i in range(num_traces):
        for j in range(num_traces):
            x = -total_span / 2 + i * spacing
            y = -total_span / 2 + j * spacing
            via = gf.components.rectangle(size=(1.0, 1.0), layer=VIA1, centered=True)
            ref = c.add_ref(via)
            ref.dmove((x, y))

    return c

if __name__ == "__main__":
    c = interconnect()
    c.write_gds("interconnect.gds")
    c.write("interconnect.oas")
''',
    "mim_capacitor": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
METAL2 = (3, 0)

@gf.cell
def mim_capacitor(plate_width: float = 20.0, plate_height: float = 20.0,
                  extension: float = 5.0) -> gf.Component:
    """MIM capacitor: bottom plate on metal1, top plate on metal2 with contact extensions."""
    c = gf.Component()

    # Bottom plate extends left for contact
    bottom = gf.components.rectangle(
        size=(plate_width + extension, plate_height), layer=METAL1,
    )
    ref_b = c.add_ref(bottom)
    ref_b.dmove((-extension, 0))

    # Top plate extends right for contact
    top = gf.components.rectangle(
        size=(plate_width + extension, plate_height), layer=METAL2,
    )
    c.add_ref(top)

    return c

if __name__ == "__main__":
    c = mim_capacitor()
    c.write_gds("mim_capacitor.gds")
    c.write("mim_capacitor.oas")
''',
    "guard_ring": '''import math

import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def guard_ring(inner_size: float = 50.0, ring_width: float = 5.0,
               via_size: float = 1.0, via_pitch: float = 3.0) -> gf.Component:
    """Guard ring with via stitching between metal1 and metal2."""
    c = gf.Component()
    ring_center_radius = (inner_size + ring_width) / 2

    # Rings on both metal layers
    for layer in [METAL1, METAL2]:
        ring = gf.components.ring(radius=ring_center_radius, width=ring_width, layer=layer)
        c.add_ref(ring)

    # Via stitching around the ring
    circumference = 2 * math.pi * ring_center_radius
    num_vias = int(circumference / via_pitch)

    for i in range(num_vias):
        angle = 2 * math.pi * i / num_vias
        x = ring_center_radius * math.cos(angle)
        y = ring_center_radius * math.sin(angle)
        via = gf.components.rectangle(size=(via_size, via_size), layer=VIA1, centered=True)
        ref = c.add_ref(via)
        ref.dmove((x, y))

    return c

if __name__ == "__main__":
    c = guard_ring()
    c.write_gds("guard_ring.gds")
    c.write("guard_ring.oas")
''',
    "bond_pad": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def single_via(via_size: float = 2.0) -> gf.Component:
    c = gf.Component()
    via = gf.components.rectangle(size=(via_size, via_size), layer=VIA1, centered=True)
    c.add_ref(via)
    return c

@gf.cell
def bond_pad(pad_size: float = 80.0, via_rows: int = 5, via_cols: int = 5,
             via_size: float = 2.0, via_pitch: float = 10.0) -> gf.Component:
    """Bond pad on metal2 with via array down to metal1."""
    c = gf.Component()

    # Metal1 ground plane
    m1 = gf.components.rectangle(size=(pad_size, pad_size), layer=METAL1, centered=True)
    c.add_ref(m1)

    # Via array
    sv = single_via(via_size=via_size)
    arr = gf.components.array(
        component=sv, columns=via_cols, rows=via_rows,
        column_pitch=via_pitch, row_pitch=via_pitch,
    )
    ref_arr = c.add_ref(arr)
    # Center the array on the pad
    arr_width = (via_cols - 1) * via_pitch
    arr_height = (via_rows - 1) * via_pitch
    ref_arr.dmove((-arr_width / 2, -arr_height / 2))

    # Metal2 top pad
    m2 = gf.components.rectangle(size=(pad_size, pad_size), layer=METAL2, centered=True)
    c.add_ref(m2)

    return c

if __name__ == "__main__":
    c = bond_pad()
    c.write_gds("bond_pad.gds")
    c.write("bond_pad.oas")
''',
    "inductor": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def inductor(turns: int = 3, inner_radius: float = 10.0,
             trace_width: float = 2.0, spacing: float = 2.0) -> gf.Component:
    """Spiral inductor on metal2 with underpass on metal1."""
    c = gf.Component()

    # Main spiral on top metal
    spiral = gf.components.spiral(
        nturns=turns, min_radius=inner_radius,
        width=trace_width, spacing=spacing, layer=METAL2,
    )
    c.add_ref(spiral)

    # Underpass on metal1 (connects inner end to outside)
    outer_radius = inner_radius + turns * (trace_width + spacing)
    underpass = gf.components.rectangle(
        size=(outer_radius + 5, trace_width), layer=METAL1, centered=True,
    )
    c.add_ref(underpass)

    # Via connections at underpass ends
    for x_pos in [-(outer_radius + 5) / 2 + trace_width / 2,
                  (outer_radius + 5) / 2 - trace_width / 2]:
        via = gf.components.rectangle(size=(trace_width, trace_width), layer=VIA1, centered=True)
        ref = c.add_ref(via)
        ref.dmove((x_pos, 0))

    return c

if __name__ == "__main__":
    c = inductor()
    c.write_gds("inductor.gds")
    c.write("inductor.oas")
''',
    "layer_stack": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

@gf.cell
def layer_stack(width: float = 10.0, height: float = 10.0,
                layers: list[tuple[int, int]] | None = None) -> gf.Component:
    """Simple overlapping rectangles on multiple layers."""
    c = gf.Component()
    if layers is None:
        layers = [(1, 0), (2, 0), (3, 0)]

    for layer in layers:
        rect = gf.components.rectangle(size=(width, height), layer=layer, centered=True)
        c.add_ref(rect)

    return c

if __name__ == "__main__":
    c = layer_stack()
    c.write_gds("layer_stack.gds")
    c.write("layer_stack.oas")
''',
    "test_pattern": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)

@gf.cell
def test_pattern(size: float = 10.0, spacing: float = 20.0) -> gf.Component:
    """Test pattern with various geometric shapes."""
    c = gf.Component()

    # Rectangle
    rect = gf.components.rectangle(size=(size, size), layer=METAL1, centered=True)
    c.add_ref(rect)

    # Circle
    circle = gf.components.circle(radius=size / 2, layer=METAL1)
    ref_c = c.add_ref(circle)
    ref_c.dmove((spacing, 0))

    # Cross (two overlapping rectangles)
    h_bar = gf.components.rectangle(size=(size, size / 3), layer=METAL1, centered=True)
    v_bar = gf.components.rectangle(size=(size / 3, size), layer=METAL1, centered=True)
    ref_h = c.add_ref(h_bar)
    ref_h.dmove((2 * spacing, 0))
    ref_v = c.add_ref(v_bar)
    ref_v.dmove((2 * spacing, 0))

    return c

if __name__ == "__main__":
    c = test_pattern()
    c.write_gds("test_pattern.gds")
    c.write("test_pattern.oas")
''',
    "curved_trace_bundle": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)

@gf.cell
def curved_trace_bundle(
    num_traces: int = 10,
    trace_width: float = 8.0,
    trace_spacing: float = 8.0,
    lead_length: float = 250.0,
    bend_radius_start: float = 120.0,
    bend_angle: float = 200.0,
    tail_length: float = 45.0,
) -> gf.Component:
    """Parallel curved bundle with concentric bends."""
    c = gf.Component()
    pitch = trace_width + trace_spacing
    bundle_height = (num_traces - 1) * pitch + trace_width

    # Left bus bar where traces start
    left_bus = gf.components.rectangle(
        size=(trace_width * 2, bundle_height), layer=METAL1, centered=True,
    )
    left_ref = c.add_ref(left_bus)
    left_ref.dmove((0, 0))

    for i in range(num_traces):
        y_offset = -bundle_height / 2 + trace_width / 2 + i * pitch
        radius = bend_radius_start + i * pitch

        path = gf.path.Path()
        path.append(gf.path.straight(length=lead_length))
        path.append(gf.path.euler(radius=radius, angle=bend_angle))
        path.append(gf.path.straight(length=tail_length))

        route = gf.path.extrude(path, width=trace_width, layer=METAL1)
        ref = c.add_ref(route)
        ref.dmove((trace_width, y_offset))

    return c

if __name__ == "__main__":
    c = curved_trace_bundle()
    c.write_gds("curved_trace_bundle.gds")
    c.write("curved_trace_bundle.oas")
''',
    "comb_serpentine": '''import numpy as np

import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)

@gf.cell
def comb_serpentine(
    trace_width: float = 6.0,
    center_pad_size: float = 60.0,
    num_fingers: int = 30,
    finger_pitch: float = 8.0,
    rail_y: float = 0.0,
    pad_center_y: float = 55.0,
    min_finger_length: float = 80.0,
    max_finger_length: float = 420.0,
    bend_radius: float = 8.0,
) -> gf.Component:
    """Horizontal comb fanout from center pad into bottom rail."""
    c = gf.Component()

    # Center pad
    center_pad = gf.components.rectangle(
        size=(center_pad_size, center_pad_size), layer=METAL1, centered=True,
    )
    pad_ref = c.add_ref(center_pad)
    pad_ref.dmove((0, pad_center_y))

    # Bottom rail
    rail_length = 2 * (max_finger_length + center_pad_size / 2 + 80.0)
    rail = gf.components.rectangle(size=(rail_length, trace_width), layer=METAL1, centered=True)
    rail_ref = c.add_ref(rail)
    rail_ref.dmove((0, rail_y))

    # Fingers from center pad to rail
    per_side = max(1, num_fingers // 2)
    y_start = pad_center_y - ((per_side - 1) * finger_pitch) / 2
    y_coords = [y_start + i * finger_pitch for i in range(per_side)]
    lengths = np.linspace(min_finger_length, max_finger_length, per_side)

    for y_local, finger_length in zip(y_coords, lengths, strict=False):
        vertical_drop = max(10.0, y_local - rail_y - trace_width)

        # Right side finger
        p = gf.path.Path()
        p.append(gf.path.straight(finger_length))
        p.append(gf.path.euler(radius=bend_radius, angle=-90))
        p.append(gf.path.straight(vertical_drop))
        right_finger = gf.path.extrude(p, width=trace_width, layer=METAL1)
        right_ref = c.add_ref(right_finger)
        right_ref.dmove((center_pad_size / 2, y_local))

        # Left side finger (mirror of right)
        left_finger = gf.path.extrude(p, width=trace_width, layer=METAL1)
        left_ref = c.add_ref(left_finger)
        left_ref.dmove((-center_pad_size / 2, y_local))
        left_ref.mirror_y()

    return c

if __name__ == "__main__":
    c = comb_serpentine()
    c.write_gds("comb_serpentine.gds")
    c.write("comb_serpentine.oas")
''',
    "mom_capacitor": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)

@gf.cell
def mom_capacitor(finger_width: float = 1.0, finger_length: float = 20.0,
                  finger_gap: float = 1.0, num_fingers: int = 10) -> gf.Component:
    """Metal-Oxide-Metal interdigitated capacitor."""
    c = gf.Component()
    pitch = finger_width + finger_gap

    for i in range(num_fingers):
        x = i * pitch
        finger = gf.components.rectangle(size=(finger_width, finger_length), layer=METAL1)
        ref = c.add_ref(finger)
        # Alternate fingers extend from top and bottom buses
        if i % 2 == 0:
            ref.dmove((x, 0))
        else:
            ref.dmove((x, finger_gap))

    # Top bus
    total_width = num_fingers * pitch
    top_bus = gf.components.rectangle(
        size=(total_width, finger_width), layer=METAL1,
    )
    ref_top = c.add_ref(top_bus)
    ref_top.dmove((0, finger_length + finger_gap))

    # Bottom bus
    bottom_bus = gf.components.rectangle(
        size=(total_width, finger_width), layer=METAL1,
    )
    ref_bot = c.add_ref(bottom_bus)
    ref_bot.dmove((0, -finger_width))

    return c

if __name__ == "__main__":
    c = mom_capacitor()
    c.write_gds("mom_capacitor.gds")
    c.write("mom_capacitor.oas")
''',
    "differential_pair": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
METAL2 = (3, 0)

@gf.cell
def differential_pair(trace_width: float = 2.0, spacing: float = 5.0,
                      length: float = 100.0, ground_width: float = 10.0) -> gf.Component:
    """Differential pair with ground shielding on metal2."""
    c = gf.Component()

    # Signal traces on metal1
    for offset in [-spacing / 2, spacing / 2]:
        trace = gf.components.rectangle(size=(length, trace_width), layer=METAL1, centered=True)
        ref = c.add_ref(trace)
        ref.dmove((0, offset))

    # Ground shield on metal2
    total_height = spacing + trace_width + 2 * ground_width
    ground = gf.components.rectangle(
        size=(length, total_height), layer=METAL2, centered=True,
    )
    c.add_ref(ground)

    return c

if __name__ == "__main__":
    c = differential_pair()
    c.write_gds("differential_pair.gds")
    c.write("differential_pair.oas")
''',
    "seal_ring": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def seal_ring(chip_width: float = 1000.0, chip_height: float = 1000.0,
              ring_width: float = 5.0) -> gf.Component:
    """Chip-edge seal ring on multiple metal layers with via stitching."""
    c = gf.Component()

    for layer in [METAL1, METAL2]:
        # Outer rectangle
        outer = gf.components.rectangle(
            size=(chip_width, chip_height), layer=layer, centered=True,
        )
        inner = gf.components.rectangle(
            size=(chip_width - 2 * ring_width, chip_height - 2 * ring_width),
            layer=layer, centered=True,
        )
        ring = gf.boolean(A=outer, B=inner, operation="not", layer=layer)
        c.add_ref(ring)

    # Via stitching along ring perimeter
    via_pitch = 5.0
    for side_length, axis, sign in [
        (chip_width, "x", 1), (chip_width, "x", -1),
        (chip_height, "y", 1), (chip_height, "y", -1),
    ]:
        num_vias = int(side_length / via_pitch)
        for i in range(num_vias):
            pos = -side_length / 2 + i * via_pitch + via_pitch / 2
            via = gf.components.rectangle(size=(1.0, 1.0), layer=VIA1, centered=True)
            ref = c.add_ref(via)
            if axis == "x":
                ref.dmove((pos, sign * (chip_height / 2 - ring_width / 2)))
            else:
                ref.dmove((sign * (chip_width / 2 - ring_width / 2), pos))

    return c

if __name__ == "__main__":
    c = seal_ring()
    c.write_gds("seal_ring.gds")
    c.write("seal_ring.oas")
''',
    "alignment_mark": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)

@gf.cell
def alignment_mark(size: float = 50.0, bar_width: float = 5.0) -> gf.Component:
    """Cross-shaped alignment mark for lithography."""
    c = gf.Component()

    # Horizontal bar
    h_bar = gf.components.rectangle(size=(size, bar_width), layer=METAL1, centered=True)
    c.add_ref(h_bar)

    # Vertical bar
    v_bar = gf.components.rectangle(size=(bar_width, size), layer=METAL1, centered=True)
    c.add_ref(v_bar)

    return c

if __name__ == "__main__":
    c = alignment_mark()
    c.write_gds("alignment_mark.gds")
    c.write("alignment_mark.oas")
''',
    "transmission_line": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)
METAL2 = (3, 0)

@gf.cell
def transmission_line(signal_width: float = 5.0, gap_width: float = 3.0,
                      length: float = 200.0, ground_width: float = 20.0) -> gf.Component:
    """Coplanar waveguide transmission line."""
    c = gf.Component()

    # Signal trace on metal2
    signal = gf.components.rectangle(size=(length, signal_width), layer=METAL2, centered=True)
    c.add_ref(signal)

    # Ground planes on metal1 (both sides)
    for sign in [-1, 1]:
        y_offset = sign * (signal_width / 2 + gap_width + ground_width / 2)
        ground = gf.components.rectangle(
            size=(length, ground_width), layer=METAL1, centered=True,
        )
        ref = c.add_ref(ground)
        ref.dmove((0, y_offset))

    return c

if __name__ == "__main__":
    c = transmission_line()
    c.write_gds("transmission_line.gds")
    c.write("transmission_line.oas")
''',
    "meander": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

METAL1 = (1, 0)

@gf.cell
def meander(trace_width: float = 2.0, num_turns: int = 5,
            meander_width: float = 20.0, meander_length: float = 50.0) -> gf.Component:
    """Meandered trace for delay matching."""
    c = gf.Component()
    pitch = meander_width + trace_width

    # Create meander segments
    for i in range(num_turns):
        y_base = i * pitch

        # Horizontal segment
        h_seg = gf.components.rectangle(
            size=(meander_length, trace_width), layer=METAL1,
        )
        ref_h = c.add_ref(h_seg)
        ref_h.dmove((0, y_base))

        # Vertical connector to next row (alternating left/right)
        if i < num_turns - 1:
            v_seg = gf.components.rectangle(
                size=(trace_width, pitch), layer=METAL1,
            )
            ref_v = c.add_ref(v_seg)
            if i % 2 == 0:
                ref_v.dmove((meander_length - trace_width, y_base))
            else:
                ref_v.dmove((0, y_base))

    return c

if __name__ == "__main__":
    c = meander()
    c.write_gds("meander.gds")
    c.write("meander.oas")
''',
    "multi_layer_rectangles": '''import gdsfactory as gf

gf.gpdk.PDK.activate()

@gf.cell
def multi_layer_rectangles() -> gf.Component:
    """Multi-range rectangles with per-range sizing and placement."""
    c = gf.Component("multi_layer_rectangles")

    # Range 1: layers 1-10 (squares, growing from 100 to 1000 um)
    for i in range(1, 11):
        t = (i - 1) / max(1, 9)
        w = 100.0 + t * (1000.0 - 100.0)
        h = 100.0 + t * (1000.0 - 100.0)
        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
        c.add_ref(rect)

    # Range 2: layers 11-20 (rectangles, offset to y=+2000)
    for i in range(11, 21):
        t = (i - 11) / max(1, 9)
        w = 200.0 + t * (2000.0 - 200.0)
        h = 50.0 + t * (500.0 - 50.0)
        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
        ref = c.add_ref(rect)
        ref.dmove((0.0, 2000.0))

    return c

if __name__ == "__main__":
    c = multi_layer_rectangles()
    c.write_gds("multi_layer_rectangles.gds")
    c.write("multi_layer_rectangles.oas")
''',
}


# ---------------------------------------------------------------------------
# Common mistakes to include in code generation prompts
# ---------------------------------------------------------------------------

COMMON_MISTAKES = """
=== COMMON MISTAKES TO AVOID ===

1. ref.dmove(x=10, y=5) -- WRONG! Use ref.dmove((10, 5)) with a TUPLE
2. c.add_array(...) -- WRONG! Use gf.components.array(...) then c.add_ref(arr)
3. Putting the SAME shape on every layer -- this is NOT multi-layer geometry.
   Each layer must have a DIFFERENT shape (e.g., via is smaller than metal pads).
4. Using movex/movey on a Component instead of a Reference
5. Forgetting gf.gpdk.PDK.activate() at the start
6. Writing .oas with write_oas() -- WRONG! Use c.write("file.oas")
7. Nesting @gf.cell definitions inside other @gf.cell functions
8. Making via_size >= metal_size -- via MUST be smaller than metal pads
9. Passing a function instead of a function call to gf.components.array:
   WRONG: gf.components.array(component=my_func, ...)
   RIGHT: gf.components.array(component=my_func(), ...)
10. Missing c.write_gds("output.gds") and c.write("output.oas") in __main__ block
11. Using Path.end_point or Path.end_orientation directly (these attributes may not exist)
"""


# ---------------------------------------------------------------------------
# Template for code generation prompt (stage 2)
# ---------------------------------------------------------------------------

GDSFACTORY_CODE_PROMPT_TEMPLATE = """You are a GDSFactory expert. Generate valid Python code \
using GDSFactory to create the specified semiconductor geometry.

=== SPECIFICATION ===
Component Type: {component_type}
Description: {description}
Parameters: {parameters}
Layers: {layers}
{geometry_ranges_section}{original_prompt_section}

CRITICAL: GDS is a 2D format. Create DIFFERENT shapes on DIFFERENT layers for true multi-layer \
geometry. Do NOT just put the same rectangle on every layer - that's unrealistic.

=== VERIFIED WORKING EXAMPLE FOR {component_type} ===
The following code is verified to work correctly. Use it as a reference for style and API usage:

```python
{component_example}
```

=== GDSFACTORY API QUICK REFERENCE ===

1. SETUP:
```python
import gdsfactory as gf
gf.gpdk.PDK.activate()
```

2. SHAPES:
```python
gf.components.rectangle(size=(w, h), layer=(L, D), centered=True)
gf.components.circle(radius=r, layer=(L, D))
gf.components.ring(radius=R, width=W, layer=(L, D))
```

3. POSITIONING - dmove takes a TUPLE:
```python
ref = c.add_ref(shape)
ref.dmove((dx, dy))  # CORRECT: tuple
# ref.dmove(x=dx, y=dy)  # WRONG - will error!
```

4. ARRAYS - use gf.components.array, NOT c.add_array:
```python
arr = gf.components.array(component=element(), columns=N, rows=M,
                          column_pitch=X, row_pitch=Y)
c.add_ref(arr)
```

5. BOOLEAN OPERATIONS:
```python
gf.boolean(A=shape1, B=shape2, operation="not", layer=(L, D))
```

6. PATH ROUTING COMPATIBILITY:
```python
p = gf.path.Path()
p.append(gf.path.straight(50))
p.append(gf.path.arc(radius=20, angle=90))
trace = gf.path.extrude(p, width=2.0, layer=(1, 0))
ref = c.add_ref(trace)

# Do NOT use p.end_point / p.end_orientation (not available in some versions).
# Use ports from the extruded reference instead.
end_port = ref.ports["o2"]
cap_center = tuple(end_port.center)
cap_orientation = end_port.orientation
```

7. EXPORT (always include BOTH):
```python
c.write_gds("output.gds")
c.write("output.oas")
```

{common_mistakes}

=== REQUIRED OUTPUT STRUCTURE ===
Your code MUST:
1. Import gdsfactory and activate PDK
2. Define layer constants matching the specification
3. Use @gf.cell decorator on component functions
4. Create DIFFERENT shapes on DIFFERENT layers
5. Use the EXACT parameter values from the specification
6. Include if __name__ == "__main__" block
7. Call BOTH write_gds() and write() for GDS and OAS export

Generate complete, runnable Python code:"""


# ---------------------------------------------------------------------------
# Legacy template (kept for backward compatibility)
# ---------------------------------------------------------------------------

GDSFACTORY_CODE_PROMPT = """You are a GDSFactory expert. Generate valid Python code using \
GDSFactory to create the specified semiconductor geometry.

Component Type: {component_type}
Description: {description}
Parameters: {parameters}
Layers: {layers}

CRITICAL: GDS is a 2D format. Create DIFFERENT shapes on DIFFERENT layers for true multi-layer \
geometry.
Do NOT just put the same rectangle on every layer - that's unrealistic.

=== GDSFACTORY API REFERENCE ===

1. BASIC SETUP:
```python
import gdsfactory as gf
gf.gpdk.PDK.activate()  # Required first!
```

2. COMPONENT CREATION:
```python
@gf.cell
def my_component() -> gf.Component:
    c = gf.Component()
    # Add shapes here
    return c
```

3. BASIC SHAPES:
```python
rect = gf.components.rectangle(size=(width, height), layer=(layer_num, datatype), centered=True)
c.add_ref(rect)
circle = gf.components.circle(radius=r, layer=(L, D))
ring = gf.components.ring(radius=R, width=W, layer=(L, D))
```

4. POSITIONING SHAPES:
```python
ref = c.add_ref(shape)
ref.dmove((dx, dy))  # CORRECT: tuple argument
# WRONG: ref.dmove(x=dx, y=dy) - this will error!
```

5. ARRAYS (CRITICAL - use gf.components.array, NOT c.add_array):
```python
arr = gf.components.array(
    component=single_element,
    columns=5, rows=5,
    column_pitch=2.0, row_pitch=2.0,
)
c.add_ref(arr)
```

6. MULTI-LAYER VIA STACK (KEY PATTERN):
```python
METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def via_stack(metal_size=2.0, via_size=0.5):
    c = gf.Component()
    m1 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL1, centered=True)
    c.add_ref(m1)
    via = gf.components.rectangle(size=(via_size, via_size), layer=VIA1, centered=True)
    c.add_ref(via)
    m2 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL2, centered=True)
    c.add_ref(m2)
    return c
```

=== CODE STRUCTURE ===
```python
import gdsfactory as gf
gf.gpdk.PDK.activate()

METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)

@gf.cell
def my_component(...parameters...) -> gf.Component:
    c = gf.Component()
    return c

if __name__ == "__main__":
    c = my_component()
    c.show()
    c.write_gds("output.gds")
    c.write("output.oas")
```

Generate complete, runnable Python code with TRUE multi-layer geometry (different shapes per \
layer):"""


# ---------------------------------------------------------------------------
# Helper: build a component-specific code prompt
# ---------------------------------------------------------------------------


def build_code_prompt(
    spec_dict: dict[str, Any],
    original_prompt: str | None = None,
) -> str:
    """Build a component-specific code generation prompt.

    Uses verified working code examples for the given component type
    instead of the generic prompt template.

    Args:
        spec_dict: Dictionary with component_type, description, parameters, layers.
        original_prompt: The original natural-language user request (optional).

    Returns:
        Formatted prompt string for the LLM.
    """
    component_type = spec_dict.get("component_type", "")
    description = spec_dict.get("description", "")
    parameters = spec_dict.get("parameters", {})
    layers = spec_dict.get("layers", [])

    # Select the best matching example
    example = COMPONENT_EXAMPLES.get(component_type)
    if example is None:
        # Fallback: try to match partial names
        for key, val in COMPONENT_EXAMPLES.items():
            if key in component_type or component_type in key:
                example = val
                break
    if example is None:
        # Ultimate fallback: use via_array as a general pattern
        example = COMPONENT_EXAMPLES["via_array"]

    # Enhance "test_pattern" routing for custom curved/comb requests.
    # These prompts are often classified as test_pattern in stage 1,
    # but need a more specific example to get visually similar geometry.
    if component_type == "test_pattern":
        combined_text = f"{description} {original_prompt or ''}".lower()
        serpentine_keywords = ("comb", "serpentine", "finger", "fanout", "bottom rail")
        curved_keywords = ("curved", "arc", "bundle", "euler", "concentric")

        if any(keyword in combined_text for keyword in serpentine_keywords):
            example = COMPONENT_EXAMPLES["comb_serpentine"]
        elif any(keyword in combined_text for keyword in curved_keywords):
            example = COMPONENT_EXAMPLES["curved_trace_bundle"]

    # Build the geometry_ranges section (optional)
    geometry_ranges = spec_dict.get("geometry_ranges", [])
    geometry_ranges_section = ""
    if geometry_ranges:
        geometry_ranges_section = f"Geometry Ranges: {json.dumps(geometry_ranges, indent=2)}\n"

    # Build the original prompt section
    original_prompt_section = ""
    if original_prompt:
        original_prompt_section = f"Original User Request: {original_prompt}"

    return GDSFACTORY_CODE_PROMPT_TEMPLATE.format(
        component_type=component_type,
        description=description,
        parameters=json.dumps(parameters, indent=2) if isinstance(parameters, dict) else parameters,
        layers=json.dumps(layers, indent=2) if isinstance(layers, list) else layers,
        geometry_ranges_section=geometry_ranges_section,
        original_prompt_section=original_prompt_section,
        component_example=example,
        common_mistakes=COMMON_MISTAKES,
    )
