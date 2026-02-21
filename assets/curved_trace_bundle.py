import gdsfactory as gf

gf.gpdk.PDK.activate()

LAYER = (1, 0)

@gf.cell
def curved_trace_bundle() -> gf.Component:
    """Parallel curved trace bundle with concentric bends."""
    c = gf.Component()

    num_traces = 10
    trace_width = 8.0
    trace_spacing = 8.0
    pitch = trace_width + trace_spacing
    lead_length = 250.0
    bend_radius_start = 100.0
    bend_angle = 200.0
    tail_length = 20.0

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
