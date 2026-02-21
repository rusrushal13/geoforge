import numpy as np
import gdsfactory as gf

gf.gpdk.PDK.activate()

LAYER = (1, 0)

@gf.cell
def comb_serpentine() -> gf.Component:
    """Horizontal comb fanout from center pad into bottom rail."""
    c = gf.Component()

    trace_width = 6.0
    center_pad_size = 60.0
    num_fingers = 30
    finger_pitch = 10.0
    rail_y = -150.0
    pad_center_y = 0.0
    min_finger_length = 50.0
    max_finger_length = 200.0
    bend_radius = 10.0

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
