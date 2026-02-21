import gdsfactory as gf

# 1. Setup: Activate PDK
gf.gpdk.PDK.activate()

# 2. Define layer constants matching the specification
METAL1 = (1, 0)
VIA1 = (2, 0)
METAL2 = (3, 0)


@gf.cell
def single_via(metal_size: float = 0.6, via_size: float = 0.2) -> gf.Component:
    """
    Single via element with metal pads on both layers.
    Connects metal1 to metal2 using via1.
    """
    c = gf.Component()

    # Metal1 pad
    m1 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL1, centered=True)
    c.add_ref(m1)

    # Via1 connecting metal1 and metal2
    # CRITICAL: via_size must be smaller than metal_size
    if via_size >= metal_size:
        raise ValueError("via_size must be smaller than metal_size for proper enclosure.")
    via = gf.components.rectangle(size=(via_size, via_size), layer=VIA1, centered=True)
    c.add_ref(via)

    # Metal2 pad
    m2 = gf.components.rectangle(size=(metal_size, metal_size), layer=METAL2, centered=True)
    c.add_ref(m2)

    return c


@gf.cell
def via_array(
    rows: int = 5, cols: int = 5, pitch: float = 1.0, metal_size: float = 0.6, via_size: float = 0.2
) -> gf.Component:
    """
    Creates an array of via stacks connecting metal1 to metal2.

    Args:
        rows: Number of rows in the array.
        cols: Number of columns in the array.
        pitch: Center-to-center distance between adjacent vias in both X and Y.
        metal_size: Size of the square metal pads (metal1 and metal2).
        via_size: Size of the square via (via1).
    """
    c = gf.Component()

    # Create a single instance of the via stack
    single = single_via(metal_size=metal_size, via_size=via_size)

    # Create an array of the single via stack
    arr = gf.components.array(
        component=single,
        columns=cols,
        rows=rows,
        column_pitch=pitch,
        row_pitch=pitch,
    )
    c.add_ref(arr)

    return c


if __name__ == "__main__":
    # Create the via_array component with specified parameters
    c = via_array(rows=5, cols=5, pitch=1.0, metal_size=0.6, via_size=0.2)

    # Export the component to GDS and OAS files
    c.write_gds("via_array.gds")
    c.write("via_array.oas")

    print(f"Generated via_array.gds and via_array.oas with {c.name}")
    # Optional: Show the component in KLayout if available
    # gf.show(c)
