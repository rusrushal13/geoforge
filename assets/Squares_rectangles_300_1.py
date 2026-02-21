import gdsfactory as gf

gf.gpdk.PDK.activate()

@gf.cell
def multi_layer_rectangles() -> gf.Component:
    """Multi-range rectangles with per-range sizing and placement."""
    c = gf.Component("LAYERS_1_300")

    # Layers 1-100: square
    for i in range(1, 100 + 1):
        t = (i - 1) / max(1, 99)
        w = 1000.0 + t * (100000.0 - 1000.0)
        h = 1000.0 + t * (100000.0 - 1000.0)
        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
        c.add_ref(rect)

    # Layers 101-200: rectangle
    for i in range(101, 200 + 1):
        t = (i - 101) / max(1, 99)
        w = 5000.0 + t * (100000.0 - 5000.0)
        h = 1000.0 + t * (20000.0 - 1000.0)
        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
        c.add_ref(rect)

    # Layers 201-300: rectangle
    for i in range(201, 300 + 1):
        t = (i - 201) / max(1, 99)
        w = 5000.0 + t * (100000.0 - 5000.0)
        h = 1000.0 + t * (20000.0 - 1000.0)
        rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
        ref = c.add_ref(rect)
        ref.dmove((0.0, 30000.0))

    return c

if __name__ == "__main__":
    c = multi_layer_rectangles()
    c.write_gds("multi_layer_rectangles.gds")
    c.write("multi_layer_rectangles.oas")
