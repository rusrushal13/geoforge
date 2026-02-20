"""Tests for visualization module."""

import tempfile
from pathlib import Path

import pytest


class TestRenderComponent:
    """Tests for render_component."""

    @pytest.mark.slow
    def test_render_component_returns_figure(self):
        """Rendering a component should return a matplotlib Figure."""
        import gdsfactory as gf

        from geoforge.viz.renderer import render_component

        gf.gpdk.PDK.activate()
        c = gf.components.rectangle(size=(10, 10), layer=(1, 0))
        fig = render_component(c)
        assert fig is not None
        assert type(fig).__name__ == "Figure"

        import matplotlib.pyplot as plt

        plt.close(fig)

    @pytest.mark.slow
    def test_render_component_saves_png(self):
        """Rendering with save_path should write a PNG file."""
        import gdsfactory as gf

        from geoforge.viz.renderer import render_component

        gf.gpdk.PDK.activate()
        c = gf.components.rectangle(size=(10, 10), layer=(1, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "test.png"
            fig = render_component(c, save_path=png_path)
            assert png_path.exists()
            assert png_path.stat().st_size > 0

            import matplotlib.pyplot as plt

            plt.close(fig)


class TestRenderCode:
    """Tests for render_code."""

    @pytest.mark.slow
    def test_render_code_with_component(self):
        """Code that creates a component should render."""
        from geoforge.viz.renderer import render_code

        code = """
import gdsfactory as gf
gf.gpdk.PDK.activate()
c = gf.Component("test_render")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
"""
        fig = render_code(code)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_render_code_no_component(self):
        """Code without a component should return None."""
        from geoforge.viz.renderer import render_code

        code = "x = 1 + 1"
        fig = render_code(code)
        assert fig is None

    def test_render_code_bad_code(self):
        """Bad code should return None (not raise)."""
        from geoforge.viz.renderer import render_code

        code = "raise RuntimeError('test')"
        fig = render_code(code)
        assert fig is None

    @pytest.mark.slow
    def test_render_code_saves_png(self):
        """render_code with save_path should write a PNG file."""
        from geoforge.viz.renderer import render_code

        code = """
import gdsfactory as gf
gf.gpdk.PDK.activate()
c = gf.Component("save_test")
rect = gf.components.rectangle(size=(5, 5), layer=(1, 0))
c.add_ref(rect)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "preview.png"
            fig = render_code(code, save_path=png_path)
            assert fig is not None
            assert png_path.exists()

            import matplotlib.pyplot as plt

            plt.close(fig)


class TestRenderGdsFile:
    """Tests for render_gds_file."""

    @pytest.mark.slow
    def test_render_gds_file(self):
        """Should render a GDS file to a figure."""
        import gdsfactory as gf

        from geoforge.viz.renderer import render_gds_file

        gf.gpdk.PDK.activate()
        c = gf.components.rectangle(size=(10, 10), layer=(1, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = Path(tmpdir) / "test.gds"
            c.write_gds(str(gds_path))

            fig = render_gds_file(gds_path)
            assert fig is not None
            assert type(fig).__name__ == "Figure"

            import matplotlib.pyplot as plt

            plt.close(fig)
