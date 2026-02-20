"""Tests for deterministic template generation."""

import pytest

from geoforge.llm.base import GeometryRange, GeometrySpec, LayerSpec
from geoforge.templates import generate_from_template, has_template, list_templates


class TestTemplateRegistry:
    """Tests for template registration and lookup."""

    def test_curved_trace_bundle_registered(self):
        assert has_template("curved_trace_bundle")

    def test_comb_serpentine_registered(self):
        assert has_template("comb_serpentine")

    def test_multi_layer_rectangles_registered(self):
        assert has_template("multi_layer_rectangles")

    def test_unknown_type_not_registered(self):
        assert not has_template("nonexistent_type_xyz")

    def test_list_templates_includes_all(self):
        names = list_templates()
        assert "curved_trace_bundle" in names
        assert "comb_serpentine" in names
        assert "multi_layer_rectangles" in names

    def test_generate_unknown_type_raises(self):
        spec = GeometrySpec(
            component_type="nonexistent_xyz",
            description="test",
        )
        with pytest.raises(KeyError, match="No template registered"):
            generate_from_template(spec)


class TestCurvedTraceBundleTemplate:
    """Tests for curved_trace_bundle code generation."""

    def test_generates_valid_python(self):
        spec = GeometrySpec(
            component_type="curved_trace_bundle",
            description="10-trace curved bundle",
            parameters={
                "num_traces": 10,
                "trace_width": 8.0,
                "trace_spacing": 8.0,
                "lead_length": 300.0,
                "bend_radius_start": 140.0,
                "bend_angle": 200.0,
                "tail_length": 50.0,
            },
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        code = generate_from_template(spec)
        assert "import gdsfactory as gf" in code
        assert "def curved_trace_bundle" in code
        assert "gf.path.euler" in code
        assert "write_gds" in code
        assert "write(" in code

    def test_uses_spec_parameters(self):
        spec = GeometrySpec(
            component_type="curved_trace_bundle",
            description="custom bundle",
            parameters={"num_traces": 5, "trace_width": 12.0, "bend_angle": 180.0},
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        code = generate_from_template(spec)
        assert "num_traces = 5" in code
        assert "trace_width = 12.0" in code
        assert "bend_angle = 180.0" in code

    def test_uses_spec_layer(self):
        spec = GeometrySpec(
            component_type="curved_trace_bundle",
            description="test",
            parameters={"num_traces": 3},
            layers=[LayerSpec(layer_number=5, datatype=0, name="metal5")],
        )
        code = generate_from_template(spec)
        assert "LAYER = (5, 0)" in code

    def test_defaults_layer_when_empty(self):
        spec = GeometrySpec(
            component_type="curved_trace_bundle",
            description="test",
            parameters={},
        )
        code = generate_from_template(spec)
        assert "LAYER = (1, 0)" in code

    @pytest.mark.slow
    def test_generated_code_executes(self):
        from geoforge.core.validator import validate_execution, validate_syntax

        spec = GeometrySpec(
            component_type="curved_trace_bundle",
            description="test",
            parameters={"num_traces": 3, "lead_length": 50.0, "bend_radius_start": 30.0},
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        code = generate_from_template(spec)
        ok, err = validate_syntax(code)
        assert ok, f"Syntax error: {err}"
        ok, err, info = validate_execution(code)
        assert ok, f"Execution error: {err}"
        assert info["gds_files"] or info["oas_files"]


class TestCombSerpentineTemplate:
    """Tests for comb_serpentine code generation."""

    def test_generates_valid_python(self):
        spec = GeometrySpec(
            component_type="comb_serpentine",
            description="30-finger comb",
            parameters={
                "trace_width": 6.0,
                "center_pad_size": 60.0,
                "num_fingers": 30,
                "bend_radius": 8.0,
            },
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        code = generate_from_template(spec)
        assert "import gdsfactory as gf" in code
        assert "def comb_serpentine" in code
        assert "gf.path.euler" in code
        assert "mirror_y" in code
        assert "write_gds" in code

    def test_uses_spec_parameters(self):
        spec = GeometrySpec(
            component_type="comb_serpentine",
            description="test",
            parameters={"num_fingers": 20, "trace_width": 4.0, "max_finger_length": 300.0},
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        code = generate_from_template(spec)
        assert "num_fingers = 20" in code
        assert "trace_width = 4.0" in code
        assert "max_finger_length = 300.0" in code

    @pytest.mark.slow
    def test_generated_code_executes(self):
        from geoforge.core.validator import validate_execution, validate_syntax

        spec = GeometrySpec(
            component_type="comb_serpentine",
            description="test",
            parameters={"num_fingers": 6, "min_finger_length": 20.0, "max_finger_length": 60.0},
            layers=[LayerSpec(layer_number=1, datatype=0, name="metal1")],
        )
        code = generate_from_template(spec)
        ok, err = validate_syntax(code)
        assert ok, f"Syntax error: {err}"
        ok, err, info = validate_execution(code)
        assert ok, f"Execution error: {err}"
        assert info["gds_files"] or info["oas_files"]


class TestMultiLayerRectanglesTemplate:
    """Tests for multi_layer_rectangles code generation."""

    def test_generates_valid_python(self):
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="10-layer test",
            parameters={"num_layers": 10, "base_width": 1000.0, "base_height": 1000.0},
        )
        code = generate_from_template(spec)
        assert "import gdsfactory as gf" in code
        assert "def multi_layer_rectangles" in code
        assert "num_layers = 10" in code
        assert "write_gds" in code

    @pytest.mark.slow
    def test_generated_code_executes(self):
        from geoforge.core.validator import validate_execution, validate_syntax

        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            parameters={"num_layers": 5, "base_width": 100.0, "base_height": 100.0},
        )
        code = generate_from_template(spec)
        ok, err = validate_syntax(code)
        assert ok, f"Syntax error: {err}"
        ok, err, info = validate_execution(code)
        assert ok, f"Execution error: {err}"
        assert info["gds_files"] or info["oas_files"]


class TestTemplatePipelineIntegration:
    """Tests that templates integrate correctly with the LLM pipeline."""

    @pytest.mark.asyncio
    async def test_generate_uses_template_when_available(self):
        """Pipeline should use template instead of LLM for registered types."""
        from tests.mock_provider import MockLLMProvider

        provider = MockLLMProvider()
        # Override the spec response to return a template-backed type
        provider.spec_response = {
            "component_type": "curved_trace_bundle",
            "description": "test curved bundle",
            "parameters": {"num_traces": 3, "lead_length": 50.0, "bend_radius_start": 30.0},
            "layers": [{"layer_number": 1, "datatype": 0, "name": "metal1"}],
        }

        spec = await provider.generate("Create a curved trace bundle")

        # Code should come from template, not the mock's default code
        assert spec.gdsfactory_code is not None
        assert "def curved_trace_bundle" in spec.gdsfactory_code
        # Mock's code_gen should NOT have been called (template was used)
        assert provider._code_calls == 0


class TestMultiLayerRangesTemplate:
    """Tests for multi_layer_rectangles template with geometry_ranges."""

    def test_generates_per_range_loops(self):
        """Template with ranges should generate per-range for-loops."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            parameters={"num_layers": 20},
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=10,
                    shape="square",
                    width_start=100.0,
                    width_end=1000.0,
                    height_start=100.0,
                    height_end=1000.0,
                ),
                GeometryRange(
                    start_layer=11,
                    end_layer=20,
                    shape="rectangle",
                    width_start=500.0,
                    width_end=5000.0,
                    height_start=200.0,
                    height_end=2000.0,
                    center_y=5000.0,
                ),
            ],
        )
        code = generate_from_template(spec)
        # Both ranges should have for-loops
        assert "range(1, 10 + 1)" in code
        assert "range(11, 20 + 1)" in code
        # Non-zero center_y should produce dmove
        assert "dmove" in code
        assert "5000.0" in code

    def test_falls_back_to_single_formula_without_ranges(self):
        """Template without ranges should use the legacy single-formula path."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            parameters={"num_layers": 10, "base_width": 100.0, "base_height": 100.0},
        )
        code = generate_from_template(spec)
        # Should use the legacy path variables
        assert "base_width" in code
        assert "width_step" in code

    def test_single_layer_range(self):
        """Range with a single layer should not cause division by zero."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            parameters={"num_layers": 1},
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=1,
                    shape="rectangle",
                    width_start=500.0,
                    width_end=500.0,
                    height_start=300.0,
                    height_end=300.0,
                ),
            ],
        )
        code = generate_from_template(spec)
        assert "range(1, 1 + 1)" in code
        # max(1, 0) prevents division by zero
        assert "max(1, 0)" in code

    def test_origin_ranges_omit_dmove(self):
        """Ranges centered at origin should not produce dmove calls."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            parameters={"num_layers": 5},
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=5,
                    shape="square",
                    width_start=100.0,
                    width_end=500.0,
                    height_start=100.0,
                    height_end=500.0,
                ),
            ],
        )
        code = generate_from_template(spec)
        assert "dmove" not in code

    @pytest.mark.slow
    def test_range_template_executes(self):
        """Generated code from range template should execute successfully."""
        from geoforge.core.validator import validate_execution, validate_syntax

        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            parameters={"num_layers": 6},
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=3,
                    shape="square",
                    width_start=100.0,
                    width_end=300.0,
                    height_start=100.0,
                    height_end=300.0,
                ),
                GeometryRange(
                    start_layer=4,
                    end_layer=6,
                    shape="rectangle",
                    width_start=200.0,
                    width_end=600.0,
                    height_start=50.0,
                    height_end=150.0,
                    center_y=500.0,
                ),
            ],
        )
        code = generate_from_template(spec)
        ok, err = validate_syntax(code)
        assert ok, f"Syntax error: {err}"
        ok, err, info = validate_execution(code)
        assert ok, f"Execution error: {err}"
        assert info["gds_files"] or info["oas_files"]
