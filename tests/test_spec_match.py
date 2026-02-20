"""Tests for spec-to-code matching validation."""

from geoforge.core.validator import SpecMatchResult, validate_spec_match
from geoforge.llm.base import GeometryRange, GeometrySpec, LayerSpec


class TestSpecMatchResult:
    """Tests for SpecMatchResult model."""

    def test_empty_result(self):
        result = SpecMatchResult()
        assert result.errors == []
        assert result.warnings == []

    def test_with_errors(self):
        result = SpecMatchResult(errors=["missing layer"], warnings=["param mismatch"])
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


class TestValidateSpecMatch:
    """Tests for validate_spec_match function."""

    def test_matching_code(self):
        """Code that matches the spec should have no errors."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
            layers=[
                LayerSpec(layer_number=1, datatype=0, name="metal1"),
                LayerSpec(layer_number=2, datatype=0, name="via1"),
            ],
            parameters={"pitch": 2.0},
        )
        code = """
import gdsfactory as gf
c = gf.Component("via_array")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
via = gf.components.rectangle(size=(2, 2), layer=(2, 0))
c.add_ref(rect)
c.add_ref(via)
pitch = 2.0
"""
        result = validate_spec_match(spec, code)
        assert len(result.errors) == 0

    def test_no_layers_found(self):
        """Code missing ALL layers should produce an error."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
            layers=[
                LayerSpec(layer_number=10, datatype=0, name="custom1"),
                LayerSpec(layer_number=20, datatype=0, name="custom2"),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("via_array")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert any("ANY" in err for err in result.errors)

    def test_some_layers_found(self):
        """Code with some layers should produce warnings, not errors."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
            layers=[
                LayerSpec(layer_number=1, datatype=0, name="metal1"),
                LayerSpec(layer_number=99, datatype=0, name="custom"),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("via_array")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        # Should NOT have "no layers" error since (1, 0) was found
        assert not any("ANY" in err for err in result.errors)
        # But should have a warning for the missing layer
        assert any("99" in w for w in result.warnings)

    def test_component_type_missing(self):
        """Code not referencing the component type should produce an error."""
        spec = GeometrySpec(
            component_type="mim_capacitor",
            description="test",
        )
        code = """
import gdsfactory as gf
c = gf.Component("my_design")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert any("mim_capacitor" in err for err in result.errors)

    def test_component_type_found_by_parts(self):
        """Component type found when split by underscore."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
        )
        code = """
import gdsfactory as gf
# Creating a via array pattern
c = gf.Component("test")
"""
        result = validate_spec_match(spec, code)
        # "via" or "array" should be found in the code
        assert not any("component_type" in err.lower() for err in result.errors)

    def test_parameter_warning(self):
        """Missing numeric parameter should produce a warning."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
            parameters={"pitch": 7.77, "rows": 5},
        )
        code = """
import gdsfactory as gf
c = gf.Component("via_array")
pitch = 2.0
rows = 5
"""
        result = validate_spec_match(spec, code)
        # pitch=7.77 not in code -> warning
        assert any("pitch" in w for w in result.warnings)
        # rows=5 IS in code -> no warning for it
        assert not any("rows" in w for w in result.warnings)

    def test_no_layers_no_error(self):
        """Spec with no layers should not produce layer errors."""
        spec = GeometrySpec(
            component_type="test",
            description="test",
            layers=[],
        )
        code = """
import gdsfactory as gf
c = gf.Component("test")
"""
        result = validate_spec_match(spec, code)
        assert not any("layer" in err.lower() for err in result.errors)

    def test_dynamic_layers_via_loop(self):
        """Code generating layers dynamically via loops should not produce layer errors."""
        spec = GeometrySpec(
            component_type="test_pattern",
            description="test",
            layers=[LayerSpec(layer_number=i, datatype=0, name=f"layer_{i}") for i in range(1, 11)],
        )
        code = """
import gdsfactory as gf
c = gf.Component("test_pattern")
LAYERS = {}
for i in range(1, 301):
    LAYERS[f"layer_{i}"] = (i, 0)
for layer_num in range(1, 301):
    gds_layer = LAYERS[f"layer_{layer_num}"]
    rect = gf.components.rectangle(size=(10, 10), layer=gds_layer, centered=True)
    c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert len(result.errors) == 0
        assert any("dynamically" in w for w in result.warnings)

    def test_dynamic_layers_via_range(self):
        """Code with range() and layer_num pattern should be detected as dynamic."""
        spec = GeometrySpec(
            component_type="layer_stack",
            description="test",
            layers=[
                LayerSpec(layer_number=1, datatype=0, name="metal1"),
                LayerSpec(layer_number=50, datatype=0, name="metal50"),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("layer_stack")
for layer_num in range(1, 51):
    rect = gf.components.rectangle(size=(10, 10), layer=(layer_num, 0), centered=True)
    c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert len(result.errors) == 0

    def test_no_dynamic_pattern_still_fails(self):
        """Code without dynamic patterns and missing all layers should still error."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
            layers=[
                LayerSpec(layer_number=10, datatype=0, name="custom1"),
                LayerSpec(layer_number=20, datatype=0, name="custom2"),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("via_array")
rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert any("ANY" in err for err in result.errors)


class TestSpecMatchWithRanges:
    """Tests for validate_spec_match with geometry_ranges."""

    def test_code_with_ranges_passes(self):
        """Code that handles ranges with dynamic patterns should pass."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            layers=[],
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=10,
                    width_start=100.0,
                    width_end=1000.0,
                    height_start=100.0,
                    height_end=1000.0,
                ),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("multi_layer_rectangles")
for i in range(1, 10 + 1):
    t = (i - 1) / max(1, 9)
    w = 100.0 + t * 900.0
    h = 100.0 + t * 900.0
    rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
    c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert len(result.errors) == 0

    def test_missing_dmove_for_offset_range_errors(self):
        """Code missing placement for non-zero offsets should error."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            layers=[],
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=5,
                    width_start=100.0,
                    width_end=500.0,
                    height_start=100.0,
                    height_end=500.0,
                    center_y=5000.0,
                ),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("multi_layer_rectangles")
for i in range(1, 6):
    rect = gf.components.rectangle(size=(100, 100), layer=(i, 0), centered=True)
    c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert any("placement operations" in err for err in result.errors)

    def test_multi_range_single_loop_errors(self):
        """Single-loop legacy code should fail multi-range coverage checks."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            layers=[],
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=100,
                    shape="square",
                    width_start=1000.0,
                    width_end=100000.0,
                    height_start=1000.0,
                    height_end=100000.0,
                ),
                GeometryRange(
                    start_layer=101,
                    end_layer=200,
                    shape="rectangle",
                    width_start=5000.0,
                    width_end=100000.0,
                    height_start=1000.0,
                    height_end=20000.0,
                ),
                GeometryRange(
                    start_layer=201,
                    end_layer=300,
                    shape="rectangle",
                    width_start=5000.0,
                    width_end=100000.0,
                    height_start=1000.0,
                    height_end=20000.0,
                    center_y=30000.0,
                ),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("multi_layer_rectangles")
num_layers = 300
base_width = 1000.0
base_height = 1000.0
width_step = 1000.0
height_step = 1000.0
for i in range(1, num_layers + 1):
    w = base_width + (i - 1) * width_step
    h = base_height + (i - 1) * height_step
    rect = gf.components.rectangle(size=(w, h), layer=(i, 0), centered=True)
    c.add_ref(rect)
"""
        result = validate_spec_match(spec, code)
        assert any("handle all geometry_ranges" in err for err in result.errors)
        assert any("201-300" in err and "placement operations" in err for err in result.errors)

    def test_data_driven_ranges_pass_without_literal_layer_bounds(self):
        """Data-driven range loops should satisfy range checks."""
        spec = GeometrySpec(
            component_type="multi_layer_rectangles",
            description="test",
            layers=[],
            geometry_ranges=[
                GeometryRange(
                    start_layer=1,
                    end_layer=5,
                    width_start=100.0,
                    width_end=500.0,
                    height_start=100.0,
                    height_end=500.0,
                    center_y=1000.0,
                ),
            ],
        )
        code = """
import gdsfactory as gf
c = gf.Component("multi_layer_rectangles")
geometry_ranges = [{
    "start_layer": 1,
    "end_layer": 5,
    "center_x": 0.0,
    "center_y": 1000.0,
}]
for r in geometry_ranges:
    for i in range(r["start_layer"], r["end_layer"] + 1):
        rect = gf.components.rectangle(size=(100, 100), layer=(i, 0), centered=True)
        ref = c.add_ref(rect)
        ref.dmove((r["center_x"], r["center_y"]))
"""
        result = validate_spec_match(spec, code)
        assert len(result.errors) == 0

    def test_empty_ranges_no_warnings(self):
        """Spec without geometry_ranges should not produce range-related warnings."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
            layers=[],
        )
        code = """
import gdsfactory as gf
c = gf.Component("via_array")
"""
        result = validate_spec_match(spec, code)
        assert not any("range" in w.lower() for w in result.warnings)
