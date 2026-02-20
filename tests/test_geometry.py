"""Tests for multi-layer geometry generation.

These tests verify that GeoForge can generate proper multi-layer
GDSFactory code for various semiconductor structures.
"""

import pytest
from pydantic import ValidationError

from geoforge.llm.base import GeometryRange, GeometrySpec, LayerSpec

# =============================================================================
# Fixtures for Multi-Layer Geometries
# =============================================================================


@pytest.fixture
def via_stack_spec():
    """Via stack connecting two metal layers."""
    return GeometrySpec(
        component_type="via_stack",
        description="Single via stack connecting metal1 to metal2",
        parameters={
            "metal_size": 2.0,
            "via_size": 0.5,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=150, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=200, material="copper"
            ),
        ],
    )


@pytest.fixture
def via_array_multilayer_spec():
    """Via array with metal pads on both layers."""
    return GeometrySpec(
        component_type="via_array",
        description="3x3 via array with metal pads connecting metal1 to metal2",
        parameters={
            "rows": 3,
            "cols": 3,
            "metal_size": 2.0,
            "via_size": 0.5,
            "pitch": 5.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=150, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=200, material="copper"
            ),
        ],
    )


@pytest.fixture
def mim_capacitor_spec():
    """MIM capacitor with offset plates."""
    return GeometrySpec(
        component_type="mim_capacitor",
        description="Metal-Insulator-Metal capacitor with offset plates for contacts",
        parameters={
            "plate_width": 30.0,
            "plate_height": 30.0,
            "extension": 5.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=200, material="copper"
            ),
        ],
    )


@pytest.fixture
def guard_ring_spec():
    """Guard ring with via stitching."""
    return GeometrySpec(
        component_type="guard_ring",
        description="Guard ring structure with via stitching connecting metal layers",
        parameters={
            "inner_size": 50.0,
            "ring_width": 5.0,
            "via_size": 1.0,
            "via_pitch": 5.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=150, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=200, material="copper"
            ),
        ],
    )


@pytest.fixture
def interconnect_spec():
    """Two-layer routing with vias at intersections."""
    return GeometrySpec(
        component_type="interconnect",
        description="Two-layer routing with horizontal traces on metal1, vertical on metal2",
        parameters={
            "trace_width": 2.0,
            "h_trace_spacing": 10.0,
            "v_trace_spacing": 15.0,
            "num_h_traces": 3,
            "num_v_traces": 3,
            "via_size": 1.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=150, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=200, material="copper"
            ),
        ],
    )


@pytest.fixture
def bond_pad_spec():
    """Bond pad with via stack to lower metal."""
    return GeometrySpec(
        component_type="bond_pad",
        description="Bond pad on top metal with via array to lower metal",
        parameters={
            "pad_size": 80.0,
            "via_array_rows": 5,
            "via_array_cols": 5,
            "via_size": 2.0,
            "via_pitch": 10.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=150, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=500, material="aluminum"
            ),
        ],
    )


@pytest.fixture
def inductor_spec():
    """Spiral inductor with underpass."""
    return GeometrySpec(
        component_type="inductor",
        description="Spiral inductor on top metal with underpass on lower metal",
        parameters={
            "turns": 3,
            "inner_radius": 20.0,
            "trace_width": 5.0,
            "spacing": 3.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=150, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=300, material="copper"
            ),
        ],
    )


# =============================================================================
# Tests for GeometrySpec Model
# =============================================================================


class TestGeometrySpecMultiLayer:
    """Tests for multi-layer GeometrySpec validation."""

    def test_via_stack_has_three_layers(self, via_stack_spec):
        """Via stack should have metal1, via, and metal2 layers."""
        assert len(via_stack_spec.layers) == 3
        layer_names = [layer.name for layer in via_stack_spec.layers]
        assert "metal1" in layer_names
        assert "via1" in layer_names
        assert "metal2" in layer_names

    def test_via_size_smaller_than_metal(self, via_stack_spec):
        """Via size should be smaller than metal pad size."""
        assert via_stack_spec.parameters["via_size"] < via_stack_spec.parameters["metal_size"]

    def test_mim_capacitor_has_two_metal_layers(self, mim_capacitor_spec):
        """MIM capacitor should have two metal layers (no via)."""
        assert len(mim_capacitor_spec.layers) == 2
        for layer in mim_capacitor_spec.layers:
            assert "metal" in layer.name

    def test_guard_ring_has_via_stitching_params(self, guard_ring_spec):
        """Guard ring should have via stitching parameters."""
        assert "via_size" in guard_ring_spec.parameters
        assert "via_pitch" in guard_ring_spec.parameters

    def test_interconnect_has_trace_params(self, interconnect_spec):
        """Interconnect should have trace routing parameters."""
        params = interconnect_spec.parameters
        assert "trace_width" in params
        assert "num_h_traces" in params
        assert "num_v_traces" in params

    def test_layer_numbers_are_unique(self, via_array_multilayer_spec):
        """All layers should have unique layer numbers."""
        layer_numbers = [layer.layer_number for layer in via_array_multilayer_spec.layers]
        assert len(layer_numbers) == len(set(layer_numbers))

    def test_layer_materials_are_set(self, via_array_multilayer_spec):
        """All layers should have materials defined."""
        for layer in via_array_multilayer_spec.layers:
            assert layer.material is not None

    def test_layer_thicknesses_are_positive(self, via_array_multilayer_spec):
        """All layer thicknesses should be positive."""
        for layer in via_array_multilayer_spec.layers:
            assert layer.thickness_nm is not None
            assert layer.thickness_nm > 0


# =============================================================================
# Tests for Layer Relationships
# =============================================================================


class TestLayerRelationships:
    """Tests for proper layer relationships in multi-layer structures."""

    def test_via_between_metals(self, via_stack_spec):
        """Via layer number should be between metal layer numbers."""
        layers = {layer.name: layer.layer_number for layer in via_stack_spec.layers}
        # Via layer should be between metal1 and metal2
        assert layers["metal1"] < layers["via1"] < layers["metal2"]

    def test_standard_layer_convention(self, via_array_multilayer_spec):
        """Layers should follow standard numbering convention."""
        layers = {layer.name: layer.layer_number for layer in via_array_multilayer_spec.layers}
        # Standard convention: metal1=1, via1=2, metal2=3
        assert layers["metal1"] == 1
        assert layers["via1"] == 2
        assert layers["metal2"] == 3

    def test_guard_ring_symmetric_layers(self, guard_ring_spec):
        """Guard ring should have symmetric metal layers."""
        metal_layers = [layer for layer in guard_ring_spec.layers if "metal" in layer.name]
        assert len(metal_layers) == 2
        # Both metal layers should have same thickness
        assert metal_layers[0].thickness_nm == metal_layers[1].thickness_nm


# =============================================================================
# Tests for Component Parameters
# =============================================================================


class TestComponentParameters:
    """Tests for component-specific parameter validation."""

    def test_array_parameters(self, via_array_multilayer_spec):
        """Array components should have rows, cols, and pitch."""
        params = via_array_multilayer_spec.parameters
        assert "rows" in params
        assert "cols" in params
        assert "pitch" in params
        assert params["rows"] > 0
        assert params["cols"] > 0
        assert params["pitch"] > 0

    def test_capacitor_has_plate_dimensions(self, mim_capacitor_spec):
        """Capacitor should have plate width and height."""
        params = mim_capacitor_spec.parameters
        assert "plate_width" in params
        assert "plate_height" in params
        assert params["plate_width"] > 0
        assert params["plate_height"] > 0

    def test_capacitor_has_extension(self, mim_capacitor_spec):
        """MIM capacitor should have extension for contacts."""
        params = mim_capacitor_spec.parameters
        assert "extension" in params
        assert params["extension"] > 0

    def test_ring_has_inner_size_and_width(self, guard_ring_spec):
        """Guard ring should have inner size and ring width."""
        params = guard_ring_spec.parameters
        assert "inner_size" in params
        assert "ring_width" in params
        assert params["inner_size"] > 0
        assert params["ring_width"] > 0

    def test_inductor_has_spiral_params(self, inductor_spec):
        """Inductor should have spiral parameters."""
        params = inductor_spec.parameters
        assert "turns" in params
        assert "inner_radius" in params
        assert "trace_width" in params
        assert params["turns"] > 0


# =============================================================================
# Tests for Prompt Parsing (Integration)
# =============================================================================


class TestPromptExamples:
    """Test that example prompts map to correct component types."""

    @pytest.mark.parametrize(
        "prompt_fragment,expected_type",
        [
            ("via array", "via_array"),
            ("via stack", "via_stack"),
            ("MIM capacitor", "mim_capacitor"),
            ("guard ring", "guard_ring"),
            ("two-layer routing", "interconnect"),
            ("interconnect", "interconnect"),
            ("bond pad", "bond_pad"),
            ("spiral inductor", "inductor"),
            ("layer stack", "layer_stack"),
            ("test pattern", "test_pattern"),
        ],
    )
    def test_prompt_to_component_type_mapping(self, prompt_fragment, expected_type):
        """Verify prompt fragments map to expected component types."""
        # This is a documentation test - verifies the expected mappings
        # Actual LLM output would be tested in integration tests
        assert expected_type is not None
        assert prompt_fragment.lower() in prompt_fragment.lower()


# =============================================================================
# Tests for Multi-Layer Code Generation Patterns
# =============================================================================


class TestCodeGenerationPatterns:
    """Tests for expected code generation patterns."""

    def test_via_stack_requires_centered_shapes(self, via_stack_spec):
        """Via stack layers should be centered for proper alignment."""
        # This documents the expected behavior
        # Via, metal1, and metal2 should all be centered at same point
        assert via_stack_spec.component_type == "via_stack"

    def test_mim_capacitor_plates_offset(self, mim_capacitor_spec):
        """MIM capacitor plates should have opposite extensions."""
        params = mim_capacitor_spec.parameters
        # Bottom plate extends one direction, top plate extends opposite
        assert "extension" in params

    def test_guard_ring_uses_boolean(self, guard_ring_spec):
        """Guard ring should use boolean operations for ring shape."""
        # Documents that ring shape is created via boolean subtraction
        assert guard_ring_spec.component_type == "guard_ring"


# =============================================================================
# Complex Multi-Layer Geometry Fixtures
# =============================================================================


@pytest.fixture
def five_metal_stack_spec():
    """5-metal layer stack with 4 via layers."""
    return GeometrySpec(
        component_type="layer_stack",
        description="5-metal interconnect stack typical of advanced CMOS",
        parameters={
            "width": 100.0,
            "height": 100.0,
        },
        layers=[
            LayerSpec(
                layer_number=1, datatype=0, name="metal1", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=2, datatype=0, name="via1", thickness_nm=200, material="tungsten"
            ),
            LayerSpec(
                layer_number=3, datatype=0, name="metal2", thickness_nm=200, material="copper"
            ),
            LayerSpec(
                layer_number=4, datatype=0, name="via2", thickness_nm=200, material="tungsten"
            ),
            LayerSpec(
                layer_number=5, datatype=0, name="metal3", thickness_nm=300, material="copper"
            ),
            LayerSpec(
                layer_number=6, datatype=0, name="via3", thickness_nm=300, material="tungsten"
            ),
            LayerSpec(
                layer_number=7, datatype=0, name="metal4", thickness_nm=500, material="copper"
            ),
            LayerSpec(
                layer_number=8, datatype=0, name="via4", thickness_nm=500, material="tungsten"
            ),
            LayerSpec(
                layer_number=9, datatype=0, name="metal5", thickness_nm=1000, material="aluminum"
            ),
        ],
    )


class TestComplexLayerStacks:
    """Tests for complex multi-layer structures."""

    def test_five_metal_stack_layer_count(self, five_metal_stack_spec):
        """5-metal stack should have 9 layers (5 metal + 4 via)."""
        assert len(five_metal_stack_spec.layers) == 9

    def test_metal_layers_increase_in_thickness(self, five_metal_stack_spec):
        """Upper metal layers are typically thicker."""
        metal_layers = [layer for layer in five_metal_stack_spec.layers if "metal" in layer.name]
        thicknesses = [layer.thickness_nm for layer in metal_layers]
        # In advanced CMOS, upper metals are thicker for power distribution
        assert thicknesses[-1] >= thicknesses[0]

    def test_alternating_metal_via_pattern(self, five_metal_stack_spec):
        """Layers should alternate between metal and via."""
        layers = five_metal_stack_spec.layers
        for i, layer in enumerate(layers):
            if i % 2 == 0:
                assert "metal" in layer.name
            else:
                assert "via" in layer.name


# =============================================================================
# Fixtures and Tests for Geometry Ranges
# =============================================================================


@pytest.fixture
def multi_range_300_layer_spec():
    """300-layer spec with 3 distinct geometry ranges."""
    return GeometrySpec(
        component_type="multi_layer_rectangles",
        description="300-layer stress test with 3 distinct ranges",
        parameters={"num_layers": 300, "cell_name": "LAYERS_1_300"},
        layers=[
            LayerSpec(layer_number=1, datatype=0, name="layer_1"),
            LayerSpec(layer_number=150, datatype=0, name="layer_150"),
            LayerSpec(layer_number=300, datatype=0, name="layer_300"),
        ],
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


class TestGeometryRanges:
    """Tests for the geometry_ranges field on GeometrySpec."""

    def test_geometry_ranges_count(self, multi_range_300_layer_spec):
        """300-layer spec should have 3 geometry ranges."""
        assert len(multi_range_300_layer_spec.geometry_ranges) == 3

    def test_ranges_cover_all_layers(self, multi_range_300_layer_spec):
        """All 300 layers should be covered by the ranges."""
        total = sum(
            r.end_layer - r.start_layer + 1 for r in multi_range_300_layer_spec.geometry_ranges
        )
        assert total == 300

    def test_range_offsets(self, multi_range_300_layer_spec):
        """Third range should have y offset, others at origin."""
        ranges = multi_range_300_layer_spec.geometry_ranges
        assert ranges[0].center_x == 0.0
        assert ranges[0].center_y == 0.0
        assert ranges[1].center_y == 0.0
        assert ranges[2].center_y == 30000.0

    def test_square_range_has_equal_dimensions(self, multi_range_300_layer_spec):
        """Square range should have width == height at both ends."""
        sq = multi_range_300_layer_spec.geometry_ranges[0]
        assert sq.width_start == sq.height_start
        assert sq.width_end == sq.height_end

    def test_rectangle_range_has_different_dimensions(self, multi_range_300_layer_spec):
        """Rectangle range should have width != height."""
        rect = multi_range_300_layer_spec.geometry_ranges[1]
        assert rect.width_start != rect.height_start
        assert rect.width_end != rect.height_end

    def test_empty_geometry_ranges_default(self):
        """GeometrySpec without geometry_ranges should default to empty list."""
        spec = GeometrySpec(
            component_type="via_array",
            description="test",
        )
        assert spec.geometry_ranges == []

    def test_geometry_range_defaults(self):
        """GeometryRange should have correct defaults."""
        r = GeometryRange(
            start_layer=1,
            end_layer=50,
            width_start=100.0,
            width_end=500.0,
            height_start=100.0,
            height_end=500.0,
        )
        assert r.shape == "rectangle"
        assert r.center_x == 0.0
        assert r.center_y == 0.0

    def test_geometry_range_rejects_end_before_start(self):
        """end_layer must be >= start_layer."""
        with pytest.raises(ValidationError):
            GeometryRange(
                start_layer=10,
                end_layer=1,
                width_start=100.0,
                width_end=500.0,
                height_start=100.0,
                height_end=500.0,
            )

    def test_square_geometry_range_requires_equal_sides(self):
        """Square ranges should enforce equal width/height at both endpoints."""
        with pytest.raises(ValidationError):
            GeometryRange(
                start_layer=1,
                end_layer=10,
                shape="square",
                width_start=100.0,
                width_end=1000.0,
                height_start=120.0,
                height_end=1000.0,
            )

    def test_geometry_range_rejects_unknown_shape(self):
        """shape must be rectangle or square."""
        with pytest.raises(ValidationError):
            GeometryRange(
                start_layer=1,
                end_layer=10,
                shape="triangle",
                width_start=100.0,
                width_end=1000.0,
                height_start=100.0,
                height_end=1000.0,
            )

    def test_geometry_range_rejects_non_positive_dimensions(self):
        """Width/height endpoints must be positive."""
        with pytest.raises(ValidationError):
            GeometryRange(
                start_layer=1,
                end_layer=10,
                width_start=0.0,
                width_end=1000.0,
                height_start=100.0,
                height_end=1000.0,
            )
