"""Tests for prompt template routing and example selection."""

from geoforge.prompts.gdsfactory import build_code_prompt


def test_test_pattern_routes_to_comb_serpentine_example():
    """Comb/serpentine wording should select the comb_serpentine example."""
    prompt = build_code_prompt(
        {
            "component_type": "test_pattern",
            "description": "comb serpentine fanout with bottom rail and fingers",
            "parameters": {"trace_width": 6.0},
            "layers": [{"layer_number": 1, "datatype": 0, "name": "metal1"}],
        },
        original_prompt=(
            "Create a comb serpentine with mirrored fingers, fanout, and a bottom rail"
        ),
    )
    assert "def comb_serpentine(" in prompt


def test_test_pattern_routes_to_curved_bundle_example():
    """Curved bundle wording should select the curved_trace_bundle example."""
    prompt = build_code_prompt(
        {
            "component_type": "test_pattern",
            "description": "curved trace bundle with concentric arcs",
            "parameters": {"num_traces": 10},
            "layers": [{"layer_number": 1, "datatype": 0, "name": "metal1"}],
        },
        original_prompt="Create a curved parallel bundle with arc bends and euler smoothing",
    )
    assert "def curved_trace_bundle(" in prompt


def test_test_pattern_defaults_to_generic_example_without_keywords():
    """Generic test pattern wording should continue to use the default example."""
    prompt = build_code_prompt(
        {
            "component_type": "test_pattern",
            "description": "simple alignment test pattern",
            "parameters": {"size": 10.0, "spacing": 20.0},
            "layers": [{"layer_number": 1, "datatype": 0, "name": "metal1"}],
        },
        original_prompt="Create a simple test pattern for alignment checks",
    )
    assert "def test_pattern(" in prompt


def test_build_code_prompt_includes_primitive_list():
    """Primitive geometry should be injected into the code-generation prompt."""
    prompt = build_code_prompt(
        {
            "component_type": "test_pattern",
            "description": "custom mascot shape",
            "parameters": {"size": 200.0},
            "layers": [{"layer_number": 1, "datatype": 0, "name": "metal1"}],
            "primitives": [
                {
                    "primitive_type": "polygon",
                    "layer_number": 1,
                    "datatype": 0,
                    "points": [[-10, -10], [0, 20], [10, -10]],
                }
            ],
        },
        original_prompt="Create a mascot silhouette from primitive polygons",
    )
    assert "Primitive List:" in prompt
    assert '"primitive_type": "polygon"' in prompt
