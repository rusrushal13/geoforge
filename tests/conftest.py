"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_prompt():
    """Sample geometry prompt for testing."""
    return "Create a 3x3 via array with 0.5um via size and 2um pitch"


@pytest.fixture
def sample_spec():
    """Sample geometry specification for testing."""
    from geoforge.llm.base import GeometrySpec, LayerSpec

    return GeometrySpec(
        component_type="via_array",
        description="3x3 array of vias for metal layer connections",
        parameters={
            "rows": 3,
            "cols": 3,
            "via_size_um": 0.5,
            "pitch_um": 2.0,
        },
        layers=[
            LayerSpec(layer_number=1, datatype=0, name="via1", thickness_nm=500, material="copper"),
        ],
    )
