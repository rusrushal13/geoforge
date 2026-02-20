"""Mock LLM provider for deterministic testing."""

from __future__ import annotations

from geoforge.llm.base import GeometrySpec, LLMProvider, RetryContext

# A simple known-good code snippet for testing
MOCK_GDSFACTORY_CODE = """
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component("mock_via_array")

# Metal 1 layer
m1 = gf.components.rectangle(size=(10, 10), layer=(1, 0))
c.add_ref(m1)

# Via layer
via = gf.components.rectangle(size=(2, 2), layer=(2, 0))
ref = c.add_ref(via)
ref.dmove((4, 4))

# Metal 2 layer
m2 = gf.components.rectangle(size=(10, 10), layer=(3, 0))
c.add_ref(m2)

c.write_gds("mock_via_array.gds")
c.write("mock_via_array.oas")
"""

MOCK_SPEC_DICT = {
    "component_type": "via_array",
    "description": "3x3 array of vias for metal layer connections",
    "parameters": {
        "rows": 3,
        "cols": 3,
        "via_size_um": 0.5,
        "pitch_um": 2.0,
    },
    "layers": [
        {"layer_number": 1, "datatype": 0, "name": "metal1"},
        {"layer_number": 2, "datatype": 0, "name": "via1"},
        {"layer_number": 3, "datatype": 0, "name": "metal2"},
    ],
}


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns predetermined responses."""

    name = "mock"

    def __init__(
        self,
        spec_response: dict | None = None,
        code_response: str | None = None,
        fail_spec: bool = False,
        fail_code: bool = False,
        fail_count: int = 0,
    ):
        self.spec_response = spec_response or MOCK_SPEC_DICT
        self.code_response = code_response or MOCK_GDSFACTORY_CODE
        self.fail_spec = fail_spec
        self.fail_code = fail_code
        self.fail_count = fail_count  # Fail this many times before succeeding
        self._spec_calls = 0
        self._code_calls = 0

    async def _generate_geometry_spec_impl(
        self,
        prompt: str,
        retry_context: RetryContext | None = None,
    ) -> dict:
        self._spec_calls += 1
        if self.fail_spec:
            raise ValueError("Mock spec generation failure")
        if self._spec_calls <= self.fail_count:
            raise ValueError(f"Mock failure on attempt {self._spec_calls}")
        return self.spec_response

    async def _generate_gdsfactory_code_impl(
        self,
        spec: GeometrySpec,
        original_prompt: str | None = None,
        retry_context: RetryContext | None = None,
    ) -> str:
        self._code_calls += 1
        if self.fail_code:
            raise ValueError("Mock code generation failure")
        if self._code_calls <= self.fail_count:
            raise ValueError(f"Mock failure on attempt {self._code_calls}")
        return self.code_response
