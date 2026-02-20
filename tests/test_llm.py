"""Tests for LLM providers."""

from types import SimpleNamespace

import pytest

from geoforge.llm import list_providers
from geoforge.llm.base import GeometryRange, GeometrySpec, LayerSpec


def test_list_providers():
    """Test that providers are registered."""
    providers = list_providers()
    assert "gemini" in providers
    assert "ollama" in providers


def test_geometry_spec_model():
    """Test GeometrySpec Pydantic model."""
    spec = GeometrySpec(
        component_type="via_array",
        description="Test via array",
        parameters={"rows": 5, "cols": 5},
        layers=[
            LayerSpec(layer_number=1, datatype=0, name="via1"),
        ],
    )
    assert spec.component_type == "via_array"
    assert spec.parameters["rows"] == 5
    assert len(spec.layers) == 1


def test_layer_spec_defaults():
    """Test LayerSpec default values."""
    layer = LayerSpec(layer_number=1, name="metal1")
    assert layer.datatype == 0
    assert layer.thickness_nm is None
    assert layer.material is None


def _spec_with_geometry_ranges() -> GeometrySpec:
    return GeometrySpec(
        component_type="multi_layer_rectangles",
        description="test",
        parameters={"num_layers": 10},
        layers=[LayerSpec(layer_number=1, datatype=0, name="layer_1")],
        geometry_ranges=[
            GeometryRange(
                start_layer=1,
                end_layer=10,
                shape="square",
                width_start=100.0,
                width_end=1000.0,
                height_start=100.0,
                height_end=1000.0,
            )
        ],
    )


class _FakeOpenAICompletions:
    @staticmethod
    async def create(**kwargs):
        del kwargs
        msg = SimpleNamespace(content="import gdsfactory as gf\nc = gf.Component('x')")
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions())


class _FakeOllamaClient:
    @staticmethod
    async def chat(**kwargs):
        del kwargs
        return {"message": {"content": "import gdsfactory as gf\nc = gf.Component('x')"}}


class _FakeGeminiModels:
    @staticmethod
    async def generate_content(**kwargs):
        del kwargs
        return SimpleNamespace(text="import gdsfactory as gf\nc = gf.Component('x')")


class _FakeGeminiClient:
    def __init__(self):
        self.aio = SimpleNamespace(models=_FakeGeminiModels())


class _FakeAnthropicMessages:
    @staticmethod
    async def create(**kwargs):
        del kwargs
        return SimpleNamespace(content=[SimpleNamespace(text="import gdsfactory as gf")])


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeAnthropicMessages()


@pytest.mark.asyncio
async def test_openai_provider_passes_geometry_ranges_to_prompt(monkeypatch):
    from geoforge.llm import openai as openai_module

    captured: dict = {}

    def _fake_build_code_prompt(spec_dict, original_prompt=None):
        captured["spec_dict"] = spec_dict
        captured["original_prompt"] = original_prompt
        return "prompt"

    monkeypatch.setattr(openai_module, "build_code_prompt", _fake_build_code_prompt)

    provider = openai_module.OpenAIProvider.__new__(openai_module.OpenAIProvider)
    provider.model = "test-model"
    provider.client = _FakeOpenAIClient()

    code = await provider._generate_gdsfactory_code_impl(_spec_with_geometry_ranges(), "original")
    assert "gdsfactory" in code
    assert "geometry_ranges" in captured["spec_dict"]
    assert len(captured["spec_dict"]["geometry_ranges"]) == 1


@pytest.mark.asyncio
async def test_ollama_provider_passes_geometry_ranges_to_prompt(monkeypatch):
    from geoforge.llm import ollama as ollama_module

    captured: dict = {}

    def _fake_build_code_prompt(spec_dict, original_prompt=None):
        captured["spec_dict"] = spec_dict
        captured["original_prompt"] = original_prompt
        return "prompt"

    monkeypatch.setattr(ollama_module, "build_code_prompt", _fake_build_code_prompt)

    provider = ollama_module.OllamaProvider.__new__(ollama_module.OllamaProvider)
    provider.model = "test-model"
    provider.client = _FakeOllamaClient()

    code = await provider._generate_gdsfactory_code_impl(_spec_with_geometry_ranges(), "original")
    assert "gdsfactory" in code
    assert "geometry_ranges" in captured["spec_dict"]
    assert len(captured["spec_dict"]["geometry_ranges"]) == 1


@pytest.mark.asyncio
async def test_gemini_provider_passes_geometry_ranges_to_prompt(monkeypatch):
    from geoforge.llm import google as google_module

    captured: dict = {}

    def _fake_build_code_prompt(spec_dict, original_prompt=None):
        captured["spec_dict"] = spec_dict
        captured["original_prompt"] = original_prompt
        return "prompt"

    monkeypatch.setattr(google_module, "build_code_prompt", _fake_build_code_prompt)

    provider = google_module.GeminiProvider.__new__(google_module.GeminiProvider)
    provider.model = "test-model"
    provider.client = _FakeGeminiClient()

    code = await provider._generate_gdsfactory_code_impl(_spec_with_geometry_ranges(), "original")
    assert "gdsfactory" in code
    assert "geometry_ranges" in captured["spec_dict"]
    assert len(captured["spec_dict"]["geometry_ranges"]) == 1


@pytest.mark.asyncio
async def test_anthropic_provider_passes_geometry_ranges_to_prompt(monkeypatch):
    from geoforge.llm import anthropic as anthropic_module

    captured: dict = {}

    def _fake_build_code_prompt(spec_dict, original_prompt=None):
        captured["spec_dict"] = spec_dict
        captured["original_prompt"] = original_prompt
        return "prompt"

    monkeypatch.setattr(anthropic_module, "build_code_prompt", _fake_build_code_prompt)

    provider = anthropic_module.AnthropicProvider.__new__(anthropic_module.AnthropicProvider)
    provider.model = "test-model"
    provider.client = _FakeAnthropicClient()

    code = await provider._generate_gdsfactory_code_impl(_spec_with_geometry_ranges(), "original")
    assert "gdsfactory" in code
    assert "geometry_ranges" in captured["spec_dict"]
    assert len(captured["spec_dict"]["geometry_ranges"]) == 1
