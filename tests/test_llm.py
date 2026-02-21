"""Tests for LLM providers."""

from types import SimpleNamespace

import pytest

from geoforge.llm import list_providers
from geoforge.llm.base import GeometryRange, GeometrySpec, LayerSpec, PrimitiveSpec, RetryContext


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


def test_primitive_spec_polygon_requires_points():
    """Polygon primitives should enforce at least three points."""
    with pytest.raises(ValueError, match="at least 3 points"):
        PrimitiveSpec(
            primitive_type="polygon",
            layer_number=1,
            points=[(0.0, 0.0), (1.0, 1.0)],
        )


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
        primitives=[
            PrimitiveSpec(
                primitive_type="polygon",
                layer_number=1,
                points=[(-1.0, -1.0), (0.0, 2.0), (1.0, -1.0)],
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


class _QueuedOpenAICompletions:
    def __init__(self, responses: list[str | None]):
        self.responses = responses
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        msg = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


class _QueuedOpenAIClient:
    def __init__(self, responses: list[str | None]):
        self.completions = _QueuedOpenAICompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class _QueuedGeminiModels:
    def __init__(self, responses: list[str | None]):
        self.responses = responses
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.responses.pop(0))


class _QueuedGeminiClient:
    def __init__(self, responses: list[str | None]):
        self.models = _QueuedGeminiModels(responses)
        self.aio = SimpleNamespace(models=self.models)


class _QueuedAnthropicMessages:
    def __init__(self, responses: list[str | None]):
        self.responses = responses
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        blocks = [] if content is None else [SimpleNamespace(text=content)]
        return SimpleNamespace(content=blocks)


class _QueuedAnthropicClient:
    def __init__(self, responses: list[str | None]):
        self.messages = _QueuedAnthropicMessages(responses)


class _QueuedOllamaClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.responses.pop(0)}}


class _FakeOllamaSyncClient:
    def __init__(self, models: list[str] | None = None, raises: Exception | None = None):
        self._models = models or []
        self._raises = raises

    def list(self):
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(models=[SimpleNamespace(model=name) for name in self._models])

    def pull(self, model: str, stream: bool = True):
        del stream
        return [SimpleNamespace(status=f"downloading {model}")]


def test_cloud_provider_init_requires_api_key(monkeypatch):
    from geoforge.llm import anthropic as anthropic_module
    from geoforge.llm import google as google_module
    from geoforge.llm import openai as openai_module

    monkeypatch.setattr(openai_module.settings, "openai_api_key", None)
    monkeypatch.setattr(google_module.settings, "gemini_api_key", None)
    monkeypatch.setattr(anthropic_module.settings, "anthropic_api_key", None)

    with pytest.raises(ValueError, match="OpenAI API key"):
        openai_module.OpenAIProvider(model="test-model")
    with pytest.raises(ValueError, match="Gemini API key"):
        google_module.GeminiProvider(model="test-model")
    with pytest.raises(ValueError, match="Anthropic API key"):
        anthropic_module.AnthropicProvider(model="test-model")


@pytest.mark.asyncio
async def test_openai_enhancement_and_spec_branches():
    from geoforge.llm import openai as openai_module

    provider = openai_module.OpenAIProvider.__new__(openai_module.OpenAIProvider)
    provider.model = "test-model"
    provider.client = _QueuedOpenAIClient(
        [
            '{"rewritten_prompt":"Centered single-layer logo","key_constraints":["single layer"]}',
            '{"component_type":"test_pattern","description":"x","parameters":{},"layers":[]}',
        ]
    )

    retry = RetryContext(
        attempt_number=1,
        previous_error="bad json",
        previous_response_snippet='{"oops": 1}',
        error_category="json_parse",
    )

    enhanced = await provider._enhance_prompt_impl("draw logo", retry)
    spec = await provider._generate_geometry_spec_impl("draw logo", retry)

    assert enhanced["rewritten_prompt"].startswith("Centered")
    assert spec["component_type"] == "test_pattern"
    assert provider.client.completions.calls[0]["messages"][2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_openai_enhancement_empty_response_raises():
    from geoforge.llm import openai as openai_module

    provider = openai_module.OpenAIProvider.__new__(openai_module.OpenAIProvider)
    provider.model = "test-model"
    provider.client = _QueuedOpenAIClient([None])

    with pytest.raises(ValueError, match="prompt enhancement"):
        await provider._enhance_prompt_impl("draw logo")


@pytest.mark.asyncio
async def test_gemini_enhancement_and_spec_branches():
    from geoforge.llm import google as google_module

    provider = google_module.GeminiProvider.__new__(google_module.GeminiProvider)
    provider.model = "test-model"
    provider.client = _QueuedGeminiClient(
        [
            '{"rewritten_prompt":"Use polygons and circles","key_constraints":["centered"]}',
            '{"component_type":"test_pattern","description":"y","parameters":{},"layers":[]}',
        ]
    )
    retry = RetryContext(attempt_number=1, previous_error="x", error_category="schema_validation")

    enhanced = await provider._enhance_prompt_impl("draw", retry)
    spec = await provider._generate_geometry_spec_impl("draw", retry)

    assert enhanced["key_constraints"] == ["centered"]
    assert spec["description"] == "y"
    assert "Error type" in provider.client.models.calls[0]["contents"]


@pytest.mark.asyncio
async def test_gemini_empty_response_raises():
    from geoforge.llm import google as google_module

    provider = google_module.GeminiProvider.__new__(google_module.GeminiProvider)
    provider.model = "test-model"
    provider.client = _QueuedGeminiClient([None])

    with pytest.raises(ValueError, match="empty response for prompt enhancement"):
        await provider._enhance_prompt_impl("draw")


@pytest.mark.asyncio
async def test_anthropic_enhancement_and_spec_branches():
    from geoforge.llm import anthropic as anthropic_module

    provider = anthropic_module.AnthropicProvider.__new__(anthropic_module.AnthropicProvider)
    provider.model = "test-model"
    provider.client = _QueuedAnthropicClient(
        [
            '{"rewritten_prompt":"Precise constraints","key_constraints":["200um"]}',
            '{"component_type":"test_pattern","description":"z","parameters":{},"layers":[]}',
        ]
    )
    retry = RetryContext(
        attempt_number=1,
        previous_error="invalid",
        previous_response_snippet='{"bad": true}',
        error_category="json_parse",
    )

    enhanced = await provider._enhance_prompt_impl("draw", retry)
    spec = await provider._generate_geometry_spec_impl("draw", retry)

    assert enhanced["key_constraints"] == ["200um"]
    assert spec["component_type"] == "test_pattern"
    assert provider.client.messages.calls[0]["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_anthropic_empty_response_raises():
    from geoforge.llm import anthropic as anthropic_module

    provider = anthropic_module.AnthropicProvider.__new__(anthropic_module.AnthropicProvider)
    provider.model = "test-model"
    provider.client = _QueuedAnthropicClient([None])

    with pytest.raises(ValueError, match="prompt enhancement"):
        await provider._enhance_prompt_impl("draw")


def test_ollama_ensure_model_available_and_connection_errors(monkeypatch):
    from geoforge.llm import ollama as ollama_module

    provider = ollama_module.OllamaProvider.__new__(ollama_module.OllamaProvider)
    provider.model = "wanted-model"
    provider.host = "http://localhost:11434"
    provider._sync_client = _FakeOllamaSyncClient(models=["other-model"])
    called = {"pull": False}
    provider._pull_model = lambda: called.__setitem__("pull", True)

    provider._ensure_model_available()
    assert called["pull"] is True

    provider._sync_client = _FakeOllamaSyncClient(raises=RuntimeError("connection refused"))
    with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
        provider._ensure_model_available()


def test_ollama_is_available_and_list_models(monkeypatch):
    from geoforge.llm import ollama as ollama_module

    class HealthyClient:
        def __init__(self, host=None):
            del host

        def list(self):
            return SimpleNamespace(models=[SimpleNamespace(model="model-a")])

    class BrokenClient:
        def __init__(self, host=None):
            del host
            raise RuntimeError("offline")

    monkeypatch.setattr(ollama_module.ollama_client, "Client", HealthyClient)
    assert ollama_module.OllamaProvider.is_available() is True
    assert ollama_module.OllamaProvider.list_local_models() == ["model-a"]

    monkeypatch.setattr(ollama_module.ollama_client, "Client", BrokenClient)
    assert ollama_module.OllamaProvider.is_available() is False
    assert ollama_module.OllamaProvider.list_local_models() == []


@pytest.mark.asyncio
async def test_ollama_enhancement_and_spec_branches():
    from geoforge.llm import ollama as ollama_module

    provider = ollama_module.OllamaProvider.__new__(ollama_module.OllamaProvider)
    provider.model = "test-model"
    provider.client = _QueuedOllamaClient(
        [
            '{"rewritten_prompt":"explicit constraints","key_constraints":["single layer"]}',
            '{"component_type":"test_pattern","description":"o","parameters":{},"layers":[]}',
        ]
    )

    retry = RetryContext(
        attempt_number=2,
        previous_error="bad",
        previous_response_snippet='{"old": 1}',
        error_category="schema_validation",
    )

    enhanced = await provider._enhance_prompt_impl("draw", retry)
    spec = await provider._generate_geometry_spec_impl("draw", retry)

    assert enhanced["rewritten_prompt"] == "explicit constraints"
    assert spec["description"] == "o"
    assert provider.client.calls[0]["format"] == "json"


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
    assert "primitives" in captured["spec_dict"]
    assert len(captured["spec_dict"]["primitives"]) == 1


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
    assert "primitives" in captured["spec_dict"]
    assert len(captured["spec_dict"]["primitives"]) == 1


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
    assert "primitives" in captured["spec_dict"]
    assert len(captured["spec_dict"]["primitives"]) == 1


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
    assert "primitives" in captured["spec_dict"]
    assert len(captured["spec_dict"]["primitives"]) == 1
