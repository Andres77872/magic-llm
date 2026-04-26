"""Tests for embedding response transformation.

Classical TDD — feed known JSON dicts, assert ModelEmbeddingResponse structure.
No network calls, no HTTP mocks — pure input→output verification.

Covers:
- OpenAiBaseProvider.transform_embedding_response: standard response parsing
- ModelEmbeddingResponse: Pydantic model validation
- Edge cases: missing usage, multiple embeddings, empty embeddings
"""

import pytest

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelEmbeddingResponse import ModelEmbeddingResponse, EmbeddingData
from magic_llm.model.ModelChatStream import UsageModel


class TestEmbeddingResponseTransform:
    """OpenAiBaseProvider.transform_embedding_response — parses raw embedding API dicts."""

    def setup_method(self):
        self.provider = OpenAiBaseProvider(
            base_url="https://api.test.com",
            api_key="test",
            model="text-embedding-3-small",
        )

    def test_parses_standard_openai_embedding_response(self):
        """Standard OpenAI embedding response with usage."""
        raw = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.0023064255, -0.009327292, 0.014567891],
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

        result = self.provider.transform_embedding_response(raw)

        assert isinstance(result, ModelEmbeddingResponse)
        assert result.object == "list"
        assert result.model == "text-embedding-3-small"
        assert len(result.data) == 1
        assert result.data[0].index == 0
        assert result.data[0].embedding == [0.0023064255, -0.009327292, 0.014567891]
        assert result.usage is not None
        assert result.usage.prompt_tokens == 8
        assert result.usage.total_tokens == 8

    def test_handles_multiple_embeddings(self):
        """Response with multiple embedding vectors (batch input)."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"object": "embedding", "index": 2, "embedding": [0.7, 0.8, 0.9]},
            ],
            "model": "text-embedding-3-small",
        }

        result = self.provider.transform_embedding_response(raw)

        assert len(result.data) == 3
        assert result.data[0].index == 0
        assert result.data[1].index == 1
        assert result.data[2].index == 2
        assert result.data[2].embedding == [0.7, 0.8, 0.9]

    def test_handles_missing_usage(self):
        """Some providers don't return usage info — should still parse."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.5, -0.3]},
            ],
            "model": "bge-m3",
        }

        result = self.provider.transform_embedding_response(raw)

        assert isinstance(result, ModelEmbeddingResponse)
        assert result.model == "bge-m3"
        assert result.usage is None

    def test_high_dimensional_embedding(self):
        """Realistic embedding dimension (e.g., text-embedding-3-small = 1536 dims)."""
        embedding = [0.001 * i for i in range(1536)]
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": embedding},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }

        result = self.provider.transform_embedding_response(raw)

        assert len(result.data[0].embedding) == 1536
        assert result.data[0].embedding[0] == 0.0
        assert result.data[0].embedding[1535] == pytest.approx(1.535, rel=1e-9)

    def test_embedding_data_is_typed(self):
        """Each embedding entry is a proper EmbeddingData instance."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ],
            "model": "test-model",
        }

        result = self.provider.transform_embedding_response(raw)

        assert isinstance(result.data[0], EmbeddingData)
        assert result.data[0].object == "embedding"

    def test_usage_is_typed(self):
        """Usage field is a proper UsageModel instance when present."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1]},
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100},
        }

        result = self.provider.transform_embedding_response(raw)

        assert isinstance(result.usage, UsageModel)
        assert result.usage.prompt_tokens == 100


class TestEmbeddingResponseModelValidation:
    """ModelEmbeddingResponse Pydantic validation — rejects malformed input."""

    def test_rejects_missing_data_field(self):
        """Response without 'data' field should fail validation."""
        raw = {
            "object": "list",
            "model": "test-model",
        }

        with pytest.raises(Exception):
            ModelEmbeddingResponse(**raw)

    def test_rejects_missing_model_field(self):
        """Response without 'model' field should fail validation."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1]},
            ],
        }

        with pytest.raises(Exception):
            ModelEmbeddingResponse(**raw)

    def test_rejects_non_list_embedding(self):
        """Embedding must be a list of floats."""
        raw = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": "not-a-list"},
            ],
            "model": "test-model",
        }

        with pytest.raises(Exception):
            ModelEmbeddingResponse(**raw)

    def test_defaults_object_to_list(self):
        """When 'object' is missing, defaults to 'list'."""
        raw = {
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1]},
            ],
            "model": "test-model",
        }

        result = ModelEmbeddingResponse(**raw)

        assert result.object == "list"

    def test_defaults_embedding_data_object(self):
        """When embedding 'object' is missing, defaults to 'embedding'."""
        raw = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
            ],
            "model": "test-model",
        }

        result = ModelEmbeddingResponse(**raw)

        assert result.data[0].object == "embedding"
