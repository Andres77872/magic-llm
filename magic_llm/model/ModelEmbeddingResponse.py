"""Pydantic models for embedding API responses.

Follows the OpenAI embedding response format:
https://platform.openai.com/docs/api-reference/embeddings/object
"""

from typing import List, Optional

from pydantic import BaseModel

from magic_llm.model.ModelChatStream import UsageModel


class EmbeddingData(BaseModel):
    """A single embedding vector from the API response."""
    object: str = "embedding"
    index: int
    embedding: List[float]


class ModelEmbeddingResponse(BaseModel):
    """Structured embedding API response.

    Attributes:
        object: Always "list" for embedding responses.
        data: List of embedding vectors (one per input text).
        model: The embedding model used.
        usage: Token usage information (optional, not all providers return it).
    """
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Optional[UsageModel] = None
