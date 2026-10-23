"""
OpenAI Embedding Adapter with timeout and API key validation.
"""

import os
from typing import List, Optional
from rag_core.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingAdapter(BaseEmbeddingModel):
    """Adapter for OpenAI text-embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self._dimension = dimension
        if not self.api_key or "your" in self.api_key.lower():
            raise ValueError(
                "A valid OpenAI API key must be provided in OPENAI_API_KEY environment variable."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=texts, model=self.model_name)
            return [data.embedding for data in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding call failed: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(input=[text], model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI query embedding failed: {e}") from e
