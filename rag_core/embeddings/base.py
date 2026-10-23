"""
Abstract base class for embedding models.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """Abstract interface for text embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of generated embedding vectors."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of document chunks."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute embedding for a single user query."""
        pass
