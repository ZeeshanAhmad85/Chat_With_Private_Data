"""
Abstract base class for vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from rag_core.models import DocumentChunk, RetrievalResult


class BaseVectorStore(ABC):
    """Abstract interface for indexing and searching document chunk embeddings."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Index a list of chunks and their embedding vectors."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Perform similarity search returning top-k matching chunks."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed chunks."""
        pass

    @abstractmethod
    def list_sources(self) -> List[str]:
        """Return list of unique document sources currently indexed."""
        pass

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        """Delete all chunks associated with a specific document source."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed data from the store."""
        pass
