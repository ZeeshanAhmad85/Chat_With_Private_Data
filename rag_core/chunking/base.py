"""
Abstract base class for document chunkers.
"""

from abc import ABC, abstractmethod
from typing import List
from rag_core.models import Document, DocumentChunk


class BaseChunker(ABC):
    """Abstract interface for splitting documents into chunks with metadata."""

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Split a list of documents into chunks."""
        pass
