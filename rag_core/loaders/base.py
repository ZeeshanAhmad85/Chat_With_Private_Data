"""
Abstract base class for document loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, List, Union
from rag_core.models import Document


class BaseDocumentLoader(ABC):
    """Abstract interface for extracting text and metadata from documents."""

    @abstractmethod
    def load(self, source: Union[str, Path, BinaryIO], filename: str) -> List[Document]:
        """Extract documents or document pages from the provided source."""
        pass
