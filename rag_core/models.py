"""
Core data structures for the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class DocumentMetadata:
    """Metadata associated with an ingested document or page."""
    source: str
    page: Optional[int] = None
    file_type: str = "text/plain"
    file_size_bytes: int = 0
    file_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "extra": self.extra,
        }


@dataclass
class Document:
    """An ingested document or document page."""
    text: str
    metadata: DocumentMetadata
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            content = f"{self.metadata.source}:{self.metadata.page}:{self.text[:100]}"
            self.doc_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


@dataclass
class DocumentChunk:
    """A granular chunk of text derived from a Document."""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: DocumentMetadata
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class RetrievalResult:
    """A single retrieved chunk with similarity score and ranking."""
    chunk: DocumentChunk
    score: float
    rank: int = 1


@dataclass
class Citation:
    """Source attribution for generated claims."""
    citation_id: int
    source: str
    page: Optional[int]
    chunk_id: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }


@dataclass
class GenerationResult:
    """The final generated response including grounding citations and telemetry."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    latency_seconds: float = 0.0
    model_name: str = "offline-grounded"
    grounded: bool = True
    insufficient_context: bool = False
