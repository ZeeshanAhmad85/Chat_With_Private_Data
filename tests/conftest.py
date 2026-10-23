"""
Shared pytest fixtures and test configuration.
"""

import io
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from rag_core.models import Document, DocumentChunk, DocumentMetadata
from rag_core.embeddings.local_embedding import LocalDenseEmbedding
from rag_core.vector_store.local_store import LocalVectorStore


@pytest.fixture
def sample_text_document() -> Document:
    text = (
        "Retrieval-Augmented Generation (RAG) grounds language model outputs "
        "in external reference documents. It combines dense semantic retrieval "
        "with generative language models to produce verifiable answers."
    )
    meta = DocumentMetadata(
        source="test_rag_overview.txt",
        page=1,
        file_type="text/plain",
        file_size_bytes=len(text),
    )
    return Document(text=text, metadata=meta)


@pytest.fixture
def sample_documents() -> list:
    docs = []
    texts = [
        ("doc_ai.txt", "Artificial Intelligence and Machine Learning are core pillars of modern computer science."),
        ("doc_db.txt", "Relational databases use SQL, while vector databases index high-dimensional embeddings."),
        ("doc_net.txt", "TCP/IP protocol stack ensures reliable packet transmission across interconnected networks."),
    ]
    for source, content in texts:
        meta = DocumentMetadata(source=source, page=1, file_type="text/plain", file_size_bytes=len(content))
        docs.append(Document(text=content, metadata=meta))
    return docs


@pytest.fixture
def embedding_model() -> LocalDenseEmbedding:
    return LocalDenseEmbedding(dimension=128, seed=42)


@pytest.fixture
def temp_vector_store(tmp_path: Path) -> LocalVectorStore:
    return LocalVectorStore(persist_dir=tmp_path / "vstore")
