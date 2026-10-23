"""
Unit tests for vector database operations, persistence, and index isolation.
"""

from pathlib import Path
import numpy as np
import pytest
from rag_core.models import DocumentChunk, DocumentMetadata
from rag_core.vector_store.local_store import LocalVectorStore


def make_chunk(cid: str, source: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id=f"doc_{source}",
        text=text,
        chunk_index=0,
        metadata=DocumentMetadata(source=source, page=1),
    )


class TestLocalVectorStore:
    def test_add_and_search_exact_cosine(self):
        store = LocalVectorStore()
        c1 = make_chunk("c1", "ml.txt", "Machine learning algorithms learn from data.")
        c2 = make_chunk("c2", "history.txt", "Ancient Roman architecture featured arches and aqueducts.")

        # Orthogonal embedding vectors
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        store.add_chunks([c1, c2], [v1, v2])

        assert store.count() == 2

        # Query aligned with v1
        results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk.chunk_id == "c1"
        assert np.isclose(results[0].score, 1.0)
        assert np.isclose(results[1].score, 0.0)

    def test_score_threshold_filtering(self):
        store = LocalVectorStore()
        c1 = make_chunk("c1", "doc1.txt", "Relevant text")
        c2 = make_chunk("c2", "doc2.txt", "Irrelevant text")
        store.add_chunks([c1, c2], [[1.0, 0.0], [0.0, 1.0]])

        # Query aligned with c1, set threshold 0.5
        results = store.search(query_vector=[1.0, 0.0], top_k=5, score_threshold=0.5)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_persistence_save_and_reload(self, tmp_path: Path):
        persist_dir = tmp_path / "saved_vstore"
        store1 = LocalVectorStore(persist_dir=persist_dir)

        c1 = make_chunk("c1", "a.txt", "Alpha content")
        c2 = make_chunk("c2", "b.txt", "Beta content")
        store1.add_chunks([c1, c2], [[0.5, 0.5], [0.1, 0.9]])

        # Create new store instance pointing to same directory
        store2 = LocalVectorStore(persist_dir=persist_dir)
        assert store2.count() == 2
        sources = store2.list_sources()
        assert "a.txt" in sources
        assert "b.txt" in sources

        # Search on reloaded store
        results = store2.search([0.5, 0.5], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"

    def test_delete_by_source_index_isolation(self, tmp_path: Path):
        store = LocalVectorStore(persist_dir=tmp_path / "iso_store")
        c1 = make_chunk("c1", "private_contract.pdf", "Confidential clause 1")
        c2 = make_chunk("c2", "private_contract.pdf", "Confidential clause 2")
        c3 = make_chunk("c3", "public_report.pdf", "Public annual report")

        store.add_chunks([c1, c2, c3], [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
        assert store.count() == 3

        # Delete private document
        deleted = store.delete_by_source("private_contract.pdf")
        assert deleted == 2
        assert store.count() == 1
        assert store.list_sources() == ["public_report.pdf"]

        # Ensure search never returns deleted document
        results = store.search([1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].chunk.metadata.source == "public_report.pdf"

    def test_clear_vector_store(self):
        store = LocalVectorStore()
        store.add_chunks([make_chunk("c1", "a.txt", "test")], [[1.0, 0.0]])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert store.search([1.0, 0.0]) == []
