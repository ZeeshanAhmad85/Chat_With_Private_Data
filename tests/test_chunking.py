"""
Unit tests for text chunkers and metadata preservation.
"""

import pytest
from rag_core.chunking.character import CharacterTextSplitter
from rag_core.chunking.recursive import RecursiveCharacterTextSplitter
from rag_core.models import Document, DocumentMetadata


def make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(
        text=text,
        metadata=DocumentMetadata(source=source, page=1, file_type="text/plain", file_size_bytes=len(text)),
    )


class TestCharacterTextSplitter:
    def test_basic_character_split(self):
        text = "Paragraph one with some text.\n\nParagraph two with other facts.\n\nParagraph three."
        doc = make_doc(text)
        splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=10, separator="\n\n")
        chunks = splitter.split_documents([doc])

        assert len(chunks) >= 2
        for c in chunks:
            assert c.metadata.source == "test.txt"
            assert c.document_id == doc.doc_id
            assert len(c.text) > 0

    def test_overlap_validation_error(self):
        with pytest.raises(ValueError, match="strictly less than chunk_size"):
            CharacterTextSplitter(chunk_size=100, chunk_overlap=100)


class TestRecursiveCharacterTextSplitter:
    def test_recursive_semantic_boundaries(self):
        text = (
            "Section 1: Information Retrieval.\n\n"
            "Vector databases store high-dimensional embeddings. "
            "Cosine similarity is used for dense ranking.\n\n"
            "Section 2: Generation.\n\n"
            "LLMs generate answers conditioned on retrieved context."
        )
        doc = make_doc(text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents([doc])

        assert len(chunks) > 1
        for idx, c in enumerate(chunks):
            assert c.chunk_index == idx
            assert c.document_id == doc.doc_id
            assert c.chunk_id.startswith(f"{doc.doc_id}_c{idx}_")
            assert c.metadata.source == "test.txt"

    def test_empty_document_handling(self):
        doc = make_doc("")
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents([doc])
        assert len(chunks) == 0

    def test_chunk_ids_deterministic(self):
        text = "Deterministic chunking content."
        doc = make_doc(text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks1 = splitter.split_documents([doc])
        chunks2 = splitter.split_documents([doc])
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
