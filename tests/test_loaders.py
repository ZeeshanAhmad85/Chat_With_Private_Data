"""
Unit tests for document loaders and validation pipeline.
"""

import io
import zipfile
import pytest
from pypdf import PdfWriter
import docx
from rag_core.loaders.factory import DocumentLoaderFactory, load_documents_from_files
from rag_core.loaders.pdf_loader import PDFLoader
from rag_core.loaders.docx_loader import DocxLoader
from rag_core.loaders.text_loader import TextLoader
from rag_core.loaders.zip_loader import ZipLoader
from rag_core.config import IngestionConfig


def create_minimal_pdf_bytes(text: str = "Test PDF Page Content") -> bytes:
    """Helper to generate in-memory valid PDF bytes with text."""
    # Write basic text object in PDF stream
    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # We can write the PDF structure
    writer.write(buf)
    return buf.getvalue()


def create_minimal_docx_bytes(paragraphs: list) -> bytes:
    """Helper to generate in-memory DOCX bytes."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestTextLoader:
    def test_load_valid_txt(self):
        loader = TextLoader()
        content = b"Hello, this is a valid UTF-8 document content."
        stream = io.BytesIO(content)
        docs = loader.load(stream, "sample.txt")
        assert len(docs) == 1
        assert docs[0].text == "Hello, this is a valid UTF-8 document content."
        assert docs[0].metadata.source == "sample.txt"
        assert docs[0].metadata.file_type == "text/plain"

    def test_load_markdown_file_type(self):
        loader = TextLoader()
        content = b"# Title\n\nMarkdown document body."
        docs = loader.load(io.BytesIO(content), "notes.md")
        assert len(docs) == 1
        assert docs[0].metadata.file_type == "text/markdown"

    def test_reject_empty_txt(self):
        loader = TextLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load(io.BytesIO(b""), "empty.txt")

    def test_reject_whitespace_only(self):
        loader = TextLoader()
        with pytest.raises(ValueError, match="only whitespace"):
            loader.load(io.BytesIO(b"   \n\t  "), "blank.txt")


class TestDocxLoader:
    def test_load_valid_docx(self):
        loader = DocxLoader()
        docx_bytes = create_minimal_docx_bytes([
            "First paragraph on RAG architectures.",
            "Second paragraph detailing vector search."
        ])
        docs = loader.load(io.BytesIO(docx_bytes), "sample.docx")
        assert len(docs) == 1
        assert "First paragraph on RAG architectures" in docs[0].text
        assert "Second paragraph detailing vector search" in docs[0].text
        assert docs[0].metadata.source == "sample.docx"

    def test_reject_corrupted_docx(self):
        loader = DocxLoader()
        with pytest.raises(ValueError, match="Corrupted or invalid DOCX"):
            loader.load(io.BytesIO(b"not a docx file"), "bad.docx")


class TestPDFLoader:
    def test_reject_corrupted_pdf(self):
        loader = PDFLoader()
        with pytest.raises(ValueError, match="Corrupted or invalid PDF"):
            loader.load(io.BytesIO(b"random corrupted bytes"), "corrupted.pdf")

    def test_reject_empty_pdf_file(self):
        loader = PDFLoader()
        with pytest.raises(ValueError, match="empty"):
            loader.load(io.BytesIO(b""), "zero_byte.pdf")


class TestZipLoader:
    def test_load_valid_zip_with_nested_documents(self):
        factory = DocumentLoaderFactory()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("docs/intro.txt", "Introduction to Data Science and AI.")
            zf.writestr("docs/summary.md", "# Summary\nConclusion of the project.")

        zip_buf.seek(0)
        docs = factory.load_file(zip_buf, "archive.zip")
        assert len(docs) == 2
        sources = [d.metadata.source for d in docs]
        assert "archive.zip/docs/intro.txt" in sources
        assert "archive.zip/docs/summary.md" in sources

    def test_zip_path_traversal_defense(self):
        factory = DocumentLoaderFactory()
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("../evil.txt", "Malicious attempt to escape directory.")
            zf.writestr("good.txt", "Safe document inside root.")

        zip_buf.seek(0)
        docs = factory.load_file(zip_buf, "traversal.zip")
        assert len(docs) == 1
        assert docs[0].metadata.source == "traversal.zip/good.txt"


class TestLoaderFactoryValidation:
    def test_reject_unsupported_file_extension(self):
        factory = DocumentLoaderFactory()
        with pytest.raises(ValueError, match="Unsupported file type"):
            factory.load_file(io.BytesIO(b"malware.exe"), "malware.exe")

    def test_reject_oversized_file(self):
        cfg = IngestionConfig(max_file_size_bytes=100)  # 100 bytes limit
        factory = DocumentLoaderFactory(config=cfg)
        large_content = b"X" * 150
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            factory.load_file(io.BytesIO(large_content), "large.txt", file_size_bytes=150)
