"""
Document loaders and ingestion utilities.
"""

from rag_core.loaders.base import BaseDocumentLoader
from rag_core.loaders.pdf_loader import PDFLoader
from rag_core.loaders.docx_loader import DocxLoader
from rag_core.loaders.text_loader import TextLoader
from rag_core.loaders.zip_loader import ZipLoader
from rag_core.loaders.factory import DocumentLoaderFactory, load_documents_from_files

__all__ = [
    "BaseDocumentLoader",
    "PDFLoader",
    "DocxLoader",
    "TextLoader",
    "ZipLoader",
    "DocumentLoaderFactory",
    "load_documents_from_files",
]
