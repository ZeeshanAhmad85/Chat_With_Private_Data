"""
PDF Document Loader with page metadata and graceful fallback.
Uses pypdf for clean extraction without heavy external dependencies.
"""

import hashlib
import io
from pathlib import Path
from typing import BinaryIO, List, Union
from pypdf import PdfReader
from rag_core.models import Document, DocumentMetadata


class PDFLoader:
    """Extracts text page by page from PDF files."""

    def load(self, source: Union[str, Path, BinaryIO], filename: str) -> List[Document]:
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                content = f.read()
        elif hasattr(source, "read"):
            content = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if not content:
            raise ValueError(f"File '{filename}' is empty (0 bytes).")

        file_hash = hashlib.sha256(content).hexdigest()
        stream = io.BytesIO(content)

        try:
            reader = PdfReader(stream)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid PDF format in '{filename}': {e}") from e

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise ValueError(f"PDF '{filename}' is password-protected: {e}") from e

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise ValueError(f"PDF '{filename}' contains 0 pages.")

        documents: List[Document] = []
        total_extracted_chars = 0

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            cleaned_text = page_text.strip()
            total_extracted_chars += len(cleaned_text)

            if cleaned_text:
                metadata = DocumentMetadata(
                    source=filename,
                    page=page_idx,
                    file_type="application/pdf",
                    file_size_bytes=len(content),
                    file_hash=file_hash,
                    extra={"total_pages": num_pages},
                )
                documents.append(Document(text=cleaned_text, metadata=metadata))

        if total_extracted_chars == 0:
            raise ValueError(
                f"PDF '{filename}' contains no readable text across {num_pages} page(s). "
                "The document may consist entirely of scanned images (OCR not currently enabled)."
            )

        return documents
