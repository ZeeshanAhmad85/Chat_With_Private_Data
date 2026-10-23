"""
Document loader factory and validation pipeline.
"""

from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
from rag_core.config import IngestionConfig
from rag_core.models import Document
from rag_core.loaders.pdf_loader import PDFLoader
from rag_core.loaders.docx_loader import DocxLoader
from rag_core.loaders.text_loader import TextLoader
from rag_core.loaders.zip_loader import ZipLoader


class DocumentLoaderFactory:
    """Dispatches files to appropriate loaders with strict validation."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self._loaders: Dict[str, object] = {
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader(),
        }
        # ZipLoader uses a factory callback to resolve member files
        self._loaders[".zip"] = ZipLoader(self._resolve_subloader)

    def _resolve_subloader(self, filename: str):
        suffix = Path(filename).suffix.lower()
        if suffix in self._loaders and suffix != ".zip":
            return self._loaders[suffix]
        raise ValueError(f"No subloader for file extension: {suffix}")

    def validate_file(self, filename: str, file_size_bytes: int) -> None:
        """Validate filename extension and file size."""
        suffix = Path(filename).suffix.lower()
        if suffix not in self.config.allowed_extensions:
            allowed = ", ".join(self.config.allowed_extensions)
            raise ValueError(
                f"Unsupported file type '{suffix}' for file '{filename}'. Allowed extensions: {allowed}"
            )

        if file_size_bytes > self.config.max_file_size_bytes:
            max_mb = self.config.max_file_size_bytes / (1024 * 1024)
            actual_mb = file_size_bytes / (1024 * 1024)
            raise ValueError(
                f"File '{filename}' ({actual_mb:.1f} MB) exceeds maximum allowed size ({max_mb:.1f} MB)."
            )

    def load_file(
        self, source: Union[str, Path, BinaryIO], filename: str, file_size_bytes: Optional[int] = None
    ) -> List[Document]:
        """Validate and load documents from a file path or file-like object."""
        if file_size_bytes is None:
            if isinstance(source, (str, Path)):
                file_size_bytes = Path(source).stat().st_size
            elif hasattr(source, "size"):
                # Streamlit UploadedFile object has .size
                file_size_bytes = getattr(source, "size")
            elif hasattr(source, "seek") and hasattr(source, "tell"):
                pos = source.tell()
                source.seek(0, 2)
                file_size_bytes = source.tell()
                source.seek(pos)
            else:
                file_size_bytes = 0

        self.validate_file(filename, file_size_bytes)
        suffix = Path(filename).suffix.lower()
        loader = self._loaders[suffix]

        return loader.load(source, filename)


def load_documents_from_files(
    files: List[Tuple[str, Union[str, Path, BinaryIO]]],
    config: Optional[IngestionConfig] = None,
) -> Tuple[List[Document], List[str], List[Dict[str, str]]]:
    """
    Batch load documents from a list of (filename, file_source) pairs.
    Returns:
        (all_documents, successful_filenames, list_of_errors)
    """
    factory = DocumentLoaderFactory(config)
    all_docs: List[Document] = []
    successes: List[str] = []
    errors: List[Dict[str, str]] = []

    for filename, file_source in files:
        try:
            docs = factory.load_file(file_source, filename)
            all_docs.extend(docs)
            successes.append(filename)
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})

    return all_docs, successes, errors
