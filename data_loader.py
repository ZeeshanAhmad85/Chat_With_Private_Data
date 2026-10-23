"""
Legacy data loader adapter for backwards compatibility.
Delegates to rag_core.loaders with robust validation and memory safety.
"""

from typing import Any, List, Tuple
from rag_core.loaders.factory import DocumentLoaderFactory


def process_document(uploaded_file: Any, loader_type: str = "PyPDFLoader", split_pages: bool = True) -> Tuple[List[str], List[Any]]:
    """
    Backwards-compatible wrapper around modern DocumentLoaderFactory.
    Returns:
        (file_names, documents)
    """
    factory = DocumentLoaderFactory()
    file_names = []
    docs = []

    try:
        if isinstance(uploaded_file, tuple):
            filename, file_obj = uploaded_file
        elif hasattr(uploaded_file, "name"):
            filename = uploaded_file.name
            file_obj = uploaded_file
        else:
            filename = str(uploaded_file)
            file_obj = uploaded_file

        file_names.append(filename)
        docs = factory.load_file(file_obj, filename)
    except Exception as e:
        # Gracefully return empty list with error
        print(f"Error loading document '{filename}': {e}")
        return file_names, []

    return file_names, docs


def process_zip(zip_file: Any) -> Tuple[List[str], List[Any]]:
    """
    Backwards-compatible wrapper for unpacking ZIP archives.
    Returns:
        (file_names, flat_list_of_documents)
    """
    factory = DocumentLoaderFactory()
    filename = getattr(zip_file, "name", "archive.zip")
    try:
        docs = factory.load_file(zip_file, filename)
        file_names = sorted(list({d.metadata.source for d in docs}))
        return file_names, docs
    except Exception as e:
        print(f"Error processing ZIP archive: {e}")
        return [], []