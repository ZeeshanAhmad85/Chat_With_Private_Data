"""
Legacy text splitters adapter for backwards compatibility.
Delegates to rag_core.chunking with metadata retention and stable IDs.
"""

from typing import List
from rag_core.chunking.character import CharacterTextSplitter
from rag_core.chunking.recursive import RecursiveCharacterTextSplitter


def apply_char_text_splitter(pages: List[object], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[object]:
    """Apply character text splitting with configurable chunk size and overlap."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def apply_recursive_text_splitter(pages: List[object], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[object]:
    """Apply recursive hierarchical text splitting with configurable chunk size and overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)