"""
Recursive character text splitter with semantic separator hierarchy.
"""

import hashlib
from typing import List, Optional
from rag_core.chunking.base import BaseChunker
from rag_core.config import ChunkingConfig
from rag_core.models import Document, DocumentChunk


class RecursiveCharacterTextSplitter(BaseChunker):
    """Splits text recursively across semantic boundaries with overlap."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        config: Optional[ChunkingConfig] = None,
    ):
        if config:
            self.chunk_size = config.chunk_size
            self.chunk_overlap = config.chunk_overlap
            self.separators = config.separators or self.DEFAULT_SEPARATORS
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge splits up to chunk_size with chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk and separator else 0)
            if current_len + piece_len > self.chunk_size and current_chunk:
                chunk_str = separator.join(current_chunk) if separator else "".join(current_chunk)
                if chunk_str.strip():
                    final_chunks.append(chunk_str.strip())

                # Build overlap
                overlap_len = 0
                overlap_pieces: List[str] = []
                for p in reversed(current_chunk):
                    p_len = len(p) + (len(separator) if overlap_pieces and separator else 0)
                    if overlap_len + p_len <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += p_len
                    else:
                        break
                current_chunk = overlap_pieces
                current_len = sum(len(p) for p in current_chunk) + (
                    len(separator) * (len(current_chunk) - 1) if current_chunk and separator else 0
                )

            current_chunk.append(piece)
            current_len += len(piece) + (len(separator) if len(current_chunk) > 1 and separator else 0)

        if current_chunk:
            chunk_str = separator.join(current_chunk) if separator else "".join(current_chunk)
            if chunk_str.strip():
                final_chunks.append(chunk_str.strip())

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            text_splits = self._split_text(doc.text, self.separators)
            char_cursor = 0

            for idx, split_text in enumerate(text_splits):
                start_char = doc.text.find(split_text, char_cursor)
                if start_char == -1:
                    start_char = char_cursor
                end_char = start_char + len(split_text)
                char_cursor = max(char_cursor, start_char + 1)

                chunk_hash = hashlib.sha256(
                    f"{doc.doc_id}:{idx}:{split_text[:50]}".encode("utf-8")
                ).hexdigest()[:12]

                chunk = DocumentChunk(
                    chunk_id=f"{doc.doc_id}_c{idx}_{chunk_hash}",
                    document_id=doc.doc_id,
                    text=split_text,
                    chunk_index=idx,
                    metadata=doc.metadata,
                    start_char=start_char,
                    end_char=end_char,
                )
                all_chunks.append(chunk)

        return all_chunks
