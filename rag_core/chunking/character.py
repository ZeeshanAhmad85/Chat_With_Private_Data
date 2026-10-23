"""
Fixed-separator character text splitter.
"""

import hashlib
from typing import List, Optional
from rag_core.chunking.base import BaseChunker
from rag_core.config import ChunkingConfig
from rag_core.models import Document, DocumentChunk


class CharacterTextSplitter(BaseChunker):
    """Splits documents on a single separator with chunk size and overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        config: Optional[ChunkingConfig] = None,
    ):
        if config:
            self.chunk_size = config.chunk_size
            self.chunk_overlap = config.chunk_overlap
            self.separator = config.separator
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separator = separator

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        splits = text.split(self.separator) if self.separator else list(text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s) + (len(self.separator) if current_chunk else 0)
            if current_len + s_len > self.chunk_size and current_chunk:
                chunk_str = self.separator.join(current_chunk)
                chunks.append(chunk_str)

                # Keep overlap pieces from the end of current_chunk
                overlap_len = 0
                overlap_pieces: List[str] = []
                for p in reversed(current_chunk):
                    if overlap_len + len(p) <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_chunk = overlap_pieces
                current_len = sum(len(p) for p in current_chunk)

            current_chunk.append(s)
            current_len += len(s)

        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []

        for doc in documents:
            text_splits = self.split_text(doc.text)
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
