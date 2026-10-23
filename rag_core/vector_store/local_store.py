"""
Lightweight, persistent local vector database using NumPy.
Provides exact cosine similarity search, source filtering, and JSON/NPZ persistence.
Zero external server dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from rag_core.models import DocumentChunk, DocumentMetadata, RetrievalResult
from rag_core.vector_store.base import BaseVectorStore


class LocalVectorStore(BaseVectorStore):
    """
    In-memory and disk-persisted vector database utilizing NumPy
    for vectorized cosine similarity matrix multiplications.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._chunks: List[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None

        if self.persist_dir and self.persist_dir.exists():
            self.load()

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        new_matrix = np.array(embeddings, dtype=np.float32)

        # Normalize rows to unit length for fast dot-product cosine similarity
        norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        new_matrix = new_matrix / norms

        if self._embeddings is None or len(self._chunks) == 0:
            self._embeddings = new_matrix
            self._chunks = list(chunks)
        else:
            self._embeddings = np.vstack([self._embeddings, new_matrix])
            self._chunks.extend(chunks)

        if self.persist_dir:
            self.save()

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        # Matrix-vector dot product for all cosine similarities
        scores = np.dot(self._embeddings, q_vec)

        # Apply source filtering if requested
        valid_indices = []
        for idx, chunk in enumerate(self._chunks):
            if source_filter and chunk.metadata.source != source_filter:
                continue
            if scores[idx] >= score_threshold:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        # Sort filtered indices descending by score
        sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            results.append(
                RetrievalResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results

    def count(self) -> int:
        return len(self._chunks)

    def list_sources(self) -> List[str]:
        sources = set()
        for chunk in self._chunks:
            sources.add(chunk.metadata.source)
        return sorted(list(sources))

    def delete_by_source(self, source: str) -> int:
        if not self._chunks:
            return 0

        keep_indices = [i for i, c in enumerate(self._chunks) if c.metadata.source != source]
        removed_count = len(self._chunks) - len(keep_indices)

        if removed_count == 0:
            return 0

        if not keep_indices:
            self.clear()
            return removed_count

        self._chunks = [self._chunks[i] for i in keep_indices]
        self._embeddings = self._embeddings[keep_indices]

        if self.persist_dir:
            self.save()

        return removed_count

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None
        if self.persist_dir and self.persist_dir.exists():
            meta_file = self.persist_dir / "chunks_metadata.json"
            emb_file = self.persist_dir / "embeddings.npz"
            if meta_file.exists():
                meta_file.unlink()
            if emb_file.exists():
                emb_file.unlink()

    def save(self) -> None:
        if not self.persist_dir:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        meta_file = self.persist_dir / "chunks_metadata.json"
        emb_file = self.persist_dir / "embeddings.npz"

        # Serialize chunks
        serialized_chunks = [c.to_dict() for c in self._chunks]
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(serialized_chunks, f, indent=2)

        # Save embeddings matrix
        if self._embeddings is not None:
            np.savez_compressed(emb_file, embeddings=self._embeddings)

    def load(self) -> None:
        if not self.persist_dir:
            return

        meta_file = self.persist_dir / "chunks_metadata.json"
        emb_file = self.persist_dir / "embeddings.npz"

        if not meta_file.exists() or not emb_file.exists():
            return

        with open(meta_file, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        chunks: List[DocumentChunk] = []
        for raw in raw_chunks:
            meta_dict = raw["metadata"]
            meta = DocumentMetadata(
                source=meta_dict["source"],
                page=meta_dict.get("page"),
                file_type=meta_dict.get("file_type", "text/plain"),
                file_size_bytes=meta_dict.get("file_size_bytes", 0),
                file_hash=meta_dict.get("file_hash", ""),
                extra=meta_dict.get("extra", {}),
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=raw["chunk_id"],
                    document_id=raw["document_id"],
                    text=raw["text"],
                    chunk_index=raw["chunk_index"],
                    metadata=meta,
                    start_char=raw.get("start_char", 0),
                    end_char=raw.get("end_char", 0),
                )
            )

        data = np.load(emb_file)
        self._embeddings = data["embeddings"]
        self._chunks = chunks
