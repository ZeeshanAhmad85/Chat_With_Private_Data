"""
Qdrant Vector Store Adapter.
Preserves Qdrant cloud and local container compatibility with robust error handling.
"""

from typing import List, Optional
from rag_core.models import DocumentChunk, DocumentMetadata, RetrievalResult
from rag_core.vector_store.base import BaseVectorStore


class QdrantVectorStoreAdapter(BaseVectorStore):
    """Adapter for Qdrant client."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "chat_private_docs",
        dimension: int = 384,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.url or "YOUR DATABASE" in self.url:
            raise ValueError(
                "A valid Qdrant URL must be provided in QDRANT_URL or DB_URL. "
                "For local-only operation without external servers, use 'local' vector store mode."
            )

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as e:
            raise ImportError(
                "qdrant-client is not installed. To use Qdrant: pip install qdrant-client"
            ) from e

        self._client = QdrantClient(url=self.url, api_key=self.api_key)
        # Ensure collection exists
        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        from qdrant_client.http import models as qmodels

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                qmodels.PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63 - 1),
                    vector=emb,
                    payload=chunk.to_dict(),
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        from qdrant_client.http import models as qmodels

        query_filter = None
        if source_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.source",
                        match=qmodels.MatchValue(value=source_filter),
                    )
                ]
            )

        search_result = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=query_filter,
        )

        results: List[RetrievalResult] = []
        for rank, hit in enumerate(search_result, start=1):
            p = hit.payload
            meta_dict = p.get("metadata", {})
            meta = DocumentMetadata(
                source=meta_dict.get("source", "unknown"),
                page=meta_dict.get("page"),
                file_type=meta_dict.get("file_type", "text/plain"),
                file_size_bytes=meta_dict.get("file_size_bytes", 0),
                file_hash=meta_dict.get("file_hash", ""),
            )
            chunk = DocumentChunk(
                chunk_id=p.get("chunk_id", str(hit.id)),
                document_id=p.get("document_id", ""),
                text=p.get("text", ""),
                chunk_index=p.get("chunk_index", 0),
                metadata=meta,
                start_char=p.get("start_char", 0),
                end_char=p.get("end_char", 0),
            )
            results.append(RetrievalResult(chunk=chunk, score=hit.score, rank=rank))

        return results

    def count(self) -> int:
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0

    def list_sources(self) -> List[str]:
        # Qdrant scroll for payload sources
        points, _ = self._client.scroll(
            collection_name=self.collection_name, limit=1000, with_payload=True, with_vectors=False
        )
        sources = set()
        for p in points:
            if p.payload and "metadata" in p.payload:
                sources.add(p.payload["metadata"].get("source", "unknown"))
        return sorted(list(sources))

    def delete_by_source(self, source: str) -> int:
        from qdrant_client.http import models as qmodels
        del_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.source",
                    match=qmodels.MatchValue(value=source),
                )
            ]
        )
        self._client.delete(collection_name=self.collection_name, points_selector=del_filter)
        return 1

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._init_client()
