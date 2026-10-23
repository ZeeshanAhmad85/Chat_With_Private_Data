"""
Retrieval engine coordinating vector similarity and optional lexical re-ranking.
"""

from typing import List, Optional
from rag_core.config import RetrievalConfig
from rag_core.embeddings.base import BaseEmbeddingModel
from rag_core.models import DocumentChunk, RetrievalResult
from rag_core.vector_store.base import BaseVectorStore


class RetrievalEngine:
    """Manages chunk retrieval with configurable similarity filtering and ranking."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
        config: Optional[RetrievalConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve most relevant chunks for the user query."""
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold

        # 1. Compute query embedding
        query_vector = self.embedding_model.embed_query(query.strip())

        # 2. Query vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            score_threshold=threshold,
            source_filter=source_filter,
        )

        return results
