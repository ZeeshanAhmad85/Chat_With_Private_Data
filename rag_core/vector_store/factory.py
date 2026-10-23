"""
Factory for creating vector store instances based on configuration.
"""

from typing import Optional
from rag_core.config import VectorStoreConfig
from rag_core.vector_store.base import BaseVectorStore
from rag_core.vector_store.local_store import LocalVectorStore
from rag_core.vector_store.qdrant_store import QdrantVectorStoreAdapter


def get_vector_store(config: Optional[VectorStoreConfig] = None) -> BaseVectorStore:
    """Return a vector store based on configuration."""
    config = config or VectorStoreConfig()

    store_type = config.store_type.lower()
    if store_type == "qdrant":
        try:
            return QdrantVectorStoreAdapter(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                collection_name=config.collection_name,
            )
        except Exception as e:
            # If Qdrant is unavailable, fallback to local store with warning
            return LocalVectorStore(persist_dir=config.persist_directory)

    # Default to LocalVectorStore
    return LocalVectorStore(persist_dir=config.persist_directory)
