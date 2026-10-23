"""
Factory for creating embedding models based on configuration.
"""

from typing import Optional
from rag_core.config import EmbeddingConfig
from rag_core.embeddings.base import BaseEmbeddingModel
from rag_core.embeddings.local_embedding import LocalDenseEmbedding
from rag_core.embeddings.huggingface import HuggingFaceEmbeddingAdapter
from rag_core.embeddings.openai import OpenAIEmbeddingAdapter


def get_embedding_model(config: Optional[EmbeddingConfig] = None) -> BaseEmbeddingModel:
    """Return an embedding model based on configuration with safe fallback."""
    config = config or EmbeddingConfig()

    provider = config.provider.lower()

    if provider == "openai":
        return OpenAIEmbeddingAdapter(model_name=config.model_name)

    if provider in ("huggingface", "sentence_transformers"):
        try:
            return HuggingFaceEmbeddingAdapter(model_name=config.model_name, device=config.device)
        except Exception:
            # Fallback to local dense embedding if sentence-transformers is missing
            return LocalDenseEmbedding(dimension=config.embedding_dimension)

    # Default: local dense embedding (reproducible, 100% offline)
    return LocalDenseEmbedding(dimension=config.embedding_dimension)
