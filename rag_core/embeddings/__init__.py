"""
Embedding models interface and factory.
"""

from rag_core.embeddings.base import BaseEmbeddingModel
from rag_core.embeddings.local_embedding import LocalDenseEmbedding
from rag_core.embeddings.huggingface import HuggingFaceEmbeddingAdapter
from rag_core.embeddings.openai import OpenAIEmbeddingAdapter
from rag_core.embeddings.factory import get_embedding_model

__all__ = [
    "BaseEmbeddingModel",
    "LocalDenseEmbedding",
    "HuggingFaceEmbeddingAdapter",
    "OpenAIEmbeddingAdapter",
    "get_embedding_model",
]
