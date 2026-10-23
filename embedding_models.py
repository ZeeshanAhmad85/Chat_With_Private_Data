"""
Legacy embedding models adapter for backwards compatibility.
Delegates to rag_core.embeddings with safe offline fallback.
"""

from rag_core.config import EmbeddingConfig
from rag_core.embeddings.factory import get_embedding_model


def apply_embedding_model1():
    """Load BAAI/bge-small-en-v1.5 embedding model with safe fallback."""
    cfg = EmbeddingConfig(provider="huggingface", model_name="BAAI/bge-small-en-v1.5")
    return get_embedding_model(cfg)


def apply_embedding_model2():
    """Load intfloat/e5-large-v2 embedding model with safe fallback."""
    cfg = EmbeddingConfig(provider="huggingface", model_name="intfloat/e5-large-v2")
    return get_embedding_model(cfg)