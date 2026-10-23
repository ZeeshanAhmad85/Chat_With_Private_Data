"""
Hugging Face / Sentence-Transformers embedding model adapter.
Preserves support for BAAI/bge-small-en-v1.5 and intfloat/e5-large-v2.
"""

from typing import List, Optional
from rag_core.embeddings.base import BaseEmbeddingModel


class HuggingFaceEmbeddingAdapter(BaseEmbeddingModel):
    """Adapter for sentence_transformers and HuggingFace models."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Inspect dimension
            test_emb = self._model.encode("test", convert_to_numpy=True)
            self._dimension = len(test_emb)
        except ImportError as e:
            raise ImportError(
                f"sentence-transformers is not installed. To use '{self.model_name}', "
                f"install sentence-transformers: pip install sentence-transformers"
            ) from e

    @property
    def dimension(self) -> int:
        return self._dimension or 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
