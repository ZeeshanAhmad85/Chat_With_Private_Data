"""
Deterministic local dense embedding model based on Random Indexing.
Implements the Johnson-Lindenstrauss lemma for dimension-reduced semantic hashing.
Runs 100% locally with NumPy; requires no GPU, external downloads, or API keys.
"""

import hashlib
import re
from typing import List
import numpy as np
from rag_core.embeddings.base import BaseEmbeddingModel


class LocalDenseEmbedding(BaseEmbeddingModel):
    """
    A lightweight, reproducible dense embedding model based on Random Projection
    and subword n-gram frequency distributions.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        self._dim = dimension
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        tokens = list(words)
        # Add character tri-grams for subword morphological matching
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(f"##{word[i:i+3]}")
        return tokens

    def _hash_token(self, token: str) -> np.ndarray:
        """Hash token into a deterministic sparse pseudo-random vector."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(h % (2**31 - 1))
        # Sparse ternary projection: +1, -1, or 0
        indices = rng.choice(self._dim, size=min(12, self._dim), replace=False)
        signs = rng.choice([-1.0, 1.0], size=len(indices))
        vec = np.zeros(self._dim, dtype=np.float32)
        vec[indices] = signs
        return vec

    def _embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self._dim, dtype=np.float32)

        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokens:
            vec += self._hash_token(token)

        # L2 Normalization for Cosine Similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text).tolist()
