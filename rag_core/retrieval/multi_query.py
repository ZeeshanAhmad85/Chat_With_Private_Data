"""
Multi-Query Expansion and Reciprocal Rank Fusion (RRF).
Enhances distance-based similarity search by exploring multi-perspective formulations
and fusing ranking results using standard Information Retrieval algorithms (Cormack et al., 2009).
"""

from collections import defaultdict
from typing import Callable, List, Optional
from rag_core.models import DocumentChunk, RetrievalResult
from rag_core.retrieval.engine import RetrievalEngine


class MultiQueryRetriever:
    """
    Expands a single user query into multiple perspectives and fuses
    the retrieved chunk rankings using Reciprocal Rank Fusion (RRF).
    """

    RRF_K = 60  # Standard smoothing constant from Cormack et al. (2009)

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        query_generator_fn: Optional[Callable[[str, int], List[str]]] = None,
    ):
        self.engine = retrieval_engine
        self.query_generator_fn = query_generator_fn or self._default_heuristic_expansion

    def _default_heuristic_expansion(self, query: str, num_queries: int = 3) -> List[str]:
        """Deterministic query variations for offline execution."""
        variations = [query]
        # Clean question words to form keyword-focused query
        keywords = " ".join([w for w in query.split() if w.lower() not in {"what", "how", "why", "when", "is", "are", "the", "a", "an"}])
        if keywords and keywords != query:
            variations.append(keywords)
        # Add summary/overview intent
        variations.append(f"overview summary of {query}")
        return variations[:num_queries]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        num_queries: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            return []

        expanded_queries = self.query_generator_fn(query.strip(), num_queries)
        if query not in expanded_queries:
            expanded_queries.insert(0, query)

        rrf_scores = defaultdict(float)
        chunk_map = {}

        for q in expanded_queries:
            results = self.engine.retrieve(q, top_k=top_k * 2, source_filter=source_filter)
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk.chunk_id
                chunk_map[chunk_id] = item.chunk
                # Reciprocal Rank Fusion: 1 / (k + rank)
                rrf_scores[chunk_id] += 1.0 / (self.RRF_K + rank)

        # Sort chunks by fused RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        fused_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_chunk_ids, start=1):
            fused_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=float(rrf_scores[cid]),
                    rank=rank,
                )
            )

        return fused_results
