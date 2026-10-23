"""
Evaluation and benchmarking framework.
"""

from rag_core.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    compute_evidence_faithfulness,
    verify_citation_coverage,
)
from rag_core.evaluation.dataset import BENCHMARK_DOCUMENTS, BENCHMARK_QUERIES, EvalSample
from rag_core.evaluation.benchmark import RAGBenchmarkRunner

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "compute_evidence_faithfulness",
    "verify_citation_coverage",
    "BENCHMARK_DOCUMENTS",
    "BENCHMARK_QUERIES",
    "EvalSample",
    "RAGBenchmarkRunner",
]
