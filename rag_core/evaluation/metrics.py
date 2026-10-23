"""
Evaluation metrics for Information Retrieval and Answer Grounding.
Implements: Recall@k, Precision@k, Mean Reciprocal Rank (MRR),
Retrieval Latency, and Evidence Faithfulness.
"""

from typing import List, Set
import re


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Precision@k."""
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved_ids[:k]
    if not top_k_retrieved:
        return 0.0
    hits = sum(1 for cid in top_k_retrieved if cid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant_ids:
        return 1.0  # vacuously true if nothing relevant
    top_k_retrieved = retrieved_ids[:k]
    unique_hits = len(set(top_k_retrieved).intersection(relevant_ids))
    return min(1.0, unique_hits / len(relevant_ids))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute Reciprocal Rank (1/rank of first relevant chunk)."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_evidence_faithfulness(answer: str, context_texts: List[str]) -> float:
    """
    Measure lexical overlap grounding between the answer and provided context.
    Returns a score between 0.0 and 1.0.
    """
    if not answer or not context_texts:
        return 0.0

    # Tokenize
    answer_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))
    if not answer_words:
        return 0.0

    combined_context = " ".join(context_texts).lower()
    context_words = set(re.findall(r"\b\w{3,}\b", combined_context))

    grounded_words = answer_words.intersection(context_words)
    return round(len(grounded_words) / len(answer_words), 4)


def verify_citation_coverage(answer: str, num_citations: int) -> bool:
    """Verify whether the answer properly references citations [X]."""
    if num_citations == 0:
        return False
    found = re.findall(r"\[\d+\]", answer)
    return len(found) > 0
