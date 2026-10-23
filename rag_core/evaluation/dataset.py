"""
Benchmark Evaluation Dataset and Schema.
Includes verified computer science & RAG domain reference texts,
golden queries, reference answers, and negative/out-of-domain queries to test grounding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalSample:
    """A single evaluation benchmark item."""
    query_id: str
    question: str
    expected_answer: str
    target_source: str
    key_facts: List[str]
    is_answerable: bool = True


# Standard academic evaluation corpus (RAG Systems & Data Privacy in Computer Science)
BENCHMARK_DOCUMENTS = [
    {
        "source": "rag_architecture_overview.txt",
        "text": """Retrieval-Augmented Generation (RAG) is an architectural pattern that combines parametric memory 
(a pretrained large language model) with non-parametric memory (a dense vector index or document store).
In standard dense retrieval, documents are segmented into contiguous text chunks, typically between 200 and 1000 tokens, 
with a small overlap such as 50 to 200 tokens to preserve context across chunk boundaries.
An embedding model maps both document chunks and user queries into a shared continuous vector space.
During inference, maximum inner product search or cosine similarity is computed between the query embedding and indexed vectors.
The top-k most relevant passages are concatenated into a structured prompt containing strict instructions and delimiters.
RAG reduces hallucinations by grounding responses in external factual sources, though it does not eliminate hallucinations entirely.
System performance depends heavily on chunk granularity, embedding quality, vector index efficiency, and prompt construction."""
    },
    {
        "source": "gdpr_data_privacy.txt",
        "text": """The General Data Protection Regulation (GDPR) establishes strict requirements for processing personal and private data in the European Union.
Article 5 mandates data minimization, ensuring that personal data collected is adequate, relevant, and limited to what is necessary.
Article 17 enshrines the right to erasure ('right to be forgotten'), requiring systems to remove stored personal data upon legitimate request.
In private document AI systems, transmitting sensitive enterprise documents to external cloud APIs creates compliance and confidentiality risks.
To comply with GDPR, architectures should implement local-first processing, vector database deletion endpoints, and zero-retention data policies.
Organisations must maintain clear audit trails of where document embeddings and cache files are stored on disk."""
    },
    {
        "source": "vector_indexing_tradeoffs.txt",
        "text": """Vector databases employ various indexing structures to balance search latency, memory consumption, and recall.
Exact flat indexing computes full brute-force cosine distances across all vectors, guaranteeing 100% recall at O(N) query time.
Hierarchical Navigable Small World (HNSW) graphs offer logarithmic O(log N) search latency with high recall, but require substantial RAM.
Inverted File with Product Quantization (IVF-PQ) compresses high-dimensional vectors into compact byte codes, reducing memory footprints at the cost of approximate recall.
Reciprocal Rank Fusion (RRF) is an effective algorithmic technique for combining rankings from dense semantic search and sparse lexical BM25 matching."""
    }
]

BENCHMARK_QUERIES = [
    EvalSample(
        query_id="q1",
        question="What components combine parametric and non-parametric memory in RAG?",
        expected_answer="A pretrained large language model (parametric) and a dense vector index or document store (non-parametric).",
        target_source="rag_architecture_overview.txt",
        key_facts=["parametric", "large language model", "non-parametric", "vector index"],
        is_answerable=True,
    ),
    EvalSample(
        query_id="q2",
        question="What is the standard chunk overlap used in document chunking?",
        expected_answer="Between 50 and 200 tokens.",
        target_source="rag_architecture_overview.txt",
        key_facts=["50 to 200 tokens", "overlap"],
        is_answerable=True,
    ),
    EvalSample(
        query_id="q3",
        question="Which GDPR article covers the right to erasure?",
        expected_answer="Article 17 establishes the right to erasure ('right to be forgotten').",
        target_source="gdpr_data_privacy.txt",
        key_facts=["Article 17", "right to erasure"],
        is_answerable=True,
    ),
    EvalSample(
        query_id="q4",
        question="What trade-off does exact flat indexing have compared to HNSW?",
        expected_answer="Flat indexing guarantees 100% recall with O(N) complexity, while HNSW provides O(log N) latency but requires high RAM.",
        target_source="vector_indexing_tradeoffs.txt",
        key_facts=["100% recall", "O(N)", "HNSW", "O(log N)"],
        is_answerable=True,
    ),
    EvalSample(
        query_id="q5",
        question="What is the capital city of Australia?",
        expected_answer="The provided documents do not contain sufficient information to answer this question.",
        target_source="none",
        key_facts=["not contain sufficient information"],
        is_answerable=False,  # Out-of-domain negative test
    ),
]
