"""
Automated benchmark runner for quantitative RAG evaluation.
Generates genuine empirical metrics: Recall@k, Precision@k, MRR, latency, and faithfulness.
"""

import csv
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from rag_core.chunking.recursive import RecursiveCharacterTextSplitter
from rag_core.config import ChunkingConfig, RetrievalConfig
from rag_core.embeddings.local_embedding import LocalDenseEmbedding
from rag_core.evaluation.dataset import BENCHMARK_DOCUMENTS, BENCHMARK_QUERIES, EvalSample
from rag_core.evaluation.metrics import (
    compute_evidence_faithfulness,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_core.generation.generator import GroundedGenerator
from rag_core.generation.providers import OfflineGroundedProvider
from rag_core.models import Document, DocumentMetadata
from rag_core.retrieval.engine import RetrievalEngine
from rag_core.vector_store.local_store import LocalVectorStore


class RAGBenchmarkRunner:
    """Runs repeatable evaluation experiments across configurations."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def setup_pipeline(self):
        """Index benchmark documents into a fresh vector store."""
        docs = []
        for raw in BENCHMARK_DOCUMENTS:
            meta = DocumentMetadata(
                source=raw["source"],
                page=1,
                file_type="text/plain",
                file_size_bytes=len(raw["text"]),
            )
            docs.append(Document(text=raw["text"], metadata=meta))

        chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = chunker.split_documents(docs)

        embedding_model = LocalDenseEmbedding(dimension=384, seed=42)
        texts = [c.text for c in chunks]
        embeddings = embedding_model.embed_documents(texts)

        vector_store = LocalVectorStore()
        vector_store.add_chunks(chunks, embeddings)

        engine = RetrievalEngine(
            vector_store=vector_store,
            embedding_model=embedding_model,
            config=RetrievalConfig(top_k=self.top_k),
        )

        generator = GroundedGenerator(provider=OfflineGroundedProvider())

        return engine, generator, chunks

    def run_benchmark(self) -> Dict[str, any]:
        """Execute all benchmark queries and record metrics."""
        engine, generator, chunks = self.setup_pipeline()

        query_results = []
        recalls = []
        precisions = []
        mrrs = []
        latencies_ms = []
        faithfulness_scores = []
        refusal_correctness = []

        for item in BENCHMARK_QUERIES:
            t0 = time.time()
            retrieved = engine.retrieve(item.question, top_k=self.top_k)
            retrieval_time_ms = (time.time() - t0) * 1000

            generation = generator.generate_answer(item.question, retrieved)

            retrieved_sources = [r.chunk.metadata.source for r in retrieved]
            relevant_sources = {item.target_source} if item.is_answerable else set()

            rec = recall_at_k(retrieved_sources, relevant_sources, k=self.top_k)
            prec = precision_at_k(retrieved_sources, relevant_sources, k=self.top_k)
            rr = reciprocal_rank(retrieved_sources, relevant_sources)

            context_texts = [r.chunk.text for r in retrieved]
            faith = compute_evidence_faithfulness(generation.answer, context_texts)

            # Check unanswerable query refusal
            if not item.is_answerable:
                refusal_ok = "not contain sufficient information" in generation.answer.lower()
                refusal_correctness.append(1.0 if refusal_ok else 0.0)
            else:
                recalls.append(rec)
                precisions.append(prec)
                mrrs.append(rr)

            latencies_ms.append(retrieval_time_ms)
            faithfulness_scores.append(faith)

            query_results.append({
                "query_id": item.query_id,
                "question": item.question,
                "is_answerable": item.is_answerable,
                "target_source": item.target_source,
                "retrieved_sources": retrieved_sources,
                "recall@k": rec,
                "precision@k": prec,
                "mrr": rr,
                "retrieval_latency_ms": round(retrieval_time_ms, 2),
                "faithfulness": faith,
                "answer": generation.answer,
            })

        summary = {
            "experiment_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "configuration": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "embedding_dimension": 384,
            },
            "metrics": {
                "mean_recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
                "mean_precision_at_k": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
                "mean_reciprocal_rank": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
                "mean_retrieval_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
                "mean_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
                "refusal_accuracy_on_unanswerable": round(
                    sum(refusal_correctness) / len(refusal_correctness), 4
                ) if refusal_correctness else 1.0,
            },
            "queries": query_results,
        }

        return summary

    def export_results(self, summary: Dict[str, any], output_dir: Path) -> Tuple[Path, Path]:
        """Save results to JSON and CSV formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "evaluation_results.json"
        csv_path = output_dir / "evaluation_results.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write query-level CSV
        fieldnames = [
            "query_id", "question", "is_answerable", "target_source",
            "recall@k", "precision@k", "mrr", "retrieval_latency_ms", "faithfulness"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in summary["queries"]:
                row = {k: q[k] for k in fieldnames}
                writer.writerow(row)

        return json_path, csv_path
