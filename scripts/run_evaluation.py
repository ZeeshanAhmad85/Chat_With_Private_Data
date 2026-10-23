"""
Executable benchmark script for RAG evaluation.
Runs quantitative experiments and outputs evaluation metrics to JSON and CSV.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from rag_core.evaluation.benchmark import RAGBenchmarkRunner


def main():
    print("=" * 60)
    print("  RAG Framework Quantitative Evaluation Benchmark")
    print("=" * 60)

    runner = RAGBenchmarkRunner(chunk_size=500, chunk_overlap=100, top_k=3)
    print("Executing benchmark suite across golden queries...")
    results = runner.run_benchmark()

    output_dir = root_dir / "evaluation"
    json_path, csv_path = runner.export_results(results, output_dir)

    print("\n--- Summary Benchmark Results ---")
    metrics = results["metrics"]
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    print(f"\nSaved detailed evaluation JSON: {json_path}")
    print(f"Saved query-level CSV:          {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
