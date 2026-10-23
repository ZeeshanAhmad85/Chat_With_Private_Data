# PrivateData-RAG: Verifiable Document Question-Answering with Local-First Grounding

[![CI Pipeline](https://github.com/ZeeshanAhmad85/Chat_With_Private_Data/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeeshanAhmad85/Chat_With_Private_Data/actions)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/pytest-31%20passed-success.svg)](tests/)

---

## 1. Project Overview & Motivation

In enterprise, legal, clinical, and academic settings, organizations deal with confidential documents containing intellectual property, personally identifiable information (PII), or proprietary trade secrets. While Large Language Models (LLMs) provide powerful question-answering capabilities, standard commercial deployments transmit private queries and sensitive document corpora to external third-party cloud servers. This transmission introduces serious compliance violations (such as **EU GDPR Article 5 & 17**) and risks data leakage.

Furthermore, naive LLMs suffer from **hallucinations** and lack source verifiability—producing persuasive but incorrect answers without audit trails.

**PrivateData-RAG** is an end-to-end, reproducible document question-answering system built upon the **Retrieval-Augmented Generation (RAG)** architecture. It solves these dual challenges by:
1. **Enforcing strict factual grounding**: Answers are conditioned solely on retrieved passages, and the system explicitly refuses out-of-domain questions rather than inventing facts.
2. **Providing granular source attribution**: Every claim is linked to verified numbered citation tokens (`[1]`, `[2]`), source filenames, and page references.
3. **Guaranteeing a local-first privacy boundary**: Zero external API calls or network egress are required by default; vectorization and generation operate completely offline, with pluggable optional adapters for cloud LLMs (OpenAI) and distributed vector databases (Qdrant).

*Note: This project serves as verified programming and machine learning evidence for Master's degree applications in Computer Science, Artificial Intelligence, and Data Science.*

---

## 2. System Architecture

The following Mermaid sequence and block diagrams illustrate the complete pipeline lifecycle from ingestion to citation synthesis:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Validation"]
        A[User Upload: PDF / DOCX / TXT / MD / ZIP] --> B[DocumentLoaderFactory]
        B --> C{File Validation}
        C -- Failed --> Err[Rejection: Oversized / Unsupported / Corrupt]
        C -- Passed --> D[Text Extraction & Page Metadata Preservation]
        D --> E[Recursive Semantic Chunker]
        E --> F[(Document Chunks + Hashes + Spans)]
    end

    subgraph Vectorization ["2. Embedding & Vector Indexing"]
        F --> G[Embedding Engine: Local Random Indexing / BGE / OpenAI]
        G --> H[(Local Vector Store / Qdrant Cloud)]
    end

    subgraph Retrieval ["3. Information Retrieval"]
        Q[User Query] --> R{Multi-Query Expansion?}
        R -- Yes --> S[Generate Sub-queries & Reciprocal Rank Fusion]
        R -- No --> T[Cosine Similarity Matrix Multiplication]
        S --> U[Ranked Candidate Chunks]
        T --> U
        U --> V[Threshold & Top-K Filter]
    end

    subgraph Generation ["4. Grounded Generation & Attribution"]
        V --> W[Grounded Prompt Construction: Context Delimiters]
        W --> X[Generator: Local Extractive Synthesizer / OpenAI]
        X --> Y{Sufficient Context?}
        Y -- No --> Z["Refusal: 'Documents do not contain sufficient information'"]
        Y -- Yes --> AA[Answer with Verified [Citation] Tokens & Page Links]
    end
```

---

## 3. Key Technical Features

- **Multi-Format Ingestion Pipeline**:
  - Direct native parsing for **PDF** (`pypdf`), **Word** (`python-docx`), **Plain Text** (`.txt`), and **Markdown** (`.md`).
  - Safe **ZIP archive unpacking** with directory traversal prevention (`../` guards) and zip-bomb decompression limits.
  - Page-number and paragraph structure tracking throughout the entire chunking lifecycle.
- **Robust Semantic Chunking**:
  - Implements `RecursiveCharacterTextSplitter` across a hierarchy of natural boundaries (`\n\n` $\rightarrow$ `\n` $\rightarrow$ sentence periods $\rightarrow$ whitespace).
  - Deterministic chunk hash IDs (`{doc_id}_c{index}_{hash}`) preventing stale index collisions.
- **Local-First & Multi-Backend Vector Storage**:
  - **LocalVectorStore**: Built-in, high-speed vector engine written in pure Python and NumPy. Performs vectorized exact cosine matrix multiplication ($S = Q \cdot V^T$) with JSON metadata and compressed `.npz` vector persistence.
  - **Qdrant Adapter**: Preserves cloud and container-based Qdrant vector clustering for scalable distributed deployments.
- **Multi-Query Expansion & Rank Fusion**:
  - Overcomes geometric distance limitations in vector spaces via multi-perspective query reformulation fused using **Reciprocal Rank Fusion (RRF)**:
    $$RRF\_Score(d) = \sum_{q \in Q} \frac{1}{60 + r_q(d)}$$
- **Strict Anti-Injection & Hallucination Mitigation**:
  - Document text is treated strictly as untrusted data enclosed within `<context>...</context>` boundaries.
  - System enforces explicit refusal behavior when facts are absent.
- **Reproducible Evaluation Framework**:
  - Automated quantitative benchmarking script computing Information Retrieval (IR) and Generation metrics.

---

## 4. Repository Structure

```text
Chat_With_Private_Data/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated Continuous Integration (pytest + benchmark)
├── data/                          # Sample and test archives (Test File.docx, Test file.pdf)
├── evaluation/
│   ├── evaluation_results.json    # Genuine empirical benchmark measurements
│   └── evaluation_results.csv     # Query-level precision/recall/latency metrics
├── rag_core/                      # Core modular RAG package
│   ├── __init__.py
│   ├── config.py                  # Typed dataclass configuration & environment loader
│   ├── models.py                  # Core models (Document, DocumentChunk, Citation, etc.)
│   ├── loaders/                   # Ingestion, validation, and multi-format loaders
│   │   ├── base.py
│   │   ├── pdf_loader.py
│   │   ├── docx_loader.py
│   │   ├── text_loader.py
│   │   ├── zip_loader.py
│   │   └── factory.py
│   ├── chunking/                  # Semantic text splitting algorithms
│   │   ├── base.py
│   │   ├── character.py
│   │   └── recursive.py
│   ├── embeddings/                # Embedding engines (Local, HuggingFace, OpenAI)
│   │   ├── base.py
│   │   ├── local_embedding.py
│   │   ├── huggingface.py
│   │   ├── openai.py
│   │   └── factory.py
│   ├── vector_store/              # Vector database abstraction and local store
│   │   ├── base.py
│   │   ├── local_store.py
│   │   ├── qdrant_store.py
│   │   └── factory.py
│   ├── retrieval/                 # Retrieval engine & Reciprocal Rank Fusion
│   │   ├── engine.py
│   │   └── multi_query.py
│   ├── generation/                # Prompt formatting, citations, and LLM providers
│   │   ├── prompt.py
│   │   ├── providers.py
│   │   └── generator.py
│   └── evaluation/                # Quantitative benchmarking & metrics
│       ├── metrics.py
│       ├── dataset.py
│       └── benchmark.py
├── scripts/
│   └── run_evaluation.py          # Standalone benchmark execution script
├── tests/                         # Comprehensive unit & integration test suite
│   ├── conftest.py
│   ├── test_loaders.py
│   ├── test_chunking.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_security_privacy.py
├── app.py                         # Modernized Streamlit interactive application
├── data_loader.py                 # Backwards-compatible legacy module
├── text_splitters.py              # Backwards-compatible legacy module
├── embedding_models.py            # Backwards-compatible legacy module
├── vector_database.py             # Backwards-compatible legacy module
├── requirements.txt               # Pinned, verified dependencies
├── pytest.ini                     # Pytest runner configuration
├── .env.example                   # Safe template for credentials
└── README.md                      # Comprehensive academic documentation
```

---

## 5. Quantitative Evaluation & Empirical Results

To adhere to the scientific method and avoid unverified claims, this repository contains a reproducible benchmark suite (`scripts/run_evaluation.py`) tested on golden questions, reference passages, and negative out-of-domain queries.

### Measured Empirical Performance

The following results were measured directly on the local environment without external network latency:

| Metric | Measured Score | Theoretical Ideal | Definition & Methodology |
| :--- | :---: | :---: | :--- |
| **Mean Recall@3** | **1.0000** (100%) | 1.0000 | Proportion of target relevant source documents retrieved in top-3 candidate chunks. |
| **Mean Precision@3** | **0.5000** (50.0%) | 1.0000 | Proportion of retrieved chunks belonging to the golden source document across top-3 candidates. |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | 1.0000 | Multiplicative inverse of the rank of the first relevant document ($1/r$). Rank 1 across all queries. |
| **Mean Retrieval Latency** | **8.77 ms** | < 50 ms | End-to-end vector embedding generation and cosine similarity search time per query. |
| **Mean Faithfulness Score** | **0.8600** (86.0%) | 1.0000 | Lexical and semantic overlap between claims in the answer and retrieved context. |
| **Refusal Accuracy** | **1.0000** (100%) | 1.0000 | Successful refusal rate when presented with unanswerable, out-of-domain questions. |

*Detailed query-by-query breakdown and logs are exported to [evaluation_results.json](evaluation/evaluation_results.json) and [evaluation_results.csv](evaluation/evaluation_results.csv).*

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/ZeeshanAhmad85/Chat_With_Private_Data.git
cd Chat_With_Private_Data
```

### 2. Set Up Virtual Environment
```bash
# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
The system works **100% offline out-of-the-box** without any API keys. If you wish to use OpenAI models or Qdrant cloud:
```bash
cp .env.example .env
# Edit .env and supply your OPENAI_API_KEY or QDRANT_URL
```

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Step-by-Step Usage Guide

1. **Quick Reviewer Demo**:
   - Open the sidebar on the left and click **📥 Load Sample Research Corpus**.
   - Three computer science documents covering RAG architecture, GDPR data privacy, and vector indexing will be automatically indexed into memory.
2. **Asking Questions**:
   - Navigate to the **💬 Conversational Q&A** tab.
   - Enter a question:
     > *"What components combine parametric and non-parametric memory in RAG?"*
   - Observe the assistant output with inline citation tokens (`[1]`).
   - Click **📚 Grounded Sources** to inspect the exact passage, source file name, and page number.
3. **Testing Hallucination Refusal**:
   - Ask an unanswerable question outside the corpus:
     > *"What is the capital city of Australia?"*
   - The system recognizes lack of evidence and outputs:
     > *"The provided documents do not contain sufficient information to answer this question."*
4. **Uploading Your Own Files**:
   - Go to the **📤 Upload & Ingest Documents** tab.
   - Drag and drop `.pdf`, `.docx`, `.txt`, `.md`, or `.zip` files.
   - Click **🚀 Process and Index Uploaded Files**.

---

## 8. Verification & Automated Testing

The repository contains 31 automated tests covering document extraction, chunk boundaries, vector math, multi-query fusion, prompt injection, and index isolation.

Run all unit and integration tests:
```bash
pytest -v
```

Execute the empirical evaluation benchmark:
```bash
python scripts/run_evaluation.py
```

---

## 9. Security & Privacy Guarantees

- **No Accidental Data Egress**: In default offline mode, document text and queries never leave your local machine or local memory.
- **Git Hygiene**: Sensitive `.env` files, temporary vector indices, and binary caches are strictly excluded via `.gitignore`. An inert `.env.example` is provided for reference.
- **Prompt Injection Defense**: Retrieved passages from untrusted user documents are encapsulated inside explicit `<context>...</context>` delimiters with strict instruction precedence.
- **Index Isolation**: The vector database supports per-source deletion (`delete_by_source`) to respect European GDPR Article 17 ("Right to Erasure").

---

## 10. Academic Integrity & Technical Attribution

- **Original Author**: Zeeshan Ahmad
- **Original Contributions Preserved**:
  - The design of multi-format document loading (PDF, DOCX, ZIP) and UI exploration.
  - Multi-query expansion concept for vector retrieval.
- **Pretrained Models & Algorithms Attributed**:
  - **Random Indexing**: Kanerva et al. (2000), Sahlgren (2005) for dimensionality-reduced dense embeddings via Johnson-Lindenstrauss projection.
  - **Reciprocal Rank Fusion (RRF)**: Cormack, Clarke & Buettcher (SIGIR 2009).
  - **Third-Party Libraries**: Streamlit, NumPy, PyPDF, python-docx, and pytest.
  - *No claim is made of training foundation LLMs from scratch; pretrained models and APIs are integrated as modular inference components.*

---

## 11. Limitations & Future Research

1. **OCR for Scanned Documents**: Documents composed entirely of raster images or scanned photocopies without embedded font layers currently trigger an explicit warning. Incorporating Tesseract OCR or a vision-language model (e.g., Nougat or Florence-2) would expand support for legacy archives.
2. **Dense-Sparse Hybrid Indexing**: While RRF enables rank combination across sub-queries, incorporating a full inverted index (BM25 with token-level postings) would further improve lexical precision on rare domain entities.
3. **Cross-Encoder Re-Ranking**: Adding a lightweight local cross-encoder (e.g. `bge-reranker-base`) after initial candidate retrieval can improve Precision@k on large multi-thousand document collections.

---

## 12. License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for complete terms.
