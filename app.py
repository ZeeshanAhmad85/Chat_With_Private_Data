"""
PrivateData-RAG: Production-Grade Document Question-Answering Application.
Features strict factual grounding, verifiable source citations, session persistence,
and local-first privacy guarantees.
"""

import os
import sys
from pathlib import Path
import time
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag_core.config import AppConfig, ChunkingConfig, EmbeddingConfig, IngestionConfig, RetrievalConfig, GenerationConfig
from rag_core.loaders.factory import DocumentLoaderFactory
from rag_core.chunking.recursive import RecursiveCharacterTextSplitter
from rag_core.chunking.character import CharacterTextSplitter
from rag_core.embeddings.factory import get_embedding_model
from rag_core.vector_store.local_store import LocalVectorStore
from rag_core.retrieval.engine import RetrievalEngine
from rag_core.retrieval.multi_query import MultiQueryRetriever
from rag_core.generation.generator import GroundedGenerator
from rag_core.generation.providers import OfflineGroundedProvider, OpenAILLMProvider
from rag_core.evaluation.dataset import BENCHMARK_DOCUMENTS
from rag_core.evaluation.benchmark import RAGBenchmarkRunner
from rag_core.models import Document, DocumentMetadata

# Set page configuration
st.set_page_config(
    page_title="PrivateData-RAG | Document Question-Answering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-privacy-local {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
        display: inline-block;
    }
    .badge-privacy-remote {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #FDE68A;
        display: inline-block;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .citation-card {
        background-color: #F1F5F9;
        border-left: 3px solid #3B82F6;
        padding: 0.6rem 0.8rem;
        margin-top: 0.4rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = LocalVectorStore()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = {}  # filename -> {chunks: int, chars: int}
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None

init_session_state()


# Sidebar: Configuration & Document Management
with st.sidebar:
    st.title("⚙️ System Control")

    # Privacy status indicator
    has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    gen_choice = st.radio(
        "LLM Generation Engine",
        ["Offline Grounded (100% Local / Zero-Cost)", "OpenAI GPT-4o-mini (Cloud API)"],
        index=0 if not has_api_key else 1,
        help="Offline mode extracts and synthesizes verbatim facts locally without internet egress. OpenAI sends retrieved context to OpenAI API."
    )
    use_openai = "OpenAI" in gen_choice

    if use_openai:
        st.markdown('<div class="badge-privacy-remote">🌐 Remote Cloud LLM Active</div>', unsafe_allow_html=True)
        if not has_api_key:
            st.warning("⚠️ No OpenAI API key detected in `.env`. Falling back to offline local provider.")
    else:
        st.markdown('<div class="badge-privacy-local">🔒 100% Local Privacy (Zero Egress)</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛠️ RAG Pipeline Settings")

    emb_choice = st.selectbox(
        "Embedding Model",
        ["Local Dense (Random Indexing 384d - Offline)", "HuggingFace BGE-small-en-v1.5", "OpenAI text-embedding-3-small"],
        index=0,
    )

    chunk_strategy = st.selectbox("Chunking Strategy", ["Recursive Character", "Character Separator"], index=0)
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=400, value=150, step=25)

    top_k = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=8, value=3)
    score_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=0.9, value=0.0, step=0.05)
    use_multi_query = st.checkbox("Enable Multi-Query Expansion (RRF)", value=False)

    st.markdown("---")
    st.subheader("📁 Document Index")
    st.write(f"**Total Indexed Chunks:** {st.session_state.vector_store.count()}")
    st.write(f"**Unique Files:** {len(st.session_state.indexed_files)}")

    if st.session_state.indexed_files:
        for fname, info in st.session_state.indexed_files.items():
            col1, col2 = st.columns([3, 1])
            col1.caption(f"📄 {fname} ({info['chunks']} chunks)")
            if col2.button("✕", key=f"del_{fname}", help=f"Delete {fname}"):
                st.session_state.vector_store.delete_by_source(fname)
                del st.session_state.indexed_files[fname]
                st.success(f"Removed {fname}")
                st.rerun()

        if st.button("🗑️ Clear Entire Index", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.indexed_files = {}
            st.session_state.all_chunks = []
            st.session_state.chat_history = []
            st.info("Vector database index cleared.")
            st.rerun()

    # Preloaded sample corpus for instant reviewer testing
    if st.button("📥 Load Sample Research Corpus", use_container_width=True, help="Loads 3 computer science & privacy research documents for instant testing."):
        with st.spinner("Indexing sample research corpus..."):
            docs = []
            for item in BENCHMARK_DOCUMENTS:
                meta = DocumentMetadata(
                    source=item["source"],
                    page=1,
                    file_type="text/plain",
                    file_size_bytes=len(item["text"]),
                )
                docs.append(Document(text=item["text"], metadata=meta))

            chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.split_documents(docs)

            emb_cfg = EmbeddingConfig(embedding_dimension=384)
            emb_model = get_embedding_model(emb_cfg)
            embeddings = emb_model.embed_documents([c.text for c in chunks])

            st.session_state.vector_store.add_chunks(chunks, embeddings)
            st.session_state.all_chunks.extend(chunks)

            for d in docs:
                file_chunks = sum(1 for c in chunks if c.metadata.source == d.metadata.source)
                st.session_state.indexed_files[d.metadata.source] = {
                    "chunks": file_chunks,
                    "chars": len(d.text),
                }

            st.success("Sample corpus indexed! You can now ask questions immediately.")
            st.rerun()


# Main Application Interface
st.markdown('<div class="main-title">PrivateData-RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Verifiable Document Question-Answering with Fact Grounding & Source Attribution</div>',
    unsafe_allow_html=True,
)

# Tab layout: Chat, Upload & Ingestion, Empirical Evaluation
tab_chat, tab_upload, tab_eval = st.tabs(["💬 Conversational Q&A", "📤 Upload & Ingest Documents", "📊 Quantitative Evaluation"])

with tab_upload:
    st.subheader("Document Ingestion & File Validation")
    st.markdown(
        "Supported formats: **PDF (`.pdf`)**, **Word (`.docx`)**, **Plain Text (`.txt`)**, "
        "**Markdown (`.md`)**, and **ZIP archives (`.zip`)**. "
        "Scanned PDFs without text layers are detected and flagged."
    )

    uploaded_files = st.file_uploader(
        "Select documents for private indexing",
        type=["pdf", "docx", "txt", "md", "zip"],
        accept_multiple_files=True,
        help="Upload files up to 25MB each. Text is processed locally or sent to your chosen LLM endpoint.",
    )

    if uploaded_files:
        if st.button("🚀 Process and Index Uploaded Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            factory = DocumentLoaderFactory()

            all_new_docs = []
            errors = []

            for i, file_obj in enumerate(uploaded_files):
                status_text.text(f"Extracting text from {file_obj.name}...")
                try:
                    docs = factory.load_file(file_obj, file_obj.name)
                    all_new_docs.extend(docs)
                except Exception as e:
                    errors.append((file_obj.name, str(e)))
                progress_bar.progress((i + 1) / len(uploaded_files) * 0.4)

            if errors:
                for fname, err in errors:
                    st.error(f"❌ Error in '{fname}': {err}")

            if all_new_docs:
                status_text.text("Segmenting documents into semantic chunks...")
                if chunk_strategy == "Recursive Character":
                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                else:
                    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                new_chunks = splitter.split_documents(all_new_docs)
                progress_bar.progress(0.7)

                status_text.text(f"Computing embeddings for {len(new_chunks)} chunks...")
                emb_cfg = EmbeddingConfig(embedding_dimension=384)
                emb_model = get_embedding_model(emb_cfg)
                embeddings = emb_model.embed_documents([c.text for c in new_chunks])
                progress_bar.progress(0.9)

                status_text.text("Persisting embeddings into vector store...")
                st.session_state.vector_store.add_chunks(new_chunks, embeddings)
                st.session_state.all_chunks.extend(new_chunks)

                for doc in all_new_docs:
                    c_count = sum(1 for c in new_chunks if c.metadata.source == doc.metadata.source)
                    st.session_state.indexed_files[doc.metadata.source] = {
                        "chunks": c_count,
                        "chars": len(doc.text),
                    }

                progress_bar.progress(1.0)
                status_text.empty()
                st.success(f"✅ Ingestion complete! Indexed {len(new_chunks)} chunks across {len(uploaded_files)} file(s).")
                st.rerun()


with tab_chat:
    col_head, col_clear = st.columns([5, 1])
    with col_head:
        if st.session_state.vector_store.count() == 0:
            st.info("💡 No documents currently indexed. Go to 'Upload & Ingest' or click 'Load Sample Research Corpus' in the sidebar to get started.")
    with col_clear:
        if st.session_state.chat_history:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 Grounded Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        page_str = f" • Page {c['page']}" if c.get('page') else ""
                        st.markdown(f"**[{c['citation_id']}] {c['source']}**{page_str}")
                        st.caption(f'"{c["snippet"]}"')
            if msg.get("latency"):
                st.caption(f"⏱️ Retrieval & Generation Latency: {msg['latency']}s • Model: {msg.get('model', 'offline')}")

    # Chat user input
    user_query = st.chat_input("Ask a question based strictly on your indexed documents...")

    if user_query:
        if st.session_state.vector_store.count() == 0:
            st.warning("Please upload or index documents first before querying.")
        else:
            # Display user question
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Retrieve and generate
            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence and synthesizing grounded response..."):
                    t_start = time.time()

                    # Retrieval setup
                    emb_cfg = EmbeddingConfig(embedding_dimension=384)
                    emb_model = get_embedding_model(emb_cfg)
                    retrieval_cfg = RetrievalConfig(top_k=top_k, score_threshold=score_threshold)
                    engine = RetrievalEngine(st.session_state.vector_store, emb_model, retrieval_cfg)

                    if use_multi_query:
                        mq_retriever = MultiQueryRetriever(engine)
                        retrieved_results = mq_retriever.retrieve(user_query, top_k=top_k)
                    else:
                        retrieved_results = engine.retrieve(user_query, top_k=top_k, score_threshold=score_threshold)

                    # Generation setup
                    if use_openai and has_api_key:
                        try:
                            provider = OpenAILLMProvider(model_name="gpt-4o-mini")
                            generator = GroundedGenerator(provider=provider)
                        except Exception as e:
                            st.error(f"OpenAI connection error: {e}. Falling back to local offline provider.")
                            generator = GroundedGenerator(provider=OfflineGroundedProvider())
                    else:
                        generator = GroundedGenerator(provider=OfflineGroundedProvider())

                    gen_result = generator.generate_answer(user_query, retrieved_results)
                    latency = round(time.time() - t_start, 3)

                    st.markdown(gen_result.answer)

                    if gen_result.citations:
                        with st.expander(f"📚 Grounded Sources ({len(gen_result.citations)})"):
                            for c in gen_result.citations:
                                page_str = f" • Page {c.page}" if c.page else ""
                                st.markdown(f"**[{c.citation_id}] {c.source}**{page_str}")
                                st.caption(f'"{c.snippet}"')

                    st.caption(f"⏱️ Retrieval & Generation Latency: {latency}s • Model: {gen_result.model_name}")

                    # Append to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": gen_result.answer,
                        "citations": [c.to_dict() for c in gen_result.citations],
                        "latency": latency,
                        "model": gen_result.model_name,
                    })


with tab_eval:
    st.subheader("Quantitative Evaluation & Reproducible Experiments")
    st.markdown(
        "Assess retrieval quality and factual grounding against verified benchmarks. "
        "Metrics computed: **Recall@k**, **Precision@k**, **Mean Reciprocal Rank (MRR)**, "
        "**Retrieval Latency**, and **Faithfulness**."
    )

    if st.button("▶️ Run Empirical Benchmark", type="primary"):
        with st.spinner("Running benchmark suite across test queries..."):
            runner = RAGBenchmarkRunner(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
            summary = runner.run_benchmark()
            st.session_state.benchmark_results = summary
            output_dir = ROOT_DIR / "evaluation"
            runner.export_results(summary, output_dir)
            st.success("Benchmark completed! Results exported to `evaluation/evaluation_results.json` and `.csv`.")

    if st.session_state.benchmark_results:
        summary = st.session_state.benchmark_results
        metrics = summary["metrics"]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Recall@k", f"{metrics['mean_recall_at_k'] * 100:.1f}%")
        col2.metric("Precision@k", f"{metrics['mean_precision_at_k'] * 100:.1f}%")
        col3.metric("MRR", f"{metrics['mean_reciprocal_rank']:.2f}")
        col4.metric("Latency", f"{metrics['mean_retrieval_latency_ms']:.1f} ms")
        col5.metric("Faithfulness", f"{metrics['mean_faithfulness'] * 100:.1f}%")

        st.subheader("Detailed Query-Level Performance")
        query_rows = []
        for q in summary["queries"]:
            query_rows.append({
                "ID": q["query_id"],
                "Question": q["question"],
                "Answerable": "Yes" if q["is_answerable"] else "No (Negative Test)",
                "Target Source": q["target_source"],
                "Recall@k": f"{q['recall@k']:.2f}",
                "Precision@k": f"{q['precision@k']:.2f}",
                "MRR": f"{q['mrr']:.2f}",
                "Latency (ms)": f"{q['retrieval_latency_ms']:.1f}",
            })
        st.dataframe(query_rows, use_container_width=True)
