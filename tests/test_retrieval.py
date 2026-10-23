"""
Unit tests for retrieval engine and multi-query fusion.
"""

from rag_core.embeddings.local_embedding import LocalDenseEmbedding
from rag_core.models import DocumentChunk, DocumentMetadata
from rag_core.retrieval.engine import RetrievalEngine
from rag_core.retrieval.multi_query import MultiQueryRetriever
from rag_core.vector_store.local_store import LocalVectorStore


def test_retrieval_engine_top_k():
    vstore = LocalVectorStore()
    emb_model = LocalDenseEmbedding(dimension=64, seed=42)

    chunks = [
        DocumentChunk(
            chunk_id=f"c{i}",
            document_id=f"doc{i}",
            text=f"Content for chunk number {i} talking about computer science and algorithms.",
            chunk_index=i,
            metadata=DocumentMetadata(source=f"doc{i}.txt", page=1),
        )
        for i in range(5)
    ]
    embeddings = emb_model.embed_documents([c.text for c in chunks])
    vstore.add_chunks(chunks, embeddings)

    engine = RetrievalEngine(vector_store=vstore, embedding_model=emb_model)
    results = engine.retrieve("computer science algorithms", top_k=3)

    assert len(results) == 3
    for r in results:
        assert r.rank in (1, 2, 3)
        assert r.score >= 0.0


def test_retrieval_empty_query():
    vstore = LocalVectorStore()
    emb_model = LocalDenseEmbedding(dimension=64)
    engine = RetrievalEngine(vector_store=vstore, embedding_model=emb_model)
    assert engine.retrieve("") == []
    assert engine.retrieve("   ") == []


def test_multi_query_retriever_rrf():
    vstore = LocalVectorStore()
    emb_model = LocalDenseEmbedding(dimension=64, seed=42)

    c1 = DocumentChunk("c1", "d1", "Neural networks gradient descent backpropagation", 0, DocumentMetadata("nn.txt"))
    c2 = DocumentChunk("c2", "d2", "Relational database SQL indexing join algorithms", 0, DocumentMetadata("db.txt"))
    vstore.add_chunks([c1, c2], emb_model.embed_documents([c1.text, c2.text]))

    engine = RetrievalEngine(vstore, emb_model)
    mq_retriever = MultiQueryRetriever(engine)

    results = mq_retriever.retrieve("What is gradient descent in neural networks?", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "c1"
    # RRF score should be positive
    assert results[0].score > 0.0
