"""
Legacy vector database adapter for backwards compatibility.
Supports local persistence and Qdrant cloud with graceful fallback.
"""

import os
from dotenv import load_dotenv
from rag_core.vector_store.local_store import LocalVectorStore
from rag_core.vector_store.qdrant_store import QdrantVectorStoreAdapter

load_dotenv()


def vector_database(docs, embeddings, url=None, api_key=None, collection_name="my_collection"):
    """
    Initialize vector store and index documents.
    Defaults to LocalVectorStore if Qdrant URL is unset or placeholder.
    """
    qdrant_url = url or os.getenv("QDRANT_URL") or os.getenv("DB_URL")
    db_api_key = api_key or os.getenv("DB_API_KEY")

    if qdrant_url and "YOUR DATABASE" not in qdrant_url and qdrant_url.startswith("http"):
        try:
            store = QdrantVectorStoreAdapter(
                url=qdrant_url,
                api_key=db_api_key,
                collection_name=collection_name,
                dimension=getattr(embeddings, "dimension", 384),
            )
            # Embed and index
            texts = [d.text for d in docs]
            emb_vectors = embeddings.embed_documents(texts)
            store.add_chunks(docs, emb_vectors)
            return store
        except Exception as e:
            print(f"Warning: Failed to connect to Qdrant ({e}). Falling back to LocalVectorStore.")

    # Local vector store fallback
    store = LocalVectorStore()
    if docs:
        texts = [d.text for d in docs]
        if hasattr(embeddings, "embed_documents"):
            emb_vectors = embeddings.embed_documents(texts)
        else:
            # Fallback
            from rag_core.embeddings.local_embedding import LocalDenseEmbedding
            emb_vectors = LocalDenseEmbedding().embed_documents(texts)
        store.add_chunks(docs, emb_vectors)

    return store
