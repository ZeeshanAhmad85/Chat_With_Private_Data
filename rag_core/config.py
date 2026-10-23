"""
Configuration and settings management for RAG pipeline.
Supports environment variables with sensible local-first defaults.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class IngestionConfig:
    """Document ingestion constraints and thresholds."""
    max_file_size_bytes: int = 25 * 1024 * 1024  # 25 MB
    allowed_extensions: tuple = (".pdf", ".docx", ".txt", ".md", ".zip")
    extract_page_numbers: bool = True
    normalize_unicode: bool = True
    ocr_enabled: bool = False  # Explicit flag; scanned documents will be rejected gracefully unless OCR is configured


@dataclass
class ChunkingConfig:
    """Text chunking parameters."""
    strategy: str = "recursive"  # "recursive" or "character"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str = "local"  # "local", "huggingface", "openai"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    embedding_dimension: int = 384
    normalize_embeddings: bool = True


@dataclass
class VectorStoreConfig:
    """Vector database persistence configuration."""
    store_type: str = "local"  # "local" (in-memory/persisted JSON/NumPy) or "qdrant"
    persist_directory: Path = field(default_factory=lambda: Path("./data/vector_store"))
    qdrant_url: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_URL") or os.getenv("DB_URL"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DB_API_KEY") or os.getenv("QDRANT_API_KEY"))
    collection_name: str = "chat_private_docs"


@dataclass
class RetrievalConfig:
    """Retrieval parameters and thresholds."""
    top_k: int = 3
    score_threshold: float = 0.0  # Minimum cosine similarity (0 to 1)
    multi_query: bool = False
    num_queries: int = 3
    hybrid_search: bool = False  # Dense + lexical (BM25)


@dataclass
class GenerationConfig:
    """LLM generation and safety parameters."""
    provider: str = "auto"  # "auto", "openai", "offline"
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    temperature: float = 0.2  # Low temperature for factual grounding
    max_tokens: int = 1024
    request_timeout: float = 30.0
    require_citations: bool = True
    local_privacy_mode: bool = False  # If True, never sends text to external APIs


@dataclass
class AppConfig:
    """Master configuration encompassing all pipeline stages."""
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Instantiate configuration from environment variables."""
        config = cls()
        # Detect if OpenAI API key is present and valid
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        if not api_key or "your" in api_key.lower():
            config.generation.provider = "offline"
        else:
            config.generation.provider = "openai"
        return config
