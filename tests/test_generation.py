"""
Unit tests for prompt formatting, grounding, and generator citation synthesis.
"""

from rag_core.generation.generator import GroundedGenerator
from rag_core.generation.prompt import format_grounded_prompt
from rag_core.generation.providers import OfflineGroundedProvider
from rag_core.models import DocumentChunk, DocumentMetadata, RetrievalResult


def test_prompt_formatting_and_citations():
    chunk = DocumentChunk(
        chunk_id="chunk_42",
        document_id="doc_1",
        text="GDPR Article 17 grants individuals the right to have personal data erased.",
        chunk_index=0,
        metadata=DocumentMetadata(source="gdpr_law.pdf", page=5),
    )
    result = RetrievalResult(chunk=chunk, score=0.92, rank=1)

    prompt, citations = format_grounded_prompt("What is GDPR Article 17?", [result])

    assert "<context>" in prompt
    assert "</context>" in prompt
    assert "[1] Source: gdpr_law.pdf, Page 5" in prompt
    assert len(citations) == 1
    assert citations[0].citation_id == 1
    assert citations[0].source == "gdpr_law.pdf"
    assert citations[0].page == 5


def test_grounded_generator_offline_success():
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text="The Fourier Transform decomposes a function of time into frequencies.",
        chunk_index=0,
        metadata=DocumentMetadata(source="dsp.txt", page=1),
    )
    res = RetrievalResult(chunk=chunk, score=0.95, rank=1)

    generator = GroundedGenerator(provider=OfflineGroundedProvider())
    gen_result = generator.generate_answer("What does the Fourier Transform decompose?", [res])

    assert gen_result.grounded is True
    assert gen_result.insufficient_context is False
    assert "[1]" in gen_result.answer
    assert "Fourier Transform" in gen_result.answer
    assert len(gen_result.citations) == 1


def test_grounded_generator_empty_context_refusal():
    generator = GroundedGenerator(provider=OfflineGroundedProvider())
    gen_result = generator.generate_answer("What is the capital of Mars?", [])

    assert gen_result.insufficient_context is True
    assert "not contain sufficient information" in gen_result.answer
    assert len(gen_result.citations) == 0
