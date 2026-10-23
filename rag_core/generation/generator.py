"""
Master response generation coordinator with citation mapping and telemetry.
"""

import time
from typing import List, Optional
from rag_core.config import GenerationConfig
from rag_core.models import Citation, GenerationResult, RetrievalResult
from rag_core.generation.prompt import SYSTEM_PROMPT, format_grounded_prompt
from rag_core.generation.providers import BaseLLMProvider, OfflineGroundedProvider, OpenAILLMProvider


class GroundedGenerator:
    """Coordinates prompt construction, LLM execution, citation extraction, and latency metrics."""

    REFUSAL_PHRASE = "The provided documents do not contain sufficient information"

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        config: Optional[GenerationConfig] = None,
    ):
        self.config = config or GenerationConfig()
        if provider:
            self.provider = provider
        else:
            self.provider = self._init_provider()

    def _init_provider(self) -> BaseLLMProvider:
        if self.config.provider == "openai":
            return OpenAILLMProvider(
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )
        # Default fallback
        return OfflineGroundedProvider()

    def generate_answer(
        self, query: str, retrieved_chunks: List[RetrievalResult]
    ) -> GenerationResult:
        start_time = time.time()

        if not retrieved_chunks:
            return GenerationResult(
                answer=f"{self.REFUSAL_PHRASE} to answer this question.",
                citations=[],
                retrieved_chunks=[],
                latency_seconds=round(time.time() - start_time, 4),
                model_name=self.config.model_name,
                grounded=True,
                insufficient_context=True,
            )

        user_prompt, citations = format_grounded_prompt(query, retrieved_chunks)

        try:
            raw_answer = self.provider.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            raw_answer = f"Error during generation: {str(e)}"

        latency = round(time.time() - start_time, 4)
        is_refusal = self.REFUSAL_PHRASE.lower() in raw_answer.lower()

        # Filter citations to only those referenced in raw_answer (e.g. "[1]")
        active_citations: List[Citation] = []
        if not is_refusal:
            for c in citations:
                token = f"[{c.citation_id}]"
                if token in raw_answer:
                    active_citations.append(c)

        # If model didn't include tokens or offline provider, include top chunk citation
        if not active_citations and not is_refusal and citations:
            active_citations = [citations[0]]

        return GenerationResult(
            answer=raw_answer,
            citations=active_citations,
            retrieved_chunks=retrieved_chunks,
            latency_seconds=latency,
            model_name=self.config.model_name,
            grounded=True,
            insufficient_context=is_refusal,
        )
