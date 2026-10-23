"""
LLM providers: OpenAI and Local Offline Grounded synthesis engine.
"""

from abc import ABC, abstractmethod
import os
import re
from typing import List, Optional
from rag_core.models import RetrievalResult


class BaseLLMProvider(ABC):
    """Abstract interface for LLM response generation."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """Generates answers via OpenAI Chat Completions API."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if not self.api_key or "your" in self.api_key.lower():
            raise ValueError(
                "A valid OpenAI API key must be set in OPENAI_API_KEY environment variable. "
                "For local offline evaluation, switch generator provider to 'offline'."
            )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}") from e


class OfflineGroundedProvider(BaseLLMProvider):
    """
    Extractive grounding engine for local, deterministic, zero-cost execution.
    Extracts high-salience sentences from retrieved contexts directly answering the query,
    enforcing citation tags without external LLM API dependency.
    """

    def __init__(self):
        pass

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Extract question from user_prompt
        q_match = re.search(r"Question:\s*(.*?)\s*Answer", user_prompt, re.DOTALL)
        query = q_match.group(1).strip() if q_match else ""

        # Extract context blocks
        c_match = re.search(r"<context>(.*?)</context>", user_prompt, re.DOTALL)
        context_str = c_match.group(1).strip() if c_match else ""

        if not context_str or not query:
            return "The provided documents do not contain sufficient information to answer this question."

        # Parse excerpts by citation token [X]
        pattern = r"\[(\d+)\] Source: (.*?)\n(.*?)(?=\n\n\[\d+\]|\Z)"
        matches = re.findall(pattern, context_str, re.DOTALL)

        if not matches:
            return "The provided documents do not contain sufficient information to answer this question."

        STOPWORDS = {
            "what", "where", "when", "which", "does", "have", "this", "that", "from",
            "the", "and", "for", "with", "are", "can", "why", "how", "used", "city"
        }
        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower())) - STOPWORDS
        min_overlap = max(1, min(2, len(q_terms)))

        best_sentences = []
        for cite_id, source, text in matches:
            # Split text into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r"\b\w{3,}\b", s_clean.lower()))
                overlap = len(q_terms.intersection(s_words))
                if overlap >= min_overlap:
                    best_sentences.append((overlap, cite_id, s_clean))

        if not best_sentences:
            return "The provided documents do not contain sufficient information to answer this question."

        # Sort sentences descending by overlap score
        best_sentences.sort(key=lambda x: x[0], reverse=True)
        top_picks = best_sentences[:2]

        answer_parts = []
        for _, cite_id, s_text in top_picks:
            # Ensure sentence ends with citation
            clean_s = s_text.rstrip(".")
            answer_parts.append(f"{clean_s} [{cite_id}].")

        return " ".join(answer_parts)
