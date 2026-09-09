"""Swappable LLM provider interface protocol and client implementations."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class interface for swappable LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion from the LLM provider for a given prompt."""
        pass


class GroqProvider(LLMProvider):
    """Groq LLM provider implementation stub."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion using the Groq API."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation stub."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion using the Google Gemini API."""
        raise NotImplementedError
