"""LLM abstraction layer — swap providers without touching agent code."""

from src.llm.factory import get_llm

__all__ = ["get_llm"]
