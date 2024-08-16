"""LLM factory — returns the right model based on LLM_PROVIDER env var.

Aug 2024: DeepSeek accessed via ChatOpenAI compatible endpoint.
Future: Ollama support planned for fully local, zero-API-key runs.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_llm() -> BaseChatModel:
    """Return the configured LLM. Every agent calls this — never imports a provider directly."""
    provider = settings.llm_provider.lower()

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        log.info("Using DeepSeek via ChatOpenAI compatible endpoint")
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        log.info("Using Ollama (fully local)")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose 'deepseek' or 'ollama'.")
