"""LLM factory — returns the right model based on LLM_PROVIDER env var.

Timeline-compatible:
- Aug 2024: DeepSeek via ChatOpenAI shim (langchain-deepseek didn't exist yet)
- Feb 2025: langchain-deepseek official package (preferred, auto-detected)
- Future:   Ollama for fully local, zero-API-key runs
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
        try:
            from langchain_deepseek import ChatDeepSeek

            log.info("Using langchain-deepseek (official package)")
            return ChatDeepSeek(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                temperature=0.3,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI

            log.info("langchain-deepseek not found — using ChatOpenAI shim (Aug 2024 compat)")
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
