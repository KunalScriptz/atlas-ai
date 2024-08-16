"""Centralized settings via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──
    llm_provider: str = "deepseek"  # "deepseek" | "ollama"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # Ollama
    ollama_model: str = "qwen3:14b"
    ollama_base_url: str = "http://localhost:11434"

    # ── Milvus ──
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "atlas_documents"

    # ── Redis ──
    redis_uri: str = "redis://localhost:6379"

    # ── Embeddings ──
    embedding_model: str = "BAAI/bge-m3"

    # ── App ──
    api_port: int = 9734
    streamlit_port: int = 8501
    log_level: str = "INFO"


settings = Settings()
