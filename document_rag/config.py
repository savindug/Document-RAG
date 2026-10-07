"""Environment-backed application configuration."""

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig(BaseModel):
    """Configuration shared by ingestion and question pipelines."""

    model_config = ConfigDict(frozen=True)

    google_api_key: SecretStr | None = None
    unstructured_api_key: SecretStr | None = None
    unstructured_api_url: str | None = None
    gemini_model: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    gemini_embedding_model: str = Field(default="gemini-embedding-2", min_length=1)
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_candidate_pool: int = Field(default=20, ge=1, le=200)
    query_decomposition_enabled: bool = True
    max_subqueries: int = Field(default=3, ge=1, le=5)
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    max_files: int = Field(default=10, ge=1)
    max_file_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_chunks: int = Field(default=200, ge=1)
    chunk_max_chars: int = Field(default=2000, ge=100)
    chunk_target_chars: int = Field(default=1400, ge=50)
    chunk_min_chars: int = Field(default=80, ge=1)

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "AppConfig":
        if not self.chunk_min_chars <= self.chunk_target_chars <= self.chunk_max_chars:
            raise ValueError("Chunk sizes must satisfy min <= target <= max.")
        return self

    @field_validator("google_api_key", "unstructured_api_key", mode="before")
    @classmethod
    def blank_key_is_missing(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def require_google_api_key(self) -> str:
        """Return the key when an API call is needed, with a setup hint if absent."""
        if self.google_api_key is None:
            raise ValueError("Set GOOGLE_API_KEY in .env before using Gemini.")
        return self.google_api_key.get_secret_value()

    @field_validator("unstructured_api_url", mode="before")
    @classmethod
    def blank_url_is_missing(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        if value is not None:
            parsed = urlparse(value)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.hostname in {"transform.unstructured.io", "mcp.transform.unstructured.io"}
                or parsed.path.rstrip("/") not in {"", "/general/v0/general"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "UNSTRUCTURED_API_URL must be an HTTPS Partition API base URL "
                    "or end in /general/v0/general; Transform URLs are not compatible."
                )
        return value


def load_config() -> AppConfig:
    """Load local .env values without overriding existing environment variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_url = os.getenv("UNSTRUCTURED_API_URL")
    if unstructured_api_url_warning(api_url):
        api_url = None
    return AppConfig(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        unstructured_api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        unstructured_api_url=api_url,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
        retrieval_top_k=os.getenv("TOP_K", "5"),
        retrieval_candidate_pool=os.getenv("RETRIEVAL_CANDIDATES", "20"),
        query_decomposition_enabled=os.getenv("QUERY_DECOMPOSITION_ENABLED", "true"),
        max_subqueries=os.getenv("MAX_SUBQUERIES", "3"),
        rerank_enabled=os.getenv("RERANK_ENABLED", "true"),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
    )


def unstructured_api_url_warning(value: str | None) -> str | None:
    """Explain why an environment URL cannot be used for hosted partitioning."""
    try:
        AppConfig.blank_url_is_missing(value)
    except ValueError:
        return (
            "UNSTRUCTURED_API_URL is not a Partition API endpoint. "
            "Using local Unstructured parsing. Set a Partition API URL to enable hosted parsing."
        )
    return None
