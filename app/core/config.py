from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemma_model: str = Field(default="gemma-4-26b-a4b-it", alias="GEMMA_MODEL")
    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="gemma4hack-study-chunks", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    pinecone_embedding_model: str = Field(default="llama-text-embed-v2", alias="PINECONE_EMBEDDING_MODEL")
    rag_top_k: int = Field(default=5, ge=1, le=20, alias="RAG_TOP_K")
    rag_keyword_top_k: int = Field(default=4, ge=1, le=20, alias="RAG_KEYWORD_TOP_K")
    rag_semantic_timeout_seconds: float = Field(default=8.0, ge=0.5, le=60.0, alias="RAG_SEMANTIC_TIMEOUT_SECONDS")
    rag_chunk_chars: int = Field(default=1800, ge=500, le=5000, alias="RAG_CHUNK_CHARS")
    rag_chunk_overlap_chars: int = Field(default=250, ge=0, le=1000, alias="RAG_CHUNK_OVERLAP_CHARS")
    cors_allowed_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ALLOWED_ORIGINS",
    )
    max_upload_mb: int = Field(default=25, ge=1, le=100, alias="MAX_UPLOAD_MB")
    max_pdf_pages: int = Field(default=80, ge=1, le=500, alias="MAX_PDF_PAGES")
    max_extracted_chars: int = Field(default=120_000, ge=1_000, le=1_000_000, alias="MAX_EXTRACTED_CHARS")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
