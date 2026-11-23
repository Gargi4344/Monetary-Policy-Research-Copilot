"""Settings from environment variables (documented in .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration. Defaults run fully locally and free."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # LLM through LiteLLM. Default: local Ollama. Paid providers are optional, e.g.
    # LLM_MODEL=openai/gpt-4o-mini with OPENAI_API_KEY, or anthropic/... with ANTHROPIC_API_KEY.
    llm_model: str = "ollama/qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 180.0
    llm_temperature: float = 0.0

    # Retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    finbert_model: str = "ProsusAI/finbert"
    chunk_tokens: int = 500
    chunk_overlap: float = 0.15
    retrieve_candidates: int = 20
    retrieve_top_k: int = 5
    collection: str = "boc_chunks"

    # Scraper
    scrape_from_year: int = 2015
    scrape_delay_seconds: float = 1.5
    user_agent: str = (
        "monetary-policy-research-copilot/0.1 (academic research; respects robots.txt)"
    )

    # SQL tool
    sql_max_rows: int = 500

    @property
    def cache_dir(self) -> Path:
        """Raw HTML/PDF cache (gitignored)."""
        return self.data_dir / "cache"

    @property
    def docs_path(self) -> Path:
        """Extracted documents, one JSON object per line."""
        return self.data_dir / "docs.jsonl"

    @property
    def chunks_path(self) -> Path:
        """Chunks with metadata, one JSON object per line."""
        return self.data_dir / "chunks.jsonl"

    @property
    def qdrant_path(self) -> Path:
        """Embedded Qdrant storage (no server needed)."""
        return self.data_dir / "qdrant"

    @property
    def bm25_path(self) -> Path:
        """Pickled BM25 index."""
        return self.data_dir / "bm25.pkl"

    @property
    def duckdb_path(self) -> Path:
        """Market data and app tables."""
        return self.data_dir / "copilot.duckdb"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings object."""
    return Settings()
