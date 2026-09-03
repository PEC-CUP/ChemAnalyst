from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ChemAnalyst"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_trust_env: bool = Field(default=False, alias="LLM_TRUST_ENV")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")

    # Backward-compatible aliases for older local .env files and offline scripts.
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_trust_env: bool = Field(default=False, alias="DEEPSEEK_TRUST_ENV")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="", alias="OPENAI_MODEL")
    openai_strong_model: str = Field(default="", alias="OPENAI_STRONG_MODEL")
    openai_balanced_model: str = Field(default="", alias="OPENAI_BALANCED_MODEL")
    openai_fast_model: str = Field(default="", alias="OPENAI_FAST_MODEL")
    openai_cheap_model: str = Field(default="", alias="OPENAI_CHEAP_MODEL")
    tesseract_cmd: str = Field(default="", alias="TESSERACT_CMD")
    tessdata_prefix: str = Field(default="", alias="TESSDATA_PREFIX")

    knowledge_dir: Path = BASE_DIR / "knowledge"
    knowledge_extra_dirs_raw: str = Field(default="", alias="KNOWLEDGE_EXTRA_DIRS")
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    petroleum_kb_enabled: bool = Field(default=True, alias="PETROLEUM_KB_ENABLED")
    petroleum_kb_dir: Path = BASE_DIR / "petroleum_kb"
    petroleum_kb_embedding_local_files_only: bool = Field(
        default=False,
        alias="PETROLEUM_KB_EMBEDDING_LOCAL_FILES_ONLY",
        description="Force petroleum_kb embedder to load models from local cache only (no HF network).",
    )
    petroleum_kb_enable_reranker: bool = Field(
        default=True,
        alias="PETROLEUM_KB_ENABLE_RERANKER",
        description="Enable petroleum_kb cross-encoder reranker.",
    )
    petroleum_kb_rerank_model: str = Field(
        default="",
        alias="PETROLEUM_KB_RERANK_MODEL",
        description="Override reranker model name for petroleum_kb.",
    )
    petroleum_kb_rerank_local_files_only: bool = Field(
        default=False,
        alias="PETROLEUM_KB_RERANK_LOCAL_FILES_ONLY",
        description="Force petroleum_kb reranker to load models from local cache only (no HF network).",
    )
    vector_dim: int = 256
    rag_top_k: int = 4
    chunk_max_chars: int = 800
    advanced_rag_enabled: bool = Field(default=False, alias="ADVANCED_RAG_ENABLED")
    advanced_rag_max_rounds: int = Field(default=2, alias="ADVANCED_RAG_MAX_ROUNDS")
    advanced_rag_top_docs: int = Field(default=5, alias="ADVANCED_RAG_TOP_DOCS")
    advanced_rag_top_k_chunks: int = Field(default=12, alias="ADVANCED_RAG_TOP_K_CHUNKS")
    qa_oriented_initial_top_docs: int = Field(default=12, alias="QA_ORIENTED_INITIAL_TOP_DOCS")
    qa_oriented_initial_top_k_chunks: int = Field(default=30, alias="QA_ORIENTED_INITIAL_TOP_K_CHUNKS")
    qa_oriented_chunks_per_doc: int = Field(default=4, alias="QA_ORIENTED_CHUNKS_PER_DOC")
    rag_evidence_filter_model: str = Field(default="", alias="RAG_EVIDENCE_FILTER_MODEL")
    rag_evidence_filter_thinking: bool = Field(default=False, alias="RAG_EVIDENCE_FILTER_THINKING")
    rag_evidence_filter_batch_size: int = Field(default=6, alias="RAG_EVIDENCE_FILTER_BATCH_SIZE")
    rag_evidence_filter_min_score: float = Field(default=6.0, alias="RAG_EVIDENCE_FILTER_MIN_SCORE")
    rag_evidence_filter_top_k: int = Field(default=10, alias="RAG_EVIDENCE_FILTER_TOP_K")
    rag_evidence_filter_max_chars: int = Field(default=1800, alias="RAG_EVIDENCE_FILTER_MAX_CHARS")

    def knowledge_source_dirs(self) -> list[Path]:
        raw_items = [item.strip() for item in self.knowledge_extra_dirs_raw.split(";") if item.strip()]
        dirs = [self.knowledge_dir]
        seen: set[str] = set()

        for item in raw_items:
            path = Path(item).expanduser()
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(path)

        return dirs


    @property
    def active_llm_api_key(self) -> str:
        return (self.llm_api_key or self.deepseek_api_key or self.openai_api_key or "").strip()

    @property
    def active_llm_base_url(self) -> str:
        return (self.llm_base_url or self.deepseek_base_url or self.openai_base_url or "").strip()

    @property
    def active_llm_model(self) -> str:
        return (self.llm_model or self.openai_model or "").strip()

    @property
    def active_llm_trust_env(self) -> bool:
        return bool(self.llm_trust_env or self.deepseek_trust_env)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    return settings
