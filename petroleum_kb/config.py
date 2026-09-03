from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class KBConfig:
    """Project-level configuration for the standalone petroleum knowledge base."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    package_root: Path = field(init=False)
    data_root: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    index_dir: Path = field(init=False)
    registry_dir: Path = field(init=False)
    source_registry_file: Path = field(init=False)
    # For academic PDFs, prefer GROBID for structure-aware extraction.
    pdf_parser_backend: str = "grobid"  # pymupdf | grobid
    grobid_url: str = "http://localhost:8070"
    grobid_timeout_seconds: int = 300
    chunk_mode: str = "scientific_sentence"  # char | sentence | scientific_sentence
    chunk_size: int = 900
    chunk_overlap: int = 150
    chunk_token_size: int = 384
    chunk_sentence_overlap: int = 2
    scientific_sentence_model: str = "en_core_web_sm"
    # Strict controls (no fallback):
    # - PDF must be parsed by GROBID (no PyMuPDF fallback)
    # - scientific_sentence must be split by spaCy model (no regex fallback)
    strict_pdf_parser: bool = True
    strict_scientific_sentence: bool = True
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_local_files_only: bool = False
    embedding_fallback_dim: int = 384
    allow_hash_fallback: bool = False
    index_name: str = "petroleum_knowledge"
    # Retrieval / reranking (stage-2):
    retrieval_top_n: int = 60  # larger candidate pool helps recover the 2nd gold doc in multi-doc QA
    enable_reranker: bool = True
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_local_files_only: bool = False
    rerank_max_chars: int = 1800
    # Semantic benchmark oriented retrieval improvements:
    enable_bm25: bool = True
    vector_weight: float = 0.65
    bm25_weight: float = 0.35
    bm25_persist: bool = True
    bm25_index_file: Path = field(init=False)
    # Document-level evidence aggregation:
    enable_doc_aggregation: bool = False
    docs_top_k: int = 5
    chunks_per_doc: int = 3
    enable_clustering: bool = False
    # Final ranking:
    # score = alpha * semantic_similarity + beta * tag_match + gamma * source_weight
    # The current production KB is literature-only and generated tags are not
    # reliable enough for ranking, so runtime retrieval uses semantic/rerank
    # scores only by default.
    alpha: float = 1.00
    beta: float = 0.00
    gamma: float = 0.00
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "text": 1.00,
            "text": 0.92,
            "text": 0.85,
            "text": 0.75,
            "text": 0.72,
            "text": 0.60,
            "text": 0.50,
        }
    )

    def __post_init__(self) -> None:
        self.package_root = self.project_root / "petroleum_kb"
        self.data_root = self.package_root / "data"
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.metadata_dir = self.data_root / "metadata"
        self.index_dir = self.data_root / "index"
        self.registry_dir = self.data_root / "registry"
        self.source_registry_file = self.registry_dir / "sources.json"
        self.bm25_index_file = self.index_dir / f"{self.index_name}.bm25.pkl"


DEFAULT_CONFIG = KBConfig()
