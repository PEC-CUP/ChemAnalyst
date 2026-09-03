from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import get_settings
from app.llm.unified_llm_client import UnifiedLLMClient
from app.rag.kb_backend import _get_searcher, invalidate_kb_searcher
from app.schemas import ExperimentalDBWorkbenchRequest, RAGPipelineRequest
from app.tools.experimental_db import SampleDatabase
from app.tools.experimental_db.evidence_retriever import (
    attach_petroleum_expert_reasoning,
    build_existing_sample_database_evidence,
)
from app.tools.experimental_db.petroleum_fraction_spectral_evidence import build_petroleum_fraction_spectral_test_evidence
from app.tools.experimental_db.petroleum_fraction_spectral_evidence import build_petroleum_fraction_spectral_workbook_evidence


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DB_WORKSPACE = ROOT / "data" / "experimental_db"


def sample_database() -> SampleDatabase:
    return SampleDatabase(SAMPLE_DB_WORKSPACE)


def rag_workbench_status(settings: Any) -> dict[str, Any]:
    searcher = _get_searcher()
    config = getattr(searcher, "config", None)
    processed = getattr(config, "processed_dir", None)
    index_dir = getattr(config, "index_dir", None)
    return {
        "status": "ready" if searcher is not None else "not_ready",
        "modes": ["naive", "qa_oriented", "iterative_review"],
        "mode_labels": {
            "naive": "Naive RAG",
            "qa_oriented": "QA-oriented RAG",
            "iterative_review": "Iterative review RAG",
        },
        "runtime": {
            "advanced_rag_enabled": bool(getattr(settings, "advanced_rag_enabled", False)),
            "rag_top_k": getattr(settings, "rag_top_k", None),
            "qa_oriented_top_docs": getattr(settings, "advanced_rag_top_docs", None),
            "qa_oriented_top_k_chunks": getattr(settings, "advanced_rag_top_k_chunks", None),
            "iterative_review_max_rounds": getattr(settings, "advanced_rag_max_rounds", None),
        },
        "kb_pipeline": {
            "chunk_mode": getattr(config, "chunk_mode", None),
            "chunk_token_size": getattr(config, "chunk_token_size", None),
            "chunk_overlap": getattr(config, "chunk_overlap", None),
            "embedding_model": getattr(config, "embedding_model_name", None),
            "embedding_local_only": getattr(config, "embedding_local_files_only", None),
            "reranker_enabled": getattr(config, "enable_reranker", None),
            "reranker_model": getattr(config, "rerank_model_name", None),
            "processed_dir": str(processed) if processed else None,
            "index_dir": str(index_dir) if index_dir else None,
        },
        "test_controls": [
            "RAG route selection: naive RAG / QA-oriented RAG / iterative review RAG",
            "Query refinement override for QA-oriented and iterative-review retrieval",
            "BM25 fusion and reranker retrieval overrides",
        ],
    }


def _kb_config() -> Any:
    from petroleum_kb.config import KBConfig

    settings = get_settings()
    config = KBConfig(project_root=settings.petroleum_kb_dir)
    if bool(getattr(settings, "petroleum_kb_embedding_local_files_only", False)):
        config.embedding_local_files_only = True
        config.rerank_local_files_only = True
    return config


def _apply_kb_pipeline_request(config: Any, request: RAGPipelineRequest) -> None:
    if request.chunk_mode:
        config.chunk_mode = request.chunk_mode
    if request.chunk_token_size is not None:
        config.chunk_token_size = request.chunk_token_size
    if request.chunk_sentence_overlap is not None:
        config.chunk_sentence_overlap = request.chunk_sentence_overlap
    if request.pdf_parser_backend:
        config.pdf_parser_backend = request.pdf_parser_backend
    if request.embedding_model:
        config.embedding_model_name = request.embedding_model
    if request.embedding_local_files_only:
        config.embedding_local_files_only = True
    if request.allow_hash_fallback:
        config.allow_hash_fallback = True


def run_rag_pipeline_action(request: RAGPipelineRequest) -> dict[str, Any]:
    from petroleum_kb.petroleum_kb.pipeline.build_index import build_vector_index
    from petroleum_kb.petroleum_kb.pipeline.ingest import ingest_registered_sources, scan_registered_sources
    from petroleum_kb.petroleum_kb.pipeline.source_registry import list_sources

    config = _kb_config()
    _apply_kb_pipeline_request(config, request)
    if request.action == "list_sources":
        result = list_sources(config)
    elif request.action == "scan_sources":
        result = scan_registered_sources(config)
    elif request.action == "ingest_sources":
        result = ingest_registered_sources(config, label_mode=request.label_mode, append=request.append)
        invalidate_kb_searcher()
    elif request.action == "build_index":
        result = build_vector_index(config.processed_dir / "chunks.jsonl", config)
        invalidate_kb_searcher()
    else:  # pragma: no cover - Pydantic constrains action names.
        raise ValueError(f"Unsupported RAG pipeline action: {request.action}")
    return {
        "status": "success",
        "action": request.action,
        "pipeline_config": {
            "source_registry_file": str(config.source_registry_file),
            "processed_file": str(config.processed_dir / "chunks.jsonl"),
            "index_dir": str(config.index_dir),
            "pdf_parser_backend": config.pdf_parser_backend,
            "chunk_mode": config.chunk_mode,
            "chunk_token_size": config.chunk_token_size,
            "chunk_sentence_overlap": config.chunk_sentence_overlap,
            "embedding_model": config.embedding_model_name,
            "embedding_local_files_only": config.embedding_local_files_only,
            "allow_hash_fallback": config.allow_hash_fallback,
        },
        "result": result,
    }


def experimental_db_status() -> dict[str, Any]:
    sample_db = sample_database()
    sample_status = sample_db.status()
    sample_examples = [item.get("sample_id") for item in sample_db.list_samples(limit=30)]
    return {
        "status": "ready" if int(sample_status.get("sample_count") or 0) > 0 else "sample_database_empty",
        "workspace": str(SAMPLE_DB_WORKSPACE.relative_to(ROOT)),
        "sample_id_examples": sample_examples,
        "feature_requirements": {
            "supported_main_evidence": [
                "sample information",
                "bulk properties as analysis-layer data",
                "GC/IR chromatographic or spectral records",
                "hydrocarbon-type composition",
                "molecular-composition records",
            ],
            "multisource_context": [
                "current-session tool outputs",
                "curated historical sample records",
                "sample relations and provenance",
            ],
            "template_governance": [
                "the active Excel template is updatable from the workbench",
                "template workbook and template definition are stored in data/experimental_db/templates",
                "new laboratories can define their own Experimental DB sheet structure by uploading a revised template",
            ],
        },
        "training_free_property_inference": {
            "policy": "Historical-neighbor evidence, inverse-distance property summaries, and evidence-guided constrained reasoning provide property support; no trained property regressor is required.",
            "main_path": "sample-centered Experimental DB evidence for Planner",
            "isolated_tests": [
                "support-layer Experimental DB workbench",
                "planner evidence sufficiency gate",
            ],
        },
        "sample_centered_database": {
            **sample_status,
            "record_policy": "Only explicit Experimental DB Workbench imports become historical records. Session uploads and tool outputs stay transient evidence.",
            "sample_layers": [
                "sample information",
                "bulk properties and distillation curve",
                "spectral/chromatographic analysis records",
                "hydrocarbon-type and molecular-composition records",
                "sample relations and provenance",
            ],
            "paper_ready_outputs": [
                "query-sample feature snapshot",
                "historical neighbor ranking table",
                "property evidence summary table",
                "manuscript-ready evidence statement",
            ],
        },
    }


def test_experimental_db_evidence(request: ExperimentalDBWorkbenchRequest, llm: Any | None = None) -> dict[str, Any]:
    evidence = build_petroleum_fraction_spectral_test_evidence(
        sample_id=request.sample_id,
        query=request.query,
        top_k=request.top_k,
    )
    if evidence is None:
        evidence = build_existing_sample_database_evidence(
            SAMPLE_DB_WORKSPACE,
            sample_id=request.sample_id,
            query=request.query,
            top_k=request.top_k,
        )
    if request.use_llm_reasoning:
        llm_client = UnifiedLLMClient()
        return attach_petroleum_expert_reasoning(evidence, llm_client)
    return evidence


def test_experimental_db_workbook_evidence(
    workbook: Path,
    *,
    query: str,
    top_k: int,
    use_llm_reasoning: bool,
    llm: Any | None = None,
) -> dict[str, Any]:
    evidence = build_petroleum_fraction_spectral_workbook_evidence(
        workbook=workbook,
        template_definition=sample_database().get_template_definition(),
        query=query,
        top_k=top_k,
    )
    if use_llm_reasoning:
        llm_client = UnifiedLLMClient()
        return attach_petroleum_expert_reasoning(evidence, llm_client)
    return evidence


