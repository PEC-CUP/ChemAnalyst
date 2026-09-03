from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalOptions(BaseModel):
    """Per-request retrieval overrides for the ChemAnalyst RAG backend.

    When omitted, the backend uses petroleum_kb's default config.
    """

    enable_refinement_agent: bool | None = Field(
        default=None,
        description="Enable LLM Query Refinement Agent (multi-query generation handled outside retrieval).",
    )
    refinement_max_queries: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Max number of queries produced by the Query Refinement Agent.",
    )

    enable_bm25: bool | None = Field(default=None, description="Enable BM25 keyword recall route.")
    vector_weight: float | None = Field(default=None, ge=0.0, le=1.0, description="Vector route weight.")
    bm25_weight: float | None = Field(default=None, ge=0.0, le=1.0, description="BM25 route weight.")

    enable_reranker: bool | None = Field(default=None, description="Enable cross-encoder reranker.")
    rerank_model: str | None = Field(default=None, description="Reranker model name (HF).")
    rerank_local_files_only: bool | None = Field(default=None, description="Load reranker from local cache only.")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User input query")
    task_type: Literal["auto", "chat", "rag", "tool", "reasoning"] = Field(
        default="auto",
        description="Optional frontend override for routing mode",
    )
    rag_mode: Literal[
        "naive",
        "qa_oriented",
        "iterative_review",
    ] = Field(
        default="qa_oriented",
        description="RAG route selector when task_type routes to rag: naive, QA-oriented, or iterative review.",
    )
    retrieval: RetrievalOptions | None = Field(
        default=None,
        description="Optional per-request retrieval overrides (multi-query/BM25/rerank).",
    )
    session_id: str | None = Field(default=None, description="Conversation session id")


class PlannerQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User input query")
    task_type: Literal["auto", "chat", "rag", "tool", "reasoning"] = Field(
        default="auto",
        description="Optional fallback runtime mode",
    )
    rag_mode: Literal[
        "naive",
        "qa_oriented",
        "iterative_review",
    ] = Field(
        default="qa_oriented",
        description="RAG route selector when planner routes to rag: naive, QA-oriented, or iterative review.",
    )
    retrieval: RetrievalOptions | None = Field(
        default=None,
        description="Optional per-request retrieval overrides (multi-query/BM25/rerank).",
    )
    template_override: Literal["auto", "concept_qa", "knowledge_qa", "data_analysis", "processing_evaluation", "hybrid_analysis"] = Field(
        default="auto",
        description="Optional planner template override",
    )
    session_id: str | None = Field(default=None, description="Conversation session id")


class QueryResponse(BaseModel):
    session_id: str
    task_type: str
    used_model: str
    used_tools: list[str]
    answer: str
    raw_result: Any


class HealthResponse(BaseModel):
    status: str
    app_name: str


class ToolResponse(BaseModel):
    status: str
    result: Any
    message: str
    session_id: str | None = None
    answer: str | None = None


class ToolOnboardingRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, description="Human-readable analytical tool name")
    description: str = Field(..., min_length=1, description="Functional description for capability conversion")
    invocation_type: Literal["python_script", "http_api", "python_callable", "manual"] = Field(
        default="manual",
        description="How the tool will eventually be invoked after adapter review",
    )
    entrypoint: str | None = Field(
        default=None,
        description="Script path, import path, or API URL. It is not executed by onboarding.",
    )
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    evidence_outputs: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    artifacts_produced: list[str] = Field(default_factory=list)
    example_queries: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    upstream_capabilities: list[str] = Field(default_factory=list)
    downstream_capabilities: list[str] = Field(default_factory=list)
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    output_bindings: dict[str, Any] = Field(default_factory=dict)
    database_integration_contract: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional record-channel contract for persisting structured tool outputs into Experimental DB.",
    )
    adapter_template: str | None = Field(
        default=None,
        description="Optional generic adapter template, e.g. generic_python_function.",
    )
    adapter_config: dict[str, Any] = Field(default_factory=dict)


class ToolOnboardingResponse(BaseModel):
    status: str
    package: dict[str, Any]
    message: str


class CapabilityProviderSaveRequest(BaseModel):
    provider_id: str | None = Field(default=None)
    package: dict[str, Any]


class CapabilityProviderEvalRequest(BaseModel):
    execute: bool = Field(default=False)
    sample_payload: dict[str, Any] = Field(default_factory=dict)


class CapabilityProviderReviewRequest(BaseModel):
    reviewed: bool = Field(default=True)
    enabled: bool = Field(default=True)
    notes: str | None = Field(default=None)


class CapabilityProviderRevokeRequest(BaseModel):
    confirm: bool = Field(default=False)
    reason: str | None = Field(default=None)


class CapabilityProviderResponse(BaseModel):
    status: str
    result: dict[str, Any]
    message: str


class BenchmarkGenerationRequest(BaseModel):
    benchmark_name: str = Field(default="benchmark.ui")
    write_mode: Literal["new", "append"] = Field(default="new")
    benchmark_mode: Literal["reviewed", "naive_baseline"] = Field(default="reviewed")
    source_mode: Literal["existing_chunks", "pdf_directory"] = Field(default="existing_chunks")
    chunks_path: str | None = Field(default=None)
    pdf_dir: str | None = Field(default=None)
    pdf_parser_backend: Literal["grobid", "pymupdf"] = Field(default="grobid")
    chunk_mode: Literal["char", "sentence", "scientific_sentence"] = Field(default="scientific_sentence")
    benchmark_style: Literal["semantic", "specific_fact"] = Field(default="semantic")
    answer_format: Literal["open", "mcq"] = Field(default="open")
    question_count: int = Field(default=10, ge=1, le=200)
    language: Literal["en", "zh"] = Field(default="en")
    single_doc_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    two_doc_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    seed: int = Field(default=7)


class EvaluationWorkbenchRequest(BaseModel):
    level: Literal["planner_sufficiency", "task_specific_schema", "agent_workflow"] = Field(
        default="planner_sufficiency"
    )
    case_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)


class RAGWorkbenchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    rag_mode: Literal["naive", "qa_oriented", "iterative_review"] = Field(default="qa_oriented")
    top_k: int | None = Field(default=None, ge=1, le=30)
    enable_refinement_agent: bool | None = Field(default=None)
    enable_bm25: bool | None = Field(default=None)
    enable_reranker: bool | None = Field(default=None)


class RAGPipelineRequest(BaseModel):
    action: Literal["list_sources", "scan_sources", "ingest_sources", "build_index"] = Field(
        default="list_sources"
    )
    chunk_mode: Literal["char", "sentence", "scientific_sentence"] | None = Field(default=None)
    chunk_token_size: int | None = Field(default=None, ge=32, le=4096)
    chunk_sentence_overlap: int | None = Field(default=None, ge=0, le=20)
    label_mode: Literal["none", "rule", "llm", "hybrid"] = Field(default="hybrid")
    append: bool = Field(default=True)
    pdf_parser_backend: Literal["pymupdf", "grobid"] | None = Field(default=None)
    embedding_model: str | None = Field(default=None)
    embedding_local_files_only: bool = Field(default=True)
    allow_hash_fallback: bool = Field(default=False)


class KBConstructionRequest(BaseModel):
    source_dir: str = Field(..., min_length=1)
    build_name: str = Field(default="kb_build.ui", min_length=1)
    pdf_parser_backend: Literal["pymupdf", "grobid"] = Field(default="grobid")
    chunk_mode: Literal["char", "sentence", "scientific_sentence"] = Field(default="scientific_sentence")
    chunk_token_size: int = Field(default=384, ge=32, le=4096)
    chunk_sentence_overlap: int = Field(default=2, ge=0, le=20)
    label_mode: Literal["none", "rule", "llm", "hybrid"] = Field(default="none")
    build_index: bool = Field(default=True)
    build_keyword_graph: bool = Field(default=True)
    embedding_model: str | None = Field(default=None)
    embedding_local_files_only: bool = Field(default=True)
    allow_hash_fallback: bool = Field(default=False)


class ExperimentalDBWorkbenchRequest(BaseModel):
    sample_id: str = Field(..., min_length=1)
    query: str = Field(default="Infer petroleum properties from available historical experimental evidence.")
    top_k: int = Field(default=4, ge=1, le=50)
    use_llm_reasoning: bool = Field(default=True)


class ExperimentalSampleImportRequest(BaseModel):
    payload: dict[str, Any] = Field(
        ...,
        description="Sample-centered JSON payload with sample_information, bulk_properties, analyses, and optional relations.",
    )
