import json
from io import BytesIO
from pathlib import Path
from shutil import copyfileobj
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agents import ChemAnalyst
from app.agents.orchestration_agent import OrchestrationAgent
from app.agents.task_classifier import classify_front_door
from app.config import get_settings
from app.rag.kb_backend import retrieve
from app.schemas import (
    BenchmarkGenerationRequest,
    CapabilityProviderEvalRequest,
    CapabilityProviderResponse,
    CapabilityProviderRevokeRequest,
    CapabilityProviderReviewRequest,
    CapabilityProviderSaveRequest,
    EvaluationWorkbenchRequest,
    ExperimentalDBWorkbenchRequest,
    ExperimentalSampleImportRequest,
    HealthResponse,
    KBConstructionRequest,
    PlannerQueryRequest,
    QueryRequest,
    QueryResponse,
    RAGPipelineRequest,
    RAGWorkbenchRequest,
    ToolOnboardingRequest,
    ToolOnboardingResponse,
    ToolResponse,
)
from app.benchmark_workbench import (
    get_benchmark_generation_job,
    get_kb_construction_job,
    kb_construction_status,
    list_benchmark_datasets,
    run_benchmark_generation,
    run_evaluation_workbench,
    start_kb_construction_job,
    start_benchmark_generation_job,
    stop_benchmark_generation_job,
)
from app.support_layer_workbench import (
    experimental_db_status,
    rag_workbench_status,
    run_rag_pipeline_action,
    test_experimental_db_evidence,
    test_experimental_db_workbook_evidence,
    sample_database,
)
from app.capabilities.provider_registry import (
    PROVIDER_ROOT,
    evaluate_provider_package,
    list_provider_packages,
    provider_runtime_spec,
    register_reviewed_provider,
    revoke_provider,
    save_provider_package,
    update_provider_review,
)
from app.tools.tool_onboarding import ToolOnboardingSpec, build_tool_onboarding_package
from app.utils.logger import setup_logger


settings = get_settings()
logger = setup_logger(__name__)
app = FastAPI(title=settings.app_name, version="0.1.0")
agent = ChemAnalyst()
planner_agent = OrchestrationAgent(agent)
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGO_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "logo.png"
MANUALS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "manuals"
DOCUMENT_UPLOAD_DIR = Path(__file__).resolve().parent / "tools" / "document_uploads"
PUBLIC_DEFAULT_TOOL_IDS = {"database.experimental_history"}
DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _register_reviewed_provider_packages_on_startup() -> None:
    for provider in list_provider_packages():
        review = provider.get("review") or {}
        if not (review.get("reviewed") and review.get("enabled")):
            continue
        provider_id = str(provider.get("provider_id") or "").strip()
        if not provider_id:
            continue
        result = register_reviewed_provider(agent.registry, provider_id)
        if result.get("status") == "success":
            logger.info("Registered reviewed capability provider on startup: %s", provider_id)
        else:
            logger.warning("Skipped reviewed capability provider %s: %s", provider_id, result.get("message"))


def _resolve_registered_tool_name(tool_name: str) -> str:
    if agent.registry.get(tool_name) is not None:
        return tool_name
    if not tool_name.startswith("tool."):
        prefixed = f"tool.{tool_name}"
        if agent.registry.get(prefixed) is not None:
            return prefixed
    return tool_name


def _upload_tool_candidates() -> list[dict]:
    candidates: list[dict] = []
    if hasattr(agent.registry, "list_capabilities"):
        for schema in agent.registry.list_capabilities():
            capability_id = str(schema.capability_id)
            if capability_id in PUBLIC_DEFAULT_TOOL_IDS or capability_id.startswith("tool."):
                candidates.append(
                    {
                        "capability_id": capability_id,
                        "description": schema.description,
                        "input_schema": schema.input_schema,
                        "output_schema": schema.output_schema,
                        "preconditions": schema.preconditions,
                        "artifacts_produced": schema.artifacts_produced,
                    }
                )
    elif hasattr(agent.registry, "list_specs"):
        for spec in agent.registry.list_specs():
            capability_id = str(spec.name)
            if capability_id in PUBLIC_DEFAULT_TOOL_IDS or capability_id.startswith("tool."):
                candidates.append(
                    {
                        "capability_id": capability_id,
                        "description": spec.description,
                        "input_schema": spec.parameters,
                        "output_schema": {},
                        "preconditions": list(spec.required_upload_fields or ()),
                        "artifacts_produced": [],
                    }
                )
    return candidates


def _select_upload_tool_name(*, requested_tool_name: str, query: str | None, file_path: Path) -> str:
    requested = (requested_tool_name or "").strip()
    if requested and requested.lower() != "auto":
        return _resolve_registered_tool_name(requested)

    candidates = _upload_tool_candidates()
    if not candidates:
        return "database.experimental_history"

    query_text = (query or "").lower()
    file_text = file_path.name.lower()
    best_score = 0
    best_id = ""
    for candidate in candidates:
        capability_id = str(candidate.get("capability_id") or "")
        haystack = " ".join(
            [
                capability_id,
                str(candidate.get("description") or ""),
                json.dumps(candidate.get("input_schema") or {}, ensure_ascii=False),
                json.dumps(candidate.get("output_schema") or {}, ensure_ascii=False),
                " ".join(candidate.get("preconditions") or []),
                " ".join(candidate.get("artifacts_produced") or []),
            ]
        ).lower()
        score = sum(1 for token in set(query_text.replace("_", " ").replace("-", " ").split()) if len(token) > 1 and token in haystack)
        score += sum(1 for token in set(file_text.replace("_", " ").replace("-", " ").split()) if len(token) > 2 and token in haystack)
        score += _score_upload_file_schema_match(file_path=file_path, candidate_text=haystack)
        if score > best_score:
            best_score = score
            best_id = capability_id
    if best_score > 0 and best_id:
        return _resolve_registered_tool_name(best_id)

    selected = _llm_select_upload_tool(query=query or "", file_path=file_path, candidates=candidates)
    if selected:
        return selected
    if best_id:
        return _resolve_registered_tool_name(best_id)
    if file_path.suffix.lower() in {".xlsx", ".xls"} and agent.registry.get("database.experimental_history") is not None:
        return "database.experimental_history"
    return "database.experimental_history"


def _score_upload_file_schema_match(*, file_path: Path, candidate_text: str) -> int:
    if file_path.suffix.lower() not in {".xlsx", ".xls"}:
        return 0
    try:
        sheet_names = [str(item).lower() for item in pd.ExcelFile(file_path).sheet_names]
    except Exception:
        return 0
    score = 0
    for sheet_name in sheet_names:
        compact = sheet_name.replace(" ", "").replace("_", "")
        if sheet_name and sheet_name in candidate_text:
            score += 3
        elif compact and compact in candidate_text.replace(" ", "").replace("_", ""):
            score += 3
    return score


def _llm_select_upload_tool(*, query: str, file_path: Path, candidates: list[dict]) -> str | None:
    if not query.strip():
        return None
    compact_candidates = [
        {
            "capability_id": item.get("capability_id"),
            "description": item.get("description"),
            "input_schema": item.get("input_schema"),
            "artifacts_produced": item.get("artifacts_produced"),
        }
        for item in candidates
    ]
    prompt = (
        "You are ChemAnalyst's upload tool selector. Select the best capability for the uploaded file and user request.\n"
        "Return JSON only: {\"capability_id\":\"...\", \"reason\":\"...\"}.\n"
        "If no analytical tool is appropriate, return an empty capability_id.\n"
        f"Uploaded filename: {file_path.name}\n"
        f"User request: {query}\n"
        f"Candidate capabilities: {json.dumps(compact_candidates, ensure_ascii=False)}"
    )
    try:
        raw = agent.llm.chat(prompt)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(raw[start : end + 1])
        capability_id = _resolve_registered_tool_name(str(data.get("capability_id") or "").strip())
        valid_ids = {_resolve_registered_tool_name(str(item.get("capability_id") or "")) for item in candidates}
        return capability_id if capability_id in valid_ids else None
    except Exception as exc:
        logger.info("LLM upload tool selection fallback triggered: %s", exc)
        return None


_register_reviewed_provider_packages_on_startup()


def _is_relative_to(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False


def _validate_artifact(path_str: str) -> Path:
    artifact_path = Path(path_str).resolve()
    allowed_roots = [DOCUMENT_UPLOAD_DIR.resolve(), MANUALS_DIR.resolve(), PROVIDER_ROOT.resolve()]
    if not any(_is_relative_to(artifact_path, root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Artifact path is outside the public workspace.")
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file was not found.")
    return artifact_path


def _artifact_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return f"/tools/artifact?path={quote(path_str)}"


def _artifact_preview_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return f"/tools/artifact/preview?path={quote(path_str)}"


def _attach_artifact_urls(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return payload

    for nested_key in ("delegated_result",):
        nested_payload = payload.get(nested_key)
        if isinstance(nested_payload, dict):
            payload[nested_key] = _attach_artifact_urls(nested_payload)

    download_urls = payload.get("download_urls", {})
    preview_urls = payload.get("preview_urls", {})
    for key in (
        "output_excel",
        "chart_html",
        "extracted_text_file",
        "input_file",
        "template_input_file",
        "template_reference_file",
        "manual_file",
    ):
        if payload.get(key):
            download_urls[key] = _artifact_url(payload.get(key))
    for key in ("template_input_file", "template_reference_file", "manual_file"):
        if payload.get(key):
            preview_urls[key] = _artifact_preview_url(payload.get(key))

    if download_urls.get("chart_html"):
        payload["chart_embed_url"] = download_urls["chart_html"]
    if download_urls:
        payload["download_urls"] = download_urls
    if preview_urls:
        payload["preview_urls"] = preview_urls
    return payload


def _utf8_json_response(payload: QueryResponse, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(payload),
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )


def _ndjson_line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _preview_tabular_file(path: Path, *, limit: int | None = 10) -> dict:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        display_df = df if limit is None else df.head(limit)
        return {
            "title": path.name,
            "kind": "table",
            "truncated": limit is not None and len(df) > len(display_df),
            "limit": limit,
            "sheets": [
                {
                    "name": "CSV",
                    "columns": [str(col) for col in df.columns.tolist()],
                    "rows": display_df.fillna("").astype(str).values.tolist(),
                    "total_rows": int(len(df)),
                    "displayed_rows": int(len(display_df)),
                }
            ],
        }

    excel = pd.ExcelFile(path)
    sheets = []
    truncated = False
    for sheet_name in excel.sheet_names[:5]:
        full_df = excel.parse(sheet_name)
        display_df = full_df if limit is None else full_df.head(limit)
        if limit is not None and len(full_df) > len(display_df):
            truncated = True
        sheets.append(
            {
                "name": sheet_name,
                "columns": [str(col) for col in full_df.columns.tolist()],
                "rows": display_df.fillna("").astype(str).values.tolist(),
                "total_rows": int(len(full_df)),
                "displayed_rows": int(len(display_df)),
            }
        )
    return {
        "title": path.name,
        "kind": "table",
        "sheets": sheets,
        "truncated": truncated,
        "limit": limit,
    }


def _preview_text_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return {"title": path.name, "kind": "text", "content": text}


def _stream_query_events(request: QueryRequest):
    session = agent.sessions.get_or_create(request.session_id)
    retrieval_overrides = request.retrieval.model_dump(exclude_none=True) if getattr(request, "retrieval", None) else None
    entry = classify_front_door(
        query=request.query,
        session=session,
        task_type_override=request.task_type,
        rag_mode_override=request.rag_mode,
        llm=agent.llm,
    )
    task_type = "planner" if entry.route == "complex" else str(entry.simple_mode or "chat")
    used_tools: list[str] = []

    yield _ndjson_line(
        {
            "type": "meta",
            "session_id": session.session_id,
            "task_type": task_type,
        }
    )

    try:
        if entry.route == "complex":
            yield _ndjson_line({"type": "status", "message": "Executing complex task orchestration..."} )
            result = planner_agent.answer_query(
                request.query,
                session_id=session.session_id,
                task_type_override=request.task_type,
                template_override="auto",
                rag_mode_override=request.rag_mode,
                retrieval_overrides=retrieval_overrides,
            )
            result["raw_result"] = _attach_artifact_urls(result.get("raw_result"))
            yield _ndjson_line({"type": "final", "data": result})
            return

        if request.task_type == "auto":
            yield _ndjson_line({"type": "status", "message": "Generating answer..."})
            result = agent.answer_query(
                request.query,
                task_type_override=request.task_type,
                session_id=session.session_id,
                rag_mode_override=request.rag_mode,
                retrieval_overrides=retrieval_overrides,
            )
            result["raw_result"] = _attach_artifact_urls(result.get("raw_result"))
            yield _ndjson_line({"type": "final", "data": result})
            return

        if entry.simple_mode == "reasoning":
            yield _ndjson_line({"type": "status", "message": "Running reasoning analysis..."})
            context = agent._build_context(session.session_id, request.query)
            scoped_query = agent._build_response_query(request.query)
            reason_query = scoped_query if not context else f"{context}\n\nUser query: {scoped_query}"
            answer_parts: list[str] = []
            for part in agent.llm.reason_stream(reason_query):
                answer_parts.append(part)
                yield _ndjson_line({"type": "delta", "content": part})
            answer = "".join(answer_parts).strip()
            agent._record_turn(session.session_id, request.query, answer)
            yield _ndjson_line(
                {
                    "type": "final",
                    "data": {
                        "session_id": session.session_id,
                        "task_type": task_type,
                        "used_model": settings.active_llm_model,
                        "used_tools": used_tools,
                        "answer": answer,
                        "raw_result": None,
                    },
                }
            )
            return

        if entry.simple_mode == "rag":
            yield _ndjson_line({"type": "status", "message": "Retrieving knowledge base evidence..."})
            yield _ndjson_line({"type": "status", "message": "RAG route selected; generating answer..."})
            result = agent.answer_query(
                request.query,
                task_type_override=request.task_type,
                session_id=session.session_id,
                rag_mode_override=request.rag_mode,
                retrieval_overrides=retrieval_overrides,
            )
            result["raw_result"] = _attach_artifact_urls(result.get("raw_result"))
            yield _ndjson_line({"type": "final", "data": result})
            return

        yield _ndjson_line({"type": "status", "message": "Generating answer..."})
        context = agent._build_context(session.session_id, request.query)
        answer_parts: list[str] = []
        for part in agent.llm.chat_stream(agent._build_response_query(request.query), context=context):
            answer_parts.append(part)
            yield _ndjson_line({"type": "delta", "content": part})
        answer = "".join(answer_parts).strip()
        agent._record_turn(session.session_id, request.query, answer)
        yield _ndjson_line(
            {
                "type": "final",
                "data": {
                    "session_id": session.session_id,
                    "task_type": task_type,
                    "used_model": settings.active_llm_model,
                    "used_tools": used_tools,
                    "answer": answer,
                    "raw_result": None,
                },
            }
        )
    except Exception as exc:
        logger.exception("Streaming query failed.")
        yield _ndjson_line(
            {
                "type": "final",
                "data": {
                    "session_id": session.session_id,
                    "task_type": task_type,
                    "used_model": "none",
                    "used_tools": used_tools,
                    "answer": f"Request failed: {exc}",
                    "raw_result": None,
                },
            }
        )


def _stream_planner_query_events(request: PlannerQueryRequest):
    session = agent.sessions.get_or_create(request.session_id)
    retrieval_overrides = request.retrieval.model_dump(exclude_none=True) if getattr(request, "retrieval", None) else None
    yield _ndjson_line(
        {
            "type": "meta",
            "session_id": session.session_id,
            "task_type": "planner",
        }
    )
    yield _ndjson_line({"type": "status", "message": "Executing Planner..."})
    try:
        result = planner_agent.answer_query(
            request.query,
            session_id=session.session_id,
            task_type_override=request.task_type,
            template_override=request.template_override,
            rag_mode_override=request.rag_mode,
            retrieval_overrides=retrieval_overrides,
        )
        result["raw_result"] = _attach_artifact_urls(result.get("raw_result"))
        yield _ndjson_line({"type": "final", "data": result})
    except Exception as exc:
        logger.exception("Planner streaming query failed.")
        yield _ndjson_line(
            {
                "type": "final",
                "data": {
                    "session_id": session.session_id,
                    "task_type": "planner",
                    "used_model": "none",
                    "used_tools": [],
                    "answer": f"Request failed: {exc}",
                    "raw_result": None,
                },
            }
        )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat_v3.html")


@app.get("/tool-onboarding", include_in_schema=False)
def tool_onboarding_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "tool_onboarding.html", headers={"Cache-Control": "no-store"})


@app.get("/benchmark-workbench", include_in_schema=False)
def benchmark_workbench_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "benchmark_workbench.html", headers={"Cache-Control": "no-store"})


@app.get("/rag-workbench", include_in_schema=False)
def rag_workbench_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "rag_workbench.html", headers={"Cache-Control": "no-store"})


@app.get("/experimental-db-workbench", include_in_schema=False)
def experimental_db_workbench_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "experimental_db_workbench.html", headers={"Cache-Control": "no-store"})


@app.get("/experimental-db-manager", include_in_schema=False)
def experimental_db_manager_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "experimental_db_manager.html", headers={"Cache-Control": "no-store"})


@app.get("/support-layer", include_in_schema=False)
def support_layer_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "support_layer.html", headers={"Cache-Control": "no-store"})


@app.get("/brand/logo", include_in_schema=False)
def brand_logo() -> FileResponse:
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="Logo not found.")
    return FileResponse(LOGO_PATH, media_type="image/png", filename="logo.png")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", app_name=settings.app_name)


@app.get("/benchmark-workbench/options")
def benchmark_workbench_options() -> JSONResponse:
    return _utf8_json_response(
        {
            "benchmark_datasets": list_benchmark_datasets(),
            "source_modes": [
                {
                    "id": "existing_chunks",
                    "label": "Existing processed chunks JSONL",
                    "description": "Use a prepared chunks.jsonl produced by the KB ingest pipeline.",
                },
                {
                    "id": "pdf_directory",
                    "label": "PDF directory",
                    "description": "Run PDF parsing and chunking first, then generate QA from the new chunks.",
                },
            ],
            "benchmark_styles": [
                {
                    "id": "semantic",
                    "label": "Semantic literature QA",
                    "description": "Literature-derived explanatory QA for answer relevance and context coverage.",
                },
            ],
            "quality_metrics": [
                "self_contained_score",
                "answerable_score",
                "objective_score",
                "specificity_score",
                "ambiguity_score",
                "evidence_locality_score",
                "reasoning_type",
                "difficulty",
                "hallucination_risk_score",
                "quality_score",
            ],
            "evaluation_levels": [
                {
                    "id": "planner_sufficiency",
                    "label": "Planner evidence sufficiency",
                    "description": "Offline rule-case benchmark for evidence status and remediation actions.",
                },
                {
                    "id": "task_specific_schema",
                    "label": "Task-specific evaluation schema",
                    "description": "Inspect workflow case definitions, expected evidence, templates, and conclusion fields.",
                },
                {
                    "id": "agent_workflow",
                    "label": "Agent workflow evaluation",
                    "description": "Run a selected workflow case through Planner and evaluate EvidenceBundle behavior.",
                },
            ],
        }
    )


@app.get("/benchmark-workbench/kb/status")
def benchmark_workbench_kb_status() -> JSONResponse:
    return _utf8_json_response(kb_construction_status())


@app.get("/benchmark-workbench/kb/active-keyword-graph")
def benchmark_workbench_active_keyword_graph() -> FileResponse:
    graph_path = (
        Path(__file__).resolve().parent.parent
        / "petroleum_kb"
        / "petroleum_kb"
        / "data"
        / "analysis"
        / "keyword_graph_text_clean"
        / "topic_keyword_graph"
        / "keyword_graph.html"
    )
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Active keyword graph not found.")
    return FileResponse(graph_path, media_type="text/html")


@app.post("/benchmark-workbench/kb/start")
def start_kb_construction_from_workbench(request: KBConstructionRequest) -> JSONResponse:
    try:
        result = start_kb_construction_job(request)
    except Exception as exc:
        logger.exception("KB construction job failed to start.")
        return _utf8_json_response(
            {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
            status_code=400,
        )
    return _utf8_json_response(result)


@app.get("/benchmark-workbench/kb/jobs/{job_id}")
def kb_construction_job_status(job_id: str) -> JSONResponse:
    try:
        return _utf8_json_response(get_kb_construction_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/benchmark-workbench/benchmark/generate")
def generate_benchmark_from_workbench(request: BenchmarkGenerationRequest) -> JSONResponse:
    try:
        result = run_benchmark_generation(request)
    except Exception as exc:
        logger.exception("Benchmark generation failed.")
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        return _utf8_json_response(result, status_code=400)
    return _utf8_json_response(result, status_code=200 if result.get("status") == "success" else 400)


@app.post("/benchmark-workbench/benchmark/start")
def start_benchmark_from_workbench(request: BenchmarkGenerationRequest) -> JSONResponse:
    try:
        result = start_benchmark_generation_job(request)
    except Exception as exc:
        logger.exception("Benchmark generation job failed to start.")
        return _utf8_json_response(
            {"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
            status_code=400,
        )
    return _utf8_json_response(result)


@app.get("/benchmark-workbench/benchmark/jobs/{job_id}")
def benchmark_job_status(job_id: str) -> JSONResponse:
    try:
        return _utf8_json_response(get_benchmark_generation_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/benchmark-workbench/benchmark/jobs/{job_id}/stop")
def stop_benchmark_job(job_id: str) -> JSONResponse:
    try:
        return _utf8_json_response(stop_benchmark_generation_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/benchmark-workbench/evaluate")
def evaluate_from_workbench(request: EvaluationWorkbenchRequest) -> JSONResponse:
    try:
        result = run_evaluation_workbench(
            level=request.level,
            case_id=request.case_id,
            session_id=request.session_id,
            planner_agent=planner_agent,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _utf8_json_response(result)


@app.get("/support-layer/rag/status")
def support_layer_rag_status() -> JSONResponse:
    return _utf8_json_response(rag_workbench_status(settings))


@app.post("/support-layer/rag/test")
def support_layer_rag_test(request: RAGWorkbenchRequest) -> JSONResponse:
    retrieval = {
        "enable_refinement_agent": request.enable_refinement_agent,
        "enable_bm25": request.enable_bm25,
        "enable_reranker": request.enable_reranker,
    }
    result = agent.answer_query(
        request.query,
        task_type_override="rag",
        rag_mode_override=request.rag_mode,
        retrieval_overrides={key: value for key, value in retrieval.items() if value is not None},
    )
    return _utf8_json_response(result)


@app.post("/support-layer/rag/pipeline")
def support_layer_rag_pipeline(request: RAGPipelineRequest) -> JSONResponse:
    try:
        result = run_rag_pipeline_action(request)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _utf8_json_response(result)


@app.get("/support-layer/experimental-db/status")
def support_layer_experimental_db_status() -> JSONResponse:
    return _utf8_json_response(experimental_db_status())


@app.post("/support-layer/experimental-db/test")
def support_layer_experimental_db_test(request: ExperimentalDBWorkbenchRequest) -> JSONResponse:
    return _utf8_json_response(test_experimental_db_evidence(request, llm=agent.llm))


@app.post("/support-layer/experimental-db/test-workbook")
def support_layer_experimental_db_test_workbook(
    file: UploadFile = File(...),
    query: str = Form("Use IR and GC-FID to infer density, distillation distribution, and saturates for this petroleum fraction sample."),
    top_k: int = Form(4),
    use_llm_reasoning: bool = Form(True),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="query_workbook_must_be_excel")
    saved_path = DOCUMENT_UPLOAD_DIR / f"{uuid4().hex}_{Path(file.filename or 'query_sample.xlsx').name}"
    try:
        with saved_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)
        result = test_experimental_db_workbook_evidence(
            saved_path,
            query=query,
            top_k=top_k,
            use_llm_reasoning=use_llm_reasoning,
            llm=agent.llm,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        file.file.close()
    return _utf8_json_response(result)


@app.get("/support-layer/experimental-db/samples")
def support_layer_experimental_db_samples(limit: int = 100) -> JSONResponse:
    return _utf8_json_response({"status": "success", "samples": sample_database().list_samples(limit=limit)})


@app.get("/support-layer/experimental-db/samples/{sample_id}")
def support_layer_experimental_db_sample(sample_id: str) -> JSONResponse:
    record = sample_database().get_sample(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail="experimental_sample_not_found")
    return _utf8_json_response({"status": "success", "sample": record})


@app.post("/support-layer/experimental-db/samples/import")
def support_layer_experimental_db_import(request: ExperimentalSampleImportRequest) -> JSONResponse:
    try:
        return _utf8_json_response(sample_database().import_payload(request.payload))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/support-layer/experimental-db/template")
def support_layer_experimental_db_template() -> StreamingResponse:
    content = sample_database().template_xlsx()
    headers = {"Content-Disposition": 'attachment; filename="chemanalyst_experimental_sample_template.xlsx"'}
    return StreamingResponse(
        BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/support-layer/experimental-db/template/definition")
def support_layer_experimental_db_template_definition() -> JSONResponse:
    return _utf8_json_response({"status": "success", "template": sample_database().get_template_definition()})


@app.post("/support-layer/experimental-db/template/update")
def support_layer_experimental_db_template_update(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="template_workbook_must_be_excel")
    saved_path = DOCUMENT_UPLOAD_DIR / f"{uuid4().hex}_{Path(file.filename or 'experimental_db_template.xlsx').name}"
    try:
        with saved_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)
        result = sample_database().update_template_from_workbook(saved_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        file.file.close()
    return _utf8_json_response(result)


@app.post("/support-layer/experimental-db/workbook/import")
def support_layer_experimental_db_workbook_import(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="sample_workbook_must_be_excel")
    saved_path = DOCUMENT_UPLOAD_DIR / f"{uuid4().hex}_{Path(file.filename or 'sample.xlsx').name}"
    try:
        with saved_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)
        result = sample_database().import_workbook(saved_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        file.file.close()
    return _utf8_json_response(result)


@app.post("/support-layer/experimental-db/workbooks/import-batch")
def support_layer_experimental_db_workbooks_import_batch(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="sample_workbooks_required")
    imported: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    database = sample_database()
    for file in files:
        filename = Path(file.filename or "sample.xlsx").name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".xlsx", ".xls"}:
            failed.append({"filename": filename, "error": "sample_workbook_must_be_excel"})
            file.file.close()
            continue
        saved_path = DOCUMENT_UPLOAD_DIR / f"{uuid4().hex}_{filename}"
        try:
            with saved_path.open("wb") as buffer:
                copyfileobj(file.file, buffer)
            result = database.import_workbook(saved_path)
            imported.append({"filename": filename, **result})
        except (FileNotFoundError, ValueError) as exc:
            failed.append({"filename": filename, "error": str(exc)})
        finally:
            file.file.close()
    status = "success" if not failed else ("partial_success" if imported else "error")
    return _utf8_json_response(
        {
            "status": status,
            "message": "sample_workbook_batch_imported",
            "submitted_count": len(files),
            "imported_count": len(imported),
            "failed_count": len(failed),
            "imported": imported,
            "failed": failed,
        }
    )


@app.post("/support-layer/experimental-db/samples/{sample_id}/artifact")
def support_layer_experimental_db_artifact(
    sample_id: str,
    analysis_type: str = Form(...),
    method: str | None = Form(default=None),
    description: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> JSONResponse:
    saved_path = DOCUMENT_UPLOAD_DIR / f"{uuid4().hex}_{Path(file.filename or 'analysis.bin').name}"
    try:
        with saved_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)
        result = sample_database().add_artifact(
            sample_id,
            saved_path,
            analysis_type=analysis_type,
            method=method,
            description=description,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        file.file.close()
    return _utf8_json_response(result)


@app.delete("/support-layer/experimental-db/samples/{sample_id}")
def support_layer_experimental_db_delete_sample(sample_id: str) -> JSONResponse:
    try:
        return _utf8_json_response(sample_database().delete_sample(sample_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/capabilities/onboard", response_model=ToolOnboardingResponse)
def onboard_tool(request: ToolOnboardingRequest) -> ToolOnboardingResponse:
    package = build_tool_onboarding_package(
        ToolOnboardingSpec(
            tool_name=request.tool_name,
            description=request.description,
            invocation_type=request.invocation_type,
            entrypoint=request.entrypoint,
            input_schema=request.input_schema,
            output_schema=request.output_schema,
            evidence_outputs=request.evidence_outputs,
            preconditions=request.preconditions,
            artifacts_produced=request.artifacts_produced,
            example_queries=request.example_queries,
            quality_checks=request.quality_checks,
            safety_notes=request.safety_notes,
            upstream_capabilities=request.upstream_capabilities,
            downstream_capabilities=request.downstream_capabilities,
            input_bindings=request.input_bindings,
            output_bindings=request.output_bindings,
            database_integration_contract=request.database_integration_contract,
            adapter_template=request.adapter_template,
            adapter_config=request.adapter_config,
        )
    )
    return ToolOnboardingResponse(
        status="success",
        package=package.to_dict(),
        message="Capability package generated. Review and register a safe adapter before enabling runtime execution.",
    )


@app.post("/capabilities/tool-onboarding/input-template")
def upload_tool_onboarding_input_template(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    allowed_suffixes = {".xlsx", ".xls", ".csv", ".json", ".txt"}
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported input template type: {suffix or 'unknown'}")

    safe_name = Path(file.filename or f"input_template{suffix}").name
    template_dir = DOCUMENT_UPLOAD_DIR / "tool_onboarding_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    saved_path = template_dir / f"{uuid4().hex}_{safe_name}"
    try:
        with saved_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    return _utf8_json_response(
        {
            "status": "success",
            "file_name": safe_name,
            "source_path": str(saved_path),
            "download_url": _artifact_url(str(saved_path)),
            "message": "Input template uploaded. Prepare or deploy the provider so it is copied into the provider package.",
        }
    )


@app.post("/capabilities/providers/save", response_model=CapabilityProviderResponse)
def save_capability_provider(request: CapabilityProviderSaveRequest) -> CapabilityProviderResponse:
    result = save_provider_package(request.package, provider_id=request.provider_id)
    return CapabilityProviderResponse(status="success", result=result, message="Capability provider package saved.")


@app.get("/capabilities/providers", response_model=CapabilityProviderResponse)
def list_capability_providers() -> CapabilityProviderResponse:
    return CapabilityProviderResponse(status="success", result={"providers": list_provider_packages()}, message="Capability providers listed.")


@app.get("/capabilities/providers/{provider_id}/runtime-spec", response_model=CapabilityProviderResponse)
def get_capability_provider_runtime_spec(provider_id: str) -> CapabilityProviderResponse:
    result = provider_runtime_spec(provider_id)
    result = _attach_artifact_urls(result) or result
    status = "success" if result.get("status") == "success" else "error"
    return CapabilityProviderResponse(status=status, result=result, message=str(result.get("message") or "provider_runtime_spec"))


@app.get("/capabilities/options", response_model=CapabilityProviderResponse)
def list_capability_options() -> CapabilityProviderResponse:
    runtime_capabilities = []
    if hasattr(agent.registry, "list_capabilities"):
        runtime_capabilities = [
            {
                "capability_id": schema.capability_id,
                "capability_type": schema.capability_type,
                "description": schema.description,
                "source": "runtime_registry",
            }
            for schema in agent.registry.list_capabilities()
        ]
    elif hasattr(agent.registry, "list_specs"):
        runtime_capabilities = [
            {
                "capability_id": spec.name,
                "capability_type": "tool",
                "description": spec.description,
                "source": "runtime_registry",
            }
            for spec in agent.registry.list_specs()
        ]
    provider_capabilities = [
        {
            "capability_id": provider.get("capability_id"),
            "capability_type": "provider",
            "description": provider.get("description"),
            "provider_id": provider.get("provider_id"),
            "source": "provider_package",
        }
        for provider in list_provider_packages()
    ]
    return CapabilityProviderResponse(
        status="success",
        result={"capabilities": runtime_capabilities + provider_capabilities},
        message="Capability options listed.",
    )


@app.post("/capabilities/providers/{provider_id}/evaluate", response_model=CapabilityProviderResponse)
def evaluate_capability_provider(provider_id: str, request: CapabilityProviderEvalRequest) -> CapabilityProviderResponse:
    result = evaluate_provider_package(provider_id, execute=request.execute, sample_payload=request.sample_payload)
    return CapabilityProviderResponse(status="success", result=result, message="Capability provider evaluated.")


@app.post("/capabilities/providers/{provider_id}/review", response_model=CapabilityProviderResponse)
def review_capability_provider(provider_id: str, request: CapabilityProviderReviewRequest) -> CapabilityProviderResponse:
    result = update_provider_review(provider_id, reviewed=request.reviewed, enabled=request.enabled, notes=request.notes)
    status = "success" if result.get("status") == "success" else "error"
    return CapabilityProviderResponse(status=status, result=result, message=str(result.get("message") or "review_updated"))


@app.post("/capabilities/providers/{provider_id}/register", response_model=CapabilityProviderResponse)
def register_capability_provider(provider_id: str) -> CapabilityProviderResponse:
    result = register_reviewed_provider(agent.registry, provider_id)
    status = "success" if result.get("status") == "success" else "error"
    return CapabilityProviderResponse(status=status, result=result, message=str(result.get("message") or "registration_finished"))


@app.post("/capabilities/providers/{provider_id}/revoke", response_model=CapabilityProviderResponse)
def revoke_capability_provider(provider_id: str, request: CapabilityProviderRevokeRequest) -> CapabilityProviderResponse:
    if not request.confirm:
        return CapabilityProviderResponse(
            status="error",
            result={"provider_id": provider_id, "required": "confirm=true"},
            message="revoke_confirmation_required",
        )
    result = revoke_provider(agent.registry, provider_id, reason=request.reason)
    status = "success" if result.get("status") == "success" else "error"
    return CapabilityProviderResponse(status=status, result=result, message=str(result.get("message") or "provider_revoked"))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    logger.info("Received query request.")
    retrieval_overrides = request.retrieval.model_dump(exclude_none=True) if request.retrieval else None
    session = agent.sessions.get_or_create(request.session_id)
    entry = classify_front_door(
        query=request.query,
        session=session,
        task_type_override=request.task_type,
        rag_mode_override=request.rag_mode,
        llm=agent.llm,
    )
    if entry.route == "complex":
        result = planner_agent.answer_query(
            request.query,
            session_id=session.session_id,
            task_type_override=request.task_type,
            template_override="auto",
            rag_mode_override=request.rag_mode,
            retrieval_overrides=retrieval_overrides,
        )
    else:
        result = agent.answer_query(
            request.query,
            task_type_override=request.task_type,
            session_id=session.session_id,
            rag_mode_override=request.rag_mode,
            retrieval_overrides=retrieval_overrides,
        )
    result["raw_result"] = _attach_artifact_urls(result.get("raw_result"))
    return _utf8_json_response(QueryResponse(**result))


@app.post("/planner/query", response_model=QueryResponse)
def planner_query(request: PlannerQueryRequest) -> QueryResponse:
    logger.info("Received planner query request.")
    retrieval_overrides = request.retrieval.model_dump(exclude_none=True) if request.retrieval else None
    result = planner_agent.answer_query(
        request.query,
        session_id=request.session_id,
        task_type_override=request.task_type,
        template_override=request.template_override,
        rag_mode_override=request.rag_mode,
        retrieval_overrides=retrieval_overrides,
    )
    result["raw_result"] = _attach_artifact_urls(result.get("raw_result"))
    return _utf8_json_response(QueryResponse(**result))


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    logger.info("Received streaming query request.")
    return StreamingResponse(_stream_query_events(request), media_type="application/x-ndjson; charset=utf-8")


@app.post("/planner/query/stream")
def planner_query_stream(request: PlannerQueryRequest) -> StreamingResponse:
    logger.info("Received planner streaming query request.")
    return StreamingResponse(_stream_planner_query_events(request), media_type="application/x-ndjson; charset=utf-8")


def _save_tool_upload(*, tool_name: str, file: UploadFile = File(...)) -> tuple[Path, Path]:
    tool_name = _resolve_registered_tool_name(tool_name)
    logger.info("Received tool upload request for tool=%s.", tool_name)

    if tool_name.lower() == "auto":
        allowed_suffixes = {".xlsx", ".xls", ".csv", ".json", ".txt", ".pdf", ".docx", ".png", ".jpg", ".jpeg"}
        base_dir = DOCUMENT_UPLOAD_DIR
        default_name = "input.bin"
    elif agent.registry.get(tool_name) is not None:
        allowed_suffixes = {".xlsx", ".xls", ".csv", ".json", ".txt"}
        base_dir = DOCUMENT_UPLOAD_DIR
        default_name = "input.bin"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    saved_name = f"{uuid4().hex}_{Path(file.filename or default_name).name}"
    saved_path = base_dir / saved_name
    job_output_dir = base_dir / Path(saved_name).stem
    job_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with saved_path.open("wb") as buffer:
            copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.exception("Failed to save uploaded file.")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {exc}") from exc
    finally:
        file.file.close()
    return saved_path, job_output_dir


@app.post("/tools/upload", response_model=ToolResponse)
def run_tool_upload(
    tool_name: str = Form(...),
    session_id: str | None = Form(default=None),
    query: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> ToolResponse:
    return _run_tool_upload_with_agent(
        tool_name=tool_name,
        session_id=session_id,
        query=query,
        file=file,
    )


def _run_tool_upload_with_agent(
    *,
    tool_name: str,
    session_id: str | None,
    query: str | None,
    file: UploadFile,
) -> ToolResponse:
    saved_path, job_output_dir = _save_tool_upload(tool_name=tool_name, file=file)
    tool_name = _select_upload_tool_name(requested_tool_name=tool_name, query=query, file_path=saved_path)

    handled = agent.handle_uploaded_tool(
        tool_name=tool_name,
        payload={
            "excel_file": str(saved_path),
            "file_path": str(saved_path),
            "output_dir": str(job_output_dir),
        },
        session_id=session_id,
        user_query=query or f"Run {tool_name}",
    )
    if isinstance(handled.get("result"), dict):
        handled["result"] = _attach_artifact_urls(handled["result"])
    return ToolResponse(**handled)


@app.get("/tools/artifact", include_in_schema=False)
def get_tool_artifact(path: str) -> FileResponse:
    artifact_path = _validate_artifact(path)
    media_type = "application/octet-stream"
    suffix = artifact_path.suffix.lower()
    if suffix == ".html":
        media_type = "text/html; charset=utf-8"
    elif suffix in {".xlsx", ".xls"}:
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif suffix in {".txt", ".md", ".json", ".csv"}:
        media_type = "text/plain; charset=utf-8"
    elif suffix == ".pdf":
        media_type = "application/pdf"
    elif suffix == ".docx":
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".bmp":
        media_type = "image/bmp"
    elif suffix in {".tif", ".tiff"}:
        media_type = "image/tiff"
    if suffix == ".html":
        return FileResponse(artifact_path, media_type=media_type, content_disposition_type="inline")
    return FileResponse(artifact_path, media_type=media_type, filename=artifact_path.name)


@app.get("/tools/artifact/preview", include_in_schema=False)
def preview_tool_artifact(path: str, all_rows: bool = False) -> dict:
    artifact_path = _validate_artifact(path)
    suffix = artifact_path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".csv"}:
        return _preview_tabular_file(artifact_path, limit=None if all_rows else 10)
    if suffix in {".md", ".txt", ".json"}:
        return _preview_text_file(artifact_path)
    raise HTTPException(status_code=400, detail="Unsupported preview file type.")


