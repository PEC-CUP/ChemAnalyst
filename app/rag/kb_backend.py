from pathlib import Path
from contextlib import contextmanager
import os
import sys

from app.config import get_settings
from app.llm.deepseek_client import DeepSeekClient
from app.utils.logger import setup_logger
from app.rag.advanced_rag import (
    QAOrientedRAG,
    IterativeReviewRAG,
    _build_evidence_context,
    _expand_selected_evidences,
    _has_invalid_doc_refs,
)
from app.rag.evidence_filter import LLMEvidenceFilter
from app.rag.run_log import log_event, new_run_id


logger = setup_logger(__name__)

# Cache only successful initialization. Do not cache failures (None), because
# embedding/model loading may fail transiently (offline flags, first-run cache).
_SEARCHER = None
_SEARCHER_READY = False

_INVALID_LABEL_KEYS = {
    "source_group",
    "task_group",
    "entity_tags",
    "method_tags",
    "evidence_type",
    "confidence",
}


def _disable_invalid_label_signals(config) -> None:
    """Prevent generated metadata labels from affecting runtime RAG."""
    if hasattr(config, "beta"):
        config.beta = 0.0
    if hasattr(config, "gamma"):
        config.gamma = 0.0
    if hasattr(config, "source_weights"):
        config.source_weights = {}


def _strip_invalid_label_metadata(value):
    if isinstance(value, dict):
        return {
            key: _strip_invalid_label_metadata(item)
            for key, item in value.items()
            if key not in _INVALID_LABEL_KEYS and key not in {"tag_score", "source_score"}
        }
    if isinstance(value, list):
        return [_strip_invalid_label_metadata(item) for item in value]
    return value


def invalidate_kb_searcher() -> None:
    """Force the next RAG request to reload processed chunks and vector index."""
    global _SEARCHER, _SEARCHER_READY
    _SEARCHER = None
    _SEARCHER_READY = False


def _doc_id_from_metadata(metadata: dict) -> str:
    doi = (metadata.get("doi") or "").strip() if isinstance(metadata, dict) else ""
    if doi:
        return f"doi:{doi}"
    title = (metadata.get("title") or "").strip() if isinstance(metadata, dict) else ""
    if title:
        year = metadata.get("year") or ""
        return f"title:{title}::{year}"
    file_name = (metadata.get("file_name") or "").strip() if isinstance(metadata, dict) else ""
    source = (metadata.get("source") or "").strip() if isinstance(metadata, dict) else ""
    return f"file:{file_name}::{source}"


def _get_searcher():
    """Create the active layered-label knowledge searcher.

    The main app now treats `petroleum_kb` as the formal knowledge backend.
    This adapter keeps the rest of the agent code simple and hides the
    underlying searcher implementation details.
    """
    global _SEARCHER, _SEARCHER_READY
    if _SEARCHER_READY:
        return _SEARCHER

    settings = get_settings()
    if not settings.petroleum_kb_enabled:
        _SEARCHER_READY = True
        _SEARCHER = None
        return None

    try:
        from petroleum_kb.config import KBConfig
        from petroleum_kb.petroleum_kb.retrieval.searcher import TagAwareSearcher
    except Exception as exc:  # pragma: no cover
        logger.warning("Petroleum KB import failed: %s", exc)
        return None

    config = KBConfig(project_root=settings.petroleum_kb_dir)
    _disable_invalid_label_signals(config)
    # App runtime often runs in environments without outbound HF access.
    # Allow a dedicated switch to force offline/local cache model loading.
    if bool(getattr(settings, "petroleum_kb_embedding_local_files_only", False)):
        config.embedding_local_files_only = True
        # Keep embedder/reranker behavior consistent in offline mode.
        config.rerank_local_files_only = True
    # Reranker runtime overrides (important for offline/local-cache deployments).
    config.enable_reranker = bool(getattr(settings, "petroleum_kb_enable_reranker", config.enable_reranker))
    rerank_model = str(getattr(settings, "petroleum_kb_rerank_model", "") or "").strip()
    if rerank_model:
        config.rerank_model_name = rerank_model
    if bool(getattr(settings, "petroleum_kb_rerank_local_files_only", False)):
        config.rerank_local_files_only = True
    # If process is globally configured to offline mode, force local-only reranker.
    if os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
        config.rerank_local_files_only = True
    processed_file = config.processed_dir / "chunks.jsonl"
    ids_file = config.index_dir / f"{config.index_name}_ids.json"
    faiss_file = config.index_dir / f"{config.index_name}.faiss"
    numpy_file = config.index_dir / f"{config.index_name}.npy"
    if not processed_file.exists() or not ids_file.exists() or (not faiss_file.exists() and not numpy_file.exists()):
        logger.info("Petroleum KB index is not ready.")
        return None

    try:
        logger.info(
            "KB searcher init: reranker_enabled=%s model=%s local_only=%s embed_local_only=%s exe=%s",
            config.enable_reranker,
            config.rerank_model_name,
            config.rerank_local_files_only,
            config.embedding_local_files_only,
            sys.executable,
        )
        _SEARCHER = TagAwareSearcher.from_config(config)
        _SEARCHER_READY = True
        return _SEARCHER
    except Exception as exc:  # pragma: no cover
        logger.warning("Petroleum KB searcher init failed: %s", exc)
        # Do NOT mark ready; allow future retries after env/model/cache changes.
        return None


@contextmanager
def _temporary_kb_config_overrides(config, overrides: dict | None):
    """Temporarily patch KBConfig fields for a single retrieval call.

    app/rag is multi-tenant (multiple sessions). The searcher is cached, so we
    must revert overrides after the call to avoid leaking settings.
    """

    if not overrides:
        yield
        return

    applied: dict[str, object] = {}
    try:
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"beta", "gamma", "source_weights"}:
                continue
            if not hasattr(config, key):
                continue
            applied[key] = getattr(config, key)
            setattr(config, key, value)
        _disable_invalid_label_signals(config)
        yield
    finally:
        for key, old in applied.items():
            try:
                setattr(config, key, old)
            except Exception:
                # Best-effort restore; don't mask retrieval errors.
                pass
        _disable_invalid_label_signals(config)


def retrieve(
    query: str,
    top_k: int = 4,
    source_group: str | None = None,
    task_group: str | None = None,
    entity_tags: list[str] | None = None,
    retrieval_overrides: dict | None = None,
) -> list[dict]:
    searcher = _get_searcher()
    if searcher is None:
        return []

    with _temporary_kb_config_overrides(searcher.config, retrieval_overrides):
        results = searcher.search(
            query=query,
            top_k=top_k,
        )
    mapped: list[dict] = []
    for item in results:
        payload = item.to_dict()
        metadata = payload.get("metadata", {})
        mapped.append(
            {
                "doc_id": _doc_id_from_metadata(metadata if isinstance(metadata, dict) else {}),
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "source": metadata.get("source", ""),
                "file_name": metadata.get("file_name", ""),
                "title": metadata.get("title"),
                "year": metadata.get("year"),
                "doi": metadata.get("doi"),
                "score": float(payload["final_score"]),
                "semantic_score": float(payload["semantic_score"]),
                "backend": payload.get("debug", {}).get("backend", "petroleum_kb"),
            }
        )
    logger.info("Retrieved %s chunks from petroleum_kb for query.", len(mapped))
    return mapped


def retrieve_naive(
    query: str,
    top_k: int = 4,
    source_group: str | None = None,
    task_group: str | None = None,
    entity_tags: list[str] | None = None,
    retrieval_overrides: dict | None = None,
) -> list[dict]:
    searcher = _get_searcher()
    if searcher is None:
        return []

    with _temporary_kb_config_overrides(searcher.config, retrieval_overrides):
        results = searcher.naive_search(
            query=query,
            top_k=top_k,
        )
    mapped: list[dict] = []
    for item in results:
        payload = item.to_dict()
        metadata = payload.get("metadata", {})
        mapped.append(
            {
                "doc_id": _doc_id_from_metadata(metadata if isinstance(metadata, dict) else {}),
                "chunk_id": payload["chunk_id"],
                "text": payload["text"],
                "source": metadata.get("source", ""),
                "file_name": metadata.get("file_name", ""),
                "title": metadata.get("title"),
                "year": metadata.get("year"),
                "doi": metadata.get("doi"),
                "score": float(payload["final_score"]),
                "semantic_score": float(payload["semantic_score"]),
                "backend": payload.get("debug", {}).get("backend", "petroleum_kb"),
                "mode": "naive",
            }
        )
    logger.info("Retrieved %s naive chunks from petroleum_kb for query.", len(mapped))
    return mapped


def retrieve_documents(
    query: str,
    *,
    top_docs: int = 5,
    top_k: int = 12,
    chunks_per_doc: int | None = None,
    source_group: str | None = None,
    task_group: str | None = None,
    entity_tags: list[str] | None = None,
    retrieval_overrides: dict | None = None,
) -> list[dict]:
    """
    Document-level evidence retrieval with citations (RAG).

    This uses petroleum_kb's `search_documents` and returns a stable dict payload
    for the app layer.
    """

    searcher = _get_searcher()
    if searcher is None:
        return []
    try:
        with _temporary_kb_config_overrides(searcher.config, retrieval_overrides):
            evidences = searcher.search_documents(
                query=query,
                top_docs=top_docs,
                top_k=top_k,
                chunks_per_doc=chunks_per_doc,
            )
    except Exception as exc:  # pragma: no cover
        logger.warning("Petroleum KB search_docs failed: %s", exc)
        return []
    return [_strip_invalid_label_metadata(e.to_dict()) for e in evidences]


def _chunks_to_document_evidences(chunks: list[dict], *, chunks_per_doc: int) -> list[dict]:
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        doc_id = str(chunk.get("doc_id") or "").strip()
        if not doc_id:
            doc_id = _doc_id_from_metadata(chunk)
        if not doc_id:
            continue
        if doc_id not in grouped:
            grouped[doc_id] = {
                "doc_id": doc_id,
                "score": float(chunk.get("score") or chunk.get("semantic_score") or 0.0),
                "citation": {
                    "title": chunk.get("title"),
                    "authors": chunk.get("authors") or [],
                    "year": chunk.get("year"),
                    "journal": chunk.get("journal"),
                    "doi": chunk.get("doi"),
                    "file_name": chunk.get("file_name"),
                    "source": chunk.get("source"),
                },
                "chunks": [],
                "retrieval_route": "naive_vector_anchor",
            }
            order.append(doc_id)
        grouped[doc_id]["chunks"].append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
                "score": float(chunk.get("score") or chunk.get("semantic_score") or 0.0),
                "semantic_score": float(chunk.get("semantic_score") or chunk.get("score") or 0.0),
                "metadata": {
                    "source": chunk.get("source"),
                    "file_name": chunk.get("file_name"),
                    "title": chunk.get("title"),
                    "year": chunk.get("year"),
                    "journal": chunk.get("journal"),
                    "doi": chunk.get("doi"),
                },
                "retrieval_route": "naive_vector_anchor",
            }
        )
        grouped[doc_id]["chunks"] = grouped[doc_id]["chunks"][: max(1, int(chunks_per_doc))]
    return [grouped[doc_id] for doc_id in order]


def _merge_document_evidence_payloads(primary: list[dict], secondary: list[dict], *, chunks_per_doc: int) -> list[dict]:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for evidence in [*(primary or []), *(secondary or [])]:
        if not isinstance(evidence, dict):
            continue
        doc_id = str(evidence.get("doc_id") or "").strip()
        if not doc_id:
            continue
        if doc_id not in merged:
            item = dict(evidence)
            item["chunks"] = list(evidence.get("chunks") or [])[: max(1, int(chunks_per_doc))]
            merged[doc_id] = item
            order.append(doc_id)
            continue
        current = merged[doc_id]
        try:
            current["score"] = max(float(current.get("score") or 0.0), float(evidence.get("score") or 0.0))
        except Exception:
            pass
        chunks_by_id: dict[str, dict] = {}
        chunk_order: list[str] = []
        fallback_index = 0
        for chunk in [*(current.get("chunks") or []), *(evidence.get("chunks") or [])]:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                fallback_index += 1
                chunk_id = f"__fallback_{fallback_index}"
            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = dict(chunk)
                chunk_order.append(chunk_id)
        current["chunks"] = [chunks_by_id[cid] for cid in chunk_order[: max(1, int(chunks_per_doc))]]
    return [merged[doc_id] for doc_id in order]


def retrieve_documents_with_naive_anchors(
    query: str,
    *,
    top_docs: int = 5,
    top_k: int = 12,
    chunks_per_doc: int | None = None,
    retrieval_overrides: dict | None = None,
) -> list[dict]:
    settings = get_settings()
    chunk_limit = max(1, int(chunks_per_doc or 4))
    naive_anchor_chunks = retrieve_naive(
        query,
        top_k=max(8, int(settings.rag_top_k or 0)),
        retrieval_overrides=retrieval_overrides,
    )
    naive_anchor_docs = _chunks_to_document_evidences(naive_anchor_chunks, chunks_per_doc=chunk_limit)
    document_evidences = retrieve_documents(
        query,
        top_docs=top_docs,
        top_k=top_k,
        chunks_per_doc=chunk_limit,
        retrieval_overrides=retrieval_overrides,
    )
    # Start with vector anchors so QA-oriented RAG cannot have less raw evidence than
    # naive RAG, then merge broader document-level/re-ranked evidence.
    return _merge_document_evidence_payloads(naive_anchor_docs, document_evidences, chunks_per_doc=chunk_limit)


def _summarize_sources(docs: list[dict]) -> list[str]:
    seen: list[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        source = str(doc.get("source") or doc.get("file_name") or doc.get("title") or "").strip()
        if not source or source in seen:
            continue
        seen.append(source)
    return seen[:10]


def _reference_from_doc(doc: dict, *, index: int) -> dict:
    citation = doc.get("citation") if isinstance(doc.get("citation"), dict) else doc
    source = citation.get("source") or doc.get("source")
    return {
        "label": f"DOC {index}",
        "doc_id": doc.get("doc_id"),
        "title": citation.get("title") or citation.get("file_name") or doc.get("title") or doc.get("file_name"),
        "authors": citation.get("authors") or doc.get("authors") or [],
        "year": citation.get("year") or doc.get("year"),
        "journal": citation.get("journal") or doc.get("journal"),
        "doi": citation.get("doi") or doc.get("doi"),
        "file_name": citation.get("file_name") or doc.get("file_name") or (Path(source).name if source else None),
        "source": source,
    }


def build_reference_list(docs: list[dict]) -> list[dict]:
    references: list[dict] = []
    seen: set[str] = set()
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        reference = _reference_from_doc(doc, index=len(references) + 1)
        key = str(reference.get("doc_id") or reference.get("doi") or reference.get("title") or reference.get("source") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        reference["label"] = f"DOC {len(references) + 1}"
        references.append(reference)
    return references


def build_structured_rag_evidence(
    raw_result: Any,
    *,
    mode: str,
    fallback_docs: list[dict] | None = None,
) -> dict[str, Any]:
    docs: list[dict] = []
    rounds_count = 0

    if isinstance(raw_result, list):
        docs = [item for item in raw_result if isinstance(item, dict)]
    elif isinstance(raw_result, dict):
        evidences = raw_result.get("evidences")
        if isinstance(evidences, list):
            docs = [item for item in evidences if isinstance(item, dict)]
        elif isinstance(raw_result.get("documents"), list):
            docs = [item for item in raw_result.get("documents", []) if isinstance(item, dict)]
        rounds = raw_result.get("rounds")
        if isinstance(rounds, list):
            rounds_count = len(rounds)
    if not docs and fallback_docs:
        docs = [item for item in fallback_docs if isinstance(item, dict)]

    knowledge_evidence = {
        "mode": mode,
        "documents": docs,
        "doc_count": len(docs),
        "top_sources": _summarize_sources(docs),
        "references": build_reference_list(docs),
        "rounds_count": rounds_count,
    }
    return {
        "knowledge_evidence": knowledge_evidence,
    }


def rag_answer(
    query: str,
    top_k: int | None = None,
    thinking: bool = False,
    source_group: str | None = None,
    task_group: str | None = None,
    entity_tags: list[str] | None = None,
    retrieval_overrides: dict | None = None,
) -> dict:
    settings = get_settings()
    run_id = new_run_id()
    docs = retrieve(
        query,
        top_k=top_k or settings.rag_top_k,
        retrieval_overrides=retrieval_overrides,
    )
    if not docs:
        log_event(
            {
                "type": "rag_run",
                "run_id": run_id,
                "chain": "basic",
                "question": query,
                "success": False,
                "reason": "no_chunks",
                "retrieval_overrides": retrieval_overrides or {},
            }
        )
        return {
            "used_model": settings.active_llm_model,
            "answer": "No relevant knowledge chunks were retrieved. Please ingest or rebuild the knowledge base first.",
            "raw_result": [],
        }

    context_parts = []
    for doc in docs:
        source_name = Path(doc["source"]).name if doc.get("source") else "unknown"
        context_parts.append(f"[source: {source_name} | score={doc['score']:.4f}]\n{doc['text']}")
    context = "\n\n".join(context_parts)
    answer = DeepSeekClient().chat(query=query, context=context, thinking=thinking)
    log_event(
        {
            "type": "rag_run",
            "run_id": run_id,
            "chain": "basic",
            "question": query,
            "success": True,
            "retrieval_overrides": retrieval_overrides or {},
            "num_chunks": len(docs),
            "top_sources": list({Path(d.get("source", "")).name for d in docs if isinstance(d, dict)})[:10],
        }
    )
    return {
        "used_model": settings.active_llm_model,
        "answer": answer,
        "raw_result": docs,
    }


def naive_rag_answer(
    query: str,
    top_k: int | None = None,
    thinking: bool = False,
    source_group: str | None = None,
    task_group: str | None = None,
    entity_tags: list[str] | None = None,
    retrieval_overrides: dict | None = None,
) -> dict:
    settings = get_settings()
    run_id = new_run_id()
    docs = retrieve_naive(
        query,
        top_k=top_k or settings.rag_top_k,
        retrieval_overrides=retrieval_overrides,
    )
    if not docs:
        log_event(
            {
                "type": "rag_run",
                "run_id": run_id,
                "chain": "naive",
                "question": query,
                "success": False,
                "reason": "no_chunks",
                "retrieval_overrides": retrieval_overrides or {},
            }
        )
        return {
            "used_model": settings.active_llm_model,
            "answer": "No relevant knowledge chunks were retrieved. Please ingest or rebuild the knowledge base first.",
            "raw_result": [],
        }

    context = "\n\n".join((doc.get("text") or "") for doc in docs if isinstance(doc, dict))
    answer = DeepSeekClient().chat(query=query, context=context, thinking=thinking)
    log_event(
        {
            "type": "rag_run",
            "run_id": run_id,
            "chain": "naive",
            "question": query,
            "success": True,
            "retrieval_overrides": retrieval_overrides or {},
            "num_chunks": len(docs),
        }
    )
    return {
        "used_model": settings.active_llm_model,
        "answer": answer,
        "raw_result": docs,
    }


def naive_llm_filter_rag_answer(
    query: str,
    *,
    top_k: int | None = None,
    thinking: bool = False,
    retrieval_overrides: dict | None = None,
) -> dict:
    """
    Ablation path: naive chunk retrieval + LLM evidence filter + final synthesis.

    This intentionally does not use document-level retrieval, naive anchors,
    query refinement, or iterative review. It isolates the contribution of the
    LLM evidence filter/synthesis layer over the same naive candidate pool.
    """

    del thinking
    settings = get_settings()
    run_id = new_run_id()
    chunks = retrieve_naive(
        query,
        top_k=top_k or settings.rag_top_k,
        retrieval_overrides=retrieval_overrides,
    )
    if not chunks:
        log_event(
            {
                "type": "rag_run",
                "run_id": run_id,
                "chain": "naive_llm_filter",
                "question": query,
                "success": False,
                "reason": "no_chunks",
                "retrieval_overrides": retrieval_overrides or {},
            }
        )
        return {
            "used_model": settings.active_llm_model,
            "answer": "No relevant knowledge chunks were retrieved. Please ingest or rebuild the knowledge base first.",
            "raw_result": {
                "chain": "naive_llm_filter",
                "evidences": [],
                "candidate_chunks": [],
                "run_id": run_id,
            },
        }

    chunk_limit = max(1, int(top_k or settings.rag_top_k))
    candidate_evidences = _chunks_to_document_evidences(chunks, chunks_per_doc=chunk_limit)
    llm = DeepSeekClient()
    filtered_evidences, filter_debug = LLMEvidenceFilter(llm).filter(
        question=query,
        evidences=candidate_evidences,
        top_k=settings.rag_evidence_filter_top_k,
        min_score=settings.rag_evidence_filter_min_score,
        batch_size=settings.rag_evidence_filter_batch_size,
        max_chars=settings.rag_evidence_filter_max_chars,
    )
    final_evidences = _expand_selected_evidences(
        filtered_evidences or candidate_evidences[: max(1, int(settings.rag_evidence_filter_top_k))],
        candidate_evidences,
        max_chunks_per_doc=max(1, min(4, chunk_limit)),
        anchor_docs=2,
    )
    reviewer = IterativeReviewRAG(llm)
    final_context = _build_evidence_context(final_evidences, max_chars_per_chunk=1800)
    answer = reviewer._final_synthesis(question=query, context=final_context)
    allowed_max = len(final_evidences)
    if _has_invalid_doc_refs(answer, allowed_max=allowed_max):
        answer = reviewer._repair_citations(
            question=query,
            answer=answer,
            context=final_context,
            allowed_max=allowed_max,
        )

    log_event(
        {
            "type": "rag_run",
            "run_id": run_id,
            "chain": "naive_llm_filter",
            "question": query,
            "success": True,
            "retrieval_overrides": retrieval_overrides or {},
            "num_chunks": len(chunks),
            "num_docs": len(final_evidences),
        }
    )
    return {
        "used_model": settings.active_llm_model,
        "answer": answer,
        "raw_result": {
            "chain": "naive_llm_filter",
            "evidences": final_evidences,
            "candidate_evidences": candidate_evidences,
            "candidate_chunks": chunks,
            "evidence_filter": filter_debug,
            "run_id": run_id,
            "retrieval_top_k": top_k or settings.rag_top_k,
        },
    }


def _run_advanced_rag_answer(
    *,
    query: str,
    rag_engine,
    chain: str,
    top_docs: int | None = None,
    top_k_chunks: int | None = None,
    max_rounds: int | None = None,
    retrieval_overrides: dict | None = None,
    include_naive_anchors: bool = False,
) -> dict:
    settings = get_settings()
    document_retriever = retrieve_documents_with_naive_anchors if include_naive_anchors else retrieve_documents
    payload = rag_engine.answer(
        question=query,
        retrieve_fn=lambda q, top_docs, top_k, chunks_per_doc=None: document_retriever(
            q,
            top_docs=top_docs,
            top_k=top_k,
            chunks_per_doc=chunks_per_doc,
            retrieval_overrides=retrieval_overrides,
        ),
        max_rounds=max_rounds or settings.advanced_rag_max_rounds,
        top_docs=top_docs or settings.advanced_rag_top_docs,
        top_k_chunks=top_k_chunks or settings.advanced_rag_top_k_chunks,
    )
    raw = payload.get("raw_result", {}) if isinstance(payload, dict) else {}
    run_id = raw.get("run_id") if isinstance(raw, dict) else None
    log_event(
        {
            "type": "rag_run",
            "run_id": run_id or new_run_id(),
            "chain": chain,
            "question": query,
            "success": True,
            "retrieval_overrides": retrieval_overrides or {},
            "rag_rounds": len((raw or {}).get("rounds", []) if isinstance(raw, dict) else []),
            "num_docs": len((raw or {}).get("evidences", []) if isinstance(raw, dict) else []),
        }
    )
    return {
        "used_model": settings.active_llm_model,
        "answer": payload.get("answer", ""),
        "raw_result": payload.get("raw_result", {}),
    }


def qa_oriented_rag_answer(
    query: str,
    *,
    top_docs: int | None = None,
    top_k_chunks: int | None = None,
    max_rounds: int | None = None,
    retrieval_overrides: dict | None = None,
) -> dict:
    return _run_advanced_rag_answer(
        query=query,
        rag_engine=QAOrientedRAG(DeepSeekClient()),
        chain="qa_oriented",
        top_docs=top_docs,
        top_k_chunks=top_k_chunks,
        max_rounds=max_rounds,
        retrieval_overrides=retrieval_overrides,
        include_naive_anchors=True,
    )


def iterative_review_rag_answer(
    query: str,
    *,
    top_docs: int | None = None,
    top_k_chunks: int | None = None,
    max_rounds: int | None = None,
    retrieval_overrides: dict | None = None,
) -> dict:
    return _run_advanced_rag_answer(
        query=query,
        rag_engine=IterativeReviewRAG(DeepSeekClient()),
        chain="iterative_review",
        top_docs=top_docs,
        top_k_chunks=top_k_chunks,
        max_rounds=max_rounds,
        retrieval_overrides=retrieval_overrides,
    )


def advanced_rag_answer(
    query: str,
    *,
    top_docs: int | None = None,
    top_k_chunks: int | None = None,
    max_rounds: int | None = None,
    retrieval_overrides: dict | None = None,
) -> dict:
    """Backward-compatible alias for the QA-oriented RAG route."""
    return qa_oriented_rag_answer(
        query,
        top_docs=top_docs,
        top_k_chunks=top_k_chunks,
        max_rounds=max_rounds,
        retrieval_overrides=retrieval_overrides,
    )
