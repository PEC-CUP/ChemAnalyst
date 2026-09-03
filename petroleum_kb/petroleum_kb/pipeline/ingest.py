from __future__ import annotations

import json
from pathlib import Path

try:
    from config import KBConfig
except ImportError:  # pragma: no cover
    from petroleum_kb.config import KBConfig
from ..indexing.embedder import TextEmbedder
from ..io.chunker import chunk_document
from ..io.loaders import load_documents_from_dirs, load_documents_from_sources
from ..labeling.cluster_labeler import suggest_cluster_labels
from ..labeling.llm_labeler import llm_label_chunk
from ..labeling.rule_labeler import build_rule_labels
from ..schemas import ChunkMetadata, DocumentChunk, LoadedDocument
from ..utils import ensure_dir, stable_chunk_uid, write_jsonl
from .source_registry import load_source_registry


def ingest_directory(input_dir: Path, config: KBConfig) -> dict:
    return ingest_directories([input_dir], config)


def ingest_directories(
    input_dirs: list[Path],
    config: KBConfig,
    label_mode: str = "hybrid",
    *,
    append: bool = False,
) -> dict:
    load_errors: list[dict] = []
    if bool(getattr(config, "strict_pdf_parser", False)) and str(config.pdf_parser_backend).strip().lower() != "grobid":
        raise ValueError("strict_pdf_parser=true: pdf_parser_backend text grobid")
    if bool(getattr(config, "strict_scientific_sentence", False)) and str(config.chunk_mode).strip().lower() != "scientific_sentence":
        raise ValueError("strict_scientific_sentence=true: chunk_mode text scientific_sentence")
    if bool(getattr(config, "strict_scientific_sentence", False)):
        model_name = str(getattr(config, "scientific_sentence_model", "")).strip()
        if model_name not in {"en_core_sci_md", "en_core_web_sm"}:
            raise ValueError("strict_scientific_sentence=true: scientific_sentence_model text en_core_sci_md text en_core_web_sm")
    documents = load_documents_from_dirs(
        input_dirs,
        pdf_parser_backend=config.pdf_parser_backend,
        grobid_url=config.grobid_url,
        grobid_timeout_seconds=getattr(config, "grobid_timeout_seconds", 300),
        strict_pdf_parser=bool(getattr(config, "strict_pdf_parser", False)),
        errors=load_errors,
    )
    return _persist_documents(
        documents=documents,
        config=config,
        mode="directories",
        source_dirs=[str(path) for path in input_dirs],
        label_mode=label_mode,
        load_errors=load_errors,
        append=append,
    )


def ingest_registered_sources(config: KBConfig, label_mode: str = "hybrid", *, append: bool = False) -> dict:
    sources = load_source_registry(config)
    load_errors: list[dict] = []
    if bool(getattr(config, "strict_pdf_parser", False)) and str(config.pdf_parser_backend).strip().lower() != "grobid":
        raise ValueError("strict_pdf_parser=true: pdf_parser_backend text grobid")
    if bool(getattr(config, "strict_scientific_sentence", False)) and str(config.chunk_mode).strip().lower() != "scientific_sentence":
        raise ValueError("strict_scientific_sentence=true: chunk_mode text scientific_sentence")
    if bool(getattr(config, "strict_scientific_sentence", False)):
        model_name = str(getattr(config, "scientific_sentence_model", "")).strip()
        if model_name not in {"en_core_sci_md", "en_core_web_sm"}:
            raise ValueError("strict_scientific_sentence=true: scientific_sentence_model text en_core_sci_md text en_core_web_sm")
    documents = load_documents_from_sources(
        sources,
        pdf_parser_backend=config.pdf_parser_backend,
        grobid_url=config.grobid_url,
        grobid_timeout_seconds=getattr(config, "grobid_timeout_seconds", 300),
        strict_pdf_parser=bool(getattr(config, "strict_pdf_parser", False)),
        errors=load_errors,
    )
    return _persist_documents(
        documents=documents,
        config=config,
        mode="registry",
        source_dirs=[item.path for item in sources if item.enabled],
        label_mode=label_mode,
        load_errors=load_errors,
        append=append,
    )


def scan_directories(input_dirs: list[Path]) -> dict:
    documents = load_documents_from_dirs(input_dirs)
    return _scan_documents(
        documents=documents,
        mode="directories",
        source_dirs=[str(path) for path in input_dirs],
    )


def scan_registered_sources(config: KBConfig) -> dict:
    sources = load_source_registry(config)
    documents = load_documents_from_sources(sources)
    return _scan_documents(
        documents=documents,
        mode="registry",
        source_dirs=[item.path for item in sources if item.enabled],
        registry_file=str(config.source_registry_file),
    )


def _scan_documents(
    *,
    documents: list[LoadedDocument],
    mode: str,
    source_dirs: list[str],
    registry_file: str | None = None,
) -> dict:
    suffix_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for doc in documents:
        suffix = Path(doc.file_name).suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        source_name = doc.source_name or Path(doc.source).parent.name or "text"
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
    payload = {
        "status": "success",
        "mode": mode,
        "documents": len(documents),
        "source_dirs": source_dirs,
        "suffix_counts": suffix_counts,
        "source_counts": source_counts,
    }
    if registry_file:
        payload["registry_file"] = registry_file
    return payload


def _persist_documents(
    *,
    documents: list[LoadedDocument],
    config: KBConfig,
    mode: str,
    source_dirs: list[str],
    label_mode: str,
    load_errors: list[dict] | None = None,
    append: bool = False,
) -> dict:
    ensure_dir(config.processed_dir)
    ensure_dir(config.metadata_dir)

    total_docs = len(documents)
    chunks: list[DocumentChunk] = []
    for doc_idx, document in enumerate(documents, start=1):
        print(f"[ingest] ({doc_idx}/{total_docs}) chunking: {document.file_name}")
        try:
            raw_chunks = chunk_document(
                document,
                config.chunk_size,
                config.chunk_overlap,
                chunk_mode=config.chunk_mode,
                chunk_token_size=config.chunk_token_size,
                chunk_sentence_overlap=config.chunk_sentence_overlap,
                scientific_sentence_model=config.scientific_sentence_model,
                strict_scientific_sentence=bool(getattr(config, "strict_scientific_sentence", False)),
            )
        except Exception as exc:
            msg = str(exc)
            print(f"[ingest] ERROR chunking {document.file_name}: {msg}")
            if load_errors is not None:
                load_errors.append(
                    {
                        "stage": "chunk",
                        "path": str(document.source),
                        "file_name": document.file_name,
                        "suffix": Path(document.file_name).suffix.lower(),
                        "error": msg,
                        "chunk_mode": config.chunk_mode,
                        "scientific_sentence_model": getattr(config, "scientific_sentence_model", None),
                        "strict_scientific_sentence": bool(getattr(config, "strict_scientific_sentence", False)),
                    }
                )
            continue

        for raw_chunk in raw_chunks:
            if label_mode == "none":
                merged = _default_labels()
            else:
                rule_labels = build_rule_labels(
                    raw_chunk["text"],
                    raw_chunk["source"],
                    raw_chunk["file_name"],
                    source_group_hint=document.source_group_hint,
                    task_group_hint=document.task_group_hint,
                )
                llm_seed = rule_labels if label_mode == "hybrid" else None
                llm_labels = (
                    llm_label_chunk(
                        raw_chunk["text"],
                        raw_chunk["source"],
                        raw_chunk["file_name"],
                        rule_labels=llm_seed,
                    )
                    if label_mode in {"llm", "hybrid"}
                    else {}
                )
                merged = _merge_labels(rule_labels=rule_labels, llm_labels=llm_labels, label_mode=label_mode)
            metadata = ChunkMetadata(
                source=raw_chunk["source"],
                file_name=raw_chunk["file_name"],
                chunk_id=raw_chunk["chunk_index"],
                page=raw_chunk.get("page"),
                section=raw_chunk.get("section"),
                source_group=merged["source_group"],
                task_group=merged["task_group"],
                entity_tags=merged["entity_tags"],
                method_tags=merged["method_tags"],
                evidence_type=merged["evidence_type"],
                confidence=merged["confidence"],
                source_name=document.source_name,
                title=document.title,
                authors=document.authors,
                year=document.year,
                journal=document.journal,
                doi=document.doi,
                abstract=document.abstract,
                document_type=document.document_type,
                chunk_mode=raw_chunk.get("chunk_mode"),
                chunk_token_size=raw_chunk.get("chunk_token_size"),
                chunk_overlap=raw_chunk.get("chunk_overlap"),
                sentence_start=raw_chunk.get("sentence_start"),
                sentence_end=raw_chunk.get("sentence_end"),
            )
            chunk_uid = stable_chunk_uid(metadata.source, metadata.chunk_id, raw_chunk["text"])
            chunks.append(DocumentChunk(chunk_id=chunk_uid, text=raw_chunk["text"], metadata=metadata))

    processed_path = config.processed_dir / "chunks.jsonl"
    metadata_path = config.metadata_dir / "metadata.json"

    existing_chunks: dict[str, dict] = {}
    if append and processed_path.exists():
        try:
            with processed_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    cid = item.get("chunk_id")
                    if isinstance(cid, str) and cid:
                        existing_chunks[cid] = item
        except Exception as exc:
            print(f"[ingest] WARN: failed to load existing chunks for append: {exc}")
            existing_chunks = {}

    new_payload = [chunk.to_dict() for chunk in chunks]
    merged_payload = new_payload
    append_merge = False
    if existing_chunks:
        for item in new_payload:
            cid = item.get("chunk_id")
            if isinstance(cid, str) and cid:
                existing_chunks[cid] = item
        merged_payload = list(existing_chunks.values())
        append_merge = True

    if append_merge:
        print(
            f"[ingest] writing chunks (append merge) -> {processed_path} "
            f"(merged={len(merged_payload)}; new={len(new_payload)})"
        )
    else:
        print(f"[ingest] writing chunks -> {processed_path} (count={len(merged_payload)})")
    write_jsonl(processed_path, merged_payload)

    if append_merge:
        print(f"[ingest] writing metadata (append merge) -> {metadata_path} (count={len(merged_payload)})")
    else:
        print(f"[ingest] writing metadata -> {metadata_path} (count={len(merged_payload)})")
    metadata_path.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cluster_payload = None
    if bool(getattr(config, "enable_clustering", False)):
        print("[ingest] building cluster summary (optional)")
        cluster_payload = _build_cluster_payload(chunks, config)
    cluster_summary_file = None
    if cluster_payload is not None:
        cluster_summary_path = config.metadata_dir / "cluster_summary.json"
        cluster_summary_path.write_text(
            json.dumps(cluster_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        cluster_summary_file = str(cluster_summary_path)

    payload = {
        "status": "success",
        "mode": mode,
        "label_mode": label_mode,
        "documents": len(documents),
        "chunks": len(merged_payload),
        "processed_file": str(processed_path),
        "metadata_file": str(metadata_path),
        "failed_files": load_errors or [],
        "source_dirs": source_dirs,
        "pdf_parser_backend": config.pdf_parser_backend,
        "chunk_mode": config.chunk_mode,
        "chunk_token_size": config.chunk_token_size if config.chunk_mode != "char" else None,
        "chunk_sentence_overlap": config.chunk_sentence_overlap if config.chunk_mode != "char" else None,
        "append_merge": bool(append_merge),
    }
    if load_errors:
        failures_path = config.metadata_dir / "ingest_failures.jsonl"
        write_jsonl(failures_path, load_errors)
        payload["failures_file"] = str(failures_path)
    if cluster_summary_file:
        payload["cluster_summary_file"] = cluster_summary_file
        payload["cluster_count"] = int(cluster_payload.get("cluster_count", 0))
        payload["cluster_embedding_backend"] = cluster_payload.get("embedding_backend")
    return payload


def _default_labels() -> dict:
    return {
        "source_group": "text",
        "task_group": "text",
        "entity_tags": [],
        "method_tags": [],
        "evidence_type": "text",
        "confidence": 0.0,
    }


def _merge_labels(*, rule_labels: dict, llm_labels: dict, label_mode: str) -> dict:
    if label_mode == "rule":
        return rule_labels
    if label_mode == "llm":
        return {**_default_labels(), **llm_labels}
    return {**rule_labels, **llm_labels}


def _build_cluster_payload(chunks: list[DocumentChunk], config: KBConfig) -> dict | None:
    if len(chunks) < 2:
        return None

    try:
        embedder = TextEmbedder(
            config.embedding_model_name,
            config.embedding_fallback_dim,
            allow_hash_fallback=config.allow_hash_fallback,
            local_files_only=config.embedding_local_files_only,
        )
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"cluster embedding unavailable: {exc}",
            "embedding_model_name": config.embedding_model_name,
            "chunk_count": len(chunks),
            "cluster_count": 0,
            "singleton_count": len(chunks),
            "clusters": [],
        }
    try:
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"cluster embedding failed: {exc}",
            "embedding_backend": getattr(embedder, "backend", "unknown"),
            "embedding_model_name": config.embedding_model_name,
            "chunk_count": len(chunks),
            "cluster_count": 0,
            "singleton_count": len(chunks),
            "clusters": [],
        }
    cluster_items = suggest_cluster_labels(
        [chunk.to_dict() for chunk in chunks],
        embeddings,
    )
    singleton_count = len(chunks) - sum(int(item["size"]) for item in cluster_items)
    return {
        "status": "success",
        "embedding_backend": embedder.backend,
        "chunk_count": len(chunks),
        "cluster_count": len(cluster_items),
        "singleton_count": singleton_count,
        "clusters": cluster_items,
    }
