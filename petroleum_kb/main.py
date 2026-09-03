from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DEFAULT_CONFIG
from petroleum_kb.pipeline.build_index import build_vector_index
from petroleum_kb.pipeline.ingest import (
    ingest_directories,
    ingest_registered_sources,
    scan_directories,
    scan_registered_sources,
)
from petroleum_kb.pipeline.source_registry import list_sources, register_source
from petroleum_kb.retrieval.searcher import TagAwareSearcher


def _load_repo_dotenv() -> None:
    """Load repo-root .env for CLI convenience (HF cache paths, API keys, etc.)."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env", override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Petroleum knowledge base CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load, chunk, label and persist raw documents")
    ingest_parser.add_argument("--input_dir", action="append", type=Path, default=None)
    ingest_parser.add_argument("--use_registry", action="store_true")
    ingest_parser.add_argument("--pdf-parser-backend", choices=("pymupdf", "grobid"), default=None)
    ingest_parser.add_argument("--grobid-url", type=str, default=None)
    ingest_parser.add_argument("--grobid-timeout", type=int, default=None, help="GROBID request timeout seconds")
    ingest_parser.add_argument("--chunk-mode", choices=("char", "sentence", "scientific_sentence"), default=None)
    ingest_parser.add_argument("--chunk-token-size", type=int, default=None)
    ingest_parser.add_argument("--chunk-sentence-overlap", type=int, default=None)
    ingest_parser.add_argument(
        "--label-mode",
        choices=("none", "rule", "llm", "hybrid"),
        default="hybrid",
        help="Labeling strategy: none, rule only, llm only, or hybrid rule+llm",
    )
    ingest_parser.add_argument(
        "--enable-clustering",
        action="store_true",
        help="Run optional clustering summary after ingest (may require embeddings/model download).",
    )
    ingest_parser.add_argument(
        "--append",
        action="store_true",
        help="Append-merge into existing chunks.jsonl (incremental ingest). Without this, ingest overwrites chunks.jsonl.",
    )

    scan_parser = subparsers.add_parser("scan", help="Preview registered or direct source directories")
    scan_parser.add_argument("--input_dir", action="append", type=Path, default=None)
    scan_parser.add_argument("--use_registry", action="store_true")

    register_parser = subparsers.add_parser("register_source", help="Register an external knowledge source directory")
    register_parser.add_argument("--path", required=True, type=Path)
    register_parser.add_argument("--name", type=str, default=None)
    register_parser.add_argument("--source_group", type=str, default=None)
    register_parser.add_argument("--task_group", type=str, default=None)
    register_parser.add_argument("--notes", type=str, default=None)

    subparsers.add_parser("list_sources", help="List registered knowledge source directories")

    index_parser = subparsers.add_parser("build_index", help="Build FAISS or fallback vector index")
    index_parser.add_argument("--processed_file", type=Path, default=DEFAULT_CONFIG.processed_dir / "chunks.jsonl")
    index_parser.add_argument("--embedding-model", type=str, default=None)
    index_parser.add_argument("--local-files-only", action="store_true")
    index_parser.add_argument("--allow-hash-fallback", action="store_true")

    search_parser = subparsers.add_parser("search", help="Run tag-aware retrieval")
    search_parser.add_argument("--query", required=True, type=str)
    search_parser.add_argument("--top_k", type=int, default=5)
    search_parser.add_argument("--source_group", type=str, default=None)
    search_parser.add_argument("--task_group", type=str, default=None)
    search_parser.add_argument("--entity_tag", action="append", default=None)
    # Optional runtime overrides. If an index meta file exists, it will take precedence.
    search_parser.add_argument("--embedding-model", type=str, default=None)
    search_parser.add_argument("--local-files-only", action="store_true")
    search_parser.add_argument("--allow-hash-fallback", action="store_true")
    # Stage-2 reranker controls
    search_parser.add_argument("--enable-reranker", action="store_true")
    search_parser.add_argument("--rerank-model", type=str, default=None)
    search_parser.add_argument("--rerank-local-files-only", action="store_true")
    search_parser.add_argument("--retrieval-top-n", type=int, default=None)
    search_parser.add_argument("--enable-bm25", action="store_true")
    search_parser.add_argument("--vector-weight", type=float, default=None)
    search_parser.add_argument("--bm25-weight", type=float, default=None)

    search_docs_parser = subparsers.add_parser("search_docs", help="Retrieve and aggregate document-level evidence")
    search_docs_parser.add_argument("--query", required=True, type=str)
    search_docs_parser.add_argument("--top_docs", type=int, default=None)
    search_docs_parser.add_argument("--top_k", type=int, default=8, help="How many chunks to retrieve before grouping")
    search_docs_parser.add_argument("--chunks_per_doc", type=int, default=None)
    search_docs_parser.add_argument("--source_group", type=str, default=None)
    search_docs_parser.add_argument("--task_group", type=str, default=None)
    search_docs_parser.add_argument("--entity_tag", action="append", default=None)
    # Optional runtime overrides. If an index meta file exists, it will take precedence
    # for model name/dim, but local/offline flags are still respected.
    search_docs_parser.add_argument("--embedding-model", type=str, default=None)
    search_docs_parser.add_argument("--local-files-only", action="store_true")
    search_docs_parser.add_argument("--allow-hash-fallback", action="store_true")
    search_docs_parser.add_argument("--enable-reranker", action="store_true")
    search_docs_parser.add_argument("--rerank-model", type=str, default=None)
    search_docs_parser.add_argument("--rerank-local-files-only", action="store_true")
    search_docs_parser.add_argument("--retrieval-top-n", type=int, default=None)
    search_docs_parser.add_argument("--enable-bm25", action="store_true")
    search_docs_parser.add_argument("--vector-weight", type=float, default=None)
    search_docs_parser.add_argument("--bm25-weight", type=float, default=None)
    return parser


def _apply_runtime_config(args) -> None:
    if getattr(args, "pdf_parser_backend", None):
        if bool(getattr(DEFAULT_CONFIG, "strict_pdf_parser", False)) and args.pdf_parser_backend != "grobid":
            raise SystemExit("strict_pdf_parser=true: text pymupdf，text --pdf-parser-backend grobid")
        DEFAULT_CONFIG.pdf_parser_backend = args.pdf_parser_backend
    if getattr(args, "grobid_url", None):
        DEFAULT_CONFIG.grobid_url = args.grobid_url
    if getattr(args, "grobid_timeout", None) is not None:
        DEFAULT_CONFIG.grobid_timeout_seconds = int(args.grobid_timeout)
    if getattr(args, "chunk_mode", None):
        if bool(getattr(DEFAULT_CONFIG, "strict_scientific_sentence", False)) and args.chunk_mode != "scientific_sentence":
            raise SystemExit("strict_scientific_sentence=true: text scientific_sentence chunk_mode")
        DEFAULT_CONFIG.chunk_mode = args.chunk_mode
    if getattr(args, "chunk_token_size", None):
        DEFAULT_CONFIG.chunk_token_size = args.chunk_token_size
    if getattr(args, "chunk_sentence_overlap", None):
        DEFAULT_CONFIG.chunk_sentence_overlap = args.chunk_sentence_overlap
    if getattr(args, "embedding_model", None):
        DEFAULT_CONFIG.embedding_model_name = args.embedding_model
    if getattr(args, "local_files_only", False):
        DEFAULT_CONFIG.embedding_local_files_only = True
    if getattr(args, "allow_hash_fallback", False):
        DEFAULT_CONFIG.allow_hash_fallback = True
    if getattr(args, "enable_reranker", False):
        DEFAULT_CONFIG.enable_reranker = True
    if getattr(args, "rerank_model", None):
        DEFAULT_CONFIG.rerank_model_name = args.rerank_model
    if getattr(args, "rerank_local_files_only", False):
        DEFAULT_CONFIG.rerank_local_files_only = True
    if getattr(args, "retrieval_top_n", None) is not None:
        DEFAULT_CONFIG.retrieval_top_n = int(args.retrieval_top_n)
    if getattr(args, "enable_bm25", False):
        DEFAULT_CONFIG.enable_bm25 = True
    if getattr(args, "vector_weight", None) is not None:
        DEFAULT_CONFIG.vector_weight = float(args.vector_weight)
    if getattr(args, "bm25_weight", None) is not None:
        DEFAULT_CONFIG.bm25_weight = float(args.bm25_weight)
    if getattr(args, "enable_clustering", False):
        DEFAULT_CONFIG.enable_clustering = True


def run() -> None:
    _load_repo_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        _apply_runtime_config(args)
        if args.use_registry:
            payload = ingest_registered_sources(DEFAULT_CONFIG, label_mode=args.label_mode, append=bool(getattr(args, "append", False)))
        else:
            input_dirs = args.input_dir or [DEFAULT_CONFIG.raw_dir]
            payload = ingest_directories(input_dirs, DEFAULT_CONFIG, label_mode=args.label_mode, append=bool(getattr(args, "append", False)))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "scan":
        if args.use_registry:
            payload = scan_registered_sources(DEFAULT_CONFIG)
        else:
            input_dirs = args.input_dir or [DEFAULT_CONFIG.raw_dir]
            payload = scan_directories(input_dirs)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "register_source":
        payload = register_source(
            config=DEFAULT_CONFIG,
            path=args.path,
            name=args.name,
            source_group_hint=args.source_group,
            task_group_hint=args.task_group,
            notes=args.notes,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "list_sources":
        print(json.dumps(list_sources(DEFAULT_CONFIG), ensure_ascii=False, indent=2))
        return

    if args.command == "build_index":
        _apply_runtime_config(args)
        print(json.dumps(build_vector_index(args.processed_file, DEFAULT_CONFIG), ensure_ascii=False, indent=2))
        return

    if args.command == "search":
        _apply_runtime_config(args)
        searcher = TagAwareSearcher.from_config(DEFAULT_CONFIG)
        results = searcher.search(
            query=args.query,
            top_k=args.top_k,
            source_group=args.source_group,
            task_group=args.task_group,
            entity_tags=args.entity_tag,
        )
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        return

    if args.command == "search_docs":
        _apply_runtime_config(args)
        searcher = TagAwareSearcher.from_config(DEFAULT_CONFIG)
        evidences = searcher.search_documents(
            query=args.query,
            top_docs=args.top_docs,
            top_k=args.top_k,
            chunks_per_doc=args.chunks_per_doc,
            source_group=args.source_group,
            task_group=args.task_group,
            entity_tags=args.entity_tag,
        )
        print(json.dumps([e.to_dict() for e in evidences], ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    run()
