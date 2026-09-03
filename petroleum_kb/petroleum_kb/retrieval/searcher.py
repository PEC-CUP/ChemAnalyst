from __future__ import annotations

import json

import numpy as np
from pathlib import Path

try:
    from config import KBConfig
except ImportError:  # pragma: no cover
    from petroleum_kb.config import KBConfig
from ..indexing.embedder import TextEmbedder
from ..indexing.vector_store import VectorStore
from .reranker import CrossEncoderReranker, rerank_score
from .bm25 import BM25Index, minmax_normalize
from .evidence import EvidenceAggregator, DocumentEvidence
from ..schemas import ChunkMetadata, SearchResult


DEFAULT_SOURCE_WEIGHTS = {
    "text": 1.00,
    "text": 0.92,
    "text": 0.85,
    "text": 0.75,
    "text": 0.72,
    "text": 0.60,
    "text": 0.50,
}


class TagAwareSearcher:
    def __init__(
        self,
        config: KBConfig,
        metadata_records: dict[str, dict],
        embedder: TextEmbedder,
        backend: str,
        index_obj,
        chunk_ids: list[str],
    ) -> None:
        self.config = config
        self.metadata_records = metadata_records
        self.embedder = embedder
        self.backend = backend
        self.index_obj = index_obj
        self.chunk_ids = chunk_ids
        self._bm25: BM25Index | None = None

    @classmethod
    def from_config(cls, config: KBConfig) -> "TagAwareSearcher":
        metadata_path = config.processed_dir / "chunks.jsonl"
        metadata_records = {}
        with metadata_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                metadata_records[item["chunk_id"]] = item
        store = VectorStore(config.index_dir, config.index_name)
        backend, index_obj, chunk_ids = store.load()
        index_meta = store.read_meta()

        # Fail fast if the index id list doesn't match the processed chunk file.
        # This typically happens when chunks.jsonl was regenerated but build_index
        # was not rerun (or built from a different processed_file).
        if chunk_ids:
            missing = sum(1 for cid in chunk_ids if cid not in metadata_records)
            if missing:
                ratio = missing / max(len(chunk_ids), 1)
                # If most ids are missing, it's almost certainly a stale index.
                if ratio >= 0.2:
                    raise RuntimeError(
                        "Index ids do not match the processed chunks file.\n"
                        f"- chunks: {metadata_path}\n"
                        f"- index ids: {store.meta_path}\n"
                        f"- missing ids: {missing}/{len(chunk_ids)}\n"
                        "text chunks.jsonl text build_index（text processed_file）。"
                    )

        # Use the embedding settings recorded during build_index to avoid
        # dimension mismatch (e.g. config default changed after indexing).
        embedding_model_name = index_meta.get("embedding_model_name") or config.embedding_model_name
        # Runtime-safe overrides:
        # - If user explicitly forces local_files_only / allow_hash_fallback at runtime, respect it.
        # - Otherwise fall back to the index meta (then config).
        embedding_local_files_only = bool(
            bool(getattr(config, "embedding_local_files_only", False))
            or bool(index_meta.get("embedding_local_files_only", False))
        )
        allow_hash_fallback = bool(
            bool(getattr(config, "allow_hash_fallback", False))
            or bool(index_meta.get("allow_hash_fallback", False))
        )

        embedder = TextEmbedder(
            embedding_model_name,
            config.embedding_fallback_dim,
            allow_hash_fallback=allow_hash_fallback,
            local_files_only=embedding_local_files_only,
        )

        # Validate embedder dimension matches index dimension and fail fast with a clear message.
        index_dim = None
        if backend == "faiss":
            index_dim = getattr(index_obj, "d", None)
        else:
            try:
                index_dim = int(index_obj.shape[1])
            except Exception:
                index_dim = None
        if index_dim is not None and int(embedder.dimension) != int(index_dim):
            raise RuntimeError(
                "Embedding dimension does not match the loaded index. "
                f" embedder={embedder.dimension}, index={index_dim}. "
                "Rebuild the index with the active embedding configuration."
            )
        return cls(config, metadata_records, embedder, backend, index_obj, chunk_ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_group: str | None = None,
        task_group: str | None = None,
        entity_tags: list[str] | None = None,
    ) -> list[SearchResult]:
        top_n = max(int(self.config.retrieval_top_n), int(top_k) * 5, 20)
        candidate_ids, candidate_scores, retrieval_debug = self._retrieve_candidates_water_rag(query=query, top_n=top_n)

        # Runtime RAG must not use generated labels for filtering or ranking.
        # source_group/task_group/entity_tags/method_tags/evidence_type are
        # retained only as raw metadata in the processed file.
        staged: list[dict] = []
        for chunk_id, semantic_score in zip(candidate_ids, candidate_scores):
            item = self.metadata_records.get(chunk_id)
            if item is None:
                continue
            metadata = ChunkMetadata(**item["metadata"])
            staged.append(
                {
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "metadata": metadata,
                    "vector_score": float(semantic_score),
                    "tag_score": 0.0,
                    "source_score": 0.0,
                }
            )

        # Stage-2: model-based reranker (optional). It re-scores only the topN candidates.
        rerank_scores: list[float] | None = None
        reranker_error: str | None = None
        if getattr(self.config, "enable_reranker", False) and staged:
            try:
                reranker = CrossEncoderReranker(
                    getattr(self.config, "rerank_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                    local_files_only=bool(getattr(self.config, "rerank_local_files_only", False)),
                )
                max_chars = int(getattr(self.config, "rerank_max_chars", 1800))
                texts = [s["text"][:max_chars] for s in staged]
                rerank_scores = reranker.score(query, texts)
            except Exception as exc:
                # Fail-soft: keep vector ranking when reranker is unavailable.
                reranker_error = str(exc)
                rerank_scores = None

        results: list[SearchResult] = []
        for i, s in enumerate(staged):
            # If reranker is enabled and available, use it as the primary semantic score for fusion.
            semantic_for_fusion = s["vector_score"]
            debug: dict = {"backend": self.backend, **(retrieval_debug or {})}
            if rerank_scores is not None:
                rr = float(rerank_scores[i])
                semantic_for_fusion = rr
                debug["rerank_score"] = rr
                debug["vector_score"] = s["vector_score"]
                debug["rerank_model"] = getattr(self.config, "rerank_model_name", None)
            elif reranker_error is not None:
                debug["rerank_error"] = reranker_error
            debug["retrieval_top_n"] = top_n
            debug["bm25"] = bool(getattr(self.config, "enable_bm25", False))

            final_score = rerank_score(
                semantic_similarity=float(semantic_for_fusion),
                tag_match=0.0,
                source_weight=0.0,
                alpha=1.0,
                beta=0.0,
                gamma=0.0,
            )
            results.append(
                SearchResult(
                    chunk_id=s["chunk_id"],
                    text=s["text"],
                    metadata=s["metadata"],
                    semantic_score=float(s["vector_score"]),
                    tag_score=float(s["tag_score"]),
                    source_score=float(s["source_score"]),
                    final_score=float(final_score),
                    debug=debug,
                )
            )

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]

    def naive_search(
        self,
        query: str,
        top_k: int = 5,
        source_group: str | None = None,
        task_group: str | None = None,
        entity_tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        True naive RAG retrieval:
        - one vector query
        - no BM25
        - no reranker
        - no tag/source fusion
        - final ranking == raw vector similarity
        """
        qv = self.embedder.embed_query(query).astype(np.float32)
        candidate_ids, candidate_scores = self._retrieve_candidates(qv, max(int(top_k), 1))
        results: list[SearchResult] = []
        for chunk_id, semantic_score in zip(candidate_ids, candidate_scores):
            item = self.metadata_records.get(chunk_id)
            if item is None:
                continue
            metadata = ChunkMetadata(**item["metadata"])
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=item["text"],
                    metadata=metadata,
                    semantic_score=float(semantic_score),
                    tag_score=0.0,
                    source_score=0.0,
                    final_score=float(semantic_score),
                    debug={"backend": self.backend, "mode": "naive", "bm25": False, "reranker": False},
                )
            )
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]

    def search_documents(
        self,
        query: str,
        *,
        top_docs: int | None = None,
        top_k: int = 8,
        chunks_per_doc: int | None = None,
        source_group: str | None = None,
        task_group: str | None = None,
        entity_tags: list[str] | None = None,
    ) -> list[DocumentEvidence]:
        """
        Document-level evidence aggregation:
        - Retrieve chunks (same as search)
        - Group by document
        - Output citations + top supporting chunks per doc
        """

        rs = self.search(
            query=query,
            top_k=top_k,
        )
        agg = EvidenceAggregator(chunks_per_doc=int(chunks_per_doc or getattr(self.config, "chunks_per_doc", 3)))
        return agg.aggregate(rs, top_docs=int(top_docs or getattr(self.config, "docs_top_k", 5)))

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is not None:
            return self._bm25
        bm25_path: Path = Path(getattr(self.config, "bm25_index_file", self.config.index_dir / f"{self.config.index_name}.bm25.pkl"))
        persist = bool(getattr(self.config, "bm25_persist", True))

        ids = sorted(self.metadata_records.keys())
        fp = BM25Index.fingerprint(ids)

        if persist and bm25_path.exists():
            try:
                idx, saved_fp = BM25Index.load(bm25_path)
                if saved_fp == fp and idx.doc_ids:
                    self._bm25 = idx
                    return self._bm25
            except Exception:
                # Corrupted or stale file -> rebuild below.
                pass

        texts = [self.metadata_records[cid].get("text", "") for cid in ids]
        self._bm25 = BM25Index().build(doc_ids=ids, texts=texts)
        if persist:
            try:
                self._bm25.save(bm25_path, fingerprint=fp)
            except Exception:
                # Fail-soft: persistence is an optimization.
                pass
        return self._bm25

    def _retrieve_candidates_water_rag(self, *, query: str, top_n: int) -> tuple[list[str], list[float], dict]:
        """
        RAG retrieval:
        - Multi-route recall: vector + BM25 (optional)
        - Merge candidates and keep vector_score for downstream rerank/debug
        """

        # NOTE: Query refinement is handled by the app-level QueryRefinementAgent
        # (and/or the iterative review route), not inside the retrieval backend.
        queries = [query]

        # 2) Vector route: union top_n per query, keep max vector score.
        vec_scores: dict[str, float] = {}
        for q in queries:
            qv = self.embedder.embed_query(q).astype(np.float32)
            cids, cscores = self._retrieve_candidates(qv, top_n)
            for cid, sc in zip(cids, cscores):
                prev = vec_scores.get(cid)
                if prev is None or sc > prev:
                    vec_scores[cid] = float(sc)

        # 3) BM25 route (optional).
        bm25_enabled = bool(getattr(self.config, "enable_bm25", False))
        bm25_scores: dict[str, float] = {}
        if bm25_enabled:
            bm25 = self._ensure_bm25()
            for q in queries:
                for cid, sc in bm25.search(q, top_n=top_n):
                    prev = bm25_scores.get(cid)
                    if prev is None or sc > prev:
                        bm25_scores[cid] = float(sc)

        # 4) Merge candidates. We prioritize union coverage.
        candidate_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        if not candidate_ids:
            return [], [], {
                "queries": queries,
                "vector_candidate_count": len(vec_scores),
                "bm25_candidate_count": len(bm25_scores),
                "merged_candidate_count": 0,
            }

        vec_norm = minmax_normalize(vec_scores)
        bm_norm = minmax_normalize(bm25_scores)
        vw = float(getattr(self.config, "vector_weight", 0.75))
        bw = float(getattr(self.config, "bm25_weight", 0.25))
        if not bm25_enabled:
            vw, bw = 1.0, 0.0
        else:
            s = vw + bw
            if s > 0:
                vw, bw = vw / s, bw / s

        combined: list[tuple[str, float, float]] = []
        for cid in candidate_ids:
            cn = vw * float(vec_norm.get(cid, 0.0)) + bw * float(bm_norm.get(cid, 0.0))
            combined.append((cid, float(cn), float(vec_scores.get(cid, 0.0))))

        combined.sort(key=lambda x: x[1], reverse=True)
        trimmed = combined[: int(max(top_n, 20))]
        retrieval_debug = {
            "queries": queries,
            "vector_candidate_count": len(vec_scores),
            "bm25_candidate_count": len(bm25_scores),
            "merged_candidate_count": len(candidate_ids),
            "vector_weight": vw,
            "bm25_weight": bw,
            "bm25_enabled": bm25_enabled,
        }
        return [cid for cid, _, _ in trimmed], [vscore for _, _, vscore in trimmed], retrieval_debug

    def _retrieve_candidates(self, query_vector: np.ndarray, top_n: int) -> tuple[list[str], list[float]]:
        if self.backend == "faiss":
            scores, indices = self.index_obj.search(query_vector.reshape(1, -1), top_n)
            valid = [(self.chunk_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
            return [cid for cid, _ in valid], [score for _, score in valid]

        embeddings = self.index_obj
        scores = embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.chunk_ids[idx] for idx in top_indices], [float(scores[idx]) for idx in top_indices]

