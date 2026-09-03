from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from config import KBConfig
except ImportError:  # pragma: no cover
    from petroleum_kb.config import KBConfig
from ..indexing.embedder import TextEmbedder
from ..indexing.vector_store import VectorStore


def build_vector_index(processed_file: Path, config: KBConfig) -> dict:
    items = []
    with processed_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    texts = [item["text"] for item in items]
    chunk_ids = [item["chunk_id"] for item in items]

    try:
        embedder = TextEmbedder(
            config.embedding_model_name,
            config.embedding_fallback_dim,
            allow_hash_fallback=config.allow_hash_fallback,
            local_files_only=config.embedding_local_files_only,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"embedding backend init failed: {exc}",
            "processed_file": str(processed_file),
            "embedding_model_name": config.embedding_model_name,
            "embedding_local_files_only": config.embedding_local_files_only,
            "allow_hash_fallback": config.allow_hash_fallback,
        }

    try:
        print(f"[build_index] embedding texts: count={len(texts)} backend={embedder.backend} model={config.embedding_model_name}")
        embeddings = embedder.embed_texts(texts, show_progress=True, batch_size=32)
    except Exception as exc:
        return {
            "status": "error",
            "error": f"embedding failed: {exc}",
            "processed_file": str(processed_file),
            "embedding_backend": embedder.backend,
            "embedding_model_name": config.embedding_model_name,
            "embedding_local_files_only": config.embedding_local_files_only,
            "allow_hash_fallback": config.allow_hash_fallback,
        }

    store = VectorStore(config.index_dir, config.index_name)
    persisted = store.save(embeddings, chunk_ids)
    # Persist how this index was built so retrieval can always reproduce the same embedder.
    # This avoids FAISS dimension mismatch when config defaults change later.
    store.write_meta(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "index_name": config.index_name,
            "index_backend": persisted.get("backend"),
            "num_vectors": persisted.get("num_vectors"),
            "embedding_backend": embedder.backend,
            "embedding_model_name": config.embedding_model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_local_files_only": bool(config.embedding_local_files_only),
            "allow_hash_fallback": bool(config.allow_hash_fallback),
            "processed_file": str(processed_file),
        }
    )
    return {
        "status": "success",
        "processed_file": str(processed_file),
        "index_dir": str(config.index_dir),
        "embedding_backend": embedder.backend,
        "embedding_model_name": config.embedding_model_name,
        "embedding_local_files_only": config.embedding_local_files_only,
        "allow_hash_fallback": config.allow_hash_fallback,
        "warning": "text hash text，text" if embedder.backend == "hash" else None,
        **persisted,
    }
