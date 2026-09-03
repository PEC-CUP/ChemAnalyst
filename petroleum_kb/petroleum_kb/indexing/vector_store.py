from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..utils import ensure_dir


class VectorStore:
    def __init__(self, index_dir: Path, index_name: str) -> None:
        self.index_dir = ensure_dir(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f"{index_name}.faiss"
        self.npy_path = self.index_dir / f"{index_name}.npy"
        self.meta_path = self.index_dir / f"{index_name}_ids.json"
        self.info_path = self.index_dir / f"{index_name}_meta.json"

    def save(self, embeddings: np.ndarray, chunk_ids: list[str]) -> dict:
        ensure_dir(self.index_dir)
        backend = "numpy"
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, str(self.index_path))
            backend = "faiss"
        except Exception as exc:  # pragma: no cover
            print(f"[vector_store] Fallback to numpy persistence because faiss failed: {exc}")
            np.save(self.npy_path, embeddings.astype(np.float32))

        self.meta_path.write_text(json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"backend": backend, "num_vectors": int(embeddings.shape[0])}

    def load(self) -> tuple[str, object, list[str]]:
        chunk_ids = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.index_path.exists():
            import faiss  # type: ignore

            return "faiss", faiss.read_index(str(self.index_path)), chunk_ids
        embeddings = np.load(self.npy_path)
        return "numpy", embeddings, chunk_ids

    def write_meta(self, meta: dict) -> None:
        self.info_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_meta(self) -> dict:
        if not self.info_path.exists():
            return {}
        try:
            return json.loads(self.info_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
