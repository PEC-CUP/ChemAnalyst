from __future__ import annotations

import hashlib

import numpy as np


class TextEmbedder:
    """
    Text embedding wrapper with graceful degradation.

    Preferred backend:
    - sentence-transformers

    Fallback:
    - deterministic hash-based vectors

    The fallback keeps the pipeline runnable when transformer dependencies are
    unavailable, and can later be replaced by stronger petroleum-specific models.
    """

    def __init__(
        self,
        model_name: str,
        fallback_dim: int = 384,
        *,
        allow_hash_fallback: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.allow_hash_fallback = allow_hash_fallback
        self.local_files_only = local_files_only
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name, local_files_only=local_files_only)
            dimension_getter = getattr(self._model, "get_embedding_dimension", None)
            if callable(dimension_getter):
                self.dimension = int(dimension_getter())
            else:  # backward compatibility with older sentence-transformers
                self.dimension = int(self._model.get_sentence_embedding_dimension())
            self.backend = "sentence-transformers"
        except Exception as exc:  # pragma: no cover
            if not allow_hash_fallback:
                raise RuntimeError(
                    f"Transformer embedding backend failed for {model_name!r}; "
                    "hash fallback is disabled for this build."
                ) from exc
            print(f"[embedder] Fallback to hash embeddings because transformer backend failed: {exc}")
            self.dimension = fallback_dim
            self.backend = "hash"

    def embed_texts(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if self._model is not None:
            encode_kwargs = {"normalize_embeddings": True}
            if show_progress:
                encode_kwargs["show_progress_bar"] = True
            if batch_size is not None:
                encode_kwargs["batch_size"] = int(batch_size)
            embeddings = self._model.encode(texts, **encode_kwargs)
            return np.asarray(embeddings, dtype=np.float32)

        # Hash fallback: add a lightweight progress indicator when requested.
        if not show_progress:
            return np.vstack([self._hash_embed(text) for text in texts]).astype(np.float32)

        total = len(texts)
        if total == 0:
            return np.zeros((0, int(self.dimension)), dtype=np.float32)
        step = max(1, min(200, total // 20))
        rows: list[np.ndarray] = []
        for idx, text in enumerate(texts, start=1):
            if idx == 1 or idx % step == 0 or idx == total:
                print(f"[embedder] hashing embeddings: {idx}/{total}")
            rows.append(self._hash_embed(text))
        return np.vstack(rows).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
