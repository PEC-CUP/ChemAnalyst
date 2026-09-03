from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


def rerank_score(
    *,
    semantic_similarity: float,
    tag_match: float,
    source_weight: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """
    Final ranking score fusion.

    By default:
      final_score = alpha * semantic + beta * tag + gamma * source

    In stage-2 reranker mode, callers may pass a model-based rerank score as
    semantic_similarity to keep the rest of the fusion logic unchanged.
    """

    return alpha * semantic_similarity + beta * tag_match + gamma * source_weight


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalize_rerank_scores(raw_scores: list[float]) -> list[float]:
    """
    Cross-encoder models often output logits (unbounded). We map them to (0, 1)
    so fusion weights behave predictably.
    """

    return [_sigmoid(float(s)) for s in raw_scores]


@dataclass(slots=True)
class RerankResult:
    chunk_id: str
    vector_score: float
    rerank_score: float


class CrossEncoderReranker:
    """
    Stage-2 "true reranker" using sentence-transformers CrossEncoder.

    It scores (query, chunk_text) pairs and is typically much more precise than
    vector similarity for topN candidates.
    """

    def __init__(self, model_name: str, *, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(model_name, local_files_only=local_files_only)
        except Exception as exc:  # pragma: no cover
            # Try local snapshots from common HF cache locations (robust across
            # different shells/services that use different HF_HOME settings).
            if "/" in model_name:
                local_snapshot = self._find_local_snapshot(model_name)
                if local_snapshot is not None:
                    try:
                        self._model = CrossEncoder(str(local_snapshot), local_files_only=True)
                        self.local_files_only = True
                        self.model_name = str(local_snapshot)
                        return
                    except Exception:
                        pass
            if not local_files_only:
                try:
                    self._model = CrossEncoder(model_name, local_files_only=True)
                    self.local_files_only = True
                    return
                except Exception:
                    pass
            raise RuntimeError(
                f"CrossEncoder reranker init failed for {model_name!r}. "
                f"local_files_only={local_files_only}. "
                "Reranker unavailable because sentence-transformers is not installed or the model is not cached. "
                f"root_error={type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _find_local_snapshot(model_name: str) -> Path | None:
        repo_fs = f"models--{model_name.replace('/', '--')}"
        candidates: list[Path] = []

        hf_home = os.getenv("HF_HOME")
        hf_hub_cache = os.getenv("HF_HUB_CACHE")
        st_home = os.getenv("SENTENCE_TRANSFORMERS_HOME")
        user_hf_hub = Path.home() / ".cache" / "huggingface" / "hub"

        if hf_hub_cache:
            candidates.append(Path(hf_hub_cache))
        if hf_home:
            candidates.append(Path(hf_home) / "hub")
        candidates.append(user_hf_hub)
        if st_home:
            # sentence-transformers specific cache can also contain model dirs.
            candidates.append(Path(st_home))

        for base in candidates:
            try:
                repo_dir = base / repo_fs
                snapshots = repo_dir / "snapshots"
                if not snapshots.exists():
                    continue
                all_snaps = [p for p in snapshots.iterdir() if p.is_dir()]
                if not all_snaps:
                    continue
                all_snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return all_snaps[0]
            except Exception:
                continue
        return None

    def score(self, query: str, texts: list[str]) -> list[float]:
        # sentence-transformers CrossEncoder expects list[(query, doc)]
        pairs = [(query, t) for t in texts]
        raw = self._model.predict(pairs)
        # predict() may return numpy array
        return normalize_rerank_scores([float(x) for x in raw])
