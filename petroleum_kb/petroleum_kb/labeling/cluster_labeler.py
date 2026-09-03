from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np


EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}

ZH_STOPWORDS = {
    "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text",
    "text", "text", "text", "text", "text", "text", "text", "text", "text", "text",
}

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{1,}")


def suggest_cluster_labels(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    similarity_threshold: float = 0.58,
    min_cluster_size: int = 2,
    max_clusters: int = 50,
) -> list[dict]:
    """
    Build lightweight semantic clusters and candidate topic labels.

    This implementation intentionally avoids heavy external dependencies such as
    BERTopic/HDBSCAN so the ingest pipeline remains runnable in restricted
    environments. It uses normalized embedding similarity with greedy centroid
    assignment, then derives human-readable candidate labels from metadata tags
    and frequent keywords.
    """

    if not chunks:
        return []

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(chunks):
        return []

    normalized = _normalize_rows(matrix)
    clusters: list[dict[str, Any]] = []
    for index, vector in enumerate(normalized):
        best_idx = -1
        best_score = -1.0
        for cluster_idx, cluster in enumerate(clusters):
            score = float(np.dot(vector, cluster["centroid"]))
            if score > best_score:
                best_score = score
                best_idx = cluster_idx
        if best_idx >= 0 and best_score >= similarity_threshold:
            _append_to_cluster(clusters[best_idx], index, vector)
        elif len(clusters) < max_clusters:
            clusters.append(
                {
                    "member_indices": [index],
                    "vector_sum": vector.copy(),
                    "centroid": vector.copy(),
                }
            )
        else:
            fallback_idx = min(range(len(clusters)), key=lambda idx: len(clusters[idx]["member_indices"]))
            _append_to_cluster(clusters[fallback_idx], index, vector)

    summaries: list[dict[str, Any]] = []
    for cluster_idx, cluster in enumerate(clusters, start=1):
        if len(cluster["member_indices"]) < min_cluster_size:
            continue
        member_chunks = [chunks[item_idx] for item_idx in cluster["member_indices"]]
        summaries.append(_summarize_cluster(cluster_idx, member_chunks, cluster["centroid"]))

    summaries.sort(key=lambda item: item["size"], reverse=True)
    return summaries


def _append_to_cluster(cluster: dict[str, Any], index: int, vector: np.ndarray) -> None:
    cluster["member_indices"].append(index)
    cluster["vector_sum"] = cluster["vector_sum"] + vector
    cluster["centroid"] = _normalize_vector(cluster["vector_sum"])


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe_norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if math.isclose(norm, 0.0):
        return vector
    return vector / norm


def _summarize_cluster(cluster_idx: int, member_chunks: list[dict[str, Any]], centroid: np.ndarray) -> dict[str, Any]:
    texts = [str(item.get("text", "")) for item in member_chunks]
    chunk_ids = [str(item.get("chunk_id", "")) for item in member_chunks]
    metadatas = [item.get("metadata", {}) for item in member_chunks]

    entity_counter = Counter(tag for meta in metadatas for tag in meta.get("entity_tags", []))
    method_counter = Counter(tag for meta in metadatas for tag in meta.get("method_tags", []))
    task_counter = Counter(meta.get("task_group") for meta in metadatas if meta.get("task_group"))
    source_counter = Counter(meta.get("source_group") for meta in metadatas if meta.get("source_group"))
    keyword_counter = _collect_keywords(texts)

    top_entities = [item for item, _ in entity_counter.most_common(3)]
    top_methods = [item for item, _ in method_counter.most_common(3)]
    top_keywords = [item for item, _ in keyword_counter.most_common(6)]

    candidate_labels: list[str] = []
    if top_entities:
        candidate_labels.extend(top_entities[:2])
    if top_methods:
        candidate_labels.extend(top_methods[:1])
    candidate_labels.extend(top_keywords[:3])

    deduped_labels: list[str] = []
    for label in candidate_labels:
        if label and label not in deduped_labels:
            deduped_labels.append(label)

    summary_label = " / ".join(deduped_labels[:3]) if deduped_labels else f"text {cluster_idx}"
    sample_members = []
    for item in member_chunks[:3]:
        meta = item.get("metadata", {})
        sample_members.append(
            {
                "chunk_id": item.get("chunk_id"),
                "file_name": meta.get("file_name"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "preview": str(item.get("text", "")).replace("\n", " ")[:180],
            }
        )

    return {
        "cluster_id": f"cluster_{cluster_idx:03d}",
        "candidate_label": summary_label,
        "size": len(member_chunks),
        "top_keywords": top_keywords,
        "entity_tags": top_entities,
        "method_tags": top_methods,
        "task_groups": [item for item, _ in task_counter.most_common(3)],
        "source_groups": [item for item, _ in source_counter.most_common(3)],
        "chunk_ids": chunk_ids,
        "sample_members": sample_members,
        "centroid_norm": float(np.linalg.norm(centroid)),
    }


def _collect_keywords(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for match in TOKEN_PATTERN.findall(text):
            token = match.strip()
            lowered = token.lower()
            if len(token) < 2:
                continue
            if lowered in EN_STOPWORDS or token in ZH_STOPWORDS:
                continue
            counter[token] += 1
    return counter
