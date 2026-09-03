from __future__ import annotations

from .taxonomy import ENTITY_LEXICON, FILE_TASK_HINTS, METHOD_LEXICON, PATH_GROUP_HINTS


def build_rule_labels(
    chunk_text: str,
    source_path: str,
    file_name: str,
    *,
    source_group_hint: str | None = None,
    task_group_hint: str | None = None,
) -> dict:
    """
    Build hierarchical labels through rule anchors.

    Rule anchors come from:
    1. path rules
    2. file-name rules
    3. body keywords
    4. petroleum term lexicons
    """

    source_group = source_group_hint or detect_source_group(source_path)
    task_group = task_group_hint or detect_task_group(chunk_text, file_name)
    entity_tags = collect_tags(chunk_text, ENTITY_LEXICON)
    method_tags = collect_tags(chunk_text, METHOD_LEXICON)
    evidence_type = detect_evidence_type(chunk_text, source_group)
    confidence = estimate_confidence(source_group, task_group, entity_tags, method_tags, evidence_type)
    return {
        "source_group": source_group,
        "task_group": task_group,
        "entity_tags": entity_tags,
        "method_tags": method_tags,
        "evidence_type": evidence_type,
        "confidence": confidence,
    }


def detect_source_group(source_path: str) -> str:
    lower_path = source_path.lower()
    for hint, label in PATH_GROUP_HINTS.items():
        if hint in lower_path:
            return label
    return "text"


def detect_task_group(chunk_text: str, file_name: str) -> str:
    searchable = f"{file_name.lower()} {chunk_text.lower()}"
    for keyword, task_group in FILE_TASK_HINTS.items():
        if keyword in searchable:
            return task_group
    if any(word in searchable for word in ["text", "text", "text", "text"]):
        return "text"
    if any(word in searchable for word in ["text", "dbe", "text"]):
        return "text"
    if any(word in searchable for word in ["api", "text", "text"]):
        return "text"
    if any(word in searchable for word in ["text", "text", "text"]):
        return "text"
    return "text"


def collect_tags(chunk_text: str, lexicon: dict[str, list[str]]) -> list[str]:
    searchable = chunk_text.lower()
    tags: list[str] = []
    for canonical, keywords in lexicon.items():
        if any(keyword.lower() in searchable for keyword in keywords):
            tags.append(canonical)
    return sorted(tags)


def detect_evidence_type(chunk_text: str, source_group: str) -> str:
    searchable = chunk_text.lower()
    if source_group == "text":
        return "text"
    if any(token in searchable for token in ["text", "text", "text"]):
        return "text"
    if any(token in searchable for token in ["text", "text", "text", "observed"]):
        return "text"
    if any(token in searchable for token in ["text", "text", "text", "text"]):
        return "text"
    if any(token in searchable for token in ["text", "text", "text"]):
        return "text"
    return "text"


def estimate_confidence(
    source_group: str,
    task_group: str,
    entity_tags: list[str],
    method_tags: list[str],
    evidence_type: str,
) -> float:
    score = 0.20
    if source_group != "text":
        score += 0.20
    if task_group != "text":
        score += 0.20
    score += min(0.20, len(entity_tags) * 0.05)
    score += min(0.10, len(method_tags) * 0.05)
    if evidence_type != "text":
        score += 0.10
    return round(min(score, 0.95), 2)
