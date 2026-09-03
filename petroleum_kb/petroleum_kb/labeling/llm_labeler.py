from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from .taxonomy import EVIDENCE_TYPES, ENTITY_LEXICON, METHOD_LEXICON, SOURCE_GROUPS, TASK_GROUPS

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


LLM_LABELER_ENABLED_ENV = "PETROLEUM_KB_ENABLE_LLM_LABELER"
LLM_LABELER_MAX_TEXT_CHARS_ENV = "PETROLEUM_KB_LLM_LABELER_MAX_TEXT_CHARS"
LLM_LABELER_DEBUG_ENV = "PETROLEUM_KB_LLM_LABELER_DEBUG"
LLM_LABELER_DEBUG_FILE_ENV = "PETROLEUM_KB_LLM_LABELER_DEBUG_FILE"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
LLM_TIMEOUT_SECONDS_ENV = "LLM_TIMEOUT_SECONDS"


if load_dotenv is not None:
    MODULE_PATH = Path(__file__).resolve()
    REPO_ROOT = MODULE_PATH.parents[3]
    load_dotenv(REPO_ROOT / ".env", override=False)


def llm_label_chunk(
    chunk_text: str,
    source_path: str,
    file_name: str,
    *,
    rule_labels: dict | None = None,
) -> dict:
    """
    Optional LLM-assisted constrained labeling.

    The labeler is disabled by default. When enabled, it can only choose from
    the existing closed taxonomy and is expected to complement the rule-based
    labels rather than freely inventing new labels.
    """

    if not _llm_labeler_enabled():
        _debug_log({"status": "disabled"})
        return {}

    client = _get_llm_client()
    if client is None:
        _debug_log({"status": "no_client"})
        return {}

    prompt = _build_prompt(
        chunk_text=chunk_text,
        source_path=source_path,
        file_name=file_name,
        rule_labels=rule_labels or {},
    )
    try:
        response = _chat(client, prompt)
        parsed = _parse_response(response)
        _debug_log(
            {
                "status": "ok",
                "file_name": file_name,
                "source_path": source_path,
                "rule_labels": rule_labels or {},
                "raw_response": response,
                "parsed_labels": parsed,
                "chunk_preview": chunk_text[:300],
            }
        )
        return parsed
    except Exception as exc:
        _debug_log(
            {
                "status": "error",
                "file_name": file_name,
                "source_path": source_path,
                "rule_labels": rule_labels or {},
                "error": repr(exc),
                "chunk_preview": chunk_text[:300],
            }
        )
        return {}


def _llm_labeler_enabled() -> bool:
    raw = os.getenv(LLM_LABELER_ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _get_llm_client() -> OpenAI | None:
    if not _llm_labeler_enabled():
        return None
    api_key = (os.getenv("LLM_API_KEY") or os.getenv(DEEPSEEK_API_KEY_ENV, "")).strip()
    if not api_key:
        return None
    try:
        return OpenAI(
            api_key=api_key,
            base_url=(os.getenv("LLM_BASE_URL") or os.getenv(DEEPSEEK_BASE_URL_ENV, "https://api.deepseek.com")).strip() or "https://api.deepseek.com",
            timeout=float(os.getenv(LLM_TIMEOUT_SECONDS_ENV, "60") or "60"),
        )
    except Exception:
        return None


def _chat(client: OpenAI, prompt: str) -> str:
    model = os.getenv("LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError("Missing LLM model. Set LLM_MODEL before enabling LLM labeling.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a petroleum-domain metadata labeler. "
                    "You must only return strict JSON and only choose values from provided options."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    content = response.choices[0].message.content if response.choices else None
    return (content or "").strip()


def _debug_log(payload: dict) -> None:
    raw = os.getenv(LLM_LABELER_DEBUG_ENV, "").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return
    path = Path(os.getenv(LLM_LABELER_DEBUG_FILE_ENV, "petroleum_kb/petroleum_kb/data/metadata/llm_labeler_debug.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_prompt(*, chunk_text: str, source_path: str, file_name: str, rule_labels: dict) -> str:
    max_chars = int(os.getenv(LLM_LABELER_MAX_TEXT_CHARS_ENV, "1800") or "1800")
    safe_text = chunk_text[:max_chars]
    schema = {
        "source_group": "must choose exactly one from source_group_options",
        "task_group": "must choose exactly one from task_group_options",
        "entity_tags": "must choose zero or more from entity_tag_options",
        "method_tags": "must choose zero or more from method_tag_options",
        "evidence_type": "must choose exactly one from evidence_type_options",
        "confidence": "float between 0 and 1",
    }
    return (
        "You are labeling a petroleum-domain text chunk.\n"
        "You must only return valid JSON and must only choose labels from the provided options.\n"
        "Do not invent any new labels. If uncertain, keep the rule-based label.\n\n"
        f"source_group_options: {json.dumps(SOURCE_GROUPS, ensure_ascii=False)}\n"
        f"task_group_options: {json.dumps(TASK_GROUPS, ensure_ascii=False)}\n"
        f"entity_tag_options: {json.dumps(sorted(ENTITY_LEXICON.keys()), ensure_ascii=False)}\n"
        f"method_tag_options: {json.dumps(sorted(METHOD_LEXICON.keys()), ensure_ascii=False)}\n"
        f"evidence_type_options: {json.dumps(EVIDENCE_TYPES, ensure_ascii=False)}\n"
        f"rule_labels: {json.dumps(rule_labels, ensure_ascii=False)}\n"
        f"source_path: {source_path}\n"
        f"file_name: {file_name}\n"
        f"chunk_text: {safe_text}\n\n"
        f"Return JSON with schema: {json.dumps(schema, ensure_ascii=False)}"
    )


def _parse_response(response: str) -> dict:
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return {}

    source_group = payload.get("source_group")
    task_group = payload.get("task_group")
    entity_tags = payload.get("entity_tags")
    method_tags = payload.get("method_tags")
    evidence_type = payload.get("evidence_type")
    confidence = payload.get("confidence")

    normalized = {}
    if source_group in SOURCE_GROUPS:
        normalized["source_group"] = source_group
    if task_group in TASK_GROUPS:
        normalized["task_group"] = task_group
    if isinstance(entity_tags, list):
        normalized["entity_tags"] = [item for item in entity_tags if item in ENTITY_LEXICON]
    if isinstance(method_tags, list):
        normalized["method_tags"] = [item for item in method_tags if item in METHOD_LEXICON]
    if evidence_type in EVIDENCE_TYPES:
        normalized["evidence_type"] = evidence_type
    try:
        score = float(confidence)
        if 0.0 <= score <= 1.0:
            normalized["confidence"] = round(score, 2)
    except (TypeError, ValueError):
        pass
    return normalized
