"""Export representative benchmark answers from evaluation outputs.

Expected inputs are one custom dataset JSONL, one closed-book result JSONL, and
one RAG result JSONL produced from the same custom dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MODE_LABELS = {
    "closed_book": "Closed-book",
    "naive": "Naive RAG",
    "naive_llm_filter": "Naive + LLM filter",
    "qa_oriented": "QA-oriented RAG",
    "iterative_review": "Iterative review RAG",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _item_id(row: dict[str, Any]) -> str:
    for key in ("id", "qid", "item_id", "question_id"):
        value = row.get(key)
        if value:
            return str(value)
    item = row.get("item")
    if isinstance(item, dict):
        for key in ("id", "qid", "item_id", "question_id"):
            value = item.get(key)
            if value:
                return str(value)
    return ""


def _mode(row: dict[str, Any], fallback: str = "") -> str:
    for key in ("rag_mode", "mode"):
        value = row.get(key)
        if value:
            return str(value)
    request = row.get("request")
    if isinstance(request, dict) and request.get("rag_mode"):
        return str(request["rag_mode"])
    return fallback


def _answer(row: dict[str, Any]) -> str:
    for key in ("answer", "response", "output", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    result = row.get("result")
    if isinstance(result, dict):
        for key in ("answer", "response", "output", "final_answer"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raw = row.get("raw")
    if isinstance(raw, dict):
        for key in ("answer", "response", "output", "final_answer"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    error = row.get("error") or row.get("error_message")
    if error:
        return f"[ERROR] {error}"
    return "[No answer captured]"


def _score_summary(row: dict[str, Any]) -> str:
    keys = [
        "correct_and_relevant_pass",
        "faithful_pass",
        "correctness_score",
        "faithfulness_score",
        "key_point_coverage_score",
        "evidence_sufficiency_score",
        "direct_correct",
        "accuracy",
    ]
    parts = []
    for key in keys:
        value = row.get(key)
        if value is None and isinstance(row.get("judge"), dict):
            value = row["judge"].get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def export_answers(dataset: Path, closed_results: Path, rag_results: Path, out_md: Path) -> None:
    items = _read_jsonl(dataset)
    closed_rows = _read_jsonl(closed_results)
    rag_rows = _read_jsonl(rag_results)

    by_item_mode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in closed_rows:
        item_id = _item_id(row)
        if item_id:
            by_item_mode[item_id]["closed_book"] = row
    for row in rag_rows:
        item_id = _item_id(row)
        mode = _mode(row)
        if item_id and mode:
            by_item_mode[item_id][mode] = row

    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Representative QA Example Answers",
        "",
        "This file summarizes closed-book, RAG, and ablation outputs for the six representative benchmark questions.",
        "",
    ]
    mode_order = [
        "closed_book",
        "naive",
        "naive_llm_filter",
        "qa_oriented",
        "iterative_review",
    ]

    for idx, item in enumerate(items, start=1):
        item_id = str(item.get("id", ""))
        label = item.get("example_label", "")
        question = item.get("question", "")
        answer_key = item.get("answer_key") or item.get("answer") or ""
        lines.extend(
            [
                f"## {idx}. {label}",
                "",
                f"**Question:** {question}",
                "",
                f"**Reference answer:** {answer_key}",
                "",
            ]
        )
        for mode in mode_order:
            row = by_item_mode.get(item_id, {}).get(mode)
            label_text = MODE_LABELS.get(mode, mode)
            lines.append(f"### {label_text}")
            if row is None:
                lines.append("")
                lines.append("[Missing result]")
                lines.append("")
                continue
            score_text = _score_summary(row)
            if score_text:
                lines.append("")
                lines.append(f"**Scores:** {score_text}")
            lines.append("")
            lines.append(_answer(row))
            lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Wrote representative answers to {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--closed-results", type=Path, required=True)
    parser.add_argument("--rag-results", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    export_answers(args.dataset, args.closed_results, args.rag_results, args.out_md)


if __name__ == "__main__":
    main()
