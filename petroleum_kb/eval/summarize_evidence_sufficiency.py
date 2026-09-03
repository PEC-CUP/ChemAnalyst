import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def as_text(value: Any) -> str:
    return str(value or "").strip()


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def iter_jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            obj["_line_no"] = line_no
            yield obj


def recover_score_and_model_pass(evidence_sufficiency: Any) -> tuple[float | None, bool | None, bool]:
    if not isinstance(evidence_sufficiency, dict):
        return None, None, False
    score = evidence_sufficiency.get("evidence_sufficiency_score")
    model_pass = evidence_sufficiency.get("model_sufficient")
    if not isinstance(model_pass, bool):
        model_pass = evidence_sufficiency.get("sufficient")
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score))), model_pass if isinstance(model_pass, bool) else None, False

    raw = evidence_sufficiency.get("raw")
    if isinstance(raw, str):
        score_match = re.search(r'"evidence_sufficiency_score"\s*:\s*([0-9.]+)', raw)
        pass_match = re.search(r'"sufficient"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
        if score_match:
            score = max(0.0, min(1.0, float(score_match.group(1))))
            recovered_pass = None
            if pass_match:
                recovered_pass = pass_match.group(1).lower() == "true"
            return score, recovered_pass, True

    return None, model_pass if isinstance(model_pass, bool) else None, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing evidence-sufficiency JSONL with a chosen pass threshold.")
    parser.add_argument("--input", type=Path, required=True, help="Evidence-sufficiency JSONL produced by run_evidence_sufficiency_eval.py.")
    parser.add_argument("--summary", type=Path, required=True, help="Output CSV summary.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Pass threshold for evidence_sufficiency_score. Default: 0.6.")
    args = parser.parse_args()

    summary_values: defaultdict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    recovered = 0
    missing_score = 0
    rows = 0

    for row in iter_jsonl(args.input):
        rows += 1
        qtype = as_text(row.get("question_type") or "unknown")
        mode = as_text(row.get("rag_mode") or row.get("mode") or "unknown")
        source = as_text(row.get("context_source") or "unknown")
        score, model_pass, was_recovered = recover_score_and_model_pass(row.get("evidence_sufficiency"))
        if was_recovered:
            recovered += 1
        if score is None:
            missing_score += 1
            continue
        key = (qtype, mode, source)
        pass_value = 1.0 if score >= float(args.threshold) else 0.0
        summary_values[key]["evidence_sufficiency_score"].append(score)
        summary_values[key]["evidence_sufficient_pass"].append(pass_value)
        if isinstance(model_pass, bool):
            summary_values[key]["model_sufficient_pass"].append(1.0 if model_pass else 0.0)
        context_doc_count = row.get("context_doc_count")
        if isinstance(context_doc_count, (int, float)):
            summary_values[key]["context_doc_count"].append(float(context_doc_count))
        latency = row.get("judge_latency_s")
        if isinstance(latency, (int, float)):
            summary_values[key]["judge_latency_s"].append(float(latency))

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_type",
        "rag_mode",
        "context_source",
        "threshold",
        "n",
        "evidence_sufficiency_score",
        "evidence_sufficient_pass",
        "model_sufficient_pass",
        "context_doc_count",
        "judge_latency_s",
    ]
    with args.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(summary_values):
            qtype, mode, source = key
            vals = summary_values[key]
            writer.writerow(
                {
                    "question_type": qtype,
                    "rag_mode": mode,
                    "context_source": source,
                    "threshold": f"{float(args.threshold):.3f}",
                    "n": len(vals["evidence_sufficiency_score"]),
                    "evidence_sufficiency_score": fmt(mean(vals["evidence_sufficiency_score"])),
                    "evidence_sufficient_pass": fmt(mean(vals["evidence_sufficient_pass"])),
                    "model_sufficient_pass": fmt(mean(vals["model_sufficient_pass"])),
                    "context_doc_count": fmt(mean(vals["context_doc_count"])),
                    "judge_latency_s": fmt(mean(vals["judge_latency_s"])),
                }
            )

    print(f"Wrote summary to {args.summary}")
    print(f"Rows: {rows}; recovered invalid_json scores: {recovered}; missing scores: {missing_score}")


if __name__ == "__main__":
    main()
