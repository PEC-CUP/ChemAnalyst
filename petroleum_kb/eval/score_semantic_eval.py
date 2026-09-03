from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _bucket_for(row: dict[str, Any]) -> str:
    qtype = str(row.get("question_type") or "").strip().lower()
    if qtype == "two_doc":
        return "two_doc-only"
    if qtype == "single":
        return "single-only"
    return "overall"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--correctness-threshold", type=float, default=0.7)
    ap.add_argument("--relevance-threshold", type=float, default=0.7)
    ap.add_argument("--faithfulness-threshold", type=float, default=0.7)
    ap.add_argument("--keypoint-threshold", type=float, default=0.65)
    args = ap.parse_args()

    out = args.out or args.results.with_suffix(".semantic.summary.csv")
    buckets = ("overall", "single-only", "two_doc-only")
    modes: dict[str, dict[str, dict[str, list[float]]]] = {}

    rows = list(iter_jsonl(args.results))
    for row in rows:
        mode = str(row.get("rag_mode") or "").strip().lower()
        if not mode:
            continue
        modes.setdefault(
            mode,
            {
                name: {
                    "answer_correctness_pass_rate": [],
                    "answer_relevancy_pass_rate": [],
                    "answer_correctness_and_relevancy_pass_rate": [],
                    "faithfulness_score": [],
                    "faithfulness_pass_rate": [],
                    "key_point_coverage_score": [],
                    "key_point_pass_rate": [],
                    "context_precision": [],
                    "context_recall": [],
                    "context_relevance": [],
                    "hit_at_k": [],
                    "recall_at_k": [],
                    "map": [],
                    "two_doc_coverage": [],
                    "two_doc_coverage_strict": [],
                }
                for name in buckets
            },
        )
        target_buckets = ["overall", _bucket_for(row)]

        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        context_judge = row.get("context_judge") if isinstance(row.get("context_judge"), dict) else {}
        rm = row.get("retrieval_metrics") if isinstance(row.get("retrieval_metrics"), dict) else {}

        correctness_score = float(judge.get("correctness_score", 0.0)) if judge else 0.0
        relevance_score = float(judge.get("relevance_score", 0.0)) if judge else 0.0
        faithfulness_score = float(judge.get("faithfulness_score", 0.0)) if judge else 0.0
        key_point_coverage_score = float(judge.get("key_point_coverage_score", 0.0)) if judge else 0.0

        for bucket in target_buckets:
            stats = modes[mode][bucket]
            stats["answer_correctness_pass_rate"].append(1.0 if correctness_score >= float(args.correctness_threshold) else 0.0)
            stats["answer_relevancy_pass_rate"].append(1.0 if relevance_score >= float(args.relevance_threshold) else 0.0)
            stats["answer_correctness_and_relevancy_pass_rate"].append(
                1.0 if (correctness_score >= float(args.correctness_threshold) and relevance_score >= float(args.relevance_threshold)) else 0.0
            )
            stats["faithfulness_score"].append(faithfulness_score)
            stats["faithfulness_pass_rate"].append(1.0 if faithfulness_score >= float(args.faithfulness_threshold) else 0.0)
            stats["key_point_coverage_score"].append(key_point_coverage_score)
            stats["key_point_pass_rate"].append(1.0 if key_point_coverage_score >= float(args.keypoint_threshold) else 0.0)

            if context_judge:
                stats["context_precision"].append(float(context_judge.get("context_precision", 0.0)))
                stats["context_recall"].append(float(context_judge.get("context_recall", 0.0)))
                stats["context_relevance"].append(float(context_judge.get("context_relevance", 0.0)))
            if rm:
                stats["hit_at_k"].append(float(rm.get("hit_at_k", 0.0)))
                stats["recall_at_k"].append(float(rm.get("recall_at_k", 0.0)))
                stats["map"].append(float(rm.get("map", 0.0)))
            if row.get("two_doc_coverage") is not None:
                stats["two_doc_coverage"].append(float(row.get("two_doc_coverage", 0.0)))
            if row.get("two_doc_coverage_strict") is not None:
                stats["two_doc_coverage_strict"].append(float(row.get("two_doc_coverage_strict", 0.0)))

    fieldnames = [
        "split",
        "mode",
        "n",
        "answer_correctness_pass_rate",
        "answer_relevancy_pass_rate",
        "answer_correctness_and_relevancy_pass_rate",
        "faithfulness_score",
        "faithfulness_pass_rate",
        "key_point_coverage_score",
        "key_point_pass_rate",
        "context_precision",
        "context_recall",
        "context_relevance",
        "hit_at_k",
        "recall_at_k",
        "map",
        "two_doc_coverage",
        "two_doc_coverage_strict",
        "correctness_threshold",
        "relevance_threshold",
        "faithfulness_threshold",
        "keypoint_threshold",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for split in buckets:
            for mode in sorted(modes):
                stats = modes[mode][split]
                n = len(stats["answer_correctness_pass_rate"])
                if n == 0:
                    continue
                writer.writerow(
                    {
                        "split": split,
                        "mode": mode,
                        "n": n,
                        "answer_correctness_pass_rate": f"{_mean(stats['answer_correctness_pass_rate']):.4f}",
                        "answer_relevancy_pass_rate": f"{_mean(stats['answer_relevancy_pass_rate']):.4f}",
                        "answer_correctness_and_relevancy_pass_rate": f"{_mean(stats['answer_correctness_and_relevancy_pass_rate']):.4f}",
                        "faithfulness_score": f"{_mean(stats['faithfulness_score']):.4f}",
                        "faithfulness_pass_rate": f"{_mean(stats['faithfulness_pass_rate']):.4f}",
                        "key_point_coverage_score": f"{_mean(stats['key_point_coverage_score']):.4f}",
                        "key_point_pass_rate": f"{_mean(stats['key_point_pass_rate']):.4f}",
                        "context_precision": f"{_mean(stats['context_precision']):.4f}",
                        "context_recall": f"{_mean(stats['context_recall']):.4f}",
                        "context_relevance": f"{_mean(stats['context_relevance']):.4f}",
                        "hit_at_k": f"{_mean(stats['hit_at_k']):.4f}",
                        "recall_at_k": f"{_mean(stats['recall_at_k']):.4f}",
                        "map": f"{_mean(stats['map']):.4f}",
                        "two_doc_coverage": f"{_mean(stats['two_doc_coverage']):.4f}",
                        "two_doc_coverage_strict": f"{_mean(stats['two_doc_coverage_strict']):.4f}",
                        "correctness_threshold": float(args.correctness_threshold),
                        "relevance_threshold": float(args.relevance_threshold),
                        "faithfulness_threshold": float(args.faithfulness_threshold),
                        "keypoint_threshold": float(args.keypoint_threshold),
                    }
                )
    print(f"Wrote semantic benchmark summary to {out}")


if __name__ == "__main__":
    main()
