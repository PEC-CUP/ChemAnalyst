from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_eval import _deepseek_judge, apply_evidence_gate, load_dotenv_simple


def load_results(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("results", "rows", "data"):
                rows = obj.get(key)
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]
            return [obj]
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def normalize_equivalent_groups(groups: Any) -> list[list[str]]:
    out: list[list[str]] = []
    if not isinstance(groups, list):
        return out
    for group in groups:
        if isinstance(group, dict):
            doc_ids = group.get("doc_ids")
            if isinstance(doc_ids, list):
                vals = [str(x).strip() for x in doc_ids if str(x).strip()]
                if vals:
                    out.append(vals)
            continue
        if isinstance(group, list):
            vals = [str(x).strip() for x in group if str(x).strip()]
            if vals:
                out.append(vals)
    return out


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("id") or ""), str(row.get("rag_mode") or row.get("mode") or ""))


def question_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    done: set[tuple[str, str, str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        done.add(
            (
                str(row.get("answer_source") or ""),
                str(row.get("judge_profile") or ""),
                str(row.get("id") or ""),
                str(row.get("rag_mode") or ""),
            )
        )
    return done


def summarize(records: list[dict[str, Any]], out_csv: Path) -> None:
    buckets: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    errors: dict[tuple[str, str, str], int] = defaultdict(int)
    for rec in records:
        key = (str(rec["answer_source"]), str(rec["judge_profile"]), str(rec["rag_mode"]))
        counts[key] += 1
        judge = rec.get("judge") if isinstance(rec.get("judge"), dict) else {}
        if rec.get("error") or "error" in judge:
            errors[key] += 1
            continue
        raw = judge.get("raw_judge_pass") if isinstance(judge.get("raw_judge_pass"), dict) else {}
        gated = judge.get("gated_judge_pass") if isinstance(judge.get("gated_judge_pass"), dict) else {}
        buckets[key]["answer_correct_and_relevant_pass"].append(1.0 if raw.get("correct_and_relevant") is True else 0.0)
        buckets[key]["faithful_pass"].append(1.0 if raw.get("faithful") is True else 0.0)
        buckets[key]["gated_correct_and_relevant_pass"].append(1.0 if gated.get("correct_and_relevant") is True else 0.0)
        buckets[key]["gated_faithful_pass"].append(1.0 if gated.get("faithful") is True else 0.0)
        for metric in ("correctness_score", "relevance_score", "faithfulness_score", "key_point_coverage_score"):
            try:
                buckets[key][metric].append(float(judge.get(metric, 0.0)))
            except Exception:
                pass
        gate = judge.get("evidence_gate") if isinstance(judge.get("evidence_gate"), dict) else {}
        for metric in ("strict_coverage_score", "soft_coverage_score"):
            try:
                buckets[key][metric].append(float(gate.get(metric, 0.0)))
            except Exception:
                pass
        if isinstance(rec.get("judge_latency_s"), (int, float)):
            buckets[key]["judge_latency_s"].append(float(rec["judge_latency_s"]))

    fieldnames = [
        "answer_source",
        "judge_profile",
        "rag_mode",
        "n",
        "errors",
        "answer_correct_and_relevant_pass",
        "faithful_pass",
        "gated_correct_and_relevant_pass",
        "gated_faithful_pass",
        "correctness_score",
        "relevance_score",
        "faithfulness_score",
        "key_point_coverage_score",
        "strict_coverage_score",
        "soft_coverage_score",
        "judge_latency_s",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(counts):
            row: dict[str, Any] = {
                "answer_source": key[0],
                "judge_profile": key[1],
                "rag_mode": key[2],
                "n": counts[key],
                "errors": errors[key],
            }
            for metric in fieldnames[5:]:
                vals = buckets[key].get(metric, [])
                row[metric] = f"{(sum(vals) / len(vals)):.4f}" if vals else ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-judge existing RAG result files without rerunning retrieval.")
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--new-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--modes", default="naive,qa_oriented,iterative_review")
    parser.add_argument("--judge-profiles", default="baseline:")
    parser.add_argument(
        "--answer-sources",
        default="old_answer,new_answer",
        help="Comma-separated answer sources to judge: old_answer,new_answer.",
    )
    parser.add_argument("--matched-only", action="store_true")
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_dotenv_simple(args.env_file)

    modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    profiles: list[tuple[str, str]] = []
    for item in args.judge_profiles.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise SystemExit(f"Invalid judge profile: {item}")
        label, model = item.split(":", 1)
        profiles.append((label.strip(), model.strip()))

    old_rows = [r for r in load_results(args.old_results) if str(r.get("rag_mode") or r.get("mode") or "") in modes and r.get("ok", True)]
    new_rows = [r for r in load_results(args.new_results) if str(r.get("rag_mode") or r.get("mode") or "") in modes and r.get("ok", True)]
    if args.matched_only:
        new_ids = {question_id(r) for r in new_rows if question_id(r)}
        old_rows = [r for r in old_rows if question_id(r) in new_ids]
    if args.limit_questions > 0:
        allowed_ids: list[str] = []
        seen: set[str] = set()
        for r in new_rows:
            qid = question_id(r)
            if qid and qid not in seen:
                seen.add(qid)
                allowed_ids.append(qid)
            if len(allowed_ids) >= args.limit_questions:
                break
        allowed = set(allowed_ids)
        old_rows = [r for r in old_rows if question_id(r) in allowed]
        new_rows = [r for r in new_rows if question_id(r) in allowed]

    answer_sources = {x.strip() for x in args.answer_sources.split(",") if x.strip()}
    valid_answer_sources = {"old_answer", "new_answer"}
    bad_answer_sources = sorted(answer_sources - valid_answer_sources)
    if bad_answer_sources:
        raise SystemExit(f"Invalid answer sources: {bad_answer_sources}. Allowed: {sorted(valid_answer_sources)}")

    jobs: list[tuple[str, dict[str, Any], str, str]] = []
    for source, rows in (("old_answer", old_rows), ("new_answer", new_rows)):
        if source not in answer_sources:
            continue
        for row in rows:
            rag_mode = str(row.get("rag_mode") or row.get("mode") or "")
            for profile, model in profiles:
                jobs.append((source, row, profile, model))

    done = load_done_keys(args.out)
    total = len(jobs)
    completed = len(done)
    print(f"[rejudge] jobs={total} already_done={completed} out={args.out}")
    started = time.perf_counter()
    for idx, (source, row, profile, model) in enumerate(jobs, 1):
        rag_mode = str(row.get("rag_mode") or row.get("mode") or "")
        key = (source, profile, question_id(row), rag_mode)
        if key in done:
            continue
        t0 = time.perf_counter()
        rec: dict[str, Any] = {
            "answer_source": source,
            "judge_profile": profile,
            "judge_model": model,
            "id": row.get("id"),
            "line_no": row.get("line_no"),
            "rag_mode": rag_mode,
            "question_type": row.get("question_type"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "answer_key": row.get("answer_key"),
            "key_points": row.get("key_points") or [],
            "gold_doc_ids": row.get("gold_doc_ids") or [],
            "pred_doc_ids": row.get("pred_doc_ids") or [],
        }
        try:
            judge = _deepseek_judge(
                question=str(row.get("question") or ""),
                answer=str(row.get("answer") or ""),
                answer_key=str(row.get("answer_key") or ""),
                key_points=row.get("key_points") or [],
                question_type=str(row.get("question_type") or "single"),
                model_override=model,
            )
            if isinstance(judge, dict) and "error" not in judge:
                judge = apply_evidence_gate(
                    judge_obj=judge,
                    question_type=str(row.get("question_type") or "single"),
                    gold_doc_ids=[str(x) for x in (row.get("gold_doc_ids") or [])],
                    pred_doc_ids=[str(x) for x in (row.get("pred_doc_ids") or [])],
                    equivalent_doc_groups=normalize_equivalent_groups(row.get("equivalent_doc_groups")),
                )
            rec["judge"] = judge
        except Exception as exc:
            rec["error"] = str(exc)
        rec["judge_latency_s"] = time.perf_counter() - t0
        append_jsonl(args.out, rec)
        done.add(key)
        if args.progress_every and idx % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"[rejudge] progress={idx}/{total} written={len(done)} elapsed_s={elapsed:.1f}")

    records = load_results(args.out)
    summarize(records, args.summary)
    print(f"[rejudge] wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
