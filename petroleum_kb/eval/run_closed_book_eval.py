from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from run_eval import _deepseek_judge, infer_question_type, load_dotenv_simple


def iter_rows(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            return
        for line_no, obj in enumerate(data, start=1):
            if isinstance(obj, dict):
                obj["_line_no"] = line_no
                yield obj
        return

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            obj["_line_no"] = line_no
            yield obj


def as_text(value: Any) -> str:
    return str(value or "").strip()


def get_answer_model(args: argparse.Namespace) -> str:
    return (
        args.model
        or os.getenv("LLM_MODEL")
        
        
        or os.getenv("OPENAI_MODEL")
        or ""
    )


def call_closed_book(question: str, args: argparse.Namespace) -> tuple[str, str]:
    api_key = (args.api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (args.base_url or os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = get_answer_model(args)
    if not api_key:
        raise RuntimeError("Missing API key. Set LLM_API_KEY, or pass --api-key.")
    if not model:
        raise RuntimeError("Missing LLM model. Set LLM_MODEL, or pass --model.")

    prompt = (
        "Answer the question using your own knowledge only.\n"
        "Do not claim access to retrieved documents, local files, a knowledge base, or the benchmark answer.\n"
        "If the question asks for paper metadata such as DOI, author, title, or year, provide it only if you are confident; otherwise state uncertainty.\n"
        "Use concise scientific English.\n\n"
        f"Question:\n{question}\n"
    )
    body = {
        "model": model,
        "temperature": float(args.temperature),
        "messages": [
            {"role": "system", "content": "You are a closed-book scientific QA baseline."},
            {"role": "user", "content": prompt},
        ],
    }
    if args.max_tokens > 0:
        body["max_tokens"] = int(args.max_tokens)

    req = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urlopen(req, timeout=float(args.timeout_s)) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return as_text(raw["choices"][0]["message"]["content"]), model


def pass_value(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv_simple(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Closed-book LLM baseline for benchmark QA without retrieval.")
    parser.add_argument("--dataset", type=Path, required=True, help="Benchmark JSONL/JSON dataset.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL result file.")
    parser.add_argument("--summary", type=Path, required=True, help="Output CSV summary.")
    parser.add_argument("--model", type=str, default="", help="Closed-book answer model. Defaults to LLM_MODEL.")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--judge", action="store_true", help="Run the same answer judge used by run_eval.py.")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Evaluator model. Defaults to LLM_MODEL.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                completed.add(str(obj.get("id") or obj.get("line_no") or ""))

    mode = "a" if args.resume and args.out.exists() else "w"
    total = 0
    ok = 0
    errors = 0
    judge_errors = 0
    summary_values: defaultdict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    start = time.perf_counter()

    with args.out.open(mode, encoding="utf-8") as fh:
        for row in iter_rows(args.dataset):
            key = str(row.get("id") or row.get("_line_no") or "")
            if key in completed:
                continue
            question = as_text(row.get("question"))
            if not question:
                continue

            rec: dict[str, Any] = {
                "id": row.get("id"),
                "line_no": row.get("_line_no"),
                "question": question,
                "rag_mode": "closed_book_deepseek",
                "question_type": infer_question_type(row),
                "gold_doc_ids": row.get("gold_doc_ids"),
                "equivalent_doc_groups": row.get("equivalent_doc_groups"),
                "answer_key": as_text(row.get("answer_key")),
                "key_points": [as_text(x) for x in (row.get("key_points") or []) if as_text(x)],
            }
            t0 = time.perf_counter()
            try:
                answer, used_model = call_closed_book(question, args)
                latency = time.perf_counter() - t0
                rec.update(
                    {
                        "ok": True,
                        "latency_s": latency,
                        "used_model": used_model,
                        "answer": answer,
                        "raw_result": None,
                        "pred_doc_ids": [],
                        "retrieval_metrics": {},
                    }
                )
                ok += 1
                if args.judge and rec["answer_key"]:
                    jt0 = time.perf_counter()
                    judge = _deepseek_judge(
                        question,
                        answer,
                        rec["answer_key"],
                        key_points=rec["key_points"],
                        question_type=rec["question_type"],
                        api_key_override=args.judge_api_key,
                        model_override=args.judge_model,
                    )
                    rec["judge"] = judge
                    rec["judge_latency_s"] = time.perf_counter() - jt0
                    if isinstance(judge, dict) and "error" not in judge:
                        qtype = str(rec["question_type"])
                        summary_values[qtype]["correct_and_relevant_pass"].append(
                            pass_value(judge.get("correct_and_relevant"))
                        )
                        summary_values[qtype]["faithful_pass"].append(pass_value(judge.get("faithful")))
                        for metric in (
                            "correctness_score",
                            "relevance_score",
                            "faithfulness_score",
                            "key_point_coverage_score",
                        ):
                            try:
                                summary_values[qtype][metric].append(float(judge.get(metric, 0.0)))
                            except Exception:
                                pass
                    else:
                        judge_errors += 1
            except Exception as exc:
                rec.update({"ok": False, "error": str(exc), "latency_s": time.perf_counter() - t0})
                errors += 1

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            total += 1
            if args.progress_every and total % int(args.progress_every) == 0:
                elapsed = time.perf_counter() - start
                print(f"[closed_book] total={total} ok={ok} errors={errors} judge_errors={judge_errors} elapsed_s={elapsed:.1f}")
            if args.limit and total >= int(args.limit):
                break

    fieldnames = [
        "question_type",
        "rag_mode",
        "n",
        "correct_and_relevant_pass",
        "faithful_pass",
        "correctness_score",
        "relevance_score",
        "faithfulness_score",
        "key_point_coverage_score",
        "judge_enabled",
        "closed_book_model",
        "judge_model",
    ]
    with args.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        if summary_values:
            for qtype, values in sorted(summary_values.items()):
                writer.writerow(
                    {
                        "question_type": qtype,
                        "rag_mode": "closed_book_deepseek",
                        "n": len(values.get("correctness_score", [])),
                        "correct_and_relevant_pass": fmt(mean(values.get("correct_and_relevant_pass", []))),
                        "faithful_pass": fmt(mean(values.get("faithful_pass", []))),
                        "correctness_score": fmt(mean(values.get("correctness_score", []))),
                        "relevance_score": fmt(mean(values.get("relevance_score", []))),
                        "faithfulness_score": fmt(mean(values.get("faithfulness_score", []))),
                        "key_point_coverage_score": fmt(mean(values.get("key_point_coverage_score", []))),
                        "judge_enabled": bool(args.judge),
                        "closed_book_model": get_answer_model(args),
                        "judge_model": args.judge_model
                        
                        or os.getenv("LLM_MODEL") 
                        or "",
                    }
                )
        else:
            writer.writerow(
                {
                    "question_type": "",
                    "rag_mode": "closed_book_deepseek",
                    "n": total,
                    "judge_enabled": bool(args.judge),
                    "closed_book_model": get_answer_model(args),
                    "judge_model": args.judge_model
                    
                    or os.getenv("LLM_MODEL") 
                    or "",
                }
            )

    print(f"Wrote closed-book rows to {args.out}")
    print(f"Wrote closed-book summary to {args.summary}")


if __name__ == "__main__":
    main()
