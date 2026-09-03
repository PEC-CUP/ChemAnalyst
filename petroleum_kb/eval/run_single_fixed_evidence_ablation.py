from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm.deepseek_client import DeepSeekClient
from app.rag.kb_backend import retrieve_naive
from petroleum_kb.eval.run_eval import (
    _compute_equivalent_coverage,
    _mean,
    _normalize_equivalent_doc_groups,
    compute_retrieval_metrics,
    iter_jsonl,
    load_dotenv_simple,
)


def _doc_id(doc: dict[str, Any]) -> str:
    did = str(doc.get("doc_id") or "").strip()
    if did:
        return did
    doi = str(doc.get("doi") or "").strip()
    if doi:
        return doi if doi.startswith("doi:") else f"doi:{doi}"
    title = str(doc.get("title") or "").strip()
    year = doc.get("year") or ""
    if title:
        return f"title:{title}::{year}"
    return ""


def _pred_doc_ids(docs: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        did = _doc_id(doc)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def _fixed_context(docs: list[dict[str, Any]], *, max_chars_per_chunk: int) -> str:
    blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        title = doc.get("title") or doc.get("file_name") or "unknown"
        year = doc.get("year") or ""
        doi = doc.get("doi") or ""
        score = doc.get("score")
        text = str(doc.get("text") or "")[:max_chars_per_chunk]
        blocks.append(
            f"[CHUNK {idx}] doc_id={_doc_id(doc)} | title={title} | year={year} | doi={doi} | score={score}\n{text}"
        )
    return "\n\n".join(blocks).strip()


def _naive_style_answer(llm: DeepSeekClient, *, question: str, context: str) -> str:
    return llm.chat(query=question, context=context)


def _qa_oriented_style_answer(llm: DeepSeekClient, *, question: str, context: str) -> str:
    prompt = (
        "You are an QA-oriented evidence synthesis module for a fixed-evidence ablation.\n"
        "The retrieved evidence below is fixed and identical to the naive baseline evidence. "
        "Do not retrieve, assume hidden documents, or use outside knowledge.\n\n"
        "Task:\n"
        "- Identify the chunks that directly answer the question.\n"
        "- Synthesize a precise answer from those chunks only.\n"
        "- Preserve exact numbers, method names, compound classes, experimental conditions, and trends.\n"
        "- Do not replace specific evidence with generic domain knowledge.\n"
        "- If evidence is incomplete, state the missing part briefly.\n"
        "- Cite supporting chunks as [CHUNK n].\n"
        "- Answer in the same language as the question. For English questions, answer in English.\n\n"
        f"Question:\n{question}\n\nFixed evidence:\n{context}\n"
    )
    return llm.chat(query=prompt)


def _judge_with_client(
    llm: DeepSeekClient,
    question: str,
    answer: str,
    answer_key: str,
    *,
    key_points: list[str] | None = None,
    question_type: str = "single",
) -> dict[str, Any]:
    key_points = [str(x).strip() for x in (key_points or []) if str(x).strip()]
    key_points_block = ""
    if key_points:
        key_points_block = "key_points:\n- " + "\n- ".join(key_points) + "\n"

    prompt = (
        "You are a QA judge. Return JSON only with keys:\n"
        '{"correctness_score": number, "relevance_score": number, "faithfulness_score": number, '
        '"key_point_coverage_score": number, "correct": bool, "relevant": bool, '
        '"correct_and_relevant": bool, "faithful": bool, "notes": string}\n'
        "Scoring in [0,1].\n"
        "Rules:\n"
        "- correctness_score: factual alignment to answer_key and key_points.\n"
        "- relevance_score: whether answer addresses the question.\n"
        "- faithfulness_score: whether the answer stays faithful to the evidence implied by answer_key/key_points.\n"
        "- key_point_coverage_score: how completely the answer covers the required key_points.\n"
        "- Use key_points as the primary rubric. Do not require wording overlap with answer_key.\n"
        "- Do not over-penalize additional correct details not explicitly in answer_key.\n"
        "- Penalize clear contradictions, unsupported key claims, or missing required core points.\n\n"
        f"question_type: {question_type}\n"
        f"question: {question}\n"
        f"answer_key: {answer_key}\n"
        f"{key_points_block}"
        f"answer: {answer}\n"
    )
    content = llm.chat(query=prompt)
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return {"error": "invalid_judge_json", "raw": content}
    try:
        parsed = json.loads(content[start : end + 1])
    except Exception:
        return {"error": "invalid_judge_json", "raw": content}
    if not isinstance(parsed, dict):
        return {"error": "invalid_judge_type"}
    out = dict(parsed)
    for key in ("correctness_score", "relevance_score", "faithfulness_score", "key_point_coverage_score"):
        try:
            out[key] = max(0.0, min(1.0, float(out.get(key, 0.0))))
        except Exception:
            out[key] = 0.0
    out["correct"] = bool(out.get("correct", out["correctness_score"] >= 0.5))
    out["relevant"] = bool(out.get("relevant", out["relevance_score"] >= 0.5))
    out["faithful"] = bool(out.get("faithful", out["faithfulness_score"] >= 0.5))
    out["correct_and_relevant"] = bool(out.get("correct_and_relevant", (out["correct"] and out["relevant"])))
    return out


def _safe_float(obj: dict[str, Any], key: str) -> float | None:
    val = obj.get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _write_summary(path: Path, records: list[dict[str, Any]], *, dataset: Path, retrieval_k: int) -> None:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_mode.setdefault(str(rec.get("ablation_mode")), []).append(rec)

    row: dict[str, Any] = {
        "dataset": str(dataset),
        "question_count": len({r.get("id") for r in records}),
        "total_rows": len(records),
        "retrieval_k": retrieval_k,
    }
    metrics = [
        "correctness_score",
        "faithfulness_score",
        "key_point_coverage_score",
        "relevance_score",
        "hit_at_k",
        "recall_at_k",
        "map",
        "mrr",
        "precision_at_k",
        "latency_s",
    ]
    bool_metrics = ["correct", "relevant", "correct_and_relevant", "faithful"]
    for mode, rows in sorted(by_mode.items()):
        row[f"{mode}_n"] = len(rows)
        for metric in bool_metrics:
            vals = [
                bool((r.get("judge") or {}).get(metric))
                for r in rows
                if isinstance((r.get("judge") or {}).get(metric), bool)
            ]
            row[f"{mode}_{metric}_pass"] = _mean([1.0 if v else 0.0 for v in vals])
        for metric in metrics:
            vals: list[float] = []
            for rec in rows:
                if metric in {"hit_at_k", "recall_at_k", "map", "mrr", "precision_at_k"}:
                    val = _safe_float(rec.get("retrieval_metrics") or {}, metric)
                elif metric == "latency_s":
                    val = _safe_float(rec, metric)
                else:
                    val = _safe_float(rec.get("judge") or {}, metric)
                if val is not None:
                    vals.append(val)
            row[f"{mode}_{metric}"] = _mean(vals)

    summary_path = path.with_suffix(".summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fixed-evidence single-doc ablation: same naive top-k chunks, naive-style vs QA-oriented synthesis."
    )
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--retrieval-k", type=int, default=8)
    ap.add_argument("--max-chars-per-chunk", type=int, default=1200)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--dotenv", type=Path, default=ROOT / ".env")
    args = ap.parse_args()

    load_dotenv_simple(args.dotenv)
    rows = list(iter_jsonl(args.dataset))[: max(1, int(args.num))]
    llm = DeepSeekClient()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        total_requests = len(rows) * 2
        done = 0
        started = time.perf_counter()
        for row in rows:
            question = str(row.get("question") or "")
            answer_key = str(row.get("answer_key") or "")
            key_points = [str(x) for x in (row.get("key_points") or []) if str(x).strip()]
            gold_doc_ids = [str(x) for x in (row.get("gold_doc_ids") or []) if str(x).strip()]
            eq_groups = _normalize_equivalent_doc_groups(row)
            docs = retrieve_naive(question, top_k=int(args.retrieval_k))
            context = _fixed_context(docs, max_chars_per_chunk=int(args.max_chars_per_chunk))
            pred_doc_ids = _pred_doc_ids(docs)
            retrieval_metrics = compute_retrieval_metrics(pred_doc_ids, gold_doc_ids, int(args.retrieval_k))
            equivalent_group_coverage = _compute_equivalent_coverage(pred_doc_ids, eq_groups)

            for mode in ("fixed_naive", "fixed_qa_oriented_synthesis"):
                t0 = time.perf_counter()
                try:
                    if mode == "fixed_naive":
                        answer = _naive_style_answer(llm, question=question, context=context)
                    else:
                        answer = _qa_oriented_style_answer(llm, question=question, context=context)
                    ok = True
                    error = None
                except Exception as exc:
                    answer = ""
                    ok = False
                    error = str(exc)

                rec: dict[str, Any] = {
                    "id": row.get("id"),
                    "question": question,
                    "question_type": row.get("question_type") or row.get("type") or "single",
                    "ablation_mode": mode,
                    "gold_doc_ids": gold_doc_ids,
                    "answer_key": answer_key,
                    "key_points": key_points,
                    "ok": ok,
                    "error": error,
                    "latency_s": time.perf_counter() - t0,
                    "answer": answer,
                    "pred_doc_ids": pred_doc_ids,
                    "retrieval_metrics": retrieval_metrics,
                    "equivalent_group_coverage": equivalent_group_coverage,
                    "raw_result": docs,
                }
                if args.judge and answer_key and answer:
                    rec["judge"] = _judge_with_client(
                        llm,
                        question,
                        answer,
                        answer_key,
                        key_points=key_points,
                        question_type="single",
                    )
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(rec)
                done += 1
                if args.progress_every > 0 and done % args.progress_every == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    avg = elapsed / done
                    eta = avg * max(total_requests - done, 0)
                    print(
                        f"[ablation] progress: {done}/{total_requests} "
                        f"questions={len({r.get('id') for r in records})}/{len(rows)} "
                        f"avg_latency_s={avg:.3f} eta_s={eta:.1f}",
                        flush=True,
                    )

    _write_summary(args.out, records, dataset=args.dataset, retrieval_k=int(args.retrieval_k))
    print(json.dumps({"status": "success", "out": str(args.out), "rows": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
