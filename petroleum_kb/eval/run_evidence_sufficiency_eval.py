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


def load_dotenv_simple(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip("'").strip('"'))


def iter_result_rows(path: Path):
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


def parse_modes(raw: str) -> set[str] | None:
    modes = {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}
    return modes or None


def rag_mode(row: dict[str, Any]) -> str:
    return as_text(row.get("rag_mode") or row.get("mode") or "unknown").lower()


def question_type(row: dict[str, Any]) -> str:
    qtype = as_text(row.get("question_type")).lower()
    if qtype:
        return qtype
    gold = row.get("gold_doc_ids")
    return "two_doc" if isinstance(gold, list) and len(gold) >= 2 else "single"


def unwrap_raw_result(row: dict[str, Any]) -> Any:
    raw = row.get("raw_result")
    if isinstance(raw, dict) and isinstance(raw.get("raw_result"), (dict, list)):
        return raw.get("raw_result")
    return raw


def doc_id_from_evidence(evidence: dict[str, Any]) -> str:
    doc_id = as_text(evidence.get("doc_id"))
    if doc_id:
        return doc_id
    doi = as_text(evidence.get("doi"))
    if doi:
        return doi if doi.startswith("doi:") else f"doi:{doi}"
    citation = evidence.get("citation") if isinstance(evidence.get("citation"), dict) else {}
    doi = as_text(citation.get("doi"))
    if doi:
        return doi if doi.startswith("doi:") else f"doi:{doi}"
    title = as_text(evidence.get("title") or citation.get("title"))
    year = as_text(evidence.get("year") or citation.get("year"))
    if title:
        return f"title:{title}::{year}"
    return ""


def format_evidence_block(evidence: dict[str, Any], index: int, max_chars_per_doc: int) -> str:
    citation = evidence.get("citation") if isinstance(evidence.get("citation"), dict) else {}
    title = as_text(evidence.get("title") or citation.get("title"))
    doi = as_text(evidence.get("doi") or citation.get("doi"))
    year = as_text(evidence.get("year") or citation.get("year"))
    authors = evidence.get("authors") or citation.get("authors") or []
    if isinstance(authors, list):
        authors_text = "; ".join(as_text(x) for x in authors[:5] if as_text(x))
    else:
        authors_text = as_text(authors)

    chunks = evidence.get("chunks") if isinstance(evidence.get("chunks"), list) else []
    texts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        txt = as_text(chunk.get("text"))
        if txt:
            texts.append(txt)
    if not texts:
        txt = as_text(evidence.get("text"))
        if txt:
            texts.append(txt)

    body = "\n".join(texts).strip()
    if len(body) > max_chars_per_doc:
        body = body[:max_chars_per_doc] + "..."

    header = [
        f"[Doc {index}]",
        f"doc_id: {doc_id_from_evidence(evidence)}",
        f"title: {title}",
        f"doi: {doi}",
        f"year: {year}",
        f"authors: {authors_text}",
    ]
    return "\n".join(header) + "\ntext:\n" + body


def extract_evidences(row: dict[str, Any], context_source: str) -> list[dict[str, Any]]:
    raw = unwrap_raw_result(row)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not isinstance(raw, dict):
        return []

    source = context_source.strip() or "evidences"
    if source == "auto":
        source = "evidences"

    if source in {
        "evidences",
        "candidate_evidences",
        "accumulated_evidences",
        "filtered_evidences",
        "synthesis_evidences",
    }:
        evidences = raw.get(source)
        return [item for item in evidences if isinstance(item, dict)] if isinstance(evidences, list) else []

    if source == "rounds":
        out: list[dict[str, Any]] = []
        rounds = raw.get("rounds") if isinstance(raw.get("rounds"), list) else []
        for rd in rounds:
            if not isinstance(rd, dict):
                continue
            evidences = rd.get("evidences") if isinstance(rd.get("evidences"), list) else []
            out.extend(item for item in evidences if isinstance(item, dict))
        return out

    return []


def build_context(
    row: dict[str, Any],
    *,
    context_source: str,
    max_docs: int,
    max_chars_per_doc: int,
) -> tuple[str, list[str]]:
    evidences = extract_evidences(row, context_source)
    selected = evidences[: max(1, int(max_docs))]
    blocks = [format_evidence_block(ev, idx, max_chars_per_doc) for idx, ev in enumerate(selected, start=1)]
    doc_ids = [doc_id_from_evidence(ev) for ev in selected if doc_id_from_evidence(ev)]
    return "\n\n".join(blocks), doc_ids


def key_points_text(row: dict[str, Any]) -> str:
    points = [as_text(x) for x in (row.get("key_points") or []) if as_text(x)]
    return "\n".join(f"- {point}" for point in points)


def call_deepseek_json(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    api_key = (args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (args.base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = (
        args.model
        
        
        or os.getenv("LLM_MODEL") 
        or os.getenv("OPENAI_MODEL")
        or ""
    )
    if not api_key:
        return {"error": "missing_api_key"}

    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
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
    content = raw["choices"][0]["message"]["content"]
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return {"error": "invalid_json", "raw": content}
    try:
        parsed = json.loads(content[start : end + 1])
    except Exception:
        return {"error": "invalid_json", "raw": content}
    return parsed if isinstance(parsed, dict) else {"error": "invalid_json_type", "raw": content}


def normalize_sufficiency(obj: dict[str, Any], threshold: float) -> dict[str, Any]:
    out = dict(obj)
    try:
        score = float(out.get("evidence_sufficiency_score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))
    if "model_sufficient" not in out and "sufficient" in out:
        out["model_sufficient"] = bool(out.get("sufficient"))
    out["evidence_sufficiency_score"] = score
    out["sufficient"] = bool(score >= threshold)
    out["pass_threshold"] = float(threshold)
    for key in ("missing_evidence", "support_summary", "notes"):
        out[key] = as_text(out.get(key))
    return out


def build_prompt(row: dict[str, Any], context_text: str) -> str:
    return (
        "You are a strict scientific RAG evidence sufficiency judge. Return JSON only with keys:\n"
        '{"evidence_sufficiency_score": number, "sufficient": bool, '
        '"missing_evidence": string, "support_summary": string, "notes": string}\n'
        "Scoring in [0,1]. Evaluate whether the retrieved evidence contains enough information "
        "to answer the question correctly.\n"
        "Rules:\n"
        "- Use only the retrieved evidence.\n"
        "- Do not require exact benchmark gold documents unless the question explicitly asks for those specific papers.\n"
        "- Alternative papers are acceptable if they contain enough evidence to support the answer.\n"
        "- For multi-document questions, check whether the evidence supports all required aspects, comparisons, or references.\n"
        "- Penalize missing key facts, unsupported claims, one-sided evidence, or missing reference metadata requested by the question.\n"
        "- Do not penalize extra relevant evidence.\n"
        "- Be stricter than answer relevance, but less strict than exact gold-document coverage.\n\n"
        f"question_type: {question_type(row)}\n"
        f"question: {as_text(row.get('question'))}\n"
        f"answer_key: {as_text(row.get('answer_key'))}\n"
        f"key_points:\n{key_points_text(row)}\n\n"
        f"answer_to_evaluate:\n{as_text(row.get('answer'))}\n\n"
        f"retrieved_evidence:\n{context_text}\n"
    )


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    load_dotenv_simple(Path(".env"))
    parser = argparse.ArgumentParser(description="Offline evidence sufficiency evaluation for saved RAG results.")
    parser.add_argument("--input", type=Path, required=True, help="Saved run_eval JSON or JSONL result file.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL with per-row evidence sufficiency scores.")
    parser.add_argument("--summary", type=Path, required=True, help="Output CSV summary.")
    parser.add_argument("--modes", type=str, default="", help="Comma-separated rag modes to include.")
    parser.add_argument(
        "--context-source",
        choices=("auto", "evidences", "candidate_evidences", "accumulated_evidences", "filtered_evidences", "synthesis_evidences", "rounds"),
        default="evidences",
    )
    parser.add_argument("--max-docs", type=int, default=12)
    parser.add_argument("--max-chars-per-doc", type=int, default=1600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit-per-mode", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Pass threshold for evidence_sufficiency_score. Default: 0.6.",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Evaluator model. Defaults to LLM_MODEL.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    per_mode_counts: defaultdict[str, int] = defaultdict(int)
    summary_values: defaultdict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows_written = 0
    skipped: defaultdict[str, int] = defaultdict(int)
    start_time = time.perf_counter()

    with args.out.open("w", encoding="utf-8") as out_fh:
        for row in iter_result_rows(args.input):
            mode = rag_mode(row)
            if modes is not None and mode not in modes:
                skipped["mode_filter"] += 1
                continue
            if args.limit_per_mode and per_mode_counts[mode] >= int(args.limit_per_mode):
                skipped["limit_per_mode"] += 1
                continue

            context_text, context_doc_ids = build_context(
                row,
                context_source=args.context_source,
                max_docs=int(args.max_docs),
                max_chars_per_doc=int(args.max_chars_per_doc),
            )
            if not context_text.strip():
                skipped["missing_context"] += 1
                continue

            prompt = build_prompt(row, context_text)
            t0 = time.perf_counter()
            try:
                judged = call_deepseek_json(prompt, args)
                latency = time.perf_counter() - t0
                if "error" not in judged:
                    judged = normalize_sufficiency(judged, float(args.threshold))
            except Exception as exc:
                latency = time.perf_counter() - t0
                judged = {"error": str(exc)}

            record = {
                "id": row.get("id"),
                "line_no": row.get("_line_no"),
                "rag_mode": mode,
                "question_type": question_type(row),
                "context_source": args.context_source,
                "context_doc_count": len(context_doc_ids),
                "context_doc_ids": context_doc_ids,
                "gold_doc_ids": row.get("gold_doc_ids"),
                "evidence_sufficiency": judged,
                "judge_latency_s": latency,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

            rows_written += 1
            per_mode_counts[mode] += 1
            if "error" not in judged:
                key = (question_type(row), mode, args.context_source)
                summary_values[key]["evidence_sufficiency_score"].append(float(judged.get("evidence_sufficiency_score", 0.0)))
                summary_values[key]["evidence_sufficient_pass"].append(1.0 if judged.get("sufficient") else 0.0)
                summary_values[key]["context_doc_count"].append(float(len(context_doc_ids)))
                summary_values[key]["judge_latency_s"].append(float(latency))
            else:
                skipped["judge_error"] += 1

            if args.progress_every and rows_written % int(args.progress_every) == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[sufficiency] scored={rows_written} elapsed_s={elapsed:.1f} skipped={dict(skipped)}")

            if args.limit and rows_written >= int(args.limit):
                break

    fieldnames = [
        "question_type",
        "rag_mode",
        "context_source",
        "n",
        "evidence_sufficiency_score",
        "evidence_sufficient_pass",
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
                    "n": len(vals["evidence_sufficiency_score"]),
                    "evidence_sufficiency_score": fmt(mean(vals["evidence_sufficiency_score"])),
                    "evidence_sufficient_pass": fmt(mean(vals["evidence_sufficient_pass"])),
                    "context_doc_count": fmt(mean(vals["context_doc_count"])),
                    "judge_latency_s": fmt(mean(vals["judge_latency_s"])),
                }
            )

    print(f"Wrote evidence sufficiency rows to {args.out}")
    print(f"Wrote evidence sufficiency summary to {args.summary}")
    print(f"Skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
