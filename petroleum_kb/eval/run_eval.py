from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def load_dotenv_simple(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip("'").strip('"')
        os.environ.setdefault(key, val)


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
    )
    with urlopen(req, timeout=timeout_s) as resp:
        status = int(getattr(resp, "status", 200))
        data = json.loads(resp.read().decode("utf-8"))
    return status, data


def get_json(url: str, timeout_s: float) -> tuple[int, dict[str, Any]]:
    req = Request(url, method="GET", headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout_s) as resp:
        status = int(getattr(resp, "status", 200))
        data = json.loads(resp.read().decode("utf-8"))
    return status, data


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def infer_question_type(row: dict[str, Any]) -> str:
    t = str(row.get("question_type") or row.get("type") or "").strip().lower()
    if t in {"single", "two_doc"}:
        return t
    gold = row.get("gold_doc_ids")
    if isinstance(gold, list) and len(gold) >= 2:
        return "two_doc"
    return "single"


def _normalize_equivalent_doc_groups(row: dict[str, Any]) -> list[list[str]]:
    gold_doc_ids = [str(x).strip() for x in (row.get("gold_doc_ids") or []) if str(x).strip()]
    raw = row.get("equivalent_doc_groups")
    groups: list[list[str]] = []
    if isinstance(raw, list):
        for item in raw:
            docs: list[str] = []
            if isinstance(item, dict):
                if isinstance(item.get("gold_doc_id"), str) and item.get("gold_doc_id", "").strip():
                    docs.append(str(item["gold_doc_id"]).strip())
                if isinstance(item.get("doc_ids"), list):
                    docs.extend(str(x).strip() for x in item["doc_ids"] if str(x).strip())
            elif isinstance(item, list):
                docs.extend(str(x).strip() for x in item if str(x).strip())
            docs = list(dict.fromkeys(docs))
            if docs:
                groups.append(docs)
    if not groups:
        return [[g] for g in gold_doc_ids]
    used = {d for grp in groups for d in grp}
    for g in gold_doc_ids:
        if g not in used:
            groups.append([g])
    return groups


def _compute_equivalent_coverage(pred_doc_ids: list[str], eq_groups: list[list[str]]) -> dict[str, Any]:
    pred_set = {str(x).strip() for x in pred_doc_ids if str(x).strip()}
    total = len(eq_groups)
    hits = 0
    matched_groups: list[int] = []
    missing_groups: list[int] = []
    for idx, grp in enumerate(eq_groups, start=1):
        grp_set = {str(x).strip() for x in grp if str(x).strip()}
        if grp_set & pred_set:
            hits += 1
            matched_groups.append(idx)
        else:
            missing_groups.append(idx)
    coverage = (hits / total) if total > 0 else 1.0
    return {
        "coverage_score": float(coverage),
        "groups_total": total,
        "groups_hit": hits,
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
        "pass_all_groups": bool(total == 0 or hits == total),
        "pass_partial_threshold": bool(total == 0 or coverage >= 0.5),
    }


def _unwrap_eval_raw_result(raw_result: Any) -> Any:
    current = raw_result
    seen: set[int] = set()
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        # Runtime/planner wrappers place the actual retrieval payload under rag_result.
        nested = current.get("rag_result")
        if isinstance(nested, (dict, list)):
            current = nested
            continue
        # Some planner flows place knowledge payload one level deeper.
        knowledge = current.get("knowledge")
        if isinstance(knowledge, dict):
            nested = knowledge.get("raw_result")
            if isinstance(nested, (dict, list)):
                current = nested
                continue
        break
    return current


def _doc_id_from_item(item: dict[str, Any]) -> str | None:
    did = str(item.get("doc_id") or "").strip()
    if did:
        return did
    doi = str(item.get("doi") or "").strip()
    if doi:
        return doi if doi.startswith("doi:") else f"doi:{doi}"
    title = str(item.get("title") or "").strip()
    year = item.get("year") or ""
    if title:
        return f"title:{title}::{year}"
    src = str(item.get("source") or "").strip()
    fn = str(item.get("file_name") or "").strip()
    if fn or src:
        return f"file:{fn}::{src}"
    return None


def _extract_doc_ids_from_evidences(evidences: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        did = _doc_id_from_item(ev)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def extract_pred_doc_ids(rag_mode: str, raw_result: Any) -> list[str]:
    unwrapped = _unwrap_eval_raw_result(raw_result)
    if isinstance(unwrapped, dict):
        evidences = unwrapped.get("evidences") if isinstance(unwrapped.get("evidences"), list) else []
        if evidences:
            return _extract_doc_ids_from_evidences([ev for ev in evidences if isinstance(ev, dict)])
        documents = unwrapped.get("documents") if isinstance(unwrapped.get("documents"), list) else []
        if documents:
            return _extract_doc_ids_from_evidences([doc for doc in documents if isinstance(doc, dict)])
        rounds = unwrapped.get("rounds") if isinstance(unwrapped.get("rounds"), list) else []
        round_evidences: list[dict[str, Any]] = []
        for rd in rounds:
            if not isinstance(rd, dict):
                continue
            evs = rd.get("evidences") if isinstance(rd.get("evidences"), list) else []
            round_evidences.extend([ev for ev in evs if isinstance(ev, dict)])
        if round_evidences:
            return _extract_doc_ids_from_evidences(round_evidences)

    chunks = unwrapped if isinstance(unwrapped, list) else []
    return _extract_doc_ids_from_evidences([ch for ch in chunks if isinstance(ch, dict)])


def compute_retrieval_metrics(pred_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> dict[str, float]:
    pred = pred_doc_ids[: max(1, int(k))]
    gold = [g for g in gold_doc_ids if isinstance(g, str) and g.strip()]
    gold_set = set(gold)
    if not gold_set:
        return {}

    hit = 1.0 if any(p in gold_set for p in pred) else 0.0
    recall = sum(1 for p in pred if p in gold_set) / max(len(gold_set), 1)
    precision = sum(1 for p in pred if p in gold_set) / max(len(pred), 1)

    rr = 0.0
    for i, p in enumerate(pred, start=1):
        if p in gold_set:
            rr = 1.0 / i
            break

    ap_hits = 0
    ap_sum = 0.0
    for i, p in enumerate(pred, start=1):
        if p in gold_set:
            ap_hits += 1
            ap_sum += ap_hits / i
    ap = ap_sum / max(len(gold_set), 1)

    return {
        "hit_at_k": hit,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr": rr,
        "map": ap,
        "k": float(k),
    }


def _eval_record_key(rec: dict[str, Any]) -> tuple[str, str]:
    return (str(rec.get("id") or "").strip(), str(rec.get("rag_mode") or "").strip().lower())


def _has_retryable_judge_error(rec: dict[str, Any], *, require_judge: bool) -> bool:
    if str(rec.get("judge_error") or "").strip():
        return True
    if not require_judge:
        return False
    for key in ("judge", "context_judge", "evidence_sufficiency"):
        value = rec.get(key)
        if not isinstance(value, dict):
            return True
        if "error" in value:
            return True
    return False


def _is_retryable_eval_error(rec: dict[str, Any], *, require_judge: bool = False) -> bool:
    if rec.get("ok") is not True:
        return True
    if str(rec.get("error") or "").strip():
        return True
    status = rec.get("http_status")
    if status not in (None, 200, "200"):
        return True
    if _has_retryable_judge_error(rec, require_judge=require_judge):
        return True
    return False


def _accumulate_existing_record_metrics(
    rec: dict[str, Any],
    by_mode: dict[str, dict[str, list[float]]],
    latencies: list[float],
    judge_latencies: list[float],
) -> tuple[int, int, int, int]:
    total = 1
    ok = 1 if rec.get("ok") is True else 0
    errors = 0 if rec.get("ok") is True else 1
    judge_errors = 0
    mode = str(rec.get("rag_mode") or "").strip().lower()
    bucket = by_mode.get(mode)
    if not bucket:
        return total, ok, errors, judge_errors

    try:
        latencies.append(float(rec.get("latency_s")))
    except (TypeError, ValueError):
        pass
    try:
        judge_latencies.append(float(rec.get("judge_latency_s")))
    except (TypeError, ValueError):
        pass

    rm = rec.get("retrieval_metrics")
    if isinstance(rm, dict):
        bucket["hit"].append(float(rm.get("hit_at_k", 0.0)))
        bucket["recall"].append(float(rm.get("recall_at_k", 0.0)))
        bucket["precision"].append(float(rm.get("precision_at_k", 0.0)))
        bucket["mrr"].append(float(rm.get("mrr", 0.0)))
        bucket["map"].append(float(rm.get("map", 0.0)))

    if "two_doc_coverage" in rec:
        bucket["cov2"].append(float(rec.get("two_doc_coverage") or 0.0))
    if "two_doc_coverage_strict" in rec:
        bucket["cov2_strict"].append(float(rec.get("two_doc_coverage_strict") or 0.0))
    eq_cov = rec.get("equivalent_group_coverage")
    if isinstance(eq_cov, dict):
        bucket["evidence_cov"].append(float(eq_cov.get("coverage_score", 0.0)))

    judge = rec.get("judge")
    if isinstance(judge, dict) and "error" not in judge:
        raw_pass = judge.get("raw_judge_pass") if isinstance(judge.get("raw_judge_pass"), dict) else {}
        gated_pass = judge.get("gated_judge_pass") if isinstance(judge.get("gated_judge_pass"), dict) else {}
        bucket["corr_rel"].append(1.0 if raw_pass.get("correct_and_relevant") is True else 0.0)
        bucket["faithful"].append(1.0 if raw_pass.get("faithful") is True else 0.0)
        bucket["corr_rel_gated"].append(1.0 if gated_pass.get("correct_and_relevant") is True else 0.0)
        bucket["faithful_gated"].append(1.0 if gated_pass.get("faithful") is True else 0.0)
        bucket["correctness_score"].append(float(judge.get("correctness_score", 0.0)))
        bucket["faithfulness_score"].append(float(judge.get("faithfulness_score", 0.0)))
        bucket["key_point_coverage_score"].append(float(judge.get("key_point_coverage_score", 0.0)))
    elif isinstance(judge, dict) and "error" in judge:
        judge_errors += 1
    if str(rec.get("judge_error") or "").strip():
        judge_errors += 1

    context_judge = rec.get("context_judge")
    if isinstance(context_judge, dict) and "error" not in context_judge:
        bucket["ctx_rel"].append(float(context_judge.get("context_relevance", 0.0)))
        bucket["ctx_prec"].append(float(context_judge.get("context_precision", 0.0)))
        bucket["ctx_rec"].append(float(context_judge.get("context_recall", 0.0)))
    elif isinstance(context_judge, dict) and "error" in context_judge:
        judge_errors += 1

    suff = rec.get("evidence_sufficiency")
    if isinstance(suff, dict) and "error" not in suff:
        bucket["suff_score"].append(float(suff.get("evidence_sufficiency_score", 0.0)))
        bucket["suff_pass"].append(1.0 if suff.get("sufficient") is True else 0.0)
    elif isinstance(suff, dict) and "error" in suff:
        judge_errors += 1

    return total, ok, errors, judge_errors


def _deepseek_judge(
    question: str,
    answer: str,
    answer_key: str,
    *,
    key_points: list[str] | None = None,
    question_type: str = "single",
    api_key_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    api_key = (api_key_override or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    model = (
        model_override
        or os.getenv("LLM_MODEL")
        
        
        or ""
    )
    if not api_key:
        return {"error": "missing_api_key"}
    if not model:
        return {"error": "missing_model"}

    key_points = [str(x).strip() for x in (key_points or []) if str(x).strip()]
    key_points_block = ""
    if key_points:
        key_points_block = "key_points:\n- " + "\n- ".join(key_points) + "\n"

    prompt = (
        "You are a QA judge. Return JSON only with keys:\n"
        '{"correctness_score": number, "relevance_score": number, "faithfulness_score": number, "key_point_coverage_score": number, '
        '"correct": bool, "relevant": bool, "correct_and_relevant": bool, "faithful": bool, "notes": string}\n'
        "Scoring in [0,1].\n"
        "Rules:\n"
        "- correctness_score: factual alignment to answer_key and key_points.\n"
        "- relevance_score: whether answer addresses the question.\n"
        "- faithfulness_score: whether the answer stays faithful to the evidence implied by answer_key/key_points.\n"
        "- key_point_coverage_score: how completely the answer covers the required key_points.\n"
        "- Use key_points as the primary rubric for two-doc questions. Do not require wording overlap with answer_key.\n"
        "- DO NOT over-penalize additional correct details not explicitly in answer_key;\n"
        "  penalize only clear contradictions, unsupported key claims, or missing required core points.\n"
        "- Keep strict on contradictions, but tolerant to paraphrase and harmless elaboration.\n\n"
        f"question_type: {question_type}\n"
        f"question: {question}\n"
        f"answer_key: {answer_key}\n"
        f"{key_points_block}"
        f"answer: {answer}\n"
    )
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
    with urlopen(req, timeout=120.0) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    content = raw["choices"][0]["message"]["content"]
    s = content.find("{")
    e = content.rfind("}")
    if s < 0 or e <= s:
        return {"error": "invalid_judge_json", "raw": content}
    try:
        parsed = json.loads(content[s : e + 1])
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
    # Backward-compatible bools if model omitted them.
    out["correct"] = bool(out.get("correct", out["correctness_score"] >= 0.5))
    out["relevant"] = bool(out.get("relevant", out["relevance_score"] >= 0.5))
    out["faithful"] = bool(out.get("faithful", out["faithfulness_score"] >= 0.5))
    out["correct_and_relevant"] = bool(out.get("correct_and_relevant", (out["correct"] and out["relevant"])))
    return out


def _collect_context_text(raw_result: Any, rag_mode: str, max_chars: int) -> str:
    parts: list[str] = []
    unwrapped = _unwrap_eval_raw_result(raw_result)
    if rag_mode in {"naive", "qa_oriented", "iterative_review"} and isinstance(unwrapped, dict):
        # Prefer final evidences; fallback to round evidences.
        evidences = unwrapped.get("evidences") if isinstance(unwrapped.get("evidences"), list) else []
        if not evidences:
            rounds = unwrapped.get("rounds") if isinstance(unwrapped.get("rounds"), list) else []
            for rd in rounds:
                if not isinstance(rd, dict):
                    continue
                evs = rd.get("evidences") if isinstance(rd.get("evidences"), list) else []
                evidences.extend([e for e in evs if isinstance(e, dict)])
        for ev in evidences:
            chunks = ev.get("chunks") if isinstance(ev.get("chunks"), list) else []
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                txt = str(ch.get("text") or "").strip()
                if txt:
                    parts.append(txt)
    elif isinstance(unwrapped, list):
        for ch in unwrapped:
            if not isinstance(ch, dict):
                continue
            txt = str(ch.get("text") or "").strip()
            if txt:
                parts.append(txt)
    merged = "\n\n".join(parts)
    return merged[: max(256, int(max_chars))]


def _resolve_eval_model(model_override: str | None = None) -> str:
    return (
        model_override
        or os.getenv("LLM_MODEL")
        
        
        or ""
    )


def _resolve_sufficiency_model(model_override: str | None = None) -> str:
    return (
        model_override
        
        or os.getenv("LLM_MODEL")
        
        
        or ""
    )


def _deepseek_context_judge(
    question: str,
    answer_key: str,
    context_text: str,
    api_key_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    api_key = (api_key_override or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    model = _resolve_eval_model(model_override)
    if not api_key:
        return {"error": "missing_api_key"}
    if not context_text.strip():
        return {"error": "empty_context"}

    prompt = (
        "You are a strict RAG context evaluator. Return JSON only with keys:\n"
        '{"context_relevance": number, "context_precision": number, "context_recall": number, "notes": string}\n'
        "Scoring in [0,1], use only retrieved context:\n"
        "- context_relevance: how relevant the retrieved context is to the user question.\n"
        "- context_precision: proportion of retrieved context that is useful/non-noisy for answering.\n"
        "- context_recall: whether retrieved context covers the key evidence required by answer_key.\n"
        "Do not use external knowledge.\n\n"
        f"question: {question}\n"
        f"answer_key: {answer_key}\n"
        f"retrieved_context:\n{context_text}\n"
    )
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
    with urlopen(req, timeout=120.0) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    content = raw["choices"][0]["message"]["content"]
    s = content.find("{")
    e = content.rfind("}")
    if s < 0 or e <= s:
        return {"error": "invalid_context_judge_json", "raw": content}
    try:
        parsed = json.loads(content[s : e + 1])
    except Exception:
        return {"error": "invalid_context_judge_json", "raw": content}
    if not isinstance(parsed, dict):
        return {"error": "invalid_context_judge_type"}
    out = dict(parsed)
    for key in ("context_relevance", "context_precision", "context_recall"):
        try:
            val = float(out.get(key, 0.0))
            out[key] = max(0.0, min(1.0, val))
        except Exception:
            out[key] = 0.0
    return out


def _normalize_evidence_sufficiency(obj: dict[str, Any], threshold: float) -> dict[str, Any]:
    out = dict(obj)
    try:
        score = float(out.get("evidence_sufficiency_score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))
    if "model_sufficient" not in out and "sufficient" in out:
        out["model_sufficient"] = bool(out.get("sufficient"))
    out["evidence_sufficiency_score"] = score
    out["sufficient"] = bool(score >= float(threshold))
    out["pass_threshold"] = float(threshold)
    for key in ("missing_evidence", "support_summary", "notes"):
        out[key] = str(out.get(key) or "").strip()
    return out


def _deepseek_evidence_sufficiency_judge(
    *,
    question: str,
    answer: str,
    answer_key: str,
    key_points: list[str] | None,
    question_type: str,
    context_text: str,
    threshold: float,
    api_key_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    api_key = (api_key_override or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    model = _resolve_sufficiency_model(model_override)
    if not api_key:
        return {"error": "missing_api_key"}
    if not context_text.strip():
        return {"error": "empty_context"}

    points = [str(x).strip() for x in (key_points or []) if str(x).strip()]
    key_points_block = "\n".join(f"- {point}" for point in points)
    prompt = (
        "You are a strict scientific RAG evidence sufficiency judge. Return JSON only with keys:\n"
        '{"evidence_sufficiency_score": number, "sufficient": bool, '
        '"missing_evidence": string, "support_summary": string, "notes": string}\n'
        "Scoring in [0,1]. Evaluate whether the retrieved evidence contains enough information "
        "to support the answer and answer the question correctly.\n"
        "Rules:\n"
        "- Use only the retrieved evidence.\n"
        "- Do not require exact benchmark gold documents unless the question explicitly asks for those specific papers.\n"
        "- Alternative papers are acceptable if they contain enough evidence to support the answer.\n"
        "- For multi-document questions, check whether the evidence supports all required aspects, comparisons, or references.\n"
        "- Penalize missing key facts, unsupported claims, one-sided evidence, or missing reference metadata requested by the question.\n"
        "- Do not penalize extra relevant evidence.\n"
        "- This is an evidence sufficiency score, not exact gold-document coverage.\n\n"
        f"question_type: {question_type}\n"
        f"question: {question}\n"
        f"answer_key: {answer_key}\n"
        f"key_points:\n{key_points_block}\n\n"
        f"answer_to_evaluate:\n{answer}\n\n"
        f"retrieved_evidence:\n{context_text}\n"
    )
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
    with urlopen(req, timeout=120.0) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    content = raw["choices"][0]["message"]["content"]
    s = content.find("{")
    e = content.rfind("}")
    parsed: dict[str, Any] | None = None
    if s >= 0 and e > s:
        try:
            candidate = json.loads(content[s : e + 1])
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None
    if parsed is None:
        score_match = re.search(r'"evidence_sufficiency_score"\s*:\s*([0-9.]+)', content)
        pass_match = re.search(r'"sufficient"\s*:\s*(true|false)', content, flags=re.IGNORECASE)
        if not score_match:
            return {"error": "invalid_evidence_sufficiency_json", "raw": content}
        parsed = {
            "evidence_sufficiency_score": float(score_match.group(1)),
            "sufficient": pass_match.group(1).lower() == "true" if pass_match else False,
            "missing_evidence": "",
            "support_summary": "",
            "notes": "Recovered score from non-strict JSON.",
            "raw": content,
            "parse_warning": "invalid_json_recovered",
        }
    return _normalize_evidence_sufficiency(parsed, threshold)


def apply_evidence_gate(
    *,
    judge_obj: dict[str, Any],
    question_type: str,
    gold_doc_ids: list[str],
    pred_doc_ids: list[str],
    equivalent_doc_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Gate judge result by retrieval evidence, with soft support for equivalent doc groups."""
    gold = [g for g in gold_doc_ids if isinstance(g, str) and g.strip()]
    pred_set = set([p for p in pred_doc_ids if isinstance(p, str) and p.strip()])
    eq_groups = equivalent_doc_groups or [[g] for g in gold]
    if not gold:
        judge_obj["evidence_gate"] = {"applied": False, "pass": True, "reason": "no_gold_doc_ids"}
        return judge_obj

    strict_coverage = 1.0 if set(gold).issubset(pred_set) else (sum(1 for g in gold if g in pred_set) / max(len(gold), 1))
    soft = _compute_equivalent_coverage(pred_doc_ids, eq_groups)

    if question_type == "two_doc":
        strict_pass = set(gold).issubset(pred_set)
        soft_pass = bool(soft["pass_partial_threshold"])
        reason = "all_equivalent_groups_hit" if bool(soft["pass_all_groups"]) else "missing_some_gold_groups"
    else:
        strict_pass = any(g in pred_set for g in gold)
        soft_pass = bool(soft["pass_all_groups"])
        reason = "hit_equivalent_group" if soft_pass else "miss_equivalent_group"

    judge_obj["evidence_gate"] = {
        "applied": True,
        "pass": soft_pass,
        "reason": reason,
        "strict_pass": strict_pass,
        "strict_coverage_score": float(strict_coverage),
        "soft_coverage_score": float(soft["coverage_score"]),
        "groups_total": int(soft["groups_total"]),
        "groups_hit": int(soft["groups_hit"]),
        "matched_groups": soft["matched_groups"],
        "missing_groups": soft["missing_groups"],
    }
    notes = str(judge_obj.get("notes") or "")
    gate_note = f"[evidence_gate:{reason}]"
    judge_obj["notes"] = f"{notes} {gate_note}".strip()
    # Keep raw judge untouched; expose gated view in parallel.
    raw_correct = bool(judge_obj.get("correct", False))
    raw_relevant = bool(judge_obj.get("relevant", False))
    raw_faithful = bool(judge_obj.get("faithful", False))
    judge_obj["raw_judge_pass"] = {
        "correct_and_relevant": bool(judge_obj.get("correct_and_relevant", raw_correct and raw_relevant)),
        "faithful": raw_faithful,
    }
    judge_obj["gated_judge_pass"] = {
        "correct_and_relevant": bool(soft_pass and bool(judge_obj["raw_judge_pass"]["correct_and_relevant"])),
        "faithful": bool(soft_pass and raw_faithful),
    }
    judge_obj["gated_scores"] = {
        "correctness_score": float(judge_obj.get("correctness_score", 0.0)) * float(soft["coverage_score"]),
        "faithfulness_score": float(judge_obj.get("faithfulness_score", 0.0)) * float(soft["coverage_score"]),
        "key_point_coverage_score": float(judge_obj.get("key_point_coverage_score", 0.0)) * float(soft["coverage_score"]),
    }
    return judge_obj


def main() -> None:
    load_dotenv_simple(Path(".env"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, default=Path("petroleum_kb/eval/results.jsonl"))
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--modes", type=str, default="naive,qa_oriented")
    ap.add_argument("--timeout-s", type=float, default=240.0)
    ap.add_argument("--retrieval-k", type=int, default=5)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N dataset questions. Useful for quick checks and ablations.",
    )
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--judge-context-max-chars", type=int, default=3000)
    ap.add_argument(
        "--sufficiency-context-max-chars",
        type=int,
        default=20000,
        help="Maximum retrieved-context characters passed to evidence sufficiency judge.",
    )
    ap.add_argument(
        "--evidence-sufficiency-threshold",
        type=float,
        default=0.6,
        help="Pass threshold for evidence_sufficiency_score when --judge is enabled.",
    )
    ap.add_argument("--progress-style", choices=("summary", "per_request"), default="summary")
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--show-eta", action="store_true")
    ap.add_argument("--judge-api-key", type=str, default=None)
    ap.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Evaluator model for answer/context/sufficiency judge. Defaults to LLM_MODEL.",
    )
    ap.add_argument(
        "--retry-errors-from",
        type=Path,
        default=None,
        help="Existing JSONL result file. Only rows with ok=false, non-200 http_status, or error will be rerun and replaced.",
    )
    args = ap.parse_args()

    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    allowed = {"naive", "qa_oriented", "iterative_review"}
    bad = sorted(set(modes) - allowed)
    if bad:
        raise SystemExit(f"Unknown modes: {bad}. Allowed: {sorted(allowed)}")

    dataset_rows = list(iter_jsonl(args.dataset))
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be a positive integer.")
        dataset_rows = dataset_rows[: int(args.limit)]

    query_url = urljoin(args.base_url.rstrip("/") + "/", "query")
    health_url = urljoin(args.base_url.rstrip("/") + "/", "health")

    try:
        hs, _ = get_json(health_url, timeout_s=5.0)
        if hs != 200:
            raise RuntimeError(f"health={hs}")
    except Exception as exc:
        raise SystemExit(
            f"Could not connect to the ChemAnalyst FastAPI service.\n"
            f"- base_url: {args.base_url}\n- health: {health_url}\n- error: {exc}\n"
            f"Start the service with:\n  uvicorn app.main:app --host 127.0.0.1 --port 8000"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json if args.out_json is not None else args.out.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out.with_suffix(".summary.csv")
    eval_model = _resolve_eval_model(args.judge_model)
    sufficiency_model = _resolve_sufficiency_model(args.judge_model)

    total = 0
    ok = 0
    errors = 0
    judge_errors = 0
    latencies: list[float] = []
    judge_latencies: list[float] = []
    records_out: list[dict[str, Any]] = []

    by_mode: dict[str, dict[str, list[float]]] = {
        m: {
            "hit": [],
            "recall": [],
            "precision": [],
            "mrr": [],
            "map": [],
            "corr_rel": [],
            "faithful": [],
            "corr_rel_gated": [],
            "faithful_gated": [],
            "correctness_score": [],
            "faithfulness_score": [],
            "key_point_coverage_score": [],
            "cov2": [],
            "cov2_strict": [],
            "evidence_cov": [],
            "ctx_rel": [],
            "ctx_prec": [],
            "ctx_rec": [],
            "suff_score": [],
            "suff_pass": [],
        }
        for m in modes
    }

    retry_existing_records: list[dict[str, Any]] = []
    retry_keys: set[tuple[str, str]] = set()
    if args.retry_errors_from is not None:
        if not args.retry_errors_from.exists():
            raise SystemExit(f"Retry source does not exist: {args.retry_errors_from}")
        loaded = 0
        for rec in iter_jsonl(args.retry_errors_from):
            loaded += 1
            key = _eval_record_key(rec)
            if not key[0] or not key[1]:
                continue
            if _is_retryable_eval_error(rec, require_judge=bool(args.judge)):
                retry_keys.add(key)
            else:
                retry_existing_records.append(rec)
        print(
            f"[eval] retry mode: loaded={loaded} keep_success={len(retry_existing_records)} "
            f"retry_errors={len(retry_keys)} source={args.retry_errors_from}"
        )
        if not retry_keys:
            print("[eval] retry mode: no retryable timeout/error rows found.")

    expected_total = None
    if args.show_eta:
        if args.retry_errors_from is not None:
            expected_total = len(retry_keys)
        else:
            expected_total = len(dataset_rows) * len(modes)

    def maybe_print():
        if args.progress_every <= 0:
            return
        if total % args.progress_every != 0:
            return
        avg_lat = _mean(latencies) or 0.0
        if expected_total and total > 0:
            eta = max(int((expected_total - total) * avg_lat), 0)
            eta_str = f"{eta//60:02d}m{eta%60:02d}s"
        else:
            eta_str = "n/a"
        print(
            f"[eval] progress: total_requests={total} ok={ok} errors={errors} judge_errors={judge_errors} "
            f"avg_latency_s={avg_lat:.3f} eta={eta_str}"
        )

    with args.out.open("w", encoding="utf-8") as out:
        for old_rec in retry_existing_records:
            out.write(json.dumps(old_rec, ensure_ascii=False) + "\n")
            records_out.append(old_rec)
            dt_total, dt_ok, dt_errors, dt_judge_errors = _accumulate_existing_record_metrics(
                old_rec, by_mode, latencies, judge_latencies
            )
            total += dt_total
            ok += dt_ok
            errors += dt_errors
            judge_errors += dt_judge_errors

        for row in dataset_rows:
            qid = row.get("id")
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            qtype = infer_question_type(row)
            gold_doc_ids = row.get("gold_doc_ids") if isinstance(row.get("gold_doc_ids"), list) else []
            equivalent_doc_groups = _normalize_equivalent_doc_groups(row)
            answer_key = str(row.get("answer_key") or "")
            key_points = [str(x).strip() for x in (row.get("key_points") or []) if str(x).strip()]
            for mode in modes:
                if args.retry_errors_from is not None and (str(qid or "").strip(), mode) not in retry_keys:
                    continue
                rec: dict[str, Any] = {
                    "id": qid,
                    "question": question,
                    "rag_mode": mode,
                    "question_type": qtype,
                    "gold_doc_ids": gold_doc_ids,
                    "equivalent_doc_groups": equivalent_doc_groups,
                    "answer_key": answer_key,
                    "key_points": key_points,
                }
                payload = {"query": question, "task_type": "rag", "rag_mode": mode, "session_id": None}
                t0 = time.perf_counter()
                try:
                    status, resp = post_json(query_url, payload, timeout_s=float(args.timeout_s))
                    dt = time.perf_counter() - t0
                    total += 1
                    latencies.append(dt)
                    rec.update({"ok": status == 200, "http_status": status, "latency_s": dt})
                    if status == 200:
                        ok += 1
                        rec["task_type"] = resp.get("task_type")
                        rec["used_model"] = resp.get("used_model")
                        rec["used_tools"] = resp.get("used_tools")
                        rec["answer"] = resp.get("answer")
                        raw = resp.get("raw_result")
                        if args.save_raw:
                            rec["raw_result"] = raw
                        pred_doc_ids = extract_pred_doc_ids(mode, raw)
                        rec["pred_doc_ids"] = pred_doc_ids
                        rm = compute_retrieval_metrics(pred_doc_ids, gold_doc_ids, int(args.retrieval_k))
                        rec["retrieval_metrics"] = rm
                        if rm:
                            by_mode[mode]["hit"].append(float(rm.get("hit_at_k", 0.0)))
                            by_mode[mode]["recall"].append(float(rm.get("recall_at_k", 0.0)))
                            by_mode[mode]["precision"].append(float(rm.get("precision_at_k", 0.0)))
                            by_mode[mode]["mrr"].append(float(rm.get("mrr", 0.0)))
                            by_mode[mode]["map"].append(float(rm.get("map", 0.0)))
                        if qtype == "two_doc" and isinstance(gold_doc_ids, list) and len(gold_doc_ids) >= 2:
                            cov2_strict = 1.0 if set(gold_doc_ids).issubset(set(pred_doc_ids)) else 0.0
                            soft_cov = _compute_equivalent_coverage(pred_doc_ids, equivalent_doc_groups)
                            cov2 = float(soft_cov.get("coverage_score", 0.0))
                            rec["two_doc_coverage_strict"] = cov2_strict
                            rec["two_doc_coverage"] = cov2
                            rec["equivalent_group_coverage"] = soft_cov
                            by_mode[mode]["cov2_strict"].append(cov2_strict)
                            by_mode[mode]["cov2"].append(cov2)
                            by_mode[mode]["evidence_cov"].append(cov2)
                        else:
                            soft_cov = _compute_equivalent_coverage(pred_doc_ids, equivalent_doc_groups)
                            rec["equivalent_group_coverage"] = soft_cov
                            by_mode[mode]["evidence_cov"].append(float(soft_cov.get("coverage_score", 0.0)))

                        if args.judge and answer_key:
                            jt0 = time.perf_counter()
                            try:
                                j = _deepseek_judge(
                                    question,
                                    str(resp.get("answer") or ""),
                                    answer_key,
                                    key_points=key_points,
                                    question_type=qtype,
                                    api_key_override=args.judge_api_key,
                                    model_override=args.judge_model,
                                )
                                if isinstance(j, dict) and "error" not in j:
                                    j = apply_evidence_gate(
                                        judge_obj=j,
                                        question_type=qtype,
                                        gold_doc_ids=gold_doc_ids,
                                        pred_doc_ids=pred_doc_ids,
                                        equivalent_doc_groups=equivalent_doc_groups,
                                    )
                                jdt = time.perf_counter() - jt0
                                rec["judge"] = j
                                rec["judge_latency_s"] = jdt
                                rec["judge_model"] = eval_model
                                judge_latencies.append(jdt)
                                if isinstance(j, dict) and "error" not in j:
                                    by_mode[mode]["corr_rel"].append(1.0 if j.get("raw_judge_pass", {}).get("correct_and_relevant") is True else 0.0)
                                    by_mode[mode]["faithful"].append(1.0 if j.get("raw_judge_pass", {}).get("faithful") is True else 0.0)
                                    by_mode[mode]["corr_rel_gated"].append(1.0 if j.get("gated_judge_pass", {}).get("correct_and_relevant") is True else 0.0)
                                    by_mode[mode]["faithful_gated"].append(1.0 if j.get("gated_judge_pass", {}).get("faithful") is True else 0.0)
                                    by_mode[mode]["correctness_score"].append(float(j.get("correctness_score", 0.0)))
                                    by_mode[mode]["faithfulness_score"].append(float(j.get("faithfulness_score", 0.0)))
                                    by_mode[mode]["key_point_coverage_score"].append(float(j.get("key_point_coverage_score", 0.0)))
                                else:
                                    judge_errors += 1

                                # Semantic context judge (LLM-based)
                                ctx = _collect_context_text(raw, mode, int(args.judge_context_max_chars))
                                cj = _deepseek_context_judge(
                                    question=question,
                                    answer_key=answer_key,
                                    context_text=ctx,
                                    api_key_override=args.judge_api_key,
                                    model_override=args.judge_model,
                                )
                                rec["context_judge"] = cj
                                rec["context_judge_model"] = eval_model
                                if isinstance(cj, dict) and "error" not in cj:
                                    by_mode[mode]["ctx_rel"].append(float(cj.get("context_relevance", 0.0)))
                                    by_mode[mode]["ctx_prec"].append(float(cj.get("context_precision", 0.0)))
                                    by_mode[mode]["ctx_rec"].append(float(cj.get("context_recall", 0.0)))

                                suff_ctx = _collect_context_text(raw, mode, int(args.sufficiency_context_max_chars))
                                sj = _deepseek_evidence_sufficiency_judge(
                                    question=question,
                                    answer=str(resp.get("answer") or ""),
                                    answer_key=answer_key,
                                    key_points=key_points,
                                    question_type=qtype,
                                    context_text=suff_ctx,
                                    threshold=float(args.evidence_sufficiency_threshold),
                                    api_key_override=args.judge_api_key,
                                    model_override=args.judge_model,
                                )
                                rec["evidence_sufficiency"] = sj
                                rec["evidence_sufficiency_model"] = sufficiency_model
                                rec["evidence_sufficiency_context_chars"] = len(suff_ctx)
                                if isinstance(sj, dict) and "error" not in sj:
                                    by_mode[mode]["suff_score"].append(float(sj.get("evidence_sufficiency_score", 0.0)))
                                    by_mode[mode]["suff_pass"].append(1.0 if sj.get("sufficient") is True else 0.0)
                                else:
                                    judge_errors += 1
                            except Exception as exc:
                                judge_errors += 1
                                rec["judge_error"] = str(exc)
                    else:
                        errors += 1
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    dt = time.perf_counter() - t0
                    total += 1
                    errors += 1
                    latencies.append(dt)
                    rec.update({"ok": False, "latency_s": dt, "error": str(exc)})

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records_out.append(rec)
                maybe_print()

    out_json.write_text(json.dumps(records_out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_modes = ("naive", "qa_oriented", "iterative_review")
    metric_suffixes = [
        "answer_correct_and_relevant_pass",
        "faithful_pass",
        "gated_correct_and_relevant_pass",
        "gated_faithful_pass",
        "correctness_score",
        "faithfulness_score",
        "key_point_coverage_score",
        "hit_at_k",
        "recall_at_k",
        "map",
        "evidence_coverage_score",
        "two_doc_coverage",
        "two_doc_coverage_strict",
        "context_relevance_llm",
        "context_precision_llm",
        "context_recall_llm",
        "evidence_sufficiency_score",
        "evidence_sufficient_pass",
    ]
    summary_fieldnames = [
        "split",
        "dataset",
        "base_url",
        "modes",
        "judge_enabled",
        "save_raw",
        "judge_model",
        "judge_context_max_chars",
        "sufficiency_model",
        "sufficiency_context_max_chars",
        "evidence_sufficiency_threshold",
        "question_count",
        "total",
        "ok",
        "errors",
        "avg_latency_s",
        "judge_errors",
        "retrieval_k",
        "avg_judge_latency_s",
    ]
    for mode_name in summary_modes:
        for suffix in metric_suffixes:
            summary_fieldnames.append(f"{mode_name}_{suffix}")

    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=summary_fieldnames,
        )
        writer.writeheader()
        row = {
            "split": "overall",
            "dataset": str(args.dataset),
            "base_url": str(args.base_url),
            "modes": ",".join(modes),
            "judge_enabled": bool(args.judge),
            "save_raw": bool(args.save_raw),
            "judge_model": eval_model if args.judge else "",
            "judge_context_max_chars": int(args.judge_context_max_chars),
            "sufficiency_model": sufficiency_model if args.judge else "",
            "sufficiency_context_max_chars": int(args.sufficiency_context_max_chars),
            "evidence_sufficiency_threshold": float(args.evidence_sufficiency_threshold),
            "question_count": len(dataset_rows),
            "total": total,
            "ok": ok,
            "errors": errors,
            "avg_latency_s": f"{(_mean(latencies) or 0.0):.4f}",
            "judge_errors": judge_errors,
            "retrieval_k": int(args.retrieval_k),
            "avg_judge_latency_s": f"{(_mean(judge_latencies) or 0.0):.4f}",
        }
        for mode in summary_modes:
            row[f"{mode}_answer_correct_and_relevant_pass"] = (
                f"{(_mean(by_mode.get(mode, {}).get('corr_rel', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_faithful_pass"] = (
                f"{(_mean(by_mode.get(mode, {}).get('faithful', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_gated_correct_and_relevant_pass"] = (
                f"{(_mean(by_mode.get(mode, {}).get('corr_rel_gated', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_gated_faithful_pass"] = (
                f"{(_mean(by_mode.get(mode, {}).get('faithful_gated', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_correctness_score"] = (
                f"{(_mean(by_mode.get(mode, {}).get('correctness_score', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_faithfulness_score"] = (
                f"{(_mean(by_mode.get(mode, {}).get('faithfulness_score', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_key_point_coverage_score"] = (
                f"{(_mean(by_mode.get(mode, {}).get('key_point_coverage_score', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_hit_at_k"] = f"{(_mean(by_mode.get(mode, {}).get('hit', [])) or 0.0):.4f}" if mode in modes else ""
            row[f"{mode}_recall_at_k"] = (
                f"{(_mean(by_mode.get(mode, {}).get('recall', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_map"] = f"{(_mean(by_mode.get(mode, {}).get('map', [])) or 0.0):.4f}" if mode in modes else ""
            row[f"{mode}_evidence_coverage_score"] = (
                f"{(_mean(by_mode.get(mode, {}).get('evidence_cov', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_two_doc_coverage"] = (
                f"{(_mean(by_mode.get(mode, {}).get('cov2', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_two_doc_coverage_strict"] = (
                f"{(_mean(by_mode.get(mode, {}).get('cov2_strict', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_context_relevance_llm"] = (
                f"{(_mean(by_mode.get(mode, {}).get('ctx_rel', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_context_precision_llm"] = (
                f"{(_mean(by_mode.get(mode, {}).get('ctx_prec', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_context_recall_llm"] = (
                f"{(_mean(by_mode.get(mode, {}).get('ctx_rec', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_evidence_sufficiency_score"] = (
                f"{(_mean(by_mode.get(mode, {}).get('suff_score', [])) or 0.0):.4f}" if mode in modes else ""
            )
            row[f"{mode}_evidence_sufficient_pass"] = (
                f"{(_mean(by_mode.get(mode, {}).get('suff_pass', [])) or 0.0):.4f}" if mode in modes else ""
            )
        writer.writerow(row)

    print(f"Wrote results to {args.out}")
    print(f"Wrote results (json) to {out_json}")
    print(f"Wrote summary to {summary_csv}")
    print(f"Total={total} OK={ok} Errors={errors} AvgLatency={(_mean(latencies) or 0.0):.3f}s")


if __name__ == "__main__":
    main()

