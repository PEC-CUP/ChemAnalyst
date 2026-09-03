from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


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
        os.environ.setdefault(key, val.strip().strip("'").strip('"'))


def iter_jsonl(path: Path):
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


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _mode(row: dict[str, Any]) -> str:
    return _as_text(row.get("rag_mode") or row.get("mode") or "unknown").lower()


def _question_type(row: dict[str, Any]) -> str:
    qtype = _as_text(row.get("question_type")).lower()
    if qtype:
        return qtype
    gold = row.get("gold_doc_ids")
    return "two_doc" if isinstance(gold, list) and len(gold) >= 2 else "single"


def _reference_text(row: dict[str, Any], *, include_key_points: bool) -> str:
    parts = [_as_text(row.get("answer_key"))]
    if include_key_points:
        key_points = [str(x).strip() for x in (row.get("key_points") or []) if str(x).strip()]
        if key_points:
            parts.append("Key points:\n" + "\n".join(f"- {point}" for point in key_points))
    return "\n\n".join(part for part in parts if part)


def _unwrap_raw_result(row: dict[str, Any]) -> Any:
    raw = row.get("raw_result")
    if isinstance(raw, dict) and isinstance(raw.get("raw_result"), (dict, list)):
        return raw.get("raw_result")
    return raw


def _extract_contexts_from_evidences(
    evidences: list[Any],
    *,
    max_contexts: int,
    max_context_chars: int,
) -> list[str]:
    contexts: list[str] = []
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        chunks = ev.get("chunks") if isinstance(ev.get("chunks"), list) else []
        if chunks:
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                text = _as_text(ch.get("text"))
                if text:
                    contexts.append(text[:max_context_chars])
                    if len(contexts) >= max_contexts:
                        return contexts
        else:
            text = _as_text(ev.get("text"))
            if text:
                contexts.append(text[:max_context_chars])
                if len(contexts) >= max_contexts:
                    return contexts
    return contexts


def extract_retrieved_contexts(
    row: dict[str, Any],
    *,
    context_source: str,
    max_contexts: int,
    max_context_chars: int,
) -> list[str]:
    raw = _unwrap_raw_result(row)
    if isinstance(raw, list):
        return _extract_contexts_from_evidences(
            raw,
            max_contexts=max_contexts,
            max_context_chars=max_context_chars,
        )
    if not isinstance(raw, dict):
        return []

    source = context_source.strip()
    if source == "auto":
        source = "evidences"

    if source in {"evidences", "filtered_evidences", "synthesis_evidences", "candidate_evidences"}:
        evidences = raw.get(source) if isinstance(raw.get(source), list) else []
        contexts = _extract_contexts_from_evidences(
            evidences,
            max_contexts=max_contexts,
            max_context_chars=max_context_chars,
        )
        if contexts:
            return contexts

    if source == "rounds" or not source:
        evidences: list[Any] = []
        rounds = raw.get("rounds") if isinstance(raw.get("rounds"), list) else []
        for rd in rounds:
            if not isinstance(rd, dict):
                continue
            evs = rd.get("evidences") if isinstance(rd.get("evidences"), list) else []
            evidences.extend(evs)
        return _extract_contexts_from_evidences(
            evidences,
            max_contexts=max_contexts,
            max_context_chars=max_context_chars,
        )

    return []


def _parse_modes(raw: str) -> set[str] | None:
    items = {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}
    return items or None


def _get_nested(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if val == val else None


def _to_pass(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "yes", "pass", "passed", "1"}:
            return 1.0
        if raw in {"false", "no", "fail", "failed", "0"}:
            return 0.0
    return None


def _existing_eval_metrics(row: dict[str, Any]) -> dict[str, Any]:
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    context_judge = row.get("context_judge") if isinstance(row.get("context_judge"), dict) else {}
    raw_pass = judge.get("raw_judge_pass") if isinstance(judge.get("raw_judge_pass"), dict) else {}
    gated_pass = judge.get("gated_judge_pass") if isinstance(judge.get("gated_judge_pass"), dict) else {}
    return {
        "existing_correctness_pass": _to_pass(judge.get("correct")),
        "existing_answer_relevancy_pass": _to_pass(judge.get("relevant")),
        "existing_correct_and_relevant_pass": _to_pass(raw_pass.get("correct_and_relevant")),
        "existing_faithful_pass": _to_pass(raw_pass.get("faithful")),
        "existing_gated_correct_and_relevant_pass": _to_pass(gated_pass.get("correct_and_relevant")),
        "existing_gated_faithful_pass": _to_pass(gated_pass.get("faithful")),
        "existing_correctness_score": _to_float(judge.get("correctness_score")),
        "existing_answer_relevancy_score": _to_float(judge.get("relevance_score")),
        "existing_faithfulness_score": _to_float(judge.get("faithfulness_score")),
        "existing_key_point_coverage_score": _to_float(judge.get("key_point_coverage_score")),
        "existing_context_relevance_score": _to_float(context_judge.get("context_relevance")),
        "existing_context_precision_score": _to_float(context_judge.get("context_precision")),
        "existing_context_recall_score": _to_float(context_judge.get("context_recall")),
    }


def build_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    modes = _parse_modes(args.modes)
    samples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    per_mode_counts: dict[str, int] = defaultdict(int)

    for row in iter_jsonl(args.input):
        mode = _mode(row)
        if modes is not None and mode not in modes:
            skipped["mode_filter"] += 1
            continue
        if args.limit_per_mode and per_mode_counts[mode] >= args.limit_per_mode:
            skipped["limit_per_mode"] += 1
            continue

        question = _as_text(row.get("question"))
        answer = _as_text(row.get("answer"))
        reference = _reference_text(row, include_key_points=not args.no_key_points)
        contexts = extract_retrieved_contexts(
            row,
            context_source=args.context_source,
            max_contexts=args.max_contexts,
            max_context_chars=args.max_context_chars,
        )
        if not question:
            skipped["missing_question"] += 1
            continue
        if not answer:
            skipped["missing_answer"] += 1
            continue
        if not reference:
            skipped["missing_reference"] += 1
            continue
        if not contexts:
            skipped["missing_contexts"] += 1
            continue

        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )
        metadata.append(
            {
                "id": row.get("id"),
                "line_no": row.get("_line_no"),
                "rag_mode": mode,
                "question_type": _question_type(row),
                "context_count": len(contexts),
                "context_source": args.context_source,
                **_existing_eval_metrics(row),
            }
        )
        per_mode_counts[mode] += 1
        if args.limit and len(samples) >= args.limit:
            break

    return samples, metadata, dict(skipped)


def _import_ragas():
    try:
        from ragas import EvaluationDataset, SingleTurnSample  # type: ignore
        from ragas.metrics import Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "Missing or incompatible RAGAS dependencies. Install optional dependencies first:\n"
            "  pip install -r petroleum_kb/eval/requirements-ragas.txt\n"
            f"Original import error: {exc}"
        ) from exc

    try:
        from ragas.metrics import ResponseRelevancy  # type: ignore
    except Exception:
        ResponseRelevancy = None  # type: ignore
    return EvaluationDataset, SingleTurnSample, Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference, ResponseRelevancy


def _prime_openai_platform(langchain_client: Any) -> None:
    """Avoid OpenAI 2.x lazy platform detection inside async worker contexts."""
    try:
        from openai._base_client import get_platform  # type: ignore
    except Exception:
        return

    platform_value = get_platform()
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(obj, "_platform"):
            try:
                if getattr(obj, "_platform", None) is None:
                    setattr(obj, "_platform", platform_value)
            except Exception:
                pass

        for attr in ("root_client", "root_async_client", "client", "async_client", "_client"):
            try:
                visit(getattr(obj, attr, None))
            except Exception:
                pass

    visit(langchain_client)


def _make_sync_backed_chat_openai():
    from langchain_openai import ChatOpenAI  # type: ignore

    class SyncBackedChatOpenAI(ChatOpenAI):
        async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
            return await asyncio.to_thread(
                self.generate_prompt,
                prompts,
                stop=stop,
                callbacks=callbacks,
                **kwargs,
            )

    return SyncBackedChatOpenAI


def _make_sync_backed_openai_embeddings():
    from langchain_openai import OpenAIEmbeddings  # type: ignore

    class SyncBackedOpenAIEmbeddings(OpenAIEmbeddings):
        async def aembed_documents(self, texts, chunk_size=None, **kwargs):
            return await asyncio.to_thread(
                self.embed_documents,
                texts,
                chunk_size=chunk_size,
                **kwargs,
            )

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

    return SyncBackedOpenAIEmbeddings


def _build_llm(args: argparse.Namespace):
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
    model = (
        args.model
        
        
        
        or os.getenv("LLM_MODEL") 
        or os.getenv("OPENAI_MODEL")
        or ""
    )
    if not api_key:
        raise SystemExit("Missing API key. Set LLM_API_KEY, or pass --api-key.")

    try:
        from ragas.llms import LangchainLLMWrapper  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "Missing RAGAS LangChain adapter dependencies. Install:\n"
            "  pip install -r petroleum_kb/eval/requirements-ragas.txt\n"
            f"Original import error: {exc}"
        ) from exc

    ChatOpenAI = _make_sync_backed_chat_openai()
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.0,
        timeout=args.timeout_s,
    )
    _prime_openai_platform(llm)
    return LangchainLLMWrapper(llm)


def _build_embeddings(args: argparse.Namespace):
    if not args.embedding_model:
        return None
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise SystemExit(
            "Missing RAGAS embedding adapter dependencies. Install:\n"
            "  pip install -r petroleum_kb/eval/requirements-ragas.txt\n"
            f"Original import error: {exc}"
        ) from exc
    OpenAIEmbeddings = _make_sync_backed_openai_embeddings()
    embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url, model=args.embedding_model)
    _prime_openai_platform(embeddings)
    return LangchainEmbeddingsWrapper(embeddings)


def _metric_names(raw: str) -> list[str]:
    names = [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]
    return names or ["context_precision", "context_recall", "faithfulness"]


def _canonical_metric_name(name: str) -> str:
    raw = str(name)
    key = raw.strip().lower()
    if "context_precision" in key:
        return "ragas_context_precision_score"
    if "context_recall" in key:
        return "ragas_context_recall_score"
    if "faithfulness" in key:
        return "ragas_faithfulness_score"
    if "answer_relev" in key or "response_relev" in key:
        return "ragas_answer_relevancy_score"
    return raw


def run_ragas(samples: list[dict[str, Any]], args: argparse.Namespace):
    (
        EvaluationDataset,
        SingleTurnSample,
        Faithfulness,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    ) = _import_ragas()
    llm = _build_llm(args)
    embeddings = _build_embeddings(args)

    metric_objects: list[tuple[str, Any]] = []
    for name in _metric_names(args.metrics):
        if name in {"context_precision", "context_precision_with_reference"}:
            metric_objects.append(("ragas_context_precision_score", LLMContextPrecisionWithReference(llm=llm)))
        elif name == "context_recall":
            metric_objects.append(("ragas_context_recall_score", LLMContextRecall(llm=llm)))
        elif name == "faithfulness":
            metric_objects.append(("ragas_faithfulness_score", Faithfulness(llm=llm)))
        elif name in {"answer_relevancy", "answer_relevance", "response_relevancy"}:
            if ResponseRelevancy is None:
                raise SystemExit("This RAGAS version does not expose ResponseRelevancy.")
            if embeddings is None:
                raise SystemExit(
                    "answer_relevancy/response_relevancy requires embeddings in RAGAS. "
                    "Pass --embedding-model with an OpenAI-compatible embedding model, or omit this metric."
                )
            metric_objects.append(("ragas_answer_relevancy_score", ResponseRelevancy(llm=llm, embeddings=embeddings)))
        else:
            raise SystemExit(f"Unsupported metric: {name}")

    # Keep this construction as an explicit schema check against official RAGAS
    # single-turn samples. Scoring is done manually because ragas.evaluate()
    # currently fails on Python 3.14 in its synchronous executor path.
    EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=item["user_input"],
                response=item["response"],
                retrieved_contexts=item["retrieved_contexts"],
                reference=item["reference"],
            )
            for item in samples
        ]
    )

    async def _score_all():
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(samples, start=1):
            sample = SingleTurnSample(
                user_input=item["user_input"],
                response=item["response"],
                retrieved_contexts=item["retrieved_contexts"],
                reference=item["reference"],
            )
            row = {
                "user_input": item["user_input"],
                "response": item["response"],
                "retrieved_contexts": item["retrieved_contexts"],
                "reference": item["reference"],
            }
            for metric_name, metric in metric_objects:
                try:
                    if args.use_ragas_timeout_wrapper:
                        row[metric_name] = await metric.single_turn_ascore(
                            sample,
                            timeout=None if args.metric_timeout_s <= 0 else float(args.metric_timeout_s),
                        )
                    else:
                        # RAGAS 0.3.x wraps single_turn_ascore() in
                        # asyncio.wait_for(). On Python 3.14 this can fail with
                        # "Timeout should be used inside a task". Calling the
                        # metric implementation directly preserves the official
                        # metric logic while bypassing only that wrapper.
                        row[metric_name] = await metric._single_turn_ascore(sample, callbacks=None)
                except Exception:
                    if not args.allow_ragas_nan:
                        raise
                    row[metric_name] = None
            rows.append(row)
            if args.progress_every and idx % int(args.progress_every) == 0:
                print(f"[ragas] scored {idx}/{len(samples)}")
        return rows

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Missing pandas dependency: {exc}") from exc
    try:
        import sniffio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing sniffio dependency required by the OpenAI/LangChain async stack. Install:\n"
            "  pip install -r petroleum_kb/eval/requirements-ragas.txt\n"
            f"Original import error: {exc}"
        ) from exc

    token = sniffio.current_async_library_cvar.set("asyncio")
    try:
        return pd.DataFrame(asyncio.run(_score_all()))
    finally:
        sniffio.current_async_library_cvar.reset(token)


def _build_ragas_metrics(args: argparse.Namespace):
    (
        EvaluationDataset,
        SingleTurnSample,
        Faithfulness,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    ) = _import_ragas()
    llm = _build_llm(args)
    embeddings = _build_embeddings(args)

    metric_objects: list[tuple[str, Any]] = []
    for name in _metric_names(args.metrics):
        if name in {"context_precision", "context_precision_with_reference"}:
            metric_objects.append(("ragas_context_precision_score", LLMContextPrecisionWithReference(llm=llm)))
        elif name == "context_recall":
            metric_objects.append(("ragas_context_recall_score", LLMContextRecall(llm=llm)))
        elif name == "faithfulness":
            metric_objects.append(("ragas_faithfulness_score", Faithfulness(llm=llm)))
        elif name in {"answer_relevancy", "answer_relevance", "response_relevancy"}:
            if ResponseRelevancy is None:
                raise SystemExit("This RAGAS version does not expose ResponseRelevancy.")
            if embeddings is None:
                raise SystemExit(
                    "answer_relevancy/response_relevancy requires embeddings in RAGAS. "
                    "Pass --embedding-model with an OpenAI-compatible embedding model, or omit this metric."
                )
            metric_objects.append(("ragas_answer_relevancy_score", ResponseRelevancy(llm=llm, embeddings=embeddings)))
        else:
            raise SystemExit(f"Unsupported metric: {name}")

    return EvaluationDataset, SingleTurnSample, metric_objects


def _record_key(rec: dict[str, Any]) -> str:
    return "|".join(
        [
            str(rec.get("id") or ""),
            str(rec.get("line_no") or ""),
            str(rec.get("rag_mode") or ""),
            str(rec.get("question_type") or ""),
            str(rec.get("context_source") or ""),
        ]
    )


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _fill_rag_quality_columns(rec: dict[str, Any], args: argparse.Namespace) -> None:
    rec["rag_quality_correctness_pass_rate"] = rec.get("existing_correctness_pass")
    rec["rag_quality_answer_relevancy_pass_rate"] = rec.get("existing_answer_relevancy_pass")
    rec["rag_quality_correct_and_relevant_pass_rate"] = rec.get("existing_correct_and_relevant_pass")
    rec["rag_quality_faithful_pass_rate"] = rec.get("existing_faithful_pass")
    rec["rag_quality_context_precision_score"] = (
        rec.get("ragas_context_precision_score")
        if rec.get("ragas_context_precision_score") is not None
        else rec.get("existing_context_precision_score")
    )
    rec["rag_quality_context_recall_score"] = (
        rec.get("ragas_context_recall_score")
        if rec.get("ragas_context_recall_score") is not None
        else rec.get("existing_context_recall_score")
    )
    rec["rag_quality_faithfulness_score"] = (
        rec.get("ragas_faithfulness_score")
        if rec.get("ragas_faithfulness_score") is not None
        else rec.get("existing_faithfulness_score")
    )
    rec["rag_quality_context_relevance_score"] = rec.get("existing_context_relevance_score")
    rec["rag_quality_answer_relevancy_score"] = (
        rec.get("ragas_answer_relevancy_score")
        if rec.get("ragas_answer_relevancy_score") is not None
        else rec.get("existing_answer_relevancy_score")
    )
    if rec.get("ragas_answer_relevancy_score") is not None:
        score = _to_float(rec.get("ragas_answer_relevancy_score"))
        rec["ragas_answer_relevancy_pass_rate"] = 1.0 if score is not None and score >= float(args.pass_threshold) else 0.0


def write_summary_from_records(records: list[dict[str, Any]], args: argparse.Namespace, metric_cols: list[str]) -> None:
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    water_cols = [
        "rag_quality_correctness_pass_rate",
        "rag_quality_answer_relevancy_pass_rate",
        "rag_quality_correct_and_relevant_pass_rate",
        "rag_quality_faithful_pass_rate",
        "rag_quality_context_precision_score",
        "rag_quality_context_recall_score",
        "rag_quality_faithfulness_score",
        "rag_quality_context_relevance_score",
        "rag_quality_answer_relevancy_score",
    ]
    existing_cols = [
        "existing_correctness_pass",
        "existing_answer_relevancy_pass",
        "existing_correct_and_relevant_pass",
        "existing_faithful_pass",
        "existing_correctness_score",
        "existing_answer_relevancy_score",
        "existing_faithfulness_score",
        "existing_context_relevance_score",
        "existing_context_precision_score",
        "existing_context_recall_score",
    ]
    output_metric_cols = list(dict.fromkeys([*water_cols, *metric_cols, *existing_cols]))
    if any("ragas_answer_relevancy_pass_rate" in rec for rec in records):
        output_metric_cols.append("ragas_answer_relevancy_pass_rate")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if str(rec.get("_ragas_error") or ""):
            continue
        grouped[(str(rec.get("question_type") or ""), str(rec.get("rag_mode") or ""))].append(rec)

    fieldnames = ["question_type", "rag_mode", "n", *output_metric_cols]
    with args.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (qtype, mode), rows in sorted(grouped.items()):
            out = {"question_type": qtype, "rag_mode": mode, "n": len(rows)}
            for col in output_metric_cols:
                vals = []
                for rec in rows:
                    val = _to_float(rec.get(col))
                    if val is not None:
                        vals.append(val)
                out[col] = f"{(sum(vals) / len(vals)):.4f}" if vals else ""
            writer.writerow(out)


def run_ragas_streaming(samples: list[dict[str, Any]], metadata: list[dict[str, Any]], args: argparse.Namespace) -> None:
    EvaluationDataset, SingleTurnSample, metric_objects = _build_ragas_metrics(args)
    EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=item["user_input"],
                response=item["response"],
                retrieved_contexts=item["retrieved_contexts"],
                reference=item["reference"],
            )
            for item in samples
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing_records = _load_existing_records(args.out) if args.resume else []
    completed_keys = {_record_key(rec) for rec in existing_records if not str(rec.get("_ragas_error") or "")}
    metric_cols = [name for name, _ in metric_objects]

    async def _score_one(item: dict[str, Any]) -> dict[str, Any]:
        sample = SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"],
            reference=item["reference"],
        )
        out = {
            "user_input": item["user_input"],
            "response": item["response"],
            "retrieved_contexts": item["retrieved_contexts"],
            "reference": item["reference"],
        }
        for metric_name, metric in metric_objects:
            try:
                if args.use_ragas_timeout_wrapper:
                    out[metric_name] = await metric.single_turn_ascore(
                        sample,
                        timeout=None if args.metric_timeout_s <= 0 else float(args.metric_timeout_s),
                    )
                else:
                    out[metric_name] = await metric._single_turn_ascore(sample, callbacks=None)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                out[metric_name] = None
                out.setdefault("_ragas_metric_errors", {})[metric_name] = str(exc)
        return out

    async def _score_remaining() -> list[dict[str, Any]]:
        records = list(existing_records)
        mode = "a" if args.resume and args.out.exists() else "w"
        with args.out.open(mode, encoding="utf-8") as fh:
            for idx, (item, meta) in enumerate(zip(samples, metadata), start=1):
                base = dict(meta)
                key = _record_key(base)
                if key in completed_keys:
                    if args.progress_every and idx % int(args.progress_every) == 0:
                        print(f"[ragas] skipped existing {idx}/{len(samples)}")
                    continue
                try:
                    row = await _score_one(item)
                    rec = dict(base)
                    for col in metric_cols:
                        rec[col] = row.get(col)
                    if row.get("_ragas_metric_errors"):
                        rec["_ragas_metric_errors"] = row.get("_ragas_metric_errors")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rec = dict(base)
                    rec["_ragas_error"] = str(exc)
                    for col in metric_cols:
                        rec[col] = None
                _fill_rag_quality_columns(rec, args)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(rec)
                if args.progress_every and idx % int(args.progress_every) == 0:
                    print(f"[ragas] scored {idx}/{len(samples)}")
        return records

    try:
        import sniffio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing sniffio dependency required by the OpenAI/LangChain async stack. Install:\n"
            "  pip install -r petroleum_kb/eval/requirements-ragas.txt\n"
            f"Original import error: {exc}"
        ) from exc

    token = sniffio.current_async_library_cvar.set("asyncio")
    try:
        records = asyncio.run(_score_remaining())
    finally:
        sniffio.current_async_library_cvar.reset(token)

    if not args.allow_ragas_nan:
        valid = sum(1 for rec in records for col in metric_cols if _to_float(rec.get(col)) is not None)
        if valid == 0:
            raise SystemExit("RAGAS evaluation produced no valid metric values.")
    write_summary_from_records(records, args, metric_cols)


def write_outputs(df, metadata: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    raw_metric_cols = [
        col
        for col in list(df.columns)
        if col
        not in {
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        }
    ]
    metric_cols: list[str] = []
    rename_map: dict[str, str] = {}
    seen_cols: set[str] = set()
    for col in raw_metric_cols:
        canonical = _canonical_metric_name(str(col))
        safe = canonical
        if safe in seen_cols:
            safe = str(col)
        rename_map[str(col)] = safe
        seen_cols.add(safe)
        metric_cols.append(safe)

    water_cols = [
        "rag_quality_correctness_pass_rate",
        "rag_quality_answer_relevancy_pass_rate",
        "rag_quality_correct_and_relevant_pass_rate",
        "rag_quality_faithful_pass_rate",
        "rag_quality_context_precision_score",
        "rag_quality_context_recall_score",
        "rag_quality_faithfulness_score",
        "rag_quality_context_relevance_score",
        "rag_quality_answer_relevancy_score",
    ]

    for idx, row in df.iterrows():
        meta = metadata[int(idx)]
        rec = dict(meta)
        for col in raw_metric_cols:
            value = row[col]
            try:
                if value != value:  # NaN
                    value = None
            except Exception:
                pass
            rec[rename_map[str(col)]] = value

        rec["rag_quality_correctness_pass_rate"] = rec.get("existing_correctness_pass")
        rec["rag_quality_answer_relevancy_pass_rate"] = rec.get("existing_answer_relevancy_pass")
        rec["rag_quality_correct_and_relevant_pass_rate"] = rec.get("existing_correct_and_relevant_pass")
        rec["rag_quality_faithful_pass_rate"] = rec.get("existing_faithful_pass")
        rec["rag_quality_context_precision_score"] = (
            rec.get("ragas_context_precision_score")
            if rec.get("ragas_context_precision_score") is not None
            else rec.get("existing_context_precision_score")
        )
        rec["rag_quality_context_recall_score"] = (
            rec.get("ragas_context_recall_score")
            if rec.get("ragas_context_recall_score") is not None
            else rec.get("existing_context_recall_score")
        )
        rec["rag_quality_faithfulness_score"] = (
            rec.get("ragas_faithfulness_score")
            if rec.get("ragas_faithfulness_score") is not None
            else rec.get("existing_faithfulness_score")
        )
        rec["rag_quality_context_relevance_score"] = rec.get("existing_context_relevance_score")
        rec["rag_quality_answer_relevancy_score"] = (
            rec.get("ragas_answer_relevancy_score")
            if rec.get("ragas_answer_relevancy_score") is not None
            else rec.get("existing_answer_relevancy_score")
        )
        if rec.get("ragas_answer_relevancy_score") is not None:
            score = _to_float(rec.get("ragas_answer_relevancy_score"))
            rec["ragas_answer_relevancy_pass_rate"] = (
                1.0 if score is not None and score >= float(args.pass_threshold) else 0.0
            )
        records.append(rec)

    ragas_cols = [col for col in metric_cols if col.startswith("ragas_")]
    if ragas_cols and not args.allow_ragas_nan:
        valid = 0
        for rec in records:
            for col in ragas_cols:
                if _to_float(rec.get(col)) is not None:
                    valid += 1
        if valid == 0:
            raise SystemExit(
                "RAGAS evaluation produced no valid metric values. "
                "Re-run with a tiny --limit-per-mode and inspect the raised RAGAS/LLM error, "
                "or pass --allow-ragas-nan only for debugging."
            )

    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec.get("question_type") or ""), str(rec.get("rag_mode") or ""))].append(rec)

    existing_cols = [
        "existing_correctness_pass",
        "existing_answer_relevancy_pass",
        "existing_correct_and_relevant_pass",
        "existing_faithful_pass",
        "existing_correctness_score",
        "existing_answer_relevancy_score",
        "existing_faithfulness_score",
        "existing_context_relevance_score",
        "existing_context_precision_score",
        "existing_context_recall_score",
    ]
    output_metric_cols = [*water_cols, *metric_cols, *existing_cols]
    if any("ragas_answer_relevancy_pass_rate" in rec for rec in records):
        output_metric_cols.append("ragas_answer_relevancy_pass_rate")
    # Preserve order while removing duplicates.
    output_metric_cols = list(dict.fromkeys(output_metric_cols))

    fieldnames = ["question_type", "rag_mode", "n", *output_metric_cols]
    with args.summary.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (qtype, mode), rows in sorted(grouped.items()):
            out = {"question_type": qtype, "rag_mode": mode, "n": len(rows)}
            for col in output_metric_cols:
                vals = []
                for rec in rows:
                    try:
                        val = float(rec.get(col))
                    except Exception:
                        continue
                    if val == val:
                        vals.append(val)
                out[col] = f"{(sum(vals) / len(vals)):.4f}" if vals else ""
            writer.writerow(out)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv_simple(repo_root / ".env")

    ap = argparse.ArgumentParser(description="Offline official RAGAS evaluation for saved run_eval JSONL results.")
    ap.add_argument("--input", type=Path, required=True, help="run_eval.py JSONL output.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL with per-row RAGAS metrics.")
    ap.add_argument("--summary", type=Path, required=True, help="Output CSV summary grouped by question_type and rag_mode.")
    ap.add_argument("--modes", type=str, default="", help="Comma-separated rag modes to include.")
    ap.add_argument("--metrics", type=str, default="context_precision,context_recall,faithfulness", help="Comma-separated RAGAS metrics.")
    ap.add_argument("--context-source", choices=("auto", "evidences", "filtered_evidences", "synthesis_evidences", "candidate_evidences", "rounds"), default="evidences")
    ap.add_argument("--max-contexts", type=int, default=8)
    ap.add_argument("--max-context-chars", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0, help="Maximum total rows for smoke testing.")
    ap.add_argument("--limit-per-mode", type=int, default=0, help="Maximum rows per mode for balanced smoke testing.")
    ap.add_argument("--no-key-points", action="store_true", help="Use answer_key only as the RAGAS reference.")
    ap.add_argument("--api-key", type=str, default="")
    ap.add_argument("--base-url", type=str, default="")
    ap.add_argument(
        "--model",
        type=str,
        default="",
        help="Evaluator LLM model. Defaults to LLM_MODEL.",
    )
    ap.add_argument("--embedding-model", type=str, default="", help="Required only for answer_relevancy/response_relevancy.")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--pass-threshold", type=float, default=0.8, help="Threshold for optional RAGAS score-derived pass rates.")
    ap.add_argument("--metric-timeout-s", type=float, default=0.0, help="Per-metric RAGAS timeout. Use 0 to disable.")
    ap.add_argument("--progress-every", type=int, default=0, help="Print progress every N scored samples.")
    ap.add_argument("--batch-size", type=int, default=0, help="Reserved for compatibility; manual RAGAS scoring is sequential.")
    ap.add_argument("--use-ragas-timeout-wrapper", action="store_true", help="Use RAGAS single_turn_ascore() wrapper. Not recommended on Python 3.14.")
    ap.add_argument("--allow-ragas-nan", action="store_true", help="Allow writing outputs even if RAGAS metrics are NaN.")
    ap.add_argument("--stream-write", action="store_true", help="Write each scored row immediately to JSONL and write summary at the end.")
    ap.add_argument("--resume", action="store_true", help="When --stream-write is enabled, skip rows already present in --out.")
    ap.add_argument("--continue-on-error", action="store_true", help="When --stream-write is enabled, keep scoring after per-row metric errors.")
    args = ap.parse_args()

    samples, metadata, skipped = build_samples(args)
    if not samples:
        raise SystemExit(f"No valid samples built. Skipped: {skipped}")
    print(f"[ragas] samples={len(samples)} skipped={skipped}")
    print(f"[ragas] metrics={_metric_names(args.metrics)} context_source={args.context_source}")
    if args.stream_write:
        run_ragas_streaming(samples, metadata, args)
    else:
        df = run_ragas(samples, args)
        write_outputs(df, metadata, args)
    print(f"Wrote RAGAS rows to {args.out}")
    print(f"Wrote RAGAS summary to {args.summary}")


if __name__ == "__main__":
    main()
