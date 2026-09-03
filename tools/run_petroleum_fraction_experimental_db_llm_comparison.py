from __future__ import annotations

import os
import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.llm.unified_llm_client import UnifiedLLMClient
from app.tools.experimental_db.sample_database import SampleDatabase
from app.tools.experimental_db.petroleum_fraction_spectral_evidence import build_petroleum_fraction_spectral_workbook_evidence


def _default_test_dir() -> Path:
    preferred = Path(os.getenv("PETRO_FRACTION_TEST_DIR", "data/experimental_db/petroleum_fraction_dataset/test"))
    if preferred.exists():
        return preferred
    legacy_name = "v" + "go-database"
    legacy = Path(os.getenv("PETRO_FRACTION_LEGACY_ROOT", "data/private")) / legacy_name / "test"
    return legacy if legacy.exists() else preferred


DEFAULT_TEST_DIR = _default_test_dir()
DEFAULT_OUTPUT_ROOT = Path("outputs") / "petroleum_fraction_experimental_db" / "llm_runs"
DEFAULT_QUERY = "Use IR and GC-FID to infer density, distillation distribution, and saturates for this petroleum fraction sample."
DEFAULT_TOPK_SET = (2, 4, 6, 10)
DEFAULT_TARGETS = ("density_20c", "BP_20", "BP_35", "BP_50", "BP_65", "BP_80", "saturates")
SAMPLE_DB_WORKSPACE = ROOT / "data" / "experimental_db"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    template_definition = SampleDatabase(SAMPLE_DB_WORKSPACE).get_template_definition()
    workbooks = sorted(Path(args.test_dir).glob("*.xlsx"))
    if args.limit:
        workbooks = workbooks[: args.limit]
    if not workbooks:
        raise FileNotFoundError(f"No test workbooks found under {args.test_dir}")

    db_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    evidence_jsonl = out_dir / "evidence_bundles.jsonl"
    llm_checkpoint_jsonl = out_dir / "llm_results_checkpoint.jsonl"
    llm_cache = load_llm_checkpoint_cache(llm_checkpoint_jsonl) if args.resume else {}

    experiments = build_experiment_plan(args)
    evidence_mode = "a" if args.resume and evidence_jsonl.exists() else "w"
    with evidence_jsonl.open(evidence_mode, encoding="utf-8") as handle:
        for exp_index, experiment in enumerate(experiments, start=1):
            top_k = int(experiment["top_k"])
            model_label = experiment["model_label"]
            print(f"[experiment {exp_index}/{len(experiments)}] {model_label}, top_k={top_k}", flush=True)
            llm = build_llm(experiment) if not args.skip_llm else None
            for index, workbook in enumerate(workbooks, start=1):
                evidence = build_petroleum_fraction_spectral_workbook_evidence(
                    workbook=workbook,
                    template_definition=template_definition,
                    query=args.query,
                    top_k=top_k,
                )
                handle.write(json.dumps({"experiment": experiment, "workbook": str(workbook), "evidence": evidence}, ensure_ascii=False) + "\n")
                summary = summarize_evidence(evidence, experiment, workbook)
                evidence_rows.append(summary)
                db_rows.extend(db_prediction_rows(evidence, experiment, workbook))
                if llm is not None and evidence.get("status") == "success":
                    print(f"  [{index:02d}/{len(workbooks):02d}] {workbook.name}", flush=True)
                    cache_key = (
                        str(experiment["provider"]),
                        str(experiment["model"]),
                        bool(experiment.get("thinking")),
                        int(experiment["top_k"]),
                        int(experiment.get("replicate") or 1),
                        workbook.name,
                    )
                    if cache_key in llm_cache:
                        llm_result = dict(llm_cache[cache_key])
                        llm_result["reused_from_cache"] = True
                        should_checkpoint = False
                    else:
                        llm_result = call_llm_adjusted_estimates(evidence, experiment, llm)
                        if llm_result.get("status") == "success":
                            llm_cache[cache_key] = dict(llm_result)
                        should_checkpoint = True
                    prediction_rows = llm_prediction_rows(evidence, experiment, workbook, llm_result)
                    llm_rows.extend(prediction_rows)
                    if should_checkpoint:
                        append_llm_checkpoint(
                            llm_checkpoint_jsonl,
                            cache_key=cache_key,
                            experiment=experiment,
                            workbook=workbook,
                            evidence=evidence,
                            llm_result=llm_result,
                            prediction_rows=prediction_rows,
                        )

    db_predictions = pd.DataFrame(db_rows)
    llm_predictions = pd.DataFrame(llm_rows)
    evidence_summary = pd.DataFrame(evidence_rows)
    db_metrics = metric_table(db_predictions)
    llm_metrics = metric_table(llm_predictions)
    comparison_summary = aggregate_experiment_metrics(db_metrics, llm_metrics)
    repeatability_summary = aggregate_repeatability_metrics(comparison_summary)

    evidence_summary.to_csv(out_dir / "evidence_summary.csv", index=False, encoding="utf-8-sig")
    db_predictions.to_csv(out_dir / "db_evidence_center_predictions.csv", index=False, encoding="utf-8-sig")
    db_metrics.to_csv(out_dir / "db_evidence_center_metrics.csv", index=False, encoding="utf-8-sig")
    llm_predictions.to_csv(out_dir / "llm_adjusted_predictions.csv", index=False, encoding="utf-8-sig")
    llm_metrics.to_csv(out_dir / "llm_adjusted_metrics.csv", index=False, encoding="utf-8-sig")
    comparison_summary.to_csv(out_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    repeatability_summary.to_csv(out_dir / "repeatability_summary.csv", index=False, encoding="utf-8-sig")
    write_repeatability_errorbar_svg(repeatability_summary, out_dir / "repeatability_mean_mape_errorbars.svg")

    summary = {
        "schema_version": "petroleum_fraction-experimental-db-llm-comparison.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_dir": str(args.test_dir),
        "output_dir": str(out_dir),
        "query": args.query,
        "workbook_count": len(workbooks),
        "experiments": experiments,
        "replicates": args.replicates,
        "replicate_start": args.replicate_start,
        "llm_enabled": not args.skip_llm,
        "numeric_baseline": "DB evidence center from inverse-distance weighted interpolation over top-k historical neighbors",
        "llm_adjusted_estimate": "Evidence-constrained expert estimate parsed from LLM JSON; validation actuals are excluded from the LLM prompt.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir, summary, db_metrics, llm_metrics, comparison_summary, repeatability_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run petroleum fraction experimental DB top-k and LLM-model comparison experiments.")
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--topk", nargs="*", type=int, default=list(DEFAULT_TOPK_SET))
    parser.add_argument("--skip-llm", action="store_true", help="Only compute deterministic DB evidence-center metrics.")
    parser.add_argument("--skip-model-comparison", action="store_true", help="Only run top-k sensitivity experiments.")
    parser.add_argument("--resume", action="store_true", help="Reuse llm_results_checkpoint.jsonl in --output-dir and skip completed LLM calls.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of test workbooks for smoke tests.")
    parser.add_argument("--replicates", type=int, default=1, help="Number of independent LLM repeats per model/top-k condition.")
    parser.add_argument("--replicate-start", type=int, default=1, help="First replicate index written into outputs.")
    parser.add_argument(
        "--topk-models",
        nargs="*",
        default=["gpt5.5"],
        choices=["gpt5.5", "gpt4.1", "deepseek"],
        help="LLM labels used for top-k sensitivity. Defaults to GPT-5.5; use gpt4.1/deepseek when GPT-5.5 structured JSON is unstable.",
    )
    parser.add_argument("--openai-gpt55-model", default="gpt-5.5")
    parser.add_argument("--openai-gpt41-model", default="gpt-4.1")
    parser.add_argument("--deepseek-label", default="deepseek_v4_reasoning")
    parser.add_argument("--deepseek-model", default="")
    return parser.parse_args()


def build_experiment_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for repeat_offset in range(max(1, int(args.replicates))):
        replicate = int(args.replicate_start) + repeat_offset
        for label in args.topk_models:
            for top_k in args.topk:
                experiments.append(experiment_for_label(args, label, "topk_sensitivity", int(top_k), replicate))
        if not args.skip_model_comparison:
            experiments.extend(
                [
                    experiment_for_label(args, "deepseek", "model_comparison", 10, replicate),
                    experiment_for_label(args, "gpt5.5", "model_comparison", 10, replicate),
                    experiment_for_label(args, "gpt4.1", "model_comparison", 10, replicate),
                ]
            )
    # Preserve repeated gpt5.5/top_k=10 rows because they belong to two experiment groups.
    return experiments


def load_llm_checkpoint_cache(path: Path) -> dict[tuple[str, str, bool, int, int, str], dict[str, Any]]:
    cache: dict[tuple[str, str, bool, int, int, str], dict[str, Any]] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        key_items = item.get("cache_key")
        if not isinstance(key_items, list | tuple) or len(key_items) not in (5, 6):
            continue
        if len(key_items) == 5:
            key = (
                str(key_items[0]),
                str(key_items[1]),
                bool(key_items[2]),
                int(key_items[3]),
                1,
                str(key_items[4]),
            )
        else:
            key = (
                str(key_items[0]),
                str(key_items[1]),
                bool(key_items[2]),
                int(key_items[3]),
                int(key_items[4]),
                str(key_items[5]),
            )
        result = item.get("llm_result")
        if isinstance(result, dict) and result.get("status") == "success":
            cache[key] = result
    return cache


def append_llm_checkpoint(
    path: Path,
    *,
    cache_key: tuple[str, str, bool, int, int, str],
    experiment: dict[str, Any],
    workbook: Path,
    evidence: dict[str, Any],
    llm_result: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_key": list(cache_key),
        "experiment": experiment,
        "workbook": str(workbook),
        "sample_id": evidence.get("sample_id"),
        "llm_result": llm_result,
        "prediction_rows": prediction_rows,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def experiment_for_label(args: argparse.Namespace, label: str, experiment_type: str, top_k: int, replicate: int = 1) -> dict[str, Any]:
    if label == "deepseek":
        return {
            "experiment_type": experiment_type,
            "model_label": args.deepseek_label,
            "provider": "deepseek",
            "model": args.deepseek_model or get_settings().active_llm_model,
            "thinking": True,
            "top_k": top_k,
            "replicate": replicate,
        }
    if label == "gpt4.1":
        return {
            "experiment_type": experiment_type,
            "model_label": "gpt4.1",
            "provider": "openai",
            "model": args.openai_gpt41_model,
            "thinking": False,
            "top_k": top_k,
            "replicate": replicate,
        }
    return {
        "experiment_type": experiment_type,
        "model_label": "gpt5.5",
        "provider": "openai",
        "model": args.openai_gpt55_model,
        "thinking": False,
        "top_k": top_k,
        "replicate": replicate,
    }


def build_llm(experiment: dict[str, Any]) -> Any:
    return UnifiedLLMClient()


def call_llm_adjusted_estimates(evidence: dict[str, Any], experiment: dict[str, Any], llm: Any) -> dict[str, Any]:
    prompt = build_adjustment_prompt(evidence, experiment)
    system_prompt = (
        "You are a petroleum-chemistry expert performing evidence-constrained property inference. "
        "Use only the supplied historical Experimental DB evidence. Return valid JSON only."
    )
    selected_model = str(experiment["model"]).lower()
    max_tokens = 6000 if experiment["provider"] == "openai" and selected_model.startswith("gpt-5") else 3000
    try:
        answer = llm.chat_with_model(
            model=experiment["model"],
            query=prompt,
            system_prompt=system_prompt,
            thinking=bool(experiment.get("thinking")),
            temperature=None if experiment["provider"] == "openai" else 0.1,
            max_tokens=max_tokens,
        )
        parsed = parse_json_answer(answer)
        return {
            "status": "success",
            "provider": experiment["provider"],
            "model": experiment["model"],
            "raw_answer": answer,
            "parsed": parsed,
        }
    except Exception as exc:
        return {
            "status": "llm_failed",
            "provider": experiment["provider"],
            "model": experiment["model"],
            "error": str(exc),
        }


def build_adjustment_prompt(evidence: dict[str, Any], experiment: dict[str, Any]) -> str:
    presentation = evidence.get("presentation") or {}
    properties = []
    for item in presentation.get("property_table") or []:
        properties.append(
            {
                "property_name": item.get("property_name"),
                "db_evidence_center": item.get("weighted_estimate"),
                "evidence_range": [item.get("range_min"), item.get("range_max")],
                "support_count": item.get("neighbor_count"),
                "supporting_samples": item.get("supporting_samples"),
            }
        )
    neighbors = []
    for item in (evidence.get("neighbors") or [])[: int(experiment["top_k"])]:
        neighbors.append(
            {
                "rank": len(neighbors) + 1,
                "sample_id": item.get("sample_id"),
                "distance": item.get("distance"),
                "shared_feature_count": item.get("shared_feature_count"),
                "top_weighted_shared_features": item.get("top_weighted_shared_features"),
                "neighbor_properties": item.get("bulk_properties"),
            }
        )
    payload = {
        "sample_id": evidence.get("sample_id"),
        "query": evidence.get("query"),
        "top_k": experiment["top_k"],
        "channels": evidence.get("evidence_channels"),
        "feature_basis": evidence.get("feature_basis"),
        "method_context": {
            "query_features": "GC-FID and IR are projected by saved training-set PCA/PLS models.",
            "feature_weights": "Feature-target correlation weights are computed from train-split Pearson/Spearman correlations.",
            "neighbor_ranking": "Weighted shared PCA/PLS latent-feature distance.",
            "db_center": "Inverse-distance weighted interpolation over retrieved historical neighbors.",
            "validation_actuals": "Excluded from this prompt; do not infer or request them.",
        },
        "neighbors": neighbors,
        "property_evidence": properties,
    }
    return (
        "Produce evidence-constrained expert-adjusted estimates for the queried petroleum fraction properties.\n"
        "Do not simply copy the DB evidence center. You may keep it if chemically justified, but you must state why.\n"
        "Use the neighbor distances, feature relevance, evidence range, distillation-curve consistency, density/composition relationships, and uncertainty.\n"
        "The adjusted estimate must usually stay within the evidence range unless you explicitly mark it as extrapolated.\n"
        "Return valid JSON only, with this schema:\n"
        "{\n"
        '  "sample_id": "...",\n'
        '  "model_reasoning_summary": "...",\n'
        '  "properties": [\n'
        '    {"property_name": "density_20c", "adjusted_estimate": 0.0, "evidence_center": 0.0, "evidence_range": [0.0, 0.0], "confidence": "low|moderate|high", "rationale": "..."}\n'
        "  ]\n"
        "}\n"
        "Use numeric values, not vague phrases. For BP values, use deg C numeric values.\n\n"
        f"Evidence JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_json_answer(answer: str) -> dict[str, Any]:
    text = (answer or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(top:json)top\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def summarize_evidence(evidence: dict[str, Any], experiment: dict[str, Any], workbook: Path) -> dict[str, Any]:
    neighbors = evidence.get("neighbors") or []
    nearest = neighbors[0] if neighbors else {}
    profile = evidence.get("query_profile") or {}
    return {
        **experiment_keys(experiment),
        "workbook": workbook.name,
        "sample_id": evidence.get("sample_id"),
        "raw_sample_id": profile.get("raw_sample_id"),
        "status": evidence.get("status"),
        "feature_count": profile.get("feature_count"),
        "neighbor_count": len(neighbors),
        "nearest_neighbor": nearest.get("sample_id"),
        "nearest_distance": nearest.get("distance"),
    }


def db_prediction_rows(evidence: dict[str, Any], experiment: dict[str, Any], workbook: Path) -> list[dict[str, Any]]:
    if evidence.get("status") != "success":
        return []
    profile = evidence.get("query_profile") or {}
    rows = []
    for item in (evidence.get("presentation") or {}).get("property_table") or []:
        estimate = to_float(item.get("weighted_estimate"))
        actual = to_float(item.get("actual_for_validation"))
        rows.append(
            {
                **experiment_keys(experiment),
                "prediction_source": "db_evidence_center",
                "workbook": workbook.name,
                "sample_id": evidence.get("sample_id"),
                "raw_sample_id": profile.get("raw_sample_id"),
                "target_property": item.get("property_name"),
                "estimate": estimate,
                "actual": actual,
                "error": safe_subtract(estimate, actual),
                "abs_error": safe_abs(safe_subtract(estimate, actual)),
                "absolute_relative_error_percent": abs_relative_error(estimate, actual),
                "range_min": item.get("range_min"),
                "range_max": item.get("range_max"),
                "support_count": item.get("neighbor_count"),
            }
        )
    return rows


def llm_prediction_rows(evidence: dict[str, Any], experiment: dict[str, Any], workbook: Path, llm_result: dict[str, Any]) -> list[dict[str, Any]]:
    profile = evidence.get("query_profile") or {}
    actual_by_property = {
        item.get("property_name"): to_float(item.get("actual_for_validation"))
        for item in (evidence.get("presentation") or {}).get("property_table") or []
    }
    center_by_property = {
        item.get("property_name"): to_float(item.get("weighted_estimate"))
        for item in (evidence.get("presentation") or {}).get("property_table") or []
    }
    rows = []
    parsed = llm_result.get("parsed") if llm_result.get("status") == "success" else {}
    properties = parsed.get("properties") if isinstance(parsed, dict) else []
    if not isinstance(properties, list):
        properties = []
    for item in properties:
        if not isinstance(item, dict):
            continue
        target = item.get("property_name")
        estimate = to_float(item.get("adjusted_estimate"))
        actual = actual_by_property.get(target)
        rows.append(
            {
                **experiment_keys(experiment),
                "prediction_source": "llm_expert_adjusted",
                "workbook": workbook.name,
                "sample_id": evidence.get("sample_id"),
                "raw_sample_id": profile.get("raw_sample_id"),
                "target_property": target,
                "estimate": estimate,
                "db_evidence_center": center_by_property.get(target),
                "actual": actual,
                "error": safe_subtract(estimate, actual),
                "abs_error": safe_abs(safe_subtract(estimate, actual)),
                "absolute_relative_error_percent": abs_relative_error(estimate, actual),
                "confidence": item.get("confidence"),
                "rationale": item.get("rationale"),
                "llm_status": llm_result.get("status"),
                "llm_error": llm_result.get("error"),
                "raw_answer": llm_result.get("raw_answer"),
            }
        )
    if rows:
        return rows
    return [
        {
            **experiment_keys(experiment),
            "prediction_source": "llm_expert_adjusted",
            "workbook": workbook.name,
            "sample_id": evidence.get("sample_id"),
            "raw_sample_id": profile.get("raw_sample_id"),
            "target_property": None,
            "estimate": None,
            "actual": None,
            "llm_status": llm_result.get("status"),
            "llm_error": llm_result.get("error"),
            "raw_answer": llm_result.get("raw_answer"),
        }
    ]


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "target_property" not in predictions:
        return pd.DataFrame()
    valid = predictions.dropna(subset=["target_property", "estimate", "actual"]).copy()
    if valid.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["experiment_type", "model_label", "provider", "model", "top_k", "replicate", "prediction_source", "target_property"]
    for keys, group in valid.groupby(group_cols, sort=False, dropna=False):
        errors = pd.to_numeric(group["error"], errors="coerce").dropna()
        abs_errors = pd.to_numeric(group["abs_error"], errors="coerce").dropna()
        ape = pd.to_numeric(group["absolute_relative_error_percent"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "n": int(len(errors)),
                "MAE": round_or_none(abs_errors.mean()),
                "RMSE": round_or_none(math.sqrt(float((errors**2).mean())) if len(errors) else None),
                "MRE_percent": round_or_none(pd.to_numeric(group["error"] / group["actual"] * 100, errors="coerce").mean()),
                "MAPE_percent": round_or_none(ape.mean()),
                "median_abs_error": round_or_none(abs_errors.median()),
                "max_abs_error": round_or_none(abs_errors.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_experiment_metrics(db_metrics: pd.DataFrame, llm_metrics: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (db_metrics, llm_metrics) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    rows = []
    group_cols = ["experiment_type", "model_label", "provider", "model", "top_k", "replicate", "prediction_source"]
    for keys, group in data.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "mean_MAE_across_targets": round_or_none(pd.to_numeric(group["MAE"], errors="coerce").mean()),
                "mean_MAPE_across_targets": round_or_none(pd.to_numeric(group["MAPE_percent"], errors="coerce").mean()),
                "target_count": int(group["target_property"].nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_repeatability_metrics(comparison_summary: pd.DataFrame) -> pd.DataFrame:
    if comparison_summary.empty or "replicate" not in comparison_summary:
        return pd.DataFrame()
    rows = []
    group_cols = ["experiment_type", "model_label", "provider", "model", "top_k", "prediction_source"]
    for keys, group in comparison_summary.groupby(group_cols, sort=False, dropna=False):
        mae = pd.to_numeric(group["mean_MAE_across_targets"], errors="coerce").dropna()
        mape = pd.to_numeric(group["mean_MAPE_across_targets"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "replicate_count": int(group["replicate"].nunique()),
                "mean_MAE": round_or_none(mae.mean()),
                "std_MAE": round_or_none(mae.std(ddof=1) if len(mae) > 1 else 0),
                "mean_MAPE": round_or_none(mape.mean()),
                "std_MAPE": round_or_none(mape.std(ddof=1) if len(mape) > 1 else 0),
                "min_MAPE": round_or_none(mape.min()),
                "max_MAPE": round_or_none(mape.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_repeatability_errorbar_svg(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("", encoding="utf-8")
        return
    data = summary.copy()
    data = data[data["prediction_source"] == "llm_expert_adjusted"].copy()
    if data.empty:
        path.write_text("", encoding="utf-8")
        return
    model_order = {"gpt5.5": 0, "gpt4.1": 1, "deepseek_v4_reasoning": 2}
    data["model_order"] = data["model_label"].map(model_order).fillna(99)
    data = data.sort_values(["experiment_type", "model_order", "top_k"])

    rows = data.to_dict("records")
    width = 1120
    row_h = 34
    top = 86
    left = 250
    plot_w = 700
    height = top + row_h * len(rows) + 70
    max_x = max(float(row.get("mean_MAPE") or 0) + float(row.get("std_MAPE") or 0) for row in rows)
    max_x = max(1.0, math.ceil(max_x + 1))

    colors = {
        "gpt5.5": "#356fd6",
        "gpt4.1": "#178a7a",
        "deepseek_v4_reasoning": "#d97706",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="40" font-family="Arial" font-size="24" font-weight="700" fill="#10213d">petroleum fraction experimental DB LLM repeatability</text>',
        '<text x="32" y="66" font-family="Arial" font-size="13" fill="#50627d">Mean MAPE across target properties; error bars show standard deviation across repeated LLM runs.</text>',
        f'<line x1="{left}" y1="{top-24}" x2="{left + plot_w}" y2="{top-24}" stroke="#d7e0ee" stroke-width="1"/>',
    ]
    for tick in range(0, int(max_x) + 1, max(1, int(max_x // 6) or 1)):
        x = left + tick / max_x * plot_w
        svg.append(f'<line x1="{x:.1f}" y1="{top-30}" x2="{x:.1f}" y2="{height-42}" stroke="#eef3fa" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-20}" font-family="Arial" font-size="11" fill="#50627d" text-anchor="middle">{tick}%</text>')

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        mean = float(row.get("mean_MAPE") or 0)
        std = float(row.get("std_MAPE") or 0)
        x_mean = left + mean / max_x * plot_w
        x_low = left + max(0.0, mean - std) / max_x * plot_w
        x_high = left + (mean + std) / max_x * plot_w
        model = str(row.get("model_label") or "")
        color = colors.get(model, "#4b5563")
        label = f'{model}, k={int(row.get("top_k") or 0)}, n={int(row.get("replicate_count") or 0)}'
        svg.append(f'<text x="32" y="{y+5}" font-family="Arial" font-size="13" fill="#10213d">{escape_svg(label)}</text>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="{color}" stroke-width="2.2"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y-7}" x2="{x_low:.1f}" y2="{y+7}" stroke="{color}" stroke-width="2.2"/>')
        svg.append(f'<line x1="{x_high:.1f}" y1="{y-7}" x2="{x_high:.1f}" y2="{y+7}" stroke="{color}" stroke-width="2.2"/>')
        svg.append(f'<circle cx="{x_mean:.1f}" cy="{y}" r="5.2" fill="{color}"/>')
        svg.append(f'<text x="{x_high + 10:.1f}" y="{y+5}" font-family="Arial" font-size="12" fill="#10213d">{mean:.2f} +/- {std:.2f}%</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def escape_svg(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def experiment_keys(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_type": experiment.get("experiment_type"),
        "model_label": experiment.get("model_label"),
        "provider": experiment.get("provider"),
        "model": experiment.get("model"),
        "top_k": experiment.get("top_k"),
        "replicate": experiment.get("replicate") or 1,
    }


def to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def safe_subtract(left: Any, right: Any) -> float | None:
    x = to_float(left)
    y = to_float(right)
    if x is None or y is None:
        return None
    return x - y


def safe_abs(value: Any) -> float | None:
    x = to_float(value)
    return abs(x) if x is not None else None


def abs_relative_error(estimate: Any, actual: Any) -> float | None:
    e = to_float(estimate)
    a = to_float(actual)
    if e is None or a in (None, 0):
        return None
    return abs((e - a) / a) * 100


def round_or_none(value: Any, digits: int = 4) -> float | None:
    numeric = to_float(value)
    return round(numeric, digits) if numeric is not None else None


def write_readme(
    out_dir: Path,
    summary: dict[str, Any],
    db_metrics: pd.DataFrame,
    llm_metrics: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    repeatability_summary: pd.DataFrame,
) -> None:
    lines = [
        "# petroleum fraction experimental DB LLM Comparison",
        "",
        "This experiment follows the Experimental DB evidence-test path: each test-set workbook is uploaded as a transient query sample, projected into saved GC-FID/IR PCA/PLS latent features, matched against train-split historical records, and summarized as evidence tables.",
        "",
        "Numerical baselines are DB evidence centers from inverse-distance weighted interpolation. LLM adjusted estimates are requested as structured JSON and are evaluated against held-out validation values after the LLM call.",
        "",
        "## Settings",
        f"- Query: `{summary.get('query')}`",
        f"- Workbooks: {summary.get('workbook_count')}",
        f"- LLM enabled: {summary.get('llm_enabled')}",
        "",
        "## Output files",
        "- `evidence_summary.csv`: query workbook, top-k setting, feature count, and nearest-neighbor summary.",
        "- `db_evidence_center_predictions.csv`: deterministic DB evidence-center predictions and validation errors.",
        "- `db_evidence_center_metrics.csv`: DB-center MAE/RMSE/MAPE by target and top-k.",
        "- `llm_adjusted_predictions.csv`: parsed LLM adjusted estimates, rationales, and validation errors.",
        "- `llm_adjusted_metrics.csv`: LLM adjusted-estimate MAE/RMSE/MAPE by target/model/top-k.",
        "- `comparison_summary.csv`: compact cross-target summary for top-k and model comparison.",
        "- `repeatability_summary.csv`: mean and standard deviation of repeated runs for each model/top-k condition.",
        "- `repeatability_mean_mape_errorbars.svg`: error-bar plot of repeated-run mean MAPE.",
        "- `evidence_bundles.jsonl`: full evidence bundles used for audit.",
        "",
        "## DB Evidence-Center Metrics",
        markdown_table(db_metrics),
        "",
        "## LLM Adjusted Metrics",
        markdown_table(llm_metrics),
        "",
        "## Comparison Summary",
        markdown_table(comparison_summary),
        "",
        "## Repeatability Summary",
        markdown_table(repeatability_summary),
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy().where(pd.notna(frame), "")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
