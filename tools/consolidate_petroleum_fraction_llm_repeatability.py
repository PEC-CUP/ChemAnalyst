from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BASE_FINAL = Path("outputs") / "petroleum_fraction_experimental_db" / "final_evaluation"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_FINAL / "repeatability"


def main() -> None:
    args = parse_args()
    base_final = Path(args.base_final)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_comparison_frames(base_final, [Path(p) for p in args.repeat_run])
    property_metrics = load_property_metric_frames(base_final, [Path(p) for p in args.repeat_run])

    comparison_for_repeatability = comparison[comparison["experiment_type"] == "topk_sensitivity"].copy() if "experiment_type" in comparison else comparison
    property_metrics_for_repeatability = (
        property_metrics[property_metrics["experiment_type"] == "topk_sensitivity"].copy()
        if "experiment_type" in property_metrics
        else property_metrics
    )

    overall_summary = repeatability_summary(
        comparison_for_repeatability,
        value_cols=("mean_MAE_across_targets", "mean_MAPE_across_targets"),
        group_cols=["experiment_type", "model_label", "provider", "model", "top_k", "prediction_source"],
    )
    property_summary = repeatability_summary(
        property_metrics_for_repeatability,
        value_cols=("MAE", "RMSE", "MAPE_percent"),
        group_cols=[
            "experiment_type",
            "model_label",
            "provider",
            "model",
            "top_k",
            "prediction_source",
            "target_property",
        ],
    )

    comparison.to_csv(out_dir / "combined_comparison_summary.csv", index=False, encoding="utf-8-sig")
    property_metrics.to_csv(out_dir / "combined_property_metrics.csv", index=False, encoding="utf-8-sig")
    overall_summary.to_csv(out_dir / "overall_repeatability_summary.csv", index=False, encoding="utf-8-sig")
    property_summary.to_csv(out_dir / "property_repeatability_summary.csv", index=False, encoding="utf-8-sig")
    write_overall_errorbar_svg(overall_summary, out_dir / "overall_mape_errorbars.svg")
    write_property_errorbar_svg(property_summary, out_dir / "property_mape_errorbars.svg")
    write_readme(out_dir, args, comparison, property_metrics, overall_summary, property_summary)
    print(f"Wrote repeatability summary to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate petroleum fraction LLM repeated-run metrics and generate error-bar summaries.")
    parser.add_argument("--base-final", default=str(DEFAULT_BASE_FINAL), help="Existing final_evaluation directory used as replicate 1.")
    parser.add_argument("--repeat-run", nargs="*", default=[], help="Additional run directories produced by run_petroleum_fraction_experimental_db_llm_comparison.py.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_comparison_frames(base_final: Path, repeat_runs: list[Path]) -> pd.DataFrame:
    frames = []
    base_path = base_final / "comparison_summary.csv"
    if base_path.exists():
        frame = pd.read_csv(base_path)
        if "replicate" not in frame:
            frame["replicate"] = 1
        frames.append(frame)
    for run_dir in repeat_runs:
        metrics = metric_table(load_checkpoint_predictions(run_dir))
        if not metrics.empty:
            frames.append(aggregate_experiment_metrics(metrics))
            continue
        path = run_dir / "comparison_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "replicate" not in frame:
                frame["replicate"] = infer_replicate(run_dir)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_property_metric_frames(base_final: Path, repeat_runs: list[Path]) -> pd.DataFrame:
    frames = []
    base_path = base_final / "all_llm_adjusted_metrics_by_property.csv"
    if base_path.exists():
        frame = pd.read_csv(base_path)
        if "replicate" not in frame:
            frame["replicate"] = 1
        frames.append(frame)
    for run_dir in repeat_runs:
        metrics = metric_table(load_checkpoint_predictions(run_dir))
        if not metrics.empty:
            frames.append(metrics)
            continue
        path = run_dir / "llm_adjusted_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "replicate" not in frame:
                frame["replicate"] = infer_replicate(run_dir)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data[data.get("prediction_source", "") == "llm_expert_adjusted"].copy()
    return data


def load_checkpoint_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "llm_results_checkpoint.jsonl"
    if not path.exists():
        return pd.DataFrame()
    latest_success: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        llm_result = item.get("llm_result")
        if not isinstance(llm_result, dict) or llm_result.get("status") != "success":
            continue
        key_items = item.get("cache_key")
        if isinstance(key_items, list | tuple):
            key = tuple(key_items)
        else:
            exp = item.get("experiment") or {}
            key = (
                exp.get("provider"),
                exp.get("model"),
                exp.get("thinking"),
                exp.get("top_k"),
                exp.get("replicate") or 1,
                Path(item.get("workbook") or "").name,
            )
        latest_success[key] = item

    rows: list[dict[str, Any]] = []
    for item in latest_success.values():
        for row in item.get("prediction_rows") or []:
            if isinstance(row, dict) and row.get("prediction_source") == "llm_expert_adjusted":
                rows.append(row)
    return pd.DataFrame(rows)


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


def aggregate_experiment_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["experiment_type", "model_label", "provider", "model", "top_k", "replicate", "prediction_source"]
    for keys, group in metrics.groupby(group_cols, sort=False, dropna=False):
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


def infer_replicate(path: Path) -> int:
    for part in reversed(path.parts):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            return int(digits[-2:])
    return 1


def repeatability_summary(frame: pd.DataFrame, *, value_cols: tuple[str, ...], group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in frame.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys, strict=False))
        row["replicate_count"] = int(group["replicate"].nunique()) if "replicate" in group else 1
        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna() if col in group else pd.Series(dtype=float)
            row[f"{col}_mean"] = round_or_none(values.mean())
            row[f"{col}_std"] = round_or_none(values.std(ddof=1) if len(values) > 1 else 0)
            row[f"{col}_min"] = round_or_none(values.min())
            row[f"{col}_max"] = round_or_none(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_overall_errorbar_svg(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("", encoding="utf-8")
        return
    data = summary[summary["prediction_source"] == "llm_expert_adjusted"].copy()
    data = data.sort_values(["model_label", "top_k"])
    write_errorbar_svg(
        data,
        path,
        title="petroleum fraction experimental DB LLM repeatability",
        label_fn=lambda r: f"{r['model_label']}, k={int(r['top_k'])}, n={int(r['replicate_count'])}",
        mean_col="mean_MAPE_across_targets_mean",
        std_col="mean_MAPE_across_targets_std",
    )


def write_property_errorbar_svg(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("", encoding="utf-8")
        return
    data = summary[summary["prediction_source"] == "llm_expert_adjusted"].copy()
    data = data.sort_values(["target_property", "model_label", "top_k"])
    write_errorbar_svg(
        data,
        path,
        title="Property-level LLM repeatability",
        label_fn=lambda r: f"{r['target_property']} | {r['model_label']} k={int(r['top_k'])}",
        mean_col="MAPE_percent_mean",
        std_col="MAPE_percent_std",
        max_rows=80,
    )


def write_errorbar_svg(
    data: pd.DataFrame,
    path: Path,
    *,
    title: str,
    label_fn: Any,
    mean_col: str,
    std_col: str,
    max_rows: int = 48,
) -> None:
    if data.empty or mean_col not in data:
        path.write_text("", encoding="utf-8")
        return
    rows = data.head(max_rows).to_dict("records")
    width = 1180
    row_h = 30
    top = 84
    left = 330
    plot_w = 670
    height = top + row_h * len(rows) + 62
    max_x = max(float(r.get(mean_col) or 0) + float(r.get(std_col) or 0) for r in rows)
    max_x = max(1.0, math.ceil(max_x + 1))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="38" font-family="Arial" font-size="23" font-weight="700" fill="#10213d">{escape_svg(title)}</text>',
        '<text x="30" y="62" font-family="Arial" font-size="13" fill="#50627d">Mean MAPE with standard deviation across repeated LLM runs.</text>',
    ]
    tick_step = max(1, int(max_x // 6) or 1)
    for tick in range(0, int(max_x) + 1, tick_step):
        x = left + tick / max_x * plot_w
        svg.append(f'<line x1="{x:.1f}" y1="{top-26}" x2="{x:.1f}" y2="{height-40}" stroke="#edf2f8"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-18}" font-family="Arial" font-size="11" fill="#50627d" text-anchor="middle">{tick}%</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        mean = float(row.get(mean_col) or 0)
        std = float(row.get(std_col) or 0)
        x_mean = left + mean / max_x * plot_w
        x_low = left + max(0.0, mean - std) / max_x * plot_w
        x_high = left + (mean + std) / max_x * plot_w
        svg.append(f'<text x="30" y="{y+5}" font-family="Arial" font-size="12" fill="#10213d">{escape_svg(label_fn(row))}</text>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="#2f6fdd" stroke-width="2.2"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y-6}" x2="{x_low:.1f}" y2="{y+6}" stroke="#2f6fdd" stroke-width="2.2"/>')
        svg.append(f'<line x1="{x_high:.1f}" y1="{y-6}" x2="{x_high:.1f}" y2="{y+6}" stroke="#2f6fdd" stroke-width="2.2"/>')
        svg.append(f'<circle cx="{x_mean:.1f}" cy="{y}" r="4.8" fill="#0f8a7b"/>')
        svg.append(f'<text x="{x_high+8:.1f}" y="{y+5}" font-family="Arial" font-size="11" fill="#10213d">{mean:.2f} +/- {std:.2f}%</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_readme(
    out_dir: Path,
    args: argparse.Namespace,
    comparison: pd.DataFrame,
    property_metrics: pd.DataFrame,
    overall: pd.DataFrame,
    property_summary: pd.DataFrame,
) -> None:
    lines = [
        "# petroleum fraction LLM Repeatability Summary",
        "",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This directory combines the existing final-evaluation run as replicate 1 with additional repeated LLM runs.",
        "",
        "## Inputs",
        f"- Base final evaluation: `{args.base_final}`",
        f"- Additional run directories: {', '.join(f'`{p}`' for p in args.repeat_run) or 'none'}",
        "",
        "## Outputs",
        "- `combined_comparison_summary.csv`: overall mean MAE/MAPE rows with replicate labels.",
        "- `combined_property_metrics.csv`: property-level MAE/RMSE/MAPE rows with replicate labels.",
        "- `overall_repeatability_summary.csv`: mean/std/min/max of overall MAPE across repeats.",
        "- `property_repeatability_summary.csv`: mean/std/min/max of property-level MAPE across repeats.",
        "- `overall_mape_errorbars.svg`: overall MAPE error bars.",
        "- `property_mape_errorbars.svg`: property-level MAPE error bars.",
        "",
        "## Overall Repeatability",
        markdown_table(overall),
        "",
        "## Property Repeatability",
        markdown_table(property_summary),
        "",
    ]
    out_dir.joinpath("README.zh.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy().where(pd.notna(frame), "")
    columns = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, digits) if math.isfinite(numeric) else None


def escape_svg(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
