from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SHEET_RAW = "raw data"
SHEET_PEAK = "peak area"
SHEET_PROP = "property"

SOLVENT_RT = 6.23
SOLVENT_WINDOW = 0.2

RAW_BINS = 64
PEAK_BINS = 32
AREA_RT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

ANCHOR_WINDOWS = {
    "c17_pr": (37.6, 38.6),
    "c18_ph": (39.0, 40.0),
}

PROPERTY_SAMPLE_NUMBER = "Sample number"
PROPERTY_DENSITY_PREFIX = "Density"
PROPERTY_BP_PREFIX = "BP_"
TEMPLATE_REFERENCE_PATH = Path("knowledge/manuals/gc_match/template_reference.xlsx")

EXPERT_PRIORS = [
    "Do not compare absolute chromatogram intensities across samples; only compare within-sample shape.",
    "Treat the strong solvent peak near 6.23 min as an artifact and exclude it from reasoning.",
    "Allow retention-time drift across runs and temperature programs; do not require strict template identity.",
    "Use paired high-abundance peaks in the C17/Pr and C18/Ph regions as anchor candidates when reasoning about carbon-number alignment.",
    "After identifying anchor candidates, use the regular recurrence of dominant n-alkane peaks to reason about neighboring carbon numbers.",
    "Infer properties from historical-neighbor evidence and feature consistency, not from a trained regression model.",
]

REFERENCE_NAME_ALIASES = {
    "text": "n-C17",
    "text": "n-C18",
    "text": "Pr",
    "text": "Ph",
}


@dataclass(slots=True)
class SampleAssets:
    sample_id: str
    workbook_path: Path
    raw_df: pd.DataFrame
    peak_df: pd.DataFrame
    property_row: dict[str, Any] | None


def _sample_sort_key(path: Path) -> tuple[int, str]:
    number = int(re.sub(r"\D", "", path.stem) or "0")
    return number, path.stem


def _sample_id_from_filename(path: Path) -> str:
    return path.stem.strip()


def _find_header_row(df: pd.DataFrame, required: list[str]) -> int:
    normalized_required = {str(value).strip().lower() for value in required}
    for idx in range(min(len(df), 25)):
        row = {str(value).strip().lower() for value in df.iloc[idx].tolist()}
        if normalized_required.issubset(row):
            return idx
    raise ValueError(f"Header row not found for columns={required}")


def _load_sheet_with_header(path: Path, sheet_name: str, required: list[str]) -> pd.DataFrame:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(preview, required)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_raw_sheet(path: Path) -> pd.DataFrame:
    df = _load_sheet_with_header(path, SHEET_RAW, ["Time", "Intensity"])
    keep = df.loc[:, [col for col in df.columns if col in {"Time", "Intensity"}]].copy()
    keep["Time"] = pd.to_numeric(keep["Time"], errors="coerce")
    keep["Intensity"] = pd.to_numeric(keep["Intensity"], errors="coerce")
    keep = keep.dropna(subset=["Time", "Intensity"]).reset_index(drop=True)
    keep = keep.loc[~keep["Time"].between(SOLVENT_RT - SOLVENT_WINDOW, SOLVENT_RT + SOLVENT_WINDOW)].reset_index(drop=True)
    return keep


def load_peak_sheet(path: Path) -> pd.DataFrame:
    df = _load_sheet_with_header(path, SHEET_PEAK, ["Apex RT", "Area"])
    rename: dict[str, str] = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered == "apex rt":
            rename[col] = "Apex_RT"
        elif lowered == "area":
            rename[col] = "Area"
        elif lowered == "height":
            rename[col] = "Height"
    keep = df.rename(columns=rename).copy()
    for col in ["Apex_RT", "Area", "Height"]:
        if col in keep.columns:
            keep[col] = pd.to_numeric(keep[col], errors="coerce")
    keep = keep.dropna(subset=["Apex_RT", "Area"]).reset_index(drop=True)
    keep = keep.loc[~keep["Apex_RT"].between(SOLVENT_RT - SOLVENT_WINDOW, SOLVENT_RT + SOLVENT_WINDOW)].reset_index(drop=True)
    return keep


def _target_property_columns(columns: Iterable[str]) -> list[str]:
    result: list[str] = []
    for col in columns:
        name = str(col).strip()
        if name == PROPERTY_SAMPLE_NUMBER:
            continue
        if name.startswith(PROPERTY_DENSITY_PREFIX) or name.startswith(PROPERTY_BP_PREFIX):
            result.append(name)
    return result


def load_property_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(col).strip() for col in df.columns]
    if PROPERTY_SAMPLE_NUMBER not in df.columns:
        raise ValueError("property.xlsx missing 'Sample number' column")
    df[PROPERTY_SAMPLE_NUMBER] = df[PROPERTY_SAMPLE_NUMBER].astype(str).str.strip()
    for col in _target_property_columns(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _property_columns(properties: pd.DataFrame) -> list[str]:
    return _target_property_columns(properties.columns)


def _workspace_property_table(out_root: Path) -> pd.DataFrame:
    prop_dir = out_root / "normalized" / "properties"
    rows: list[dict[str, Any]] = []
    if not prop_dir.exists():
        return pd.DataFrame(columns=[PROPERTY_SAMPLE_NUMBER])
    for path in sorted(prop_dir.glob("*.property.json"), key=_sample_sort_key):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    if not rows:
        return pd.DataFrame(columns=[PROPERTY_SAMPLE_NUMBER])
    df = pd.DataFrame(rows)
    df.columns = [str(col).strip() for col in df.columns]
    if PROPERTY_SAMPLE_NUMBER in df.columns:
        df[PROPERTY_SAMPLE_NUMBER] = df[PROPERTY_SAMPLE_NUMBER].astype(str).str.strip()
    for col in _target_property_columns(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(arr.sum())
    if total > 0:
        return arr / total
    max_val = float(arr.max()) if arr.size else 0.0
    return arr / max_val if max_val > 0 else arr


def _binned_profile(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    x_min: float | None = None,
    x_max: float | None = None,
) -> list[float]:
    if x.size == 0 or y.size == 0:
        return [0.0] * bins
    left = float(np.min(x)) if x_min is None else x_min
    right = float(np.max(x)) if x_max is None else x_max
    if not math.isfinite(left) or not math.isfinite(right) or right <= left:
        return [0.0] * bins
    hist, _ = np.histogram(x, bins=bins, range=(left, right), weights=y)
    profile = _normalize_series(hist)
    return [round(float(value), 6) for value in profile.tolist()]


def _weighted_rt_quantiles(rt: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, float | None]:
    if rt.size == 0 or weights.size == 0:
        return {f"q{int(q * 100):02d}": None for q in quantiles}
    order = np.argsort(rt)
    rt_sorted = np.asarray(rt[order], dtype=float)
    weight_sorted = np.asarray(weights[order], dtype=float)
    weight_sorted = np.nan_to_num(weight_sorted, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(weight_sorted.sum())
    if total <= 0:
        return {f"q{int(q * 100):02d}": None for q in quantiles}
    cumulative = np.cumsum(weight_sorted) / total
    result: dict[str, float | None] = {}
    for quantile in quantiles:
        idx = int(np.searchsorted(cumulative, quantile, side="left"))
        idx = min(max(idx, 0), len(rt_sorted) - 1)
        result[f"q{int(quantile * 100):02d}"] = round(float(rt_sorted[idx]), 4)
    return result


def _peak_top_rts(peak_df: pd.DataFrame, top_n: int = 12) -> list[float]:
    subset = peak_df.sort_values("Area", ascending=False).head(top_n)
    return [round(float(value), 4) for value in subset["Apex_RT"].tolist()]


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _anchor_candidates(peak_df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (left, right) in ANCHOR_WINDOWS.items():
        window = peak_df.loc[peak_df["Apex_RT"].between(left, right)].sort_values("Area", ascending=False).head(6)
        result[name] = [
            {
                "rt": round(float(row.Apex_RT), 4),
                "area": round(float(row.Area), 4),
            }
            for row in window.itertuples()
        ]
    return result


def _anchor_pair_summary(anchor_candidates: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for anchor_name, rows in anchor_candidates.items():
        candidates = sorted(rows or [], key=lambda item: item["rt"])
        pair = candidates[:2]
        if len(pair) < 2:
            summary[anchor_name] = {
                "available": False,
                "pair_rts": [entry["rt"] for entry in pair],
                "pair_gap": None,
                "pair_area_ratio": None,
            }
            continue
        first, second = pair
        summary[anchor_name] = {
            "available": True,
            "pair_rts": [first["rt"], second["rt"]],
            "pair_gap": round(float(second["rt"] - first["rt"]), 4),
            "pair_area_ratio": round(float(second["area"] / first["area"]), 4) if first["area"] else None,
        }
    return summary


def _raw_shape_features(raw_df: pd.DataFrame) -> dict[str, Any]:
    time = raw_df["Time"].to_numpy(dtype=float)
    intensity = raw_df["Intensity"].to_numpy(dtype=float)
    normalized_intensity = _normalize_series(intensity.copy())
    dominant_idx = int(np.argmax(normalized_intensity)) if normalized_intensity.size else 0
    return {
        "raw_bins": _binned_profile(time, normalized_intensity, bins=RAW_BINS),
        "rt_range": {
            "min": round(float(time.min()), 4) if time.size else None,
            "max": round(float(time.max()), 4) if time.size else None,
        },
        "dominant_peak_rt": round(float(time[dominant_idx]), 4) if time.size else None,
        "intensity_quantiles": {
            "q50": round(float(np.quantile(normalized_intensity, 0.5)), 6) if normalized_intensity.size else None,
            "q90": round(float(np.quantile(normalized_intensity, 0.9)), 6) if normalized_intensity.size else None,
            "q99": round(float(np.quantile(normalized_intensity, 0.99)), 6) if normalized_intensity.size else None,
        },
        "cumulative_rt_quantiles": _weighted_rt_quantiles(time, normalized_intensity, AREA_RT_QUANTILES),
    }


def _peak_area_features(peak_df: pd.DataFrame) -> dict[str, Any]:
    rt = peak_df["Apex_RT"].to_numpy(dtype=float)
    area = peak_df["Area"].to_numpy(dtype=float)
    profile = _binned_profile(rt, area, bins=PEAK_BINS)
    total_area = float(np.sum(area)) if area.size else 0.0
    area_quantiles = np.quantile(area, [0.5, 0.9, 0.99]) if area.size else [None, None, None]
    anchors = _anchor_candidates(peak_df)
    return {
        "peak_count": int(len(peak_df)),
        "total_area": round(total_area, 4),
        "area_bins": profile,
        "rt_range": {
            "min": round(float(rt.min()), 4) if rt.size else None,
            "max": round(float(rt.max()), 4) if rt.size else None,
        },
        "top_peak_rts": _peak_top_rts(peak_df),
        "anchor_candidates": anchors,
        "anchor_pair_summary": _anchor_pair_summary(anchors),
        "area_quantiles": {
            "q50": round(float(area_quantiles[0]), 4) if area.size else None,
            "q90": round(float(area_quantiles[1]), 4) if area.size else None,
            "q99": round(float(area_quantiles[2]), 4) if area.size else None,
        },
        "cumulative_rt_quantiles": _weighted_rt_quantiles(rt, area, AREA_RT_QUANTILES),
    }


def _classify_signal(value: float | None, *, low: float, high: float) -> str:
    if value is None:
        return "unknown"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def _feature_relation(
    *,
    feature: str,
    value: Any,
    direction: str,
    evidence_strength: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "feature": feature,
        "value": value,
        "direction": direction,
        "evidence_strength": evidence_strength,
        "interpretation": interpretation,
    }


def build_property_feature_profile(features: dict[str, Any]) -> dict[str, Any]:
    peak_shape = features.get("peak_area_features", {}) if isinstance(features.get("peak_area_features"), dict) else {}
    raw_shape = features.get("raw_shape_features", {}) if isinstance(features.get("raw_shape_features"), dict) else {}
    peak_quantiles = peak_shape.get("cumulative_rt_quantiles", {}) if isinstance(peak_shape.get("cumulative_rt_quantiles"), dict) else {}
    raw_quantiles = raw_shape.get("cumulative_rt_quantiles", {}) if isinstance(raw_shape.get("cumulative_rt_quantiles"), dict) else {}
    anchor_summary = peak_shape.get("anchor_pair_summary", {}) if isinstance(peak_shape.get("anchor_pair_summary"), dict) else {}

    peak_q10 = _round_or_none(peak_quantiles.get("q10"))
    peak_q50 = _round_or_none(peak_quantiles.get("q50"))
    peak_q90 = _round_or_none(peak_quantiles.get("q90"))
    raw_q90 = _round_or_none(raw_quantiles.get("q90"))
    peak_count = peak_shape.get("peak_count")
    rt_range = peak_shape.get("rt_range") if isinstance(peak_shape.get("rt_range"), dict) else {}
    rt_span = None
    if rt_range.get("min") is not None and rt_range.get("max") is not None:
        rt_span = _round_or_none(float(rt_range["max"]) - float(rt_range["min"]))
    raw_peak_divergence = _round_or_none(raw_q90 - peak_q90) if raw_q90 is not None and peak_q90 is not None else None

    light_signal = _classify_signal(peak_q10, low=18.0, high=28.0)
    median_signal = _classify_signal(peak_q50, low=34.0, high=44.0)
    heavy_signal = _classify_signal(peak_q90, low=50.0, high=62.0)
    raw_tail_signal = _classify_signal(raw_q90, low=60.0, high=75.0)
    divergence_signal = _classify_signal(raw_peak_divergence, low=8.0, high=20.0)

    density_relations = [
        _feature_relation(
            feature="peak_area_q50_rt",
            value=peak_q50,
            direction="increase_density" if median_signal == "high" else ("decrease_density" if median_signal == "low" else "neutral"),
            evidence_strength="moderate" if peak_q50 is not None else "missing",
            interpretation="Higher area-weighted median RT usually indicates a heavier average boiling/composition center and tends to increase density.",
        ),
        _feature_relation(
            feature="peak_area_q90_rt",
            value=peak_q90,
            direction="increase_density" if heavy_signal == "high" else ("decrease_density" if heavy_signal == "low" else "neutral"),
            evidence_strength="moderate" if peak_q90 is not None else "missing",
            interpretation="Higher q90 RT indicates stronger heavy-end contribution, which tends to increase density.",
        ),
        _feature_relation(
            feature="raw_q90_minus_peak_q90",
            value=raw_peak_divergence,
            direction="increase_density" if divergence_signal == "high" else "neutral",
            evidence_strength="weak" if raw_peak_divergence is not None else "missing",
            interpretation="A large raw/peak q90 gap may indicate unresolved late-eluting material; use cautiously because raw intensity is not comparable across samples.",
        ),
    ]
    bp_relations = {
        "BP_10": [
            _feature_relation(
                feature="peak_area_q10_rt",
                value=peak_q10,
                direction="increase_bp" if light_signal == "high" else ("decrease_bp" if light_signal == "low" else "neutral"),
                evidence_strength="strong" if peak_q10 is not None else "missing",
                interpretation="BP_10 is mainly constrained by the light-end fraction; lower q10 RT tends to lower BP_10, higher q10 RT tends to raise BP_10.",
            )
        ],
        "BP_50": [
            _feature_relation(
                feature="peak_area_q50_rt",
                value=peak_q50,
                direction="increase_bp" if median_signal == "high" else ("decrease_bp" if median_signal == "low" else "neutral"),
                evidence_strength="strong" if peak_q50 is not None else "missing",
                interpretation="BP_50 should track the area-weighted median RT more closely than biomarker ratios.",
            )
        ],
        "BP_90": [
            _feature_relation(
                feature="peak_area_q90_rt",
                value=peak_q90,
                direction="increase_bp" if heavy_signal == "high" else ("decrease_bp" if heavy_signal == "low" else "neutral"),
                evidence_strength="strong" if peak_q90 is not None else "missing",
                interpretation="BP_90 is mainly constrained by late-eluting heavy-end peaks and should not be freely extrapolated without heavy-tail evidence.",
            ),
            _feature_relation(
                feature="raw_q90_rt",
                value=raw_q90,
                direction="increase_bp" if raw_tail_signal == "high" else "neutral",
                evidence_strength="weak" if raw_q90 is not None else "missing",
                interpretation="Raw q90 can suggest unresolved heavy tail but is weaker than integrated peak-area evidence.",
            ),
        ],
    }
    return {
        "schema_version": "property-feature-profile.v1",
        "purpose": "property_inference_only",
        "feature_values": {
            "peak_area_q10_rt": peak_q10,
            "peak_area_q50_rt": peak_q50,
            "peak_area_q90_rt": peak_q90,
            "raw_q90_rt": raw_q90,
            "raw_q90_minus_peak_q90": raw_peak_divergence,
            "peak_count": peak_count,
            "peak_rt_span": rt_span,
            "anchor_pair_summary": anchor_summary,
        },
        "property_relations": {
            "density": density_relations,
            "BP_10": bp_relations["BP_10"],
            "BP_50": bp_relations["BP_50"],
            "BP_90": bp_relations["BP_90"],
        },
        "correction_policy": {
            "baseline_source": "neighbor_property_summary.weighted_estimate",
            "default_action": "keep_baseline",
            "allowed_actions": ["keep_baseline", "small_adjustment", "range_only"],
            "do_not_extrapolate_beyond_neighbor_min_max_without_strong_feature_evidence": True,
            "prefer_range_over_point_estimate_when_feature_evidence_is_weak_or_conflicting": True,
            "biomarker_ratios_are_secondary_for_bulk_density_and_bp": True,
        },
    }


def build_sample_features(sample: SampleAssets) -> dict[str, Any]:
    raw_shape = _raw_shape_features(sample.raw_df)
    peak_shape = _peak_area_features(sample.peak_df)
    features = {
        "schema_version": "crude-gc-db-features.v1",
        "sample_id": sample.sample_id,
        "source_workbook": str(sample.workbook_path),
        "solvent_removed": {
            "rt_center": SOLVENT_RT,
            "window": SOLVENT_WINDOW,
        },
        "raw_shape_features": raw_shape,
        "peak_area_features": peak_shape,
        "property_targets": sample.property_row or {},
        "reasoning_hints": {
            "absolute_intensity_not_comparable_across_samples": True,
            "allow_rt_shift": True,
            "expert_priors": EXPERT_PRIORS,
        },
    }
    features["property_feature_profile"] = build_property_feature_profile(features)
    return features


def build_database_evidence(out_root: Path, sample_id: str, *, k: int = 5) -> dict[str, Any]:
    features_dir = out_root / "features"
    if not features_dir.exists():
        return {"status": "error", "message": "feature_store_missing", "sample_id": sample_id}
    features_by_sample = {
        fp.name.replace(".features.json", ""): json.loads(fp.read_text(encoding="utf-8"))
        for fp in features_dir.glob("*.features.json")
    }
    sample_id = str(sample_id).strip()
    if sample_id not in features_by_sample:
        return {"status": "error", "message": "sample_not_in_database", "sample_id": sample_id}
    properties = _workspace_property_table(out_root)
    matrix = _feature_matrix(features_by_sample)
    if sample_id not in matrix.index:
        return {"status": "error", "message": "sample_not_in_feature_matrix", "sample_id": sample_id}
    neighbors_df = _neighbor_scores(matrix, sample_id).head(max(1, int(k))).copy()
    neighbors = neighbors_df.to_dict(orient="records")
    property_summary = _infer_from_neighbors(neighbors_df, properties) if not properties.empty else {}
    feature_payload = features_by_sample[sample_id]
    peak_shape = feature_payload.get("peak_area_features", {})
    raw_shape = feature_payload.get("raw_shape_features", {})
    return {
        "status": "success",
        "sample_id": sample_id,
        "self_excluded": sample_id not in {row["sample_id"] for row in neighbors},
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": property_summary,
        "feature_digest": {
            "raw_rt_range": raw_shape.get("rt_range"),
            "raw_cumulative_rt_quantiles": raw_shape.get("cumulative_rt_quantiles"),
            "peak_count": peak_shape.get("peak_count"),
            "peak_rt_range": peak_shape.get("rt_range"),
            "top_peak_rts": peak_shape.get("top_peak_rts"),
            "anchor_pair_summary": peak_shape.get("anchor_pair_summary"),
        },
    }


def build_external_database_evidence(
    out_root: Path,
    sample_id: str,
    external_features: dict[str, Any],
    *,
    k: int = 5,
) -> dict[str, Any]:
    features_dir = out_root / "features"
    if not features_dir.exists():
        return {"status": "error", "message": "feature_store_missing", "sample_id": sample_id}
    features_by_sample = {
        fp.name.replace(".features.json", ""): json.loads(fp.read_text(encoding="utf-8"))
        for fp in features_dir.glob("*.features.json")
    }
    if not features_by_sample:
        return {"status": "error", "message": "feature_store_empty", "sample_id": sample_id}

    sample_id = str(sample_id or "external_sample").strip() or "external_sample"
    coerced_features, feature_basis = _coerce_external_features_for_database(external_features, sample_id=sample_id)
    external_vector = _flatten_feature_vector(coerced_features)
    history_matrix = _feature_matrix(features_by_sample)
    comparable_cols = [col for col in history_matrix.columns if col in external_vector]
    if not comparable_cols:
        return {
            "status": "error",
            "message": "external_features_not_comparable",
            "sample_id": sample_id,
            "feature_basis": feature_basis,
        }

    external_id = f"external::{sample_id}"
    external_row = pd.DataFrame([{col: external_vector.get(col, 0.0) for col in comparable_cols}], index=[external_id])
    matrix = pd.concat([history_matrix.loc[:, comparable_cols], external_row], axis=0).fillna(0.0)
    neighbors_df = _neighbor_scores(matrix, external_id).head(max(1, int(k))).copy()
    neighbors = neighbors_df.to_dict(orient="records")
    properties = _workspace_property_table(out_root)
    property_summary = _infer_from_neighbors(neighbors_df, properties) if not properties.empty else {}
    reasoning_case = _reasoning_case(
        sample_id=sample_id,
        features=coerced_features,
        neighbors=neighbors,
        inferred=property_summary,
        actual=None,
    )
    reasoning_case["task"] = "infer_new_crude_properties_from_historical_database"
    reasoning_case["database_usage"] = "external_sample_against_historical_database"
    reasoning_case["weighted_estimates_are_evidence_not_final_answer"] = True
    reasoning_case["llm_prompt"] = _build_property_reasoning_prompt(reasoning_case)
    peak_shape = coerced_features.get("peak_area_features", {}) if isinstance(coerced_features.get("peak_area_features"), dict) else {}
    raw_shape = coerced_features.get("raw_shape_features", {}) if isinstance(coerced_features.get("raw_shape_features"), dict) else {}
    return {
        "status": "success",
        "message": "external_sample_compared_to_historical_database",
        "sample_id": sample_id,
        "feature_basis": feature_basis,
        "self_excluded": sample_id not in {row["sample_id"] for row in neighbors},
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": property_summary,
        "llm_reasoning_case": reasoning_case,
        "feature_digest": {
            "raw_rt_range": raw_shape.get("rt_range"),
            "raw_cumulative_rt_quantiles": raw_shape.get("cumulative_rt_quantiles"),
            "peak_count": peak_shape.get("peak_count"),
            "peak_rt_range": peak_shape.get("rt_range"),
            "top_peak_rts": peak_shape.get("top_peak_rts"),
            "anchor_pair_summary": peak_shape.get("anchor_pair_summary"),
        },
        "interpretation_policy": {
            "nearest_neighbors_are_evidence": True,
            "final_property_inference_requires_llm_reasoning": True,
            "do_not_copy_weighted_estimate_blindly": True,
        },
    }


def _coerce_external_features_for_database(external_features: dict[str, Any], *, sample_id: str) -> tuple[dict[str, Any], str]:
    if not isinstance(external_features, dict):
        return {"schema_version": "crude-gc-db-features.v1", "sample_id": sample_id}, "missing_external_features"
    if isinstance(external_features.get("peak_area_features"), dict):
        payload = dict(external_features)
        payload.setdefault("schema_version", "crude-gc-db-features.v1")
        payload.setdefault("sample_id", sample_id)
        payload.setdefault("reasoning_hints", {"expert_priors": EXPERT_PRIORS})
        payload.setdefault("property_feature_profile", build_property_feature_profile(payload))
        return payload, "crude_db_feature_schema"

    source_file = external_features.get("source_file")
    peak_features = _peak_area_features_from_external_source(source_file)
    if peak_features:
        payload = {
                "schema_version": "crude-gc-db-features.v1",
                "sample_id": sample_id,
                "source_workbook": str(source_file or ""),
                "peak_area_features": peak_features,
                "reasoning_hints": {"expert_priors": EXPERT_PRIORS},
            }
        payload["property_feature_profile"] = build_property_feature_profile(payload)
        return payload, "gc_feature_source_file_peak_table"

    rt_range = external_features.get("rt_range") or (external_features.get("match_summary") or {}).get("rt_range") or {}
    match_summary = external_features.get("match_summary") or {}
    payload = {
            "schema_version": "crude-gc-db-features.v1",
            "sample_id": sample_id,
            "peak_area_features": {
                "peak_count": int(match_summary.get("total_peaks") or match_summary.get("matched_peaks") or 0),
                "rt_range": rt_range,
                "top_peak_rts": [],
                "anchor_candidates": {},
                "anchor_pair_summary": {},
                "cumulative_rt_quantiles": {},
                "area_bins": [],
                "total_area": 0.0,
                "area_quantiles": {},
            },
            "reasoning_hints": {"expert_priors": EXPERT_PRIORS},
        }
    payload["property_feature_profile"] = build_property_feature_profile(payload)
    return payload, "gc_feature_summary_fallback"


def _peak_area_features_from_external_source(source_file: Any) -> dict[str, Any] | None:
    if not source_file:
        return None
    path = Path(str(source_file))
    if not path.exists() or path.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
        return None
    try:
        df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    except Exception:
        return None
    rename: dict[str, str] = {}
    for col in df.columns:
        lowered = str(col).strip().lower().replace(" ", "_")
        if lowered in {"apex_rt", "rt", "retention_time"}:
            rename[col] = "Apex_RT"
        elif lowered == "area":
            rename[col] = "Area"
    peak_df = df.rename(columns=rename).copy()
    if "Apex_RT" not in peak_df.columns:
        return None
    if "Area" not in peak_df.columns:
        peak_df["Area"] = 1.0
    peak_df["Apex_RT"] = pd.to_numeric(peak_df["Apex_RT"], errors="coerce")
    peak_df["Area"] = pd.to_numeric(peak_df["Area"], errors="coerce")
    peak_df = peak_df.dropna(subset=["Apex_RT", "Area"]).reset_index(drop=True)
    peak_df = peak_df.loc[~peak_df["Apex_RT"].between(SOLVENT_RT - SOLVENT_WINDOW, SOLVENT_RT + SOLVENT_WINDOW)].reset_index(drop=True)
    if peak_df.empty:
        return None
    return _peak_area_features(peak_df)


def _flatten_feature_vector(features: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    raw_shape = features.get("raw_shape_features", {}) if isinstance(features.get("raw_shape_features"), dict) else {}
    peak_shape = features.get("peak_area_features", {}) if isinstance(features.get("peak_area_features"), dict) else {}

    for idx, value in enumerate(raw_shape.get("raw_bins", []) or []):
        out[f"raw_bin_{idx:02d}"] = float(value)
    for idx, value in enumerate(peak_shape.get("area_bins", []) or []):
        out[f"area_bin_{idx:02d}"] = float(value)

    out["peak_count"] = float(peak_shape.get("peak_count") or 0.0)
    out["total_area_log"] = math.log1p(float(peak_shape.get("total_area") or 0.0))
    out["raw_rt_min"] = float((raw_shape.get("rt_range") or {}).get("min") or 0.0)
    out["raw_rt_max"] = float((raw_shape.get("rt_range") or {}).get("max") or 0.0)
    out["peak_rt_min"] = float((peak_shape.get("rt_range") or {}).get("min") or 0.0)
    out["peak_rt_max"] = float((peak_shape.get("rt_range") or {}).get("max") or 0.0)
    out["raw_q50"] = float((raw_shape.get("intensity_quantiles") or {}).get("q50") or 0.0)
    out["raw_q90"] = float((raw_shape.get("intensity_quantiles") or {}).get("q90") or 0.0)
    out["raw_q99"] = float((raw_shape.get("intensity_quantiles") or {}).get("q99") or 0.0)
    out["area_q50_log"] = math.log1p(float((peak_shape.get("area_quantiles") or {}).get("q50") or 0.0))
    out["area_q90_log"] = math.log1p(float((peak_shape.get("area_quantiles") or {}).get("q90") or 0.0))
    out["area_q99_log"] = math.log1p(float((peak_shape.get("area_quantiles") or {}).get("q99") or 0.0))

    raw_rt_quantiles = raw_shape.get("cumulative_rt_quantiles", {}) or {}
    peak_rt_quantiles = peak_shape.get("cumulative_rt_quantiles", {}) or {}
    for key in ["q10", "q25", "q50", "q75", "q90"]:
        out[f"raw_{key}"] = float(raw_rt_quantiles.get(key) or 0.0)
        out[f"peak_{key}"] = float(peak_rt_quantiles.get(key) or 0.0)

    top_rts = peak_shape.get("top_peak_rts", []) or []
    for idx in range(12):
        out[f"top_rt_{idx:02d}"] = float(top_rts[idx]) if idx < len(top_rts) else 0.0

    anchor_candidates = peak_shape.get("anchor_candidates", {}) if isinstance(peak_shape.get("anchor_candidates"), dict) else {}
    for prefix, rows in anchor_candidates.items():
        rows = rows or []
        for idx in range(4):
            out[f"{prefix}_rt_{idx}"] = float(rows[idx]["rt"]) if idx < len(rows) else 0.0
            out[f"{prefix}_area_log_{idx}"] = math.log1p(float(rows[idx]["area"])) if idx < len(rows) else 0.0

    anchor_pair_summary = peak_shape.get("anchor_pair_summary", {}) if isinstance(peak_shape.get("anchor_pair_summary"), dict) else {}
    for prefix, payload in anchor_pair_summary.items():
        payload = payload or {}
        out[f"{prefix}_gap"] = float(payload.get("pair_gap") or 0.0)
        out[f"{prefix}_ratio"] = float(payload.get("pair_area_ratio") or 0.0)

    return out


def _feature_matrix(features_by_sample: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sample_id, payload in features_by_sample.items():
        row = {"sample_id": sample_id}
        row.update(_flatten_feature_vector(payload))
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample_id").sort_index()


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    centered = df - df.mean(axis=0)
    std = df.std(axis=0, ddof=0).replace(0.0, 1.0)
    return centered / std


def _neighbor_scores(feature_matrix: pd.DataFrame, query_id: str) -> pd.DataFrame:
    z = _zscore(feature_matrix.fillna(0.0))
    query = z.loc[query_id].to_numpy(dtype=float)
    records = []
    for sample_id, row in z.iterrows():
        if sample_id == query_id:
            continue
        distance = float(np.linalg.norm(query - row.to_numpy(dtype=float)))
        records.append({"sample_id": sample_id, "distance": round(distance, 6)})
    return pd.DataFrame(records).sort_values("distance").reset_index(drop=True)


def select_holdout_samples(properties: pd.DataFrame, limit: int) -> list[str]:
    density_cols = [col for col in properties.columns if str(col).startswith(PROPERTY_DENSITY_PREFIX)]
    if not density_cols:
        return properties[PROPERTY_SAMPLE_NUMBER].astype(str).head(limit).tolist()
    density_col = density_cols[0]
    available = properties.loc[properties[density_col].notna(), [PROPERTY_SAMPLE_NUMBER, density_col]].copy()
    if available.empty:
        return []
    available[PROPERTY_SAMPLE_NUMBER] = available[PROPERTY_SAMPLE_NUMBER].astype(str).str.strip()
    available = available.sort_values(density_col).reset_index(drop=True)
    count = min(max(limit, 1), len(available))
    if count == len(available):
        return available[PROPERTY_SAMPLE_NUMBER].tolist()
    indices = np.linspace(0, len(available) - 1, num=count)
    picked: list[str] = []
    used: set[str] = set()
    for idx in indices:
        sample_id = str(available.iloc[int(round(idx))][PROPERTY_SAMPLE_NUMBER]).strip()
        if sample_id not in used:
            used.add(sample_id)
            picked.append(sample_id)
    if len(picked) < count:
        for sample_id in available[PROPERTY_SAMPLE_NUMBER].tolist():
            if sample_id in used:
                continue
            used.add(sample_id)
            picked.append(sample_id)
            if len(picked) >= count:
                break
    return picked


def _weighted_property_summary(
    neighbor_df: pd.DataFrame,
    properties: pd.DataFrame,
    property_col: str,
) -> dict[str, Any] | None:
    prop_lookup = properties.set_index(PROPERTY_SAMPLE_NUMBER)
    rows = []
    for row in neighbor_df.itertuples():
        sample_id = str(row.sample_id)
        if sample_id not in prop_lookup.index:
            continue
        value = pd.to_numeric(prop_lookup.at[sample_id, property_col], errors="coerce")
        if pd.isna(value):
            continue
        rows.append({"sample_id": sample_id, "distance": float(row.distance), "value": float(value)})
    if not rows:
        return None
    values = np.asarray([item["value"] for item in rows], dtype=float)
    distances = np.asarray([item["distance"] for item in rows], dtype=float)
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights = weights / weights.sum()
    return {
        "weighted_estimate": round(float(np.dot(values, weights)), 4),
        "neighbor_median": round(float(np.median(values)), 4),
        "neighbor_mean": round(float(np.mean(values)), 4),
        "neighbor_min": round(float(np.min(values)), 4),
        "neighbor_max": round(float(np.max(values)), 4),
        "neighbor_count": int(len(values)),
        "supporting_samples": [item["sample_id"] for item in rows],
    }


def _infer_from_neighbors(neighbor_df: pd.DataFrame, properties: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for property_col in _property_columns(properties):
        summary = _weighted_property_summary(neighbor_df, properties, property_col)
        if summary is not None:
            result[property_col] = summary
    return result


def _evaluation_summary(results: list[dict[str, Any]], property_columns: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for property_col in property_columns:
        errors = []
        for result in results:
            inferred = ((result.get("inferred_properties") or {}).get(property_col) or {}).get("weighted_estimate")
            actual = (result.get("actual_properties") or {}).get(property_col)
            if inferred is None or actual is None or pd.isna(actual):
                continue
            errors.append(abs(float(inferred) - float(actual)))
        metrics[property_col] = {
            "evaluated_samples": len(errors),
            "mae": round(float(np.mean(errors)), 4) if errors else None,
        }
    return metrics


def _reasoning_case(
    sample_id: str,
    features: dict[str, Any],
    neighbors: list[dict[str, Any]],
    inferred: dict[str, Any],
    actual: dict[str, Any] | None,
) -> dict[str, Any]:
    peak_shape = features.get("peak_area_features", {})
    return {
        "sample_id": sample_id,
        "task": "infer_crude_properties_without_supervised_training",
        "query_features": {
            "raw_shape_features": features.get("raw_shape_features"),
            "peak_area_features": {
                "peak_count": peak_shape.get("peak_count"),
                "rt_range": peak_shape.get("rt_range"),
                "top_peak_rts": peak_shape.get("top_peak_rts"),
                "cumulative_rt_quantiles": peak_shape.get("cumulative_rt_quantiles"),
                "anchor_candidates": peak_shape.get("anchor_candidates"),
                "anchor_pair_summary": peak_shape.get("anchor_pair_summary"),
            },
            "property_feature_profile": features.get("property_feature_profile") or build_property_feature_profile(features),
        },
        "retrieved_neighbors": neighbors,
        "neighbor_property_summary": inferred,
        "expert_priors": EXPERT_PRIORS,
        "ground_truth_hidden_for_llm": True,
        "ground_truth_for_eval": actual or {},
    }


def _standardize_reference_name(name: Any) -> str:
    raw = str(name or "").strip()
    return REFERENCE_NAME_ALIASES.get(raw, raw)


def _load_reference_template(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(col).strip() for col in df.columns]
    rename: dict[str, str] = {}
    for col in df.columns:
        lowered = col.lower()
        if lowered == "name":
            rename[col] = "Name"
        elif lowered == "apex rt":
            rename[col] = "Apex_RT"
        elif lowered == "area":
            rename[col] = "Area"
    df = df.rename(columns=rename).copy()
    if "Name" not in df.columns or "Apex_RT" not in df.columns:
        raise ValueError("Reference template missing Name/Apex RT columns")
    df["Name"] = df["Name"].map(_standardize_reference_name)
    df["Apex_RT"] = pd.to_numeric(df["Apex_RT"], errors="coerce")
    if "Area" in df.columns:
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    return df.dropna(subset=["Apex_RT"]).reset_index(drop=True)


def _reference_anchor_summary(reference_df: pd.DataFrame) -> dict[str, Any]:
    mapping = {
        "c17_pr": ("n-C17", "Pr"),
        "c18_ph": ("n-C18", "Ph"),
    }
    out: dict[str, Any] = {}
    for key, (left_name, right_name) in mapping.items():
        subset = reference_df.loc[reference_df["Name"].isin([left_name, right_name]), ["Name", "Apex_RT"]].copy()
        rows = []
        for name in [left_name, right_name]:
            hit = subset.loc[subset["Name"] == name]
            if not hit.empty:
                rows.append({"name": name, "rt": round(float(hit.iloc[0]["Apex_RT"]), 4)})
        if len(rows) == 2:
            out[key] = {
                "available": True,
                "pair_rts": [rows[0]["rt"], rows[1]["rt"]],
                "pair_gap": round(rows[1]["rt"] - rows[0]["rt"], 4),
                "members": rows,
            }
        else:
            out[key] = {
                "available": False,
                "pair_rts": [row["rt"] for row in rows],
                "pair_gap": None,
                "members": rows,
            }
    return out


def _estimate_rt_shift(features: dict[str, Any], reference_df: pd.DataFrame) -> dict[str, Any]:
    sample_summary = ((features.get("peak_area_features") or {}).get("anchor_pair_summary") or {})
    reference_summary = _reference_anchor_summary(reference_df)
    shifts = []
    detail = []
    for anchor_name in ["c17_pr", "c18_ph"]:
        sample_anchor = sample_summary.get(anchor_name) or {}
        ref_anchor = reference_summary.get(anchor_name) or {}
        sample_rts = sample_anchor.get("pair_rts") or []
        ref_rts = ref_anchor.get("pair_rts") or []
        if len(sample_rts) < 2 or len(ref_rts) < 2:
            continue
        pair_shifts = [float(sample_rts[i]) - float(ref_rts[i]) for i in range(2)]
        mean_shift = float(sum(pair_shifts) / len(pair_shifts))
        shifts.append(mean_shift)
        detail.append(
            {
                "anchor_region": anchor_name,
                "sample_pair_rts": sample_rts,
                "reference_pair_rts": ref_rts,
                "member_shifts": [_round_or_none(v) for v in pair_shifts],
                "mean_shift": _round_or_none(mean_shift),
            }
        )
    estimated_shift = float(sum(shifts) / len(shifts)) if shifts else 0.0
    return {
        "estimated_global_shift": round(estimated_shift, 4),
        "anchor_evidence": detail,
        "reference_anchor_summary": reference_summary,
    }


def _rt_tolerance(reference_rt: float) -> float:
    if reference_rt < 8.0:
        return 0.35
    if reference_rt < 20.0:
        return 0.45
    if reference_rt < 35.0:
        return 0.55
    return 0.8


def _build_peak_alignment_candidates(
    peak_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    estimated_shift: float,
    max_sample_peaks: int = 18,
) -> list[dict[str, Any]]:
    sample_peaks = peak_df.sort_values("Area", ascending=False).head(max_sample_peaks).copy()
    candidates: list[dict[str, Any]] = []
    for row in sample_peaks.itertuples():
        sample_rt = float(row.Apex_RT)
        shifted_reference_rt = sample_rt - estimated_shift
        subset = reference_df.copy()
        subset["delta"] = (subset["Apex_RT"] - shifted_reference_rt).abs()
        subset["tolerance"] = subset["Apex_RT"].map(_rt_tolerance)
        subset = subset.loc[subset["delta"] <= subset["tolerance"]].sort_values(["delta", "Apex_RT"]).head(6)
        match_rows = [
            {
                "reference_name": str(hit.Name),
                "reference_rt": round(float(hit.Apex_RT), 4),
                "rt_delta_after_shift": round(float(hit.delta), 4),
            }
            for hit in subset.itertuples()
        ]
        candidates.append(
            {
                "sample_rt": round(sample_rt, 4),
                "sample_area": round(float(row.Area), 4),
                "shift_adjusted_rt": round(shifted_reference_rt, 4),
                "candidate_matches": match_rows,
            }
        )
    return candidates


def _build_property_reasoning_prompt(case: dict[str, Any]) -> str:
    prompt_case = json.loads(json.dumps(case, ensure_ascii=False, default=str))
    prompt_case.pop("ground_truth_for_eval", None)
    prompt_case.pop("llm_prompt", None)
    prompt_case["ground_truth_hidden_for_llm"] = True
    return (
        "You are a constrained evidence-based crude-oil property inference agent.\n"
        "You must not behave as a free numerical predictor and must not use any supervised regression model.\n"
        "Use only the structured evidence below.\n\n"
        "Requirements:\n"
        "- Treat neighbor_property_summary.weighted_estimate as the baseline estimate for each property.\n"
        "- Use query_features.property_feature_profile to decide whether the baseline is supported, weakened, or should be adjusted.\n"
        "- For each property, explicitly list relevant_features, direction, evidence_strength, and correction_decision.\n"
        "- Allowed correction_decision values: keep_baseline, small_adjustment, range_only.\n"
        "- Do not make a large numerical correction unless multiple property-oriented features give strong and consistent evidence.\n"
        "- If evidence is weak or conflicting, keep the baseline or provide a range rather than forcing a point correction.\n"
        "- Biomarker/anchor ratios are secondary evidence for density and BP; do not let them override light/mid/heavy distribution features.\n"
        "- return JSON only\n\n"
        "JSON schema:\n"
        "{\n"
        '  "density_inference": {"baseline": number|null, "estimate": number|null, "range": [number, number]|null, "correction_decision": "keep_baseline|small_adjustment|range_only", "relevant_features": [{"feature": string, "direction": string, "evidence_strength": string, "effect_on_property": string}], "confidence": "low|moderate|high", "reasoning": [string]},\n'
        '  "bp_inference": {"BP_10": {"baseline": number|null, "estimate": number|null, "range": [number, number]|null, "correction_decision": "keep_baseline|small_adjustment|range_only", "relevant_features": [...], "confidence": "low|moderate|high", "reasoning": [string]}, "BP_50": {...}, "BP_90": {...}},\n'
        '  "overall_summary": string,\n'
        '  "uncertainty_notes": [string]\n'
        "}\n\n"
        f"Case:\n{json.dumps(prompt_case, ensure_ascii=False, indent=2)}"
    )


def _build_peak_match_prompt(case: dict[str, Any]) -> str:
    return (
        "You are performing crude-oil GC peak reasoning under retention-time drift.\n"
        "The historical reference template remains an important chemical reference and must not be discarded.\n"
        "Do not require exact template identity when retention-time migration is evident.\n"
        "Use the anchor-pair evidence near C17/Pr and C18/Ph to calibrate the sample against the template, then reason outward to other regularly recurring n-alkane peaks.\n"
        "Use the template heavily for light-end peaks and for the overall n-alkane series, but do not force matches when evidence is weak or locally inconsistent.\n"
        "If the chromatogram suggests nonlinear or region-specific drift, say so explicitly.\n\n"
        "Return JSON only with schema:\n"
        "{\n"
        '  "estimated_shift_assessment": {"accept_shift": true|false, "recommended_shift": number|null, "reasoning": [string]},\n'
        '  "anchor_assignments": [{"region": string, "assigned_pair": [string, string]|null, "confidence": "low|moderate|high", "reasoning": string}],\n'
        '  "peak_assignments": [{"sample_rt": number, "assigned_name": string|null, "confidence": "low|moderate|high", "reasoning": string}],\n'
        '  "notes": [string]\n'
        "}\n\n"
        f"Case:\n{json.dumps(case, ensure_ascii=False, indent=2)}"
    )


def _extract_json_from_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _build_peak_match_case(
    sample_id: str,
    peak_df: pd.DataFrame,
    features: dict[str, Any],
    reference_df: pd.DataFrame,
) -> dict[str, Any]:
    shift_info = _estimate_rt_shift(features, reference_df)
    candidate_matches = _build_peak_alignment_candidates(
        peak_df,
        reference_df,
        estimated_shift=float(shift_info["estimated_global_shift"]),
    )
    return {
        "sample_id": sample_id,
        "task": "anchor_assisted_loose_peak_matching",
        "reasoning_mode": "llm_under_rt_drift",
        "expert_priors": EXPERT_PRIORS,
        "matching_objective": {
            "preserve_template_as_reference": True,
            "allow_rt_migration": True,
            "focus": [
                "anchor-assisted calibration",
                "template-guided light-end matching",
                "template-guided n-alkane sequence matching",
                "migration detection",
                "uncertainty explanation",
            ],
        },
        "sample_anchor_summary": ((features.get("peak_area_features") or {}).get("anchor_pair_summary") or {}),
        "reference_anchor_summary": shift_info["reference_anchor_summary"],
        "estimated_shift": shift_info,
        "candidate_peak_matches": candidate_matches,
    }


def ingest_database(source_dir: Path, out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    normalized_raw_dir = out_root / "normalized" / "chromatogram"
    normalized_peak_dir = out_root / "normalized" / "peak_table"
    normalized_prop_dir = out_root / "normalized" / "properties"
    features_dir = out_root / "features"
    registry_dir = out_root / "registry"
    for folder in [normalized_raw_dir, normalized_peak_dir, normalized_prop_dir, features_dir, registry_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    property_path = source_dir / "property.xlsx"
    properties = load_property_table(property_path)
    workbooks = sorted(
        [path for path in source_dir.glob("*.xlsx") if path.name.lower() != "property.xlsx"],
        key=_sample_sort_key,
    )

    registry_rows = []
    features_by_sample: dict[str, dict[str, Any]] = {}
    for workbook in workbooks:
        sample_id = _sample_id_from_filename(workbook)
        raw_df = load_raw_sheet(workbook)
        peak_df = load_peak_sheet(workbook)
        prop_row_df = properties.loc[properties[PROPERTY_SAMPLE_NUMBER] == sample_id]
        property_row = prop_row_df.iloc[0].to_dict() if not prop_row_df.empty else None

        raw_out = normalized_raw_dir / f"{sample_id}.raw.csv"
        peak_out = normalized_peak_dir / f"{sample_id}.peak.csv"
        prop_out = normalized_prop_dir / f"{sample_id}.property.json"
        feature_out = features_dir / f"{sample_id}.features.json"

        raw_df.to_csv(raw_out, index=False)
        peak_df.to_csv(peak_out, index=False)
        if property_row is not None:
            prop_out.write_text(json.dumps(property_row, ensure_ascii=False, indent=2), encoding="utf-8")

        assets = SampleAssets(
            sample_id=sample_id,
            workbook_path=workbook,
            raw_df=raw_df,
            peak_df=peak_df,
            property_row=property_row,
        )
        features = build_sample_features(assets)
        feature_out.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
        features_by_sample[sample_id] = features

        registry_rows.append(
            {
                "sample_id": sample_id,
                "workbook_file": workbook.name,
                "sample_name": sample_id,
                "origin_basin": "",
                "field": "",
                "crude_type": "",
                "has_chromatogram_raw": True,
                "has_peak_table_integrated": True,
                "has_properties_metadata": property_row is not None,
                "feature_json": str(feature_out),
                "status": "featured",
                "notes": "",
            }
        )

    pd.DataFrame(registry_rows).to_csv(registry_dir / "sample_registry.csv", index=False)
    return {
        "sample_count": len(workbooks),
        "property_row_count": int(len(properties)),
        "feature_files": len(features_by_sample),
        "property_columns": _property_columns(properties),
    }


def run_leave_one_out(source_dir: Path, out_root: Path, *, k: int, test_limit: int) -> dict[str, Any]:
    ingest_summary = ingest_database(source_dir, out_root)
    properties = load_property_table(source_dir / "property.xlsx")
    property_columns = _property_columns(properties)
    reference_df = _load_reference_template(TEMPLATE_REFERENCE_PATH)

    features_dir = out_root / "features"
    features_by_sample = {
        fp.name.replace(".features.json", ""): json.loads(fp.read_text(encoding="utf-8"))
        for fp in features_dir.glob("*.features.json")
    }
    peak_table_dir = out_root / "normalized" / "peak_table"
    matrix = _feature_matrix(features_by_sample)
    test_samples = select_holdout_samples(properties, limit=test_limit)

    property_cases = []
    peak_match_cases = []
    results = []
    prop_lookup = properties.set_index(PROPERTY_SAMPLE_NUMBER)
    for sample_id in test_samples:
        if sample_id not in matrix.index:
            continue
        neighbors_df = _neighbor_scores(matrix, sample_id).head(k).copy()
        inferred = _infer_from_neighbors(neighbors_df, properties)
        actual = prop_lookup.loc[sample_id].to_dict() if sample_id in prop_lookup.index else {}
        neighbor_records = neighbors_df.to_dict(orient="records")
        property_case = _reasoning_case(sample_id, features_by_sample[sample_id], neighbor_records, inferred, actual)
        property_case["llm_prompt"] = _build_property_reasoning_prompt(property_case)
        property_cases.append(property_case)

        peak_path = peak_table_dir / f"{sample_id}.peak.csv"
        if peak_path.exists():
            peak_df = pd.read_csv(peak_path)
            peak_case = _build_peak_match_case(sample_id, peak_df, features_by_sample[sample_id], reference_df)
            peak_case["llm_prompt"] = _build_peak_match_prompt(peak_case)
            peak_match_cases.append(peak_case)

        results.append(
            {
                "sample_id": sample_id,
                "self_excluded": sample_id not in {row["sample_id"] for row in neighbor_records},
                "neighbors": neighbor_records,
                "inferred_properties": inferred,
                "actual_properties": actual,
            }
        )

    summary = {
        "ingest_summary": ingest_summary,
        "selected_holdout_samples": test_samples,
        "metrics": _evaluation_summary(results, property_columns),
    }

    eval_dir = out_root / "indexes"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_file = eval_dir / "leave_one_out_results.json"
    cases_file = eval_dir / "llm_reasoning_cases.jsonl"
    property_cases_file = eval_dir / "llm_property_reasoning_cases.jsonl"
    peak_cases_file = eval_dir / "llm_peak_match_cases.jsonl"
    summary_file = eval_dir / "pilot_summary.json"
    results_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with cases_file.open("w", encoding="utf-8") as handle:
        for case in property_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    with property_cases_file.open("w", encoding="utf-8") as handle:
        for case in property_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    with peak_cases_file.open("w", encoding="utf-8") as handle:
        for case in peak_match_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "tested_samples": len(results),
        "k_neighbors": k,
        "selected_holdout_samples": test_samples,
        "result_file": str(results_file),
        "llm_cases_file": str(cases_file),
        "llm_property_cases_file": str(property_cases_file),
        "llm_peak_match_cases_file": str(peak_cases_file),
        "summary_file": str(summary_file),
    }


def _run_llm_over_cases(
    *,
    cases_file: Path,
    out_file: Path,
    max_cases: int | None = None,
) -> dict[str, Any]:
    from app.llm.deepseek_client import DeepSeekClient, DeepSeekClientError

    client = DeepSeekClient()
    outputs = []
    lines = [line for line in cases_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_cases is not None:
        lines = lines[: max(0, int(max_cases))]
    for raw_line in lines:
        case = json.loads(raw_line)
        task = str(case.get("task") or "")
        if task in {"infer_crude_properties_without_supervised_training", "infer_new_crude_properties_from_historical_database"}:
            prompt = _build_property_reasoning_prompt(case)
        elif task == "anchor_assisted_loose_peak_matching":
            prompt = _build_peak_match_prompt(case)
        else:
            prompt = str(case.get("llm_prompt") or "").strip()
        sample_id = str(case.get("sample_id") or "")
        try:
            raw_answer = client.reason(prompt)
            parsed = _extract_json_from_text(raw_answer)
            outputs.append(
                {
                    "sample_id": sample_id,
                    "status": "success",
                    "raw_answer": raw_answer,
                    "parsed_answer": parsed,
                }
            )
        except DeepSeekClientError as exc:
            outputs.append(
                {
                    "sample_id": sample_id,
                    "status": "error",
                    "error": str(exc),
                    "raw_answer": None,
                    "parsed_answer": {},
                }
            )
    out_file.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "case_count": len(outputs),
        "output_file": str(out_file),
    }


def run_llm_property_reasoning(out_root: Path, *, max_cases: int | None = None) -> dict[str, Any]:
    indexes_dir = out_root / "indexes"
    cases_file = indexes_dir / "llm_property_reasoning_cases.jsonl"
    if not cases_file.exists():
        raise FileNotFoundError(f"Property reasoning cases not found: {cases_file}")
    return _run_llm_over_cases(
        cases_file=cases_file,
        out_file=indexes_dir / "llm_property_reasoning_outputs.json",
        max_cases=max_cases,
    )


def run_llm_peak_match_reasoning(out_root: Path, *, max_cases: int | None = None) -> dict[str, Any]:
    indexes_dir = out_root / "indexes"
    cases_file = indexes_dir / "llm_peak_match_cases.jsonl"
    if not cases_file.exists():
        raise FileNotFoundError(f"Peak-match cases not found: {cases_file}")
    return _run_llm_over_cases(
        cases_file=cases_file,
        out_file=indexes_dir / "llm_peak_match_outputs.json",
        max_cases=max_cases,
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Historical crude-oil DB ingestion and leave-one-out inference.")
    ap.add_argument("--source-dir", type=Path, required=True, help="Directory containing sample workbooks and property.xlsx")
    ap.add_argument("--out-root", type=Path, default=Path("data/crude_oil_db"), help="Workspace root for normalized outputs.")
    ap.add_argument(
        "--mode",
        choices=["ingest", "leave_one_out", "llm_property_reasoning", "llm_peak_match_reasoning"],
        default="leave_one_out",
    )
    ap.add_argument("--k", type=int, default=5, help="Number of nearest historical samples to use.")
    ap.add_argument("--test-limit", type=int, default=8, help="Number of holdout samples for the pilot test.")
    ap.add_argument("--max-cases", type=int, default=None, help="Optional limit when running LLM reasoning over prepared cases.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "ingest":
        result = ingest_database(args.source_dir, args.out_root)
    elif args.mode == "leave_one_out":
        result = run_leave_one_out(
            args.source_dir,
            args.out_root,
            k=max(1, int(args.k)),
            test_limit=max(1, int(args.test_limit)),
        )
    elif args.mode == "llm_property_reasoning":
        result = run_llm_property_reasoning(args.out_root, max_cases=args.max_cases)
    else:
        result = run_llm_peak_match_reasoning(args.out_root, max_cases=args.max_cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

