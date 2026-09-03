from __future__ import annotations

import os
import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tools.experimental_db.evidence_retriever import _profile_from_sample_record
from app.tools.experimental_db.sample_database import infer_template_definition, workbook_to_payload


def _default_db_root() -> Path:
    preferred = Path(os.getenv("PETRO_FRACTION_DB_ROOT", "data/experimental_db/petroleum_fraction_dataset"))
    if preferred.exists():
        return preferred
    legacy_name = "v" + "go-database"
    legacy = Path(os.getenv("PETRO_FRACTION_LEGACY_ROOT", "data/private")) / legacy_name
    return legacy if legacy.exists() else preferred


DEFAULT_DB_ROOT = _default_db_root()
DEFAULT_TEMPLATE = Path(os.getenv("PETRO_FRACTION_TEMPLATE", "data/experimental_db/templates/chemanalyst_experimental_sample_template.xlsx"))
DEFAULT_OUTPUT_ROOT = Path("outputs") / "petroleum_fraction_experimental_db" / "feature_models"
DEFAULT_TARGETS = (
    "density_20c",
    "saturates",
    "BP_20",
    "BP_35",
    "BP_50",
    "BP_65",
    "BP_80",
)
SPECTRAL_ANALYSIS_TYPES = ("gc_fid", "ir_spectrum")
SPECTRAL_GRID_POINTS = {
    "gc_fid": 600,
    "ir_spectrum": 700,
}
SPECTRAL_PCA_COMPONENTS = 5
SPECTRAL_PLS_COMPONENTS = 4
PLS_TARGETS = DEFAULT_TARGETS
SCENARIOS = ("analysis_only", "analysis_plus_non_target_bulk")
MIN_CORRELATION_N = 8
MIN_FEATURE_WEIGHT = 0.05


@dataclass
class SampleProfile:
    split: str
    sample_id: str
    raw_sample_id: str
    workbook: str
    channels: list[str]
    features: dict[str, float]
    properties: dict[str, float]
    spectra: dict[str, list[dict[str, Any]]]


def main() -> None:
    args = parse_args()
    db_root = Path(args.db_root)
    template_path = Path(args.template)
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [target for target in args.targets if target]

    if args.reuse_tables_dir:
        print(f"[1/6] Reusing existing tables from {args.reuse_tables_dir}", flush=True)
        feature_matrix, property_table, metadata = load_existing_tables(Path(args.reuse_tables_dir))
    else:
        print(f"[1/6] Loading template: {template_path}", flush=True)
        template_definition = infer_template_definition(template_path)
        print(f"[2/6] Loading petroleum fraction manifest: {db_root}", flush=True)
        manifest = load_manifest(db_root)
        print(f"[3/6] Parsing {len(manifest)} sample workbooks", flush=True)
        profiles = load_profiles(manifest, template_definition=template_definition)
        print("[4/6] Building feature/property tables and spectral PCA/PLS features", flush=True)
        feature_matrix, property_table, metadata = build_tables(profiles)
    feature_matrix = drop_legacy_generic_composition_features(feature_matrix)
    train_ids = metadata.loc[metadata["split"] == "train", "sample_id"].tolist()
    test_ids = metadata.loc[metadata["split"] == "test", "sample_id"].tolist()
    train_features = feature_matrix.loc[train_ids].copy()
    test_features = feature_matrix.loc[test_ids].copy()
    train_properties = property_table.loc[train_ids].copy()
    test_properties = property_table.loc[test_ids].copy()

    print("[5/6] Computing feature-target weights and test-set neighbor evidence", flush=True)
    correlations, weights_by_target = build_feature_target_weights(train_features, train_properties, targets)
    redundancy = feature_feature_redundancy(train_features, correlations)
    predictions, neighbors, prompts = evaluate_test_set(
        train_features=train_features,
        test_features=test_features,
        train_properties=train_properties,
        test_properties=test_properties,
        metadata=metadata,
        weights_by_target=weights_by_target,
        targets=targets,
        top_k=args.top_k,
    )
    metrics = summarize_metrics(predictions)

    print(f"[6/6] Writing outputs to {out_dir}", flush=True)
    write_outputs(
        out_dir=out_dir,
        feature_matrix=feature_matrix,
        property_table=property_table,
        metadata=metadata,
        correlations=correlations,
        redundancy=redundancy,
        predictions=predictions,
        neighbors=neighbors,
        prompts=prompts,
        metrics=metrics,
        targets=targets,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "status": "success",
                "output_dir": str(out_dir.resolve()),
                "train_samples": len(train_ids),
                "test_samples": len(test_ids),
                "targets": targets,
                "prediction_rows": int(len(predictions)),
                "metric_rows": int(len(metrics)),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate petroleum fraction Historical Experimental DB feature weighting and inference.")
    parser.add_argument("--db-root", default=str(DEFAULT_DB_ROOT), help="petroleum fraction workbook database root with manifest.csv.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="Experimental DB workbook template used to parse sheets.")
    parser.add_argument("--output-dir", default="", help="Centralized output directory. Defaults to outputs/petroleum_fraction_experimental_db/feature_models/<timestamp>.")
    parser.add_argument("--reuse-tables-dir", default="", help="Reuse feature_matrix_all.csv, property_table_all.csv, and sample_metadata.csv from a previous run.")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k historical neighbors.")
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS), help="Target bulk properties to evaluate.")
    return parser.parse_args()


def load_manifest(db_root: Path) -> pd.DataFrame:
    manifest_path = db_root / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        rows = []
        for split in ("train", "test"):
            for workbook in sorted((db_root / split).glob("*.xlsx")):
                rows.append({"sample_id": workbook.stem.split("__", 1)[0], "raw_sample_id": workbook.stem, "file": str(workbook), "split": split})
        manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No workbooks found under {db_root}")
    required = {"sample_id", "raw_sample_id", "file", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest.csv missing required columns: {sorted(missing)}")
    return manifest


def load_existing_tables(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_path = source_dir / "feature_matrix_all.csv"
    property_path = source_dir / "property_table_all.csv"
    metadata_path = source_dir / "sample_metadata.csv"
    for path in (feature_path, property_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)
    feature_matrix = pd.read_csv(feature_path, index_col=0)
    property_table = pd.read_csv(property_path, index_col=0)
    metadata = pd.read_csv(metadata_path)
    if "sample_id" not in metadata.columns:
        raise ValueError(f"{metadata_path} must contain sample_id")
    metadata = metadata.set_index("sample_id", drop=False)
    return feature_matrix, property_table, metadata


def drop_legacy_generic_composition_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    value_tokens = (
        "content",
        "value",
        "mass_percent",
        "wt_percent",
        "area_percent",
        "area",
        "half_up_sum",
        "relative_abundance",
        "abundance",
    )
    pattern = re.compile(rf"^(hydrocarbon_types|molecular_composition):numeric:({'|'.join(value_tokens)})$")
    drop_cols = [column for column in feature_matrix.columns if pattern.match(str(column))]
    if drop_cols:
        cleaned = feature_matrix.drop(columns=drop_cols)
        cleaned.attrs.update(feature_matrix.attrs)
        return cleaned
    return feature_matrix


def load_profiles(manifest: pd.DataFrame, *, template_definition: dict[str, Any]) -> list[SampleProfile]:
    profiles: list[SampleProfile] = []
    rows = manifest.sort_values(["split", "sample_id"]).to_dict(orient="records")
    for idx, row in enumerate(rows, start=1):
        workbook = Path(str(row["file"]))
        print(f"  - [{idx:02d}/{len(rows):02d}] {workbook.name}", flush=True)
        payload = workbook_to_payload(workbook, template_definition=template_definition)
        record = {
            "sample": payload.get("sample_information") or {},
            "bulk_properties": payload.get("bulk_properties") or [],
            "analyses": payload.get("analyses") or [],
        }
        profile = _profile_from_sample_record(record)
        sample_id = str(row["sample_id"])
        raw_sample_id = str(row.get("raw_sample_id") or sample_id)
        profiles.append(
            SampleProfile(
                split=str(row["split"]),
                sample_id=sample_id,
                raw_sample_id=raw_sample_id,
                workbook=str(workbook),
                channels=list(profile.get("available_channels") or []),
                features={str(k): float(v) for k, v in (profile.get("feature_vector") or {}).items() if is_finite_number(v)},
                properties={str(k): float(v) for k, v in (profile.get("bulk_properties") or {}).items() if is_finite_number(v)},
                spectra=spectral_rows_from_payload(payload),
            )
        )
    return profiles


def build_tables(profiles: list[SampleProfile]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_matrix = pd.DataFrame.from_dict({item.sample_id: item.features for item in profiles}, orient="index").sort_index()
    property_table = pd.DataFrame.from_dict({item.sample_id: item.properties for item in profiles}, orient="index").sort_index()
    feature_matrix = replace_curve_statistics_with_spectral_latent_features(feature_matrix, property_table, profiles)
    metadata = pd.DataFrame(
        [
            {
                "sample_id": item.sample_id,
                "raw_sample_id": item.raw_sample_id,
                "split": item.split,
                "workbook": item.workbook,
                "channels": ", ".join(item.channels),
                "feature_count": len(item.features),
                "property_count": len(item.properties),
            }
            for item in profiles
        ]
    ).set_index("sample_id", drop=False)
    return feature_matrix, property_table, metadata


def spectral_rows_from_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for analysis in payload.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        analysis_type = str(analysis.get("analysis_type") or analysis.get("type") or "").strip().lower()
        if analysis_type in SPECTRAL_ANALYSIS_TYPES:
            rows = analysis.get("rows") or analysis.get("data") or []
            if isinstance(rows, list) and rows:
                out[analysis_type] = rows
    return out


def replace_curve_statistics_with_spectral_latent_features(
    feature_matrix: pd.DataFrame,
    property_table: pd.DataFrame,
    profiles: list[SampleProfile],
) -> pd.DataFrame:
    cleaned = drop_curve_stat_features(feature_matrix)
    latent = build_spectral_latent_features(profiles, property_table)
    if latent.empty:
        return cleaned
    combined = cleaned.join(latent, how="left")
    combined.attrs["spectral_model_manifest"] = latent.attrs.get("spectral_model_manifest", [])
    combined.attrs["spectral_model_artifacts"] = latent.attrs.get("spectral_model_artifacts", [])
    combined.attrs["spectral_latent_feature_columns"] = list(latent.columns)
    return combined


def drop_curve_stat_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    prefixes = tuple(f"{analysis_type}:" for analysis_type in SPECTRAL_ANALYSIS_TYPES)
    drop_cols = [column for column in feature_matrix.columns if str(column).startswith(prefixes)]
    if not drop_cols:
        return feature_matrix
    cleaned = feature_matrix.drop(columns=drop_cols)
    cleaned.attrs.update(feature_matrix.attrs)
    return cleaned


def build_spectral_latent_features(profiles: list[SampleProfile], property_table: pd.DataFrame) -> pd.DataFrame:
    sample_ids = [item.sample_id for item in profiles]
    split_by_id = {item.sample_id: item.split for item in profiles}
    train_ids = [sample_id for sample_id in sample_ids if split_by_id.get(sample_id) == "train"]
    feature_blocks: list[pd.DataFrame] = []
    model_manifest: list[dict[str, Any]] = []
    model_artifacts: list[dict[str, Any]] = []

    for analysis_type in SPECTRAL_ANALYSIS_TYPES:
        print(f"    spectral channel: {analysis_type}", flush=True)
        matrix, grid, available_ids = spectral_matrix_for_analysis(analysis_type, profiles)
        if matrix.empty:
            continue
        channel_train_ids = [sample_id for sample_id in train_ids if sample_id in matrix.index]
        if len(channel_train_ids) < MIN_CORRELATION_N:
            continue
        pca_scores, pca_model = fit_transform_pca_features(
            analysis_type=analysis_type,
            matrix=matrix,
            train_ids=channel_train_ids,
            n_components=SPECTRAL_PCA_COMPONENTS,
        )
        feature_blocks.append(pca_scores)
        pca_file = f"{analysis_type}__pca.npz"
        model_manifest.append(
            {
                "analysis_type": analysis_type,
                "model_type": "PCA",
                "artifact_file": pca_file,
                "component_count": int(pca_scores.shape[1]),
                "train_sample_count": len(channel_train_ids),
                "available_sample_count": len(available_ids),
                "grid_min": float(grid[0]) if len(grid) else None,
                "grid_max": float(grid[-1]) if len(grid) else None,
                "grid_points": int(len(grid)),
                "explained_variance_ratio": [round(float(v), 6) for v in pca_model["explained_variance_ratio"]],
            }
        )
        model_artifacts.append(
            {
                "artifact_file": pca_file,
                "arrays": {
                    "grid": grid,
                    "mean": pca_model["mean"],
                    "scale": pca_model["scale"],
                    "loadings": pca_model["loadings"],
                    "explained_variance_ratio": pca_model["explained_variance_ratio"],
                },
            }
        )
        for target in PLS_TARGETS:
            if target not in property_table.columns:
                continue
            y = pd.to_numeric(property_table.loc[channel_train_ids, target], errors="coerce")
            valid_train_ids = [sample_id for sample_id in channel_train_ids if pd.notna(y.loc[sample_id])]
            if len(valid_train_ids) < MIN_CORRELATION_N:
                continue
            pls_scores, pls_meta, pls_model = fit_transform_pls_features(
                analysis_type=analysis_type,
                target=target,
                matrix=matrix,
                property_table=property_table,
                train_ids=valid_train_ids,
                n_components=SPECTRAL_PLS_COMPONENTS,
            )
            if not pls_scores.empty:
                pls_file = f"{analysis_type}__pls__{safe_key(target)}.npz"
                pls_meta["artifact_file"] = pls_file
                feature_blocks.append(pls_scores)
                model_manifest.append(pls_meta)
                model_artifacts.append(
                    {
                        "artifact_file": pls_file,
                        "arrays": {
                            "grid": grid,
                            "mean": pls_model["mean"],
                            "scale": pls_model["scale"],
                            "weights": pls_model["weights"],
                            "loadings": pls_model["loadings"],
                            "q_values": pls_model["q_values"],
                            "y_mean": np.asarray([pls_model["y_mean"]], dtype=float),
                        },
                    }
                )

    if not feature_blocks:
        return pd.DataFrame(index=sample_ids)
    latent = pd.concat(feature_blocks, axis=1).reindex(sample_ids)
    latent.attrs["spectral_model_manifest"] = model_manifest
    latent.attrs["spectral_model_artifacts"] = model_artifacts
    return latent


def spectral_matrix_for_analysis(analysis_type: str, profiles: list[SampleProfile]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    series_by_id: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for profile in profiles:
        rows = profile.spectra.get(analysis_type)
        if not rows:
            continue
        points = spectral_points(rows, analysis_type)
        if points is None:
            continue
        series_by_id[profile.sample_id] = points
    if not series_by_id:
        return pd.DataFrame(), np.asarray([]), []

    mins = [float(x.min()) for x, _ in series_by_id.values() if len(x)]
    maxs = [float(x.max()) for x, _ in series_by_id.values() if len(x)]
    lower = max(mins)
    upper = min(maxs)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        lower = min(mins)
        upper = max(maxs)
    grid = np.linspace(lower, upper, SPECTRAL_GRID_POINTS.get(analysis_type, 600))
    rows = {}
    for sample_id, (x, y) in series_by_id.items():
        interpolated = np.interp(grid, x, y)
        rows[sample_id] = row_normalize_spectrum(interpolated)
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"x_{idx:04d}" for idx in range(len(grid))]), grid, sorted(series_by_id)


def spectral_points(rows: list[dict[str, Any]], analysis_type: str) -> tuple[np.ndarray, np.ndarray] | None:
    is_ir = analysis_type.startswith("ir")
    x_keys = ("wave_number_cm-1", "wave_number", "wavenumber", "wn") if is_ir else ("rt", "apex_rt", "time")
    y_keys = ("transmittance", "absorbance", "intensity") if is_ir else ("intensity", "abundance", "area")
    points = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        x = first_numeric(row, x_keys)
        y = first_numeric(row, y_keys)
        if x is not None and y is not None:
            points.append((x, y))
    if len(points) < 8:
        return None
    points.sort(key=lambda item: item[0])
    grouped: dict[float, list[float]] = {}
    for x, y in points:
        grouped.setdefault(float(x), []).append(float(y))
    xs = np.asarray(sorted(grouped), dtype=float)
    ys = np.asarray([float(np.mean(grouped[x])) for x in xs], dtype=float)
    if not is_ir:
        ys = np.abs(ys)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    if len(xs) < 8:
        return None
    return xs, ys


def row_normalize_spectrum(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    median = np.nanmedian(arr)
    arr = np.where(np.isnan(arr), median, arr)
    mean = float(arr.mean())
    std = float(arr.std())
    if not math.isfinite(std) or std < 1e-12:
        return arr - mean
    return (arr - mean) / std


def fit_transform_pca_features(
    *,
    analysis_type: str,
    matrix: pd.DataFrame,
    train_ids: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = matrix.loc[train_ids]
    scaled, mean, scale = standardize_by_train(matrix, train)
    train_scaled = scaled.loc[train_ids].to_numpy(dtype=float)
    _, singular_values, vt = np.linalg.svd(train_scaled, full_matrices=False)
    component_count = max(1, min(n_components, vt.shape[0], len(train_ids) - 1))
    loadings = vt[:component_count]
    scores = scaled.to_numpy(dtype=float) @ loadings.T
    columns = [f"{analysis_type}:pca:pc{idx}" for idx in range(1, component_count + 1)]
    score_frame = pd.DataFrame(scores, index=matrix.index, columns=columns)
    explained = singular_values**2
    ratio = explained / explained.sum() if explained.sum() > 0 else np.zeros_like(explained)
    return score_frame, {
        "mean": mean,
        "scale": scale,
        "loadings": loadings,
        "explained_variance_ratio": ratio[:component_count],
    }


def fit_transform_pls_features(
    *,
    analysis_type: str,
    target: str,
    matrix: pd.DataFrame,
    property_table: pd.DataFrame,
    train_ids: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train = matrix.loc[train_ids]
    scaled, mean, scale = standardize_by_train(matrix, train)
    x_train = scaled.loc[train_ids].to_numpy(dtype=float)
    y_train = pd.to_numeric(property_table.loc[train_ids, target], errors="coerce").to_numpy(dtype=float)
    y_mean = float(np.nanmean(y_train))
    y_centered = y_train - y_mean
    component_count = max(1, min(n_components, len(train_ids) - 2, x_train.shape[1]))
    weights, loadings, q_values = fit_pls1_nipals(x_train, y_centered, component_count)
    if weights.size == 0:
        return pd.DataFrame(), {}, {}
    all_scores = transform_pls_scores(scaled.to_numpy(dtype=float), weights, loadings)
    columns = [f"{analysis_type}:pls:{safe_key(target)}:lv{idx}" for idx in range(1, all_scores.shape[1] + 1)]
    score_frame = pd.DataFrame(all_scores, index=matrix.index, columns=columns)
    meta = {
        "analysis_type": analysis_type,
        "model_type": "PLS",
        "target_property": target,
        "component_count": int(score_frame.shape[1]),
        "train_sample_count": len(train_ids),
        "y_mean": round(y_mean, 6),
        "q_values": [round(float(v), 6) for v in q_values],
    }
    model = {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "loadings": loadings,
        "q_values": q_values,
        "y_mean": y_mean,
    }
    return score_frame, meta, model


def standardize_by_train(matrix: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0).to_numpy(dtype=float)
    scale = train.std(axis=0, ddof=0).to_numpy(dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    values = matrix.to_numpy(dtype=float)
    scaled = (values - mean) / scale
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return pd.DataFrame(scaled, index=matrix.index, columns=matrix.columns), mean, scale


def fit_pls1_nipals(x: np.ndarray, y: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = np.asarray(x, dtype=float).copy()
    f = np.asarray(y, dtype=float).copy()
    weights = []
    loadings = []
    q_values = []
    for _ in range(n_components):
        w = e.T @ f
        norm = float(np.linalg.norm(w))
        if not math.isfinite(norm) or norm < 1e-12:
            break
        w = w / norm
        t = e @ w
        denom = float(t.T @ t)
        if not math.isfinite(denom) or denom < 1e-12:
            break
        p = e.T @ t / denom
        q = float(f.T @ t / denom)
        e = e - np.outer(t, p)
        f = f - q * t
        weights.append(w)
        loadings.append(p)
        q_values.append(q)
    if not weights:
        return np.empty((0, 0)), np.empty((0, 0)), np.asarray([])
    return np.vstack(weights), np.vstack(loadings), np.asarray(q_values)


def transform_pls_scores(x: np.ndarray, weights: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    e = np.asarray(x, dtype=float).copy()
    scores = []
    for w, p in zip(weights, loadings, strict=False):
        t = e @ w
        scores.append(t)
        e = e - np.outer(t, p)
    return np.vstack(scores).T if scores else np.empty((x.shape[0], 0))


def first_numeric(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalized = {safe_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(safe_key(key))
        numeric = to_float(value)
        if numeric is not None:
            return numeric
    return None


def build_feature_target_weights(
    train_features: pd.DataFrame,
    train_properties: pd.DataFrame,
    targets: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    weights_by_target: dict[str, dict[str, float]] = {}
    for target in targets:
        if target not in train_properties.columns:
            weights_by_target[target] = {}
            continue
        y = pd.to_numeric(train_properties[target], errors="coerce")
        target_rows: list[dict[str, Any]] = []
        for feature in train_features.columns:
            if feature == f"bulk_property:{safe_key(target)}":
                continue
            x = pd.to_numeric(train_features[feature], errors="coerce")
            valid = x.notna() & y.notna()
            n = int(valid.sum())
            if n < MIN_CORRELATION_N or x[valid].nunique(dropna=True) < 2 or y[valid].nunique(dropna=True) < 2:
                pearson = math.nan
                spearman = math.nan
                abs_corr = 0.0
            else:
                pearson = float(x[valid].corr(y[valid], method="pearson"))
                spearman = float(x[valid].corr(y[valid], method="spearman"))
                corr_values = [abs(value) for value in (pearson, spearman) if is_finite_number(value)]
                abs_corr = float(sum(corr_values) / len(corr_values)) if corr_values else 0.0
            coverage = n / max(1, len(train_features))
            weight = MIN_FEATURE_WEIGHT + abs_corr * coverage
            target_rows.append(
                {
                    "target_property": target,
                    "feature": feature,
                    "feature_channel": feature_channel(feature),
                    "n": n,
                    "coverage": round(coverage, 6),
                    "pearson": round_or_none(pearson),
                    "spearman": round_or_none(spearman),
                    "abs_corr_mean": round(abs_corr, 6),
                    "weight": round(weight, 6),
                }
            )
        rows.extend(target_rows)
        weights_by_target[target] = {row["feature"]: float(row["weight"]) for row in target_rows}
    return pd.DataFrame(rows).sort_values(["target_property", "weight"], ascending=[True, False]), weights_by_target


def feature_feature_redundancy(train_features: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return pd.DataFrame()
    selected = sorted(set(correlations.groupby("target_property").head(40)["feature"]))
    selected = [feature for feature in selected if feature in train_features.columns]
    rows = []
    for i, left in enumerate(selected):
        for right in selected[i + 1 :]:
            x = pd.to_numeric(train_features[left], errors="coerce")
            y = pd.to_numeric(train_features[right], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < MIN_CORRELATION_N:
                continue
            corr = x[valid].corr(y[valid], method="spearman")
            if is_finite_number(corr) and abs(float(corr)) >= 0.9:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "feature_left_channel": feature_channel(left),
                        "feature_right_channel": feature_channel(right),
                        "n": int(valid.sum()),
                        "spearman": round(float(corr), 6),
                        "abs_spearman": round(abs(float(corr)), 6),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False) if rows else pd.DataFrame()


def evaluate_test_set(
    *,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_properties: pd.DataFrame,
    test_properties: pd.DataFrame,
    metadata: pd.DataFrame,
    weights_by_target: dict[str, dict[str, float]],
    targets: list[str],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scales = train_features.apply(feature_scale, axis=0).to_dict()
    prediction_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for sample_id in test_features.index:
        for target in targets:
            actual = value_at(test_properties, sample_id, target)
            if actual is None:
                continue
            for scenario in SCENARIOS:
                query = scenario_features(test_features.loc[sample_id], target=target, scenario=scenario)
                candidates = train_features.apply(lambda row: scenario_features(row, target=target, scenario=scenario), axis=1)
                for method in ("equal_weight", "correlation_weighted"):
                    feature_weights = weights_by_target.get(target, {}) if method == "correlation_weighted" else {}
                    ranked = rank_neighbors(
                        query=query,
                        candidates=candidates,
                        target_values=train_properties.get(target),
                        scales=scales,
                        feature_weights=feature_weights,
                        top_k=top_k,
                    )
                    if ranked.empty:
                        continue
                    estimate = weighted_estimate(ranked)
                    error = estimate - actual if estimate is not None else None
                    relative_error = error / actual * 100.0 if error is not None and actual != 0 else None
                    pred_row = {
                        "sample_id": sample_id,
                        "raw_sample_id": metadata.loc[sample_id, "raw_sample_id"] if sample_id in metadata.index else sample_id,
                        "target_property": target,
                        "scenario": scenario,
                        "method": method,
                        "actual": round(actual, 6),
                        "estimate": round_or_none(estimate),
                        "error": round_or_none(error),
                        "abs_error": round_or_none(abs(error) if error is not None else None),
                        "relative_error_percent": round_or_none(relative_error),
                        "absolute_percentage_error": round_or_none(abs(relative_error) if relative_error is not None else None),
                        "neighbor_count": int(len(ranked)),
                        "nearest_distance": round_or_none(ranked.iloc[0]["distance"]),
                        "historical_range_min": round_or_none(ranked["target_value"].min()),
                        "historical_range_max": round_or_none(ranked["target_value"].max()),
                        "actual_in_neighbor_range": bool(ranked["target_value"].min() <= actual <= ranked["target_value"].max()),
                        "mean_shared_feature_count": round_or_none(ranked["shared_feature_count"].mean()),
                        "top_neighbor": str(ranked.iloc[0]["neighbor_sample_id"]),
                    }
                    prediction_rows.append(pred_row)
                    neighbor_rows.extend(
                        {
                            **item,
                            "query_sample_id": sample_id,
                            "target_property": target,
                            "scenario": scenario,
                            "method": method,
                        }
                        for item in ranked.to_dict(orient="records")
                    )
                    if method == "correlation_weighted":
                        prompt_rows.append(
                            build_prompt_record(
                                sample_id=sample_id,
                                target=target,
                                scenario=scenario,
                                actual=actual,
                                pred_row=pred_row,
                                ranked=ranked,
                                feature_weights=feature_weights,
                            )
                        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(neighbor_rows), pd.DataFrame(prompt_rows)


def rank_neighbors(
    *,
    query: pd.Series,
    candidates: pd.DataFrame,
    target_values: pd.Series | None,
    scales: dict[str, float],
    feature_weights: dict[str, float],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    for sample_id, candidate in candidates.iterrows():
        if target_values is None or sample_id not in target_values.index or pd.isna(target_values.loc[sample_id]):
            continue
        shared = sorted(set(query.dropna().index) & set(candidate.dropna().index))
        if not shared:
            continue
        weighted_squared = 0.0
        weight_sum = 0.0
        top_relevance = []
        for feature in shared:
            q = to_float(query.get(feature))
            c = to_float(candidate.get(feature))
            if q is None or c is None:
                continue
            scale = max(float(scales.get(feature) or 1.0), 1e-9)
            weight = float(feature_weights.get(feature, 1.0))
            delta = (q - c) / scale
            weighted_squared += weight * delta * delta
            weight_sum += weight
            top_relevance.append((feature, weight, abs(delta)))
        if weight_sum <= 0:
            continue
        distance = math.sqrt(weighted_squared / weight_sum) + 1.0 / max(len(shared), 1)
        top_relevance.sort(key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "rank": 0,
                "neighbor_sample_id": sample_id,
                "distance": round(distance, 6),
                "target_value": float(target_values.loc[sample_id]),
                "shared_feature_count": len(shared),
                "shared_feature_examples": "; ".join(shared[:12]),
                "top_weighted_shared_features": "; ".join(f"{name}({weight:.3f})" for name, weight, _ in top_relevance[:8]),
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows).sort_values(["distance", "neighbor_sample_id"]).head(max(1, int(top_k))).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def weighted_estimate(ranked: pd.DataFrame) -> float | None:
    if ranked.empty:
        return None
    distances = pd.to_numeric(ranked["distance"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(ranked["target_value"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(distances) & np.isfinite(values)
    if not valid.any():
        return None
    weights = 1.0 / np.maximum(distances[valid], 1e-6)
    return float(np.sum(values[valid] * weights) / np.sum(weights))


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in predictions.groupby(["target_property", "scenario", "method"]):
        errors = pd.to_numeric(group["error"], errors="coerce").dropna()
        abs_errors = errors.abs()
        relative_errors = pd.to_numeric(group.get("relative_error_percent"), errors="coerce").dropna()
        absolute_percentage_errors = pd.to_numeric(group.get("absolute_percentage_error"), errors="coerce").dropna()
        rows.append(
            {
                "target_property": keys[0],
                "scenario": keys[1],
                "method": keys[2],
                "n": int(len(errors)),
                "MAE": round(float(abs_errors.mean()), 6) if len(abs_errors) else None,
                "RMSE": round(float(math.sqrt((errors**2).mean())), 6) if len(errors) else None,
                "MRE_percent": round(float(relative_errors.mean()), 6) if len(relative_errors) else None,
                "MAPE_percent": round(float(absolute_percentage_errors.mean()), 6) if len(absolute_percentage_errors) else None,
                "bias": round(float(errors.mean()), 6) if len(errors) else None,
                "median_abs_error": round(float(abs_errors.median()), 6) if len(abs_errors) else None,
                "range_hit_rate": round(float(group["actual_in_neighbor_range"].mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_property", "scenario", "method"])


def scenario_features(row: pd.Series, *, target: str, scenario: str) -> pd.Series:
    cleaned = row.dropna().copy()
    target_key = f"bulk_property:{safe_key(target)}"
    if target_key in cleaned.index:
        cleaned = cleaned.drop(index=target_key)
    if scenario == "analysis_only":
        cleaned = cleaned[[not str(index).startswith("bulk_property:") for index in cleaned.index]]
    return cleaned


def build_prompt_record(
    *,
    sample_id: str,
    target: str,
    scenario: str,
    actual: float,
    pred_row: dict[str, Any],
    ranked: pd.DataFrame,
    feature_weights: dict[str, float],
) -> dict[str, Any]:
    relevant_features = sorted(feature_weights.items(), key=lambda item: item[1], reverse=True)[:12]
    evidence = {
        "query_sample_id": sample_id,
        "target_property": target,
        "scenario": scenario,
        "estimate": pred_row.get("estimate"),
        "historical_range": [pred_row.get("historical_range_min"), pred_row.get("historical_range_max")],
        "top_neighbors": ranked[["rank", "neighbor_sample_id", "distance", "target_value", "shared_feature_count", "top_weighted_shared_features"]].to_dict(orient="records"),
        "top_feature_weights": [{"feature": feature, "weight": round(weight, 6)} for feature, weight in relevant_features],
    }
    prompt = (
        "You are a petroleum-chemistry expert. Use the Historical Experimental DB evidence below for "
        "training-free property inference. Do not treat the nearest-neighbor estimate as a trained model prediction. "
        "Explain whether the retrieved samples are chemically relevant, which features support or weaken the inference, "
        "and provide a final evidence-constrained conclusion with limitations.\n\n"
        f"Evidence JSON:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    return {
        "sample_id": sample_id,
        "target_property": target,
        "scenario": scenario,
        "actual_for_evaluation_only": round(actual, 6),
        "evidence_json": json.dumps(evidence, ensure_ascii=False),
        "expert_prompt": prompt,
        "llm_status": "prompt_generated_not_called",
    }


def write_outputs(
    *,
    out_dir: Path,
    feature_matrix: pd.DataFrame,
    property_table: pd.DataFrame,
    metadata: pd.DataFrame,
    correlations: pd.DataFrame,
    redundancy: pd.DataFrame,
    predictions: pd.DataFrame,
    neighbors: pd.DataFrame,
    prompts: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: list[str],
    top_k: int,
) -> None:
    feature_matrix.to_csv(out_dir / "feature_matrix_all.csv", encoding="utf-8-sig")
    latent_cols = feature_matrix.attrs.get("spectral_latent_feature_columns") or [
        column for column in feature_matrix.columns if ":pca:" in str(column) or ":pls:" in str(column)
    ]
    if latent_cols:
        feature_matrix[latent_cols].to_csv(out_dir / "spectral_latent_features.csv", encoding="utf-8-sig")
    spectral_manifest = feature_matrix.attrs.get("spectral_model_manifest") or []
    spectral_artifacts = feature_matrix.attrs.get("spectral_model_artifacts") or []
    if spectral_manifest:
        model_dir = out_dir / "spectral_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_manifest.json").write_text(json.dumps(spectral_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        save_spectral_model_artifacts(model_dir, spectral_artifacts)
    property_table.to_csv(out_dir / "property_table_all.csv", encoding="utf-8-sig")
    metadata.to_csv(out_dir / "sample_metadata.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(out_dir / "feature_target_correlations.csv", index=False, encoding="utf-8-sig")
    correlations.groupby("target_property").head(25).to_csv(out_dir / "feature_weight_top25_by_target.csv", index=False, encoding="utf-8-sig")
    redundancy.to_csv(out_dir / "feature_feature_redundancy_top_pairs.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    neighbors.to_csv(out_dir / "neighbor_evidence.csv", index=False, encoding="utf-8-sig")
    prompts.to_json(out_dir / "expert_reasoning_prompts.jsonl", orient="records", force_ascii=False, lines=True)
    metrics.to_csv(out_dir / "metrics_by_property.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "petroleum_fraction-experimental-db-evaluation.v1",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "top_k": top_k,
                "targets": targets,
                "sample_count": int(len(metadata)),
                "train_count": int((metadata["split"] == "train").sum()),
                "test_count": int((metadata["split"] == "test").sum()),
                "feature_count": int(feature_matrix.shape[1]),
                "spectral_latent_feature_count": int(len(latent_cols)),
                "spectral_model_count": int(len(spectral_manifest)),
                "spectral_model_artifact_count": int(len(spectral_artifacts)),
                "property_count": int(property_table.shape[1]),
                "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file() or path.is_dir()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_svg_top_correlations(out_dir / "top_feature_correlations.svg", correlations, targets)
    write_report(out_dir / "README.md", metrics=metrics, correlations=correlations, predictions=predictions, metadata=metadata)


def save_spectral_model_artifacts(model_dir: Path, artifacts: list[dict[str, Any]]) -> None:
    for artifact in artifacts:
        file_name = str(artifact.get("artifact_file") or "").strip()
        arrays = artifact.get("arrays") or {}
        if not file_name or not isinstance(arrays, dict):
            continue
        numeric_arrays = {key: np.asarray(value, dtype=float) for key, value in arrays.items()}
        np.savez(model_dir / file_name, **numeric_arrays)


def write_report(path: Path, *, metrics: pd.DataFrame, correlations: pd.DataFrame, predictions: pd.DataFrame, metadata: pd.DataFrame) -> None:
    lines = [
        "# petroleum fraction Historical Experimental DB evaluation",
        "",
        "This run evaluates training-free property inference using train/test sample workbooks.",
        "",
        "## Data split",
        f"- Train samples: {int((metadata['split'] == 'train').sum())}",
        f"- Test samples: {int((metadata['split'] == 'test').sum())}",
        "",
        "## Feature processing",
        "- GC and IR curves are aligned to a training-set grid and converted into PCA/PLS latent variables.",
        "- PCA models are fitted per spectral channel using train samples only.",
        "- PLS models are fitted per spectral channel and target property using train samples only.",
        "- Hydrocarbon-type rows are converted into component-content features.",
        "- Bulk properties are numeric features only when the scenario allows them; the target property itself is always masked from the query.",
        "",
        "## Metrics",
        markdown_table(metrics) if not metrics.empty else "No metrics generated.",
        "",
        "## Top feature-target relationships",
    ]
    for target, group in correlations.groupby("target_property"):
        lines.append(f"### {target}")
        cols = ["feature", "feature_channel", "n", "spearman", "pearson", "weight"]
        lines.append(markdown_table(group.head(10)[cols]))
        lines.append("")
    if not predictions.empty:
        lines.extend(
            [
                "## Prediction examples",
                markdown_table(predictions.head(20)),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_top_correlations(path: Path, correlations: pd.DataFrame, targets: list[str]) -> None:
    rows = []
    for target in targets:
        group = correlations[correlations["target_property"] == target].head(8)
        for item in group.to_dict(orient="records"):
            rows.append(item)
    width = 1300
    row_h = 24
    height = 80 + max(1, len(rows)) * row_h
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfdff"/>',
        '<text x="24" y="30" font-family="Arial" font-size="24" font-weight="700" fill="#0f2442">Top PCA/PLS-informed feature-target weights</text>',
        '<text x="24" y="52" font-family="Arial" font-size="13" fill="#526987">w = 0.05 + mean(abs(Pearson), abs(Spearman)) * coverage; used in weighted shared-feature distance.</text>',
    ]
    y = 70
    for item in rows:
        weight = float(item.get("weight") or 0.0)
        bar_w = min(460, max(2, weight * 420))
        label = f'{item.get("target_property")} | {item.get("feature")}'
        svg.append(f'<text x="24" y="{y}" font-family="Arial" font-size="12" fill="#0f2442">{escape_xml(label[:105])}</text>')
        svg.append(f'<rect x="710" y="{y-14}" width="{bar_w:.1f}" height="14" rx="4" fill="#2f6fdd" opacity="0.78"/>')
        svg.append(f'<text x="{720 + bar_w:.1f}" y="{y}" font-family="Arial" font-size="12" fill="#0f2442">w={weight:.3f}, n={int(item.get("n") or 0)}</text>')
        y += row_h
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def feature_scale(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    std = float(numeric.std(ddof=0))
    if math.isfinite(std) and std > 1e-9:
        return std
    max_abs = float(numeric.abs().max())
    return max(max_abs, 1.0)


def value_at(frame: pd.DataFrame, index: str, column: str) -> float | None:
    if column not in frame.columns or index not in frame.index:
        return None
    return to_float(frame.loc[index, column])


def to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def is_finite_number(value: Any) -> bool:
    return to_float(value) is not None


def round_or_none(value: Any, digits: int = 6) -> float | None:
    numeric = to_float(value)
    return round(numeric, digits) if numeric is not None else None


def safe_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def feature_channel(feature: str) -> str:
    if ":" not in feature:
        return "generic"
    return feature.split(":", 1)[0]


def escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    display = display.where(pd.notna(display), "")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False):
        cells = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
