from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from app.tools.experimental_db.sample_database import workbook_to_payload


petroleum_fraction_SPECTRAL_EVIDENCE_SCHEMA = "petroleum-fraction-spectral-experimental-db-evidence.v1"
DEFAULT_EVAL_ROOT = Path("outputs") / "petroleum_fraction_experimental_db" / "feature_models"
DEFAULT_CHANNELS = ("gc_fid", "ir_spectrum")
DISTILLATION_TARGETS = ("BP_20", "BP_35", "BP_50", "BP_65", "BP_80")
DEFAULT_TARGETS = ("density_20c", *DISTILLATION_TARGETS, "saturates")


def build_petroleum_fraction_spectral_test_evidence(
    *,
    sample_id: str,
    query: str | None = None,
    top_k: int = 4,
    eval_root: Path = DEFAULT_EVAL_ROOT,
) -> dict[str, Any] | None:
    """Build Experimental DB evidence for a petroleum fraction test sample using saved PCA/PLS features.

    The query sample remains transient: only train-split samples are used as the
    historical database. Test-set actual values are attached only as validation
    fields and are not used for neighbor retrieval or weighted estimates.
    """

    result_dir = _latest_petroleum_fraction_eval_dir(eval_root)
    if result_dir is None:
        return None
    metadata = pd.read_csv(result_dir / "sample_metadata.csv")
    if sample_id not in set(metadata["sample_id"].astype(str)):
        return None
    row = metadata.loc[metadata["sample_id"].astype(str) == sample_id].iloc[0]
    if str(row.get("split") or "") != "test":
        return None

    feature_matrix = pd.read_csv(result_dir / "feature_matrix_all.csv", index_col=0)
    property_table = pd.read_csv(result_dir / "property_table_all.csv", index_col=0)
    correlations = pd.read_csv(result_dir / "feature_target_correlations.csv")
    model_manifest = _read_model_manifest(result_dir)

    channels = _target_channels_from_query(query)
    targets = _target_properties_from_query(query)
    train_ids = metadata.loc[metadata["split"].astype(str) == "train", "sample_id"].astype(str).tolist()
    train_features = feature_matrix.loc[[item for item in train_ids if item in feature_matrix.index]]
    query_features = feature_matrix.loc[sample_id]

    feature_columns = _selected_latent_features(feature_matrix.columns, channels=channels, targets=targets)
    if not feature_columns:
        return _skipped(
            "no_selected_pca_pls_features",
            sample_id=sample_id,
            result_dir=result_dir,
            channels=channels,
            targets=targets,
        )
    scales = _feature_scales(train_features[feature_columns])
    weights = _combined_feature_weights(correlations, targets=targets, feature_columns=feature_columns)
    neighbors = _rank_neighbors(
        sample_id=sample_id,
        query=query_features[feature_columns],
        candidates=train_features[feature_columns],
        metadata=metadata,
        property_table=property_table,
        scales=scales,
        weights=weights,
        targets=targets,
        top_k=top_k,
    )
    if not neighbors:
        return _skipped(
            "no_train_neighbors_in_selected_pca_pls_space",
            sample_id=sample_id,
            result_dir=result_dir,
            channels=channels,
            targets=targets,
        )

    property_rows = _property_evidence_rows(
        sample_id=sample_id,
        neighbors=neighbors,
        property_table=property_table,
        targets=targets,
    )
    query_profile = {
        "sample_id": sample_id,
        "raw_sample_id": row.get("raw_sample_id"),
        "available_channels": list(channels),
        "feature_basis": [
            "saved_training_spectral_models",
            "gc_fid_pca_pls_latent_features",
            "ir_spectrum_pca_pls_latent_features",
        ],
        "feature_vector": {
            feature: _round_or_none(query_features.get(feature), digits=6)
            for feature in feature_columns
            if _to_float(query_features.get(feature)) is not None
        },
    }
    evidence = {
        "schema_version": petroleum_fraction_SPECTRAL_EVIDENCE_SCHEMA,
        "status": "success",
        "message": "petroleum_fraction_test_sample_compared_to_train_historical_db_with_saved_spectral_models",
        "source": "sample_centered_experimental_db",
        "sample_id": sample_id,
        "query": query or "",
        "result_dir": str(result_dir),
        "feature_basis": query_profile["feature_basis"],
        "evidence_channels": list(channels),
        "target_properties": list(targets),
        "query_profile": {
            "sample_id": sample_id,
            "raw_sample_id": row.get("raw_sample_id"),
            "available_channels": list(channels),
            "feature_count": len(query_profile["feature_vector"]),
            "feature_examples": list(query_profile["feature_vector"])[:20],
        },
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": {
            item["property_name"]: {
                "weighted_estimate": item["weighted_estimate"],
                "evidence_range": [item["range_min"], item["range_max"]],
                "neighbor_count": item["neighbor_count"],
                "supporting_samples": item["supporting_samples"],
                "validation_actual": item.get("actual_for_validation"),
                "absolute_error": item.get("absolute_error"),
                "absolute_percentage_error": item.get("absolute_percentage_error"),
            }
            for item in property_rows
        },
        "spectral_model_summary": _model_summary(model_manifest, channels=channels),
        "retrieval_policy": {
            "query_sample_role": "test_set_transient_query_not_historical_record",
            "historical_records_source": "train split of petroleum fraction experimental DB evaluation output",
            "use_saved_training_spectral_models": True,
            "selected_channels": list(channels),
            "selected_target_properties": list(targets),
            "pca_pls_features_only": True,
            "target_values_masked_from_query": True,
            "nearest_neighbors_are_evidence": True,
            "weighted_estimates_are_evidence_not_final_labels": True,
        },
        "presentation": _presentation(
            sample_id=sample_id,
            query=query or "",
            query_profile=query_profile,
            neighbors=neighbors,
            property_rows=property_rows,
            channels=channels,
            targets=targets,
            result_dir=result_dir,
        ),
        "database_status": {
            "status": "ready",
            "train_sample_count": len(train_ids),
            "test_sample_count": int((metadata["split"].astype(str) == "test").sum()),
            "spectral_model_count": len(model_manifest),
            "result_dir": str(result_dir),
        },
    }
    return evidence


def build_petroleum_fraction_spectral_workbook_evidence(
    *,
    workbook: Path,
    template_definition: dict[str, Any],
    query: str | None = None,
    top_k: int = 4,
    eval_root: Path = DEFAULT_EVAL_ROOT,
) -> dict[str, Any]:
    result_dir = _latest_petroleum_fraction_eval_dir(eval_root)
    if result_dir is None:
        return _skipped(
            "petroleum_fraction_spectral_model_output_not_found",
            sample_id=Path(workbook).stem,
            result_dir=Path(eval_root),
            channels=DEFAULT_CHANNELS,
            targets=DEFAULT_TARGETS,
        )
    payload = workbook_to_payload(Path(workbook), template_definition=template_definition)
    sample_info = payload.get("sample_information") if isinstance(payload.get("sample_information"), dict) else {}
    sample_id = str(sample_info.get("sample_id") or Path(workbook).stem)
    channels = _target_channels_from_query(query)
    targets = _target_properties_from_query(query)

    metadata = pd.read_csv(result_dir / "sample_metadata.csv")
    feature_matrix = pd.read_csv(result_dir / "feature_matrix_all.csv", index_col=0)
    property_table = pd.read_csv(result_dir / "property_table_all.csv", index_col=0)
    correlations = pd.read_csv(result_dir / "feature_target_correlations.csv")
    model_manifest = _read_model_manifest(result_dir)
    query_features = _project_workbook_spectral_features(
        payload=payload,
        result_dir=result_dir,
        model_manifest=model_manifest,
        channels=channels,
        targets=targets,
    )
    feature_columns = sorted(query_features)
    if not feature_columns:
        return _skipped(
            "uploaded_workbook_has_no_projectable_gc_fid_or_ir_features",
            sample_id=sample_id,
            result_dir=result_dir,
            channels=channels,
            targets=targets,
        )

    train_ids = metadata.loc[metadata["split"].astype(str) == "train", "sample_id"].astype(str).tolist()
    train_features = feature_matrix.loc[[item for item in train_ids if item in feature_matrix.index], feature_columns]
    query_series = pd.Series(query_features, name=sample_id)
    scales = _feature_scales(train_features)
    weights = _combined_feature_weights(correlations, targets=targets, feature_columns=feature_columns)
    neighbors = _rank_neighbors(
        sample_id=sample_id,
        query=query_series,
        candidates=train_features,
        metadata=metadata,
        property_table=property_table,
        scales=scales,
        weights=weights,
        targets=targets,
        top_k=top_k,
    )
    if not neighbors:
        return _skipped(
            "no_train_neighbors_for_uploaded_workbook",
            sample_id=sample_id,
            result_dir=result_dir,
            channels=channels,
            targets=targets,
        )
    actual_properties = _properties_from_payload(payload)
    property_rows = _property_evidence_rows(
        sample_id=sample_id,
        neighbors=neighbors,
        property_table=property_table,
        targets=targets,
        actual_properties=actual_properties,
    )
    query_profile = {
        "sample_id": sample_id,
        "raw_sample_id": sample_info.get("raw_sample_id") or sample_id,
        "available_channels": list(channels),
        "feature_basis": [
            "uploaded_transient_query_workbook",
            "saved_training_spectral_models",
            "gc_fid_pca_pls_latent_features",
            "ir_spectrum_pca_pls_latent_features",
        ],
        "feature_vector": {feature: _round_or_none(value, digits=6) for feature, value in query_features.items()},
    }
    return {
        "schema_version": petroleum_fraction_SPECTRAL_EVIDENCE_SCHEMA,
        "status": "success",
        "message": "uploaded_query_workbook_compared_to_train_historical_db_with_saved_spectral_models",
        "source": "sample_centered_experimental_db",
        "sample_id": sample_id,
        "query": query or "",
        "result_dir": str(result_dir),
        "feature_basis": query_profile["feature_basis"],
        "evidence_channels": list(channels),
        "target_properties": list(targets),
        "query_profile": {
            "sample_id": sample_id,
            "raw_sample_id": query_profile["raw_sample_id"],
            "available_channels": list(channels),
            "feature_count": len(query_profile["feature_vector"]),
            "feature_examples": list(query_profile["feature_vector"])[:20],
        },
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": {
            item["property_name"]: {
                "weighted_estimate": item["weighted_estimate"],
                "evidence_range": [item["range_min"], item["range_max"]],
                "neighbor_count": item["neighbor_count"],
                "supporting_samples": item["supporting_samples"],
                "validation_actual": item.get("actual_for_validation"),
                "absolute_error": item.get("absolute_error"),
                "absolute_percentage_error": item.get("absolute_percentage_error"),
                "estimation_method": "inverse_distance_weighted_interpolation_over_top_k_neighbors",
            }
            for item in property_rows
        },
        "spectral_model_summary": _model_summary(model_manifest, channels=channels),
        "retrieval_policy": {
            "query_sample_role": "uploaded_transient_query_not_inserted_into_historical_db",
            "historical_records_source": "train split of petroleum fraction experimental DB evaluation output",
            "use_saved_training_spectral_models": True,
            "selected_channels": list(channels),
            "selected_target_properties": list(targets),
            "pca_pls_features_only": True,
            "target_values_masked_from_query": True,
            "neighbor_ranking": "weighted_shared_pca_pls_latent_feature_distance",
            "property_estimation": "inverse_distance_weighted_interpolation_over_top_k_neighbors",
            "weighted_estimates_are_evidence_not_final_labels": True,
        },
        "presentation": _presentation(
            sample_id=sample_id,
            query=query or "",
            query_profile=query_profile,
            neighbors=neighbors,
            property_rows=property_rows,
            channels=channels,
            targets=targets,
            result_dir=result_dir,
            query_role="uploaded query workbook",
        ),
        "database_status": {
            "status": "ready",
            "train_sample_count": len(train_ids),
            "spectral_model_count": len(model_manifest),
            "result_dir": str(result_dir),
        },
    }


def _latest_petroleum_fraction_eval_dir(eval_root: Path) -> Path | None:
    root = Path(eval_root)
    if not root.exists():
        return None
    candidates = [
        item
        for item in sorted(root.iterdir())
        if item.is_dir()
        and (item / "feature_matrix_all.csv").exists()
        and (item / "property_table_all.csv").exists()
        and (item / "sample_metadata.csv").exists()
        and (item / "spectral_models" / "model_manifest.json").exists()
    ]
    return candidates[-1] if candidates else None


def _read_model_manifest(result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "spectral_models" / "model_manifest.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _target_channels_from_query(query: str | None) -> tuple[str, ...]:
    text = str(query or "").lower()
    channels = []
    if any(token in text for token in ("gc-fid", "gc_fid", "gc fid", "fid", "text")):
        channels.append("gc_fid")
    if any(token in text for token in ("ir", "ftir", "infrared", "text")):
        channels.append("ir_spectrum")
    return tuple(channels or DEFAULT_CHANNELS)


def _target_properties_from_query(query: str | None) -> tuple[str, ...]:
    text = str(query or "").lower()
    targets: list[str] = []
    if any(token in text for token in ("density", "text")):
        targets.append("density_20c")
    if any(token in text for token in ("distillation", "boiling", "bp_", "text", "text")):
        targets.extend(DISTILLATION_TARGETS)
    if any(token in text for token in ("saturate", "saturates", "text")):
        targets.append("saturates")
    return tuple(dict.fromkeys(targets or DEFAULT_TARGETS))


def _selected_latent_features(columns: pd.Index, *, channels: tuple[str, ...], targets: tuple[str, ...]) -> list[str]:
    selected = []
    safe_targets = {_safe_key(target) for target in targets}
    for column in map(str, columns):
        if not any(column.startswith(f"{channel}:") for channel in channels):
            continue
        if ":pca:" in column:
            selected.append(column)
            continue
        if ":pls:" in column:
            parts = column.split(":")
            if len(parts) >= 4 and parts[2] in safe_targets:
                selected.append(column)
    return selected


def _feature_scales(train_features: pd.DataFrame) -> dict[str, float]:
    scales = {}
    for column in train_features.columns:
        values = pd.to_numeric(train_features[column], errors="coerce").dropna()
        std = float(values.std(ddof=0)) if not values.empty else 1.0
        scales[str(column)] = std if math.isfinite(std) and std > 1e-9 else 1.0
    return scales


def _combined_feature_weights(
    correlations: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    feature_columns: list[str],
) -> dict[str, float]:
    subset = correlations[
        correlations["target_property"].astype(str).isin(targets)
        & correlations["feature"].astype(str).isin(feature_columns)
    ]
    weights = defaultdict(lambda: 0.05)
    for feature, group in subset.groupby("feature"):
        values = pd.to_numeric(group["weight"], errors="coerce").dropna()
        if not values.empty:
            weights[str(feature)] = float(values.max())
    return dict(weights)


def _rank_neighbors(
    *,
    sample_id: str,
    query: pd.Series,
    candidates: pd.DataFrame,
    metadata: pd.DataFrame,
    property_table: pd.DataFrame,
    scales: dict[str, float],
    weights: dict[str, float],
    targets: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_by_id = metadata.set_index("sample_id", drop=False)
    for neighbor_id, candidate in candidates.iterrows():
        shared = sorted(set(query.dropna().index) & set(candidate.dropna().index))
        if not shared:
            continue
        weighted_squared = 0.0
        weight_sum = 0.0
        feature_items = []
        for feature in shared:
            q = _to_float(query.get(feature))
            c = _to_float(candidate.get(feature))
            if q is None or c is None:
                continue
            weight = float(weights.get(feature, 0.05))
            scale = max(float(scales.get(feature) or 1.0), 1e-9)
            delta = (q - c) / scale
            weighted_squared += weight * delta * delta
            weight_sum += weight
            feature_items.append((feature, weight, abs(delta)))
        if weight_sum <= 0:
            continue
        distance = math.sqrt(weighted_squared / weight_sum) + 1.0 / max(len(shared), 1)
        props = {
            target: _round_or_none(property_table.loc[neighbor_id, target])
            for target in targets
            if target in property_table.columns and neighbor_id in property_table.index
        }
        meta = metadata_by_id.loc[neighbor_id] if neighbor_id in metadata_by_id.index else {}
        feature_items.sort(key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "sample_id": str(neighbor_id),
                "sample_type": "petroleum fraction",
                "origin": "petroleum fraction train split",
                "distance": round(distance, 6),
                "shared_feature_count": len(shared),
                "shared_feature_examples": shared[:12],
                "top_weighted_shared_features": [
                    {"feature": name, "weight": round(weight, 6), "normalized_delta": round(delta, 6)}
                    for name, weight, delta in feature_items[:10]
                ],
                "available_channels": ["gc_fid", "ir_spectrum"],
                "bulk_properties": props,
                "raw_sample_id": meta.get("raw_sample_id") if hasattr(meta, "get") else None,
            }
        )
    rows.sort(key=lambda item: (item["distance"], -item["shared_feature_count"], item["sample_id"]))
    return rows[: max(1, int(top_k))]


def _project_workbook_spectral_features(
    *,
    payload: dict[str, Any],
    result_dir: Path,
    model_manifest: list[dict[str, Any]],
    channels: tuple[str, ...],
    targets: tuple[str, ...],
) -> dict[str, float]:
    spectra = _spectral_rows_from_payload(payload)
    safe_targets = {_safe_key(target) for target in targets}
    features: dict[str, float] = {}
    for model in model_manifest:
        analysis_type = str(model.get("analysis_type") or "")
        if analysis_type not in channels:
            continue
        if str(model.get("model_type") or "").upper() == "PLS" and _safe_key(str(model.get("target_property") or "")) not in safe_targets:
            continue
        artifact = str(model.get("artifact_file") or "")
        if not artifact:
            continue
        rows = spectra.get(analysis_type)
        if not rows:
            continue
        path = result_dir / "spectral_models" / artifact
        if not path.exists():
            continue
        arrays = np.load(path)
        grid = np.asarray(arrays["grid"], dtype=float)
        points = _spectral_points(rows, analysis_type)
        if points is None:
            continue
        x, y = points
        interpolated = np.interp(grid, x, y)
        normalized = _row_normalize_spectrum(interpolated)
        scaled = (normalized - np.asarray(arrays["mean"], dtype=float)) / np.asarray(arrays["scale"], dtype=float)
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        model_type = str(model.get("model_type") or "").upper()
        if model_type == "PCA":
            scores = scaled @ np.asarray(arrays["loadings"], dtype=float).T
            for index, value in enumerate(scores, start=1):
                features[f"{analysis_type}:pca:pc{index}"] = float(value)
        elif model_type == "PLS":
            scores = _transform_pls_scores(
                scaled.reshape(1, -1),
                np.asarray(arrays["weights"], dtype=float),
                np.asarray(arrays["loadings"], dtype=float),
            )[0]
            target = _safe_key(str(model.get("target_property") or ""))
            for index, value in enumerate(scores, start=1):
                features[f"{analysis_type}:pls:{target}:lv{index}"] = float(value)
    return features


def _spectral_rows_from_payload(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for analysis in payload.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        analysis_type = str(analysis.get("analysis_type") or analysis.get("type") or "").strip().lower()
        if analysis_type == "ir":
            analysis_type = "ir_spectrum"
        rows = analysis.get("rows") or analysis.get("data") or []
        if analysis_type in {"gc_fid", "ir_spectrum"} and isinstance(rows, list) and rows:
            out[analysis_type] = rows
    return out


def _spectral_points(rows: list[dict[str, Any]], analysis_type: str) -> tuple[np.ndarray, np.ndarray] | None:
    is_ir = analysis_type.startswith("ir")
    x_keys = ("wave_number_cm-1", "wave_number", "wavenumber", "wn") if is_ir else ("rt", "apex_rt", "time")
    y_keys = ("transmittance", "absorbance", "intensity") if is_ir else ("intensity", "abundance", "area")
    points = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        x = _first_numeric(row, x_keys)
        y = _first_numeric(row, y_keys)
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


def _row_normalize_spectrum(values: np.ndarray) -> np.ndarray:
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


def _transform_pls_scores(x: np.ndarray, weights: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    e = np.asarray(x, dtype=float).copy()
    scores = []
    for weight, loading in zip(weights, loadings, strict=False):
        score = e @ weight
        scores.append(score)
        e = e - np.outer(score, loading)
    return np.vstack(scores).T if scores else np.empty((x.shape[0], 0))


def _properties_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in payload.get("bulk_properties") or []:
        if not isinstance(row, dict):
            continue
        name = row.get("property_name") or row.get("name") or row.get("property")
        value = _to_float(row.get("value"))
        if name and value is not None:
            values[str(name)] = value
    return values


def _property_evidence_rows(
    *,
    sample_id: str,
    neighbors: list[dict[str, Any]],
    property_table: pd.DataFrame,
    targets: tuple[str, ...],
    actual_properties: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        items = []
        for neighbor in neighbors:
            value = _to_float((neighbor.get("bulk_properties") or {}).get(target))
            distance = _to_float(neighbor.get("distance"))
            if value is None or distance is None:
                continue
            items.append((value, 1.0 / max(distance, 1e-6), neighbor["sample_id"]))
        if not items:
            continue
        weight_sum = sum(weight for _, weight, _ in items) or 1.0
        estimate = sum(value * weight for value, weight, _ in items) / weight_sum
        values = [value for value, _, _ in items]
        if actual_properties and target in actual_properties:
            actual = _to_float(actual_properties.get(target))
        else:
            actual = _to_float(property_table.loc[sample_id, target]) if target in property_table.columns and sample_id in property_table.index else None
        error = estimate - actual if actual is not None else None
        ape = abs(error / actual) * 100 if error is not None and actual not in (None, 0) else None
        density_digits = _infer_decimal_digits([*values, actual]) if _is_density_property(target) else None
        rows.append(
            {
                "property_name": target,
                "weighted_estimate": _round_property_value(target, estimate, digits=density_digits),
                "range_min": _round_property_value(target, min(values), digits=density_digits),
                "range_max": _round_property_value(target, max(values), digits=density_digits),
                "neighbor_count": len(items),
                "supporting_samples": [sample for _, _, sample in items[:8]],
                "actual_for_validation": _round_property_value(target, actual, digits=density_digits),
                "absolute_error": _round_property_error(target, abs(error) if error is not None else None, digits=density_digits),
                "absolute_percentage_error": _round_or_none(ape, digits=2),
            }
        )
    return rows


def _model_summary(model_manifest: list[dict[str, Any]], *, channels: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "analysis_type": item.get("analysis_type"),
            "model_type": item.get("model_type"),
            "target_property": item.get("target_property"),
            "component_count": item.get("component_count"),
            "train_sample_count": item.get("train_sample_count"),
            "artifact_file": item.get("artifact_file"),
        }
        for item in model_manifest
        if item.get("analysis_type") in channels
    ]


def _presentation(
    *,
    sample_id: str,
    query: str,
    query_profile: dict[str, Any],
    neighbors: list[dict[str, Any]],
    property_rows: list[dict[str, Any]],
    channels: tuple[str, ...],
    targets: tuple[str, ...],
    result_dir: Path,
    query_role: str = "test-set transient query",
) -> dict[str, Any]:
    return {
        "workflow": [
            {"step": 1, "title": "Prepare transient query sample", "detail": f"Use {query_role} {sample_id}; it is not inserted into the historical train database."},
            {"step": 2, "title": "Apply saved spectral preprocessing", "detail": "Project GC-FID and IR curves into PCA/PLS latent variables fitted on the training split."},
            {"step": 3, "title": "Retrieve train-split historical neighbors", "detail": f"Ranked {len(neighbors)} train samples using weighted shared PCA/PLS latent-feature distance."},
            {"step": 4, "title": "Interpolate property evidence", "detail": f"Prepared inverse-distance weighted interpolation evidence for {', '.join(targets)}; validation values are hidden from retrieval."},
        ],
        "query_snapshot": {
            "sample_id": sample_id,
            "feature_count": len(query_profile.get("feature_vector") or {}),
            "channels": list(channels),
            "feature_examples": list(query_profile.get("feature_vector") or {})[:12],
        },
        "neighbor_table": [
            {
                "rank": index + 1,
                "sample_id": item.get("sample_id"),
                "sample_type": item.get("sample_type"),
                "origin": item.get("origin"),
                "distance": item.get("distance"),
                "shared_feature_count": item.get("shared_feature_count"),
                "shared_feature_examples": item.get("shared_feature_examples"),
                "analysis_channels": item.get("available_channels"),
            }
            for index, item in enumerate(neighbors)
        ],
        "property_table": property_rows,
        "constrained_reasoning": _constrained_reasoning(sample_id, query, property_rows, neighbors),
        "evidence_statement": _evidence_statement(sample_id, channels, targets, neighbors, property_rows),
        "reasoning_notes": [
            "This is the Experimental DB evidence-test path with a petroleum fraction spectral validation branch.",
            f"Saved PCA/PLS features were loaded from {result_dir}.",
            "Only GC-FID and IR-spectrum PCA/PLS latent variables are used for this query.",
            "Training-set neighbor properties are evidence carriers; validation actual values are reported only for method checking.",
        ],
    }


def _constrained_reasoning(
    sample_id: str,
    query: str,
    property_rows: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
) -> dict[str, Any]:
    support_counts = {item["property_name"]: item["neighbor_count"] for item in property_rows}
    property_text = "; ".join(
        f"{item['property_name']}={_format_property_value(item['property_name'], item['weighted_estimate'])} "
        f"(range {_format_property_value(item['property_name'], item['range_min'])}-"
        f"{_format_property_value(item['property_name'], item['range_max'])}, "
        f"validation actual={_format_property_value(item['property_name'], item.get('actual_for_validation'))})"
        for item in property_rows
    )
    return {
        "status": "spectral_pca_pls_evidence_prepared_for_expert_reasoning",
        "neighbor_count": len(neighbors),
        "diagnostics": {
            "neighbor_count": len(neighbors),
            "distance_min": min((item["distance"] for item in neighbors), default=None),
            "distance_max": max((item["distance"] for item in neighbors), default=None),
            "shared_feature_count_min": min((item["shared_feature_count"] for item in neighbors), default=0),
            "shared_feature_count_max": max((item["shared_feature_count"] for item in neighbors), default=0),
            "property_support_counts": support_counts,
            "diagnostic_role": "validation actuals are not used for retrieval",
        },
        "final_answer": (
            f"For test sample {sample_id}, GC-FID and IR PCA/PLS latent evidence retrieved {len(neighbors)} train-split historical neighbors. "
            f"The evidence estimates are: {property_text}. "
            "The petroleum expert reasoning step should judge chemical relevance of the spectral latent features before treating these values as conclusions."
        ),
        "property_claims": [
            {
                "property_name": item["property_name"],
                "evidence_center": item["weighted_estimate"],
                "evidence_range": [item["range_min"], item["range_max"]],
                "support_count": item["neighbor_count"],
                "supporting_samples": item["supporting_samples"],
                "interpretation": (
                    f"{item['property_name']}: spectral evidence center="
                    f"{_format_property_value(item['property_name'], item['weighted_estimate'])}, "
                    f"neighbor range={_format_property_value(item['property_name'], item['range_min'])}-"
                    f"{_format_property_value(item['property_name'], item['range_max'])}, "
                    f"support_count={item['neighbor_count']}, validation actual="
                    f"{_format_property_value(item['property_name'], item.get('actual_for_validation'))}, "
                    f"absolute relative error={_format_percent(item.get('absolute_percentage_error'))}%."
                ),
            }
            for item in property_rows
        ],
        "prompt_policy": [
            "Use GC-FID and IR PCA/PLS latent variables as spectral evidence, not as raw curve statistics.",
            "Do not use the test-set actual values in reasoning; they are validation annotations only.",
            "Explain that the numeric evidence centers are obtained by inverse-distance weighted interpolation over top-k historical neighbors.",
            "For density, consider spectral similarity together with petroleum-property relationships.",
            "For distillation distribution, discuss BP_20/BP_35/BP_50/BP_65/BP_80 as a curve rather than isolated labels.",
            "Write boiling-point values as table values or explicit intervals; do not use shorthand such as '420s °C', 'mid-450s °C', or 'around 470 °C' when numerical evidence is available.",
            "For saturates, evaluate whether spectral-neighbor evidence is chemically consistent with composition evidence.",
        ],
        "expert_reasoning_prompt": (
            "You are a petroleum-chemistry expert. Use only the GC-FID and IR PCA/PLS Experimental DB evidence. "
            "Explain density, distillation distribution, and saturates inference from train-split historical neighbors. "
            "Explicitly state that property centers are inverse-distance weighted interpolations over the retrieved top-k historical samples. "
            "For BP_20/BP_35/BP_50/BP_65/BP_80, report the numeric evidence center or an explicit numeric interval, and do not use decade shorthand such as '420s °C' or vague phrases such as 'mid-450s °C'. "
            "Do not use validation actual values except to comment on method performance if they are explicitly marked as validation. "
            f"User query: {query}"
        ),
        "expert_reasoning": None,
    }


def _evidence_statement(
    sample_id: str,
    channels: tuple[str, ...],
    targets: tuple[str, ...],
    neighbors: list[dict[str, Any]],
    property_rows: list[dict[str, Any]],
) -> str:
    nearest = neighbors[0] if neighbors else {}
    targets_text = ", ".join(targets)
    channel_text = ", ".join(channels)
    props = "; ".join(
        f"{item['property_name']}={_format_property_value(item['property_name'], item['weighted_estimate'])}"
        for item in property_rows
    )
    return (
        f"For petroleum fraction test sample {sample_id}, the Experimental DB evidence test used saved {channel_text} PCA/PLS latent features "
        f"to infer {targets_text}. The nearest train-split support sample is {nearest.get('sample_id')} "
        f"(distance={nearest.get('distance')}, shared_features={nearest.get('shared_feature_count')}). "
        f"Evidence centers: {props}."
    )


def _skipped(reason: str, *, sample_id: str, result_dir: Path, channels: tuple[str, ...], targets: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": petroleum_fraction_SPECTRAL_EVIDENCE_SCHEMA,
        "status": "skipped",
        "message": reason,
        "sample_id": sample_id,
        "result_dir": str(result_dir),
        "evidence_channels": list(channels),
        "target_properties": list(targets),
    }


def _to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    numeric = _to_float(value)
    return round(numeric, digits) if numeric is not None else None


def _property_digits(property_name: str, *, error: bool = False) -> int:
    key = _safe_key(property_name)
    if key.startswith("bp_"):
        return 2
    if key in {"saturates", "aromatics", "wax_content", "sat_ar"}:
        return 2
    return 3 if error else 2


def _is_density_property(property_name: str) -> bool:
    return "density" in _safe_key(property_name)


def _round_property_value(property_name: str, value: Any, *, digits: int | None = None) -> float | None:
    if _is_density_property(property_name):
        return _round_or_none(value, digits=digits if digits is not None else 6)
    return _round_or_none(value, digits=_property_digits(property_name))


def _round_property_error(property_name: str, value: Any, *, digits: int | None = None) -> float | None:
    if _is_density_property(property_name):
        return _round_or_none(value, digits=digits if digits is not None else 6)
    return _round_or_none(value, digits=_property_digits(property_name, error=True))


def _format_property_value(property_name: str, value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    if _is_density_property(property_name):
        return f"{numeric:g}"
    return f"{numeric:.{_property_digits(property_name)}f}"


def _format_percent(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.2f}"


def _infer_decimal_digits(values: list[Any], *, default: int = 4, maximum: int = 6) -> int:
    digits = []
    for value in values:
        numeric = _to_float(value)
        if numeric is None:
            continue
        text = f"{numeric:.10f}".rstrip("0").rstrip(".")
        if "." in text:
            digits.append(len(text.split(".", 1)[1]))
    return min(max(digits, default=default), maximum)


def _safe_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _first_numeric(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalized = {_safe_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(_safe_key(key))
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None
