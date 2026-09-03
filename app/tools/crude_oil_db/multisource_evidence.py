from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.tools.crude_oil_db.pipeline import (
    EXPERT_PRIORS,
    PROPERTY_DENSITY_PREFIX,
    PROPERTY_SAMPLE_NUMBER,
    _feature_matrix,
    _property_columns,
    _workspace_property_table,
    build_database_evidence,
    build_external_database_evidence,
)


def build_query_driven_multisource_database_evidence(
    out_root: Path,
    sample_id: str,
    *,
    query: str,
    features: dict[str, Any] | None = None,
    known_properties: dict[str, Any] | None = None,
    ir_descriptors: dict[str, Any] | None = None,
    history_limit: int = 10,
) -> dict[str, Any]:
    """Build main-program database evidence with query-driven multi-source context.

    This is the production counterpart of the isolated
    `experiments/multisource_evidence_reasoning` prototype. It does not call an
    LLM by itself; it prepares structured multi-source evidence for Planner and
    the final synthesis step.
    """

    sample_id = str(sample_id or "external_sample").strip() or "external_sample"
    features = features if isinstance(features, dict) else {}
    base = (
        build_external_database_evidence(out_root, sample_id, features, k=history_limit)
        if features
        else build_database_evidence(out_root, sample_id, k=history_limit)
    )
    if base.get("status") != "success":
        return base

    features_by_sample = _read_feature_store(out_root)
    properties = _workspace_property_table(out_root)
    ir_payload = ir_descriptors if isinstance(ir_descriptors, dict) else _load_default_ir_descriptors(out_root)
    inventory = _database_inventory(
        properties=properties,
        features_by_sample=features_by_sample,
        ir_descriptors=ir_payload,
    )
    targets = _infer_target_properties(query, list(inventory.get("property_columns") or []))
    query_known_properties = _mask_target_properties(known_properties or {}, targets)
    historical_records = _historical_records_from_neighbors(
        neighbors=base.get("neighbors") or [],
        features_by_sample=features_by_sample,
        properties=properties,
        ir_descriptors=ir_payload,
        limit=history_limit,
    )
    multisource_case = {
        "schema_version": "multisource-db-evidence.v1",
        "task": "query_driven_multisource_petroleum_reasoning",
        "sample_id": sample_id,
        "user_query": query,
        "target_properties": targets,
        "database_inventory": inventory,
        "query_evidence": {
            "gc": {
                "available": bool(features),
                "data": _compact_features(features) if features else None,
            },
            "ir": {
                "available": sample_id in ir_payload,
                "data": ir_payload.get(sample_id),
            },
            "known_properties": {
                "available": bool(query_known_properties),
                "data": query_known_properties,
                "masked_target_properties": targets,
                "masking_policy": "Only target properties requested by user_query are removed from query evidence.",
            },
        },
        "historical_database_records": historical_records,
        "expert_priors": EXPERT_PRIORS,
        "integration_policy": {
            "database_provides_multisource_records": True,
            "planner_and_llm_integrate_evidence": True,
            "no_pretrained_prediction_model": True,
        },
    }
    base["multisource_evidence"] = multisource_case
    base["database_capability"] = {
        "supports_multisource_records": True,
        "integration_owner": "planner_orchestrator_and_llm_synthesis",
    }
    return base


def _read_feature_store(out_root: Path) -> dict[str, dict[str, Any]]:
    features_dir = out_root / "features"
    if not features_dir.exists():
        return {}
    features: dict[str, dict[str, Any]] = {}
    for path in sorted(features_dir.glob("*.features.json")):
        try:
            features[path.name.replace(".features.json", "")] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return features


def _load_default_ir_descriptors(out_root: Path) -> dict[str, Any]:
    for path in (out_root / "ir_descriptors.json", out_root / "indexes" / "ir_descriptors.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _sample_properties(properties: pd.DataFrame, sample_id: str) -> dict[str, Any]:
    if properties.empty or PROPERTY_SAMPLE_NUMBER not in properties.columns:
        return {}
    indexed = properties.set_index(PROPERTY_SAMPLE_NUMBER)
    if sample_id not in indexed.index:
        return {}
    row = indexed.loc[sample_id].to_dict()
    return {
        key: None if pd.isna(value) else float(value)
        for key, value in row.items()
        if key in _property_columns(properties)
    }


def _compact_features(features: dict[str, Any]) -> dict[str, Any]:
    peak = features.get("peak_area_features") or {}
    raw = features.get("raw_shape_features") or {}
    return {
        "raw_summary": {
            "rt_range": raw.get("rt_range"),
            "dominant_peak_rt": raw.get("dominant_peak_rt"),
            "cumulative_rt_quantiles": raw.get("cumulative_rt_quantiles"),
            "intensity_quantiles": raw.get("intensity_quantiles"),
        },
        "peak_area_summary": {
            "peak_count": peak.get("peak_count"),
            "rt_range": peak.get("rt_range"),
            "top_peak_rts": peak.get("top_peak_rts"),
            "cumulative_rt_quantiles": peak.get("cumulative_rt_quantiles"),
            "anchor_pair_summary": peak.get("anchor_pair_summary"),
        },
        "property_feature_profile": features.get("property_feature_profile"),
    }


def _infer_target_properties(query: str, available_properties: list[str]) -> list[str]:
    lowered = str(query or "").lower()
    targets: list[str] = []
    density_keys = [key for key in available_properties if key.startswith(PROPERTY_DENSITY_PREFIX)]
    if any(term in lowered for term in ["density", "text"]):
        targets.extend(density_keys)
    bp_keys = [key for key in available_properties if key.startswith("BP_")]
    if any(term in lowered for term in ["boiling", "distillation", "bp", "text", "text"]):
        explicit = [key for key in bp_keys if key.lower() in lowered or key.replace("_", "").lower() in lowered]
        targets.extend(explicit or bp_keys)
    if any(term in lowered for term in ["property", "properties", "text"]) and not targets:
        targets.extend(density_keys)
        targets.extend(bp_keys)
    deduped: list[str] = []
    for target in targets:
        if target in available_properties and target not in deduped:
            deduped.append(target)
    return deduped


def _mask_target_properties(properties: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    target_set = set(targets)
    return {key: value for key, value in properties.items() if key not in target_set}


def _database_inventory(
    *,
    properties: pd.DataFrame,
    features_by_sample: dict[str, dict[str, Any]],
    ir_descriptors: dict[str, Any],
) -> dict[str, Any]:
    property_cols = _property_columns(properties) if not properties.empty else []
    property_availability = {}
    for col in property_cols:
        property_availability[col] = int(pd.to_numeric(properties[col], errors="coerce").notna().sum())
    return {
        "sample_count_with_gc_features": len(features_by_sample),
        "sample_count_with_ir_descriptors": len(ir_descriptors),
        "property_columns": property_cols,
        "property_availability": property_availability,
        "available_modalities": {
            "gc": len(features_by_sample) > 0,
            "ir": len(ir_descriptors) > 0,
            "measured_properties": bool(property_cols),
        },
    }


def _historical_records_from_neighbors(
    *,
    neighbors: list[dict[str, Any]],
    features_by_sample: dict[str, dict[str, Any]],
    properties: pd.DataFrame,
    ir_descriptors: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(neighbors[: max(1, int(limit))], start=1):
        historical_id = str(row.get("sample_id") or "")
        records.append(
            {
                "case_id": f"H{idx:02d}",
                "sample_id": historical_id,
                "retrieval_distance": row.get("distance"),
                "evidence_channels": {
                    "gc": {
                        "available": historical_id in features_by_sample,
                        "data": _compact_features(features_by_sample.get(historical_id, {})),
                    },
                    "ir": {
                        "available": historical_id in ir_descriptors,
                        "data": ir_descriptors.get(historical_id),
                    },
                    "measured_properties": {
                        "available": True,
                        "data": _sample_properties(properties, historical_id),
                    },
                },
            }
        )
    return records
