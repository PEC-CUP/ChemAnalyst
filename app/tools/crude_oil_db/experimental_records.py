from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.tools.crude_oil_db.pipeline import _infer_from_neighbors, _workspace_property_table


EXPERIMENTAL_RECORD_SCHEMA = "experimental-record.v1"
HYDROCARBON_TYPE_FEATURE_SCHEMA = "hydrocarbon-type-features.v1"


def persist_tool_result_record(
    out_root: Path,
    *,
    payload: dict[str, Any],
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    contract = manifest.get("database_integration_contract")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        return None
    tool_evidence = result.get("tool_evidence") if isinstance(result.get("tool_evidence"), dict) else {}
    structured = tool_evidence.get("structured_result")
    if structured in (None, {}, []):
        return None
    sample_id = _sample_id_from_binding(payload, tool_evidence, contract)
    record = build_tool_result_record(
        sample_id=sample_id,
        capability_id=str(manifest.get("capability_id") or tool_evidence.get("capability_id") or ""),
        channel_id=str(contract.get("channel_id") or tool_evidence.get("evidence_type") or "tool_result"),
        tool_evidence=tool_evidence,
        artifacts=result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {},
        contract=contract,
    )
    return persist_experimental_record(out_root, record)


def build_tool_result_record(
    *,
    sample_id: str,
    capability_id: str,
    channel_id: str,
    tool_evidence: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structured = tool_evidence.get("structured_result")
    features = extract_flexible_numeric_features(structured)
    record_schema = str((contract or {}).get("record_schema") or EXPERIMENTAL_RECORD_SCHEMA)
    feature_schema = str((contract or {}).get("feature_schema") or _feature_schema_for_channel(channel_id))
    return {
        "schema_version": record_schema,
        "record_id": uuid4().hex,
        "sample_id": str(sample_id or "external_sample"),
        "channel_id": str(channel_id or "tool_result"),
        "capability_id": capability_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "evidence_schema": tool_evidence.get("schema_version"),
            "evidence_type": tool_evidence.get("evidence_type"),
            "method": tool_evidence.get("method"),
            "source_file": tool_evidence.get("source_file"),
            "artifacts": artifacts or {},
        },
        "structured_result": structured,
        "feature_channel": {
            "schema_version": feature_schema,
            "extraction_policy": "flexible_label_value_numeric_intersection",
            "features": features,
            "feature_count": len(features),
        },
        "quality_flags": list(tool_evidence.get("quality_flags") or []),
        "db_integration_contract": contract or {},
    }


def persist_experimental_record(out_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    root = out_root / "experimental_records"
    channel_id = _safe_token(str(record.get("channel_id") or "tool_result"))
    sample_id = _safe_token(str(record.get("sample_id") or "external_sample"))
    directory = root / channel_id / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.get('record_id') or uuid4().hex}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "success",
        "schema_version": record.get("schema_version"),
        "record_id": record.get("record_id"),
        "sample_id": record.get("sample_id"),
        "channel_id": record.get("channel_id"),
        "feature_schema": (record.get("feature_channel") or {}).get("schema_version"),
        "feature_count": (record.get("feature_channel") or {}).get("feature_count"),
        "record_file": str(path),
    }


def build_channel_database_evidence(
    out_root: Path,
    record: dict[str, Any],
    *,
    query: str,
    history_limit: int = 10,
) -> dict[str, Any]:
    channel_id = str(record.get("channel_id") or "")
    query_features = _record_features(record)
    records = [
        item
        for item in load_channel_records(out_root, channel_id)
        if item.get("record_id") != record.get("record_id") and _record_features(item)
    ]
    if not query_features:
        return {"status": "error", "message": "experimental_record_features_missing", "channel_id": channel_id}
    if not records:
        return {
            "status": "skipped",
            "message": "channel_history_missing",
            "channel_id": channel_id,
            "sample_id": record.get("sample_id"),
            "record_count": 0,
        }

    neighbors = _channel_neighbors(record, records, history_limit=history_limit)
    if not neighbors:
        return {
            "status": "skipped",
            "message": "channel_history_not_comparable",
            "channel_id": channel_id,
            "sample_id": record.get("sample_id"),
            "record_count": len(records),
        }
    properties = _workspace_property_table(out_root)
    property_summary = {}
    if not properties.empty:
        neighbor_frame = pd.DataFrame(
            [{"sample_id": item["sample_id"], "distance": item["distance"]} for item in neighbors]
        )
        property_summary = _infer_from_neighbors(neighbor_frame, properties)
    return {
        "status": "success",
        "message": "experimental_channel_history_compared",
        "sample_id": record.get("sample_id"),
        "channel_id": channel_id,
        "feature_schema": (record.get("feature_channel") or {}).get("schema_version"),
        "feature_extraction_policy": "flexible_label_value_numeric_intersection",
        "record_count": len(records),
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": property_summary,
        "query": query,
        "query_record": {
            "record_id": record.get("record_id"),
            "feature_count": len(query_features),
            "feature_labels": sorted(feature.get("label") for feature in query_features.values()),
        },
        "comparison_policy": {
            "shared_numeric_feature_labels_only": True,
            "does_not_require_fixed_component_count": True,
            "historical_records_are_evidence_for_llm": True,
        },
    }


def build_tool_result_channel_database_evidence(
    out_root: Path,
    *,
    tool_result: dict[str, Any],
    capability_id: str,
    query: str,
    history_limit: int = 10,
) -> dict[str, Any]:
    record_ref = tool_result.get("experimental_record") if isinstance(tool_result.get("experimental_record"), dict) else {}
    record_file = record_ref.get("record_file")
    if record_file and Path(str(record_file)).exists():
        try:
            record = json.loads(Path(str(record_file)).read_text(encoding="utf-8"))
        except Exception:
            record = {}
        if isinstance(record, dict) and record:
            return build_channel_database_evidence(out_root, record, query=query, history_limit=history_limit)

    tool_evidence = tool_result.get("tool_evidence") if isinstance(tool_result.get("tool_evidence"), dict) else {}
    if not tool_evidence:
        return {"status": "skipped", "message": "dynamic_tool_evidence_missing"}
    channel_id = _channel_id_from_evidence(tool_evidence)
    record = build_tool_result_record(
        sample_id=str(record_ref.get("sample_id") or Path(str(tool_evidence.get("source_file") or "external_sample")).stem),
        capability_id=capability_id,
        channel_id=channel_id,
        tool_evidence=tool_evidence,
        artifacts=tool_result.get("artifacts") if isinstance(tool_result.get("artifacts"), dict) else {},
    )
    return build_channel_database_evidence(out_root, record, query=query, history_limit=history_limit)


def load_channel_records(out_root: Path, channel_id: str) -> list[dict[str, Any]]:
    root = out_root / "experimental_records" / _safe_token(channel_id)
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.glob("*/*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(item, dict):
            item.setdefault("record_file", str(path))
            records.append(item)
    return records


def extract_flexible_numeric_features(structured_result: Any) -> dict[str, dict[str, Any]]:
    rows = structured_result if isinstance(structured_result, list) else [structured_result]
    features: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _label_from_row(row)
        if not label:
            continue
        for field, value in row.items():
            number = _as_finite_float(value)
            if number is None or _is_identifier_field(field):
                continue
            key = _feature_key(label, str(field))
            features[key] = {"label": label, "value_field": str(field), "value": number}
    return features


def _channel_neighbors(record: dict[str, Any], records: list[dict[str, Any]], *, history_limit: int) -> list[dict[str, Any]]:
    query_features = _record_features(record)
    neighbors: list[dict[str, Any]] = []
    for item in records:
        historical = _record_features(item)
        shared = sorted(set(query_features) & set(historical))
        if not shared:
            continue
        squared = sum((query_features[key]["value"] - historical[key]["value"]) ** 2 for key in shared)
        distance = math.sqrt(squared / len(shared))
        neighbors.append(
            {
                "record_id": item.get("record_id"),
                "sample_id": item.get("sample_id"),
                "distance": round(distance, 6),
                "shared_feature_count": len(shared),
                "shared_features": [
                    {
                        "label": query_features[key]["label"],
                        "value_field": query_features[key]["value_field"],
                        "query_value": query_features[key]["value"],
                        "historical_value": historical[key]["value"],
                    }
                    for key in shared[:40]
                ],
                "record_file": item.get("record_file"),
            }
        )
    neighbors.sort(key=lambda item: (-int(item["shared_feature_count"]), float(item["distance"])))
    return neighbors[: max(1, int(history_limit))]


def _record_features(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    channel = record.get("feature_channel") if isinstance(record.get("feature_channel"), dict) else {}
    features = channel.get("features")
    return features if isinstance(features, dict) else {}


def _sample_id_from_binding(payload: dict[str, Any], tool_evidence: dict[str, Any], contract: dict[str, Any]) -> str:
    binding = str(contract.get("sample_id_binding") or "input_file_stem")
    if binding == "tool_evidence.sample_id" and tool_evidence.get("sample_id"):
        return str(tool_evidence["sample_id"])
    explicit = payload.get("sample_id")
    if explicit:
        return str(explicit)
    for field in ("input_file", "file_path", "excel_file"):
        if payload.get(field):
            return Path(str(payload[field])).stem
    return "external_sample"


def _label_from_row(row: dict[str, Any]) -> str:
    preferred = ("item", "name", "component", "class", "category", "group", "compound", "label")
    normalized = {str(key).strip().lower(): key for key in row}
    for name in preferred:
        key = normalized.get(name)
        value = row.get(key) if key is not None else None
        if value not in (None, ""):
            return str(value).strip()
    string_values = [str(value).strip() for value in row.values() if isinstance(value, str) and str(value).strip()]
    return string_values[-1] if string_values else ""


def _is_identifier_field(field: Any) -> bool:
    lowered = str(field).strip().lower()
    return lowered in {"sample", "sample_id", "id", "index", "mode", "m/z", "mz"}


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _feature_key(label: str, field: str) -> str:
    return f"{_safe_token(label)}::{_safe_token(field)}"


def _feature_schema_for_channel(channel_id: str) -> str:
    if _safe_token(channel_id) == "hydrocarbon_type":
        return HYDROCARBON_TYPE_FEATURE_SCHEMA
    return f"{_safe_token(channel_id)}-features.v1"


def _channel_id_from_evidence(tool_evidence: dict[str, Any]) -> str:
    evidence_type = _safe_token(str(tool_evidence.get("evidence_type") or ""))
    if evidence_type == "hydrocarbon_group_composition":
        return "hydrocarbon_type"
    return evidence_type or "tool_result"


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(value).strip().lower()).strip("_")
    return token or "unknown"
