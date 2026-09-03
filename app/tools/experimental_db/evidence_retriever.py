from __future__ import annotations

import math
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.tools.experimental_db.sample_database import ANALYSIS_TYPES, SampleDatabase


SAMPLE_DB_EVIDENCE_SCHEMA = "experimental-db-evidence.v1"
SUPPORTED_ANALYSIS_TYPES = set(ANALYSIS_TYPES)
CHROMATOGRAM_TYPES = {"gc_fid", "htgc_fid", "gc_ncd", "gc_scd"}


def build_sample_centered_database_evidence(
    db_root: Path,
    *,
    sample_id: str | None = None,
    features: dict[str, Any] | None = None,
    tool_result: dict[str, Any] | None = None,
    query: str | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Compare current-session evidence with sample-centered Experimental DB records.

    This function deliberately does not write the current session result into the
    database. It treats uploaded/tool-derived evidence as a transient query sample
    and compares it against curated historical records in data/experimental_db.
    """

    database = SampleDatabase(Path(db_root))
    status = database.status()
    if int(status.get("sample_count") or 0) <= 0:
        return _skipped("sample_centered_database_empty", sample_id=sample_id, database_status=status)

    query_profile = _build_query_profile(
        sample_id=sample_id,
        features=features,
        tool_result=tool_result,
        query=query,
    )
    if not query_profile["feature_vector"]:
        return _skipped(
            "no_comparable_session_features",
            sample_id=query_profile.get("sample_id") or sample_id,
            database_status=status,
            query_profile=query_profile,
        )

    historical_profiles = []
    for item in database.list_samples(limit=10000):
        historical_id = str(item.get("sample_id") or "")
        if not historical_id:
            continue
        if sample_id and historical_id == str(sample_id):
            continue
        record = database.get_sample(historical_id)
        if not record:
            continue
        profile = _profile_from_sample_record(record)
        if profile["feature_vector"]:
            historical_profiles.append(profile)

    if not historical_profiles:
        return _skipped(
            "no_comparable_historical_records",
            sample_id=query_profile.get("sample_id") or sample_id,
            database_status=status,
            query_profile=query_profile,
        )

    ranked = []
    for profile in historical_profiles:
        distance, shared_keys = _feature_distance(query_profile["feature_vector"], profile["feature_vector"])
        if not shared_keys:
            continue
        ranked.append(
            {
                "sample_id": profile["sample_id"],
                "sample_type": profile.get("sample_type"),
                "origin": profile.get("origin"),
                "distance": round(distance, 6),
                "shared_feature_count": len(shared_keys),
                "shared_feature_examples": shared_keys[:12],
                "available_channels": profile["available_channels"],
                "bulk_properties": profile["bulk_properties"],
            }
        )
    ranked.sort(key=lambda item: (item["distance"], -item["shared_feature_count"], item["sample_id"]))
    neighbors = ranked[: max(1, int(top_k))]
    if not neighbors:
        return _skipped(
            "historical_records_have_no_shared_feature_space",
            sample_id=query_profile.get("sample_id") or sample_id,
            database_status=status,
            query_profile=query_profile,
        )

    return {
        "schema_version": SAMPLE_DB_EVIDENCE_SCHEMA,
        "status": "success",
        "message": "session_sample_compared_to_sample_centered_experimental_db",
        "source": "sample_centered_experimental_db",
        "sample_id": query_profile.get("sample_id") or sample_id,
        "query": query or "",
        "feature_basis": query_profile["feature_basis"],
        "evidence_channels": query_profile["available_channels"],
        "query_profile": {
            "sample_id": query_profile.get("sample_id") or sample_id,
            "available_channels": query_profile["available_channels"],
            "feature_count": len(query_profile["feature_vector"]),
            "feature_examples": sorted(query_profile["feature_vector"])[:20],
        },
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": _summarize_properties(neighbors),
        "retrieval_policy": {
            "current_session_not_persisted": True,
            "historical_records_source": str(Path(db_root)),
            "multi_source_feature_space": True,
            "nearest_neighbors_are_evidence": True,
            "final_property_inference_requires_llm_reasoning": True,
            "do_not_copy_weighted_estimate_blindly": True,
        },
        "presentation": _presentation_bundle(
            sample_id=query_profile.get("sample_id") or sample_id,
            query=query or "",
            query_profile=query_profile,
            neighbors=neighbors,
        ),
        "database_status": status,
    }


def build_tool_result_sample_database_evidence(
    db_root: Path,
    *,
    tool_result: dict[str, Any],
    capability_id: str | None = None,
    query: str | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    return build_sample_centered_database_evidence(
        db_root,
        sample_id=_sample_id_from_tool_result(tool_result) or capability_id,
        tool_result=tool_result,
        query=query,
        top_k=top_k,
    )


def build_existing_sample_database_evidence(
    db_root: Path,
    *,
    sample_id: str,
    query: str | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Compare one curated Experimental DB sample against the other records."""

    database = SampleDatabase(Path(db_root))
    status = database.status()
    sample_id = str(sample_id or "").strip()
    if not sample_id:
        return _skipped("missing_sample_id", sample_id=sample_id, database_status=status)
    record = database.get_sample(sample_id)
    if record is None:
        return _skipped("sample_not_found", sample_id=sample_id, database_status=status)

    query_profile = _profile_from_sample_record(record)
    if not query_profile["feature_vector"]:
        return _skipped(
            "sample_has_no_comparable_features",
            sample_id=sample_id,
            database_status=status,
            query_profile=query_profile,
        )

    historical_profiles = []
    for item in database.list_samples(limit=10000):
        historical_id = str(item.get("sample_id") or "")
        if not historical_id or historical_id == sample_id:
            continue
        historical_record = database.get_sample(historical_id)
        if not historical_record:
            continue
        profile = _profile_from_sample_record(historical_record)
        if profile["feature_vector"]:
            historical_profiles.append(profile)

    ranked = []
    for profile in historical_profiles:
        distance, shared_keys = _feature_distance(query_profile["feature_vector"], profile["feature_vector"])
        if not shared_keys:
            continue
        ranked.append(
            {
                "sample_id": profile["sample_id"],
                "sample_type": profile.get("sample_type"),
                "origin": profile.get("origin"),
                "distance": round(distance, 6),
                "shared_feature_count": len(shared_keys),
                "shared_feature_examples": shared_keys[:12],
                "available_channels": profile["available_channels"],
                "bulk_properties": profile["bulk_properties"],
            }
        )
    ranked.sort(key=lambda item: (item["distance"], -item["shared_feature_count"], item["sample_id"]))
    neighbors = ranked[: max(1, int(top_k))]
    if not neighbors:
        return _skipped(
            "historical_records_have_no_shared_feature_space",
            sample_id=sample_id,
            database_status=status,
            query_profile=query_profile,
        )

    return {
        "schema_version": SAMPLE_DB_EVIDENCE_SCHEMA,
        "status": "success",
        "message": "curated_sample_compared_to_sample_centered_experimental_db",
        "source": "sample_centered_experimental_db",
        "sample_id": sample_id,
        "query": query or "",
        "feature_basis": query_profile["feature_basis"],
        "evidence_channels": query_profile["available_channels"],
        "query_profile": {
            "sample_id": sample_id,
            "available_channels": query_profile["available_channels"],
            "feature_count": len(query_profile["feature_vector"]),
            "feature_examples": sorted(query_profile["feature_vector"])[:20],
        },
        "neighbor_count": len(neighbors),
        "neighbors": neighbors,
        "property_summary": _summarize_properties(neighbors),
        "retrieval_policy": {
            "current_session_not_persisted": True,
            "historical_records_source": str(Path(db_root)),
            "multi_source_feature_space": True,
            "nearest_neighbors_are_evidence": True,
            "final_property_inference_requires_llm_reasoning": True,
            "do_not_copy_weighted_estimate_blindly": True,
        },
        "presentation": _presentation_bundle(
            sample_id=sample_id,
            query=query or "",
            query_profile=query_profile,
            neighbors=neighbors,
        ),
        "database_status": status,
    }


def _build_query_profile(
    *,
    sample_id: str | None,
    features: dict[str, Any] | None,
    tool_result: dict[str, Any] | None,
    query: str | None,
) -> dict[str, Any]:
    profile = {
        "sample_id": sample_id or _sample_id_from_tool_result(tool_result) or _sample_id_from_features(features),
        "sample_type": None,
        "origin": None,
        "available_channels": [],
        "feature_basis": [],
        "feature_vector": {},
        "bulk_properties": {},
    }
    if isinstance(features, dict) and features:
        _merge_profile(profile, _profile_from_gc_features(features), basis="gc_feature_object")
    if isinstance(tool_result, dict) and tool_result:
        _merge_profile(profile, _profile_from_tool_result(tool_result), basis="tool_result")
    if query:
        profile["query"] = query
    return profile


def _profile_from_sample_record(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    profile = {
        "sample_id": str(sample.get("sample_id") or record.get("sample_id") or ""),
        "sample_type": sample.get("sample_type"),
        "origin": sample.get("origin") or sample.get("refinery"),
        "available_channels": [],
        "feature_basis": ["sample_centered_db_record"],
        "feature_vector": {},
        "bulk_properties": _properties_to_dict(record.get("bulk_properties") or []),
    }
    for key, value in profile["bulk_properties"].items():
        profile["feature_vector"][f"bulk_property:{_safe_key(key)}"] = value
    if profile["bulk_properties"]:
        profile["available_channels"].append("bulk_properties")
    for analysis in record.get("analyses") or []:
        if not isinstance(analysis, dict):
            continue
        analysis_type = _normalize_analysis_type(analysis.get("analysis_type") or analysis.get("type"))
        rows = analysis.get("rows") or analysis.get("data") or []
        extracted = _features_from_rows(analysis_type, rows)
        if extracted:
            profile["available_channels"].append(analysis_type)
            profile["feature_vector"].update(extracted)
    profile["available_channels"] = sorted(set(profile["available_channels"]))
    return profile


def _profile_from_gc_features(features: dict[str, Any]) -> dict[str, Any]:
    profile = _empty_transient_profile()
    sample_meta = features.get("sample_meta") if isinstance(features.get("sample_meta"), dict) else {}
    if sample_meta.get("sample_id"):
        profile["sample_id"] = str(sample_meta["sample_id"])
    profile["available_channels"].append("gc_fid")
    for prefix in (
        "distribution_features",
        "diagnostic_ratios",
        "component_class_profile",
        "chromatogram_shape",
        "quality_flags",
        "match_summary",
        "property_feature_profile",
    ):
        value = features.get(prefix)
        profile["feature_vector"].update(_flatten_numeric(value, f"gc_feature:{prefix}"))
    return profile


def _profile_from_tool_result(tool_result: dict[str, Any]) -> dict[str, Any]:
    profile = _empty_transient_profile()
    tool_evidence = tool_result.get("tool_evidence") if isinstance(tool_result.get("tool_evidence"), dict) else {}
    evidence_type = str(tool_evidence.get("evidence_type") or tool_evidence.get("analysis_type") or "").lower()
    result_payload = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else {}
    structured_result = (
        tool_evidence.get("structured_result")
        or tool_result.get("structured_result")
        or result_payload.get("structured_result")
    )
    if "hydrocarbon" in evidence_type or "hydrocarbon" in str(tool_result).lower():
        analysis_type = "hydrocarbon_types"
    else:
        analysis_type = _normalize_analysis_type(tool_evidence.get("analysis_type") or tool_result.get("analysis_type"))
    rows = structured_result if isinstance(structured_result, list) else []
    extracted = _features_from_rows(analysis_type, rows)
    if extracted:
        profile["available_channels"].append(analysis_type)
        profile["feature_vector"].update(extracted)
    profile["feature_vector"].update(_flatten_numeric(tool_evidence.get("quality_flags"), "tool_quality"))
    return profile


def _features_from_rows(analysis_type: str, rows: Any) -> dict[str, float]:
    if not isinstance(rows, list) or not rows:
        return {}
    analysis_type = _normalize_analysis_type(analysis_type)
    if analysis_type in CHROMATOGRAM_TYPES:
        return _curve_features(rows, analysis_type, x_keys=("rt", "apex_rt", "time"), y_keys=("intensity", "abundance", "area"))
    if analysis_type == "ir":
        return _curve_features(
            rows,
            analysis_type,
            x_keys=("wave_number_cm-1", "wave_number", "wavenumber", "wn"),
            y_keys=("transmittance", "absorbance", "intensity"),
        )
    if analysis_type in {"hydrocarbon_types", "molecular_composition"}:
        return _composition_features(rows, analysis_type)
    return _composition_features(rows, analysis_type)


def _curve_features(rows: list[dict[str, Any]], analysis_type: str, *, x_keys: tuple[str, ...], y_keys: tuple[str, ...]) -> dict[str, float]:
    points = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        x = _first_numeric(row, x_keys)
        y = _first_numeric(row, y_keys)
        if x is not None and y is not None:
            points.append((x, abs(y)))
    if not points:
        return {}
    points.sort(key=lambda item: item[0])
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    total_y = sum(ys) or 1.0
    return {
        f"{analysis_type}:point_count": float(len(points)),
        f"{analysis_type}:x_min": min(xs),
        f"{analysis_type}:x_max": max(xs),
        f"{analysis_type}:weighted_x_mean": sum(x * y for x, y in points) / total_y,
        f"{analysis_type}:weighted_x_q10": _weighted_quantile(points, 0.10),
        f"{analysis_type}:weighted_x_q50": _weighted_quantile(points, 0.50),
        f"{analysis_type}:weighted_x_q90": _weighted_quantile(points, 0.90),
        f"{analysis_type}:dominant_x": max(points, key=lambda item: item[1])[0],
    }


def _composition_features(rows: list[dict[str, Any]], analysis_type: str) -> dict[str, float]:
    features: dict[str, float] = {}
    label_keys = (
        "component",
        "item",
        "name",
        "compound_class",
        "molecule_family",
        "molecular_formula",
        "formula",
        "class",
    )
    value_keys = (
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
    excluded_numeric_keys = {*label_keys, *value_keys, "sample_id", "unit"}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = _first_text(
            row,
            label_keys,
        )
        value = _first_numeric(
            row,
            value_keys,
        )
        if label and value is not None:
            features[f"{analysis_type}:{_safe_key(label)}"] = value
        for key, raw_value in row.items():
            safe = _safe_key(key)
            if safe in excluded_numeric_keys:
                continue
            numeric = _to_float(raw_value)
            if numeric is not None:
                features.setdefault(f"{analysis_type}:numeric:{safe}", numeric)
    return features


def _properties_to_dict(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("property_name") or row.get("name") or row.get("property")
        value = _to_float(row.get("value"))
        if name and value is not None:
            values[str(name)] = value
    return values


def _summarize_properties(neighbors: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for neighbor in neighbors:
        distance = float(neighbor.get("distance") or 0.0)
        weight = 1.0 / max(distance, 1e-6)
        for name, value in (neighbor.get("bulk_properties") or {}).items():
            numeric = _to_float(value)
            if numeric is not None:
                buckets[name].append((numeric, weight, neighbor["sample_id"]))
    summary = {}
    for name, items in buckets.items():
        weight_sum = sum(weight for _, weight, _ in items) or 1.0
        values = [value for value, _, _ in items]
        summary[name] = {
            "weighted_estimate": round(sum(value * weight for value, weight, _ in items) / weight_sum, 6),
            "evidence_range": [round(min(values), 6), round(max(values), 6)],
            "neighbor_count": len(items),
            "supporting_samples": [sample_id for _, _, sample_id in items[:8]],
            "role": "historical_db_evidence_not_pretrained_model",
        }
    return summary


def _presentation_bundle(
    *,
    sample_id: str | None,
    query: str,
    query_profile: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> dict[str, Any]:
    top_properties = _top_property_rows(neighbors)
    query_channels = query_profile.get("available_channels") or []
    query_feature_examples = sorted((query_profile.get("feature_vector") or {}).keys())[:12]
    constrained_reasoning = _constrained_reasoning(
        sample_id=sample_id,
        query=query,
        query_channels=query_channels,
        neighbors=neighbors,
        top_properties=top_properties,
    )
    return {
        "workflow": [
            {"step": 1, "title": "Build query profile", "detail": f"Extracted {len(query_profile.get('feature_vector') or {})} comparable features from the selected sample."},
            {"step": 2, "title": "Retrieve historical neighbors", "detail": f"Ranked {len(neighbors)} nearest historical samples by shared feature distance."},
            {"step": 3, "title": "Aggregate property evidence", "detail": f"Summarized {len(top_properties)} property ranges and weighted estimates from the retrieved neighbors."},
            {"step": 4, "title": "Constrain evidence-based reasoning", "detail": "Use the neighbor table and support counts to produce a training-free inference without copying a single neighbor value blindly."},
        ],
        "query_snapshot": {
            "sample_id": sample_id,
            "feature_count": len(query_profile.get("feature_vector") or {}),
            "channels": query_channels,
            "feature_examples": query_feature_examples,
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
        "property_table": top_properties,
        "constrained_reasoning": constrained_reasoning,
        "evidence_statement": _evidence_statement(sample_id=sample_id, query=query, query_channels=query_channels, neighbors=neighbors, top_properties=top_properties),
        "reasoning_notes": [
            "bulk_properties is treated as an analysis-data channel at the same level as hydrocarbon_types; sample information is kept as metadata.",
            "Similarity is computed in a shared feature space built from imported analysis-data rows, including bulk_properties and other analysis sheets.",
            "The retrieved neighbors are evidence carriers, not direct predictions or labels from a trained model.",
            "Weighted estimates should be interpreted together with the supporting range and the actual neighbor identities.",
        ],
    }


def _constrained_reasoning(
    *,
    sample_id: str | None,
    query: str,
    query_channels: list[str],
    neighbors: list[dict[str, Any]],
    top_properties: list[dict[str, Any]],
) -> dict[str, Any]:
    neighbor_count = len(neighbors)
    diagnostics = _evidence_diagnostics(neighbors=neighbors, top_properties=top_properties)
    if neighbor_count <= 0:
        return {
            "status": "evidence_unavailable",
            "diagnostics": diagnostics,
            "final_answer": "No comparable historical neighbor was retrieved. Petroleum-expert reasoning should not infer properties from the historical DB for this query.",
            "property_claims": [],
            "prompt_policy": _reasoning_prompt_policy(),
            "expert_reasoning_prompt": _reasoning_prompt_instruction(query=query),
            "expert_reasoning": None,
        }
    claims = []
    for item in top_properties:
        name = item.get("property_name")
        center = item.get("weighted_estimate")
        range_min = item.get("range_min")
        range_max = item.get("range_max")
        support_count = int(item.get("neighbor_count") or 0)
        samples = item.get("supporting_samples") or []
        interpretation = (
            f"{name}: evidence center={center}, historical range={range_min}-{range_max}, "
            f"support_count={support_count}, supporting_samples={', '.join(map(str, samples)) or 'none'}. "
            "This is a database evidence item; expert reasoning must judge whether the supporting features are chemically relevant to the queried property."
        )
        claims.append(
            {
                "property_name": name,
                "evidence_center": center,
                "evidence_range": [range_min, range_max],
                "support_count": support_count,
                "supporting_samples": samples,
                "interpretation": interpretation,
            }
        )
    channel_text = ", ".join(query_channels) or "imported analysis-data channels"
    property_text = "; ".join(
        f"{item.get('property_name')}={item.get('weighted_estimate')} (range {item.get('range_min')}-{item.get('range_max')}, n={item.get('neighbor_count')})"
        for item in top_properties[:5]
    ) or "no property summary"
    final_answer = (
        f"For {sample_id or 'the query sample'}, the Historical Experimental DB retrieved {neighbor_count} historical neighbor(s) "
        f"using {channel_text}. The property evidence table contains: {property_text}. "
        "This programmatic summary is not a confidence label; the final interpretation should be made by the petroleum-chemistry expert reasoning step using feature relevance, distance, shared-feature basis, property relationships, and evidence conflicts."
    )
    return {
        "status": "evidence_prepared_for_expert_reasoning",
        "neighbor_count": neighbor_count,
        "diagnostics": diagnostics,
        "final_answer": final_answer,
        "property_claims": claims,
        "prompt_policy": _reasoning_prompt_policy(),
        "expert_reasoning_prompt": _reasoning_prompt_instruction(query=query),
        "expert_reasoning": None,
    }


def _evidence_diagnostics(*, neighbors: list[dict[str, Any]], top_properties: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [float(item.get("distance")) for item in neighbors if _to_float(item.get("distance")) is not None]
    shared_counts = [int(item.get("shared_feature_count") or 0) for item in neighbors]
    property_support_counts = {
        str(item.get("property_name")): int(item.get("neighbor_count") or 0)
        for item in top_properties
        if item.get("property_name")
    }
    return {
        "neighbor_count": len(neighbors),
        "distance_min": round(min(distances), 6) if distances else None,
        "distance_max": round(max(distances), 6) if distances else None,
        "shared_feature_count_min": min(shared_counts) if shared_counts else 0,
        "shared_feature_count_max": max(shared_counts) if shared_counts else 0,
        "property_support_counts": property_support_counts,
        "diagnostic_role": "descriptive_evidence_metrics_not_confidence_labels",
    }


def _reasoning_prompt_policy() -> list[str]:
    return [
        "Act as a petroleum-chemistry expert, not as a generic nearest-neighbor regressor.",
        "Use retrieved historical samples as evidence carriers; do not copy a neighbor value as the final answer.",
        "Judge whether shared features are chemically relevant to the queried property. For example, density and API gravity are directly related, while some composition features may be indirect evidence.",
        "Consider distance, shared-feature basis, feature-channel relevance, property support counts, historical range, and conflicting evidence together.",
        "Do not invent numeric values outside the evidence table; if evidence is insufficient or chemically weak, state the limitation explicitly.",
        "Separate database evidence from any tool-direct, rule-based, or literature/RAG evidence.",
    ]


def _reasoning_prompt_instruction(*, query: str) -> str:
    return (
        "You are a petroleum-chemistry expert reasoning over an experimental-db-evidence.v1 bundle. "
        "Explain what counts as a similar sample by referring to the shared numerical feature space, distances, and analysis channels. "
        "Assess which shared features are relevant to the queried properties and why; note that different features can have different relevance. "
        "Use known petroleum-property relationships when applicable, such as the inverse relationship between density and API gravity, but do not invent unsupported values. "
        "Use neighbor property summaries as evidence, not as labels to copy. "
        "Return a concise final inference with evidence basis, limitations, and whether additional historical samples or analysis channels are needed. "
        f"User query: {query or 'property inference'}"
    )


def attach_petroleum_expert_reasoning(evidence: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Attach LLM expert interpretation without changing the deterministic evidence tables."""

    if not isinstance(evidence, dict) or evidence.get("status") != "success":
        return evidence
    presentation = evidence.get("presentation")
    if not isinstance(presentation, dict):
        return evidence
    constrained = presentation.get("constrained_reasoning")
    if not isinstance(constrained, dict):
        return evidence
    prompt = _build_expert_reasoning_prompt(evidence)
    constrained["expert_reasoning_prompt"] = prompt
    provider = getattr(llm, "provider", "unknown")
    model = getattr(llm, "model", None)
    system_prompt = (
        "You are ChemAnalyst's petroleum-chemistry expert reasoning module. "
        "Reason in professional scientific English. "
        "Use the structured historical Experimental DB evidence conservatively, "
        "distinguish measured evidence from inference, and do not invent property values."
    )
    selected_model = str(model or "").lower()
    max_tokens = 6000 if provider == "openai" and selected_model.startswith("gpt-5") else 1800
    try:
        if hasattr(llm, "chat_with_model"):
            answer = llm.chat_with_model(
                model=model or "",
                query=prompt,
                system_prompt=system_prompt,
                temperature=None if provider == "openai" else 0.1,
                max_tokens=max_tokens,
            )
        else:
            answer = llm.chat(prompt)
    except Exception as exc:
        constrained["expert_reasoning"] = {
            "status": "llm_reasoning_failed",
            "provider": provider,
            "model": model,
            "error": str(exc),
        }
    else:
        constrained["expert_reasoning"] = {
            "status": "success",
            "provider": provider,
            "model": model,
            "answer": answer,
        }
    return evidence


def _build_expert_reasoning_prompt(evidence: dict[str, Any]) -> str:
    compact_evidence = _compact_evidence_for_expert_reasoning(evidence)
    return (
        "You are a petroleum-chemistry expert. Perform evidence-constrained reasoning using the Historical Experimental DB evidence below.\n"
        "Do not behave like a simple KNN regressor. Do not decide confidence only by neighbor count.\n"
        "Your task:\n"
        "1. Define why the retrieved samples are or are not chemically similar to the query sample.\n"
        "2. Explain how each shared feature/channel contributes to the queried property inference, and identify weak or indirect features.\n"
        "3. Use petroleum knowledge where relevant, e.g. density/API gravity relationship, distillation-property meaning, hydrocarbon-type effects.\n"
        "4. If the evidence bundle declares an interpolation or weighting method, explicitly explain how the numerical evidence centers were obtained.\n"
        "5. For boiling-point or distillation properties, report numeric table values or explicit intervals; do not use shorthand such as '420s °C', 'mid-450s °C', or 'around 470 °C' when numerical evidence is available.\n"
        "6. Use weighted estimates and ranges as evidence only; do not copy them blindly as the final truth.\n"
        "7. If evidence is sparse, conflicting, or chemically insufficient, state what extra historical samples or analysis data are needed.\n"
        "8. Output in English with the following section headings: Similarity assessment, Feature contribution analysis, Property-evidence reasoning, Limitations, Final conclusion.\n\n"
        f"Evidence JSON:\n{json.dumps(compact_evidence, ensure_ascii=False, indent=2)}"
    )


def _compact_evidence_for_expert_reasoning(evidence: dict[str, Any]) -> dict[str, Any]:
    """Keep the expert prompt focused on evidence needed for reasoning."""

    presentation = evidence.get("presentation") if isinstance(evidence.get("presentation"), dict) else {}
    constrained = presentation.get("constrained_reasoning") if isinstance(presentation.get("constrained_reasoning"), dict) else {}
    return {
        "schema_version": evidence.get("schema_version"),
        "status": evidence.get("status"),
        "source": evidence.get("source"),
        "sample_id": evidence.get("sample_id"),
        "query": evidence.get("query"),
        "feature_basis": evidence.get("feature_basis"),
        "evidence_channels": evidence.get("evidence_channels"),
        "query_profile": evidence.get("query_profile"),
        "retrieval_policy": evidence.get("retrieval_policy"),
        "neighbor_count": evidence.get("neighbor_count"),
        "neighbors": [
            {
                "rank": index + 1,
                "sample_id": item.get("sample_id"),
                "sample_type": item.get("sample_type"),
                "distance": item.get("distance"),
                "shared_feature_count": item.get("shared_feature_count"),
                "shared_feature_examples": item.get("shared_feature_examples"),
                "available_channels": item.get("available_channels"),
            }
            for index, item in enumerate((evidence.get("neighbors") or [])[:12])
            if isinstance(item, dict)
        ],
        "property_summary": evidence.get("property_summary"),
        "presentation_tables": {
            "workflow": presentation.get("workflow"),
            "query_snapshot": presentation.get("query_snapshot"),
            "neighbor_table": presentation.get("neighbor_table"),
            "property_table": presentation.get("property_table"),
            "evidence_statement": presentation.get("evidence_statement"),
            "reasoning_notes": presentation.get("reasoning_notes"),
        },
        "constrained_reasoning": {
            "status": constrained.get("status"),
            "diagnostics": constrained.get("diagnostics"),
            "property_claims": constrained.get("property_claims"),
            "prompt_policy": constrained.get("prompt_policy"),
            "final_answer": constrained.get("final_answer"),
        },
    }


def _top_property_rows(neighbors: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    summary = _summarize_properties(neighbors)
    ranked_names = sorted(summary, key=lambda name: (-int(summary[name].get("neighbor_count") or 0), name))
    rows = []
    for name in ranked_names[:limit]:
        item = summary[name]
        rows.append(
            {
                "property_name": name,
                "weighted_estimate": item.get("weighted_estimate"),
                "range_min": (item.get("evidence_range") or [None, None])[0],
                "range_max": (item.get("evidence_range") or [None, None])[1],
                "neighbor_count": item.get("neighbor_count"),
                "supporting_samples": item.get("supporting_samples"),
            }
        )
    return rows


def _evidence_statement(
    *,
    sample_id: str | None,
    query: str,
    query_channels: list[str],
    neighbors: list[dict[str, Any]],
    top_properties: list[dict[str, Any]],
) -> str:
    if not neighbors:
        return "No comparable historical neighbor was retrieved, so no evidence statement can be produced."
    top_neighbor = neighbors[0]
    property_names = ", ".join(item["property_name"] for item in top_properties[:5]) or "bulk properties"
    channels = ", ".join(query_channels) or "imported analysis channels"
    return (
        f"For sample {sample_id or 'current_sample'}, the evidence test used {channels} to retrieve "
        f"{len(neighbors)} historical neighbors. The closest support sample is {top_neighbor.get('sample_id')} "
        f"(distance={top_neighbor.get('distance')}, shared_features={top_neighbor.get('shared_feature_count')}). "
        f"The main evidence-bearing properties summarized from the retrieved neighborhood are {property_names}. "
        f"This bundle is intended for historical-evidence reasoning in response to: {query or 'property inference'}."
    )


def _feature_distance(left: dict[str, float], right: dict[str, float]) -> tuple[float, list[str]]:
    shared = sorted(set(left) & set(right))
    if not shared:
        return math.inf, []
    squared = 0.0
    for key in shared:
        scale = max(abs(left[key]), abs(right[key]), 1.0)
        squared += ((left[key] - right[key]) / scale) ** 2
    penalty = 1.0 / max(len(shared), 1)
    return math.sqrt(squared / len(shared)) + penalty, shared


def _merge_profile(target: dict[str, Any], incoming: dict[str, Any], *, basis: str) -> None:
    if incoming.get("sample_id") and not target.get("sample_id"):
        target["sample_id"] = incoming["sample_id"]
    target["available_channels"].extend(incoming.get("available_channels") or [])
    target["available_channels"] = sorted(set(target["available_channels"]))
    target["feature_basis"].append(basis)
    target["feature_vector"].update(incoming.get("feature_vector") or {})
    target["bulk_properties"].update(incoming.get("bulk_properties") or {})


def _empty_transient_profile() -> dict[str, Any]:
    return {"sample_id": None, "available_channels": [], "feature_basis": [], "feature_vector": {}, "bulk_properties": {}}


def _normalize_analysis_type(value: Any) -> str:
    token = _safe_key(value or "")
    aliases = {
        "gc": "gc_fid",
        "gc-fid": "gc_fid",
        "gc_fid": "gc_fid",
        "htgc": "htgc_fid",
        "htgc-fid": "htgc_fid",
        "ncd": "gc_ncd",
        "scd": "gc_scd",
        "ir_spectrum": "ir",
        "ftir": "ir",
        "hydrocarbon_type": "hydrocarbon_types",
        "hydrocarbon_group_composition": "hydrocarbon_types",
        "molecular_formula": "molecular_composition",
    }
    return aliases.get(token, token if token in SUPPORTED_ANALYSIS_TYPES else token or "generic")


def _flatten_numeric(value: Any, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten_numeric(item, f"{prefix}:{_safe_key(key)}"))
    elif isinstance(value, list):
        for index, item in enumerate(value[:50]):
            out.update(_flatten_numeric(item, f"{prefix}:{index}"))
    else:
        numeric = _to_float(value)
        if numeric is not None:
            out[prefix] = numeric
    return out


def _weighted_quantile(points: list[tuple[float, float]], quantile: float) -> float:
    total = sum(weight for _, weight in points)
    threshold = total * quantile
    running = 0.0
    for value, weight in points:
        running += weight
        if running >= threshold:
            return value
    return points[-1][0]


def _first_numeric(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    normalized = {_safe_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(_safe_key(key))
        numeric = _to_float(value)
        if numeric is not None:
            return numeric
    return None


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    normalized = {_safe_key(key): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(_safe_key(key))
        if value not in (None, ""):
            return str(value)
    return ""


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace("%", "")
            if not value:
                return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _sample_id_from_features(features: dict[str, Any] | None) -> str | None:
    if not isinstance(features, dict):
        return None
    sample_meta = features.get("sample_meta")
    if isinstance(sample_meta, dict) and sample_meta.get("sample_id"):
        return str(sample_meta["sample_id"])
    return None


def _sample_id_from_tool_result(tool_result: dict[str, Any] | None) -> str | None:
    if not isinstance(tool_result, dict):
        return None
    for key in ("sample_id", "input_sample_id"):
        if tool_result.get(key):
            return str(tool_result[key])
    record = tool_result.get("experimental_record")
    if isinstance(record, dict) and record.get("sample_id"):
        return str(record["sample_id"])
    return None


def _safe_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff_.-]+", "_", text)
    return text.strip("_")


def _skipped(message: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SAMPLE_DB_EVIDENCE_SCHEMA,
        "status": "skipped",
        "source": "sample_centered_experimental_db",
        "message": message,
        **{key: value for key, value in extra.items() if value is not None},
    }
