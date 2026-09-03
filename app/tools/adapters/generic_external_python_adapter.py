from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
from typing import Any, Callable

import pandas as pd


NAMED_RECORD_VALUE_MAPPINGS: dict[str, dict[str, dict[str, str]]] = {
    "hydrocarbon_type_en": {
        "item": {
            "\u94fe\u70f7\u70c3": "paraffins",
            "\u4e00\u73af\u70f7\u70c3": "monocycloparaffins",
            "\u4e8c\u73af\u70f7\u70c3": "dicycloparaffins",
            "\u4e09\u73af\u70f7\u70c3": "tricycloparaffins",
            "\u603b\u73af\u70f7\u70c3": "total_cycloparaffins",
            "\u603b\u9971\u548c\u70c3": "total_saturates",
            "\u70f7\u57fa\u82ef": "alkylbenzenes",
            "\u831a\u6ee1\u6216\u56db\u6c22\u8418": "indanes_or_tetralins",
            "\u8418": "naphthalene",
            "\u8418\u7c7b": "naphthalenes",
            "\u831a\u7c7b\u548c/\u6216CnH2n-10": "indenes_or_cnh2n_10",
            "\u82ca\u7c7b\u548c/\u6216CnH2n-14": "acenaphthenes_or_cnh2n_14",
            "\u82b4\u7c7b\u548c/\u6216CnH2n-16": "fluorenes_or_cnh2n_16",
            "\u4e09\u73af\u82b3\u70c3": "tricyclic_aromatics",
            "\u603b\u5355\u73af\u82b3\u70c3": "total_monoaromatics",
            "\u603b\u53cc\u73af\u82b3\u70c3": "total_diaromatics",
            "\u603b\u82b3\u70c3": "total_aromatics",
            "\u603b\u91cd\u91cf": "total_weight",
        }
    }
}


def build_adapter(manifest: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Bind a reviewed provider manifest to a generic external Python adapter."""

    def _adapter(payload: dict[str, Any]) -> dict[str, Any]:
        return run_with_manifest(payload, manifest)

    return _adapter


def run_with_manifest(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    invocation = manifest.get("invocation") if isinstance(manifest.get("invocation"), dict) else {}
    config = invocation.get("adapter_config") if isinstance(invocation.get("adapter_config"), dict) else {}
    script_path = Path(str(config.get("script_path") or invocation.get("entrypoint") or "")).expanduser()
    if not script_path.exists():
        return {"status": "error", "message": "script_path_not_found", "script_path": str(script_path)}

    input_field = str(config.get("input_field") or "input_file")
    input_file = Path(str(payload.get(input_field) or payload.get("input_file") or payload.get("file_path") or "")).expanduser()
    if not input_file.exists():
        return {"status": "error", "message": "input_file_not_found", "input_file": str(input_file)}

    provider_id = str(manifest.get("provider_id") or manifest.get("capability_id") or "generic_tool")
    output_dir = Path(str(payload.get("output_dir") or config.get("default_output_dir") or Path("data") / "tool_outputs" / provider_id)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_key = str(config.get("output_artifact_key") or "output_file")
    output_path = _build_output_path(input_file, output_dir, provider_id, config)

    try:
        module = _load_external_module(script_path)
        function_name = str(config.get("function_name") or "run")
        func = getattr(module, function_name, None)
        if not callable(func):
            return {"status": "error", "message": "function_not_found", "function_name": function_name, "script_path": str(script_path)}
        _call_function(func, payload, input_file, output_path, config)
        structured_result = _apply_record_value_mappings(_parse_output(output_path, config), config)
        _rewrite_output_with_mapped_records(output_path, structured_result, config)
        _rewrite_workbook_value_mappings(output_path, config)
        evidence_config = config.get("evidence") if isinstance(config.get("evidence"), dict) else {}
        quality_flags = _quality_flags(output_path, structured_result)
        return {
            "status": "success",
            "message": "generic_python_function_completed",
            "tool_evidence": {
                "schema_version": evidence_config.get("schema_version") or "generic-tool-evidence.v1",
                "capability_id": manifest.get("capability_id"),
                "evidence_type": evidence_config.get("evidence_type") or "structured_tool_output",
                "method": evidence_config.get("method") or manifest.get("tool_name"),
                "source_file": str(input_file),
                "structured_result": structured_result,
                "quality_flags": quality_flags,
            },
            "artifacts": {output_artifact_key: str(output_path)},
            "result": {
                "structured_result": structured_result,
                output_artifact_key: str(output_path),
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": "generic_python_function_failed",
            "error": str(exc),
            "script_path": str(script_path),
            "input_file": str(input_file),
        }


def _build_output_path(input_file: Path, output_dir: Path, provider_id: str, config: dict[str, Any]) -> Path:
    suffix = str(config.get("output_suffix") or ".xlsx")
    template = str(config.get("output_filename_template") or "{input_stem}_{provider_id}_{run_id}{suffix}")
    name = template.format(
        input_stem=input_file.stem,
        provider_id=_safe_token(provider_id),
        run_id=uuid.uuid4().hex[:8],
        suffix=suffix,
    )
    path = output_dir / name
    if not path.suffix:
        path = path.with_suffix(suffix)
    return path


def _load_external_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"chemanalyst_external_{uuid.uuid4().hex}", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load external Python script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_function(func: Callable[..., Any], payload: dict[str, Any], input_file: Path, output_path: Path, config: dict[str, Any]) -> Any:
    call_style = str(config.get("call_style") or "input_output_paths")
    if call_style == "payload":
        enriched = dict(payload)
        enriched.setdefault("input_file", str(input_file))
        enriched.setdefault("output_file", str(output_path))
        return func(enriched)
    if call_style == "keyword":
        kwargs = dict(config.get("static_kwargs") or {})
        kwargs[str(config.get("input_kwarg") or "input_file")] = str(input_file)
        kwargs[str(config.get("output_kwarg") or "output_file")] = str(output_path)
        return func(**kwargs)
    return func(str(input_file), str(output_path))


def _parse_output(output_path: Path, config: dict[str, Any]) -> Any:
    parser = config.get("output_parser") if isinstance(config.get("output_parser"), dict) else {}
    parser_type = str(parser.get("type") or "excel_sheet_records")
    if parser_type == "none":
        return {"output_file": str(output_path)}
    if parser_type == "csv_records":
        return pd.read_csv(output_path).to_dict(orient="records")
    sheet_name = parser.get("sheet_name") or 0
    return pd.read_excel(output_path, sheet_name=sheet_name).to_dict(orient="records")


def _apply_record_value_mappings(structured_result: Any, config: dict[str, Any]) -> Any:
    mapping_name = str(config.get("record_value_mapping_name") or "").strip()
    mappings = NAMED_RECORD_VALUE_MAPPINGS.get(mapping_name) if mapping_name else None
    if mappings is None:
        mappings = config.get("record_value_mappings")
    if not isinstance(mappings, dict) or not mappings:
        return structured_result
    if isinstance(structured_result, list):
        return [_map_record_values(record, mappings) for record in structured_result]
    if isinstance(structured_result, dict):
        return _map_record_values(structured_result, mappings)
    return structured_result


def _rewrite_output_with_mapped_records(output_path: Path, structured_result: Any, config: dict[str, Any]) -> None:
    """Keep the downloadable artifact consistent with mapped structured evidence."""

    if config.get("rewrite_output_with_mapped_records", True) is False:
        return
    if not isinstance(structured_result, list) or not all(isinstance(record, dict) for record in structured_result):
        return
    if output_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return
    parser = config.get("output_parser") if isinstance(config.get("output_parser"), dict) else {}
    if str(parser.get("type") or "excel_sheet_records") != "excel_sheet_records":
        return
    sheet_name = parser.get("sheet_name") or 0
    if not isinstance(sheet_name, str):
        return
    df = pd.DataFrame(structured_result)
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def _rewrite_workbook_value_mappings(output_path: Path, config: dict[str, Any]) -> None:
    """Normalize mapped labels across all downloadable Excel sheets."""

    if config.get("rewrite_output_with_mapped_records", True) is False:
        return
    if output_path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return
    mapping_name = str(config.get("record_value_mapping_name") or "").strip()
    mappings = NAMED_RECORD_VALUE_MAPPINGS.get(mapping_name) if mapping_name else None
    if mappings is None:
        mappings = config.get("record_value_mappings")
    if not isinstance(mappings, dict) or not mappings:
        return
    workbook = pd.ExcelFile(output_path)
    mapped_sheets: dict[str, pd.DataFrame] = {}
    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(output_path, sheet_name=sheet_name)
        records = [_map_record_values(record, mappings) for record in df.to_dict(orient="records")]
        mapped_sheets[sheet_name] = pd.DataFrame(records)
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        for sheet_name, df in mapped_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def _map_record_values(record: Any, mappings: dict[str, Any]) -> Any:
    if not isinstance(record, dict):
        return record
    mapped = dict(record)
    for field, value_mapping in mappings.items():
        if field not in mapped or not isinstance(value_mapping, dict):
            continue
        raw_value = str(mapped[field]).strip()
        mapped[field] = value_mapping.get(raw_value, mapped[field])
    return mapped


def _quality_flags(output_path: Path, structured_result: Any) -> list[str]:
    flags: list[str] = []
    if not output_path.exists():
        flags.append("output_file_missing")
    if structured_result in ({}, [], None):
        flags.append("empty_structured_result")
    return flags


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "tool"
