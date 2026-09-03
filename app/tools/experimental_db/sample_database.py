from __future__ import annotations

import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


SAMPLE_DB_SCHEMA = "experimental-sample-db.v1"
SAMPLE_TEMPLATE_SCHEMA = "experimental-sample-template.v2"
SAMPLE_INFORMATION_SHEETS = ("sample_information", "sample_info")
ANALYSIS_TYPES = {
    "ir": {"sheet": "ir_spectrum", "axes": ["wave_number_cm-1", "transmittance"]},
    "gc_fid": {"sheet": "gc_fid", "axes": ["rt", "intensity"]},
    "htgc_fid": {"sheet": "htgc_fid", "axes": ["rt", "intensity"]},
    "gc_ncd": {"sheet": "gc_ncd", "axes": ["rt", "intensity"]},
    "gc_scd": {"sheet": "gc_scd", "axes": ["rt", "intensity"]},
    "hydrocarbon_types": {"sheet": "hydrocarbon_types", "axes": ["component", "content"]},
    "molecular_composition": {
        "sheet": "molecular_composition",
        "axes": ["molecule_family", "molecular_formula", "content"],
    },
}
DISTILLATION_YIELDS = tuple(range(5, 100, 5))


class SampleDatabase:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.samples_root = self.root / "samples"
        self.objects_root = self.root / "objects"
        self.registry_root = self.root / "registry"
        self.templates_root = self.root / "templates"
        self.catalog_path = self.root / "catalog" / "experimental_samples.sqlite"
        self.template_definition_path = self.templates_root / "template_definition.json"
        self.template_workbook_path = self.templates_root / "chemanalyst_experimental_sample_template.xlsx"
        self.sample_import_example_path = self.templates_root / "sample_import.json"
        self._ensure_layout()

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            sample_count = connection.execute("select count(*) from samples").fetchone()[0]
            property_count = connection.execute("select count(*) from property_observations").fetchone()[0]
            analysis_count = connection.execute("select count(*) from analysis_records").fetchone()[0]
            artifact_count = connection.execute("select count(*) from artifacts").fetchone()[0]
        template_definition = self.get_template_definition()
        analysis_types = [
            sheet.get("analysis_type")
            for sheet in template_definition.get("sheets", [])
            if sheet.get("role") == "analysis" and sheet.get("analysis_type")
        ]
        return {
            "schema_version": SAMPLE_DB_SCHEMA,
            "root": str(self.root),
            "sample_count": sample_count,
            "property_observation_count": property_count,
            "analysis_record_count": analysis_count,
            "artifact_count": artifact_count,
            "storage": {
                "filesystem_objects": str(self.objects_root),
                "json_registry": str(self.registry_root),
                "sqlite_catalog": str(self.catalog_path),
            },
            "sample_information_sheet_names": list(SAMPLE_INFORMATION_SHEETS),
            "analysis_types": sorted(set(analysis_types)),
            "distillation_yields": list(DISTILLATION_YIELDS),
            "active_template": {
                "schema_version": template_definition.get("schema_version"),
                "updated_at": template_definition.get("updated_at"),
                "updated_from": template_definition.get("updated_from"),
                "sample_information_sheet": template_definition.get("sample_information_sheet"),
                "bulk_properties_sheet": template_definition.get("bulk_properties_sheet"),
                "analysis_sheet_count": len([sheet for sheet in template_definition.get("sheets", []) if sheet.get("role") == "analysis"]),
                "template_definition_file": str(self.template_definition_path),
                "template_workbook_file": str(self.template_workbook_path) if self.template_workbook_path.exists() else None,
            },
        }

    def get_template_definition(self) -> dict[str, Any]:
        if not self.template_definition_path.exists():
            definition = default_template_definition()
            self._write_template_definition(definition)
            return definition
        return json.loads(self.template_definition_path.read_text(encoding="utf-8"))

    def list_samples(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select s.sample_id, s.sample_name, s.sample_type, s.origin, s.refinery, s.updated_at,
                       count(distinct a.analysis_id) as analysis_count,
                       count(distinct p.observation_id) as property_count,
                       count(distinct af.object_id) as artifact_count
                from samples s
                left join analysis_records a on a.sample_id = s.sample_id
                left join property_observations p on p.sample_id = s.sample_id
                left join artifacts af on af.sample_id = s.sample_id
                group by s.sample_id
                order by s.updated_at desc
                limit ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        path = self._sample_path(sample_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return _public_record(record)

    def import_payload(self, payload: dict[str, Any], *, source: str = "json_ui") -> dict[str, Any]:
        sample = _normalize_sample(payload.get("sample_information") or payload.get("sample") or payload)
        sample_id = sample["sample_id"]
        existing = self._load_raw_sample(sample_id) or self._empty_sample(sample)
        existing["sample_information"].update({key: value for key, value in sample.items() if value not in (None, "")})
        existing["updated_at"] = _utc_now()

        property_rows = _property_rows(payload.get("bulk_properties") or payload.get("properties") or [])
        for row in property_rows:
            row["sample_id"] = sample_id
        analysis_rows = _analysis_rows(payload.get("analyses") or [], sample_id=sample_id)
        relation_rows = _relation_rows(payload.get("relations") or [], sample_id=sample_id)
        existing["bulk_properties"] = _merge_records(existing.get("bulk_properties") or [], property_rows, "observation_id")
        existing["analyses"] = _merge_records(existing.get("analyses") or [], analysis_rows, "analysis_id")
        existing["relations"] = _merge_records(existing.get("relations") or [], relation_rows, "relation_id")
        self._write_sample(existing)
        self._write_registries(existing, property_rows, analysis_rows, relation_rows, source=source)
        self._index_records(existing, property_rows, analysis_rows, relation_rows)
        return {
            "status": "success",
            "message": "sample_imported",
            "sample_id": sample_id,
            "properties_added": len(property_rows),
            "analyses_added": len(analysis_rows),
            "relations_added": len(relation_rows),
            "sample_file": str(self._sample_path(sample_id)),
        }

    def import_workbook(self, workbook: Path) -> dict[str, Any]:
        payload = workbook_to_payload(workbook, template_definition=self.get_template_definition())
        result = self.import_payload(payload, source=f"workbook:{Path(workbook).name}")
        object_ref = self.store_object(workbook, sample_id=result["sample_id"], object_kind="import_workbook")
        result["import_workbook"] = object_ref
        return result

    def update_template_from_workbook(self, workbook: Path) -> dict[str, Any]:
        definition = infer_template_definition(workbook)
        definition["updated_at"] = _utc_now()
        definition["updated_from"] = Path(workbook).name
        shutil.copy2(workbook, self.template_workbook_path)
        self._write_template_definition(definition)
        self._write_sample_import_example()
        return {
            "status": "success",
            "message": "template_updated",
            "template_definition_file": str(self.template_definition_path),
            "template_workbook_file": str(self.template_workbook_path),
            "sample_information_sheet": definition.get("sample_information_sheet"),
            "bulk_properties_sheet": definition.get("bulk_properties_sheet"),
            "analysis_types": [
                sheet.get("analysis_type")
                for sheet in definition.get("sheets", [])
                if sheet.get("role") == "analysis"
            ],
        }

    def delete_sample(self, sample_id: str) -> dict[str, Any]:
        sample_id = str(sample_id or "").strip()
        if not sample_id:
            raise ValueError("sample_id is required")
        existing = self._load_raw_sample(sample_id)
        if not existing:
            raise ValueError(f"Unknown sample_id: {sample_id}")

        sample_dir = self._sample_path(sample_id).parent
        object_dir = self.objects_root / _safe_token(sample_id)
        if sample_dir.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
        if object_dir.exists():
            shutil.rmtree(object_dir, ignore_errors=True)

        with self._connect() as connection:
            connection.execute("delete from artifacts where sample_id = ?", (sample_id,))
            connection.execute("delete from analysis_records where sample_id = ?", (sample_id,))
            connection.execute("delete from property_observations where sample_id = ?", (sample_id,))
            connection.execute("delete from sample_relations where sample_id = ?", (sample_id,))
            connection.execute("delete from samples where sample_id = ?", (sample_id,))

        self._rebuild_registries()
        return {"status": "success", "message": "sample_deleted", "sample_id": sample_id}

    def add_artifact(
        self,
        sample_id: str,
        source_path: Path,
        *,
        analysis_type: str,
        method: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        existing = self._load_raw_sample(sample_id)
        if not existing:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        object_ref = self.store_object(source_path, sample_id=sample_id, object_kind=analysis_type)
        analysis = _analysis_rows(
            [
                {
                    "analysis_type": analysis_type,
                    "method": method,
                    "description": description,
                    "artifact": object_ref,
                    "rows": [],
                }
            ],
            sample_id=sample_id,
        )[0]
        existing["analyses"].append(analysis)
        existing["updated_at"] = _utc_now()
        self._write_sample(existing)
        self._write_registries(existing, [], [analysis], [], source="artifact_upload")
        self._index_records(existing, [], [analysis], [])
        return {"status": "success", "sample_id": sample_id, "analysis": analysis, "artifact": object_ref}

    def store_object(self, source_path: Path, *, sample_id: str, object_kind: str) -> dict[str, Any]:
        source = Path(source_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(source)
        object_id = uuid4().hex
        suffix = source.suffix or ".bin"
        target = self.objects_root / _safe_token(sample_id) / f"{object_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record = {
            "object_id": object_id,
            "sample_id": sample_id,
            "object_kind": object_kind,
            "filename": source.name,
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "created_at": _utc_now(),
        }
        _append_jsonl(self.registry_root / "artifacts.jsonl", record)
        with self._connect() as connection:
            connection.execute(
                """
                insert into artifacts(object_id, sample_id, object_kind, filename, path, size_bytes, created_at)
                values(:object_id, :sample_id, :object_kind, :filename, :path, :size_bytes, :created_at)
                """,
                record,
            )
        return record

    def template_xlsx(self) -> bytes:
        if self.template_workbook_path.exists():
            return self.template_workbook_path.read_bytes()
        definition = self.get_template_definition()
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet in definition.get("sheets", []):
                rows = sheet.get("example_rows") or [{}]
                pd.DataFrame(rows).to_excel(writer, sheet_name=str(sheet["sheet_name"]), index=False)
        return output.getvalue()

    def _ensure_layout(self) -> None:
        for directory in (self.samples_root, self.objects_root, self.registry_root, self.templates_root, self.catalog_path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        journal = self.catalog_path.with_name(f"{self.catalog_path.name}-journal")
        if self.catalog_path.exists() and self.catalog_path.stat().st_size == 0 and journal.exists():
            self.catalog_path.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
        samples_registry = self.registry_root / "samples.json"
        if not samples_registry.exists():
            samples_registry.write_text("{}", encoding="utf-8")
        self._migrate_template_files()
        with self._connect() as connection:
            connection.executescript(_sqlite_schema())
        if self._catalog_has_missing_sample_files():
            self._rebuild_catalog_from_files()

    def _migrate_template_files(self) -> None:
        if not self.template_definition_path.exists():
            self._write_template_definition(default_template_definition())
        self._write_sample_import_example()
        legacy_schema = self.templates_root / "postgres_schema.sql"
        legacy_schema.unlink(missing_ok=True)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.catalog_path)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma journal_mode = off")
        connection.execute("pragma foreign_keys = on")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _write_sample(self, record: dict[str, Any]) -> None:
        path = self._sample_path(record["sample_information"]["sample_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_raw_sample(self, sample_id: str) -> dict[str, Any] | None:
        path = self._sample_path(sample_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return _normalize_record_shape(record)

    def _write_registries(
        self,
        record: dict[str, Any],
        properties: list[dict[str, Any]],
        analyses: list[dict[str, Any]],
        relations: list[dict[str, Any]],
        *,
        source: str,
    ) -> None:
        sample = record["sample_information"]
        registry_path = self.registry_root / "samples.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry[sample["sample_id"]] = {
            **sample,
            "sample_file": str(self._sample_path(sample["sample_id"])),
            "updated_at": record["updated_at"],
            "source": source,
        }
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        for path, rows in (
            (self.registry_root / "property_observations.jsonl", properties),
            (self.registry_root / "analysis_records.jsonl", analyses),
            (self.registry_root / "sample_relations.jsonl", relations),
        ):
            for row in rows:
                _append_jsonl(path, row)

    def _rebuild_registries(self) -> None:
        samples_registry: dict[str, Any] = {}
        property_rows: list[dict[str, Any]] = []
        analysis_rows: list[dict[str, Any]] = []
        relation_rows: list[dict[str, Any]] = []
        for sample_dir in sorted(self.samples_root.iterdir()) if self.samples_root.exists() else []:
            sample_file = sample_dir / "sample.json"
            if not sample_file.exists():
                continue
            record = _normalize_record_shape(json.loads(sample_file.read_text(encoding="utf-8")))
            sample = record.get("sample_information") or {}
            sample_id = str(sample.get("sample_id") or "")
            if not sample_id:
                continue
            samples_registry[sample_id] = {
                **sample,
                "sample_file": str(sample_file),
                "updated_at": record.get("updated_at"),
                "source": "registry_rebuild",
            }
            property_rows.extend(record.get("bulk_properties") or [])
            analysis_rows.extend(record.get("analyses") or [])
            relation_rows.extend(record.get("relations") or [])
        (self.registry_root / "samples.json").write_text(json.dumps(samples_registry, ensure_ascii=False, indent=2), encoding="utf-8")
        for path, rows in (
            (self.registry_root / "property_observations.jsonl", property_rows),
            (self.registry_root / "analysis_records.jsonl", analysis_rows),
            (self.registry_root / "sample_relations.jsonl", relation_rows),
        ):
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            path.write_text(text, encoding="utf-8")

    def _catalog_has_missing_sample_files(self) -> bool:
        with self._connect() as connection:
            rows = connection.execute("select sample_id from samples limit 20").fetchall()
        return any(not self._sample_path(str(row["sample_id"])).exists() for row in rows)

    def _rebuild_catalog_from_files(self) -> None:
        with self._connect() as connection:
            for table in ("sample_relations", "artifacts", "analysis_records", "property_observations", "samples"):
                connection.execute(f"delete from {table}")
        for sample_dir in sorted(self.samples_root.iterdir()) if self.samples_root.exists() else []:
            sample_file = sample_dir / "sample.json"
            if not sample_file.exists():
                continue
            record = _normalize_record_shape(json.loads(sample_file.read_text(encoding="utf-8")))
            properties = record.get("bulk_properties") or []
            analyses = record.get("analyses") or []
            relations = record.get("relations") or []
            self._index_records(record, properties, analyses, relations)
        self._reindex_artifacts_from_registry()

    def _reindex_artifacts_from_registry(self) -> None:
        artifact_path = self.registry_root / "artifacts.jsonl"
        if not artifact_path.exists():
            return
        valid_sample_ids = {path.name for path in self.samples_root.iterdir() if path.is_dir()}
        with self._connect() as connection:
            for line in artifact_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(record.get("sample_id") or "")
                if _safe_token(sample_id) not in valid_sample_ids:
                    continue
                connection.execute(
                    """
                    insert or replace into artifacts(object_id, sample_id, object_kind, filename, path, size_bytes, created_at)
                    values(:object_id, :sample_id, :object_kind, :filename, :path, :size_bytes, :created_at)
                    """,
                    record,
                )

    def _write_template_definition(self, definition: dict[str, Any]) -> None:
        self.template_definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_sample_import_example(self) -> None:
        payload = {
            "sample_information": {"sample_id": "sample_petroleum_fraction_001", "sample_name": "petroleum fraction 001", "sample_type": "petroleum fraction", "origin": "pilot"},
            "bulk_properties": [{"property_name": "density_20c", "value": 0.912, "unit": "g_cm-3"}],
            "analyses": [{"analysis_type": "hydrocarbon_types", "rows": [{"component": "paraffins", "content": 36.2, "unit": "wt_percent"}]}],
        }
        self.sample_import_example_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _index_records(
        self,
        record: dict[str, Any],
        properties: list[dict[str, Any]],
        analyses: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> None:
        sample = record["sample_information"]
        with self._connect() as connection:
            connection.execute(
                """
                insert into samples(sample_id, sample_name, sample_type, origin, refinery, project, metadata_json, updated_at)
                values(:sample_id, :sample_name, :sample_type, :origin, :refinery, :project, :metadata_json, :updated_at)
                on conflict(sample_id) do update set
                  sample_name=excluded.sample_name, sample_type=excluded.sample_type, origin=excluded.origin,
                  refinery=excluded.refinery, project=excluded.project, metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                {
                    "sample_id": sample["sample_id"],
                    "sample_name": sample.get("sample_name"),
                    "sample_type": sample.get("sample_type"),
                    "origin": sample.get("origin"),
                    "refinery": sample.get("refinery"),
                    "project": sample.get("project"),
                    "metadata_json": json.dumps(sample, ensure_ascii=False),
                    "updated_at": record["updated_at"],
                },
            )
            for row in properties:
                connection.execute(
                    """
                    insert or replace into property_observations
                    (observation_id, sample_id, property_name, property_group, value_json, unit, method, created_at)
                    values(:observation_id, :sample_id, :property_name, :property_group, :value_json, :unit, :method, :created_at)
                    """,
                    {**row, "value_json": json.dumps(row.get("value"), ensure_ascii=False)},
                )
            for row in analyses:
                connection.execute(
                    """
                    insert or replace into analysis_records
                    (analysis_id, sample_id, analysis_type, method, data_json, artifact_json, created_at)
                    values(:analysis_id, :sample_id, :analysis_type, :method, :data_json, :artifact_json, :created_at)
                    """,
                    {
                        **row,
                        "data_json": json.dumps(row.get("rows") or [], ensure_ascii=False),
                        "artifact_json": json.dumps(row.get("artifact") or {}, ensure_ascii=False),
                    },
                )
            for row in relations:
                connection.execute(
                    """
                    insert or replace into sample_relations
                    (relation_id, sample_id, related_sample_id, relation_type, metadata_json, created_at)
                    values(:relation_id, :sample_id, :related_sample_id, :relation_type, :metadata_json, :created_at)
                    """,
                    {**row, "metadata_json": json.dumps(row.get("metadata") or {}, ensure_ascii=False)},
                )

    def _sample_path(self, sample_id: str) -> Path:
        return self.samples_root / _safe_token(sample_id) / "sample.json"

    def _empty_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": SAMPLE_DB_SCHEMA,
            "created_at": now,
            "updated_at": now,
            "sample_information": sample,
            "bulk_properties": [],
            "analyses": [],
            "relations": [],
        }


def workbook_to_payload(workbook: Path, *, template_definition: dict[str, Any] | None = None) -> dict[str, Any]:
    definition = template_definition or default_template_definition()
    sheets = pd.read_excel(workbook, sheet_name=None)
    indexed = {str(name): frame.dropna(how="all") for name, frame in sheets.items()}
    sample_sheet_name = str(definition.get("sample_information_sheet") or "sample_information")
    bulk_sheet_name = str(definition.get("bulk_properties_sheet") or "bulk_properties")
    sample_frame = indexed.get(sample_sheet_name)
    if sample_frame is None:
        sample_frame = _first_available_sheet(indexed, SAMPLE_INFORMATION_SHEETS)
    sample = _sample_from_sheet(sample_frame)
    if not sample.get("sample_id"):
        raise ValueError("sample_information sheet must provide sample_id")
    property_frame = indexed.get(bulk_sheet_name)
    if property_frame is None:
        property_frame = indexed.get("bulk_properties")
    properties = _properties_from_sheet(property_frame)
    analyses: list[dict[str, Any]] = []
    for sheet in definition.get("sheets", []):
        if sheet.get("role") != "analysis":
            continue
        sheet_name = str(sheet.get("sheet_name") or "")
        frame = indexed.get(sheet_name)
        if frame is None or frame.empty:
            continue
        analysis_type = str(sheet.get("analysis_type") or _safe_token(sheet_name))
        analyses.append({"analysis_type": analysis_type, "rows": _frame_records(frame)})
    relations = _relations_from_sample_info(sample)
    relation_sheet = next((sheet for sheet in definition.get("sheets", []) if sheet.get("role") == "relations"), None)
    if relation_sheet:
        frame = indexed.get(str(relation_sheet.get("sheet_name") or ""))
        if frame is not None:
            relations.extend(_frame_records(frame))
    elif indexed.get("sample_relations") is not None:
        relations.extend(_frame_records(indexed["sample_relations"]))
    return {"sample_information": sample, "bulk_properties": properties, "analyses": analyses, "relations": relations}


def infer_template_definition(workbook: Path) -> dict[str, Any]:
    sheets = pd.read_excel(workbook, sheet_name=None)
    if not sheets:
        raise ValueError("template workbook must contain at least one sheet")
    definition = {
        "schema_version": SAMPLE_TEMPLATE_SCHEMA,
        "sample_information_sheet": "sample_information",
        "bulk_properties_sheet": "bulk_properties",
        "sheets": [],
    }
    sample_sheet_name = None
    for name, frame in sheets.items():
        columns = [str(column).strip() for column in frame.columns.tolist() if str(column).strip()]
        compact_name = _safe_token(name)
        example_rows = _example_rows(frame)
        if compact_name in {_safe_token(item) for item in SAMPLE_INFORMATION_SHEETS}:
            sample_sheet_name = str(name)
            role = "sample_information"
            layout = "field_value" if {"field", "value"}.issubset({_safe_token(column) for column in columns}) else "record"
            definition["sample_information_sheet"] = str(name)
            definition["sheets"].append(
                {
                    "sheet_name": str(name),
                    "role": role,
                    "layout": layout,
                    "columns": columns,
                    "example_rows": example_rows or [{"field": "sample_id", "value": "sample_001"}],
                }
            )
            continue
        if compact_name == "bulk_properties":
            definition["bulk_properties_sheet"] = str(name)
            definition["sheets"].append(
                {
                    "sheet_name": str(name),
                    "role": "bulk_properties",
                    "columns": columns or ["property_name", "value", "unit", "method"],
                    "example_rows": example_rows or [{"property_name": "density_20c", "value": "", "unit": "g_cm-3", "method": ""}],
                }
            )
            continue
        if compact_name == "sample_relations":
            definition["sheets"].append(
                {
                    "sheet_name": str(name),
                    "role": "relations",
                    "columns": columns or ["related_sample_id", "relation_type", "description"],
                    "example_rows": example_rows or [{"related_sample_id": "crude_001", "relation_type": "fraction_of", "description": ""}],
                }
            )
            continue
        definition["sheets"].append(
            {
                "sheet_name": str(name),
                "role": "analysis",
                "analysis_type": compact_name,
                "columns": columns,
                "example_rows": example_rows or [{column: "" for column in (columns or ["value"])}],
            }
        )
    if not sample_sheet_name:
        raise ValueError("template workbook must contain a sample_information sheet")
    if not any(sheet.get("role") == "bulk_properties" for sheet in definition["sheets"]):
        definition["sheets"].insert(
            1,
            {
                "sheet_name": "bulk_properties",
                "role": "bulk_properties",
                "columns": ["property_name", "value", "unit", "method"],
                "example_rows": _bulk_property_template_rows()[:6],
            },
        )
    return definition


def default_template_definition() -> dict[str, Any]:
    sheets = [
        {
            "sheet_name": "sample_information",
            "role": "sample_information",
            "layout": "field_value",
            "columns": ["field", "value"],
            "example_rows": [
                {"field": "sample_id", "value": "sample_petroleum_fraction_001"},
                {"field": "sample_name", "value": "petroleum fraction 001"},
                {"field": "sample_type", "value": "petroleum fraction"},
                {"field": "origin", "value": "optional origin"},
                {"field": "refinery", "value": "optional refinery"},
                {"field": "project", "value": "optional project"},
                {"field": "related_sample_id", "value": ""},
                {"field": "relation_type", "value": ""},
                {"field": "relation_description", "value": ""},
            ],
        },
        {
            "sheet_name": "bulk_properties",
            "role": "bulk_properties",
            "columns": ["property_name", "value", "unit", "method"],
            "example_rows": _bulk_property_template_rows(),
        },
        {
            "sheet_name": "ir_spectrum",
            "role": "analysis",
            "analysis_type": "ir",
            "columns": ["wave_number_cm-1", "transmittance"],
            "example_rows": [{"wave_number_cm-1": 3000.0, "transmittance": 0.82}],
        },
    ]
    for analysis_type in ("gc_fid", "htgc_fid", "gc_ncd", "gc_scd"):
        sheets.append(
            {
                "sheet_name": analysis_type,
                "role": "analysis",
                "analysis_type": analysis_type,
                "columns": ["rt", "intensity"],
                "example_rows": [{"rt": 38.18, "intensity": 1200.0}],
            }
        )
    sheets.extend(
        [
            {
                "sheet_name": "hydrocarbon_types",
                "role": "analysis",
                "analysis_type": "hydrocarbon_types",
                "columns": ["component", "content", "unit"],
                "example_rows": [{"component": "paraffins", "content": 36.2, "unit": "wt_percent"}],
            },
            {
                "sheet_name": "molecular_composition",
                "role": "analysis",
                "analysis_type": "molecular_composition",
                "columns": ["molecule_family", "molecular_formula", "content", "unit", "dbe"],
                "example_rows": [
                    {
                        "molecule_family": "saturated_hydrocarbons",
                        "molecular_formula": "C20H42",
                        "content": 0.013,
                        "unit": "mass_fraction",
                        "dbe": 0,
                    }
                ],
            },
        ]
    )
    return {
        "schema_version": SAMPLE_TEMPLATE_SCHEMA,
        "updated_at": _utc_now(),
        "updated_from": "built_in_default",
        "sample_information_sheet": "sample_information",
        "bulk_properties_sheet": "bulk_properties",
        "sheets": sheets,
    }


def _normalize_sample(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("sample payload must be an object")
    sample_id = str(raw.get("sample_id") or raw.get("id") or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required")
    normalized = {
        "sample_id": sample_id,
        "sample_name": str(raw.get("sample_name") or raw.get("name") or sample_id).strip(),
        "sample_type": _optional_str(raw.get("sample_type")),
        "origin": _optional_str(raw.get("origin")),
        "refinery": _optional_str(raw.get("refinery")),
        "project": _optional_str(raw.get("project")),
    }
    extras = {key: value for key, value in raw.items() if key not in {*normalized, "id", "name"}}
    if extras:
        normalized["metadata"] = extras
    return normalized


def _property_rows(raw: Any) -> list[dict[str, Any]]:
    rows = []
    iterable = raw.items() if isinstance(raw, dict) else raw
    for item in iterable or []:
        if isinstance(item, tuple):
            item = {"property_name": item[0], "value": item[1]}
        if not isinstance(item, dict):
            continue
        property_name = str(item.get("property_name") or item.get("name") or "").strip()
        if not property_name:
            continue
        rows.append(
            {
                "observation_id": str(item.get("observation_id") or uuid4().hex),
                "sample_id": str(item.get("sample_id") or ""),
                "property_name": property_name,
                "property_group": str(item.get("property_group") or _property_group(property_name)),
                "value": _json_value(item.get("value")),
                "unit": _optional_str(item.get("unit")),
                "method": _optional_str(item.get("method")),
                "created_at": str(item.get("created_at") or _utc_now()),
            }
        )
    return rows


def _analysis_rows(raw: Any, *, sample_id: str) -> list[dict[str, Any]]:
    rows = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        analysis_type = str(item.get("analysis_type") or item.get("type") or "").strip().lower()
        if not analysis_type:
            continue
        rows.append(
            {
                "analysis_id": str(item.get("analysis_id") or uuid4().hex),
                "sample_id": sample_id,
                "analysis_type": analysis_type,
                "method": _optional_str(item.get("method")),
                "description": _optional_str(item.get("description")),
                "rows": _json_value(item.get("rows") or item.get("data") or []),
                "artifact": item.get("artifact") if isinstance(item.get("artifact"), dict) else {},
                "created_at": str(item.get("created_at") or _utc_now()),
            }
        )
    return rows


def _relation_rows(raw: Any, *, sample_id: str) -> list[dict[str, Any]]:
    rows = []
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("related_sample_id"):
            continue
        rows.append(
            {
                "relation_id": str(item.get("relation_id") or uuid4().hex),
                "sample_id": sample_id,
                "related_sample_id": str(item["related_sample_id"]),
                "relation_type": str(item.get("relation_type") or "related"),
                "metadata": _relation_metadata(item),
                "created_at": str(item.get("created_at") or _utc_now()),
            }
        )
    return rows


def _relation_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    reserved = {"relation_id", "sample_id", "related_sample_id", "relation_type", "metadata", "created_at"}
    return {**metadata, **{key: value for key, value in item.items() if key not in reserved}}


def _sample_from_sheet(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    columns = {_safe_token(column): column for column in frame.columns}
    if "field" in columns and "value" in columns:
        return {
            str(row[columns["field"]]).strip(): _json_value(row[columns["value"]])
            for _, row in frame.iterrows()
            if pd.notna(row[columns["field"]])
        }
    return _frame_records(frame)[0]


def _relations_from_sample_info(sample: dict[str, Any]) -> list[dict[str, Any]]:
    related_sample_id = sample.pop("related_sample_id", None)
    relation_type = sample.pop("relation_type", None)
    description = sample.pop("relation_description", None)
    if related_sample_id in (None, "") or relation_type in (None, ""):
        return []
    relation = {"related_sample_id": related_sample_id, "relation_type": relation_type}
    if description not in (None, ""):
        relation["description"] = description
    return [relation]


def _properties_from_sheet(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    rows = []
    for row in _frame_records(frame):
        property_name = row.get("property_name") or row.get("name")
        if property_name:
            rows.append(row)
    return rows


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(key).strip(): _json_value(value)
            for key, value in row.items()
            if str(key).strip() and not _missing(value)
        }
        for row in frame.to_dict(orient="records")
    ]


def _merge_records(old: list[dict[str, Any]], new: list[dict[str, Any]], id_key: str) -> list[dict[str, Any]]:
    merged = {str(item.get(id_key)): item for item in old if item.get(id_key)}
    for item in new:
        merged[str(item[id_key])] = item
    return list(merged.values())


def _first_available_sheet(indexed: dict[str, pd.DataFrame], names: tuple[str, ...]) -> pd.DataFrame | None:
    lowered = {_safe_token(key): value for key, value in indexed.items()}
    for name in names:
        frame = lowered.get(_safe_token(name))
        if frame is not None:
            return frame
    return None


def _bulk_property_template_rows() -> list[dict[str, Any]]:
    scalar = [
        ("C", "wt_percent"),
        ("H", "wt_percent"),
        ("O", "wt_percent"),
        ("N", "wt_percent"),
        ("S", "wt_percent"),
        ("saturates", "wt_percent"),
        ("aromatics", "wt_percent"),
        ("resins", "wt_percent"),
        ("asphaltenes", "wt_percent"),
        ("density_20c", "g_cm-3"),
        ("pour_point", "degC"),
        ("viscosity", "mPa_s"),
        ("wax_content", "wt_percent"),
        ("research_octane_number", ""),
        ("cetane_number", ""),
        ("api_gravity", "degAPI"),
    ]
    rows = [{"property_name": name, "value": "", "unit": unit, "method": ""} for name, unit in scalar]
    rows.extend(
        {"property_name": f"BP_{yield_percent}", "value": "", "unit": "degC", "method": "distillation"}
        for yield_percent in DISTILLATION_YIELDS
    )
    return rows


def _property_group(property_name: str) -> str:
    lowered = property_name.lower()
    if lowered.startswith("bp_"):
        return "distillation"
    if lowered in {"c", "h", "o", "n", "s"}:
        return "elemental"
    if lowered in {"saturates", "aromatics", "resins", "asphaltenes"}:
        return "sara"
    return "bulk_property"


def _sqlite_schema() -> str:
    return """
    create table if not exists samples(
      sample_id text primary key, sample_name text, sample_type text, origin text, refinery text,
      project text, metadata_json text not null, updated_at text not null
    );
    create table if not exists property_observations(
      observation_id text primary key, sample_id text not null references samples(sample_id),
      property_name text not null, property_group text not null, value_json text, unit text,
      method text, created_at text not null
    );
    create table if not exists analysis_records(
      analysis_id text primary key, sample_id text not null references samples(sample_id),
      analysis_type text not null, method text, data_json text not null, artifact_json text not null,
      created_at text not null
    );
    create table if not exists artifacts(
      object_id text primary key, sample_id text not null references samples(sample_id),
      object_kind text not null, filename text not null, path text not null, size_bytes integer not null,
      created_at text not null
    );
    create table if not exists sample_relations(
      relation_id text primary key, sample_id text not null references samples(sample_id),
      related_sample_id text not null, relation_type text not null, metadata_json text not null,
      created_at text not null
    );
    create index if not exists idx_properties_sample on property_observations(sample_id, property_group);
    create index if not exists idx_analysis_sample on analysis_records(sample_id, analysis_type);
    """


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _example_rows(frame: pd.DataFrame, *, limit: int = 8) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    rows = _frame_records(frame)
    return rows[:limit]


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(value).strip()).strip("_")
    return token or "sample"


def _json_value(value: Any) -> Any:
    if _missing(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_str(value: Any) -> str | None:
    if value in (None, "") or _missing(value):
        return None
    return str(value).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_record_shape(record: dict[str, Any]) -> dict[str, Any]:
    if "sample_information" in record:
        return record
    if "sample" in record:
        record["sample_information"] = record.pop("sample")
    return record


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_record_shape(record)
    sample_information = normalized.get("sample_information") or {}
    return {
        **normalized,
        "sample_information": sample_information,
        "sample": sample_information,
        "analysis_data": {
            "bulk_properties": normalized.get("bulk_properties") or [],
            "analyses": normalized.get("analyses") or [],
        },
    }
