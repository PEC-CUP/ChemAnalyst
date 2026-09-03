from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.tools.experimental_db import SampleDatabase
from app.tools.experimental_db.evidence_retriever import build_sample_centered_database_evidence


class SampleExperimentalDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(f"tests/_tmp_sample_experimental_db_{uuid4().hex}")
        shutil.rmtree(self.root, ignore_errors=True)
        self.database = SampleDatabase(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_json_import_indexes_properties_and_analysis_by_sample(self) -> None:
        result = self.database.import_payload(
            {
                "sample": {"sample_id": "petroleum_fraction-01", "sample_type": "petroleum fraction", "origin": "demo"},
                "bulk_properties": [
                    {"property_name": "density_20c", "value": 0.912, "unit": "g_cm-3"},
                    {"property_name": "BP_50", "value": 420, "unit": "degC"},
                ],
                "analyses": [
                    {
                        "analysis_type": "hydrocarbon_types",
                        "rows": [
                            {"component": "paraffins", "content": 36.2},
                            {"component": "extra_class", "content": 1.4},
                        ],
                    }
                ],
            }
        )

        self.assertEqual(result["sample_id"], "petroleum_fraction-01")
        record = self.database.get_sample("petroleum_fraction-01")
        self.assertEqual(record["sample"]["sample_type"], "petroleum fraction")
        self.assertEqual(len(record["bulk_properties"]), 2)
        self.assertEqual(record["analyses"][0]["rows"][1]["component"], "extra_class")
        self.assertEqual(self.database.status()["sample_count"], 1)
        self.assertEqual(self.database.list_samples()[0]["analysis_count"], 1)

    def test_generated_workbook_template_can_be_imported(self) -> None:
        workbook = self.root / "template.xlsx"
        workbook.write_bytes(self.database.template_xlsx())

        result = self.database.import_workbook(workbook)

        self.assertEqual(result["sample_id"], "sample_petroleum_fraction_001")
        record = self.database.get_sample("sample_petroleum_fraction_001")
        self.assertTrue(any(item["analysis_type"] == "gc_fid" for item in record["analyses"]))
        self.assertEqual(record["relations"], [])
        self.assertTrue((self.root / "templates" / "template_definition.json").exists())

    def test_workbook_relation_fields_live_in_sample_info(self) -> None:
        workbook = self.root / "fraction.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame(
                [
                    {"field": "sample_id", "value": "diesel_001"},
                    {"field": "sample_type", "value": "diesel"},
                    {"field": "related_sample_id", "value": "crude_001"},
                    {"field": "relation_type", "value": "fraction_of"},
                    {"field": "relation_description", "value": "diesel cut from crude_001"},
                ]
            ).to_excel(writer, sheet_name="sample_info", index=False)

        self.database.import_workbook(workbook)
        relation = self.database.get_sample("diesel_001")["relations"][0]

        self.assertEqual(relation["related_sample_id"], "crude_001")
        self.assertEqual(relation["relation_type"], "fraction_of")
        self.assertEqual(relation["metadata"]["description"], "diesel cut from crude_001")

    def test_sample_centered_retriever_compares_transient_hydrocarbon_tool_result(self) -> None:
        self.database.import_payload(
            {
                "sample": {"sample_id": "hist_petroleum_fraction_01", "sample_type": "petroleum fraction", "origin": "demo"},
                "bulk_properties": [
                    {"property_name": "density_20c", "value": 0.91, "unit": "g_cm-3"},
                    {"property_name": "BP_50", "value": 421, "unit": "degC"},
                ],
                "analyses": [
                    {
                        "analysis_type": "hydrocarbon_types",
                        "rows": [
                            {"component": "paraffins", "content": 36.0},
                            {"component": "aromatics", "content": 28.5},
                        ],
                    }
                ],
            }
        )
        self.database.import_payload(
            {
                "sample": {"sample_id": "hist_petroleum_fraction_02", "sample_type": "petroleum fraction", "origin": "demo"},
                "bulk_properties": [{"property_name": "density_20c", "value": 0.96, "unit": "g_cm-3"}],
                "analyses": [
                    {
                        "analysis_type": "hydrocarbon_types",
                        "rows": [
                            {"component": "paraffins", "content": 12.0},
                            {"component": "aromatics", "content": 58.0},
                        ],
                    }
                ],
            }
        )

        evidence = build_sample_centered_database_evidence(
            self.root,
            sample_id="session_sample",
            tool_result={
                "status": "success",
                "tool_evidence": {
                    "evidence_type": "hydrocarbon_group_composition",
                    "structured_result": [
                        {"component": "paraffins", "content": 35.8},
                        {"component": "aromatics", "content": 29.0},
                    ],
                },
            },
            query="infer density from hydrocarbon composition",
        )

        self.assertEqual(evidence["status"], "success")
        self.assertEqual(evidence["source"], "sample_centered_experimental_db")
        self.assertEqual(evidence["neighbors"][0]["sample_id"], "hist_petroleum_fraction_01")
        self.assertIn("hydrocarbon_types", evidence["evidence_channels"])
        self.assertIn("density_20c", evidence["property_summary"])
        self.assertIn("presentation", evidence)
        self.assertIn("neighbor_table", evidence["presentation"])

    def test_template_workbook_can_replace_active_definition(self) -> None:
        workbook = self.root / "custom_template.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            pd.DataFrame(
                [
                    {"field": "sample_id", "value": "lab_sample_001"},
                    {"field": "sample_name", "value": "Lab Sample 001"},
                    {"field": "sample_type", "value": "diesel"},
                ]
            ).to_excel(writer, sheet_name="sample_information", index=False)
            pd.DataFrame(
                [{"property_name": "density_20c", "value": "", "unit": "g_cm-3", "method": ""}]
            ).to_excel(writer, sheet_name="bulk_properties", index=False)
            pd.DataFrame(
                [{"retention_time": 10.1, "peak_area": 120.5, "peak_name": "nC16"}]
            ).to_excel(writer, sheet_name="custom_gc_profile", index=False)

        result = self.database.update_template_from_workbook(workbook)
        definition = self.database.get_template_definition()

        self.assertEqual(result["status"], "success")
        self.assertEqual(definition["sample_information_sheet"], "sample_information")
        self.assertTrue(any(sheet.get("analysis_type") == "custom_gc_profile" for sheet in definition["sheets"]))
        self.assertTrue((self.root / "templates" / "chemanalyst_experimental_sample_template.xlsx").exists())

    def test_delete_sample_removes_record_and_registry_entries(self) -> None:
        self.database.import_payload(
            {
                "sample_information": {"sample_id": "delete_me", "sample_type": "petroleum fraction"},
                "bulk_properties": [{"property_name": "density_20c", "value": 0.912}],
                "analyses": [{"analysis_type": "gc_fid", "rows": [{"rt": 1.0, "intensity": 2.0}]}],
            }
        )

        result = self.database.delete_sample("delete_me")

        self.assertEqual(result["status"], "success")
        self.assertIsNone(self.database.get_sample("delete_me"))
        self.assertEqual(self.database.status()["sample_count"], 0)
        registry = (self.root / "registry" / "samples.json").read_text(encoding="utf-8")
        self.assertNotIn("delete_me", registry)


if __name__ == "__main__":
    unittest.main()
