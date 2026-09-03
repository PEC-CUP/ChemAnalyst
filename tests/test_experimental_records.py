from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.tools.crude_oil_db.experimental_records import (
    build_channel_database_evidence,
    build_tool_result_channel_database_evidence,
    build_tool_result_record,
    extract_flexible_numeric_features,
    persist_experimental_record,
)


class ExperimentalRecordTests(unittest.TestCase):
    def test_flexible_hydrocarbon_features_do_not_require_fixed_category_count(self) -> None:
        features = extract_flexible_numeric_features(
            [
                {"sample": "a", "item": "chain alkane", "mass_percent": 33.2},
                {"sample": "a", "item": "mono aromatic", "mass_percent": 12.1},
                {"sample": "a", "item": "optional extra class", "mass_percent": 4.7},
            ]
        )

        self.assertEqual(len(features), 3)
        self.assertTrue(any(item["label"] == "optional extra class" for item in features.values()))

    def test_channel_comparison_uses_shared_result_labels(self) -> None:
        root = Path("tests/_tmp_experimental_records")
        try:
            shutil.rmtree(root, ignore_errors=True)
            query = build_tool_result_record(
                sample_id="query",
                capability_id="tool.hydrocarbon_type_calculator",
                channel_id="hydrocarbon_type",
                tool_evidence={
                    "schema_version": "hydrocarbon-type.v1",
                    "evidence_type": "hydrocarbon_group_composition",
                    "structured_result": [
                        {"item": "chain alkane", "mass_percent": 30.0},
                        {"item": "mono aromatic", "mass_percent": 10.0},
                    ],
                },
            )
            historical = build_tool_result_record(
                sample_id="history_13_classes",
                capability_id="tool.hydrocarbon_type_calculator",
                channel_id="hydrocarbon_type",
                tool_evidence={
                    "schema_version": "hydrocarbon-type.v1",
                    "evidence_type": "hydrocarbon_group_composition",
                    "structured_result": [
                        {"item": "chain alkane", "mass_percent": 31.0},
                        {"item": "mono aromatic", "mass_percent": 11.0},
                        {"item": "extra class", "mass_percent": 2.0},
                    ],
                },
            )
            persist_experimental_record(root, historical)

            evidence = build_channel_database_evidence(root, query, query="infer density", history_limit=5)

            self.assertEqual(evidence["status"], "success")
            self.assertEqual(evidence["channel_id"], "hydrocarbon_type")
            self.assertEqual(evidence["neighbors"][0]["shared_feature_count"], 2)
            self.assertTrue(evidence["comparison_policy"]["does_not_require_fixed_component_count"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_current_tool_result_compares_without_session_record_persistence(self) -> None:
        root = Path("tests/_tmp_experimental_records_tool_result")
        try:
            shutil.rmtree(root, ignore_errors=True)
            historical = build_tool_result_record(
                sample_id="curated_history",
                capability_id="tool.hydrocarbon_type_calculator",
                channel_id="hydrocarbon_type",
                tool_evidence={
                    "schema_version": "hydrocarbon-type.v1",
                    "evidence_type": "hydrocarbon_group_composition",
                    "structured_result": [{"item": "chain alkane", "mass_percent": 31.0}],
                },
            )
            persist_experimental_record(root, historical)
            current_result = {
                "status": "success",
                "tool_evidence": {
                    "schema_version": "hydrocarbon-type.v1",
                    "evidence_type": "hydrocarbon_group_composition",
                    "source_file": "current_sample.xlsx",
                    "structured_result": [{"item": "chain alkane", "mass_percent": 30.0}],
                },
            }

            evidence = build_tool_result_channel_database_evidence(
                root,
                tool_result=current_result,
                capability_id="tool.hydrocarbon_type_calculator",
                query="compare this current sample with the database",
            )

            self.assertNotIn("experimental_record", current_result)
            self.assertEqual(evidence["status"], "success")
            self.assertEqual(evidence["neighbors"][0]["sample_id"], "curated_history")
            self.assertFalse((root / "experimental_records" / "hydrocarbon_type" / "current_sample").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
