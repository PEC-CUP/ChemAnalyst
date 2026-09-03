from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from app.capabilities.provider_registry import (
    evaluate_provider_package,
    list_provider_packages,
    register_reviewed_provider,
    revoke_provider,
    save_provider_package,
    update_provider_review,
)
from app.tools.tool_onboarding import ToolOnboardingSpec, build_tool_onboarding_package
from app.tools.registry import CapabilityRegistry


class CapabilityProviderRegistryTests(unittest.TestCase):
    def test_save_list_and_evaluate_provider_package(self) -> None:
        root = Path("tests/_tmp_capability_providers_save")
        try:
            shutil.rmtree(root, ignore_errors=True)
            package = build_tool_onboarding_package(
                ToolOnboardingSpec(
                    tool_name="SARA Analyzer",
                    description="Analyze SARA fraction data.",
                    invocation_type="python_callable",
                    entrypoint="app.tools.sara_tool.run",
                    input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
                    evidence_outputs=["tool_evidence"],
                    upstream_capabilities=["knowledge.retrieve"],
                    downstream_capabilities=["database.experimental_history"],
                    input_bindings={"file_path": "upstream.artifact.path"},
                    output_bindings={"tool_evidence": "evidence_bundle.tool_evidence"},
                )
            ).to_dict()

            saved = save_provider_package(package, root=root)
            self.assertEqual(saved["provider_id"], "sara_analyzer")
            self.assertTrue((root / "sara_analyzer" / "manifest.json").exists())

            providers = list_provider_packages(root=root)
            self.assertEqual(providers[0]["provider_id"], "sara_analyzer")
            self.assertEqual(providers[0]["workflow_relations"]["upstream_capabilities"], ["knowledge.retrieve"])

            evaluation = evaluate_provider_package("sara_analyzer", root=root)
            self.assertTrue(evaluation["passed"])
            self.assertTrue(evaluation["checks"]["workflow_relations_present"])
            task_specific = evaluation["task_specific_evaluation"]
            self.assertEqual(task_specific["summary"]["case_count"], 1)
            self.assertEqual(task_specific["summary"]["targets"]["tool"]["passed_count"], 1)
            self.assertEqual(task_specific["results"][0]["case_id"], "tool_onboarding_sara_analyzer_1")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_unreviewed_provider_does_not_execute(self) -> None:
        root = Path("tests/_tmp_capability_providers_unreviewed")
        try:
            shutil.rmtree(root, ignore_errors=True)
            package = build_tool_onboarding_package(
                ToolOnboardingSpec(
                    tool_name="Demo Tool",
                    description="Demo.",
                    invocation_type="python_callable",
                    entrypoint="app.tools.demo_tool.run",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    evidence_outputs=["tool_evidence"],
                )
            ).to_dict()
            save_provider_package(package, root=root)
            manifest_path = root / "demo_tool" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["review"]["reviewed"])

            evaluation = evaluate_provider_package("demo_tool", execute=True, sample_payload={}, root=root)
            self.assertFalse(evaluation["passed"])
            self.assertEqual(evaluation["execution_result"]["status"], "skipped")
            self.assertFalse(evaluation["checks"]["reviewed_adapter_available"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_generic_python_script_provider_can_execute_after_review(self) -> None:
        root = Path("tests/_tmp_capability_providers_generic_script")
        try:
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)
            script_path = root / "external_tool.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import pandas as pd",
                        "def calculate_file(input_file, output_file):",
                        "    pd.DataFrame([{'sample':'demo','item':'total','mass_percent':100.0}]).to_csv(output_file, index=False)",
                    ]
                ),
                encoding="utf-8",
            )
            input_file = root / "input.csv"
            input_file.write_text("x\n1\n", encoding="utf-8")

            package = build_tool_onboarding_package(
                ToolOnboardingSpec(
                    tool_name="Generic Hydrocarbon Tool",
                    description="Run an existing external script through the generic Python adapter.",
                    invocation_type="python_script",
                    entrypoint=str(script_path),
                    adapter_template="generic_python_function",
                    adapter_config={
                        "script_path": str(script_path),
                        "function_name": "calculate_file",
                        "call_style": "input_output_paths",
                        "input_field": "input_file",
                        "output_artifact_key": "output_csv",
                        "output_suffix": ".csv",
                        "output_parser": {"type": "csv_records"},
                        "evidence": {
                            "schema_version": "hydrocarbon-type.v1",
                            "evidence_type": "hydrocarbon_group_composition",
                            "method": "demo",
                        },
                    },
                    input_schema={"type": "object", "properties": {"input_file": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"tool_evidence": {"type": "object"}}},
                    evidence_outputs=["tool_evidence", "artifacts"],
                    database_integration_contract={"enabled": True, "channel_id": "hydrocarbon_type"},
                )
            ).to_dict()
            save_provider_package(package, root=root)
            review = update_provider_review("generic_hydrocarbon_tool", reviewed=True, enabled=True, notes="test", root=root)
            self.assertEqual(review["status"], "success")

            evaluation = evaluate_provider_package(
                "generic_hydrocarbon_tool",
                execute=True,
                sample_payload={"input_file": str(input_file), "output_dir": str(root / "outputs")},
                root=root,
            )

            self.assertTrue(evaluation["checks"]["reviewed_adapter_available"])
            self.assertTrue(evaluation["checks"]["execution_success"])
            self.assertTrue(evaluation["checks"]["structured_evidence_present"])
            self.assertEqual(evaluation["execution_result"]["tool_evidence"]["schema_version"], "hydrocarbon-type.v1")
            self.assertTrue(evaluation["task_specific_evaluation"]["results"][0]["checks"]["execution_evidence"])

            registry = CapabilityRegistry()
            registered = register_reviewed_provider(registry, "generic_hydrocarbon_tool", root=root)
            runtime_result = registry.call(
                registered["capability_id"],
                {"input_file": str(input_file), "output_dir": str(root / "runtime_outputs")},
            )
            self.assertEqual(runtime_result["status"], "success")
            self.assertNotIn("experimental_record", runtime_result)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_revoke_provider_disables_and_unregisters_runtime_capability(self) -> None:
        root = Path("tests/_tmp_capability_provider_revoke")
        try:
            shutil.rmtree(root, ignore_errors=True)
            package = build_tool_onboarding_package(
                ToolOnboardingSpec(
                    tool_name="Revoke Demo",
                    description="Demo reviewed callable.",
                    invocation_type="python_callable",
                    entrypoint="app.tools.tool_onboarding.build_tool_onboarding_package",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    evidence_outputs=["tool_evidence"],
                )
            ).to_dict()
            save_provider_package(package, root=root)
            update_provider_review("revoke_demo", reviewed=True, enabled=True, notes="test", root=root)
            registry = CapabilityRegistry()
            registered = register_reviewed_provider(registry, "revoke_demo", root=root)
            self.assertEqual(registered["status"], "success")
            self.assertIsNotNone(registry.get("tool.revoke_demo"))

            revoked = revoke_provider(registry, "revoke_demo", reason="test revoke", root=root)

            self.assertEqual(revoked["status"], "success")
            self.assertTrue(revoked["runtime_unregistered"])
            self.assertIsNone(registry.get("tool.revoke_demo"))
            manifest = json.loads((root / "revoke_demo" / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["review"]["enabled"])
            self.assertTrue(manifest["revoked"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
