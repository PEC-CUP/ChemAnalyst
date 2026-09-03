from __future__ import annotations

import unittest

from app.tools.tool_onboarding import ToolOnboardingSpec, build_tool_onboarding_package


class ToolOnboardingTests(unittest.TestCase):
    def test_builds_capability_manifest_and_evaluation_cases(self) -> None:
        spec = ToolOnboardingSpec(
            tool_name="SARA Analyzer",
            description="Analyze SARA fraction data and produce structured petroleum composition evidence.",
            invocation_type="http_api",
            entrypoint="https://example.invalid/sara/analyze",
            input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"status": {"type": "string"}, "fractions": {"type": "object"}}},
            evidence_outputs=["tool_evidence"],
            preconditions=["uploaded_artifact_available"],
            artifacts_produced=["sara_report_json"],
            example_queries=["Use the uploaded SARA table to evaluate crude-oil composition."],
            quality_checks=["status_success", "fractions_sum_to_100", "warnings_present_if_invalid"],
            upstream_capabilities=["knowledge.retrieve"],
            downstream_capabilities=["database.experimental_history"],
            input_bindings={"file_path": "upstream.artifact.path"},
            output_bindings={"tool_evidence": "evidence_bundle.tool_evidence"},
        )

        package = build_tool_onboarding_package(spec)

        self.assertEqual(package.capability.capability_id, "tool.sara_analyzer")
        self.assertEqual(package.capability.capability_type, "tool")
        self.assertEqual(package.capability.invocation_mode, "external_adapter_required")
        self.assertEqual(package.mcp_manifest["capability_id"], "tool.sara_analyzer")
        self.assertTrue(package.mcp_manifest["invocation"]["review_required"])
        self.assertEqual(package.evaluation_cases[0].expected_capabilities, ["tool.sara_analyzer"])
        self.assertEqual(package.evaluation_cases[0].expected_evidence, ["tool_evidence"])
        self.assertEqual(package.mcp_manifest["workflow_relations"]["upstream_capabilities"], ["knowledge.retrieve"])
        self.assertEqual(package.mcp_manifest["workflow_relations"]["output_bindings"]["tool_evidence"], "evidence_bundle.tool_evidence")
        self.assertEqual(package.registration_plan["runtime_policy"]["do_not_execute_unreviewed_uploaded_code"], True)
        self.assertEqual(package.warnings, [])

    def test_builds_generic_python_script_adapter_config(self) -> None:
        package = build_tool_onboarding_package(
            ToolOnboardingSpec(
                tool_name="Hydrocarbon Type Calculator",
                description="Run an existing SH/T 0606 calculation script.",
                invocation_type="python_script",
                entrypoint="examples/tools/hydrocarbon_type/sht0606_official_full.py",
                adapter_template="generic_python_function",
                adapter_config={
                    "function_name": "calculate_file",
                    "call_style": "input_output_paths",
                    "output_parser": {"type": "excel_sheet_records", "sheet_name": "summary_compare"},
                },
                input_schema={"type": "object", "properties": {"input_file": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"tool_evidence": {"type": "object"}}},
                evidence_outputs=["tool_evidence", "artifacts"],
            )
        )

        invocation = package.mcp_manifest["invocation"]
        self.assertEqual(invocation["type"], "python_script")
        self.assertEqual(invocation["adapter_template"], "generic_python_function")
        self.assertEqual(invocation["adapter_config"]["function_name"], "calculate_file")
        self.assertTrue(invocation["review_required"])

    def test_warns_when_tool_description_is_incomplete(self) -> None:
        package = build_tool_onboarding_package(
            ToolOnboardingSpec(
                tool_name="",
                description="",
                invocation_type="python_script",
            )
        )
        self.assertIn("missing_tool_name", package.warnings)
        self.assertIn("missing_description", package.warnings)
        self.assertIn("missing_external_entrypoint", package.warnings)
        self.assertIn("missing_input_schema", package.warnings)
        self.assertIn("missing_output_schema", package.warnings)
        self.assertIn("missing_evidence_outputs", package.warnings)
        self.assertEqual(package.capability.capability_id, "tool.unnamed_tool")


if __name__ == "__main__":
    unittest.main()
