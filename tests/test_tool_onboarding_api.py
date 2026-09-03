from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app


class ToolOnboardingApiTests(unittest.TestCase):
    def test_onboarding_endpoint_returns_capability_package(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/capabilities/onboard",
            json={
                "tool_name": "SARA Analyzer",
                "description": "Analyze SARA fraction data and produce structured petroleum composition evidence.",
                "invocation_type": "python_callable",
                "entrypoint": "app.tools.sara_tool.run",
                "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
                "evidence_outputs": ["tool_evidence"],
                "preconditions": ["uploaded_artifact_available"],
                "artifacts_produced": ["sara_evidence_json"],
                "example_queries": ["Use the uploaded SARA table to evaluate petroleum composition."],
                "quality_checks": ["status_success", "structured_tool_evidence_present"],
                "database_integration_contract": {"enabled": True, "channel_id": "sara"},
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        package = payload["package"]
        self.assertEqual(package["capability"]["capability_id"], "tool.sara_analyzer")
        self.assertEqual(package["mcp_manifest"]["capability_id"], "tool.sara_analyzer")
        self.assertEqual(package["mcp_manifest"]["database_integration_contract"]["channel_id"], "sara")
        self.assertIn("disabled", package["mcp_manifest"]["database_integration_contract"]["runtime_ingestion"])
        self.assertEqual(package["evaluation_cases"][0]["expected_capabilities"], ["tool.sara_analyzer"])

    def test_tool_onboarding_ui_is_served(self) -> None:
        client = TestClient(app)
        response = client.get("/tool-onboarding")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Analytical Tool Onboarding", response.text)
        self.assertIn("Onboarding manual", response.text)
        self.assertIn("Analytical Tool Onboarding manual", response.text)
        self.assertIn("generic_python_function", response.text)
        self.assertIn("Deploy to Planner", response.text)
        self.assertIn("Onboarding actions", response.text)
        self.assertIn("Hydrocarbon type composition calculation", response.text)
        self.assertIn("Revoke provider", response.text)

    def test_capability_options_endpoint_returns_runtime_tools(self) -> None:
        client = TestClient(app)
        response = client.get("/capabilities/options")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        capability_ids = [item["capability_id"] for item in payload["result"]["capabilities"]]
        self.assertIn("database.experimental_history", capability_ids)

    def test_revoke_endpoint_requires_confirmation(self) -> None:
        client = TestClient(app)
        response = client.post("/capabilities/providers/not_real/revoke", json={"confirm": False})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["message"], "revoke_confirmation_required")


if __name__ == "__main__":
    unittest.main()



