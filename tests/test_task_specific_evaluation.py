from __future__ import annotations

import unittest

from app.agents.task_specific_evaluation import (
    TaskSpecificEvaluationCase,
    default_end_to_end_agent_cases,
    default_workflow_evaluation_cases,
    evaluate_tool_provider_result,
    evaluate_workflow_result,
    run_end_to_end_agent_evaluation,
    summarize_task_specific_results,
    task_specific_case_from_dict,
)


class TaskSpecificEvaluationTests(unittest.TestCase):
    def test_default_workflow_cases_define_unified_schema(self) -> None:
        cases = default_workflow_evaluation_cases()
        self.assertEqual(len(cases), 4)
        self.assertEqual({case.evaluation_target for case in cases}, {"workflow"})
        self.assertTrue(any("database.experimental_history" in case.expected_capabilities for case in cases))

    def test_evaluate_workflow_result_passes_expected_evidence(self) -> None:
        case = default_workflow_evaluation_cases()[-1]
        result = {
            "task_type": "planner",
            "raw_result": {
                "planner": {"template": "hybrid_analysis"},
                "evidence_bundle": {
                    "capabilities_used": [
                        {"capability_id": "knowledge.retrieve"},
                        {"capability_id": "database.experimental_history"},
                        {"capability_id": "database.experimental_history"},
                    ],
                    "knowledge_evidence": {"doc_count": 2},
                    "tool_evidence": {"status": "success"},
                    "database_evidence": {"status": "success"},
                },
                "evidence_sufficiency_report": {"status": "sufficient"},
                "analysis_conclusion": {
                    "direct_findings": {},
                    "database_inference": {},
                    "knowledge_interpretation": {},
                    "uncertainty": {},
                    "confidence": "moderate",
                },
            },
        }
        evaluation = evaluate_workflow_result(case, result)
        self.assertTrue(evaluation.passed)
        self.assertEqual(evaluation.failure_reasons, [])

    def test_evaluate_workflow_result_fails_missing_database_evidence(self) -> None:
        case = default_workflow_evaluation_cases()[-1]
        result = {
            "task_type": "planner",
            "raw_result": {
                "planner": {"template": "hybrid_analysis"},
                "evidence_bundle": {
                    "capabilities_used": [
                        {"capability_id": "knowledge.retrieve"},
                        {"capability_id": "database.experimental_history"},
                    ],
                    "knowledge_evidence": {"doc_count": 2},
                    "tool_evidence": {"status": "success"},
                },
                "evidence_sufficiency_report": {"status": "weak"},
                "analysis_conclusion": {"direct_findings": {}, "uncertainty": {}},
            },
        }
        evaluation = evaluate_workflow_result(case, result)
        self.assertFalse(evaluation.passed)
        self.assertIn("evidence", evaluation.failure_reasons)
        self.assertIn("sufficiency", evaluation.failure_reasons)

    def test_onboarding_tool_case_uses_declared_and_executed_evidence(self) -> None:
        case = task_specific_case_from_dict(
            {
                "case_id": "tool_hydrocarbon",
                "evaluation_target": "tool",
                "query": "Calculate hydrocarbon type composition from the uploaded workbook.",
                "expected_capabilities": ["tool.hydrocarbon_type_calculator"],
                "expected_evidence": ["tool_evidence", "artifacts"],
            }
        )
        evaluation = evaluate_tool_provider_result(
            case,
            {
                "passed": True,
                "capability": {"capability_id": "tool.hydrocarbon_type_calculator"},
                "manifest": {"evidence_contract": {"required_outputs": ["tool_evidence", "artifacts"]}},
                "execution_result": {
                    "status": "success",
                    "tool_evidence": {"evidence_type": "hydrocarbon_group_composition"},
                    "artifacts": {"output_excel": "out.xlsx"},
                },
            },
        )
        self.assertTrue(evaluation.passed)
        self.assertTrue(evaluation.checks["execution_evidence"])

    def test_task_specific_summary_groups_evaluation_targets(self) -> None:
        passed = evaluate_tool_provider_result(
            TaskSpecificEvaluationCase(
                case_id="tool_declared",
                evaluation_target="tool",
                query="Run tool.",
                expected_capabilities=["tool.demo"],
                expected_evidence=["tool_evidence"],
            ),
            {
                "passed": True,
                "capability": {"capability_id": "tool.demo"},
                "manifest": {"evidence_contract": {"required_outputs": ["tool_evidence"]}},
                "execution_result": None,
            },
        )
        summary = summarize_task_specific_results([passed])
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["targets"]["tool"]["passed_count"], 1)

    def test_end_to_end_agent_runner_evaluates_cross_capability_case(self) -> None:
        case = default_end_to_end_agent_cases()[0]

        def execute_case(_case: TaskSpecificEvaluationCase) -> dict:
            return {
                "task_type": "planner",
                "raw_result": {
                    "planner": {"template": "hybrid_analysis"},
                    "evidence_bundle": {
                        "capabilities_used": [
                            {"capability_id": "knowledge.retrieve"},
                            {"capability_id": "database.experimental_history"},
                            {"capability_id": "database.experimental_history"},
                        ],
                        "knowledge_evidence": {"doc_count": 3},
                        "tool_evidence": {"status": "success"},
                        "database_evidence": {"status": "success"},
                    },
                    "evidence_sufficiency_report": {"status": "sufficient"},
                    "analysis_conclusion": {
                        "direct_findings": {},
                        "database_inference": {},
                        "knowledge_interpretation": {},
                        "uncertainty": {},
                        "confidence": "moderate",
                    },
                },
            }

        evaluation = run_end_to_end_agent_evaluation(execute_case, [case])
        self.assertEqual(evaluation["schema_version"], "chemanalyst-end-to-end-agent-evaluation.v1")
        self.assertEqual(evaluation["summary"]["passed_count"], 1)
        self.assertEqual(evaluation["results"][0]["case_id"], case.case_id)


if __name__ == "__main__":
    unittest.main()

