from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal


EvaluationStage = Literal["offline_development", "runtime"]
EvaluationTarget = Literal["rag", "planner", "tool", "database", "workflow"]


@dataclass(slots=True)
class TaskSpecificEvaluationCase:
    """Unified offline evaluation case for ChemAnalyst components.

    This schema is for benchmark construction, development, and paper
    validation. It is not executed on every runtime user request.
    """

    case_id: str
    evaluation_target: EvaluationTarget
    query: str
    expected_route: str | None = None
    expected_template: str | None = None
    expected_capabilities: list[str] = field(default_factory=list)
    expected_evidence: list[str] = field(default_factory=list)
    expected_sufficiency_status: str | None = None
    expected_action: str | None = None
    expected_conclusion_fields: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskSpecificEvaluationResult:
    case_id: str
    evaluation_target: EvaluationTarget
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_specific_case_from_dict(payload: dict[str, Any]) -> TaskSpecificEvaluationCase:
    """Read a generated or hand-authored task-specific evaluation case."""

    return TaskSpecificEvaluationCase(
        case_id=str(payload.get("case_id") or "unnamed_case"),
        evaluation_target=payload.get("evaluation_target") or "workflow",
        query=str(payload.get("query") or ""),
        expected_route=payload.get("expected_route"),
        expected_template=payload.get("expected_template"),
        expected_capabilities=_string_list(payload.get("expected_capabilities")),
        expected_evidence=_string_list(payload.get("expected_evidence")),
        expected_sufficiency_status=payload.get("expected_sufficiency_status"),
        expected_action=payload.get("expected_action"),
        expected_conclusion_fields=_string_list(payload.get("expected_conclusion_fields")),
        notes=str(payload.get("notes") or ""),
    )


def task_specific_cases_from_payload(payload: Any) -> list[TaskSpecificEvaluationCase]:
    if not isinstance(payload, list):
        return []
    return [task_specific_case_from_dict(item) for item in payload if isinstance(item, dict)]


def default_workflow_evaluation_cases() -> list[TaskSpecificEvaluationCase]:
    """Offline end-to-end workflow cases that connect Planner, capabilities, and EvidenceBundle."""

    return [
        TaskSpecificEvaluationCase(
            case_id="workflow_knowledge_qa",
            evaluation_target="workflow",
            query="Explain the role of Pr/Ph in crude oil geochemical interpretation.",
            expected_route="planner",
            expected_template="knowledge_qa",
            expected_capabilities=["knowledge.retrieve"],
            expected_evidence=["knowledge_evidence"],
            expected_sufficiency_status="sufficient",
            expected_conclusion_fields=["knowledge_interpretation", "uncertainty", "confidence"],
            notes="Pure domain QA should be evaluated as an QA-oriented RAG / knowledge-evidence workflow.",
        ),
        TaskSpecificEvaluationCase(
            case_id="workflow_experimental_db_evidence",
            evaluation_target="workflow",
            query="Use an uploaded petroleum fraction workbook to retrieve Historical Experimental DB evidence for property inference.",
            expected_route="planner",
            expected_template="data_analysis",
            expected_capabilities=["database.experimental_history"],
            expected_evidence=["database_evidence"],
            expected_sufficiency_status="sufficient",
            expected_conclusion_fields=["direct_findings", "uncertainty", "confidence"],
            notes="Experimental DB workflow should be evaluated by structured historical-neighbor evidence.",
        ),
        TaskSpecificEvaluationCase(
            case_id="workflow_gc_property_db",
            evaluation_target="workflow",
            query="Use uploaded GC data and the historical experimental database to infer density and BP_90.",
            expected_route="planner",
            expected_template="data_analysis",
            expected_capabilities=["database.experimental_history"],
            expected_evidence=["database_evidence", "database_evidence"],
            expected_sufficiency_status="sufficient",
            expected_conclusion_fields=["direct_findings", "database_inference", "uncertainty", "confidence"],
            notes="DB workflow should evaluate historical evidence coverage for a new sample, not leave-one-out leakage controls.",
        ),
        TaskSpecificEvaluationCase(
            case_id="workflow_hybrid_kb_tool_db",
            evaluation_target="workflow",
            query="Combine uploaded GC data, petroleum knowledge, and historical database evidence to interpret hydrocarbon distribution and property implications.",
            expected_route="planner",
            expected_template="hybrid_analysis",
            expected_capabilities=["knowledge.retrieve", "database.experimental_history"],
            expected_evidence=["knowledge_evidence", "database_evidence", "database_evidence"],
            expected_sufficiency_status="sufficient",
            expected_conclusion_fields=[
                "direct_findings",
                "database_inference",
                "knowledge_interpretation",
                "uncertainty",
                "confidence",
            ],
            notes="Full workflow case connects QA-oriented RAG, analytical tool evidence, DB evidence, and planner sufficiency.",
        ),
    ]


def default_end_to_end_agent_cases() -> list[TaskSpecificEvaluationCase]:
    """Concrete offline agent workflow cases for cross-capability evaluation."""

    return [
        TaskSpecificEvaluationCase(
            case_id="e2e_uploaded_sample_property_interpretation",
            evaluation_target="workflow",
            query=(
                "Analyze the uploaded petroleum GC peak table, use petroleum knowledge "
                "to interpret the hydrocarbon distribution, and use historical experimental "
                "database evidence to assess whether density and BP_90 inference is supported."
            ),
            expected_route="planner",
            expected_template="hybrid_analysis",
            expected_capabilities=["knowledge.retrieve", "database.experimental_history"],
            expected_evidence=["knowledge_evidence", "database_evidence", "database_evidence"],
            expected_sufficiency_status="sufficient",
            expected_conclusion_fields=[
                "direct_findings",
                "database_inference",
                "knowledge_interpretation",
                "uncertainty",
                "confidence",
            ],
            notes=(
                "End-to-end case: one uploaded petroleum data artifact should trigger Planner, "
                "analytical-tool evidence, KB evidence, DB evidence, sufficiency checking, "
                "and an evidence-grounded conclusion."
            ),
        )
    ]


def evaluate_workflow_result(case: TaskSpecificEvaluationCase, result: dict[str, Any]) -> TaskSpecificEvaluationResult:
    raw = result.get("raw_result") if isinstance(result, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    planner = raw.get("planner") if isinstance(raw.get("planner"), dict) else {}
    evidence = raw.get("evidence_bundle") if isinstance(raw.get("evidence_bundle"), dict) else {}
    sufficiency = raw.get("evidence_sufficiency_report") if isinstance(raw.get("evidence_sufficiency_report"), dict) else {}
    conclusion = raw.get("analysis_conclusion") if isinstance(raw.get("analysis_conclusion"), dict) else {}

    checks = {
        "route": result.get("task_type") == case.expected_route if case.expected_route else True,
        "template": planner.get("template") == case.expected_template if case.expected_template else True,
        "capabilities": _contains_capabilities(evidence, case.expected_capabilities),
        "evidence": _contains_evidence(evidence, case.expected_evidence),
        "sufficiency": sufficiency.get("status") == case.expected_sufficiency_status if case.expected_sufficiency_status else True,
        "conclusion_fields": all(field in conclusion for field in case.expected_conclusion_fields),
    }
    failure_reasons = [name for name, ok in checks.items() if not ok]
    return TaskSpecificEvaluationResult(
        case_id=case.case_id,
        evaluation_target=case.evaluation_target,
        passed=all(checks.values()),
        checks=checks,
        actual={
            "route": result.get("task_type"),
            "template": planner.get("template"),
            "capabilities": [item.get("capability_id") for item in evidence.get("capabilities_used", []) if isinstance(item, dict)],
            "evidence_keys": [key for key in case.expected_evidence if evidence.get(key)],
            "sufficiency_status": sufficiency.get("status"),
            "conclusion_fields": [key for key in conclusion.keys()],
        },
        expected=case.to_dict(),
        failure_reasons=failure_reasons,
    )


def run_end_to_end_agent_evaluation(
    execute_case: Callable[[TaskSpecificEvaluationCase], dict[str, Any]],
    cases: list[TaskSpecificEvaluationCase] | None = None,
) -> dict[str, Any]:
    """Execute workflow cases through an agent caller and evaluate returned evidence."""

    selected_cases = cases or default_end_to_end_agent_cases()
    results: list[TaskSpecificEvaluationResult] = []
    for case in selected_cases:
        try:
            raw_result = execute_case(case)
            results.append(evaluate_workflow_result(case, raw_result))
        except Exception as exc:
            results.append(
                TaskSpecificEvaluationResult(
                    case_id=case.case_id,
                    evaluation_target=case.evaluation_target,
                    passed=False,
                    checks={"execution": False},
                    actual={"execution_error": str(exc)},
                    expected=case.to_dict(),
                    failure_reasons=["execution"],
                )
            )
    return {
        "schema_version": "chemanalyst-end-to-end-agent-evaluation.v1",
        "stage": "offline_development",
        "cases": [case.to_dict() for case in selected_cases],
        "results": [result.to_dict() for result in results],
        "summary": summarize_task_specific_results(results),
    }


def evaluate_tool_provider_result(
    case: TaskSpecificEvaluationCase,
    provider_evaluation: dict[str, Any],
) -> TaskSpecificEvaluationResult:
    """Evaluate a Tool Onboarding provider against its generated tool case.

    A metadata-only provider check validates the declared capability and evidence
    contract. When provider execution is requested, the same case also checks the
    structured outputs returned by the reviewed adapter.
    """

    capability = provider_evaluation.get("capability") if isinstance(provider_evaluation, dict) else {}
    capability = capability if isinstance(capability, dict) else {}
    manifest = provider_evaluation.get("manifest") if isinstance(provider_evaluation, dict) else {}
    manifest = manifest if isinstance(manifest, dict) else {}
    execution_result = provider_evaluation.get("execution_result") if isinstance(provider_evaluation, dict) else None
    evidence_contract = manifest.get("evidence_contract") if isinstance(manifest.get("evidence_contract"), dict) else {}
    declared_outputs = _string_list(evidence_contract.get("required_outputs"))
    executed = isinstance(execution_result, dict) and execution_result.get("status") != "skipped"

    capability_id = capability.get("capability_id") or manifest.get("capability_id")
    checks = {
        "evaluation_target": case.evaluation_target == "tool",
        "provider_package": bool(provider_evaluation.get("passed")),
        "capabilities": all(expected == capability_id for expected in case.expected_capabilities),
        "declared_evidence": all(output in declared_outputs for output in case.expected_evidence),
        "execution_evidence": _contains_provider_outputs(execution_result, case.expected_evidence) if executed else True,
    }
    failure_reasons = [name for name, ok in checks.items() if not ok]
    return TaskSpecificEvaluationResult(
        case_id=case.case_id,
        evaluation_target=case.evaluation_target,
        passed=all(checks.values()),
        checks=checks,
        actual={
            "capability_id": capability_id,
            "declared_outputs": declared_outputs,
            "execution_checked": executed,
            "execution_status": execution_result.get("status") if isinstance(execution_result, dict) else None,
            "execution_output_keys": sorted(execution_result.keys()) if isinstance(execution_result, dict) else [],
        },
        expected=case.to_dict(),
        failure_reasons=failure_reasons,
    )


def summarize_task_specific_results(results: list[TaskSpecificEvaluationResult]) -> dict[str, Any]:
    target_counts = Counter(result.evaluation_target for result in results)
    passed_counts = Counter(result.evaluation_target for result in results if result.passed)
    return {
        "case_count": len(results),
        "passed_count": sum(result.passed for result in results),
        "failed_case_ids": [result.case_id for result in results if not result.passed],
        "targets": {
            target: {
                "case_count": count,
                "passed_count": passed_counts.get(target, 0),
            }
            for target, count in sorted(target_counts.items())
        },
    }


def _contains_capabilities(evidence: dict[str, Any], expected: list[str]) -> bool:
    capabilities = evidence.get("capabilities_used", [])
    capability_ids = {item.get("capability_id") for item in capabilities if isinstance(item, dict)}
    return all(capability_id in capability_ids for capability_id in expected)


def _contains_evidence(evidence: dict[str, Any], expected: list[str]) -> bool:
    return all(bool(evidence.get(key)) for key in expected)


def _contains_provider_outputs(payload: Any, expected: list[str]) -> bool:
    if not isinstance(payload, dict):
        return False
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return all(output in payload or output in result for output in expected)


def _string_list(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if item is not None]
