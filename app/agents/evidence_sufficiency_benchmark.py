from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.agents.evidence_sufficiency import EvidenceSufficiencyEvaluator
from app.agents.schemas import AnalysisRequest, EvidenceBundle


@dataclass(slots=True)
class EvidenceSufficiencyCase:
    case_id: str
    query: str
    template: str
    evidence: dict[str, Any]
    expected_status: str
    expected_action: str
    expected_missing: list[str] = field(default_factory=list)
    expected_weak: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceSufficiencyCaseResult:
    case_id: str
    passed: bool
    expected_status: str
    actual_status: str
    expected_action: str
    actual_action: str
    missing_ok: bool
    weak_ok: bool
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_evidence_sufficiency_cases() -> list[EvidenceSufficiencyCase]:
    """Small task-specific benchmark for planner evidence sufficiency behavior."""

    return [
        EvidenceSufficiencyCase(
            case_id="knowledge_missing",
            query="Explain the geochemical meaning of Pr/Ph in crude oil GC analysis.",
            template="knowledge_qa",
            evidence={},
            expected_status="insufficient",
            expected_action="retrieve_more",
            expected_missing=["knowledge_evidence"],
        ),
        EvidenceSufficiencyCase(
            case_id="tool_missing_for_data_analysis",
            query="Analyze the uploaded GC peak table and extract structured features.",
            template="data_analysis",
            evidence={},
            expected_status="insufficient",
            expected_action="run_tool",
            expected_missing=["tool_evidence"],
        ),
        EvidenceSufficiencyCase(
            case_id="property_requires_database",
            query="Infer density and distillation-range properties from the uploaded GC data.",
            template="data_analysis",
            evidence={
                "tool_evidence": {
                    "status": "success",
                    "features": {"schema_version": "gc-features.v1"},
                    "sample_id": "external-sample",
                },
                "database_evidence": {"status": "skipped", "sample_id": "external-sample"},
            },
            expected_status="weak",
            expected_action="query_database",
            expected_weak=["database_evidence_unavailable"],
        ),
        EvidenceSufficiencyCase(
            case_id="hybrid_ready",
            query="Analyze the hydrocarbon distribution of the sample using GC features and knowledge-base evidence.",
            template="hybrid_analysis",
            evidence={
                "tool_evidence": {
                    "status": "success",
                    "features": {"schema_version": "gc-features.v1"},
                    "sample_id": "1#",
                },
                "knowledge_evidence": {"documents": [{"source": "paper.md", "text": "evidence"}], "doc_count": 1},
            },
            expected_status="sufficient",
            expected_action="synthesize",
        ),
        EvidenceSufficiencyCase(
            case_id="tool_success_missing_features",
            query="Analyze the uploaded GC peak table.",
            template="data_analysis",
            evidence={"tool_evidence": {"status": "success", "sample_id": "1#"}},
            expected_status="weak",
            expected_action="run_tool",
            expected_weak=["missing_structured_features"],
        ),
    ]


def run_evidence_sufficiency_benchmark(
    cases: list[EvidenceSufficiencyCase] | None = None,
) -> dict[str, Any]:
    evaluator = EvidenceSufficiencyEvaluator()
    results: list[EvidenceSufficiencyCaseResult] = []
    for case in cases or default_evidence_sufficiency_cases():
        request = AnalysisRequest(query=case.query, session_id=f"bench-{case.case_id}", template=case.template)
        evidence = EvidenceBundle(**case.evidence)
        report = evaluator.evaluate(request, evidence)
        missing_ok = all(item in report.missing_evidence for item in case.expected_missing)
        weak_ok = all(item in report.weak_evidence for item in case.expected_weak)
        passed = (
            report.status == case.expected_status
            and report.recommended_action == case.expected_action
            and missing_ok
            and weak_ok
        )
        results.append(
            EvidenceSufficiencyCaseResult(
                case_id=case.case_id,
                passed=passed,
                expected_status=case.expected_status,
                actual_status=report.status,
                expected_action=case.expected_action,
                actual_action=report.recommended_action,
                missing_ok=missing_ok,
                weak_ok=weak_ok,
                report=report.to_dict(),
            )
        )
    return {
        "schema_version": "planner-evidence-sufficiency-benchmark.v1",
        "case_count": len(results),
        "passed_count": sum(1 for item in results if item.passed),
        "pass_rate": round(sum(1 for item in results if item.passed) / len(results), 4) if results else 0.0,
        "results": [item.to_dict() for item in results],
    }
