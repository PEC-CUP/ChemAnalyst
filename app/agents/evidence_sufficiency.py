from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.agents.schemas import AnalysisRequest, EvidenceBundle


SufficiencyStatus = Literal["sufficient", "weak", "insufficient"]
RecommendedAction = Literal[
    "synthesize",
    "lower_confidence",
    "retrieve_more",
    "run_tool",
    "query_database",
    "ask_user_for_input",
]


@dataclass(slots=True)
class EvidenceSufficiencyReport:
    schema_version: str = "evidence-sufficiency.v1"
    task_template: str = ""
    status: SufficiencyStatus = "insufficient"
    confidence_ceiling: str = "low"
    recommended_action: RecommendedAction = "ask_user_for_input"
    required_evidence: list[str] = field(default_factory=list)
    present_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    weak_evidence: list[str] = field(default_factory=list)
    conflict_evidence: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceSufficiencyEvaluator:
    """Task-level evidence gate for Planner Orchestrator.

    This evaluator does not re-score internal RAG/tool/database quality. It reads
    capability summaries and decides whether the collected EvidenceBundle is
    adequate for the current task.
    """

    PROPERTY_QUERY_MARKERS = (
        "density",
        "boiling",
        "boiling range",
        "distillation",
        "property",
        "properties",
        "abundance",
        "abundance",
        "abundance",
        "abundance",
        "abundance",
        "abundance",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
    )

    def evaluate(self, request: AnalysisRequest, evidence: EvidenceBundle) -> EvidenceSufficiencyReport:
        template = str(request.template or "")
        report = EvidenceSufficiencyReport(task_template=template)

        if template == "concept_qa":
            self._evaluate_knowledge_only(report, evidence, min_docs=1)
        elif template == "knowledge_qa":
            self._evaluate_knowledge_only(report, evidence, min_docs=2)
        elif template == "data_analysis":
            self._evaluate_data_analysis(report, request, evidence)
        elif template == "hybrid_analysis":
            self._evaluate_hybrid_analysis(report, request, evidence)
        elif template == "processing_evaluation":
            self._evaluate_processing(report, evidence)
        else:
            self._evaluate_knowledge_only(report, evidence, min_docs=1)

        self._finalize_status(report)
        return report

    def _evaluate_knowledge_only(self, report: EvidenceSufficiencyReport, evidence: EvidenceBundle, *, min_docs: int) -> None:
        report.required_evidence.append("knowledge_evidence")
        doc_count = self._knowledge_doc_count(evidence)
        if doc_count >= min_docs:
            report.present_evidence.append("knowledge_evidence")
            report.rationale.append(f"knowledge_evidence_doc_count={doc_count}")
        elif doc_count > 0:
            report.present_evidence.append("knowledge_evidence")
            report.weak_evidence.append("low_knowledge_doc_count")
            report.rationale.append(f"knowledge_evidence_doc_count={doc_count}<required_{min_docs}")
        else:
            report.missing_evidence.append("knowledge_evidence")
            report.rationale.append("knowledge_evidence_missing")

    def _evaluate_data_analysis(self, report: EvidenceSufficiencyReport, request: AnalysisRequest, evidence: EvidenceBundle) -> None:
        report.required_evidence.append("tool_evidence")
        self._check_tool_evidence(report, evidence)
        if self._asks_property_inference(request.query):
            report.required_evidence.append("database_evidence")
            self._check_database_evidence(report, evidence)

    def _evaluate_hybrid_analysis(self, report: EvidenceSufficiencyReport, request: AnalysisRequest, evidence: EvidenceBundle) -> None:
        report.required_evidence.extend(["tool_evidence", "knowledge_evidence"])
        self._check_tool_evidence(report, evidence)
        self._check_knowledge_evidence(report, evidence, min_docs=1)
        if self._asks_property_inference(request.query):
            report.required_evidence.append("database_evidence")
            self._check_database_evidence(report, evidence)

    def _evaluate_processing(self, report: EvidenceSufficiencyReport, evidence: EvidenceBundle) -> None:
        report.required_evidence.append("knowledge_evidence")
        self._check_knowledge_evidence(report, evidence, min_docs=1)

    def _check_knowledge_evidence(self, report: EvidenceSufficiencyReport, evidence: EvidenceBundle, *, min_docs: int) -> None:
        doc_count = self._knowledge_doc_count(evidence)
        evidence_count = self._knowledge_evidence_count(evidence)
        support_count = max(doc_count, evidence_count)
        if support_count >= min_docs:
            report.present_evidence.append("knowledge_evidence")
            report.rationale.append(f"knowledge_support_count={support_count}")
        elif support_count > 0:
            report.present_evidence.append("knowledge_evidence")
            report.weak_evidence.append("low_knowledge_support")
            report.rationale.append(f"knowledge_support_count={support_count}<required_{min_docs}")
        else:
            report.missing_evidence.append("knowledge_evidence")
            report.rationale.append("knowledge_evidence_missing")

    def _check_tool_evidence(self, report: EvidenceSufficiencyReport, evidence: EvidenceBundle) -> None:
        tool = evidence.tool_evidence or {}
        status = str(tool.get("status") or "").lower()
        if status == "success":
            report.present_evidence.append("tool_evidence")
            report.rationale.append("toolchain_status=success")
        elif status:
            report.present_evidence.append("tool_evidence")
            report.weak_evidence.append("toolchain_not_success")
            report.rationale.append(f"toolchain_status={status}")
        else:
            report.missing_evidence.append("tool_evidence")
            report.rationale.append("tool_evidence_missing")

        features = tool.get("features")
        generic_structured = tool.get("structured_result") or tool.get("structured_tool_evidence")
        has_generic_structured = bool(generic_structured)
        if status == "success" and not isinstance(features, dict) and not has_generic_structured:
            report.weak_evidence.append("missing_structured_features")
            report.rationale.append("tool_features_missing")
        elif isinstance(features, dict) and features:
            report.present_evidence.append("structured_features")
        elif has_generic_structured:
            report.present_evidence.append("structured_tool_output")

    def _check_database_evidence(self, report: EvidenceSufficiencyReport, evidence: EvidenceBundle) -> None:
        database = evidence.database_evidence or {}
        status = str(database.get("status") or "").lower()
        if status == "success":
            report.present_evidence.append("database_evidence")
            report.rationale.append("database_evidence_status=success")
        elif status in {"skipped", "error"}:
            report.weak_evidence.append("database_evidence_unavailable")
            report.rationale.append(f"database_evidence_status={status}")
        else:
            report.missing_evidence.append("database_evidence")
            report.rationale.append("database_evidence_missing")

        neighbor_count = self._safe_int(database.get("neighbor_count"))
        if status == "success" and neighbor_count is not None and neighbor_count < 3:
            report.weak_evidence.append("low_database_neighbor_count")
            report.rationale.append(f"database_neighbor_count={neighbor_count}<3")

        property_summary = database.get("property_summary")
        if status == "success" and not isinstance(property_summary, dict):
            report.weak_evidence.append("missing_database_property_summary")
            report.rationale.append("database_property_summary_missing")

    def _finalize_status(self, report: EvidenceSufficiencyReport) -> None:
        if report.missing_evidence:
            report.status = "insufficient"
            report.confidence_ceiling = "low"
            report.recommended_action = self._recommended_action_for_missing(report.missing_evidence[0])
            return
        if report.weak_evidence or report.conflict_evidence:
            report.status = "weak"
            report.confidence_ceiling = "moderate"
            report.recommended_action = self._recommended_action_for_weak(report.weak_evidence)
            return
        report.status = "sufficient"
        report.confidence_ceiling = "high"
        report.recommended_action = "synthesize"

    @staticmethod
    def _recommended_action_for_missing(item: str) -> RecommendedAction:
        if item == "knowledge_evidence":
            return "retrieve_more"
        if item == "tool_evidence":
            return "run_tool"
        if item == "database_evidence":
            return "query_database"
        return "ask_user_for_input"

    @staticmethod
    def _recommended_action_for_weak(items: list[str]) -> RecommendedAction:
        weak_items = set(items)
        if weak_items & {"low_knowledge_doc_count", "low_knowledge_support"}:
            return "retrieve_more"
        if weak_items & {"toolchain_not_success", "missing_structured_features"}:
            return "run_tool"
        if weak_items & {
            "database_evidence_unavailable",
            "low_database_neighbor_count",
            "missing_database_property_summary",
        }:
            return "query_database"
        return "lower_confidence"

    @staticmethod
    def _knowledge_doc_count(evidence: EvidenceBundle) -> int:
        knowledge = evidence.knowledge_evidence or {}
        value = knowledge.get("doc_count")
        if isinstance(value, int):
            return value
        docs = knowledge.get("documents")
        return len(docs) if isinstance(docs, list) else 0

    @staticmethod
    def _knowledge_evidence_count(evidence: EvidenceBundle) -> int:
        knowledge = evidence.knowledge_evidence or {}
        value = knowledge.get("evidence_count")
        if isinstance(value, int):
            return value
        evidences = knowledge.get("evidences")
        return len(evidences) if isinstance(evidences, list) else 0

    @classmethod
    def _asks_property_inference(cls, query: str) -> bool:
        lowered = str(query or "").lower()
        return any(marker in lowered for marker in cls.PROPERTY_QUERY_MARKERS)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
