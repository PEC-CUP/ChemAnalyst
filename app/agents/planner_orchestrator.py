from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.agents.runtime_agent import ChemAnalyst
from app.agents.schemas import AnalysisRequest, EvidenceBundle


PlannerTemplate = Literal["concept_qa", "knowledge_qa", "data_analysis", "processing_evaluation", "hybrid_analysis"]


@dataclass(slots=True)
class PlanStep:
    step_id: str
    action: str
    status: str = "pending"
    detail: str = ""
    output: dict[str, Any] | None = None


@dataclass(slots=True)
class ExecutionPlan:
    template: PlannerTemplate
    goal: str
    status: str = "pending"
    steps: list[PlanStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "goal": self.goal,
            "status": self.status,
            "steps": [asdict(step) for step in self.steps],
            "metadata": self.metadata,
        }


class PlannerOrchestrator:
    """Public-release planner wrapper.

    The public package exposes the manuscript-aligned workflows only: knowledge-base
    RAG, Historical Experimental DB property evidence, and reviewed dynamic tools
    such as the hydrocarbon-type calculator. Earlier internal GC matching and
    QuantPlatform toolchains are intentionally not part of this release.
    """

    def __init__(self, agent: ChemAnalyst) -> None:
        self.agent = agent

    def _select_template(self, *, query: str, task_type_override: str | None, template_override: str | None) -> PlannerTemplate:
        override = (template_override or "").strip().lower()
        if override in {"concept_qa", "knowledge_qa", "data_analysis", "processing_evaluation", "hybrid_analysis"}:
            return override  # type: ignore[return-value]
        requested = (task_type_override or "").strip().lower()
        text = query.lower()
        if requested == "rag" or any(token in text for token in ("rag", "knowledge", "literature", "paper", "method")):
            return "knowledge_qa"
        if any(token in text for token in ("database", "experimental", "property", "density", "bp_", "distillation", "gc-fid", "ir")):
            return "data_analysis"
        if any(token in text for token in ("tool", "calculate", "composition", "workbook", "uploaded")):
            return "data_analysis"
        return "concept_qa"

    def _build_plan(self, *, query: str, session_id: str, template: PlannerTemplate) -> ExecutionPlan:
        session = self.agent.sessions.get_or_create(session_id)
        steps = [
            PlanStep("s1", "classify_intent", "completed", output={"template": template}),
            PlanStep("s2", "check_context", "completed", output={"has_uploaded_files": bool(session.artifacts)}),
            PlanStep("s3", "execute_public_workflow", "completed"),
            PlanStep("s4", "compose_answer", "completed"),
        ]
        return ExecutionPlan(template=template, goal=query, status="completed", steps=steps)

    def _build_analysis_request(
        self,
        *,
        query: str,
        session_id: str,
        template: PlannerTemplate,
        rag_mode_override: str | None = None,
        task_type_override: str | None = None,
    ) -> AnalysisRequest:
        session = self.agent.sessions.get_or_create(session_id)
        return AnalysisRequest(
            query=query,
            session_id=session_id,
            template=template,
            rag_mode=rag_mode_override,
            task_type_override=task_type_override,
            uploaded_artifacts=[str(item.path) for item in session.artifacts],
            constraints={
                "has_uploaded_artifacts": bool(session.artifacts),
                "last_tool_name": session.last_tool_name,
                "pending_tool_name": session.pending_tool_name,
            },
        )

    def _attach_public_raw_result(
        self,
        *,
        result: dict[str, Any],
        query: str,
        session_id: str,
        template: PlannerTemplate,
        rag_mode_override: str | None,
        task_type_override: str | None,
    ) -> dict[str, Any]:
        raw = result.get("raw_result") if isinstance(result.get("raw_result"), dict) else {}
        plan = self._build_plan(query=query, session_id=session_id, template=template)
        analysis_request = self._build_analysis_request(
            query=query,
            session_id=session_id,
            template=template,
            rag_mode_override=rag_mode_override,
            task_type_override=task_type_override,
        )
        raw.setdefault("planner", plan.to_dict())
        raw.setdefault("analysis_request", analysis_request.model_dump())
        if "evidence_bundle" not in raw:
            raw["evidence_bundle"] = EvidenceBundle().model_dump()
        result["raw_result"] = raw
        if result.get("task_type") != "planner":
            result["task_type"] = "planner"
        return result

    def answer_query(
        self,
        query: str,
        *,
        session_id: str = "default",
        task_type_override: str | None = None,
        template_override: str | None = None,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        template = self._select_template(
            query=query,
            task_type_override=task_type_override,
            template_override=template_override,
        )
        runtime_task = task_type_override
        if template == "knowledge_qa":
            runtime_task = "rag"
        result = self.agent.answer_query(
            query,
            session_id=session_id,
            task_type_override=runtime_task,
            rag_mode_override=rag_mode_override,
            retrieval_overrides=retrieval_overrides,
        )
        return self._attach_public_raw_result(
            result=result,
            query=query,
            session_id=session_id,
            template=template,
            rag_mode_override=rag_mode_override,
            task_type_override=task_type_override,
        )
