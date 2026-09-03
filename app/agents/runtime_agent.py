from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agents.state_store import SessionArtifact, SessionMemoryStore
from app.config import get_settings
from app.llm.deepseek_client import DeepSeekClient
from app.llm.unified_llm_client import UnifiedLLMClient
from app.rag.kb_backend import (
    qa_oriented_rag_answer,
    iterative_review_rag_answer,
    naive_llm_filter_rag_answer,
    naive_rag_answer,
    retrieve,
)
from app.router.task_router import detect_task_type, normalize_task_type
from app.tools.experimental_db import SampleDatabase
from app.tools.experimental_db.evidence_retriever import attach_petroleum_expert_reasoning
from app.tools.experimental_db.petroleum_fraction_spectral_evidence import (
    build_petroleum_fraction_spectral_workbook_evidence,
)
from app.tools.registry import ToolRegistry, ToolSpec
from app.utils.logger import setup_logger


logger = setup_logger(__name__)
session_store = SessionMemoryStore()


@dataclass(slots=True)
class PlannerDecision:
    action: str
    task_type: str
    tool_name: str | None = None
    used_tools: list[str] = field(default_factory=list)
    upload_spec: dict | None = None
    reason: str = ""


@dataclass(slots=True)
class IntentAssessment:
    route: str = "chat"
    candidate_tool: str | None = None
    confidence: float = 0.0
    reason: str = ""


class SimplePlanner:
    """Small public-release planner for chat, RAG, reasoning, and manuscript tools."""

    def __init__(self, registry: ToolRegistry, llm: DeepSeekClient) -> None:
        self.registry = registry
        self.llm = llm

    def is_pending_tool_followup(self, query: str) -> bool:
        text = str(query or "").lower()
        return any(token in text for token in ("upload", "file", "workbook", "xlsx", "run", "continue", "confirm"))

    def should_cancel_pending(self, query: str) -> bool:
        text = str(query or "").lower()
        return any(token in text for token in ("cancel", "stop", "ignore", "never mind"))

    def is_general_chat_query(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        return text in {"hi", "hello", "help", "what can you do"}

    def is_tool_confirmation(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        return any(token in text for token in ("confirm", "run", "execute", "yes"))

    def is_tool_rejection(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        return any(token in text for token in ("reject", "cancel", "no", "do not run"))

    def should_reference_last_tool(self, query: str, session) -> bool:
        if not session.last_tool_result:
            return False
        text = str(query or "").lower()
        return any(token in text for token in ("result", "evidence", "neighbor", "property", "density", "distillation", "saturates"))

    def decide(self, *, query: str, session, task_type_override: str | None = None) -> PlannerDecision:
        override = normalize_task_type(task_type_override)
        if override in {"rag", "reasoning", "chat"}:
            return PlannerDecision(override, override, reason="override")
        if override == "tool":
            return self._experimental_db_decision(query)
        task_type = detect_task_type(query)
        if task_type == "rag":
            return PlannerDecision("rag", "rag", reason="router")
        if task_type == "reasoning":
            return PlannerDecision("reasoning", "reasoning", reason="router")
        if self._looks_like_experimental_db_request(query):
            return self._experimental_db_decision(query)
        return PlannerDecision("chat", "chat", reason="default")

    def _looks_like_experimental_db_request(self, query: str) -> bool:
        text = str(query or "").lower()
        return any(
            token in text
            for token in (
                "experimental db",
                "historical database",
                "historical experimental",
                "property inference",
                "density",
                "distillation",
                "saturates",
                "gc-fid",
                "ir spectrum",
                "petroleum fraction",
            )
        )

    def _experimental_db_decision(self, query: str) -> PlannerDecision:
        spec = self.registry.get_spec("database.experimental_history")
        if spec and spec.requires_upload:
            return PlannerDecision(
                "await_upload",
                "tool",
                spec.name,
                [spec.name],
                {"accepted_file_types": list(spec.accepted_file_types), "instructions": self._upload_instruction(spec)},
                "experimental_db_request",
            )
        return PlannerDecision("run_tool", "tool", "database.experimental_history", ["database.experimental_history"], reason="experimental_db_request")

    def _upload_instruction(self, spec: ToolSpec) -> str:
        if spec.name == "database.experimental_history":
            return "Upload a petroleum fraction Experimental DB workbook, then run the property-inference request again."
        if spec.name.startswith("tool."):
            return f"Upload the required input workbook or data file for `{spec.name}`, then run the analytical-tool request again."
        return "Upload the required input file, then run the request again."


class ChemAnalyst:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = DeepSeekClient()
        self.sessions = session_store
        self.registry = ToolRegistry()
        self._register_tools()
        self.planner = SimplePlanner(self.registry, self.llm)

    def _register_tools(self) -> None:
        self.registry.register(
            "database.experimental_history",
            self._run_experimental_history_workbook_tool,
            spec=ToolSpec(
                name="database.experimental_history",
                description=(
                    "Historical Experimental DB evidence path for uploaded petroleum fraction query workbooks. "
                    "It extracts GC-FID/IR spectral features, retrieves top-k historical neighbors, "
                    "and returns property evidence for training-free inference."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "input_file": {"type": "string", "description": "Uploaded petroleum fraction experimental sample workbook."},
                        "query": {"type": "string", "description": "Property inference request."},
                        "top_k": {"type": "integer", "default": 4},
                    },
                    "required": ["input_file"],
                    "x_chemanalyst_input_requirements": {
                        "file_type": ".xlsx Excel workbook",
                        "required_workbook_sheets": ["sample_information", "gc_fid", "ir_spectrum"],
                        "notes": [
                            "The query workbook is treated as a transient sample and is not inserted into the historical database.",
                            "GC-FID and IR spectra are mapped to the saved Experimental DB feature space before neighbor retrieval.",
                        ],
                        "input_templates": ["data/experimental_db/templates/chemanalyst_experimental_sample_template.xlsx"],
                    },
                },
                requires_upload=True,
                accepted_file_types=(".xlsx", ".xls"),
                reusable_from_previous_file=True,
                intent_keywords=(
                    "experimental db",
                    "historical database",
                    "historical experimental",
                    "property inference",
                    "training-free",
                    "density",
                    "distillation",
                    "saturates",
                    "gc-fid",
                    "ir",
                    "petroleum fraction",
                ),
            ),
        )

    def _manual_root(self) -> Path:
        return Path(__file__).resolve().parents[2] / "knowledge" / "manuals"

    def _record_turn(self, session_id: str, user_query: str, answer: str) -> None:
        self.sessions.append_message(session_id, "user", user_query)
        self.sessions.append_message(session_id, "assistant", answer)

    def _build_context(self, session_id: str, query: str) -> str | None:
        session = self.sessions.get_or_create(session_id)
        parts = []
        recent = self.sessions.build_recent_context(session_id)
        if recent:
            parts.append(recent)
        task_state = self.sessions.build_task_state_context(session_id)
        if task_state:
            parts.append(task_state)
        if self.planner.should_reference_last_tool(query, session):
            last_tool = self.sessions.build_last_tool_context(session_id)
            if last_tool:
                parts.append(last_tool)
        return "\n\n".join(parts) or None

    def _build_response_query(self, query: str) -> str:
        return str(query or "")

    def _identity_answer(self) -> str:
        return (
            "ChemAnalyst supports literature RAG, Historical Experimental DB property inference, "
            "and reviewed analytical-tool onboarding for the public petroleum-fraction workflow."
        )

    def _cancel_pending_answer(self, tool_name: str | None) -> str:
        return f"Cancelled the pending `{tool_name}` upload request." if tool_name else "Cancelled the pending request."

    def _build_upload_spec_with_assets(self, tool_name: str, upload_spec: dict | None) -> dict:
        spec = dict(upload_spec or {})
        spec.setdefault("status", "awaiting_upload")
        spec.setdefault("tool_name", tool_name)
        tool_spec = self.registry.get_spec(tool_name)
        if tool_spec:
            spec.setdefault("accepted_file_types", list(tool_spec.accepted_file_types))
            spec.setdefault("instructions", self.planner._upload_instruction(tool_spec))
            template_file = self._input_template_file_from_spec(tool_spec)
            if template_file:
                spec.setdefault("template_input_file", template_file)
        return spec

    @staticmethod
    def _input_template_file_from_spec(tool_spec: ToolSpec) -> str | None:
        if not isinstance(tool_spec.parameters, dict):
            return None
        requirements = tool_spec.parameters.get("x_chemanalyst_input_requirements")
        if not isinstance(requirements, dict):
            return None
        templates = requirements.get("input_templates")
        if isinstance(templates, list) and templates:
            first = templates[0]
            if isinstance(first, dict):
                return str(first.get("path") or first.get("source_path") or "") or None
            return str(first)
        return None

    def _run_tool_with_previous_artifact(self, query: str, session_id: str, tool_name: str) -> dict | None:
        artifact = self.sessions.latest_artifact(session_id, file_types=(".xlsx", ".xls"), role="input")
        if not artifact:
            artifact = self.sessions.latest_artifact(session_id, file_types=(".xlsx", ".xls"))
        if not artifact:
            return None
        input_path = Path(artifact.path).resolve()
        output_dir = input_path.parent / f"{input_path.stem}_{tool_name.replace('.', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.handle_uploaded_tool(
            tool_name=tool_name,
            payload={"input_file": str(input_path), "file_path": str(input_path), "excel_file": str(input_path), "output_dir": str(output_dir), "query": query},
            session_id=session_id,
            user_query=query,
        )

    def handle_uploaded_tool(self, *, tool_name: str, payload: dict, session_id: str | None, user_query: str) -> dict:
        session = self.sessions.get_or_create(session_id)
        if tool_name == "database.experimental_history":
            file_path = str(payload.get("input_file") or payload.get("file_path") or payload.get("excel_file") or "").strip()
            if not file_path:
                answer = "Upload a petroleum fraction Experimental DB workbook before running property inference."
                return {"status": "awaiting_upload", "result": self._build_upload_spec_with_assets(tool_name, {"instructions": answer}), "message": "input_workbook_required", "session_id": session.session_id, "answer": answer}
            input_path = Path(file_path).resolve()
            self._remember_input_artifact(session, input_path, tool_name)
            query_text = user_query or str(payload.get("query") or "")
            tool_result = self._run_experimental_history_workbook_tool(
                {"input_file": str(input_path), "query": query_text, "top_k": self._parse_top_k(query_text, default=4)}
            )
            self.sessions.remember_tool_result(session.session_id, tool_name, tool_result)
            evidence = tool_result.get("evidence") if isinstance(tool_result, dict) else {}
            property_summary = evidence.get("property_summary") if isinstance(evidence, dict) else {}
            answer = "Experimental DB evidence inference completed."
            if isinstance(evidence, dict):
                answer += f" Retrieved {evidence.get('neighbor_count', 0)} historical neighbors."
            if isinstance(property_summary, dict) and property_summary:
                answer += f" Property evidence fields: {', '.join(list(property_summary)[:8])}."
            result_payload = {
                "evidence": evidence,
                "tool_evidence": tool_result.get("tool_evidence", {}) if isinstance(tool_result, dict) else {},
                "database_evidence": evidence,
                "input_file": str(input_path),
            }
            self._record_turn(session.session_id, user_query, answer)
            return {"status": tool_result.get("status", "success"), "result": result_payload, "message": tool_result.get("message", ""), "session_id": session.session_id, "answer": answer}

        if tool_name.startswith("tool.") and self.registry.get(tool_name) is not None:
            file_path = str(payload.get("input_file") or payload.get("file_path") or payload.get("excel_file") or "").strip()
            if not file_path:
                answer = "Upload the required input workbook or data file before running this analytical tool."
                return {"status": "awaiting_upload", "result": self._build_upload_spec_with_assets(tool_name, {"instructions": answer}), "message": "input_file_required", "session_id": session.session_id, "answer": answer}
            input_path = Path(file_path).resolve()
            self._remember_input_artifact(session, input_path, tool_name)
            output_dir = payload.get("output_dir") or str(input_path.parent / f"{input_path.stem}_{tool_name.replace('.', '_')}")
            tool_result = self.registry.call(
                tool_name,
                {"input_file": str(input_path), "file_path": str(input_path), "excel_file": str(input_path), "output_dir": str(output_dir)},
            )
            self.sessions.remember_tool_result(session.session_id, tool_name, tool_result)
            tool_evidence = tool_result.get("tool_evidence") if isinstance(tool_result, dict) else {}
            structured = tool_evidence.get("structured_result") if isinstance(tool_evidence, dict) else None
            artifacts = tool_result.get("artifacts") if isinstance(tool_result, dict) else {}
            answer = f"{tool_name} completed."
            if isinstance(structured, list):
                answer += f" Structured evidence rows: {len(structured)}."
            if isinstance(artifacts, dict) and artifacts:
                answer += " Generated artifacts are available in the tool result."
            result_payload = tool_result.get("result") if isinstance(tool_result, dict) else {}
            result_payload = dict(result_payload) if isinstance(result_payload, dict) else {}
            result_payload.setdefault("tool_evidence", tool_evidence if isinstance(tool_evidence, dict) else {})
            result_payload.setdefault("artifacts", artifacts if isinstance(artifacts, dict) else {})
            self._record_turn(session.session_id, user_query, answer)
            return {"status": tool_result.get("status", "success"), "result": result_payload, "message": tool_result.get("message", ""), "session_id": session.session_id, "answer": answer}

        return {"status": "error", "result": None, "message": f"Tool `{tool_name}` is not part of the public ChemAnalyst release.", "session_id": session.session_id}

    def _remember_input_artifact(self, session, input_path: Path, source_tool: str) -> None:
        artifact = SessionArtifact(
            path=str(input_path),
            file_name=input_path.name,
            file_type=input_path.suffix.lower(),
            source_tool=source_tool,
            role="input",
        )
        if not any(item.path == artifact.path for item in session.artifacts):
            session.artifacts.append(artifact)
            session.artifacts = session.artifacts[-12:]

    def _run_experimental_history_workbook_tool(self, payload: dict) -> dict:
        input_file = str(payload.get("input_file") or payload.get("file_path") or payload.get("excel_file") or "").strip()
        if not input_file:
            return {"status": "awaiting_upload", "message": "input_workbook_required", "result": None}
        workbook = Path(input_file).resolve()
        query = str(payload.get("query") or "Use IR and GC-FID evidence to infer petroleum fraction properties.")
        top_k = int(payload.get("top_k") or self._parse_top_k(query, default=4))
        workspace = Path(__file__).resolve().parents[2] / "data" / "experimental_db"
        evidence = build_petroleum_fraction_spectral_workbook_evidence(
            workbook=workbook,
            template_definition=SampleDatabase(workspace).get_template_definition(),
            query=query,
            top_k=top_k,
        )
        evidence = attach_petroleum_expert_reasoning(evidence, UnifiedLLMClient())
        status = str(evidence.get("status") or "success") if isinstance(evidence, dict) else "error"
        return {
            "status": status,
            "message": str(evidence.get("message") or "experimental_db_evidence_ready") if isinstance(evidence, dict) else "experimental_db_evidence_failed",
            "sample_id": evidence.get("sample_id") if isinstance(evidence, dict) else workbook.stem,
            "evidence": evidence,
            "database_evidence": evidence,
            "tool_evidence": {
                "capability_id": "database.experimental_history",
                "status": status,
                "input_file": str(workbook),
                "top_k": top_k,
                "structured_result": evidence.get("property_summary", {}) if isinstance(evidence, dict) else {},
            },
            "result": {
                "sample_id": evidence.get("sample_id") if isinstance(evidence, dict) else workbook.stem,
                "neighbor_count": evidence.get("neighbor_count") if isinstance(evidence, dict) else None,
                "property_summary": evidence.get("property_summary", {}) if isinstance(evidence, dict) else {},
            },
        }

    @staticmethod
    def _parse_top_k(query: str, *, default: int = 4) -> int:
        match = re.search(r"(?:top[-_\s]*k|k)\s*[=:]?\s*(\d+)", str(query or ""), flags=re.I)
        if not match:
            return default
        return max(1, min(50, int(match.group(1))))

    def answer_query(
        self,
        query: str,
        task_type_override: str | None = None,
        session_id: str | None = None,
        planner_decision: PlannerDecision | None = None,
        rag_mode_override: str | None = None,
        retrieval_overrides: dict | None = None,
    ) -> dict:
        session = self.sessions.get_or_create(session_id)
        if session.pending_tool_name and self.planner.should_cancel_pending(query):
            pending = session.pending_tool_name
            self.sessions.set_pending_tool(session.session_id, None)
            answer = self._cancel_pending_answer(pending)
            self._record_turn(session.session_id, query, answer)
            return {"session_id": session.session_id, "task_type": "chat", "used_model": "none", "used_tools": [], "answer": answer, "raw_result": None}

        decision = planner_decision or self.planner.decide(query=query, session=session, task_type_override=task_type_override)
        if decision.task_type == "tool":
            reusable = self._run_tool_with_previous_artifact(query, session.session_id, decision.tool_name or "")
            if reusable:
                return reusable
            self.sessions.set_pending_tool(session.session_id, decision.tool_name)
            upload_spec = self._build_upload_spec_with_assets(decision.tool_name or "", decision.upload_spec)
            answer = upload_spec.get("instructions") or "Upload the required input file before running this tool."
            self._record_turn(session.session_id, query, answer)
            return {"session_id": session.session_id, "task_type": "tool", "used_model": "none", "used_tools": decision.used_tools, "answer": answer, "raw_result": upload_spec}

        if decision.action == "rag":
            rag_mode = (rag_mode_override or "auto").strip().lower()
            if rag_mode == "naive" or (rag_mode == "auto" and not bool(getattr(self.settings, "advanced_rag_enabled", False))):
                docs = retrieve(query, top_k=self.settings.rag_top_k, retrieval_overrides=retrieval_overrides)
                if not docs:
                    answer = "No relevant knowledge chunks were retrieved. Please ingest or rebuild the knowledge base first."
                    self._record_turn(session.session_id, query, answer)
                    return {"session_id": session.session_id, "task_type": "rag", "used_model": self.settings.active_llm_model, "used_tools": [], "answer": answer, "raw_result": []}
                rag_result = naive_rag_answer(query, top_k=self.settings.rag_top_k, retrieval_overrides=retrieval_overrides)
            elif rag_mode == "iterative_review":
                rag_result = iterative_review_rag_answer(query, retrieval_overrides=retrieval_overrides)
            else:
                rag_result = qa_oriented_rag_answer(query, retrieval_overrides=retrieval_overrides)
            answer = rag_result.get("answer", "")
            self._record_turn(session.session_id, query, answer)
            return {"session_id": session.session_id, "task_type": "rag", "used_model": rag_result.get("used_model", self.settings.active_llm_model), "used_tools": [], "answer": answer, "raw_result": rag_result.get("raw_result", [])}

        if decision.action == "reasoning":
            context = self._build_context(session.session_id, query)
            reason_query = query if not context else f"{context}\n\nUser query: {query}"
            answer = self.llm.reason(reason_query)
            self._record_turn(session.session_id, query, answer)
            return {"session_id": session.session_id, "task_type": "reasoning", "used_model": self.settings.active_llm_model, "used_tools": [], "answer": answer, "raw_result": None}

        if self.planner.is_general_chat_query(query):
            answer = self._identity_answer()
        else:
            context = self._build_context(session.session_id, query)
            answer = self.llm.chat(self._build_response_query(query), context=context)
        self._record_turn(session.session_id, query, answer)
        return {"session_id": session.session_id, "task_type": "chat", "used_model": self.settings.active_llm_model, "used_tools": [], "answer": answer, "raw_result": None}
