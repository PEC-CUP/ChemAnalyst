from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4


@dataclass(slots=True)
class SessionArtifact:
    path: str
    file_name: str
    file_type: str
    source_tool: str
    role: str


@dataclass(slots=True)
class SessionState:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_tool_name: str | None = None
    last_tool_result: dict | None = None
    active_goal: str | None = None
    pending_tool_name: str | None = None
    pending_confirmation_tool_name: str | None = None
    pending_confirmation_query: str | None = None
    last_plan: dict | None = None
    artifacts: list[SessionArtifact] = field(default_factory=list)
    staged_uploads: dict[str, list[str]] = field(default_factory=dict)


class SessionMemoryStore:
    """In-memory state store for multi-turn conversation and tool execution."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str | None = None) -> SessionState:
        normalized = (session_id or "").strip() or uuid4().hex
        if normalized not in self._sessions:
            self._sessions[normalized] = SessionState(session_id=normalized)
        return self._sessions[normalized]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get_or_create(session_id)
        session.messages.append({"role": role, "content": content})
        session.messages = session.messages[-12:]

    def update_goal(self, session_id: str, goal: str | None) -> None:
        if goal:
            self.get_or_create(session_id).active_goal = goal.strip()

    def set_pending_tool(self, session_id: str, tool_name: str | None) -> None:
        self.get_or_create(session_id).pending_tool_name = tool_name

    def set_pending_confirmation(self, session_id: str, tool_name: str | None, query: str | None = None) -> None:
        session = self.get_or_create(session_id)
        session.pending_confirmation_tool_name = tool_name
        session.pending_confirmation_query = query.strip() if query else None

    def clear_pending_confirmation(self, session_id: str) -> None:
        session = self.get_or_create(session_id)
        session.pending_confirmation_tool_name = None
        session.pending_confirmation_query = None

    def set_last_plan(self, session_id: str, plan: dict | None) -> None:
        self.get_or_create(session_id).last_plan = plan

    def add_staged_upload(self, session_id: str, tool_name: str, file_path: str) -> list[str]:
        session = self.get_or_create(session_id)
        files = session.staged_uploads.setdefault(tool_name, [])
        if file_path not in files:
            files.append(file_path)
        return list(files)

    def get_staged_uploads(self, session_id: str, tool_name: str) -> list[str]:
        return list(self.get_or_create(session_id).staged_uploads.get(tool_name, []))

    def clear_staged_uploads(self, session_id: str, tool_name: str) -> None:
        self.get_or_create(session_id).staged_uploads.pop(tool_name, None)

    def latest_artifact(
        self,
        session_id: str,
        *,
        file_types: tuple[str, ...] = (),
        source_tool: str | None = None,
        role: str | None = None,
    ) -> SessionArtifact | None:
        session = self.get_or_create(session_id)
        normalized_types = {item.lower() for item in file_types}
        for artifact in reversed(session.artifacts):
            if normalized_types and artifact.file_type.lower() not in normalized_types:
                continue
            if source_tool and artifact.source_tool != source_tool:
                continue
            if role and artifact.role != role:
                continue
            return artifact
        return None

    def remember_tool_result(self, session_id: str, tool_name: str, result: dict) -> None:
        session = self.get_or_create(session_id)
        session.last_tool_name = tool_name
        session.last_tool_result = self._compact_tool_result(result)
        session.pending_tool_name = None
        session.pending_confirmation_tool_name = None
        session.pending_confirmation_query = None
        self._harvest_artifacts(session, tool_name, session.last_tool_result)

    def build_recent_context(self, session_id: str, max_messages: int = 6) -> str:
        session = self.get_or_create(session_id)
        if not session.messages:
            return ""
        lines = ["Recent conversation:"]
        for item in session.messages[-max_messages:]:
            role = "User" if item["role"] == "user" else "ChemAnalyst"
            lines.append(f"{role}: {item['content']}")
        return "\n".join(lines)

    def build_task_state_context(self, session_id: str) -> str:
        session = self.get_or_create(session_id)
        lines: list[str] = []
        if session.active_goal:
            lines.append(f"Active goal: {session.active_goal}")
        if session.pending_tool_name:
            lines.append(f"Pending tool: {session.pending_tool_name}")
        if session.artifacts:
            lines.append("Session artifacts:")
            for artifact in session.artifacts[-5:]:
                lines.append(
                    f"- {artifact.file_name} ({artifact.file_type}) | role={artifact.role} | source={artifact.source_tool}"
                )
        return "\n".join(lines)

    def build_last_tool_context(self, session_id: str) -> str:
        session = self.get_or_create(session_id)
        if not session.last_tool_result:
            return ""
        payload = json.dumps(session.last_tool_result, ensure_ascii=False, indent=2)
        return f"Last tool result ({session.last_tool_name}):\n{payload}"

    def _harvest_artifacts(self, session: SessionState, tool_name: str, result: dict) -> None:
        if not isinstance(result, dict):
            return
        nested = result.get("result")
        if not isinstance(nested, dict):
            return

        file_keys = {
            "input_file": "input",
            "template_file": "input",
            "sample_file": "input",
            "output_excel": "output",
            "chart_html": "output",
            "extracted_text_file": "output",
            "report_path": "output",
        }
        for key, role in file_keys.items():
            value = nested.get(key)
            if not value:
                continue
            path = Path(str(value))
            artifact = SessionArtifact(
                path=str(path),
                file_name=path.name,
                file_type=path.suffix.lower(),
                source_tool=tool_name,
                role=role,
            )
            if not any(item.path == artifact.path for item in session.artifacts):
                session.artifacts.append(artifact)
        session.artifacts = session.artifacts[-12:]

    def _compact_tool_result(self, result: dict) -> dict:
        if not isinstance(result, dict):
            return result

        compact = json.loads(json.dumps(result, ensure_ascii=False))
        nested = compact.get("result")
        if isinstance(nested, dict):
            text = nested.get("extracted_text")
            if isinstance(text, str) and len(text) > 4000:
                nested["extracted_text"] = text[:4000] + "\n...[truncated]"
            preview = nested.get("text_preview")
            if isinstance(preview, str) and len(preview) > 1500:
                nested["text_preview"] = preview[:1500] + "\n...[truncated]"
        return compact

