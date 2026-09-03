from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


CapabilityType = Literal["knowledge", "tool", "database"]


@dataclass(slots=True)
class CapabilitySchema:
    capability_id: str
    capability_type: CapabilityType
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)
    invocation_mode: str = "internal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisRequest:
    query: str
    session_id: str
    template: str
    rag_mode: str | None = None
    task_type_override: str | None = None
    uploaded_artifacts: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvidenceBundle:
    capabilities_used: list[dict[str, Any]] = field(default_factory=list)
    tool_evidence: dict[str, Any] = field(default_factory=dict)
    knowledge_evidence: dict[str, Any] = field(default_factory=dict)
    database_evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    uncertainty_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisConclusion:
    direct_findings: dict[str, Any] = field(default_factory=dict)
    database_inference: dict[str, Any] = field(default_factory=dict)
    knowledge_interpretation: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    confidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
