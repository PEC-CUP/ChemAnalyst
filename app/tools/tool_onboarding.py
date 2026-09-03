from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.agents.schemas import CapabilitySchema
from app.agents.task_specific_evaluation import TaskSpecificEvaluationCase


ToolInvocationType = Literal["python_script", "http_api", "python_callable", "manual"]


@dataclass(slots=True)
class ToolOnboardingSpec:
    """User-provided description for converting a new analytical tool into a capability.

    This object is metadata only. It does not execute uploaded code or call remote
    APIs. Execution adapters must be reviewed and registered separately.
    """

    tool_name: str
    description: str
    invocation_type: ToolInvocationType
    entrypoint: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    evidence_outputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)
    example_queries: list[str] = field(default_factory=list)
    quality_checks: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    upstream_capabilities: list[str] = field(default_factory=list)
    downstream_capabilities: list[str] = field(default_factory=list)
    input_bindings: dict[str, Any] = field(default_factory=dict)
    output_bindings: dict[str, Any] = field(default_factory=dict)
    database_integration_contract: dict[str, Any] = field(default_factory=dict)
    adapter_template: str | None = None
    adapter_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolOnboardingPackage:
    capability: CapabilitySchema
    mcp_manifest: dict[str, Any]
    evaluation_cases: list[TaskSpecificEvaluationCase]
    registration_plan: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.to_dict(),
            "mcp_manifest": self.mcp_manifest,
            "evaluation_cases": [case.to_dict() for case in self.evaluation_cases],
            "registration_plan": self.registration_plan,
            "warnings": self.warnings,
        }


def build_tool_onboarding_package(spec: ToolOnboardingSpec) -> ToolOnboardingPackage:
    warnings = _validate_spec(spec)
    capability_id = f"tool.{_slugify(spec.tool_name)}"
    capability = CapabilitySchema(
        capability_id=capability_id,
        capability_type="tool",
        description=spec.description,
        input_schema=spec.input_schema or {"input": "object"},
        output_schema=spec.output_schema or {"status": "string", "evidence": "object"},
        preconditions=spec.preconditions,
        artifacts_produced=spec.artifacts_produced,
        invocation_mode="external_adapter_required" if spec.invocation_type in {"python_script", "http_api"} else "internal",
    )
    return ToolOnboardingPackage(
        capability=capability,
        mcp_manifest=_build_mcp_manifest(spec, capability),
        evaluation_cases=_build_evaluation_cases(spec, capability),
        registration_plan=_build_registration_plan(spec, capability),
        warnings=warnings,
    )


def _build_mcp_manifest(spec: ToolOnboardingSpec, capability: CapabilitySchema) -> dict[str, Any]:
    return {
        "schema_version": "chemanalyst-mcp-tool.v1",
        "tool_name": spec.tool_name,
        "capability_id": capability.capability_id,
        "description": spec.description,
        "invocation": {
            "type": spec.invocation_type,
            "entrypoint": spec.entrypoint,
            "adapter_template": spec.adapter_template,
            "adapter_config": spec.adapter_config,
            "review_required": spec.invocation_type in {"python_script", "http_api"},
        },
        "input_schema": capability.input_schema,
        "output_schema": capability.output_schema,
        "evidence_contract": {
            "required_outputs": spec.evidence_outputs or ["structured_tool_evidence"],
            "quality_checks": spec.quality_checks or ["status_present", "structured_output_present"],
            "artifacts_produced": spec.artifacts_produced,
        },
        "workflow_relations": {
            "upstream_capabilities": spec.upstream_capabilities,
            "downstream_capabilities": spec.downstream_capabilities,
            "input_bindings": spec.input_bindings,
            "output_bindings": spec.output_bindings,
            "planner_hint": "Use upstream outputs as candidate inputs and expose this tool's output bindings as downstream evidence.",
        },
        "database_integration_contract": _build_database_integration_contract(spec),
    }


def _build_evaluation_cases(spec: ToolOnboardingSpec, capability: CapabilitySchema) -> list[TaskSpecificEvaluationCase]:
    queries = spec.example_queries or [f"Run {spec.tool_name} and return structured analytical evidence."]
    evidence_outputs = spec.evidence_outputs or ["tool_evidence"]
    return [
        TaskSpecificEvaluationCase(
            case_id=f"tool_onboarding_{_slugify(spec.tool_name)}_{idx}",
            evaluation_target="tool",
            query=query,
            expected_route="planner",
            expected_template="data_analysis",
            expected_capabilities=[capability.capability_id],
            expected_evidence=evidence_outputs,
            expected_sufficiency_status="sufficient",
            expected_conclusion_fields=["direct_findings", "uncertainty", "confidence"],
            notes="Generated during tool onboarding. The case checks whether the new tool can produce structured evidence for Planner.",
        )
        for idx, query in enumerate(queries, start=1)
    ]


def _build_registration_plan(spec: ToolOnboardingSpec, capability: CapabilitySchema) -> dict[str, Any]:
    return {
        "status": "adapter_required",
        "capability_id": capability.capability_id,
        "steps": [
            "review the tool script/API and define a safe execution adapter",
            "register CapabilitySchema in CapabilityRegistry or a capability provider",
            "map raw tool output to structured tool_evidence",
            "add generated evaluation cases to the task-specific evaluation suite",
            "run onboarding evaluation before enabling planner selection",
        ],
        "runtime_policy": {
            "do_not_execute_unreviewed_uploaded_code": True,
            "planner_selection_requires_registered_adapter": True,
        },
        "workflow_relations": {
            "upstream_capabilities": spec.upstream_capabilities,
            "downstream_capabilities": spec.downstream_capabilities,
            "input_bindings": spec.input_bindings,
            "output_bindings": spec.output_bindings,
        },
        "database_integration_contract": _build_database_integration_contract(spec),
    }


def _build_database_integration_contract(spec: ToolOnboardingSpec) -> dict[str, Any]:
    contract = dict(spec.database_integration_contract or {})
    if not contract:
        return {
            "enabled": False,
            "purpose": (
                "Optional. Describe how current-session tool evidence can be aligned with an Experimental DB "
                "channel; runtime tool execution does not ingest records automatically."
            ),
        }
    contract.setdefault("enabled", True)
    contract.setdefault("record_schema", "experimental-record.v1")
    contract.setdefault("feature_extraction_policy", "flexible_label_value_numeric_intersection")
    contract.setdefault("sample_id_binding", "input_file_stem")
    contract.setdefault("structured_result_path", "tool_evidence.structured_result")
    contract.setdefault(
        "comparison_policy",
        "extract numeric features from current tool evidence and curated database records at comparison time; compare shared labels without requiring a fixed component count",
    )
    contract.setdefault("runtime_ingestion", "disabled; database ingestion is an explicit Experimental DB Workbench action")
    return contract


def _validate_spec(spec: ToolOnboardingSpec) -> list[str]:
    warnings = []
    if not spec.tool_name.strip():
        warnings.append("missing_tool_name")
    if not spec.description.strip():
        warnings.append("missing_description")
    if spec.invocation_type in {"python_script", "http_api"} and not spec.entrypoint:
        warnings.append("missing_external_entrypoint")
    if not spec.input_schema:
        warnings.append("missing_input_schema")
    if not spec.output_schema:
        warnings.append("missing_output_schema")
    if not spec.evidence_outputs:
        warnings.append("missing_evidence_outputs")
    return warnings


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "unnamed_tool"
