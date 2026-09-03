from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.utils.logger import setup_logger


logger = setup_logger(__name__)

CapabilityCallable = Callable[[dict], dict]
ToolCallable = Callable[[dict], dict]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    requires_upload: bool = False
    required_upload_fields: tuple[str, ...] = ("input_file",)
    accepted_file_types: tuple[str, ...] = ()
    reusable_from_previous_file: bool = False
    intent_keywords: tuple[str, ...] = ()
    followup_keywords: tuple[str, ...] = ()
    preferred_extensions: tuple[str, ...] = ()


class CapabilityRegistry:
    """Registry for executable capabilities."""

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityCallable] = {}
        self._schemas: dict[str, CapabilitySchema] = {}

    def register(self, capability: CapabilitySchema, func: CapabilityCallable) -> None:
        self._capabilities[capability.capability_id] = func
        self._schemas[capability.capability_id] = capability
        logger.info("Registered capability: %s", capability.capability_id)

    def unregister(self, capability_id: str) -> bool:
        existed = capability_id in self._capabilities or capability_id in self._schemas
        self._capabilities.pop(capability_id, None)
        self._schemas.pop(capability_id, None)
        if existed:
            logger.info("Unregistered capability: %s", capability_id)
        return existed

    def get(self, capability_id: str) -> CapabilityCallable | None:
        return self._capabilities.get(capability_id)

    def get_schema(self, capability_id: str) -> CapabilitySchema | None:
        return self._schemas.get(capability_id)

    def call(self, capability_id: str, payload: dict) -> dict:
        capability = self.get(capability_id)
        if capability is None:
            return {"status": "error", "result": None, "message": f"Capability `{capability_id}` is not registered."}
        return capability(payload)

    def list_capabilities(self) -> list[CapabilitySchema]:
        return list(self._schemas.values())

    def get_function_calling_specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": schema.capability_id,
                    "description": schema.description,
                    "parameters": schema.input_schema,
                },
            }
            for schema in self._schemas.values()
        ]


class ToolRegistry:
    """Compatibility registry for the historical runtime agent path."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolCallable] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, name_or_capability: Any, func: ToolCallable, *, spec: ToolSpec | None = None) -> None:
        if (
            hasattr(name_or_capability, "capability_id")
            and hasattr(name_or_capability, "description")
            and hasattr(name_or_capability, "input_schema")
        ):
            capability = name_or_capability
            name = str(capability.capability_id)
            parameters = capability.input_schema if isinstance(capability.input_schema, dict) else {}
            required = parameters.get("required") if isinstance(parameters, dict) else []
            input_requirements = parameters.get("x_chemanalyst_input_requirements") if isinstance(parameters, dict) else {}
            file_type_text = ""
            if isinstance(input_requirements, dict):
                file_type_text = str(input_requirements.get("file_type") or "").lower()
            for field_name in ("input_file", "file_path", "excel_file", "artifact_path"):
                field_schema = parameters.get("properties", {}).get(field_name, {}) if isinstance(parameters.get("properties"), dict) else {}
                if isinstance(field_schema, dict):
                    file_type_text += " " + str(field_schema.get("description") or "").lower()
            accepted_file_types = (".xlsx", ".xls") if ("excel" in file_type_text or "xlsx" in file_type_text or "xls" in file_type_text) else (".xlsx", ".xls", ".csv", ".json", ".txt")
            required_upload_fields = tuple(
                field
                for field in required
                if str(field) in {"input_file", "file_path", "excel_file", "artifact_path"}
            )
            inferred_spec = ToolSpec(
                name=name,
                description=str(capability.description),
                parameters=parameters,
                requires_upload=bool(required_upload_fields),
                required_upload_fields=required_upload_fields or ("input_file",),
                accepted_file_types=accepted_file_types,
                reusable_from_previous_file=True,
                intent_keywords=tuple(token for token in name.replace("tool.", "").replace("_", " ").split() if token),
            )
            self._tools[name] = func
            self._specs[name] = inferred_spec
            logger.info("Registered tool from capability schema: %s", name)
            return

        name = str(name_or_capability)
        if spec is None:
            raise TypeError("ToolRegistry.register() missing required keyword-only argument: 'spec'")
        self._tools[name] = func
        self._specs[name] = spec
        logger.info("Registered tool: %s", name)

    def unregister(self, name: str) -> bool:
        existed = name in self._tools or name in self._specs
        self._tools.pop(name, None)
        self._specs.pop(name, None)
        if existed:
            logger.info("Unregistered tool: %s", name)
        return existed

    def get(self, name: str) -> ToolCallable | None:
        return self._tools.get(name)

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def call(self, name: str, payload: dict) -> dict:
        tool = self.get(name)
        if tool is None:
            return {"status": "error", "result": None, "message": f"Tool `{name}` is not registered."}
        return tool(payload)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def list_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def get_function_calling_specs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._specs.values()
        ]

