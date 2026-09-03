from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from app.router.task_router import detect_task_type, normalize_task_type


FrontDoorRoute = Literal["simple", "complex"]
SimpleMode = Literal["chat", "rag", "reasoning"]


@dataclass(slots=True)
class FrontDoorDecision:
    route: FrontDoorRoute
    simple_mode: SimpleMode | None = None
    reason: str = ""


_CHAT_MARKERS = ("hello", "hi", "help", "thanks")
_GENERAL_CHAT_PHRASES = ("what can you do", "who are you", "how to use", "help me")
_ANALYSIS_ACTION_MARKERS = ("analyze", "calculate", "process", "infer", "predict", "retrieve", "compare", "build", "run", "use", "upload", "uploaded")
_DATA_OBJECT_MARKERS = ("file", "workbook", "xlsx", "sample", "database", "property", "density", "bp_", "distillation", "composition", "evidence")
_CONTEXTUAL_RESULT_MARKERS = ("this", "that", "result", "previous", "uploaded", "file", "workbook", "sample", "evidence", "answer")
_CONTEXTUAL_ANALYSIS_MARKERS = ("analyze", "explain", "infer", "calculate", "continue", "use", "compare", "summarize", "interpret", "retrieve")


def is_general_chat_query(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered in _CHAT_MARKERS or any(marker in lowered for marker in _GENERAL_CHAT_PHRASES)


def _looks_like_explicit_data_analysis_intent(query: str) -> bool:
    lowered = query.lower()
    has_action = any(marker in lowered for marker in _ANALYSIS_ACTION_MARKERS)
    has_object = any(marker in lowered for marker in _DATA_OBJECT_MARKERS)
    has_english_phrase = any(
        phrase in lowered
        for phrase in (
            "analyze uploaded",
            "process uploaded",
            "uploaded file",
            "historical database",
            "database inference",
        )
    )
    return (has_action and has_object) or has_english_phrase


def _looks_like_contextual_artifact_followup(query: str) -> bool:
    lowered = query.lower()
    has_context_ref = any(marker in lowered for marker in _CONTEXTUAL_RESULT_MARKERS)
    has_analysis_intent = any(marker in lowered for marker in _CONTEXTUAL_ANALYSIS_MARKERS)
    return has_context_ref and has_analysis_intent


def classify_front_door(
    *,
    query: str,
    session,
    task_type_override: str | None = None,
    rag_mode_override: str | None = None,
    llm=None,
) -> FrontDoorDecision:
    override = normalize_task_type(task_type_override)
    rag_mode = (rag_mode_override or "qa_oriented").strip().lower()

    if override == "tool":
        return FrontDoorDecision(route="complex", reason="explicit_complex_override")

    if override in {"chat", "rag", "reasoning"}:
        return FrontDoorDecision(route="simple", simple_mode=override, reason="explicit_simple_override")

    if is_general_chat_query(query):
        return FrontDoorDecision(route="simple", simple_mode="chat", reason="general_chat")

    if getattr(session, "artifacts", None) or getattr(session, "last_tool_result", None):
        if _looks_like_explicit_data_analysis_intent(query) or _looks_like_contextual_artifact_followup(query):
            return FrontDoorDecision(route="complex", reason="session_contextual_analysis_followup")
        if llm is not None:
            decision = _llm_front_door(query=query, llm=llm)
            if decision is not None:
                return decision
        return FrontDoorDecision(route="complex", reason="session_has_analysis_context")

    if llm is not None:
        decision = _llm_front_door(query=query, llm=llm)
        if decision is not None:
            return decision

    if _looks_like_explicit_data_analysis_intent(query):
        return FrontDoorDecision(route="complex", reason="explicit_data_analysis_intent")

    task_type = override or detect_task_type(query)
    if task_type == "rag":
        return FrontDoorDecision(route="simple", simple_mode="rag", reason="rule_rag")
    if task_type == "reasoning":
        return FrontDoorDecision(route="simple", simple_mode="reasoning", reason="rule_reasoning")
    return FrontDoorDecision(route="simple", simple_mode="chat", reason="default_chat")


def _llm_front_door(*, query: str, llm) -> FrontDoorDecision | None:
    prompt = (
        "You are ChemAnalyst's front-door task classifier. Output JSON only.\n"
        "Classify the user query into either:\n"
        "- simple: general chat, knowledge QA, or reasoning without structured data processing\n"
        "- complex: uploaded-data analysis, multi-step analysis, tool/database-assisted analysis, or mixed analysis\n"
        "Important: merely mentioning GC, chromatography, or a scientific concept does not make a query complex.\n"
        "Only choose complex when the user is asking to analyze uploaded data, run a multi-step workflow, or combine tools/database evidence.\n"
        "If simple, also choose one simple_mode from: chat, rag, reasoning.\n"
        'Output format: {"route":"simple|complex","simple_mode":"chat|rag|reasoning|null","reason":""}\n'
        f"User query: {query}"
    )
    try:
        raw = llm.chat(prompt)
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        data = json.loads(match.group(0))
        route = str(data.get("route", "")).strip().lower()
        simple_mode = data.get("simple_mode")
        if route not in {"simple", "complex"}:
            return None
        if route == "simple":
            mode = str(simple_mode or "").strip().lower()
            if mode not in {"chat", "rag", "reasoning"}:
                mode = "chat"
            return FrontDoorDecision(route="simple", simple_mode=mode, reason=str(data.get("reason", "")).strip())
        return FrontDoorDecision(route="complex", reason=str(data.get("reason", "")).strip())
    except Exception:
        return None



