from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.agents.task_classifier import classify_front_door


class _SimpleRagLLM:
    def chat(self, prompt: str, context: str | None = None) -> str:
        return '{"route":"simple","simple_mode":"rag","reason":"concept explanation"}'


class TaskClassifierTests(unittest.TestCase):
    def test_public_rag_mode_stays_simple_rag(self) -> None:
        session = SimpleNamespace(artifacts=[], last_tool_result=None)
        decision = classify_front_door(
            query="What is catalytic cracking?",
            session=session,
            task_type_override="rag",
            rag_mode_override="qa_oriented",
        )
        self.assertEqual(decision.route, "simple")
        self.assertEqual(decision.simple_mode, "rag")

    def test_simple_chat_for_general_greeting(self) -> None:
        session = SimpleNamespace(artifacts=[], last_tool_result=None)
        decision = classify_front_door(query="hello", session=session)
        self.assertEqual(decision.route, "simple")
        self.assertEqual(decision.simple_mode, "chat")

    def test_complex_when_session_has_artifacts(self) -> None:
        session = SimpleNamespace(artifacts=["a.xlsx"], last_tool_result=None)
        decision = classify_front_door(
            query="Please analyze the uploaded workbook",
            session=session,
        )
        self.assertEqual(decision.route, "complex")

    def test_contextual_result_followup_with_artifacts_stays_complex_before_llm(self) -> None:
        session = SimpleNamespace(artifacts=["a.xlsx"], last_tool_result=None)
        decision = classify_front_door(
            query="Use this uploaded workbook to infer density",
            session=session,
            llm=_SimpleRagLLM(),
        )
        self.assertEqual(decision.route, "complex")

    def test_llm_can_keep_gc_principle_question_simple(self) -> None:
        session = SimpleNamespace(artifacts=[], last_tool_result=None)
        decision = classify_front_door(
            query="Explain the principle of GC-FID",
            session=session,
            llm=_SimpleRagLLM(),
        )
        self.assertEqual(decision.route, "simple")
        self.assertEqual(decision.simple_mode, "rag")

    def test_explicit_data_analysis_intent_without_llm_is_complex(self) -> None:
        session = SimpleNamespace(artifacts=[], last_tool_result=None)
        decision = classify_front_door(
            query="Analyze the uploaded file and calculate the property evidence",
            session=session,
        )
        self.assertEqual(decision.route, "complex")


if __name__ == "__main__":
    unittest.main()


