from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.llm.deepseek_client import infer_output_language
from app.rag.advanced_rag import compact_cited_evidences
from app.rag.kb_backend import build_structured_rag_evidence


class SupportLayerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_support_layer_ui_and_db_status_are_served(self) -> None:
        page = self.client.get("/support-layer")
        self.assertEqual(page.status_code, 200)
        self.assertIn("ChemAnalyst Support Layer", page.text)
        self.assertIn("/rag-workbench", page.text)
        self.assertIn("/experimental-db-workbench", page.text)

        stylesheet = self.client.get("/static/support_workbench.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn(".workbench-header", stylesheet.text)

        rag_page = self.client.get("/rag-workbench")
        self.assertEqual(rag_page.status_code, 200)
        self.assertIn("RAG manual", rag_page.text)
        self.assertIn("RAG query test", rag_page.text)

        db_page = self.client.get("/experimental-db-workbench")
        self.assertEqual(db_page.status_code, 200)
        self.assertIn("Experimental Database manual", db_page.text)
        self.assertIn("Experimental Database evidence test", db_page.text)

        status = self.client.get("/support-layer/experimental-db/status")
        self.assertEqual(status.status_code, 200)
        self.assertIn("no trained property regressor", json.dumps(status.json()).lower())

        pipeline = self.client.post("/support-layer/rag/pipeline", json={"action": "list_sources"})
        self.assertEqual(pipeline.status_code, 200)
        self.assertEqual(pipeline.json()["action"], "list_sources")
        self.assertIn("pipeline_config", pipeline.json())

    def test_structured_rag_evidence_contains_bibliographic_references(self) -> None:
        structured = build_structured_rag_evidence(
            {
                "evidences": [
                    {
                        "doc_id": "doi:10.1/demo",
                        "citation": {
                            "title": "Petroleum Analysis Demo",
                            "authors": ["A. Author", "B. Author"],
                            "year": 2025,
                            "journal": "Analytical Demo",
                            "doi": "10.1/demo",
                        },
                    }
                ]
            },
            mode="qa_oriented",
        )
        references = structured["knowledge_evidence"]["references"]
        self.assertEqual(references[0]["label"], "DOC 1")
        self.assertEqual(references[0]["authors"], ["A. Author", "B. Author"])
        self.assertEqual(references[0]["journal"], "Analytical Demo")

    def test_chat_references_open_from_inline_doc_citations(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('id="ragReferenceModal"', page.text)
        self.assertIn("function renderAssistantInline", page.text)
        self.assertIn("function showRagReference", page.text)
        self.assertNotIn('title.textContent = "References"', page.text)
        self.assertNotIn("authors.join", page.text)

    def test_final_rag_citations_are_compacted_to_used_documents(self) -> None:
        answer, evidences = compact_cited_evidences(
            "DPF separates fractions [DOC 1] and reduces complexity [DOC 7]. It is supported by [DOC 7].",
            [{"doc_id": f"doc-{index}"} for index in range(1, 8)],
        )
        self.assertEqual(
            answer,
            "DPF separates fractions [DOC 1] and reduces complexity [DOC 2]. It is supported by [DOC 2].",
        )
        self.assertEqual([item["doc_id"] for item in evidences], ["doc-1", "doc-7"])

    def test_chinese_queries_with_english_terms_stay_chinese(self) -> None:
        self.assertEqual(infer_output_language("\u8bf7\u8bf4\u660e DPF method"), "zh")
        self.assertEqual(infer_output_language("Explain the DPF method"), "en")
        self.assertEqual(infer_output_language("What are DPF applications?"), "en")


if __name__ == "__main__":
    unittest.main()
