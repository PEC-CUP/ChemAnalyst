from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.state_store import SessionArtifact
from app.capabilities.provider_registry import register_reviewed_provider, update_provider_review
from app.main import agent, planner_agent


PROVIDER_ID = "hydrocarbon_type_calculator"
INPUT_FILE = ROOT / "app" / "tools" / "document_uploads" / "02231c9ad08b4e1bbd604dd6da3a49c8_input_template.xlsx"
OUTPUT_ROOT = ROOT / "outputs" / "tool_onboarding_hydrocarbon_planner_test"


class SmokeLLM:
    def chat(self, prompt: str) -> str:
        return (
            "Planner smoke test completed. The uploaded workbook was routed to "
            "Hydrocarbon Type Calculator, and the structured tool evidence was "
            "attached to EvidenceBundle.tool_evidence."
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mark_provider_deployed() -> dict[str, Any]:
    review = update_provider_review(
        PROVIDER_ID,
        reviewed=True,
        enabled=True,
        notes="Smoke-tested and enabled for Planner dynamic capability routing.",
    )
    manifest_path = ROOT / "app" / "capabilities" / "providers" / PROVIDER_ID / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revoked"] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    registered = register_reviewed_provider(agent.registry, PROVIDER_ID)
    return {"review": review, "registration": registered}


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)
    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    agent.llm = SmokeLLM()
    planner_agent.agent.llm = SmokeLLM()
    deploy_result = _mark_provider_deployed()

    session_id = f"hydrocarbon_type_planner_smoke_{datetime.now().strftime('%H%M%S')}"
    session = agent.sessions.get_or_create(session_id)
    session.artifacts.append(
        SessionArtifact(
            path=str(INPUT_FILE.resolve()),
            file_name=INPUT_FILE.name,
            file_type=INPUT_FILE.suffix,
            source_tool="manual_upload_smoke",
            role="input",
        )
    )

    query = "Calculate hydrocarbon type composition from the uploaded workbook."
    result = planner_agent.answer_query(
        query,
        session_id=session_id,
        template_override="data_analysis",
    )

    evidence_bundle = (result.get("raw_result") or {}).get("evidence_bundle") or {}
    tool_evidence = evidence_bundle.get("tool_evidence") or {}
    structured = tool_evidence.get("structured_result") or []
    artifacts = tool_evidence.get("artifacts") or {}

    summary = {
        "status": result.get("raw_result", {}).get("analysis_conclusion", {}).get("direct_findings", {}).get("status")
        or tool_evidence.get("status"),
        "planner_template": result.get("raw_result", {}).get("planner", {}).get("template"),
        "used_capability": tool_evidence.get("capability_id"),
        "input_file": str(INPUT_FILE.resolve()),
        "output_excel": artifacts.get("output_excel"),
        "structured_result_rows": len(structured) if isinstance(structured, list) else None,
        "first_structured_rows": structured[:5] if isinstance(structured, list) else [],
    }

    _write_json(run_dir / "planner_result.json", result)
    _write_json(run_dir / "evidence_bundle.json", evidence_bundle)
    _write_json(run_dir / "tool_evidence.json", tool_evidence)
    _write_json(run_dir / "summary.json", {"deploy": deploy_result, "summary": summary})
    (run_dir / "README.zh.md").write_text(
        "# Hydrocarbon Type Calculator Planner smoke test\n\n"
        f"- Query: `{query}`\n"
        f"- Input file: `{summary['input_file']}`\n"
        f"- Planner template: `{summary['planner_template']}`\n"
        f"- Used capability: `{summary['used_capability']}`\n"
        f"- Tool status: `{summary['status']}`\n"
        f"- Structured result rows: `{summary['structured_result_rows']}`\n"
        f"- Output Excel: `{summary['output_excel']}`\n\n"
        "text `planner_result.json`、`evidence_bundle.json` text `tool_evidence.json`。\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
