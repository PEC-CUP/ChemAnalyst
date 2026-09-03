from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _repo_root() -> Path:
    # app/rag/run_log.py -> repo root
    return Path(__file__).resolve().parents[2]


def _default_log_path() -> Path:
    return _repo_root() / "data" / "logs" / "rag_runs.jsonl"


def new_run_id() -> str:
    return uuid4().hex


def log_event(event: dict) -> Path | None:
    """Append a single JSONL line for reproducible RAG runs."""

    enabled = os.getenv("RAG_RUN_LOG_ENABLED", "").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    path = Path(os.getenv("RAG_RUN_LOG_PATH", "").strip() or _default_log_path())
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(event)
    payload.setdefault("ts", datetime.now(timezone.utc).isoformat())

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path

