from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def stable_chunk_uid(source: str, chunk_id: int, text: str) -> str:
    raw = f"{source}|{chunk_id}|{text[:200]}".encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split())
