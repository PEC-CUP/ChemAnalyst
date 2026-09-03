from __future__ import annotations

import json
from pathlib import Path

try:
    from config import KBConfig
except ImportError:  # pragma: no cover
    from petroleum_kb.config import KBConfig
from ..schemas import SourceRegistration
from ..utils import ensure_dir


def load_source_registry(config: KBConfig) -> list[SourceRegistration]:
    path = config.source_registry_file
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SourceRegistration(**item) for item in payload if isinstance(item, dict)]


def save_source_registry(config: KBConfig, items: list[SourceRegistration]) -> Path:
    ensure_dir(config.registry_dir)
    config.source_registry_file.write_text(
        json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config.source_registry_file


def register_source(
    *,
    config: KBConfig,
    path: Path,
    name: str | None = None,
    source_group_hint: str | None = None,
    task_group_hint: str | None = None,
    notes: str | None = None,
) -> dict:
    resolved = path.expanduser().resolve()
    items = load_source_registry(config)
    source_name = name or resolved.name

    for index, item in enumerate(items):
        if Path(item.path).resolve() == resolved:
            items[index] = SourceRegistration(
                name=source_name,
                path=str(resolved),
                enabled=True,
                source_group_hint=source_group_hint or item.source_group_hint,
                task_group_hint=task_group_hint or item.task_group_hint,
                notes=notes or item.notes,
            )
            save_source_registry(config, items)
            return {
                "status": "updated",
                "registry_file": str(config.source_registry_file),
                "source": items[index].to_dict(),
            }

    entry = SourceRegistration(
        name=source_name,
        path=str(resolved),
        enabled=True,
        source_group_hint=source_group_hint,
        task_group_hint=task_group_hint,
        notes=notes,
    )
    items.append(entry)
    save_source_registry(config, items)
    return {
        "status": "registered",
        "registry_file": str(config.source_registry_file),
        "source": entry.to_dict(),
    }


def list_sources(config: KBConfig) -> dict:
    items = load_source_registry(config)
    return {
        "status": "success",
        "registry_file": str(config.source_registry_file),
        "sources": [item.to_dict() for item in items],
    }
