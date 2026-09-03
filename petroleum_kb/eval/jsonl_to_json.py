from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JSONL to pretty JSON array.")
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Input .jsonl path")
    ap.add_argument("--out", dest="out_path", type=Path, default=None, help="Output .json path")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path or in_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = list(iter_jsonl(in_path))
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()
