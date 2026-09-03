"""Build a small benchmark dataset for representative QA examples.

The script copies selected questions from existing benchmark files into a
standalone custom JSONL dataset. It does not modify the source benchmark files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "petroleum_kb" / "eval" / "datasets"


EXAMPLES = [
    {
        "label": "Naive baseline single-doc",
        "path": DATASETS / "benchmark.naive_baseline.single.200.en.final.jsonl",
        "id": "sq0004",
    },
    {
        "label": "Naive baseline two-doc",
        "path": DATASETS / "benchmark.naive_baseline.twodoc.200.en.final.jsonl",
        "id": "sq0003",
    },
    {
        "label": "Reviewed single-doc",
        "path": DATASETS / "benchmark.reviewed.single.200.en.final.jsonl",
        "id": "sq0044",
    },
    {
        "label": "Reviewed two-doc",
        "path": DATASETS / "benchmark.reviewed.twodoc.200.en.final.jsonl",
        "id": "sq0007",
    },
    {
        "label": "Specific-fact open QA",
        "path": DATASETS / "benchmark.reviewed.specific_fact.single.open.100.en.jsonl",
        "id": "sq0048",
    },
    {
        "label": "Specific-fact MCQ",
        "path": DATASETS / "benchmark.reviewed.specific_fact.single.mcq.100.en.jsonl",
        "id": "sq0017",
    },
]


def _load_item(path: Path, item_id: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") == item_id:
                return obj
    raise SystemExit(f"Item not found: id={item_id} path={path}")


def build_dataset(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, spec in enumerate(EXAMPLES, start=1):
        obj = _load_item(spec["path"], spec["id"])
        copied = dict(obj)
        copied["id"] = f"ex{idx:03d}"
        copied["source_id"] = obj.get("id")
        copied["example_label"] = spec["label"]
        copied["example_source_dataset"] = spec["path"].name
        rows.append(copied)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} representative examples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASETS / "custom" / "representative_qa_examples_6.jsonl",
    )
    args = parser.parse_args()
    build_dataset(args.out)


if __name__ == "__main__":
    main()
