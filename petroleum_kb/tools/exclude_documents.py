from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exclude specific documents from chunks jsonl")
    parser.add_argument("--input", type=Path, required=True, help="Input chunks jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output filtered chunks jsonl")
    parser.add_argument("--report", type=Path, required=True, help="Output exclusion report json")
    parser.add_argument("--exclude-doi", action="append", default=[], help="DOI to exclude; repeatable")
    parser.add_argument("--exclude-title-contains", action="append", default=[], help="Case-insensitive title substring to exclude; repeatable")
    parser.add_argument("--exclude-source-contains", action="append", default=[], help="Case-insensitive source path substring to exclude; repeatable")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    exclude_doi = {str(x).strip().lower() for x in args.exclude_doi if str(x).strip()}
    exclude_title_contains = [str(x).strip().lower() for x in args.exclude_title_contains if str(x).strip()]
    exclude_source_contains = [str(x).strip().lower() for x in args.exclude_source_contains if str(x).strip()]

    kept: list[dict] = []
    removed_chunks = 0
    removed_docs: dict[tuple[str, str, str], dict[str, object]] = {}

    with args.input.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            md = item.get("metadata") or {}
            title = str(md.get("title") or "").strip()
            doi = str(md.get("doi") or "").strip()
            source = str(md.get("source") or "").strip()

            title_l = title.lower()
            doi_l = doi.lower()
            source_l = source.lower()

            matched = (
                (doi_l in exclude_doi)
                or any(term in title_l for term in exclude_title_contains)
                or any(term in source_l for term in exclude_source_contains)
            )

            if matched:
                removed_chunks += 1
                key = (title, doi, source)
                row = removed_docs.setdefault(
                    key,
                    {
                        "title": title,
                        "doi": doi,
                        "source": source,
                        "chunk_count": 0,
                    },
                )
                row["chunk_count"] = int(row["chunk_count"]) + 1
                continue

            kept.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for item in kept:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_chunks": len(kept) + removed_chunks,
        "output_chunks": len(kept),
        "removed_chunks": removed_chunks,
        "removed_documents": sorted(removed_docs.values(), key=lambda x: (-int(x["chunk_count"]), str(x["title"]))),
        "criteria": {
            "exclude_doi": sorted(exclude_doi),
            "exclude_title_contains": exclude_title_contains,
            "exclude_source_contains": exclude_source_contains,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
