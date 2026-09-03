from __future__ import annotations

"""
Import documents from multiple folders into one destination with hash-based dedup.

Why:
- Different folders may contain identical PDFs/DOCX/TXT with different paths/names.
- KB ingest uses `metadata.source` as part of chunk uid; duplicates across paths will
  otherwise be ingested twice.

What it does:
- Recursively scan input folders for supported suffixes
- Compute SHA256 for each file
- Copy unique files into dest dir (default: petroleum_kb/petroleum_kb/data/raw/imported)
- Write a manifest JSONL describing originals -> imported target

Example (PowerShell):
  python -B petroleum_kb/tools/import_dedupe.py `
    --input "data/private/literature/Part1 saturated hydrocarbon" `
    --input "data/private/literature/Part2 organic geochemistry" `
    --input "data/private/literature/Part3 petroleomics" `
    --dest petroleum_kb/petroleum_kb/data/raw/imported `
    --manifest petroleum_kb/petroleum_kb/data/metadata/import_manifest.jsonl
"""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".json", ".jsonl"}


def iter_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in inputs:
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(root)
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            files.append(path)
    return files


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_name(name: str) -> str:
    # Minimal sanitization for Windows filenames.
    return "".join("_" if c in '<>:"/\\\\|top*' else c for c in name)


@dataclass(frozen=True)
class ImportItem:
    sha256: str
    src: Path
    dest: Path | None
    is_duplicate: bool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", type=Path, required=True, help="Input folder or file (repeatable).")
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path("petroleum_kb/petroleum_kb/data/raw/imported"),
        help="Destination folder for unique files.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("petroleum_kb/petroleum_kb/data/metadata/import_manifest.jsonl"),
        help="Manifest JSONL output path.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Scan and report only; do not copy files.")
    ap.add_argument("--report-top", type=int, default=10, help="Print top duplicated hashes (0 disables).")
    args = ap.parse_args()

    inputs = [Path(p) for p in (args.input or [])]
    paths = iter_files(inputs)
    total = len(paths)
    if total == 0:
        raise SystemExit("No supported files found in inputs.")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.dest.mkdir(parents=True, exist_ok=True)

    seen: dict[str, Path] = {}
    hash_sources: dict[str, list[str]] = defaultdict(list)
    input_counts: dict[str, int] = defaultdict(int)
    items: list[ImportItem] = []

    for idx, path in enumerate(paths, start=1):
        print(f"[import] ({idx}/{total}) hashing: {path.name}")
        digest = sha256_file(path)
        hash_sources[digest].append(str(path))
        for root in inputs:
            try:
                path.resolve().relative_to(root.resolve())
                input_counts[str(root)] += 1
                break
            except Exception:
                continue
        if digest in seen:
            items.append(ImportItem(sha256=digest, src=path, dest=seen[digest], is_duplicate=True))
            continue

        # Unique: choose a deterministic name so reruns are stable.
        short = digest[:12]
        dest_name = safe_name(f"{short}__{path.name}")
        dest_path = args.dest / dest_name
        if not args.dry_run:
            shutil.copy2(path, dest_path)
        seen[digest] = dest_path
        items.append(ImportItem(sha256=digest, src=path, dest=dest_path, is_duplicate=False))

    unique = sum(1 for it in items if not it.is_duplicate)
    dup = total - unique

    with args.manifest.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(
                json.dumps(
                    {
                        "sha256": it.sha256,
                        "src": str(it.src),
                        "dest": str(it.dest) if it.dest else None,
                        "duplicate": bool(it.is_duplicate),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[import] done: total={total} unique={unique} duplicates={dup}")
    print(f"[import] manifest: {args.manifest}")
    if args.dry_run:
        print("[import] dry-run: no files copied")
    else:
        print(f"[import] dest: {args.dest}")

    if input_counts:
        print("[import] per-input file counts:")
        for key in sorted(input_counts.keys()):
            print(f"  - {key}: {input_counts[key]}")

    top_n = int(args.report_top or 0)
    if top_n > 0:
        dups = [(h, len(srcs)) for h, srcs in hash_sources.items() if len(srcs) > 1]
        dups.sort(key=lambda x: x[1], reverse=True)
        print(f"[import] top duplicated sha256 (showing {min(top_n, len(dups))}):")
        for h, cnt in dups[:top_n]:
            examples = hash_sources[h][:3]
            print(f"  - count={cnt} sha256={h}")
            for ex in examples:
                print(f"    * {ex}")


if __name__ == "__main__":
    main()
