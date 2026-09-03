from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def check_service(base_url: str) -> None:
    response = requests.get(base_url.rstrip("/") + "/api/isalive", timeout=10)
    response.raise_for_status()
    print(response.text.strip() or "GROBID is alive")


def check_pdf(base_url: str, pdf_path: Path) -> None:
    endpoint = base_url.rstrip("/") + "/api/processFulltextDocument"
    with pdf_path.open("rb") as fh:
        response = requests.post(
            endpoint,
            files={"input": (pdf_path.name, fh, "application/pdf")},
            data={"consolidateHeader": "1", "consolidateCitations": "1"},
            timeout=120,
        )
    response.raise_for_status()
    text = response.text
    print(f"Parsed TEI XML characters: {len(text)}")
    print(text[:500].replace("\n", " "))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GROBID service for ChemAnalyst.")
    parser.add_argument("--url", default="http://localhost:8070")
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()

    try:
        check_service(args.url)
        if args.pdf:
            check_pdf(args.url, args.pdf)
    except Exception as exc:
        print(f"GROBID check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
