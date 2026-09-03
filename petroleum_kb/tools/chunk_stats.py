from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path


TABLE_FRAGMENT_RE = re.compile(
    r"(top:\b(top:m/z|ppm|rt|fid|tof|ei|esi|appi|api)\b|[%\d]\s*$|^[\w\-\+\./]+(top:\s+[\w\-\+\./]+){0,6}$)",
    re.IGNORECASE,
)
ACK_SECTION_RE = re.compile(
    r"\b(acknowledg(top:e)topmentstop|funding|author contributionstop|conflictstop of interest|supporting information)\b",
    re.IGNORECASE,
)
REFERENCE_LEAD_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe chunk size and quality distributions")
    parser.add_argument("--input", type=Path, required=True, help="Input chunks.jsonl or chunks_clean.jsonl")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--sample-out", type=Path, default=None, help="Optional sample text report")
    return parser


def word_count(text: str) -> int:
    return len(text.split())


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def approx_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic for mixed scientific English text.
        return max(1, round(len(text) / 4))


def quality_bucket(text: str, metadata: dict) -> str:
    wc = word_count(text)
    section = str(metadata.get("section") or "").strip()
    if wc < 40:
        if TABLE_FRAGMENT_RE.search(" ".join(text.split())):
            return "short_table_fragment"
        return "short_fragment"
    if REFERENCE_LEAD_RE.match(section) or ACK_SECTION_RE.search(section) or ACK_SECTION_RE.search(text[:240]):
        return "reference_or_ack"
    if alpha_ratio(text) < 0.55:
        return "low_alpha_ratio"
    if wc >= 450:
        return "very_long"
    if wc >= 300:
        return "long"
    return "normal"


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    idx = max(0, min(len(values) - 1, round((len(values) - 1) * q)))
    return values[idx]


def bucket_counts(values: list[int], bounds: list[tuple[str, int, int | None]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, lo, hi in bounds:
        if hi is None:
            out[label] = sum(1 for v in values if v >= lo)
        else:
            out[label] = sum(1 for v in values if lo <= v <= hi)
    return out


def main() -> None:
    args = build_parser().parse_args()

    chars: list[int] = []
    words: list[int] = []
    tokens: list[int] = []
    quality: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, object]]] = {
        "short_fragment": [],
        "short_table_fragment": [],
        "reference_or_ack": [],
        "low_alpha_ratio": [],
        "very_long": [],
    }

    with args.input.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            metadata = obj.get("metadata") or {}

            c = len(text)
            w = word_count(text)
            t = approx_tokens(text)
            q = quality_bucket(text, metadata)

            chars.append(c)
            words.append(w)
            tokens.append(t)
            quality[q] += 1

            doc_key = str(metadata.get("title") or metadata.get("source") or "<unknown>")
            doc_counts[doc_key] += 1

            if q in samples and len(samples[q]) < 3:
                samples[q].append(
                    {
                        "title": metadata.get("title"),
                        "section": metadata.get("section"),
                        "words": w,
                        "chars": c,
                        "text_preview": text[:500].replace("\n", " "),
                    }
                )

    chars_sorted = sorted(chars)
    words_sorted = sorted(words)
    tokens_sorted = sorted(tokens)

    report = {
        "input": str(args.input),
        "chunks": len(words),
        "documents": len(doc_counts),
        "avg_chunks_per_doc": round(len(words) / max(1, len(doc_counts)), 2),
        "char_stats": {
            "min": min(chars_sorted) if chars_sorted else 0,
            "p10": pct(chars_sorted, 0.10),
            "median": pct(chars_sorted, 0.50),
            "p90": pct(chars_sorted, 0.90),
            "max": max(chars_sorted) if chars_sorted else 0,
            "mean": round(statistics.mean(chars_sorted), 1) if chars_sorted else 0,
            "buckets": bucket_counts(
                chars_sorted,
                [
                    ("<200", 0, 199),
                    ("200-499", 200, 499),
                    ("500-999", 500, 999),
                    ("1000-1499", 1000, 1499),
                    ("1500+", 1500, None),
                ],
            ),
        },
        "word_stats": {
            "min": min(words_sorted) if words_sorted else 0,
            "p10": pct(words_sorted, 0.10),
            "median": pct(words_sorted, 0.50),
            "p90": pct(words_sorted, 0.90),
            "max": max(words_sorted) if words_sorted else 0,
            "mean": round(statistics.mean(words_sorted), 1) if words_sorted else 0,
            "buckets": bucket_counts(
                words_sorted,
                [
                    ("<40", 0, 39),
                    ("40-79", 40, 79),
                    ("80-119", 80, 119),
                    ("120-199", 120, 199),
                    ("200-299", 200, 299),
                    ("300-449", 300, 449),
                    ("450+", 450, None),
                ],
            ),
        },
        "token_stats": {
            "min": min(tokens_sorted) if tokens_sorted else 0,
            "p10": pct(tokens_sorted, 0.10),
            "median": pct(tokens_sorted, 0.50),
            "p90": pct(tokens_sorted, 0.90),
            "max": max(tokens_sorted) if tokens_sorted else 0,
            "mean": round(statistics.mean(tokens_sorted), 1) if tokens_sorted else 0,
            "approximation": "tiktoken cl100k_base if available, else chars/4 heuristic",
        },
        "quality_buckets": dict(quality),
        "top_documents": [
            {"title_or_source": title, "chunk_count": count}
            for title, count in doc_counts.most_common(20)
        ],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sample_out:
        args.sample_out.parent.mkdir(parents=True, exist_ok=True)
        args.sample_out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
