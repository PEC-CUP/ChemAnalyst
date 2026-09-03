from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TABLE_FRAGMENT_RE = re.compile(
    r"(top:\b(top:m/z|ppm|rt|fid|tof|ei|esi|appi|api)\b|[%\d]\s*$|^[\w\-\+\./]+(top:\s+[\w\-\+\./]+){0,6}$)",
    re.IGNORECASE,
)
MOSTLY_SYMBOL_RE = re.compile(r"^[\W\d_]+$")
REFERENCE_LEAD_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
ACK_SECTION_RE = re.compile(
    r"\b(acknowledg(top:e)topmentstop|funding|author contributionstop|conflictstop of interest|supporting information)\b",
    re.IGNORECASE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Light cleanup for petroleum_kb chunks.jsonl")
    parser.add_argument("--input", type=Path, required=True, help="Input chunks.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output cleaned jsonl")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults to <output>.report.json",
    )
    parser.add_argument("--min-words", type=int, default=40, help="Drop chunks shorter than this")
    parser.add_argument(
        "--max-words",
        type=int,
        default=0,
        help="Drop chunks longer than this. Use 0 to disable long-chunk filtering.",
    )
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.55,
        help="Drop chunks with too little alphabetic content",
    )
    parser.add_argument(
        "--keep-short-methods",
        action="store_true",
        help="Keep short chunks if they still look like method/evidence sentences",
    )
    return parser


def word_count(text: str) -> int:
    return len(text.split())


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def looks_like_reference_or_ack(text: str, metadata: dict) -> bool:
    section = str(metadata.get("section") or "").strip()
    if REFERENCE_LEAD_RE.match(section):
        return True
    if ACK_SECTION_RE.search(section):
        return True
    if ACK_SECTION_RE.search(text[:240]):
        return True
    return False


def looks_like_table_fragment(text: str) -> bool:
    compact = " ".join(text.split())
    if not compact:
        return True
    if MOSTLY_SYMBOL_RE.match(compact):
        return True
    if compact.count("|") >= 2 or compact.count("\t") >= 2:
        return True
    if TABLE_FRAGMENT_RE.search(compact) and len(compact.split()) <= 14:
        return True
    return False


def looks_like_method_sentence(text: str) -> bool:
    lower = text.lower()
    cues = (
        "used to",
        "was used",
        "were used",
        "analyzed by",
        "measured by",
        "detected by",
        "characterized by",
        "an energy-dispersive",
        "gc×gc",
        "ft-icr",
        "mass spectrometry",
        "electrospray",
        "photoionization",
    )
    return any(cue in lower for cue in cues)


def classify_drop_reason(text: str, metadata: dict, *, min_words: int, max_words: int, min_alpha_ratio: float, keep_short_methods: bool) -> str | None:
    wc = word_count(text)
    if wc < min_words:
        if keep_short_methods and looks_like_method_sentence(text):
            return None
        if looks_like_table_fragment(text):
            return "short_table_fragment"
        return "too_short"
    if max_words > 0 and wc > max_words:
        return "too_long"
    if looks_like_reference_or_ack(text, metadata):
        return "reference_or_ack"
    if alpha_ratio(text) < min_alpha_ratio:
        return "low_alpha_ratio"
    if looks_like_table_fragment(text) and wc < 80:
        return "table_fragment"
    return None


def main() -> None:
    args = build_parser().parse_args()
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")

    kept: list[dict] = []
    drop_reasons: Counter[str] = Counter()
    before_lengths: list[int] = []
    after_lengths: list[int] = []

    with args.input.open("r", encoding="utf-8", errors="ignore") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            text = str(item.get("text") or "").strip()
            metadata = item.get("metadata") or {}
            before_lengths.append(word_count(text))
            reason = classify_drop_reason(
                text,
                metadata,
                min_words=args.min_words,
                max_words=args.max_words,
                min_alpha_ratio=args.min_alpha_ratio,
                keep_short_methods=args.keep_short_methods,
            )
            if reason is not None:
                drop_reasons[reason] += 1
                continue
            kept.append(item)
            after_lengths.append(word_count(text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for item in kept:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_chunks": len(before_lengths),
        "output_chunks": len(after_lengths),
        "removed_chunks": len(before_lengths) - len(after_lengths),
        "drop_reasons": dict(drop_reasons),
        "params": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "min_alpha_ratio": args.min_alpha_ratio,
            "keep_short_methods": bool(args.keep_short_methods),
        },
        "length_summary": {
            "before_min": min(before_lengths) if before_lengths else 0,
            "before_max": max(before_lengths) if before_lengths else 0,
            "after_min": min(after_lengths) if after_lengths else 0,
            "after_max": max(after_lengths) if after_lengths else 0,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
