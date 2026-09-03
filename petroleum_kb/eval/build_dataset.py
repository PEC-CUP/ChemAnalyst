from __future__ import annotations

"""
Build an evaluation dataset (questions + gold_doc_ids) from the ingested KB.

Design goals:
- doc_id is stable across re-indexing (prefer DOI, then title+year, then file_name+source).
- questions are "paper-level evidence-driven": derived from multiple chunks grouped by paper.
- supports single-doc and two-document comparison questions.

Typical workflow:
1) Ingest docs + build index
2) Run this script to generate a draft dataset
3) Manually review/edit a subset (recommended) and freeze it as your benchmark

Example:
  python -B petroleum_kb/eval/build_dataset.py ^
    --chunks petroleum_kb/petroleum_kb/data/processed/chunks.jsonl ^
    --out petroleum_kb/eval/datasets/questions.auto.jsonl ^
    --num 50 ^
    --mode mixed
"""

import argparse
import math
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dataset_generation_utils import (
    _has_vague_paper_reference,
    build_doc_evidence,
    generate_doc_level_sample,
    generate_naive_doc_level_sample,
    review_and_rewrite_question,
)

NEAR_DUPLICATE_QUESTION_THRESHOLD = 0.88
MAX_ENGLISH_ANSWER_CJK_RATIO = 0.01

TWO_DOC_QUESTION_VARIANTS = [
    "method-selection contrast: ask when one named analytical or modeling approach is more suitable than the other, based on the evidence.",
    "mechanistic interpretation: ask how named methods, transformations, or compositional drivers differ, without referring to source-paper identity.",
    "evidence-to-conclusion contrast: ask how each paper moves from observed evidence to its main scientific conclusion.",
    "sample-matrix boundary: ask how matrix or fraction type changes what can be measured, modeled, or inferred.",
    "workflow trade-off: ask about trade-offs in resolution, specificity, throughput, assumptions, or required sample preparation.",
    "composition-dimension synthesis: ask how named compositional dimensions such as heteroatoms, hydrocarbons, fractions, or molecular classes are characterized.",
    "model-versus-measurement contrast: ask how computational reconstruction/modeling differs from direct analytical measurement or separation.",
    "limitation-and-use-case synthesis: ask what limitations in one approach are addressed or complemented by the other approach.",
]


def _two_doc_question_variant(target_index: int, attempt_index: int) -> str:
    return TWO_DOC_QUESTION_VARIANTS[(int(target_index) + int(attempt_index)) % len(TWO_DOC_QUESTION_VARIANTS)]


def _normalize_benchmark_style(raw: str) -> str:
    value = str(raw or "").strip().lower()
    aliases = {
        "semantic": "semantic",
        "semantic": "semantic",
        "specific": "specific_fact",
        "specific_fact": "specific_fact",
        "fact": "specific_fact",
        "fact_lookup": "specific_fact",
        "specific_fact": "specific_fact",
    }
    if value not in aliases:
        raise SystemExit(f"Unsupported --benchmark-style={raw!r}. Use semantic or specific_fact.")
    return aliases[value]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _doc_key(md: dict[str, Any]) -> str:
    doi = (md.get("doi") or "").strip()
    if doi:
        return f"doi:{doi}"
    title = (md.get("title") or "").strip()
    if title:
        year = md.get("year") or ""
        return f"title:{title}::{year}".strip()
    file_name = (md.get("file_name") or "").strip()
    source = (md.get("source") or "").strip()
    return f"file:{file_name}::{source}".strip()


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


@dataclass(frozen=True)
class DocChunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocPairCandidate:
    left: str
    right: str
    score: float


def load_chunks(path: Path) -> list[DocChunk]:
    out: list[DocChunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            md = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            doc_id = _doc_key(md)
            chunk_id = str(item.get("chunk_id") or md.get("chunk_id") or "")
            out.append(DocChunk(doc_id=doc_id, chunk_id=chunk_id, text=text, metadata=md))
    return out


def group_by_doc(chunks: list[DocChunk]) -> dict[str, list[DocChunk]]:
    groups: dict[str, list[DocChunk]] = {}
    for ch in chunks:
        groups.setdefault(ch.doc_id, []).append(ch)
    return groups


def _normalize_doc_count_ratios(raw: str | None, *, two_doc_ratio: float) -> dict[int, float]:
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --doc-count-ratios JSON: {exc}") from exc
        ratios = {int(key): max(float(value), 0.0) for key, value in payload.items()}
    else:
        ratios = {1: max(1.0 - float(two_doc_ratio), 0.0), 2: max(float(two_doc_ratio), 0.0)}
    ratios = {count: value for count, value in ratios.items() if count in {1, 2} and value > 0}
    total = sum(ratios.values())
    if total <= 0:
        raise SystemExit("At least one positive doc-count ratio for 1 or 2 documents is required.")
    return {count: value / total for count, value in ratios.items()}


def _existing_question_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _normalize_qa_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = " ".join(text.split())
    return text


def _cjk_char_ratio(text: str) -> float:
    chars = [ch for ch in str(text or "") if not ch.isspace()]
    if not chars:
        return 0.0
    cjk_count = sum(1 for ch in chars if "\u4e00" <= ch <= "\u9fff")
    return cjk_count / len(chars)


def _contains_cjk(text: Any) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _language_gate_passes(row: dict[str, Any], expected_language: str) -> bool:
    if expected_language != "en":
        return True
    question = str(row.get("question") or "")
    answer_key = str(row.get("answer_key") or "")
    key_points = " ".join(str(item) for item in row.get("key_points") or [])
    if _contains_cjk(question):
        return False
    return (
        _cjk_char_ratio(answer_key) <= MAX_ENGLISH_ANSWER_CJK_RATIO
        and _cjk_char_ratio(key_points) <= MAX_ENGLISH_ANSWER_CJK_RATIO
    )


def _retrievable_question_gate_passes(row: dict[str, Any]) -> bool:
    question = str(row.get("question") or "")
    return bool(question.strip()) and not _has_vague_paper_reference(question)


def _parse_mcq_choice(value: Any) -> tuple[str, str] | None:
    if isinstance(value, dict):
        label = str(value.get("label") or value.get("option") or value.get("id") or "").strip().upper()
        text = str(value.get("text") or value.get("choice") or value.get("answer") or "").strip()
        if not text:
            parsed = _parse_mcq_choice(str(value.get("value") or "").strip())
            if parsed:
                return parsed
        if label in {"A", "B", "C", "D"} and text:
            return label, " ".join(text.split())
        return None
    text = str(value or "").strip()
    match = re.match(r"^\s*([A-D])\s*[\).:]\s*(.+top)\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).upper(), " ".join(match.group(2).strip().split())


def _extract_mcq_options_from_question(question: str) -> dict[str, str]:
    text = str(question or "").strip()
    if not text:
        return {}
    matches = list(re.finditer(r"(topis)(top:^|[\n\r\s]|\\n|\\r)([A-D])[\).:]\s+", text))
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = " ".join(text[start:end].strip().split())
        if label in {"A", "B", "C", "D"} and value:
            out[label] = value
    return out


def _strip_mcq_options_from_question(question: str) -> str:
    text = str(question or "").strip()
    match = re.search(r"(topis)(top:^|[\n\r\s]|\\n|\\r)A[\).:]\s+", text)
    if not match:
        return text
    return text[: match.start()].strip()


def _normalize_mcq_text(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"^\s*[a-d]\s*[\).:]\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .;")


def _normalize_mcq_key_points(key_points: Any) -> list[str]:
    if not isinstance(key_points, list):
        return []
    out: list[str] = []
    for item in key_points:
        text = str(item or "").strip()
        if not text:
            continue
        if re.search(r"\b(correct\s+option|option\s+[A-D]|answer\s+is\s+[A-D])\b", text, flags=re.IGNORECASE):
            continue
        out.append(text)
    return out


def _infer_mcq_correct_option(row: dict[str, Any], choices_by_label: dict[str, str]) -> str | None:
    correct_option = str(row.get("correct_option") or "").strip().upper()
    answer_key = str(row.get("answer_key") or "").strip()
    answer_match = re.match(r"^\s*([A-D])\s*[\).:]\s*(.*)$", answer_key, flags=re.IGNORECASE | re.DOTALL)
    answer_option = answer_match.group(1).upper() if answer_match else ""
    answer_text = answer_match.group(2).strip() if answer_match else answer_key

    if correct_option in choices_by_label:
        if answer_option and answer_option != correct_option:
            return None
        if answer_text:
            answer_norm = _normalize_mcq_text(answer_text)
            choice_norm = _normalize_mcq_text(choices_by_label[correct_option])
            if answer_norm and answer_norm not in choice_norm and choice_norm not in answer_norm:
                return None
        return correct_option
    if answer_option in choices_by_label:
        return answer_option

    answer_norm = _normalize_mcq_text(answer_text)
    if answer_norm:
        matches = [
            label
            for label, choice_text in choices_by_label.items()
            if answer_norm == _normalize_mcq_text(choice_text)
            or answer_norm in _normalize_mcq_text(choice_text)
            or _normalize_mcq_text(choice_text) in answer_norm
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _least_used_mcq_option(counts: Counter, rng: random.Random) -> str:
    min_count = min(int(counts.get(label, 0)) for label in "ABCD")
    candidates = [label for label in "ABCD" if int(counts.get(label, 0)) == min_count]
    return rng.choice(candidates)


def _normalize_mcq_row(
    row: dict[str, Any],
    *,
    target_correct_option: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    if str(row.get("answer_format") or "").strip().lower() != "mcq":
        normalized = dict(row)
        normalized["answer_format"] = "open"
        normalized["choices"] = []
        normalized["correct_option"] = ""
        return normalized

    raw_choices = row.get("choices") if isinstance(row.get("choices"), list) else []
    parsed_choices = [_parse_mcq_choice(choice) for choice in raw_choices]
    if len(parsed_choices) != 4 or any(choice is None for choice in parsed_choices):
        return None
    choices_by_label = {label: text for label, text in parsed_choices if label and text}  # type: ignore[misc]
    if set(choices_by_label) != {"A", "B", "C", "D"}:
        return None
    normalized_choice_texts = [_normalize_mcq_text(text) for text in choices_by_label.values()]
    if any(not text for text in normalized_choice_texts) or len(set(normalized_choice_texts)) != 4:
        return None

    original_correct = _infer_mcq_correct_option(row, choices_by_label)
    if original_correct not in {"A", "B", "C", "D"}:
        return None
    correct_text = choices_by_label[original_correct]
    target = str(target_correct_option or "").strip().upper()
    if target not in {"A", "B", "C", "D"}:
        target = original_correct

    distractors = [text for label, text in choices_by_label.items() if label != original_correct]
    rng.shuffle(distractors)
    remaining_labels = [label for label in "ABCD" if label != target]
    relabeled: dict[str, str] = {target: correct_text}
    for label, text in zip(remaining_labels, distractors):
        relabeled[label] = text

    stem = _strip_mcq_options_from_question(str(row.get("question") or ""))
    if not stem:
        return None
    normalized_choices = [f"{label}. {relabeled[label]}" for label in "ABCD"]
    normalized_row = dict(row)
    normalized_row["question"] = stem.rstrip() + "\n" + "\n".join(normalized_choices)
    normalized_row["answer_format"] = "mcq"
    normalized_row["choices"] = normalized_choices
    normalized_row["correct_option"] = target
    normalized_row["answer_key"] = f"{target}. {correct_text}"
    normalized_row["key_points"] = _normalize_mcq_key_points(row.get("key_points"))
    return normalized_row


def _mcq_format_gate_passes(row: dict[str, Any]) -> bool:
    if str(row.get("answer_format") or "").strip().lower() != "mcq":
        return True
    question = str(row.get("question") or "")
    answer_key = str(row.get("answer_key") or "")
    choices = row.get("choices") if isinstance(row.get("choices"), list) else []
    correct_option = str(row.get("correct_option") or "").strip().upper()
    parsed_choices = [_parse_mcq_choice(choice) for choice in choices]
    if len(parsed_choices) != 4 or any(choice is None for choice in parsed_choices):
        return False
    choices_by_label = {label: text for label, text in parsed_choices if label and text}  # type: ignore[misc]
    if set(choices_by_label) != {"A", "B", "C", "D"}:
        return False
    if correct_option not in {"A", "B", "C", "D"}:
        return False
    question_options = _extract_mcq_options_from_question(question)
    if set(question_options) != {"A", "B", "C", "D"}:
        return False
    stem = _strip_mcq_options_from_question(question)
    if re.search(r"(topis)(top:^|[\n\r\s]|\\n|\\r)[A-D][\).:]\s+", stem):
        return False
    for label in "ABCD":
        if _normalize_mcq_text(question_options[label]) != _normalize_mcq_text(choices_by_label[label]):
            return False
    if len({_normalize_mcq_text(choices_by_label[label]) for label in "ABCD"}) != 4:
        return False
    expected_answer_key = f"{correct_option}. {choices_by_label[correct_option]}"
    if _normalize_mcq_text(answer_key) != _normalize_mcq_text(expected_answer_key):
        return False
    if not re.match(rf"^\s*{correct_option}\s*[\).:]\s+", answer_key, flags=re.IGNORECASE):
        return False
    key_points = row.get("key_points") if isinstance(row.get("key_points"), list) else []
    if not key_points:
        return False
    for key_point in key_points:
        if re.search(r"\b(correct\s+option|option\s+[A-D]|answer\s+is\s+[A-D])\b", str(key_point), flags=re.IGNORECASE):
            return False
    return True


def _rewrite_question_for_retrieval(
    *,
    qa: dict[str, Any],
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any] | None,
    max_answer_chars: int,
) -> dict[str, Any]:
    if not _has_vague_paper_reference(qa.get("question")):
        return qa
    if evidence_b is None:
        return qa
    schema = (
        "{\n"
        '  "question": "...",\n'
        '  "answer_key": "...",\n'
        '  "key_points": ["..."],\n'
        '  "tags": ["..."],\n'
        '  "notes": ""\n'
        "}\n"
    )
    prompt = (
        "Rewrite this benchmark QA item so the question is self-contained and retrievable by RAG.\n"
        "Keep the same scientific intent and answer facts, but remove hidden source references.\n"
        "The rewritten question must NOT contain: the two papers, both papers, Paper A, Paper B, "
        "the two studies, both studies, the two approaches, both approaches, the two methods, both methods, "
        "provided evidence, or publication-year paper anchors.\n"
        "Instead, explicitly name the methods, sample/fraction types, compound classes, ionization modes, "
        "or petroleum phenomena from the evidence.\n"
        f"Keep answer_key within {int(max_answer_chars)} characters.\n"
        "Return JSON only using this schema:\n"
        f"{schema}\n"
        f"ORIGINAL_QUESTION: {qa.get('question', '')}\n"
        f"ORIGINAL_ANSWER_KEY: {qa.get('answer_key', '')}\n"
        f"ORIGINAL_KEY_POINTS: {qa.get('key_points', [])}\n\n"
        f"EVIDENCE_A:\n{evidence_a['text']}\n\n"
        f"EVIDENCE_B:\n{evidence_b['text']}\n"
    )
    raw = _call_deepseek(
        prompt=prompt,
        temperature=0.0,
        system_prompt="You rewrite benchmark questions into explicit, retrievable petroleum-science questions. Output JSON only.",
    )
    rewritten = _extract_json(raw)
    if not isinstance(rewritten, dict):
        return qa
    candidate = {
        "question": str(rewritten.get("question") or qa.get("question") or "").strip(),
        "answer_key": str(rewritten.get("answer_key") or qa.get("answer_key") or "").strip(),
        "key_points": rewritten.get("key_points") if isinstance(rewritten.get("key_points"), list) else list(qa.get("key_points") or []),
        "tags": rewritten.get("tags") if isinstance(rewritten.get("tags"), list) else list(qa.get("tags") or []),
        "notes": str(rewritten.get("notes") or qa.get("notes") or "").strip(),
        "answer_format": str(qa.get("answer_format") or "open").strip().lower(),
        "choices": list(qa.get("choices") or []) if isinstance(qa.get("choices"), list) else [],
        "correct_option": str(qa.get("correct_option") or "").strip().upper(),
    }
    return candidate if not _has_vague_paper_reference(candidate.get("question")) else qa


def _qa_signature(row: dict[str, Any]) -> tuple[str, str]:
    return (_normalize_qa_text(row.get("question")), _normalize_qa_text(row.get("answer_key")))


def _question_signature(row: dict[str, Any]) -> str:
    return _normalize_qa_text(row.get("question"))


def _question_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _is_near_duplicate_question(
    question_signature: str,
    seen_questions: set[str],
    *,
    threshold: float = NEAR_DUPLICATE_QUESTION_THRESHOLD,
) -> bool:
    if not question_signature:
        return False
    return any(_question_similarity(question_signature, existing) >= float(threshold) for existing in seen_questions)


def _load_existing_qa_signatures(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    signatures: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                signature = _qa_signature(item)
                if all(signature):
                    signatures.add(signature)
    return signatures


def _load_existing_question_signatures(path: Path) -> set[str]:
    if not path.exists():
        return set()
    signatures: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                signature = _question_signature(item)
                if signature:
                    signatures.add(signature)
    return signatures


def _load_existing_mcq_correct_option_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if str(item.get("answer_format") or "").strip().lower() != "mcq":
                continue
            option = str(item.get("correct_option") or "").strip().upper()
            if option in {"A", "B", "C", "D"}:
                counts[option] += 1
    return counts


def _target_doc_counts(total: int, ratios: dict[int, float]) -> dict[int, int]:
    raw_targets = {doc_count: max(float(ratios.get(doc_count, 0.0)), 0.0) * int(total) for doc_count in (1, 2)}
    targets = {doc_count: int(raw_targets[doc_count]) for doc_count in (1, 2)}
    remainder = int(total) - sum(targets.values())
    fractional_order = sorted(
        raw_targets,
        key=lambda doc_count: (raw_targets[doc_count] - targets[doc_count], raw_targets[doc_count]),
        reverse=True,
    )
    for index in range(max(remainder, 0)):
        targets[fractional_order[index % len(fractional_order)]] += 1
    return {doc_count: count for doc_count, count in targets.items() if count > 0}


def _planned_doc_counts(*, mode: str, total: int, ratios: dict[int, float]) -> list[int]:
    if mode == "single":
        targets = {1: int(total)}
    elif mode == "two_doc":
        targets = {2: int(total)}
    else:
        targets = _target_doc_counts(int(total), ratios)
    # Keep the generation order deterministic and auditable: all single-document
    # questions first, then all two-document questions. This prevents ratio drift
    # caused by random sampling and makes progress reporting match the requested
    # benchmark composition.
    return [1] * int(targets.get(1, 0)) + [2] * int(targets.get(2, 0))


def _counts_from_plan(plan: list[int]) -> dict[int, int]:
    return {doc_count: plan.count(doc_count) for doc_count in (1, 2) if plan.count(doc_count) > 0}


def _select_single_doc(
    *,
    doc_ids: list[str],
    used_docs: set[str],
    rng: random.Random,
) -> str:
    available = [doc_id for doc_id in doc_ids if doc_id not in used_docs]
    if not available:
        used_docs.clear()
        available = list(doc_ids)
    doc_id = rng.choice(available)
    used_docs.add(doc_id)
    return doc_id


PROFILE_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "any",
    "analysis",
    "and",
    "are",
    "article",
    "arising",
    "based",
    "been",
    "between",
    "both",
    "can",
    "content",
    "data",
    "designation",
    "different",
    "during",
    "d02",
    "each",
    "effect",
    "effects",
    "francis",
    "for",
    "from",
    "have",
    "high",
    "indicates",
    "into",
    "last",
    "method",
    "methods",
    "more",
    "not",
    "number",
    "oil",
    "one",
    "only",
    "opinions",
    "other",
    "our",
    "paper",
    "properties",
    "provide",
    "reapproval",
    "result",
    "results",
    "revision",
    "sample",
    "samples",
    "show",
    "shown",
    "study",
    "studies",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "taylor",
    "using",
    "value",
    "values",
    "views",
    "were",
    "with",
    "within",
    "year",
}

PROFILE_STOPWORDS.update(
    {
        "accuracy",
        "change",
        "completeness",
        "damages",
        "direct",
        "editorial",
        "epsilon",
        "expenses",
        "following",
        "however",
        "immediately",
        "information",
        "issued",
        "jurisdiction",
        "lubricants",
        "parentheses",
        "research",
        "responsibility",
        "shall",
        "should",
        "standard",
        "superscript",
        "under",
        "use",
        "warranties",
        "whatsoever",
        "dans",
        "des",
        "les",
        "par",
        "troli",
        "une",
    }
)

PROFILE_DOMAIN_WORDS = {
    "acid",
    "acids",
    "alkane",
    "alkanes",
    "aromatic",
    "aromatics",
    "asphaltene",
    "asphaltenes",
    "biomarker",
    "biomarkers",
    "bitumen",
    "carbon",
    "catalytic",
    "characterization",
    "chemical",
    "chromatography",
    "coke",
    "coal",
    "composition",
    "compositional",
    "compound",
    "compounds",
    "cracking",
    "crude",
    "desorption",
    "diesel",
    "distillation",
    "electrospray",
    "elemental",
    "fims",
    "fuel",
    "fuels",
    "gas",
    "gasoil",
    "heavy",
    "hydrocarbon",
    "hydrocarbons",
    "hydrogenation",
    "hydrotreating",
    "ion",
    "ionization",
    "ions",
    "mass",
    "mesophase",
    "molecular",
    "naphthenes",
    "naphthenic",
    "nitrogen",
    "olefins",
    "paraffins",
    "petroleum",
    "petroleomics",
    "photoionization",
    "pitch",
    "pyrolysis",
    "reconstruction",
    "residue",
    "resins",
    "sara",
    "saturates",
    "separation",
    "shale",
    "spectra",
    "spectrometry",
    "sulfur",
    "tar",
    "vacuum",
    "wax",
}

PROFILE_DOMAIN_ACRONYMS = {
    "api",
    "appi",
    "apci",
    "ccs",
    "dbe",
    "dom",
    "ei",
    "esi",
    "fcc",
    "fid",
    "ftir",
    "ft-icr",
    "gc",
    "gc-fid",
    "gc-ms",
    "gcxgc",
    "hds",
    "hrms",
    "ldi",
    "maldi",
    "ms",
    "nmr",
    "pah",
    "pin",
    "sara",
    "tofms",
    "petroleum fraction",
}

PROFILE_GENERIC_DOMAIN_WORDS = {
    "carbon",
    "characterization",
    "chemical",
    "composition",
    "compositional",
    "compound",
    "compounds",
    "gas",
    "heavy",
    "ion",
    "ions",
    "mass",
    "molecular",
}

PROFILE_GENERIC_DOMAIN_ACRONYMS = {
    "api",
    "gc",
    "ms",
}

PROFILE_BENCHMARK_TOPIC_WORDS = {
    "alkane",
    "alkanes",
    "aromatic",
    "aromatics",
    "asphaltene",
    "asphaltenes",
    "biomarker",
    "biomarkers",
    "bitumen",
    "catalytic",
    "coke",
    "coal",
    "cracking",
    "crude",
    "diesel",
    "distillation",
    "fcc",
    "fuel",
    "fuels",
    "gasoil",
    "hydrocarbon",
    "hydrocarbons",
    "hydrogenation",
    "hydrotreating",
    "mesophase",
    "naphthenes",
    "naphthenic",
    "olefins",
    "paraffins",
    "petroleum",
    "petroleomics",
    "pitch",
    "pyrolysis",
    "residue",
    "resins",
    "sara",
    "saturates",
    "shale",
    "sulfur",
    "tar",
    "vacuum",
    "petroleum fraction",
    "wax",
}

PROFILE_PETROLEUM_CORE_WORDS = {
    "acid",
    "acids",
    "alkane",
    "alkanes",
    "aromatic",
    "aromatics",
    "asphaltene",
    "asphaltenes",
    "biomarker",
    "biomarkers",
    "bitumen",
    "catalytic",
    "coke",
    "coal",
    "cracking",
    "crude",
    "diamondoid",
    "diamondoids",
    "diesel",
    "distillation",
    "fcc",
    "fuel",
    "fuels",
    "gasoil",
    "hydrocarbon",
    "hydrocarbons",
    "hydrogenation",
    "hydroprocessing",
    "hydrotreating",
    "mesophase",
    "naphthenes",
    "naphthenic",
    "olefins",
    "paraffins",
    "petroleum",
    "petroleomics",
    "pitch",
    "pyrolysis",
    "refinery",
    "refining",
    "residue",
    "resins",
    "sara",
    "saturates",
    "shale",
    "sulfur",
    "tar",
    "vacuum",
    "petroleum fraction",
    "wax",
}

PROFILE_PETROLEUM_CORE_ACRONYMS = {
    "fcc",
    "hds",
    "pin",
    "sara",
    "petroleum fraction",
}

PROFILE_PETROLEUM_CORE_PHRASES = {
    "coal tar",
    "crude oil",
    "diesel fuel",
    "fluid catalytic cracking",
    "heavy oil",
    "petroleum products",
    "petroleum residues",
    "shale oil",
    "vacuum residue",
}

PROFILE_PETROLEUM_CORE_MIN_SCORE = 5.0

PROFILE_PETROLEUM_ANCHOR_WORDS = {
    "asphaltene",
    "asphaltenes",
    "biomarker",
    "biomarkers",
    "bitumen",
    "crude",
    "diamondoid",
    "diamondoids",
    "diesel",
    "fcc",
    "gasoil",
    "gasoils",
    "hydrocarbon",
    "hydrocarbons",
    "naphthenes",
    "naphthenic",
    "olefins",
    "paraffins",
    "petroleum",
    "petroleomics",
    "residue",
    "resins",
    "sara",
    "saturates",
    "shale",
    "petroleum fraction",
}

PROFILE_PETROLEUM_ANCHOR_PHRASES = {
    "coal tar",
    "crude oil",
    "diesel fuel",
    "fluid catalytic cracking",
    "heavy oil",
    "petroleum products",
    "petroleum residues",
    "shale oil",
    "vacuum residue",
}


def _normalize_profile_term(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("×", "x")
    text = text.replace("×", "x")
    text = re.sub(r"[^a-z0-9+\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _has_profile_domain_signal(term: str) -> bool:
    parts = [part for part in re.split(r"[\s\-/]+", term) if part]
    return any(
        (part in PROFILE_DOMAIN_WORDS and part not in PROFILE_GENERIC_DOMAIN_WORDS)
        or (part in PROFILE_DOMAIN_ACRONYMS and part not in PROFILE_GENERIC_DOMAIN_ACRONYMS)
        for part in parts
    )


def _has_benchmark_topic_signal(chunks: list[DocChunk]) -> bool:
    metadata = chunks[0].metadata if chunks else {}
    values = [
        str(metadata.get("title") or ""),
        str(metadata.get("abstract") or ""),
        str(metadata.get("journal") or ""),
        " ".join(chunk.text[:1000] for chunk in chunks[:4]),
    ]
    text = _normalize_profile_term(" ".join(values))
    parts = set(part for part in re.split(r"[\s\-/]+", text) if part)
    return bool(parts & PROFILE_BENCHMARK_TOPIC_WORDS)


def _petroleum_core_signal_score(chunks: list[DocChunk]) -> float:
    """Score whether a paper belongs to the petroleum KB benchmark scope."""

    metadata = chunks[0].metadata if chunks else {}
    lead_values = [
        str(metadata.get("title") or ""),
        str(metadata.get("abstract") or ""),
        str(metadata.get("journal") or ""),
    ]
    body_values = [chunk.text[:900] for chunk in chunks[:4]]

    lead_text = _normalize_profile_term(" ".join(lead_values))
    body_text = _normalize_profile_term(" ".join(body_values))

    def count_hits(text: str) -> int:
        parts = set(part for part in re.split(r"[\s\-/]+", text) if part)
        return len(parts & PROFILE_PETROLEUM_CORE_WORDS) + len(parts & PROFILE_PETROLEUM_CORE_ACRONYMS)

    score = 0.0
    score += 3.0 * count_hits(lead_text)
    score += 0.75 * count_hits(body_text)
    combined = f"{lead_text} {body_text}"
    for phrase in PROFILE_PETROLEUM_CORE_PHRASES:
        if phrase in combined:
            score += 4.0
    return score


def _has_petroleum_anchor_signal(chunks: list[DocChunk]) -> bool:
    metadata = chunks[0].metadata if chunks else {}
    values = [
        str(metadata.get("title") or ""),
        str(metadata.get("abstract") or ""),
        str(metadata.get("journal") or ""),
        " ".join(chunk.text[:900] for chunk in chunks[:4]),
    ]
    text = _normalize_profile_term(" ".join(values))
    parts = set(part for part in re.split(r"[\s\-/]+", text) if part)
    if parts & PROFILE_PETROLEUM_ANCHOR_WORDS:
        return True
    return any(phrase in text for phrase in PROFILE_PETROLEUM_ANCHOR_PHRASES)


def _two_doc_pairable_doc_ids(groups: dict[str, list[DocChunk]], doc_ids: list[str]) -> list[str]:
    return [
        doc_id
        for doc_id in doc_ids
        if _has_benchmark_topic_signal(groups[doc_id])
        and _has_petroleum_anchor_signal(groups[doc_id])
        and _petroleum_core_signal_score(groups[doc_id]) >= PROFILE_PETROLEUM_CORE_MIN_SCORE
    ]


def _profile_tokens(text: str) -> list[str]:
    terms: list[str] = []
    words = [
        token
        for token in re.findall(r"[a-z][a-z0-9+\-/]{2,}", _normalize_profile_term(text))
        if token not in PROFILE_STOPWORDS and _has_profile_domain_signal(token)
    ]
    terms.extend(words)
    for size in (2, 3):
        for index in range(0, max(len(words) - size + 1, 0)):
            gram = " ".join(words[index : index + size])
            if gram and _has_profile_domain_signal(gram):
                terms.append(gram)
    return terms


def _doc_profile(doc_id: str, chunks: list[DocChunk]) -> Counter[str]:
    """Build a lightweight topical profile for pairing related papers."""

    profile: Counter[str] = Counter()
    metadata = chunks[0].metadata if chunks else {}
    lead_text = " ".join(
        str(metadata.get(key) or "")
        for key in ("title", "abstract", "journal")
        if str(metadata.get(key) or "").strip()
    )
    for term in _profile_tokens(lead_text):
        profile[term] += 4.0

    # Use a small body sample so papers without abstracts still get a usable
    # profile, without letting common boilerplate dominate the pairing.
    body_text = " ".join(chunk.text[:1200] for chunk in chunks[:6])
    for term in _profile_tokens(body_text):
        profile[term] += 0.35

    profile.pop(doc_id.lower(), None)
    return profile


def _doc_lead_profile(chunks: list[DocChunk]) -> Counter[str]:
    profile: Counter[str] = Counter()
    metadata = chunks[0].metadata if chunks else {}
    lead_text = " ".join(
        str(metadata.get(key) or "")
        for key in ("title", "abstract", "journal")
        if str(metadata.get(key) or "").strip()
    )
    for term in _profile_tokens(lead_text):
        profile[term] += 4.0
    return profile


def _doc_title_key(chunks: list[DocChunk]) -> str:
    metadata = chunks[0].metadata if chunks else {}
    title = _normalize_profile_term(metadata.get("title"))
    title = title.replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", re.sub(r"\b(the|a|an)\b", " ", title)).strip()


def _build_doc_pair_candidates(
    *,
    groups: dict[str, list[DocChunk]],
    doc_ids: list[str],
    rng: random.Random,
    max_doc_freq_ratio: float = 0.35,
    min_similarity: float = 0.035,
    max_pairs_per_doc: int = 80,
) -> list[DocPairCandidate]:
    doc_ids = _two_doc_pairable_doc_ids(groups, doc_ids)
    profiles = {doc_id: _doc_profile(doc_id, groups[doc_id]) for doc_id in doc_ids}
    lead_profiles = {doc_id: _doc_lead_profile(groups[doc_id]) for doc_id in doc_ids}
    title_keys = {doc_id: _doc_title_key(groups[doc_id]) for doc_id in doc_ids}
    doc_freq: Counter[str] = Counter()
    for profile in profiles.values():
        for term in profile:
            doc_freq[term] += 1

    doc_count = max(len(doc_ids), 1)
    weighted_profiles: dict[str, dict[str, float]] = {}
    norms: dict[str, float] = {}
    inverted: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for doc_id, profile in profiles.items():
        weighted: dict[str, float] = {}
        for term, tf in profile.items():
            df = doc_freq[term]
            if df < 2 or df / doc_count > float(max_doc_freq_ratio):
                continue
            idf = math.log((doc_count + 1) / (df + 1)) + 1.0
            weight = math.sqrt(float(tf)) * idf
            if weight <= 0:
                continue
            weighted[term] = weight
            inverted[term].append((doc_id, weight))
        weighted_profiles[doc_id] = weighted
        norms[doc_id] = math.sqrt(sum(value * value for value in weighted.values()))

    pair_scores: Counter[tuple[str, str]] = Counter()
    for postings in inverted.values():
        if len(postings) < 2:
            continue
        postings = sorted(postings)
        for i in range(len(postings)):
            left, left_weight = postings[i]
            for j in range(i + 1, len(postings)):
                right, right_weight = postings[j]
                pair_scores[(left, right)] += left_weight * right_weight

    per_doc: dict[str, list[DocPairCandidate]] = defaultdict(list)
    for (left, right), dot in pair_scores.items():
        if title_keys.get(left) and title_keys.get(left) == title_keys.get(right):
            continue
        if (
            title_keys.get(left)
            and title_keys.get(right)
            and _question_similarity(title_keys[left], title_keys[right]) >= 0.96
        ):
            continue
        if not (set(lead_profiles.get(left, {})) & set(lead_profiles.get(right, {}))):
            continue
        denom = norms.get(left, 0.0) * norms.get(right, 0.0)
        if denom <= 0:
            continue
        score = float(dot) / denom
        if score < float(min_similarity):
            continue
        candidate = DocPairCandidate(left=left, right=right, score=score)
        per_doc[left].append(candidate)
        per_doc[right].append(candidate)

    selected: set[tuple[str, str]] = set()
    candidates: list[DocPairCandidate] = []
    for doc_candidates in per_doc.values():
        doc_candidates.sort(key=lambda item: item.score, reverse=True)
        for candidate in doc_candidates[: int(max_pairs_per_doc)]:
            key = tuple(sorted((candidate.left, candidate.right)))
            if key in selected:
                continue
            selected.add(key)
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    # Shuffle equal-ish score neighborhoods so runs with the same seed remain
    # reproducible but do not overuse one narrow topical cluster.
    windowed: list[DocPairCandidate] = []
    bucket: list[DocPairCandidate] = []
    last_bucket: int | None = None
    for candidate in candidates:
        bucket_key = int(candidate.score * 100)
        if last_bucket is not None and bucket_key != last_bucket:
            rng.shuffle(bucket)
            windowed.extend(bucket)
            bucket = []
        bucket.append(candidate)
        last_bucket = bucket_key
    if bucket:
        rng.shuffle(bucket)
        windowed.extend(bucket)
    return windowed


def _load_vector_index_embeddings(index_dir: Path, index_name: str) -> tuple[list[str], Any]:
    ids_path = index_dir / f"{index_name}_ids.json"
    faiss_path = index_dir / f"{index_name}.faiss"
    npy_path = index_dir / f"{index_name}.npy"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing vector index id file: {ids_path}")
    chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(chunk_ids, list):
        raise ValueError(f"Invalid vector index id file: {ids_path}")

    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Semantic two-doc pairing requires numpy.") from exc

    if faiss_path.exists():
        try:
            import faiss  # type: ignore

            index = faiss.read_index(str(faiss_path))
            vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)]).astype("float32")
        except Exception as exc:
            raise RuntimeError(f"Unable to read FAISS vectors from {faiss_path}: {exc}") from exc
    elif npy_path.exists():
        vectors = np.load(npy_path).astype("float32")
    else:
        raise FileNotFoundError(f"Missing vector index file: {faiss_path} or {npy_path}")

    if len(chunk_ids) != int(vectors.shape[0]):
        raise ValueError(
            f"Vector index id/vector count mismatch: ids={len(chunk_ids)} vectors={int(vectors.shape[0])}"
        )
    return [str(chunk_id) for chunk_id in chunk_ids], vectors


def _build_semantic_doc_pair_candidates(
    *,
    groups: dict[str, list[DocChunk]],
    doc_ids: list[str],
    rng: random.Random,
    index_dir: Path,
    index_name: str,
    min_similarity: float = 0.58,
    max_pairs_per_doc: int = 80,
) -> list[DocPairCandidate]:
    """Pair papers with the same chunk embeddings used by RAG retrieval."""

    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Semantic two-doc pairing requires numpy.") from exc

    chunk_ids, vectors = _load_vector_index_embeddings(index_dir, index_name)
    vector_by_chunk_id = {chunk_id: vectors[index] for index, chunk_id in enumerate(chunk_ids)}
    doc_ids = _two_doc_pairable_doc_ids(groups, doc_ids)
    title_keys = {doc_id: _doc_title_key(groups[doc_id]) for doc_id in doc_ids}

    doc_vectors: dict[str, Any] = {}
    for doc_id in doc_ids:
        rows = [vector_by_chunk_id[chunk.chunk_id] for chunk in groups[doc_id] if chunk.chunk_id in vector_by_chunk_id]
        if not rows:
            continue
        matrix = np.vstack(rows).astype("float32")
        vector = matrix.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            continue
        doc_vectors[doc_id] = vector / norm

    ordered_doc_ids = list(doc_vectors)
    if len(ordered_doc_ids) < 2:
        return []

    matrix = np.vstack([doc_vectors[doc_id] for doc_id in ordered_doc_ids]).astype("float32")
    scores = matrix @ matrix.T
    per_doc: dict[str, list[DocPairCandidate]] = defaultdict(list)
    for i, left in enumerate(ordered_doc_ids):
        for j in range(i + 1, len(ordered_doc_ids)):
            right = ordered_doc_ids[j]
            if title_keys.get(left) and title_keys.get(left) == title_keys.get(right):
                continue
            if (
                title_keys.get(left)
                and title_keys.get(right)
                and _question_similarity(title_keys[left], title_keys[right]) >= 0.96
            ):
                continue
            score = float(scores[i, j])
            if score < float(min_similarity):
                continue
            candidate = DocPairCandidate(left=left, right=right, score=score)
            per_doc[left].append(candidate)
            per_doc[right].append(candidate)

    selected: set[tuple[str, str]] = set()
    candidates: list[DocPairCandidate] = []
    for doc_candidates in per_doc.values():
        doc_candidates.sort(key=lambda item: item.score, reverse=True)
        for candidate in doc_candidates[: int(max_pairs_per_doc)]:
            key = tuple(sorted((candidate.left, candidate.right)))
            if key in selected:
                continue
            selected.add(key)
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    windowed: list[DocPairCandidate] = []
    bucket: list[DocPairCandidate] = []
    last_bucket: int | None = None
    for candidate in candidates:
        bucket_key = int(candidate.score * 100)
        if last_bucket is not None and bucket_key != last_bucket:
            rng.shuffle(bucket)
            windowed.extend(bucket)
            bucket = []
        bucket.append(candidate)
        last_bucket = bucket_key
    if bucket:
        rng.shuffle(bucket)
        windowed.extend(bucket)
    return windowed


def _pair_focus_terms(
    *,
    left_chunks: list[DocChunk],
    right_chunks: list[DocChunk],
    max_terms: int = 30,
) -> list[str]:
    left = _doc_lead_profile(left_chunks)
    right = _doc_lead_profile(right_chunks)
    overlap = []
    for term in set(left) & set(right):
        if len(term) < 3 or term in PROFILE_STOPWORDS:
            continue
        overlap.append((term, min(left[term], right[term]) + 0.15 * (left[term] + right[term])))
    overlap.sort(key=lambda item: item[1], reverse=True)
    terms = [term for term, _ in overlap[:max_terms]]
    if terms:
        return terms

    # If lead metadata does not overlap, use the strongest terms from both
    # papers so evidence selection still favors the pair's topical center.
    merged = left + right
    return [term for term, _ in merged.most_common(max_terms) if term not in PROFILE_STOPWORDS]


def _chunk_index(chunk: DocChunk) -> int:
    raw = chunk.metadata.get("chunk_id")
    try:
        return int(raw)
    except Exception:
        return 10**9


def _focused_doc_evidence(
    *,
    doc_id: str,
    chunks: list[DocChunk],
    focus_terms: list[str],
    max_chars: int,
    max_chunks: int,
    min_chunk_chars: int,
) -> dict[str, Any]:
    eligible = [chunk for chunk in chunks if len(str(chunk.text or "").strip()) >= int(min_chunk_chars)]
    if not eligible:
        return {"doc_id": doc_id, "text": "", "chunk_ids": [], "metadata": {}}

    normalized_terms = [_normalize_profile_term(term) for term in focus_terms if _normalize_profile_term(term)]
    scored: list[tuple[float, int, DocChunk]] = []
    for chunk in eligible:
        text = _normalize_profile_term(chunk.text)
        score = 0.0
        for term in normalized_terms:
            if not term:
                continue
            if " " in term:
                if term in text:
                    score += 4.0
            else:
                score += min(text.count(term), 3) * 1.0
        # Prefer informative chunks, but keep position as a deterministic tie-break.
        score += min(len(chunk.text) / 2500.0, 1.0)
        scored.append((score, _chunk_index(chunk), chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [chunk for score, _, chunk in scored[: max(1, int(max_chunks))] if score > 0]
    if not selected:
        selected = [chunk for _, _, chunk in sorted(scored, key=lambda item: item[1])[: max(1, int(max_chunks))]]
    selected.sort(key=_chunk_index)

    meta = selected[0].metadata if selected else {}
    title = str(meta.get("title") or "").strip()
    year = str(meta.get("year") or "").strip()
    journal = str(meta.get("journal") or "").strip()
    focus = "; ".join(focus_terms[:10])
    header_bits = [x for x in (title, year, journal) if x]
    header = f"Paper: {' | '.join(header_bits)}" if header_bits else f"Paper: {doc_id}"
    if focus:
        header += f"\nPair focus terms: {focus}"

    parts = [header]
    chunk_ids: list[str] = []
    total_chars = len(header)
    for i, chunk in enumerate(selected, start=1):
        chunk_id = str(chunk.chunk_id or chunk.metadata.get("chunk_id", i))
        text = str(chunk.text or "").strip()
        prefix = f"[Segment {i} | chunk_id={chunk_id}]\n"
        block = f"{prefix}{text}"
        remaining = int(max_chars) - total_chars - 2
        if len(block) > remaining:
            suffix = "\n[truncated]"
            available_text_chars = remaining - len(prefix) - len(suffix)
            if available_text_chars <= 80:
                break
            block = f"{prefix}{text[:available_text_chars].rstrip()}{suffix}"
            parts.append(block)
            chunk_ids.append(chunk_id)
            break
        parts.append(block)
        total_chars += len(block) + 2
        chunk_ids.append(chunk_id)

    return {
        "doc_id": doc_id,
        "text": "\n\n".join(parts),
        "chunk_ids": chunk_ids,
        "metadata": meta,
    }


def _select_doc_pair(
    *,
    doc_ids: list[str],
    used_pairs: set[tuple[str, str]],
    rng: random.Random,
    candidates: list[DocPairCandidate] | None = None,
) -> tuple[str, str]:
    if candidates:
        available = [
            candidate
            for candidate in candidates
            if tuple(sorted((candidate.left, candidate.right))) not in used_pairs
        ]
        if available:
            window_size = min(len(available), max(25, int(len(available) * 0.08)))
            candidate = rng.choice(available[:window_size])
            pair = tuple(sorted((candidate.left, candidate.right)))
            used_pairs.add(pair)
            return pair

    pairs = [(doc_ids[i], doc_ids[j]) for i in range(len(doc_ids)) for j in range(i + 1, len(doc_ids))]
    available = [pair for pair in pairs if pair not in used_pairs]
    if not available:
        used_pairs.clear()
        available = pairs
    pair = rng.choice(available)
    used_pairs.add(pair)
    return pair


def _quality_block(review: dict[str, Any]) -> dict[str, Any]:
    block = {
        "pass": bool(review.get("pass", False)),
        "self_contained_score": review["self_contained_score"],
        "answerable_score": review["answerable_score"],
        "objective_score": review["objective_score"],
        "specificity_score": review.get("specificity_score"),
        "ambiguity_score": review.get("ambiguity_score"),
        "evidence_locality_score": review.get("evidence_locality_score"),
        "reasoning_type": review.get("reasoning_type"),
        "difficulty": review.get("difficulty"),
        "hallucination_risk_score": review.get("hallucination_risk_score"),
        "quality_score": review.get("quality_score"),
        "evidence_supports": review.get("evidence_supports") if isinstance(review.get("evidence_supports"), list) else [],
        "issues": review["issues"],
        "notes": review["notes"],
    }
    if review.get("two_doc_required_score") is not None:
        block["two_doc_required_score"] = review.get("two_doc_required_score")
    if review.get("cross_doc_synthesis_score") is not None:
        block["cross_doc_synthesis_score"] = review.get("cross_doc_synthesis_score")
    if review.get("doc_balance_score") is not None:
        block["doc_balance_score"] = review.get("doc_balance_score")
    return block


def _reviewed_score_gate_passes(review: dict[str, Any], *, question_type: str, min_quality: float) -> bool:
    """Use judge scores as the reviewed gate instead of relying only on a brittle bool."""

    if bool(review.get("pass", False)):
        return True
    quality_score = float(review.get("quality_score") or 0.0)
    if quality_score < float(min_quality):
        return False
    if float(review.get("self_contained_score") or 0.0) < 0.75:
        return False
    if float(review.get("answerable_score") or 0.0) < 0.75:
        return False
    if float(review.get("objective_score") or 0.0) < 0.75:
        return False
    if float(review.get("specificity_score") or 0.0) < 0.70:
        return False
    if float(review.get("evidence_locality_score") or 0.0) < 0.65:
        return False
    if float(review.get("ambiguity_score") or 1.0) > 0.35:
        return False
    if float(review.get("hallucination_risk_score") or 1.0) > 0.40:
        return False
    if question_type == "two_doc":
        if float(review.get("two_doc_required_score") or 0.0) < 0.75:
            return False
        if float(review.get("cross_doc_synthesis_score") or 0.0) < 0.65:
            return False
        if float(review.get("doc_balance_score") or 0.0) < 0.65:
            return False
    severe_markers = (
        "language_mismatch",
        "unsupported_answer_claim",
        "vague_paper_reference_subject",
        "hidden_source_reference",
        "two_doc_requirement_not_satisfied",
    )
    issues = " ".join(str(item).lower() for item in review.get("issues", []) if str(item).strip())
    return not any(marker in issues for marker in severe_markers)


def _review_rewritten_candidate_if_needed(
    *,
    review: dict[str, Any],
    question_type: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any] | None,
    benchmark_style: str,
    min_quality: float,
) -> dict[str, Any]:
    if _reviewed_score_gate_passes(review, question_type=question_type, min_quality=min_quality):
        return review
    rewritten = review.get("rewritten") if isinstance(review.get("rewritten"), dict) else {}
    if not str(rewritten.get("question") or "").strip() or not str(rewritten.get("answer_key") or "").strip():
        return review
    second = review_and_rewrite_question(
        llm_call=_call_deepseek_reasoner,
        question_type=question_type,
        qa=rewritten,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        benchmark_style=benchmark_style,
        temperature=0.0,
    )
    if _reviewed_score_gate_passes(second, question_type=question_type, min_quality=min_quality):
        second["rewrite_retry_used"] = True
        return second
    if float(second.get("quality_score") or 0.0) > float(review.get("quality_score") or 0.0):
        second["rewrite_retry_used"] = True
        return second
    return review


def _benchmark_mode_label(mode: str) -> str:
    value = str(mode or "").strip().lower()
    if value in {"reviewed", "naive_baseline"}:
        return value
    return "reviewed"


def _call_deepseek_model(*, prompt: str, temperature: float, system_prompt: str, model_name: str) -> str:
    _ensure_repo_root_on_path()
    from openai import OpenAI  # type: ignore

    from app.config import get_settings  # type: ignore

    settings = get_settings()
    client = OpenAI(
        api_key=settings.deepseek_api_key or "MISSING_API_KEY",
        base_url=settings.deepseek_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=float(temperature),
    )
    content = resp.choices[0].message.content if resp.choices else ""
    return (content or "").strip()


def _call_deepseek(*, prompt: str, temperature: float, system_prompt: str) -> str:
    _ensure_repo_root_on_path()
    from app.config import get_settings  # type: ignore

    settings = get_settings()
    return _call_deepseek_model(
        prompt=prompt,
        temperature=temperature,
        system_prompt=system_prompt,
        model_name=settings.active_llm_model,
    )


def _call_deepseek_reasoner(*, prompt: str, temperature: float, system_prompt: str) -> str:
    _ensure_repo_root_on_path()
    from app.config import get_settings  # type: ignore

    settings = get_settings()
    return _call_deepseek_model(
        prompt=prompt,
        temperature=temperature,
        system_prompt=system_prompt,
        model_name=settings.active_llm_model,
    )


def build_single_doc_sample(
    *,
    doc_id: str,
    excerpt: str,
    max_answer_chars: int,
    temperature: float,
    language: str,
) -> dict[str, Any]:
    schema = (
        "{\n"
        '  "question": "...",\n'
        '  "answer_key": "...",\n'
        '  "key_points": ["..."],\n'
        '  "tags": ["..."],\n'
        '  "notes": ""\n'
        "}\n"
    )
    if language == "en":
        system_prompt = (
            "You are a strict dataset-construction assistant. "
            "Generate evaluable QA pairs from the provided evidence excerpt only. "
            "Do not use outside knowledge. "
            "Output JSON only."
        )
        prompt = (
            "Generate exactly 1 question and 1 reference answer_key from the evidence excerpt.\n"
            "Requirements:\n"
            "- The question must be answerable from this excerpt alone.\n"
            "- Rephrase instead of copying the original sentence.\n"
            "- Write both question and answer_key in English.\n"
            "- Provide 2-4 short key_points capturing the core facts a correct answer should cover.\n"
            f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
            '- If the evidence is insufficient, return "question" as an empty string.\n\n'
            f"JSON schema:\n{schema}\n"
            f"doc_id: {doc_id}\n\n"
            f"evidence_excerpt:\n{excerpt}\n"
        )
    else:
        system_prompt = (
            "You are a strict dataset-construction assistant. "
            "Generate evaluable QA pairs from the provided evidence only. "
            "Do not use outside knowledge. "
            "Output JSON only."
        )
        prompt = (
            "Generate exactly 1 question and 1 reference answer_key from the evidence excerpt.\n"
            "Requirements:\n"
            "- The question must be answerable from this excerpt alone.\n"
            "- Rephrase instead of copying the original sentence.\n"
            "- Write both question and answer_key in English.\n"
            "- Provide 2-4 short key_points capturing the core facts a correct answer should cover.\n"
            f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
            "- If the evidence is insufficient, return question as an empty string.\n\n"
            f"JSON schema:\n{schema}\n"
            f"doc_id: {doc_id}\n\n"
            f"evidence_excerpt:\n{excerpt}\n"
        )
    raw = _call_deepseek(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt)
    data = _extract_json(raw)
    if not isinstance(data.get("question"), str):
        data["question"] = ""
    if not isinstance(data.get("answer_key"), str):
        data["answer_key"] = ""
    if not isinstance(data.get("key_points"), list):
        data["key_points"] = []
    else:
        data["key_points"] = [str(x).strip() for x in data["key_points"] if str(x).strip()]
    if not isinstance(data.get("tags"), list):
        data["tags"] = []
    if not isinstance(data.get("notes"), str):
        data["notes"] = ""
    return data


def build_two_doc_sample(
    *,
    doc_id_a: str,
    excerpt_a: str,
    doc_id_b: str,
    excerpt_b: str,
    max_answer_chars: int,
    temperature: float,
    language: str,
) -> dict[str, Any]:
    schema = (
        "{\n"
        '  "question": "...",\n'
        '  "answer_key": "...",\n'
        '  "key_points": ["..."],\n'
        '  "tags": ["..."],\n'
        '  "notes": ""\n'
        "}\n"
    )
    if language == "en":
        system_prompt = (
            "You are a strict dataset-construction assistant. "
            "Generate evaluable multi-document QA pairs from the provided evidence only. "
            "Do not use outside knowledge. "
            "Output JSON only."
        )
        prompt = (
            'Generate exactly 1 comparison/synthesis question and 1 reference answer_key from two evidence excerpts.\n'
            "Requirements:\n"
            "- The question must require both excerpt A and excerpt B; one excerpt alone must be insufficient.\n"
            "- Do not use outside knowledge.\n"
            "- Write both question and answer_key in English.\n"
            "- Provide 3-6 short key_points; each should be an atomic fact or comparison point a good answer should cover.\n"
            f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
            '- If you cannot construct a valid question, return "question" as an empty string.\n\n'
            f"JSON schema:\n{schema}\n"
            f"doc_id_A: {doc_id_a}\n"
            f"doc_id_B: {doc_id_b}\n\n"
            f"evidence_A:\n{excerpt_a}\n\n"
            f"evidence_B:\n{excerpt_b}\n"
        )
    else:
        system_prompt = (
            "You are a strict dataset-construction assistant. "
            "Generate evaluable multi-document QA pairs from the provided evidence only. "
            "Do not use outside knowledge. "
            "Output JSON only."
        )
        prompt = (
            "Generate exactly 1 comparison/synthesis question and 1 reference answer_key from two evidence excerpts.\n"
            "Requirements:\n"
            "- The question must require both excerpt A and excerpt B; one excerpt alone must be insufficient.\n"
            "- Do not use outside knowledge.\n"
            "- Write both question and answer_key in English.\n"
            "- Provide 3-6 short key_points.\n"
            f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
            "- If you cannot construct a valid question, return question as an empty string.\n\n"
            f"JSON schema:\n{schema}\n"
            f"doc_id_A: {doc_id_a}\n"
            f"doc_id_B: {doc_id_b}\n\n"
            f"evidence_A:\n{excerpt_a}\n\n"
            f"evidence_B:\n{excerpt_b}\n"
        )
    raw = _call_deepseek(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt)
    data = _extract_json(raw)
    if not isinstance(data.get("question"), str):
        data["question"] = ""
    if not isinstance(data.get("answer_key"), str):
        data["answer_key"] = ""
    if not isinstance(data.get("key_points"), list):
        data["key_points"] = []
    else:
        data["key_points"] = [str(x).strip() for x in data["key_points"] if str(x).strip()]
    if not isinstance(data.get("tags"), list):
        data["tags"] = []
    if not isinstance(data.get("notes"), str):
        data["notes"] = ""
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks",
        type=Path,
        default=Path("petroleum_kb/petroleum_kb/data/processed/chunks.jsonl"),
        help="Path to processed chunks.jsonl",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output dataset JSONL path.")
    ap.add_argument("--num", type=int, default=50, help="Number of questions to generate.")
    ap.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["single", "two_doc", "mixed"],
        help="Question type: single-doc, two-doc, or ratio-driven mixed.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--excerpt-max-chars", type=int, default=1800)
    ap.add_argument("--max-answer-chars", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.2, help="DeepSeek temperature for dataset generation.")
    ap.add_argument("--min-chunk-chars", type=int, default=300)
    ap.add_argument("--min-doc-chunks", type=int, default=3, help="Minimum eligible chunks per paper for document-level question generation.")
    ap.add_argument("--doc-max-chunks", type=int, default=6, help="Maximum number of chunks per paper used to build evidence.")
    ap.add_argument(
        "--two-doc-pairing",
        type=str,
        default="semantic",
        choices=["semantic", "topical", "random"],
        help=(
            "Two-document pairing strategy. semantic uses the existing RAG chunk vectors; "
            "topical uses title/abstract/body-text lexical similarity."
        ),
    )
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=Path("petroleum_kb/petroleum_kb/data/index"),
        help="Vector index directory used by --two-doc-pairing semantic.",
    )
    ap.add_argument(
        "--index-name",
        type=str,
        default="petroleum_knowledge",
        help="Vector index name used by --two-doc-pairing semantic.",
    )
    ap.add_argument(
        "--two-doc-vector-min-sim",
        type=float,
        default=0.58,
        help="Minimum document-vector cosine similarity for semantic two-doc candidate pairs.",
    )
    ap.add_argument(
        "--two-doc-min-sim",
        type=float,
        default=0.035,
        help="Minimum topical cosine similarity for two-doc candidate pairs.",
    )
    ap.add_argument(
        "--two-doc-max-pairs-per-doc",
        type=int,
        default=80,
        help="Maximum topical candidate pairs retained per document.",
    )
    ap.add_argument("--quality-min-score", type=float, default=0.7, help="Minimum average self-contained/answerable/objective score.")
    ap.add_argument(
        "--max-attempts-per-item",
        type=int,
        default=40,
        help="Maximum reviewed-generation attempts for each requested item before stopping.",
    )
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N written questions.")
    ap.add_argument(
        "--benchmark-style",
        type=str,
        default="semantic",
        help="Question-generation style: semantic or specific_fact.",
    )
    ap.add_argument(
        "--answer-format",
        type=str,
        default="open",
        choices=["open", "mcq"],
        help="Answer format for generated benchmark items: open short-answer QA or strict four-option MCQ.",
    )
    ap.add_argument(
        "--two-doc-ratio",
        type=float,
        default=0.25,
        help="Legacy mixed-mode fraction for two-doc questions when --doc-count-ratios is omitted.",
    )
    ap.add_argument(
        "--doc-count-ratios",
        type=str,
        default=None,
        help='Mixed-mode JSON ratio map, e.g. \'{"1":0.7,"2":0.3}\'.',
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append generated rows to an existing benchmark instead of recreating the dataset.",
    )
    ap.add_argument(
        "--benchmark-mode",
        type=str,
        default="reviewed",
        choices=["reviewed", "naive_baseline"],
        help="Generation mode: reviewed applies all gates; naive_baseline uses a weaker generation prompt and skips gates.",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Question/answer language. Use en for the main benchmark on English papers.",
    )
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    benchmark_mode = _benchmark_mode_label(args.benchmark_mode)
    ungated_baseline_mode = benchmark_mode != "reviewed"
    naive_mode = benchmark_mode == "naive_baseline"
    benchmark_style = _normalize_benchmark_style(str(args.benchmark_style))
    answer_format = str(args.answer_format or "open").strip().lower()
    doc_count_ratios = _normalize_doc_count_ratios(args.doc_count_ratios, two_doc_ratio=float(args.two_doc_ratio))
    chunks = load_chunks(Path(args.chunks))
    groups = group_by_doc(chunks)
    doc_ids = [
        d
        for d, cs in groups.items()
        if sum(1 for c in cs if len(c.text) >= int(args.min_chunk_chars)) >= int(args.min_doc_chunks)
    ]
    if not doc_ids:
        raise SystemExit(f"No docs found in {args.chunks}. Did you run ingest firsttop")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    q_index = _existing_question_count(args.out) + 1 if args.append else 1
    seen_qa = _load_existing_qa_signatures(args.out) if args.append else set()
    seen_questions = _load_existing_question_signatures(args.out) if args.append else set()
    mcq_correct_option_counts = _load_existing_mcq_correct_option_counts(args.out) if args.append else Counter()
    generation_plan = _planned_doc_counts(mode=str(args.mode), total=int(args.num), ratios=doc_count_ratios)
    target_counts = _counts_from_plan(generation_plan)
    if target_counts.get(2, 0) and len(doc_ids) < 2:
        raise SystemExit("Two-document benchmark generation requires at least two eligible source documents.")
    doc_pair_candidates: list[DocPairCandidate] = []
    if target_counts.get(2, 0) and str(args.two_doc_pairing) == "semantic":
        pairable_doc_ids = _two_doc_pairable_doc_ids(groups, doc_ids)
        doc_pair_candidates = _build_semantic_doc_pair_candidates(
            groups=groups,
            doc_ids=doc_ids,
            rng=rng,
            index_dir=args.index_dir,
            index_name=str(args.index_name),
            min_similarity=float(args.two_doc_vector_min_sim),
            max_pairs_per_doc=int(args.two_doc_max_pairs_per_doc),
        )
        print(
            f"[dataset] two_doc_pairing=semantic candidates={len(doc_pair_candidates)} "
            f"eligible_docs={len(doc_ids)} pairable_docs={len(pairable_doc_ids)} "
            f"petroleum_core_min_score={PROFILE_PETROLEUM_CORE_MIN_SCORE:.1f} "
            f"vector_min_sim={float(args.two_doc_vector_min_sim):.3f} "
            f"index={args.index_dir / (str(args.index_name) + '.faiss')}",
            flush=True,
        )
    elif target_counts.get(2, 0) and str(args.two_doc_pairing) == "topical":
        pairable_doc_ids = _two_doc_pairable_doc_ids(groups, doc_ids)
        doc_pair_candidates = _build_doc_pair_candidates(
            groups=groups,
            doc_ids=doc_ids,
            rng=rng,
            min_similarity=float(args.two_doc_min_sim),
            max_pairs_per_doc=int(args.two_doc_max_pairs_per_doc),
        )
        print(
            f"[dataset] two_doc_pairing=topical candidates={len(doc_pair_candidates)} "
            f"eligible_docs={len(doc_ids)} pairable_docs={len(pairable_doc_ids)} "
            f"petroleum_core_min_score={PROFILE_PETROLEUM_CORE_MIN_SCORE:.1f} "
            f"min_sim={float(args.two_doc_min_sim):.3f}",
            flush=True,
        )
    written_by_type: dict[int, int] = {1: 0, 2: 0}
    duplicate_skips = 0
    near_duplicate_skips = 0
    language_skips = 0
    empty_question_skips = 0
    judge_fail_skips = 0
    quality_score_skips = 0

    with args.out.open("a" if args.append else "w", encoding="utf-8") as out:
        # Hard stops avoid infinite loops when the LLM returns low-quality or
        # duplicate questions. A failed item is explicit instead of silently
        # writing fewer rows than requested.
        max_attempts = int(max(100, args.num * 60))
        max_attempts_per_item = int(args.max_attempts_per_item)
        attempts = 0
        used_single_docs: set[str] = set()
        used_doc_pairs: set[tuple[str, str]] = set()

        for target_index, doc_count in enumerate(generation_plan, start=1):
            row: dict[str, Any] | None = None
            for _attempt_for_item in range(max_attempts_per_item):
                row = None
                attempts += 1
                if attempts > max_attempts:
                    break
                if int(args.progress_every or 0) > 0:
                    print(
                        f"[dataset] progress: target={target_index}/{len(generation_plan)} "
                        f"written={written}/{len(generation_plan)} "
                        f"current_type={'two_doc' if doc_count == 2 else 'single'} "
                        f"benchmark_mode={benchmark_mode} "
                        f"single={written_by_type.get(1, 0)}/{target_counts.get(1, 0)} "
                        f"two_doc={written_by_type.get(2, 0)}/{target_counts.get(2, 0)} "
                        f"item_attempt={_attempt_for_item + 1}/{max_attempts_per_item} "
                        f"attempts={attempts} duplicate_skips={duplicate_skips} "
                        f"near_duplicate_skips={near_duplicate_skips} "
                        f"language_skips={language_skips} "
                        f"judge_fail_skips={judge_fail_skips} "
                        f"quality_score_skips={quality_score_skips}",
                        flush=True,
                    )

                id_prefix = "sq"

                if doc_count == 2:
                    a, b = _select_doc_pair(
                        doc_ids=doc_ids,
                        used_pairs=used_doc_pairs,
                        rng=rng,
                        candidates=doc_pair_candidates,
                    )
                    if doc_pair_candidates:
                        focus_terms = _pair_focus_terms(left_chunks=groups[a], right_chunks=groups[b])
                        ev_a = _focused_doc_evidence(
                            doc_id=a,
                            chunks=groups[a],
                            focus_terms=focus_terms,
                            max_chars=int(args.excerpt_max_chars),
                            max_chunks=int(args.doc_max_chunks),
                            min_chunk_chars=int(args.min_chunk_chars),
                        )
                        ev_b = _focused_doc_evidence(
                            doc_id=b,
                            chunks=groups[b],
                            focus_terms=focus_terms,
                            max_chars=int(args.excerpt_max_chars),
                            max_chunks=int(args.doc_max_chunks),
                            min_chunk_chars=int(args.min_chunk_chars),
                        )
                    else:
                        ev_a = build_doc_evidence(
                            doc_id=a,
                            chunks=groups[a],
                            max_chars=int(args.excerpt_max_chars),
                            max_chunks=int(args.doc_max_chunks),
                            min_chunk_chars=int(args.min_chunk_chars),
                        )
                        ev_b = build_doc_evidence(
                            doc_id=b,
                            chunks=groups[b],
                            max_chars=int(args.excerpt_max_chars),
                            max_chunks=int(args.doc_max_chunks),
                            min_chunk_chars=int(args.min_chunk_chars),
                        )
                    if not ev_a["text"] or not ev_b["text"]:
                        continue
                    if naive_mode:
                        qa = generate_naive_doc_level_sample(
                            llm_call=_call_deepseek,
                            mode="two_doc",
                            evidence_a=ev_a,
                            evidence_b=ev_b,
                            max_answer_chars=int(args.max_answer_chars),
                            temperature=float(args.temperature),
                            language=str(args.language),
                        )
                    else:
                        qa = generate_doc_level_sample(
                            llm_call=_call_deepseek,
                            mode="two_doc",
                            evidence_a=ev_a,
                            evidence_b=ev_b,
                            max_answer_chars=int(args.max_answer_chars),
                            temperature=float(args.temperature),
                            language=str(args.language),
                            benchmark_style=benchmark_style,
                            question_variant=_two_doc_question_variant(target_index, _attempt_for_item),
                            answer_format=answer_format,
                        )
                        qa = _normalize_mcq_row(
                            qa,
                            target_correct_option=_least_used_mcq_option(mcq_correct_option_counts, rng),
                            rng=rng,
                        )
                        if qa is None:
                            judge_fail_skips += 1
                            continue
                    review = review_and_rewrite_question(
                        llm_call=_call_deepseek_reasoner,
                        question_type="two_doc",
                        qa=qa,
                        evidence_a=ev_a,
                        evidence_b=ev_b,
                        benchmark_style=benchmark_style,
                        temperature=0.0,
                    )
                    if not ungated_baseline_mode:
                        review = _review_rewritten_candidate_if_needed(
                            review=review,
                            question_type="two_doc",
                            evidence_a=ev_a,
                            evidence_b=ev_b,
                            benchmark_style=benchmark_style,
                            min_quality=float(args.quality_min_score),
                        )
                    avg_quality = float(review.get("quality_score") or 0.0)
                    final_qa = qa if naive_mode else review["rewritten"]
                    question = str(final_qa.get("question") or "").strip()
                    if not question:
                        empty_question_skips += 1
                        continue
                    answer_key = str(final_qa.get("answer_key") or "").strip()
                    if not answer_key:
                        empty_question_skips += 1
                        continue
                    if not ungated_baseline_mode and not _retrievable_question_gate_passes({"question": question}):
                        final_qa = _rewrite_question_for_retrieval(
                            qa=final_qa,
                            evidence_a=ev_a,
                            evidence_b=ev_b,
                            max_answer_chars=int(args.max_answer_chars),
                        )
                        question = str(final_qa.get("question") or "").strip()
                        answer_key = str(final_qa.get("answer_key") or "").strip()
                        if not question or not answer_key:
                            empty_question_skips += 1
                            continue
                    if not ungated_baseline_mode and not _reviewed_score_gate_passes(
                        review,
                        question_type="two_doc",
                        min_quality=float(args.quality_min_score),
                    ):
                        judge_fail_skips += 1
                        continue
                    if not ungated_baseline_mode and avg_quality < float(args.quality_min_score):
                        quality_score_skips += 1
                        continue
                    row = {
                        "id": f"{id_prefix}{q_index:04d}",
                        "benchmark_mode": benchmark_mode,
                        "benchmark_style": benchmark_style,
                        "question": question,
                        "question_type": "two_doc",
                        "gold_doc_ids": [a, b],
                        "equivalent_doc_groups": [
                            {"gold_doc_id": a, "doc_ids": [a]},
                            {"gold_doc_id": b, "doc_ids": [b]},
                        ],
                        "answer_key": answer_key,
                        "key_points": final_qa.get("key_points") if isinstance(final_qa.get("key_points"), list) else [],
                        "tags": final_qa.get("tags") if isinstance(final_qa.get("tags"), list) else [],
                        "notes": str(final_qa.get("notes") or ""),
                        "answer_format": str(final_qa.get("answer_format") or answer_format),
                        "choices": final_qa.get("choices") if isinstance(final_qa.get("choices"), list) else [],
                        "correct_option": str(final_qa.get("correct_option") or ""),
                        "question_quality": _quality_block(review),
                        "source_chunk_ids": {a: ev_a["chunk_ids"], b: ev_b["chunk_ids"]},
                        "type": "two_doc",
                        "language": str(args.language),
                    }
                else:
                    d = _select_single_doc(doc_ids=doc_ids, used_docs=used_single_docs, rng=rng)
                    ev = build_doc_evidence(
                        doc_id=d,
                        chunks=groups[d],
                        max_chars=int(args.excerpt_max_chars),
                        max_chunks=int(args.doc_max_chunks),
                        min_chunk_chars=int(args.min_chunk_chars),
                    )
                    if not ev["text"]:
                        continue
                    if naive_mode:
                        qa = generate_naive_doc_level_sample(
                            llm_call=_call_deepseek,
                            mode="single",
                            evidence_a=ev,
                            evidence_b=None,
                            max_answer_chars=int(args.max_answer_chars),
                            temperature=float(args.temperature),
                            language=str(args.language),
                        )
                    else:
                        qa = generate_doc_level_sample(
                            llm_call=_call_deepseek,
                            mode="single",
                            evidence_a=ev,
                            evidence_b=None,
                            max_answer_chars=int(args.max_answer_chars),
                            temperature=float(args.temperature),
                            language=str(args.language),
                            benchmark_style=benchmark_style,
                            answer_format=answer_format,
                        )
                        qa = _normalize_mcq_row(
                            qa,
                            target_correct_option=_least_used_mcq_option(mcq_correct_option_counts, rng),
                            rng=rng,
                        )
                        if qa is None:
                            judge_fail_skips += 1
                            continue
                    review = review_and_rewrite_question(
                        llm_call=_call_deepseek_reasoner,
                        question_type="single",
                        qa=qa,
                        evidence_a=ev,
                        evidence_b=None,
                        benchmark_style=benchmark_style,
                        temperature=0.0,
                    )
                    if not ungated_baseline_mode:
                        review = _review_rewritten_candidate_if_needed(
                            review=review,
                            question_type="single",
                            evidence_a=ev,
                            evidence_b=None,
                            benchmark_style=benchmark_style,
                            min_quality=float(args.quality_min_score),
                        )
                    avg_quality = float(review.get("quality_score") or 0.0)
                    final_qa = qa if naive_mode else review["rewritten"]
                    question = str(final_qa.get("question") or "").strip()
                    if not question:
                        empty_question_skips += 1
                        continue
                    answer_key = str(final_qa.get("answer_key") or "").strip()
                    if not answer_key:
                        empty_question_skips += 1
                        continue
                    if not ungated_baseline_mode and not _retrievable_question_gate_passes({"question": question}):
                        final_qa = _rewrite_question_for_retrieval(
                            qa=final_qa,
                            evidence_a=ev,
                            evidence_b=None,
                            max_answer_chars=int(args.max_answer_chars),
                        )
                        question = str(final_qa.get("question") or "").strip()
                        answer_key = str(final_qa.get("answer_key") or "").strip()
                        if not question or not answer_key:
                            empty_question_skips += 1
                            continue
                    if not ungated_baseline_mode and not _reviewed_score_gate_passes(
                        review,
                        question_type="single",
                        min_quality=float(args.quality_min_score),
                    ):
                        judge_fail_skips += 1
                        continue
                    if not ungated_baseline_mode and avg_quality < float(args.quality_min_score):
                        quality_score_skips += 1
                        continue
                    row = {
                        "id": f"{id_prefix}{q_index:04d}",
                        "benchmark_mode": benchmark_mode,
                        "benchmark_style": benchmark_style,
                        "question": question,
                        "question_type": "single",
                        "gold_doc_ids": [d],
                        "equivalent_doc_groups": [
                            {"gold_doc_id": d, "doc_ids": [d]},
                        ],
                        "answer_key": answer_key,
                        "key_points": final_qa.get("key_points") if isinstance(final_qa.get("key_points"), list) else [],
                        "tags": final_qa.get("tags") if isinstance(final_qa.get("tags"), list) else [],
                        "notes": str(final_qa.get("notes") or ""),
                        "answer_format": str(final_qa.get("answer_format") or answer_format),
                        "choices": final_qa.get("choices") if isinstance(final_qa.get("choices"), list) else [],
                        "correct_option": str(final_qa.get("correct_option") or ""),
                        "question_quality": _quality_block(review),
                        "source_chunk_ids": {d: ev["chunk_ids"]},
                        "type": "single",
                        "language": str(args.language),
                    }

                normalized_row = _normalize_mcq_row(
                    row,
                    target_correct_option=_least_used_mcq_option(mcq_correct_option_counts, rng),
                    rng=rng,
                )
                if normalized_row is None:
                    judge_fail_skips += 1
                    row = None
                    continue
                row = normalized_row

                signature = _qa_signature(row)
                question_signature = _question_signature(row)
                if not ungated_baseline_mode and not _language_gate_passes(row, str(args.language)):
                    language_skips += 1
                    row = None
                    continue
                if not ungated_baseline_mode and not _retrievable_question_gate_passes(row):
                    judge_fail_skips += 1
                    row = None
                    continue
                if not ungated_baseline_mode and not _mcq_format_gate_passes(row):
                    judge_fail_skips += 1
                    row = None
                    continue
                if not ungated_baseline_mode and (signature in seen_qa or question_signature in seen_questions):
                    duplicate_skips += 1
                    row = None
                    continue
                if not ungated_baseline_mode and _is_near_duplicate_question(question_signature, seen_questions):
                    near_duplicate_skips += 1
                    row = None
                    continue
                break

            if row is None:
                raise SystemExit(
                    "Unable to generate the requested non-duplicate benchmark composition "
                    f"for doc_count={doc_count} after {max_attempts_per_item} attempts. "
                    "Try increasing source documents, lowering quality-min-score, or reducing requested QA count. "
                    f"Skip counters: duplicate_skips={duplicate_skips}, "
                    f"near_duplicate_skips={near_duplicate_skips}, language_skips={language_skips}, "
                    f"judge_fail_skips={judge_fail_skips}, quality_score_skips={quality_score_skips}, "
                    f"empty_question_skips={empty_question_skips}."
                )

            seen_qa.add(_qa_signature(row))
            seen_questions.add(_question_signature(row))
            if str(row.get("answer_format") or "").strip().lower() == "mcq":
                correct_option = str(row.get("correct_option") or "").strip().upper()
                if correct_option in {"A", "B", "C", "D"}:
                    mcq_correct_option_counts[correct_option] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            written_by_type[doc_count] = written_by_type.get(doc_count, 0) + 1
            q_index += 1
            if int(args.progress_every or 0) > 0 and (
                written % int(args.progress_every) == 0 or written == len(generation_plan)
            ):
                print(
                    f"[dataset] progress: written={written}/{len(generation_plan)} "
                    f"single={written_by_type.get(1, 0)}/{target_counts.get(1, 0)} "
                    f"two_doc={written_by_type.get(2, 0)}/{target_counts.get(2, 0)} "
                    f"attempts={attempts} duplicate_skips={duplicate_skips} "
                    f"near_duplicate_skips={near_duplicate_skips} "
                    f"language_skips={language_skips} "
                    f"judge_fail_skips={judge_fail_skips} "
                    f"quality_score_skips={quality_score_skips}",
                    flush=True,
                )

    print(
        f"Wrote dataset to {args.out} "
        f"(benchmark_mode={benchmark_mode}, "
        f"questions={written}, single={written_by_type.get(1, 0)}/{target_counts.get(1, 0)}, "
        f"two_doc={written_by_type.get(2, 0)}/{target_counts.get(2, 0)}, "
        f"attempts={attempts}, duplicate_skips={duplicate_skips}, "
        f"near_duplicate_skips={near_duplicate_skips}, language_skips={language_skips}, "
        f"judge_fail_skips={judge_fail_skips}, quality_score_skips={quality_score_skips}, "
        f"empty_question_skips={empty_question_skips})"
    )


if __name__ == "__main__":
    main()

