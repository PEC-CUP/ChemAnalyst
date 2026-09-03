from __future__ import annotations

import json
import re
from typing import Any, Callable

QUALITY_SCORE_WEIGHTS = {
    "self_contained": 0.16,
    "answerable": 0.16,
    "objective": 0.12,
    "specificity": 0.18,
    "evidence_locality": 0.23,
    "clarity": 0.07,
    "grounding_safety": 0.08,
}


def extract_json_object(text: str) -> dict[str, Any]:
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


def _clamp_score(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _contains_cjk(text: Any) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _has_vague_paper_reference(text: Any) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "according to the paper",
        "according to the study",
        "according to the studies",
        "according to the article",
        "according to both papers",
        "according to the two papers",
        "according to the two studies",
        "according to the two approaches",
        "based on the provided evidence",
        "based on provided evidence",
        "based on the paper",
        "based on the study",
        "based on the article",
        "based on the  paper",
        "based on paper a",
        "based on paper b",
        "in the paper",
        "in the study",
        "in the studies",
        "in this paper",
        "in this study",
        "paper a",
        "paper b",
        "provided evidence",
        "the two papers",
        "these two papers",
        "both papers",
        "the two studies",
        "these two studies",
        "both studies",
        "the two approaches",
        "these two approaches",
        "both approaches",
        "the two methods",
        "these two methods",
        "both methods",
        "the two works",
        "these two works",
        "both works",
        "two articles",
        "both articles",
        "the 2016 paper",
        "the 2017 paper",
        "the 2018 paper",
        "the 2019 paper",
        "the 2020 paper",
        "the 2021 paper",
        "the 2022 paper",
        "the 2023 paper",
        "the 2024 paper",
        "the 2025 paper",
        "energy & fuels study",
        "organic geochemistry paper",
    )
    if any(marker in lowered for marker in markers):
        return True

    # Catch broader paper-referential lead-ins that remain too vague for a
    # reviewed benchmark even when a method name is present, such as
    # "According to the DPF MS method studies, ...".
    vague_leadin_patterns = (
        r"^\s*according to\s+(top:the\s+)top[a-z0-9\-\s/&]+(top:paper|papers|study|studies|article|articles)\b",
        r"^\s*in\s+(top:the\s+)top[a-z0-9\-\s/&]+(top:paper|papers|study|studies|article|articles)\b",
        r"^\s*based on\s+(top:the\s+)top[a-z0-9\-\s/&]+(top:paper|papers|study|studies|article|articles)\b",
        r"\b(top:19|20)\d{2}\s+(top:paper|papers|study|studies|article|articles)\b",
        r"\b(top:19|20)\d{2}\s+[a-z][a-z&\-\s]{2,80}\s+(top:paper|papers|study|studies|article|articles)\b",
        r"\b[a-z][a-z&\-\s]{2,80}\s+(top:19|20)\d{2}\s+(top:paper|papers|study|studies|article|articles)\b",
        r"\b(top:paper|study|article)\s+(top:from|in)\s+(top:19|20)\d{2}\b",
        r"\b(top:the|these|both)\s+two\s+(top:papers|studies|articles)\b",
        r"\bboth\s+(top:papers|studies|articles)\b",
        r"\b(top:the|these|both)\s+two\s+(top:approaches|methods|works)\b",
        r"\bboth\s+(top:approaches|methods|works)\b",
        r"\bpaper\s+[ab]\b",
    )
    return any(re.search(pattern, lowered) for pattern in vague_leadin_patterns)


def _compute_quality_score(
    *,
    self_contained: float,
    answerable: float,
    objective: float,
    specificity: float,
    evidence_locality: float,
    ambiguity: float,
    hallucination_risk: float,
) -> float:
    weighted_average = (
        QUALITY_SCORE_WEIGHTS["self_contained"] * self_contained
        + QUALITY_SCORE_WEIGHTS["answerable"] * answerable
        + QUALITY_SCORE_WEIGHTS["objective"] * objective
        + QUALITY_SCORE_WEIGHTS["specificity"] * specificity
        + QUALITY_SCORE_WEIGHTS["evidence_locality"] * evidence_locality
        + QUALITY_SCORE_WEIGHTS["clarity"] * (1.0 - ambiguity)
        + QUALITY_SCORE_WEIGHTS["grounding_safety"] * (1.0 - hallucination_risk)
    )
    core_floor = min(self_contained, answerable, objective, specificity, evidence_locality)
    risk_ceiling = min(1.0 - ambiguity, 1.0 - hallucination_risk)
    return min(weighted_average, core_floor, risk_ceiling)


def _normalize_quality_review(review: dict[str, Any]) -> dict[str, Any]:
    """Normalize benchmark quality-review dimensions.

    Positive dimensions use 1 as best. Risk dimensions use 0 as best:
    ambiguity_score and hallucination_risk_score indicate risk, not quality.
    """

    self_contained = _clamp_score(review.get("self_contained_score"), 0.0)
    answerable = _clamp_score(review.get("answerable_score"), 0.0)
    objective = _clamp_score(review.get("objective_score"), 0.0)
    specificity = _clamp_score(review.get("specificity_score"), self_contained)
    ambiguity = _clamp_score(review.get("ambiguity_score"), 1.0 - self_contained)
    evidence_locality = _clamp_score(review.get("evidence_locality_score"), answerable)
    hallucination_risk = _clamp_score(review.get("hallucination_risk_score"), 1.0 - answerable)
    reasoning_type = str(review.get("reasoning_type") or "other").strip().lower()
    if reasoning_type not in {
        "fact_lookup",
        "method_explanation",
        "comparison",
        "synthesis",
        "causal_reasoning",
        "property_reasoning",
        "other",
        "unknown",
    }:
        reasoning_type = "other"
    difficulty = str(review.get("difficulty") or "medium").strip().lower()
    if difficulty not in {"easy", "medium", "hard", "unknown"}:
        difficulty = "medium"
    quality_score = _compute_quality_score(
        self_contained=self_contained,
        answerable=answerable,
        objective=objective,
        specificity=specificity,
        evidence_locality=evidence_locality,
        ambiguity=ambiguity,
        hallucination_risk=hallucination_risk,
    )
    return {
        "self_contained_score": self_contained,
        "answerable_score": answerable,
        "objective_score": objective,
        "specificity_score": specificity,
        "ambiguity_score": ambiguity,
        "evidence_locality_score": evidence_locality,
        "reasoning_type": reasoning_type,
        "difficulty": difficulty,
        "hallucination_risk_score": hallucination_risk,
        "quality_score": quality_score,
    }


def _recompute_quality_score(quality: dict[str, Any]) -> None:
    self_contained = _clamp_score(quality.get("self_contained_score"), 0.0)
    answerable = _clamp_score(quality.get("answerable_score"), 0.0)
    objective = _clamp_score(quality.get("objective_score"), 0.0)
    specificity = _clamp_score(quality.get("specificity_score"), 0.0)
    evidence_locality = _clamp_score(quality.get("evidence_locality_score"), 0.0)
    ambiguity = _clamp_score(quality.get("ambiguity_score"), 1.0)
    hallucination_risk = _clamp_score(quality.get("hallucination_risk_score"), 1.0)
    quality["quality_score"] = _compute_quality_score(
        self_contained=self_contained,
        answerable=answerable,
        objective=objective,
        specificity=specificity,
        evidence_locality=evidence_locality,
        ambiguity=ambiguity,
        hallucination_risk=hallucination_risk,
    )


def _review_two_doc_specifics(
    *,
    llm_call: Callable[..., str],
    question: str,
    qa: dict[str, Any],
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    temperature: float = 0.0,
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict auditor for two-document petroleum-analysis benchmark QA items. "
        "Decide whether the item truly requires both papers and whether the answer genuinely performs "
        "cross-document comparison or synthesis. Output JSON only."
    )
    prompt = (
        "Evaluate this two-document benchmark item under a strict two_doc-specific rule.\n"
        "Return JSON:\n"
        "{\n"
        '  "two_doc_pass": true/false,\n'
        '  "both_docs_required_score": 0-1,\n'
        '  "cross_doc_synthesis_score": 0-1,\n'
        '  "doc_balance_score": 0-1,\n'
        '  "failure_reason": "...",\n'
        '  "notes": "..."\n'
        "}\n\n"
        "Decision rules:\n"
        "- two_doc_pass=true only if the question genuinely requires BOTH papers and the answer/key points materially use BOTH papers.\n"
        "- Fail if the answer is mostly parallel description without explicit comparison/synthesis.\n"
        "- Fail if one paper could be removed and the answer would remain mostly intact.\n"
        "- Fail if one paper contributes only a minor side fact rather than a core part of the answer.\n"
        "- both_docs_required_score: does the question truly need both docstop\n"
        "- cross_doc_synthesis_score: does the answer synthesize/compare rather than just concatenatetop\n"
        "- doc_balance_score: are both docs carrying meaningful loadtop\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER_KEY: {qa.get('answer_key', '')}\n\n"
        f"KEY_POINTS: {qa.get('key_points', [])}\n\n"
        f"PAPER_A_ID: {evidence_a['doc_id']}\nPAPER_A_EVIDENCE:\n{evidence_a['text']}\n\n"
        f"PAPER_B_ID: {evidence_b['doc_id']}\nPAPER_B_EVIDENCE:\n{evidence_b['text']}\n"
    )
    return extract_json_object(llm_call(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt))


def _chunk_order_key(chunk: Any) -> tuple[int, int]:
    md = getattr(chunk, "metadata", {}) if hasattr(chunk, "metadata") else {}
    raw_chunk_id = md.get("chunk_id")
    try:
        chunk_idx = int(raw_chunk_id)
    except Exception:
        chunk_idx = 10**9
    text_len = len(str(getattr(chunk, "text", "") or ""))
    return (chunk_idx, -text_len)


def _chunk_uid(chunk: Any, fallback: int) -> str:
    raw = getattr(chunk, "chunk_id", None)
    if raw:
        return str(raw)
    md = getattr(chunk, "metadata", {}) if hasattr(chunk, "metadata") else {}
    return str(md.get("chunk_uid") or md.get("global_chunk_id") or md.get("chunk_id", fallback))


def build_doc_evidence(
    *,
    doc_id: str,
    chunks: list[Any],
    max_chars: int,
    max_chunks: int,
    min_chunk_chars: int,
) -> dict[str, Any]:
    eligible = [c for c in chunks if len(str(getattr(c, "text", "") or "").strip()) >= int(min_chunk_chars)]
    if not eligible:
        return {"doc_id": doc_id, "text": "", "chunk_ids": [], "metadata": {}}

    ordered = sorted(eligible, key=_chunk_order_key)
    if len(ordered) <= max_chunks:
        selected = ordered
    else:
        # Evenly spread across the paper, not just one local fragment.
        step = max(len(ordered) / float(max_chunks), 1.0)
        idxs = sorted({min(int(round(i * step)), len(ordered) - 1) for i in range(max_chunks)})
        selected = [ordered[i] for i in idxs]

    parts: list[str] = []
    chunk_ids: list[str] = []
    meta = getattr(selected[0], "metadata", {}) if selected else {}
    title = str(meta.get("title") or "").strip()
    year = str(meta.get("year") or "").strip()
    journal = str(meta.get("journal") or "").strip()
    header_bits = [x for x in (title, year, journal) if x]
    header = f"Paper: {' | '.join(header_bits)}" if header_bits else f"Paper: {doc_id}"
    parts.append(header)

    total_chars = len(header)
    for i, chunk in enumerate(selected, start=1):
        md = getattr(chunk, "metadata", {}) if hasattr(chunk, "metadata") else {}
        chunk_id = _chunk_uid(chunk, i)
        text = str(getattr(chunk, "text", "") or "").strip()
        prefix = f"[Segment {i} | chunk_id={chunk_id}]\n"
        block = f"{prefix}{text}"
        remaining = int(max_chars) - total_chars - 2
        if len(block) > remaining:
            # Keep evidence traceability even when the first selected chunk is
            # longer than the excerpt budget. The previous behavior broke
            # before appending any segment, leaving only the paper header and
            # empty source_chunk_ids. Truncate the segment text but still record
            # the chunk id used as evidence.
            suffix = "\n[truncated]"
            available_text_chars = remaining - len(prefix) - len(suffix)
            if available_text_chars <= 80:
                break
            text = text[:available_text_chars].rstrip()
            block = f"{prefix}{text}{suffix}"
            parts.append(block)
            chunk_ids.append(chunk_id)
            total_chars += len(block) + 2
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


def _normalize_qa_dict(data: dict[str, Any]) -> dict[str, Any]:
    answer_format = str(data.get("answer_format") or "open").strip().lower()
    if answer_format not in {"open", "mcq"}:
        answer_format = "open"
    choices = data.get("choices")
    normalized_choices: list[str] = []
    if isinstance(choices, list):
        normalized_choices = [str(x).strip() for x in choices if str(x).strip()]
    if answer_format != "mcq":
        normalized_choices = []
    return {
        "question": str(data.get("question") or "").strip(),
        "answer_key": str(data.get("answer_key") or "").strip(),
        "key_points": [str(x).strip() for x in data.get("key_points", []) if str(x).strip()] if isinstance(data.get("key_points"), list) else [],
        "tags": [str(x).strip() for x in data.get("tags", []) if str(x).strip()] if isinstance(data.get("tags"), list) else [],
        "notes": str(data.get("notes") or "").strip(),
        "answer_format": answer_format,
        "choices": normalized_choices,
        "correct_option": str(data.get("correct_option") or "").strip().upper() if answer_format == "mcq" else "",
    }


def generate_naive_doc_level_sample(
    *,
    llm_call: Callable[..., str],
    mode: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any] | None,
    max_answer_chars: int,
    temperature: float,
    language: str,
) -> dict[str, Any]:
    """Generate a weak-prompt baseline QA item.

    This baseline keeps the document-count setup aligned with the reviewed
    benchmark, but intentionally avoids the stronger benchmark-design
    constraints such as self-containedness, specificity, or paper-reference
    avoidance. It is still asked to return structured JSON so the item can be
    scored by the same judge and persisted in the same dataset schema.
    """

    if language != "en":
        language = "en"

    schema = (
        "{\n"
        '  "question": "...",\n'
        '  "answer_key": "...",\n'
        '  "key_points": ["..."],\n'
        '  "tags": ["..."],\n'
        '  "notes": ""\n'
        "}\n"
    )
    is_two = mode == "two_doc"
    if is_two:
        system_prompt = (
            "You generate baseline QA pairs from supplied petroleum-paper evidence. "
            "Return JSON only."
        )
        prompt = (
            "Generate exactly 1 question and 1 reference answer from the two paper evidence blocks.\n"
            "Use both papers when possible.\n"
            f"Keep answer_key within {int(max_answer_chars)} characters.\n"
            "Include a short list of key_points, tags, and optional notes.\n"
            'If you cannot generate a usable QA pair, return "question" as an empty string.\n\n'
            f"JSON schema:\n{schema}\n"
            f"doc_id_A: {evidence_a['doc_id']}\n"
            f"doc_id_B: {evidence_b['doc_id'] if evidence_b else ''}\n\n"
            f"paper_A_evidence:\n{evidence_a['text']}\n\n"
            f"paper_B_evidence:\n{evidence_b['text'] if evidence_b else ''}\n"
        )
    else:
        system_prompt = (
            "You generate baseline QA pairs from supplied petroleum-paper evidence. "
            "Return JSON only."
        )
        prompt = (
            "Generate exactly 1 question and 1 reference answer from the paper evidence block.\n"
            f"Keep answer_key within {int(max_answer_chars)} characters.\n"
            "Include a short list of key_points, tags, and optional notes.\n"
            'If you cannot generate a usable QA pair, return "question" as an empty string.\n\n'
            f"JSON schema:\n{schema}\n"
            f"doc_id: {evidence_a['doc_id']}\n\n"
            f"paper_evidence:\n{evidence_a['text']}\n"
        )

    raw = llm_call(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt)
    return _normalize_qa_dict(extract_json_object(raw))


def generate_doc_level_sample(
    *,
    llm_call: Callable[..., str],
    mode: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any] | None,
    max_answer_chars: int,
    temperature: float,
    language: str,
    benchmark_style: str,
    question_variant: str | None = None,
    answer_format: str = "open",
) -> dict[str, Any]:
    if language != "en":
        # Keep the pipeline deterministic; current benchmark work is English-first.
        language = "en"

    is_two = mode == "two_doc"
    normalized_style = str(benchmark_style or "semantic").strip().lower()
    normalized_answer_format = str(answer_format or "open").strip().lower()
    if normalized_answer_format not in {"open", "mcq"}:
        normalized_answer_format = "open"
    schema = (
        "{\n"
        '  "question": "...",\n'
        '  "answer_key": "...",\n'
        '  "key_points": ["..."],\n'
        '  "tags": ["..."],\n'
        '  "notes": "",\n'
        '  "answer_format": "open|mcq",\n'
        '  "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],\n'
        '  "correct_option": "A|B|C|D"\n'
        "}\n"
    )
    if normalized_style == "specific_fact":
        style_req = (
            "- Generate a narrow evidence-local fact question, not a broad review or general explanation.\n"
            "- Prefer exact parameters, thresholds, ranges, ratios, named sample fractions, ionization modes, detector settings, formula constraints, performance values, or specific method-selection facts.\n"
            "- The answer must be short and objectively checkable from the supplied evidence.\n"
            "- The question should be difficult to answer from general model priors alone; it should require finding the specific evidence.\n"
            "- Avoid broad prompts such as advantages, implications, mechanisms in general, or literature-wide overviews.\n"
            "- Good open question: 'What resolving power threshold was used to assign elemental compositions in the FT-ICR MS workflowtop'\n"
            "- Good open question: 'Which ionization mode was used to preferentially detect acidic O2-class species in the crude-oil fractiontop'\n"
            "- Bad: 'What are the advantages and challenges of FT-ICR MS for petroleum analysistop'\n"
            "- Bad: 'How do these papers improve petroleum characterizationtop'\n"
        )
    else:
        style_req = (
            "- Prefer literature-derived explanatory questions with broad semantic answers.\n"
            "- Optimize for answer correctness, answer relevancy, and context quality rather than exact citation wording.\n"
            "- The question subject must be a concrete petroleum-analysis concept, method, phenomenon, sample class, or analytical comparison.\n"
            "- Do not ask 'what did a specific paper/study report' unless the question explicitly names a reusable method or concept from that work.\n"
            "- Avoid paper-referential anchors such as 'the 2021 Energy & Fuels study', 'the 2016 paper', 'according to the paper', or journal/year-only descriptions.\n"
            "- Good: ask about DPF method principles, APCI vs ESI applicability, ionization suppression, FT-ICR MS assignments, or crude oil fraction characterization.\n"
            "- Bad: ask what a particular year's paper found when the paper identity is not known to the user.\n"
        )
    if normalized_answer_format == "mcq":
        format_req = (
            "- Generate a four-option multiple-choice question.\n"
            "- The question text must contain the question stem followed by exactly four labeled options A-D.\n"
            "- The choices field must exactly repeat the same four labeled options shown in the question text.\n"
            "- Only one option may be correct; distractors must be plausible but contradicted by or unsupported by the evidence.\n"
            "- Set answer_format to mcq.\n"
            "- Set correct_option to the correct letter A, B, C, or D.\n"
            "- Set answer_key to the correct labeled option only, for example 'B. 30 ng/mL'. Do not add a separate explanation inside answer_key.\n"
            "- Do not put option-letter statements in key_points, such as 'the correct option is C'. Key_points must state evidence facts only.\n"
            "- Do not systematically place the correct answer at C; the code will also relabel options after generation.\n"
        )
    else:
        format_req = (
            "- Generate an open short-answer question.\n"
            "- Set answer_format to open, choices to an empty list, and correct_option to an empty string.\n"
            "- The answer_key should be concise and directly grounded in the supplied evidence.\n"
        )
    kp_req = "4-8" if not is_two else "5-10"
    if normalized_style == "specific_fact":
        kp_req = "2-4" if not is_two else "3-5"

    if is_two:
        variant_req = ""
        if question_variant:
            variant_req = f"- Use this question style for this item: {question_variant}\n"
        system_prompt = (
            "You are a semantic literature-QA benchmark constructor. "
            "Build a self-contained, retrievable literature synthesis question. "
            "Do not mention hidden source labels such as 'two papers', 'both papers', 'the two approaches', 'both methods', 'the studies', 'excerpt', 'segment', 'above', 'former', or 'latter'. "
            "Output JSON only."
        )
        prompt = (
        "Generate exactly 1 self-contained two-paper literature question and 1 reference answer.\n"
            "Requirements:\n"
            "- The question must remain understandable when shown alone, without the evidence blocks.\n"
            "- The question must be retrievable by RAG: it must name concrete searchable scientific entities such as methods, sample/fraction types, compound classes, ionization modes, or phenomena.\n"
            "- Before writing the final question, mentally replace source placeholders with the actual scientific names from the evidence.\n"
            "- If your draft contains 'the two papers', 'both papers', 'Paper A', 'Paper B', 'the two approaches', 'the two methods', or publication-year anchors, rewrite it before returning JSON.\n"
            "- The question should explicitly name the compared methods / phenomena / sample types / analytical goals.\n"
            "- The question must be about a concrete method, concept, phenomenon, sample class, or analytical comparison; not about a vaguely identified paper.\n"
            "- Do not use paper anchors such as journal name, publication year, 'the study', or 'according to the paper' as the main subject.\n"
            "- Do not refer to source identity with hidden-context phrases such as 'the two papers', 'both papers', 'the two studies', 'both studies', or 'these papers'.\n"
            "- Do not use vague comparative placeholders such as 'the two approaches', 'both approaches', 'the two methods', or 'both methods'; name the actual approaches instead.\n"
            "- Do not anchor the question to publication years, e.g. 'the 2008 paper' or 'the 2012 study'.\n"
            "- Bad: 'How do the two papers characterize sulfur compoundstop'\n"
            "- Bad: 'How do the two approaches differtop'\n"
            "- Bad: 'Based on the 2008 paper on naphthenic acids and the 2012 paper, ...'\n"
            "- Good: 'How do reactive versus non-reactive sulfur species and sulfide versus thiophenic sulfur distributions help characterize petroleum fractionstop'\n"
            "- Good: 'How do HPLC-based SARA fractionation and GC-FIMS hydrocarbon-type analysis differ in the compositional information they provide for petroleum fractionstop'\n"
            "- Good: 'How do ESI FT-ICR MS and APCI/APPI ionization strategies differ in their ability to characterize acidic, basic, and nonpolar petroleum componentstop'\n"
            "- Do not start with repetitive benchmark templates such as 'Compare the analytical goals and sample types' or 'How do the analytical goals and sample types differ'.\n"
            "- Prefer one focused scientific angle over a broad inventory of goals, samples, and methods.\n"
            "- Avoid any hidden-context wording such as 'the two excerpts' or 'the studies above'.\n"
            "- The answer_key should summarize objective facts supported by the evidence blocks.\n"
            f"- Provide {kp_req} short key_points, each an atomic fact a strong answer should cover.\n"
            f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
            f"{variant_req}"
            f"{style_req}"
            f"{format_req}"
            '- If the pair does not support a valid self-contained question, return "question" as an empty string.\n\n'
            f"JSON schema:\n{schema}\n"
            f"doc_id_A: {evidence_a['doc_id']}\n"
            f"doc_id_B: {evidence_b['doc_id'] if evidence_b else ''}\n\n"
            f"paper_A_evidence:\n{evidence_a['text']}\n\n"
            f"paper_B_evidence:\n{evidence_b['text'] if evidence_b else ''}\n"
        )
    else:
        system_prompt = (
            "You are a semantic literature-QA benchmark constructor. "
            "Build a self-contained, paper-level question from one paper. "
            "Do not mention hidden excerpts or segments. "
            "Output JSON only."
        )
        prompt = (
            "Generate exactly 1 self-contained literature question and 1 reference answer from one paper.\n"
            "Requirements:\n"
            "- The question must remain understandable when shown alone.\n"
            "- The question should refer to the actual method / finding / sample / implication, not to hidden evidence fragments.\n"
            "- The question must be about a concrete method, concept, phenomenon, sample class, or analytical implication; not about a vaguely identified paper.\n"
            "- Do not use paper anchors such as journal name, publication year, 'the study', or 'according to the paper' as the main subject.\n"
            "- Avoid any wording like 'in the excerpt', 'in the text above', or 'the evidence above'.\n"
            "- The answer_key should summarize objective facts supported by the paper evidence.\n"
            f"- Provide {kp_req} short key_points capturing the core facts a strong answer should cover.\n"
            f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
            f"{style_req}"
            f"{format_req}"
            '- If the paper evidence is too weak, return "question" as an empty string.\n\n'
            f"JSON schema:\n{schema}\n"
            f"doc_id: {evidence_a['doc_id']}\n\n"
            f"paper_evidence:\n{evidence_a['text']}\n"
        )

    raw = llm_call(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt)
    return _normalize_qa_dict(extract_json_object(raw))


def generate_multi_doc_level_sample(
    *,
    llm_call: Callable[..., str],
    evidence_docs: list[dict[str, Any]],
    max_answer_chars: int,
    temperature: float,
    language: str,
    benchmark_style: str,
) -> dict[str, Any]:
    """Generate a reviewed-candidate question that requires 3+ papers."""

    if language != "en":
        language = "en"
    doc_count = len(evidence_docs)
    key_point_range = f"{max(5, doc_count + 2)}-{max(9, doc_count * 3)}"
    style_req = "- Prefer literature-derived explanatory synthesis; semantic coverage matters more than exact wording.\n"
    evidence_blocks = "\n\n".join(
        f"paper_{index}_doc_id: {item['doc_id']}\npaper_{index}_evidence:\n{item['text']}"
        for index, item in enumerate(evidence_docs, start=1)
    )
    prompt = (
        f"Generate exactly 1 self-contained literature synthesis question and reference answer from {doc_count} papers.\n"
        "Requirements:\n"
        f"- The question must require evidence from all {doc_count} papers; do not create a question answerable from only one paper.\n"
        "- Explicitly name the scientific methods, sample types, phenomena, or analytical goals being synthesized.\n"
        "- Avoid hidden-context wording such as 'the papers above', 'former', 'latter', or 'these excerpts'.\n"
        "- Do not use outside knowledge.\n"
        f"- Provide {key_point_range} atomic key_points for judging evidence coverage.\n"
        f"- Keep answer_key within {int(max_answer_chars)} characters.\n"
        f"{style_req}"
        '- If the evidence set does not support a valid question, return "question" as an empty string.\n\n'
        "Return JSON with keys question, answer_key, key_points, tags, notes.\n\n"
        f"{evidence_blocks}"
    )
    system_prompt = (
        "You are a strict petroleum-analysis benchmark constructor. "
        "Build a multi-paper QA pair from supplied paper evidence only. Output JSON only."
    )
    raw = llm_call(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt)
    return _normalize_qa_dict(extract_json_object(raw))


def review_and_rewrite_question(
    *,
    llm_call: Callable[..., str],
    question_type: str,
    qa: dict[str, Any],
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any] | None,
    benchmark_style: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    question = str(qa.get("question") or "").strip()
    if not question:
        return {
            "pass": False,
            "self_contained_score": 0.0,
            "answerable_score": 0.0,
            "objective_score": 0.0,
            "specificity_score": 0.0,
            "ambiguity_score": 1.0,
            "evidence_locality_score": 0.0,
            "reasoning_type": "unknown",
            "difficulty": "unknown",
            "hallucination_risk_score": 1.0,
            "quality_score": 0.0,
            "issues": ["empty_question"],
            "rewritten": qa,
            "notes": "empty question",
        }

    system_prompt = (
        "You are a rigorous petroleum-analysis benchmark quality auditor. "
        "Score the ORIGINAL generated QA item only; do not score a corrected version. "
        "A suggested rewrite may be provided, but it must not improve the scores. "
        "Use conservative scores: a benchmark item is high-quality only when the question, answer_key, "
        "and key_points are all directly supported by the supplied evidence. Output JSON only."
    )
    if str(benchmark_style or "").strip().lower() == "specific_fact":
        benchmark_hint = (
            "This is a specific-fact literature-QA benchmark: the item should test a narrow, "
            "objective, evidence-local fact such as a parameter, threshold, range, method setting, "
            "sample/fraction identity, ionization mode, detector choice, or exact reported finding. "
            "Broad overview questions should be penalized."
        )
    else:
        benchmark_hint = "This is a semantic literature-QA benchmark: semantic coverage matters more than exact wording."
    evidence_text = f"evidence_A:\n{evidence_a['text']}\n"
    if evidence_b:
        evidence_text += f"\n\nevidence_B:\n{evidence_b['text']}\n"
    prompt = (
        "Review the generated benchmark item. Score the ORIGINAL generated question, answer_key, and key_points only.\n"
        "Do not give perfect 1.0 scores unless the item is flawless, language-consistent, non-generic, and every important answer/key-point claim is directly supported by the supplied evidence.\n"
        "Use this stricter scoring scale for positive dimensions: 1.0 = flawless; 0.90 = strong but has a small weakness; 0.75 = acceptable benchmark item; 0.60 = marginal; 0.40 = weak; 0.0 = unusable.\n"
        "Use this stricter scoring scale for risk dimensions: 0.0 = no visible risk; 0.20 = minor risk; 0.45 = moderate risk; 0.75 = high risk; 1.0 = unusable risk.\n"
        "If there is any issue, do not assign 1.0 to the affected dimension. If issues is non-empty, quality_score should normally be <= 0.85.\n"
        "Check especially for these failure modes:\n"
        "- language mismatch: English benchmark items must have English question, answer_key, and key_points\n"
        "- paper-referential question subject: the question is framed around a vague paper/study/year/journal rather than a concrete petroleum-analysis concept, method, phenomenon, or sample class\n"
        "- hidden-context wording like 'the excerpt', 'the two excerpts', 'the two papers', 'both papers', 'the two studies', 'both studies', 'the two approaches', 'both approaches', 'the two methods', 'both methods', 'above', 'former', 'latter'\n"
        "- publication-year paper anchors such as 'the 2008 paper', 'the 2012 study', or 'Based on the 2008 paper ... and the 2012 paper ...'\n"
        "- not retrievable by RAG because the question depends on knowing which hidden papers are being referenced\n"
        "- missing subject/object, so that even an expert cannot tell what is being asked\n"
        "- not answerable from the paper-level evidence\n"
        "- answer_key or key_points include general background knowledge, extrapolation, or tangential claims not required by the question\n"
        "- over-fragmented 'snippet semantics' instead of objective scientific facts\n"
        "- low specificity, ambiguous wording, weak evidence locality, unsupported answer claims, or high hallucination risk\n"
        "- question asks about broad background knowledge rather than the provided paper-level evidence\n"
        "- answer_key contains claims that are plausible but not explicitly supported by the evidence blocks\n"
        "- for specific_fact benchmarks: broad literature-review, advantages/challenges, general implication, or open-ended synthesis questions instead of a narrow checkable fact\n"
        "- for multiple-choice questions: missing A-D options in the question text, mismatch between question options and the choices field, multiple correct options, implausible distractors, incorrect correct_option, answer_key that does not identify the correct labeled option, or key_points that contain stale option-letter labels\n"
        f"{benchmark_hint}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "pass": true/false,\n'
        '  "self_contained_score": 0-1,\n'
        '  "answerable_score": 0-1,\n'
        '  "objective_score": 0-1,\n'
        '  "specificity_score": 0-1,\n'
        '  "ambiguity_score": 0-1,\n'
        '  "evidence_locality_score": 0-1,\n'
        '  "reasoning_type": "fact_lookup|method_explanation|comparison|synthesis|causal_reasoning|property_reasoning|other",\n'
        '  "difficulty": "easy|medium|hard",\n'
        '  "hallucination_risk_score": 0-1,\n'
        '  "evidence_supports": [{"claim": "...", "evidence_id": "A|B", "supported": true/false}],\n'
        '  "issues": ["..."],\n'
        '  "rewritten_question": "...",\n'
        '  "rewritten_answer_key": "...",\n'
        '  "rewritten_key_points": ["..."],\n'
        '  "notes": "..."\n'
        "}\n\n"
        "Use ambiguity_score and hallucination_risk_score as risk scores where 0 is low risk and 1 is high risk.\n"
        "Hard caps:\n"
        "- If language is inconsistent with the requested benchmark language, set self_contained_score <= 0.50, answerable_score <= 0.60, objective_score <= 0.50, specificity_score <= 0.60, evidence_locality_score <= 0.60, hallucination_risk_score >= 0.35, quality_score <= 0.45, and pass=false.\n"
        "- If the question subject is a vague paper/study/year/journal reference rather than a concrete method, concept, phenomenon, sample class, or analytical comparison, set self_contained_score <= 0.55, specificity_score <= 0.50, evidence_locality_score <= 0.45, ambiguity_score >= 0.45, hallucination_risk_score >= 0.40, quality_score <= 0.40, and pass=false.\n"
        "- If the question says 'the two papers', 'both papers', 'the two studies', 'the two approaches', 'both approaches', 'the two methods', or similar hidden source references, set self_contained_score <= 0.50, specificity_score <= 0.50, evidence_locality_score <= 0.45, ambiguity_score >= 0.45, quality_score <= 0.40, and pass=false.\n"
        "- If the question anchors the task to publication years or vague source identities such as 'the 2008 paper' or 'the 2012 study', apply the same cap and pass=false.\n"
        "- If any central answer claim is unsupported or answers a different entity than the question asks, set answerable_score <= 0.30, evidence_locality_score <= 0.30, ambiguity_score >= 0.25, hallucination_risk_score >= 0.80, quality_score <= 0.25, and pass=false.\n"
        "- If key_points are tangential to the question or not directly supported, set specificity_score <= 0.70, evidence_locality_score <= 0.50, ambiguity_score >= 0.25, quality_score <= 0.50.\n"
        "- If the question is generic or could be answered without these evidence blocks, set specificity_score <= 0.65, evidence_locality_score <= 0.45, ambiguity_score >= 0.35, hallucination_risk_score >= 0.35, quality_score <= 0.45, and normally pass=false.\n"
        "- For specific_fact benchmarks, if the question does not ask for a specific factual item, value, range, condition, setting, named method/fraction/species, or other short checkable answer, set specificity_score <= 0.55, evidence_locality_score <= 0.45, ambiguity_score >= 0.35, quality_score <= 0.45, and pass=false.\n"
        "- For multiple-choice questions, if the question text does not include exactly four labeled options A-D, if choices/correct_option disagree with the question text, if answer_key does not state the correct labeled option, or if key_points include stale option-letter labels, set objective_score <= 0.55, specificity_score <= 0.60, quality_score <= 0.50, and pass=false.\n"
        "- If the answer depends on an unseen table/figure or references results not explicitly present in the supplied evidence text, set evidence_locality_score <= 0.55, hallucination_risk_score >= 0.30, quality_score <= 0.50.\n"
        "- If the item has only a small wording weakness, use at most 0.90 for the affected dimension and at most 0.90 for quality_score.\n"
        f"question_type: {question_type}\n"
        f"generated_question: {question}\n"
        f"generated_answer_key: {qa.get('answer_key', '')}\n"
        f"generated_key_points: {qa.get('key_points', [])}\n\n"
        f"generated_answer_format: {qa.get('answer_format', '')}\n"
        f"generated_choices: {qa.get('choices', [])}\n"
        f"generated_correct_option: {qa.get('correct_option', '')}\n\n"
        f"{evidence_text}"
    )
    raw = llm_call(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt)
    review = extract_json_object(raw)
    normalized_quality = _normalize_quality_review(review)
    passed = bool(review.get("pass", False))
    evidence_supports = review.get("evidence_supports") if isinstance(review.get("evidence_supports"), list) else []
    issues = [str(x).strip() for x in review.get("issues", []) if str(x).strip()] if isinstance(review.get("issues"), list) else []
    generated_text = " ".join(
        [
            question,
            str(qa.get("answer_key") or ""),
            " ".join(str(x) for x in qa.get("key_points", []) if str(x).strip())
            if isinstance(qa.get("key_points"), list)
            else "",
        ]
    )
    if _contains_cjk(generated_text):
        issues.append("language_mismatch")
        normalized_quality["self_contained_score"] = min(float(normalized_quality["self_contained_score"]), 0.50)
        normalized_quality["answerable_score"] = min(float(normalized_quality["answerable_score"]), 0.60)
        normalized_quality["objective_score"] = min(float(normalized_quality["objective_score"]), 0.50)
        normalized_quality["specificity_score"] = min(float(normalized_quality["specificity_score"]), 0.60)
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.60)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.35)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.45)
        passed = False
    if _has_vague_paper_reference(question):
        issue_name = "hidden_source_reference" if re.search(
            r"\b(top:the two|these two|both)\s+(top:papers|studies|articles|approaches|methods|works)\b|\bboth\s+(top:papers|studies|articles|approaches|methods|works)\b|\b(top:19|20)\d{2}\s+(top:paper|study|article)stop\b",
            question.lower(),
        ) else "vague_paper_reference_subject"
        issues.append(issue_name)
        normalized_quality["self_contained_score"] = min(float(normalized_quality["self_contained_score"]), 0.55)
        normalized_quality["specificity_score"] = min(float(normalized_quality["specificity_score"]), 0.50)
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.45)
        normalized_quality["ambiguity_score"] = max(float(normalized_quality["ambiguity_score"]), 0.45)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.40)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.40)
        passed = False
    if not evidence_supports:
        issues.append("missing_evidence_supports")
        normalized_quality["answerable_score"] = min(float(normalized_quality["answerable_score"]), 0.70)
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.70)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.30)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.70)
    elif any(isinstance(item, dict) and item.get("supported") is False for item in evidence_supports):
        issues.append("unsupported_answer_claim")
        normalized_quality["answerable_score"] = min(float(normalized_quality["answerable_score"]), 0.30)
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.30)
        normalized_quality["ambiguity_score"] = max(float(normalized_quality["ambiguity_score"]), 0.25)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.80)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.25)
        passed = False
    lowered_issues = " ".join(issues).lower()
    if any(marker in lowered_issues for marker in ("tangential", "not directly", "not required", "do not explicitly compare", "does not explicitly compare")):
        normalized_quality["specificity_score"] = min(float(normalized_quality["specificity_score"]), 0.70)
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.50)
        normalized_quality["ambiguity_score"] = max(float(normalized_quality["ambiguity_score"]), 0.25)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.50)
    if any(marker in lowered_issues for marker in ("extrapolation", "unsupported", "not explicitly supported")):
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.55)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.55)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.45)
    if any(marker in lowered_issues for marker in ("generic", "could be answered without specific evidence", "does not target the unique contributions")):
        normalized_quality["specificity_score"] = min(float(normalized_quality["specificity_score"]), 0.65)
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.45)
        normalized_quality["ambiguity_score"] = max(float(normalized_quality["ambiguity_score"]), 0.35)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.35)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.45)
        passed = False
    if any(marker in lowered_issues for marker in ("table", "figure", "not visible in the evidence", "not present in the supplied evidence text")):
        normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.55)
        normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.30)
        _recompute_quality_score(normalized_quality)
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.50)
    two_doc_specific = None
    if question_type == "two_doc" and evidence_b is not None:
        two_doc_specific = _review_two_doc_specifics(
            llm_call=llm_call,
            question=question,
            qa=qa,
            evidence_a=evidence_a,
            evidence_b=evidence_b,
            temperature=temperature,
        )
        both_docs_required = _clamp_score(two_doc_specific.get("both_docs_required_score"), 0.0)
        cross_doc_synthesis = _clamp_score(two_doc_specific.get("cross_doc_synthesis_score"), 0.0)
        doc_balance = _clamp_score(two_doc_specific.get("doc_balance_score"), 0.0)
        two_doc_pass = bool(two_doc_specific.get("two_doc_pass", False))
        failure_reason = str(two_doc_specific.get("failure_reason") or "").strip()
        if not two_doc_pass:
            if failure_reason:
                issues.append(failure_reason)
            issues.append("two_doc_requirement_not_satisfied")
            normalized_quality["answerable_score"] = min(float(normalized_quality["answerable_score"]), 0.70)
            normalized_quality["specificity_score"] = min(float(normalized_quality["specificity_score"]), 0.75)
            normalized_quality["evidence_locality_score"] = min(float(normalized_quality["evidence_locality_score"]), 0.50)
            normalized_quality["ambiguity_score"] = max(float(normalized_quality["ambiguity_score"]), 0.30)
            normalized_quality["hallucination_risk_score"] = max(float(normalized_quality["hallucination_risk_score"]), 0.35)
            _recompute_quality_score(normalized_quality)
            normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.45)
            passed = False
        elif min(both_docs_required, cross_doc_synthesis, doc_balance) < 0.85:
            normalized_quality["evidence_locality_score"] = min(
                float(normalized_quality["evidence_locality_score"]),
                max(min(both_docs_required, cross_doc_synthesis, doc_balance), 0.60),
            )
            _recompute_quality_score(normalized_quality)
            normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.85)
    if issues:
        normalized_quality["quality_score"] = min(float(normalized_quality["quality_score"]), 0.80)
    rewritten = {
        "question": str(review.get("rewritten_question") or question).strip(),
        "answer_key": str(review.get("rewritten_answer_key") or qa.get("answer_key") or "").strip(),
        "key_points": [str(x).strip() for x in review.get("rewritten_key_points", []) if str(x).strip()]
        if isinstance(review.get("rewritten_key_points"), list)
        else list(qa.get("key_points") or []),
        "tags": list(qa.get("tags") or []),
        "notes": str(qa.get("notes") or "").strip(),
        "answer_format": str(qa.get("answer_format") or "open").strip().lower(),
        "choices": list(qa.get("choices") or []) if isinstance(qa.get("choices"), list) else [],
        "correct_option": str(qa.get("correct_option") or "").strip().upper(),
    }
    if not rewritten["key_points"]:
        rewritten["key_points"] = list(qa.get("key_points") or [])
    if not rewritten["question"]:
        rewritten["question"] = question
    return {
        "pass": passed,
        **normalized_quality,
        "two_doc_required_score": _clamp_score(two_doc_specific.get("both_docs_required_score"), 0.0)
        if isinstance(two_doc_specific, dict)
        else None,
        "cross_doc_synthesis_score": _clamp_score(two_doc_specific.get("cross_doc_synthesis_score"), 0.0)
        if isinstance(two_doc_specific, dict)
        else None,
        "doc_balance_score": _clamp_score(two_doc_specific.get("doc_balance_score"), 0.0)
        if isinstance(two_doc_specific, dict)
        else None,
        "evidence_supports": evidence_supports,
        "issues": issues,
        "rewritten": rewritten,
        "notes": str(review.get("notes") or "").strip(),
    }


def review_and_rewrite_multi_doc_question(
    *,
    llm_call: Callable[..., str],
    qa: dict[str, Any],
    evidence_docs: list[dict[str, Any]],
    benchmark_style: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Review a 3+ document benchmark item before it is persisted."""

    question = str(qa.get("question") or "").strip()
    if not question:
        return {
            "pass": False,
            "self_contained_score": 0.0,
            "answerable_score": 0.0,
            "objective_score": 0.0,
            "specificity_score": 0.0,
            "ambiguity_score": 1.0,
            "evidence_locality_score": 0.0,
            "reasoning_type": "unknown",
            "difficulty": "unknown",
            "hallucination_risk_score": 1.0,
            "quality_score": 0.0,
            "issues": ["empty_question"],
            "rewritten": qa,
            "notes": "empty question",
        }
    evidence_blocks = "\n\n".join(
        f"evidence_{index} ({item['doc_id']}):\n{item['text']}"
        for index, item in enumerate(evidence_docs, start=1)
    )
    benchmark_hint = "This is a semantic literature-QA benchmark; preserve semantic synthesis while enforcing evidence support."
    prompt = (
        f"Review this {len(evidence_docs)}-paper petroleum-analysis benchmark item.\n"
        "Check for hidden-context wording, unsupported synthesis, missing scientific subject, "
        "and failure to require all supplied evidence documents.\n"
        f"{benchmark_hint}\n\n"
        "Return JSON with keys pass, self_contained_score, answerable_score, objective_score, "
        "specificity_score, ambiguity_score, evidence_locality_score, reasoning_type, difficulty, "
        "hallucination_risk_score, issues, rewritten_question, rewritten_answer_key, rewritten_key_points, notes.\n"
        "Use ambiguity_score and hallucination_risk_score as risk scores where 0 is low risk and 1 is high risk.\n\n"
        f"generated_question: {question}\n"
        f"generated_answer_key: {qa.get('answer_key', '')}\n"
        f"generated_key_points: {qa.get('key_points', [])}\n\n"
        f"{evidence_blocks}"
    )
    system_prompt = (
        "You are a strict semantic literature benchmark quality reviewer. "
        "Reject questions that are not self-contained, objective, or evidence-answerable. Output JSON only."
    )
    review = extract_json_object(llm_call(prompt=prompt, temperature=float(temperature), system_prompt=system_prompt))
    normalized_quality = _normalize_quality_review(review)
    rewritten = {
        "question": str(review.get("rewritten_question") or question).strip(),
        "answer_key": str(review.get("rewritten_answer_key") or qa.get("answer_key") or "").strip(),
        "key_points": [str(item).strip() for item in review.get("rewritten_key_points", []) if str(item).strip()]
        if isinstance(review.get("rewritten_key_points"), list)
        else list(qa.get("key_points") or []),
        "tags": list(qa.get("tags") or []),
        "notes": str(qa.get("notes") or "").strip(),
    }
    return {
        "pass": bool(review.get("pass", False)),
        **normalized_quality,
        "issues": [str(item).strip() for item in review.get("issues", []) if str(item).strip()]
        if isinstance(review.get("issues"), list)
        else [],
        "rewritten": rewritten,
        "notes": str(review.get("notes") or "").strip(),
    }
