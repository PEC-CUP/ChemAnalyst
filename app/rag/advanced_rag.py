from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from dataclasses import asdict
from uuid import uuid4

from app.llm.deepseek_client import DeepSeekClient
from app.config import get_settings
from app.rag.evidence_filter import LLMEvidenceFilter
from app.rag.query_refinement_agent import QueryRefinementAgent


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def _format_citation(c: dict) -> str:
    title = c.get("title") or c.get("file_name") or "unknown"
    year = c.get("year") or ""
    doi = c.get("doi") or ""
    parts = [str(title)]
    if year:
        parts.append(str(year))
    if doi:
        parts.append(f"doi:{doi}")
    return " | ".join(parts)


def _build_evidence_context(evidences: list[dict], *, max_chars_per_chunk: int = 800) -> str:
    """
    Convert `search_docs` output into a compact, citation-friendly context.

    We keep doc-level citation + a few best chunks to stay within token limits.
    """

    blocks: list[str] = []
    for idx, ev in enumerate(evidences, start=1):
        doc_id = ev.get("doc_id") or f"doc_{idx}"
        citation = ev.get("citation") or {}
        chunks = ev.get("chunks") or []
        blocks.append(f"[DOC {idx}] id={doc_id}\nCITATION: {_format_citation(citation)}\nSCORE: {ev.get('score')}")
        for j, ch in enumerate(chunks, start=1):
            md = (ch.get("metadata") or {}) if isinstance(ch, dict) else {}
            sec = md.get("section") or ""
            page = md.get("page")
            loc = []
            if sec:
                loc.append(str(sec))
            if page is not None:
                loc.append(f"p.{page}")
            loc_str = (" (" + ", ".join(loc) + ")") if loc else ""
            facts = ch.get("llm_supported_facts") if isinstance(ch, dict) else None
            if isinstance(facts, list):
                facts_text = "; ".join(str(x).strip() for x in facts[:5] if str(x).strip())
                if facts_text:
                    blocks.append(f"- evidence {j} filter facts: {facts_text}")
            text = (ch.get("text") or "")[:max_chars_per_chunk]
            blocks.append(f"- evidence {j}{loc_str}: {text}")
        blocks.append("")
    return "\n".join(blocks).strip()



def compact_cited_evidences(answer: str, evidences: list[dict]) -> tuple[str, list[dict]]:
    """Keep cited documents only and renumber citations in first-use order."""

    cited_numbers: list[int] = []
    for match in re.finditer(r"\[DOC\s+(\d+)\]", answer or "", flags=re.I):
        number = int(match.group(1))
        if number not in cited_numbers:
            cited_numbers.append(number)

    mapping = {old: new for new, old in enumerate(cited_numbers, start=1)}
    compacted = [evidences[old - 1] for old in cited_numbers if 1 <= old <= len(evidences)]

    def replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        new = mapping.get(old, old)
        return f"[DOC {new}]"

    rewritten = re.sub(r"\[DOC\s+(\d+)\]", replace, answer or "", flags=re.I)
    return rewritten, compacted

def _safe_score(item: dict) -> float:
    try:
        return float(item.get("score") or 0.0)
    except Exception:
        return 0.0


def _merge_document_evidence(base: dict, incoming: dict, *, max_chunks: int) -> dict:
    """
    Merge evidence for the same document without dropping direct-hit chunks.

    Multi-query retrieval issues multiple queries. A later broad query should not
    overwrite a more precise direct-query hit for the same document.
    """

    if not isinstance(base, dict):
        return incoming
    if not isinstance(incoming, dict):
        return base

    merged = dict(base)
    if _safe_score(incoming) > _safe_score(merged):
        merged["score"] = incoming.get("score")
    if not merged.get("citation") and incoming.get("citation"):
        merged["citation"] = incoming.get("citation")

    chunks_by_id: dict[str, dict] = {}
    fallback_index = 0
    for source in (base, incoming):
        for chunk in source.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            cid = str(chunk.get("chunk_id") or "").strip()
            if not cid:
                fallback_index += 1
                cid = f"__fallback_{fallback_index}"
            prev = chunks_by_id.get(cid)
            if prev is None or _safe_score(chunk) > _safe_score(prev):
                chunks_by_id[cid] = chunk

    chunks = sorted(chunks_by_id.values(), key=_safe_score, reverse=True)
    merged["chunks"] = chunks[: max(1, int(max_chunks))]
    return merged


def _merge_evidence_lists(existing: list[dict], incoming: list[dict], *, max_chunks_per_doc: int) -> list[dict]:
    by_doc: dict[str, dict] = {}
    order: list[str] = []
    for evidence in [*(existing or []), *(incoming or [])]:
        if not isinstance(evidence, dict):
            continue
        doc_id = str(evidence.get("doc_id") or "").strip()
        if not doc_id:
            continue
        if doc_id not in by_doc:
            by_doc[doc_id] = evidence
            order.append(doc_id)
            continue
        by_doc[doc_id] = _merge_document_evidence(by_doc[doc_id], evidence, max_chunks=max_chunks_per_doc)

    ordered = [by_doc[doc_id] for doc_id in order]
    ordered.sort(key=_safe_score, reverse=True)
    return ordered


def _select_accumulated_evidences(
    existing: list[dict],
    incoming: list[dict],
    *,
    max_chunks_per_doc: int,
    max_docs: int,
) -> list[dict]:
    """
    Keep focused follow-up evidence instead of letting old broad high-score
    documents dominate the final context.

    Scores from different rewritten queries are not perfectly comparable. When
    a follow-up round searches for missing details, its documents should be
    preserved first, then earlier documents can fill the remaining context.
    """

    current = _merge_evidence_lists([], incoming, max_chunks_per_doc=max_chunks_per_doc)
    previous = _merge_evidence_lists([], existing, max_chunks_per_doc=max_chunks_per_doc)
    previous_by_doc = {
        str(ev.get("doc_id") or "").strip(): ev
        for ev in previous
        if isinstance(ev, dict) and str(ev.get("doc_id") or "").strip()
    }

    selected: list[dict] = []
    selected_ids: set[str] = set()

    for ev in current:
        doc_id = str(ev.get("doc_id") or "").strip()
        if not doc_id:
            continue
        prev = previous_by_doc.get(doc_id)
        merged = _merge_document_evidence(prev, ev, max_chunks=max_chunks_per_doc) if prev else ev
        selected.append(merged)
        selected_ids.add(doc_id)
        if len(selected) >= max(1, int(max_docs)):
            return selected

    for ev in previous:
        doc_id = str(ev.get("doc_id") or "").strip()
        if not doc_id or doc_id in selected_ids:
            continue
        selected.append(ev)
        selected_ids.add(doc_id)
        if len(selected) >= max(1, int(max_docs)):
            break

    return selected


def _combine_priority_evidences(
    priority: list[dict],
    broad: list[dict],
    *,
    max_chunks_per_doc: int,
    max_docs: int,
) -> list[dict]:
    """Return priority evidences first, then broad evidences without dropping docs."""

    selected: list[dict] = []
    by_doc: dict[str, dict] = {}

    for ev in [*(priority or []), *(broad or [])]:
        if not isinstance(ev, dict):
            continue
        doc_id = str(ev.get("doc_id") or "").strip()
        if not doc_id:
            continue
        if doc_id not in by_doc:
            by_doc[doc_id] = ev
            selected.append(ev)
        else:
            by_doc[doc_id] = _merge_document_evidence(by_doc[doc_id], ev, max_chunks=max_chunks_per_doc)
            for idx, existing in enumerate(selected):
                if str(existing.get("doc_id") or "").strip() == doc_id:
                    selected[idx] = by_doc[doc_id]
                    break
        if len(selected) >= max(1, int(max_docs)):
            break

    return selected


def _expand_selected_evidences(
    selected: list[dict],
    candidates: list[dict],
    *,
    max_chunks_per_doc: int,
    anchor_docs: int = 2,
) -> list[dict]:
    """
    Preserve answer-critical context after the LLM evidence gate.

    The filter scores individual chunks, but single-document precision QA often
    needs adjacent or later text from the same retrieved document. Once a doc is
    selected, pass through the retrieved chunks from that doc while preserving
    LLM-selected chunk annotations. A small top-retriever anchor is also kept so
    a filter miss cannot erase all direct retriever evidence.
    """

    max_chunks = max(1, int(max_chunks_per_doc))
    by_doc = {
        str(ev.get("doc_id") or "").strip(): ev
        for ev in candidates or []
        if isinstance(ev, dict) and str(ev.get("doc_id") or "").strip()
    }
    selected_by_doc = {
        str(ev.get("doc_id") or "").strip(): ev
        for ev in selected or []
        if isinstance(ev, dict) and str(ev.get("doc_id") or "").strip()
    }

    ordered_doc_ids: list[str] = []
    for ev in selected or []:
        doc_id = str(ev.get("doc_id") or "").strip()
        if doc_id and doc_id not in ordered_doc_ids:
            ordered_doc_ids.append(doc_id)
    for ev in (candidates or [])[: max(0, int(anchor_docs))]:
        doc_id = str(ev.get("doc_id") or "").strip() if isinstance(ev, dict) else ""
        if doc_id and doc_id not in ordered_doc_ids:
            ordered_doc_ids.append(doc_id)

    expanded: list[dict] = []
    for doc_id in ordered_doc_ids:
        base = by_doc.get(doc_id) or selected_by_doc.get(doc_id)
        if not isinstance(base, dict):
            continue
        merged = dict(base)
        selected_chunks = selected_by_doc.get(doc_id, {}).get("chunks") or []
        candidate_chunks = by_doc.get(doc_id, {}).get("chunks") or []
        chunks_by_id: dict[str, dict] = {}
        order: list[str] = []
        fallback_index = 0
        for chunk in [*selected_chunks, *candidate_chunks]:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                fallback_index += 1
                chunk_id = f"__fallback_{fallback_index}"
            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = dict(chunk)
                order.append(chunk_id)
            else:
                # Preserve LLM annotations from selected chunks.
                chunks_by_id[chunk_id] = {**dict(chunk), **chunks_by_id[chunk_id]}
        merged["chunks"] = [chunks_by_id[cid] for cid in order[:max_chunks]]
        expanded.append(merged)
    return expanded


def _ordered_retrieval_queries(primary: str, candidates: list[str], *, max_queries: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for query in [primary, *(candidates or [])]:
        q = str(query or "").strip()
        if not q:
            continue
        key = re.sub(r"\s+", " ", q).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= max(1, int(max_queries)):
            break
    return out


def _call_retrieve_fn(retrieve_fn, query: str, *, top_docs: int, top_k: int, chunks_per_doc: int) -> list[dict]:
    try:
        return retrieve_fn(query, top_docs=top_docs, top_k=top_k, chunks_per_doc=chunks_per_doc)
    except TypeError as exc:
        if "chunks_per_doc" not in str(exc):
            raise
        return retrieve_fn(query, top_docs=top_docs, top_k=top_k)


_DOC_REF_RE = re.compile(r"\[DOC\s+(\d+)\]")


def _extract_doc_refs(text: str) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for m in _DOC_REF_RE.finditer(text):
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue
    return out


def _strip_invalid_doc_refs(text: str, *, allowed_max: int) -> str:
    if allowed_max <= 0:
        return _DOC_REF_RE.sub("", text)

    def _replace(m: re.Match) -> str:
        try:
            n = int(m.group(1))
        except Exception:
            return ""
        return m.group(0) if 1 <= n <= allowed_max else ""

    return _DOC_REF_RE.sub(_replace, text)


def _has_invalid_doc_refs(text: str, *, allowed_max: int) -> bool:
    refs = _extract_doc_refs(text)
    return any((n <= 0 or n > allowed_max) for n in refs)



@dataclass(slots=True)
class RAGRound:
    round_index: int
    retrieval_query: str
    retrieval_queries: list[str] = field(default_factory=list)
    evidences: list[dict] = field(default_factory=list)
    draft_answer: str = ""
    evaluation: dict = field(default_factory=dict)


class IterativeReviewRAG:
    """
    Iterative evidence-grounded RAG:
      retrieve -> review/synthesize -> evaluate -> refine when needed

    This implementation intentionally keeps the rest of ChemAnalyst architecture intact:
    - Retrieval is delegated to petroleum_kb via the adapter layer.
    - LLM calls use existing DeepSeekClient configuration.
    """

    def __init__(self, llm: DeepSeekClient) -> None:
        self.llm = llm
        self.refiner = QueryRefinementAgent()

    def answer(
        self,
        *,
        question: str,
        retrieve_fn,
        max_rounds: int = 2,
        top_docs: int = 5,
        top_k_chunks: int = 12,
    ) -> dict:
        return self._answer(
            question=question,
            retrieve_fn=retrieve_fn,
            max_rounds=max_rounds,
            top_docs=top_docs,
            top_k_chunks=top_k_chunks,
            soft_filter=False,
            chain="iterative_review",
        )


    def _answer(
        self,
        *,
        question: str,
        retrieve_fn,
        max_rounds: int = 2,
        top_docs: int = 5,
        top_k_chunks: int = 12,
        soft_filter: bool = False,
        chain: str = "iterative_review",
        enable_initial_refinement: bool = True,
        retrieval_top_docs_override: int | None = None,
        retrieval_top_k_override: int | None = None,
        chunks_per_doc_override: int | None = None,
    ) -> dict:
        q = question.strip()
        rounds: list[RAGRound] = []
        accumulated: list[dict] = []
        filtered_accumulated: list[dict] = []
        run_id = uuid4().hex
        retrieval_top_docs = (
            int(retrieval_top_docs_override)
            if retrieval_top_docs_override is not None
            else max(int(top_docs or 5), 8)
        )
        retrieval_top_docs = max(1, retrieval_top_docs)
        retrieval_top_k = (
            int(retrieval_top_k_override)
            if retrieval_top_k_override is not None
            else max(int(top_k_chunks or 12), retrieval_top_docs * 6, 30)
        )
        retrieval_top_k = max(1, retrieval_top_k)
        chunks_per_doc = (
            int(chunks_per_doc_override)
            if chunks_per_doc_override is not None
            else max(3, min(4, int(top_k_chunks or 12)))
        )
        chunks_per_doc = max(1, chunks_per_doc)
        max_retrieval_queries = max(1, min(3, retrieval_top_docs))
        max_accumulated_docs = max(retrieval_top_docs + 2, 10)
        filter_top_k = max(12, retrieval_top_docs * 2)
        min_review_score = 7
        current_review = ""

        focus = ""
        for i in range(1, int(max_rounds) + 1):
            retrieval_query = q if not focus else focus
            missing_points = [p.strip() for p in focus.split(";") if p.strip()] if focus else []
            if focus:
                retrieval_queries = _ordered_retrieval_queries(
                    retrieval_query,
                    missing_points,
                    max_queries=max_retrieval_queries,
                )
            else:
                if enable_initial_refinement:
                    try:
                        refinement = self.refiner.refine(
                            question=retrieval_query,
                            missing_points=missing_points,
                            max_queries=max_retrieval_queries,
                        )
                        refined_queries = refinement.queries or []
                    except Exception:
                        refined_queries = []
                else:
                    refined_queries = []
                retrieval_queries = _ordered_retrieval_queries(
                    retrieval_query,
                    refined_queries,
                    max_queries=max_retrieval_queries,
                )

            # Multi-query retrieval: retrieve per query, then merge doc evidences by doc_id.
            round_evidences: list[dict] = []
            for rq in retrieval_queries:
                evs = _call_retrieve_fn(
                    retrieve_fn,
                    rq,
                    top_docs=retrieval_top_docs,
                    top_k=retrieval_top_k,
                    chunks_per_doc=chunks_per_doc,
                )
                round_evidences = _merge_evidence_lists(
                    round_evidences,
                    evs,
                    max_chunks_per_doc=chunks_per_doc,
                )

            filtered_evidences, filter_debug = LLMEvidenceFilter(self.llm).filter(
                question=q if not focus else f"{q}\n\nReview gap to supplement:\n{focus}",
                evidences=round_evidences,
                top_k=filter_top_k,
                min_score=5.0,
                batch_size=get_settings().rag_evidence_filter_batch_size,
                max_chars=get_settings().rag_evidence_filter_max_chars,
            )
            filter_priority_evidences = _expand_selected_evidences(
                filtered_evidences or round_evidences[:max_accumulated_docs],
                round_evidences,
                max_chunks_per_doc=chunks_per_doc,
                anchor_docs=2,
            )
            evidences = round_evidences[:max_accumulated_docs] if soft_filter else filter_priority_evidences
            accumulated = _select_accumulated_evidences(
                accumulated,
                evidences,
                max_chunks_per_doc=chunks_per_doc,
                max_docs=max_accumulated_docs,
            )
            filtered_accumulated = _combine_priority_evidences(
                filter_priority_evidences,
                filtered_accumulated,
                max_chunks_per_doc=chunks_per_doc,
                max_docs=max_accumulated_docs,
            )

            synthesis_evidences = (
                _combine_priority_evidences(
                    filtered_accumulated,
                    accumulated,
                    max_chunks_per_doc=chunks_per_doc,
                    max_docs=max_accumulated_docs,
                )
                if soft_filter
                else accumulated
            )
            context = _build_evidence_context(synthesis_evidences, max_chars_per_chunk=1800)
            if current_review:
                draft = self._update_review(
                    question=q,
                    current_review=current_review,
                    missing_aspects=missing_points,
                    context=_build_evidence_context(
                        filter_priority_evidences if soft_filter else evidences,
                        max_chars_per_chunk=1800,
                    ),
                )
            else:
                draft = self._review_synthesis(question=q, context=context)
            allowed_max = len(synthesis_evidences)
            if _has_invalid_doc_refs(draft, allowed_max=allowed_max):
                draft = self._repair_citations(question=q, answer=draft, context=context, allowed_max=allowed_max)
            evaluation = self._evaluate(question=q, answer=draft, context=context)
            evaluation["evidence_filter"] = filter_debug
            evaluation["soft_filter"] = soft_filter
            current_review = draft

            rounds.append(
                RAGRound(
                    round_index=i,
                    retrieval_query=retrieval_query,
                    retrieval_queries=retrieval_queries,
                    evidences=evidences,
                    draft_answer=draft,
                    evaluation=evaluation,
                )
            )

            score = evaluation.get("score")
            try:
                score_value = float(score)
            except Exception:
                score_value = 0.0
            missing = evaluation.get("missing_points") or []
            if evaluation.get("sufficient") is True and score_value >= min_review_score:
                break

            if isinstance(missing, list) and missing:
                focus = "; ".join(str(x) for x in missing[:5] if str(x).strip())
            else:
                followups = evaluation.get("follow_up_questions") or []
                if isinstance(followups, list) and followups:
                    focus = "; ".join(str(x) for x in followups[:3] if str(x).strip())
                else:
                    # Nothing actionable to refine.
                    break

        final_evidences = accumulated[:max_accumulated_docs]
        final_synthesis_evidences = (
            _combine_priority_evidences(
                filtered_accumulated,
                final_evidences,
                max_chunks_per_doc=chunks_per_doc,
                max_docs=max_accumulated_docs,
            )
            if soft_filter
            else final_evidences
        )
        final_context = _build_evidence_context(final_synthesis_evidences, max_chars_per_chunk=1800)
        final_answer = current_review or self._review_synthesis(question=q, context=final_context)
        allowed_max = len(final_synthesis_evidences)
        if _has_invalid_doc_refs(final_answer, allowed_max=allowed_max):
            final_answer = self._repair_citations(
                question=q,
                answer=final_answer,
                context=final_context,
                allowed_max=allowed_max,
            )
        return {
            "answer": final_answer,
            "raw_result": {
                "rounds": [asdict(r) for r in rounds],
                "evidences": final_evidences,
                "accumulated_evidences": accumulated,
                "filtered_evidences": filtered_accumulated,
                "synthesis_evidences": final_synthesis_evidences,
                "run_id": run_id,
                "retrieval_top_docs": retrieval_top_docs,
                "retrieval_top_k": retrieval_top_k,
                "chunks_per_doc": chunks_per_doc,
                "max_accumulated_docs": max_accumulated_docs,
                "chain": chain,
                "review_min_score": min_review_score,
                "soft_filter": soft_filter,
            },
        }

    def _review_synthesis(self, *, question: str, context: str) -> str:
        prompt = (
            "You are an expert scientific reviewer for petroleum-domain literature.\n"
            "Use only the provided evidence to write a literature review that directly addresses the research question.\n"
            "Requirements:\n"
            "- Cite supporting evidence with [DOC n], where n matches the document number in the context.\n"
            "- Do not invent evidence. If the evidence is insufficient, state the gap explicitly.\n"
            "- Compare methods, findings, mechanisms, limitations, and trends where the evidence supports comparison.\n"
            "- Prefer exact procedures, measured values, named methods, or explicit trends over general background.\n"
            "- Identify remaining research gaps only when they are grounded in the supplied evidence.\n"
            "- Answer in the same language as the question. For English questions, answer in English.\n\n"
            f"Question:\n{question}\n\nEvidence:\n{context}\n"
        )
        return self.llm.chat(query=prompt)

    def _evaluate(self, *, question: str, answer: str, context: str) -> dict:
        prompt = (
            "You are a strict literature review auditor for a petroleum-domain RAG system.\n"
            "Evaluate whether the review adequately addresses the research question using only the supplied evidence.\n"
            "Return strict JSON only. Do not output any additional text.\n\n"
            "JSON schema:\n"
            '{\n'
            '  "sufficient": true|false,\n'
            '  "score": 1-10,\n'
            '  "missing_points": ["..."],\n'
            '  "follow_up_questions": ["..."],\n'
            '  "notes": "..." \n'
            '}\n\n'
            "Scoring criteria:\n"
            "- 9-10: comprehensive, evidence-grounded, technically specific, and critically synthesized.\n"
            "- 7-8: good but has some missing depth, comparison, or quantitative detail.\n"
            "- 5-6: partially adequate but misses important aspects.\n"
            "- 1-4: weak, broad, unsupported, or substantially incomplete.\n"
            "Set sufficient=true only when score >= 7 and the main aspects are covered.\n"
            "Missing points must be specific searchable aspects for supplementary retrieval.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{context}\n"
        )
        raw = self.llm.chat(query=prompt)
        data = _extract_json(raw)
        # Basic normalization
        if "sufficient" not in data:
            data["sufficient"] = False
        try:
            data["score"] = max(1, min(10, int(float(data.get("score", 5)))))
        except Exception:
            data["score"] = 5
        for k in ("missing_points", "follow_up_questions"):
            if not isinstance(data.get(k), list):
                data[k] = []
        if not isinstance(data.get("notes"), str):
            data["notes"] = ""
        return data

    def _update_review(
        self,
        *,
        question: str,
        current_review: str,
        missing_aspects: list[str],
        context: str,
    ) -> str:
        aspects = [str(x).strip() for x in (missing_aspects or []) if str(x).strip()]
        prompt = (
            "You are an expert scientific reviewer. Update an existing petroleum-domain literature review by integrating new evidence.\n"
            "Use only the current review and the supplied new evidence.\n"
            "Requirements:\n"
            "- Address the missing aspects explicitly.\n"
            "- Preserve accurate useful content from the current review.\n"
            "- Integrate new evidence coherently; do not merely append disconnected notes.\n"
            "- Preserve existing valid citations and cite newly integrated claims with [DOC n] from the new evidence context.\n"
            "- Do not invent facts or unsupported citations.\n"
            "- Answer in the same language as the question. For English questions, answer in English.\n\n"
            f"Research question:\n{question}\n\n"
            f"Missing aspects:\n{json.dumps(aspects, ensure_ascii=False)}\n\n"
            f"Current review:\n{current_review}\n\n"
            f"New evidence:\n{context}\n\n"
            "Return the updated literature review only."
        )
        return self.llm.chat(query=prompt)

    def _repair_citations(self, *, question: str, answer: str, context: str, allowed_max: int) -> str:
        """
        Best-effort strict citation enforcement:
        - Only allow [DOC n] where 1 <= n <= allowed_max.
        - If a claim lacks support, mark it instead of inventing citations.
        """

        prompt = (
            "You are a citation validator for an evidence-grounded RAG system.\n"
            "Fix citations in the answer. STRICT rules:\n"
            f"- Only allow citations in the form [DOC n] where n is within 1..{allowed_max}.\n"
            "- Do NOT invent any new DOC ids.\n"
            "- If a point cannot be supported by the evidence, keep the structure but append '(evidence missing)' "
            "and avoid adding fake citations.\n"
            "Output the revised answer only.\n\n"
            f"Question:\n{question}\n\nEvidence:\n{context}\n\nOriginal answer:\n{answer}\n"
        )
        fixed = self.llm.chat(query=prompt)
        return _strip_invalid_doc_refs(fixed, allowed_max=allowed_max)

    def _final_synthesis(self, *, question: str, context: str) -> str:
        prompt = (
            "You are the final answer synthesis module for a petroleum-domain RAG system.\n"
            "Write the final answer using only the supplied evidence.\n"
            "Requirements:\n"
            "- Keep the answer concise, natural, and professional; use bullets only when they improve clarity.\n"
            "- Cite key conclusions with [DOC n].\n"
            "- Do not include content unrelated to the evidence.\n"
            "- Answer in the same language as the question. For English questions, answer in English.\n"
            "Hard constraints:\n"
            "1. Answer only what the question asks.\n"
            "2. Prefer exact evidence over broad background: numbers, units, temperatures, times, method names, compound classes, sequences, and trends.\n"
            "3. Do not use general method background or external comparisons as substitutes for the specific facts asked in the question.\n"
            "4. Do not turn a specific question into a broad review-style summary.\n"
            "5. Avoid speculative or peripheral expansion.\n"
            "6. If evidence for a sub-question is insufficient, state that briefly and do not fill the gap yourself.\n\n"
            "Style:\n"
            "- Do not write fragmented note-like phrases.\n"
            "- Use one or two short explanatory sentences when needed to connect key causal links or differences.\n"
            "- Give the direct answer first, then add only the necessary evidence support.\n\n"
            f"Question:\n{question}\n\nEvidence:\n{context}\n"
        )
        return self.llm.chat(query=prompt)


class QAOrientedRAG(IterativeReviewRAG):
    """QA-oriented RAG route: broad retrieval, evidence filtering, and final synthesis."""

    def answer(
        self,
        *,
        question: str,
        retrieve_fn,
        max_rounds: int = 1,
        top_docs: int = 5,
        top_k_chunks: int = 12,
    ) -> dict:
        del max_rounds
        q = question.strip()
        settings = get_settings()
        run_id = uuid4().hex
        retrieval_top_docs = max(int(top_docs or 0), int(settings.qa_oriented_initial_top_docs))
        retrieval_top_k = max(int(top_k_chunks or 0), int(settings.qa_oriented_initial_top_k_chunks))
        chunks_per_doc = max(1, int(settings.qa_oriented_chunks_per_doc))

        candidate_evidences = _call_retrieve_fn(
            retrieve_fn,
            q,
            top_docs=retrieval_top_docs,
            top_k=retrieval_top_k,
            chunks_per_doc=chunks_per_doc,
        )
        filtered_evidences, filter_debug = LLMEvidenceFilter(self.llm).filter(
            question=q,
            evidences=candidate_evidences,
            top_k=settings.rag_evidence_filter_top_k,
            min_score=settings.rag_evidence_filter_min_score,
            batch_size=settings.rag_evidence_filter_batch_size,
            max_chars=settings.rag_evidence_filter_max_chars,
        )
        final_evidences = _expand_selected_evidences(
            filtered_evidences or candidate_evidences[: max(1, int(settings.rag_evidence_filter_top_k))],
            candidate_evidences,
            max_chunks_per_doc=max(chunks_per_doc, 4),
            anchor_docs=2,
        )
        final_context = _build_evidence_context(final_evidences, max_chars_per_chunk=1800)
        final_answer = self._final_synthesis(question=q, context=final_context)
        allowed_max = len(final_evidences)
        if _has_invalid_doc_refs(final_answer, allowed_max=allowed_max):
            final_answer = self._repair_citations(
                question=q,
                answer=final_answer,
                context=final_context,
                allowed_max=allowed_max,
            )

        round_payload = RAGRound(
            round_index=1,
            retrieval_query=q,
            retrieval_queries=[q],
            evidences=candidate_evidences,
            draft_answer="",
            evaluation={
                "mode": "qa_oriented_evidence_filter",
                "sufficient": True,
                "evidence_filter": filter_debug,
            },
        )
        return {
            "answer": final_answer,
            "raw_result": {
                "chain": "qa_oriented",
                "rounds": [asdict(round_payload)],
                "evidences": final_evidences,
                "candidate_evidences": candidate_evidences,
                "evidence_filter": filter_debug,
                "run_id": run_id,
                "retrieval_top_docs": retrieval_top_docs,
                "retrieval_top_k": retrieval_top_k,
                "chunks_per_doc": chunks_per_doc,
            },
        }
