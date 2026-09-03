from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.llm.deepseek_client import DeepSeekClient
from app.utils.logger import setup_logger


logger = setup_logger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_score(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class CandidateChunk:
    candidate_id: str
    doc_rank: int
    chunk_rank: int
    doc_id: str
    citation: dict[str, Any]
    evidence: dict[str, Any]
    chunk: dict[str, Any]
    original_score: float
    text: str


class LLMEvidenceFilter:
    """
    QA-oriented RAG evidence gate:
    retrieve broadly, ask an LLM which chunks directly support the question,
    then pass only filtered document evidence to final synthesis.
    """

    def __init__(self, llm: DeepSeekClient) -> None:
        self.llm = llm
        self.settings = get_settings()

    def filter(
        self,
        *,
        question: str,
        evidences: list[dict[str, Any]],
        top_k: int | None = None,
        min_score: float | None = None,
        batch_size: int | None = None,
        max_chars: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = self._flatten_evidences(evidences, max_chars=max_chars or self.settings.rag_evidence_filter_max_chars)
        if not candidates:
            return [], {"enabled": True, "candidate_count": 0, "kept_count": 0, "fallback": "empty_candidates"}

        limit = max(1, int(top_k or self.settings.rag_evidence_filter_top_k))
        threshold = float(min_score if min_score is not None else self.settings.rag_evidence_filter_min_score)
        batch = max(1, int(batch_size or self.settings.rag_evidence_filter_batch_size))
        model = (self.settings.rag_evidence_filter_model or self.settings.active_llm_model).strip()
        thinking = bool(self.settings.rag_evidence_filter_thinking)

        try:
            scored = self._score_candidates(
                question=question,
                candidates=candidates,
                batch_size=batch,
                model=model,
                thinking=thinking,
            )
        except Exception as exc:
            logger.warning("LLM evidence filter failed; falling back to retriever order: %s", exc)
            kept = candidates[:limit]
            return self._regroup(kept), {
                "enabled": True,
                "model": model,
                "thinking": thinking,
                "candidate_count": len(candidates),
                "kept_count": len(kept),
                "fallback": "llm_filter_error",
                "error": str(exc),
            }

        scored.sort(
            key=lambda item: (
                -item["filter_score"],
                item["candidate"].doc_rank,
                item["candidate"].chunk_rank,
            )
        )
        selected = [item for item in scored if item["filter_score"] >= threshold and item["useful"]]
        fallback = ""
        if not selected:
            selected = scored[:limit]
            fallback = "no_chunk_passed_threshold"
        else:
            selected = selected[:limit]

        for item in selected:
            cand = item["candidate"]
            cand.chunk["llm_filter_score"] = item["filter_score"]
            cand.chunk["llm_filter_reason"] = item.get("reason", "")
            cand.chunk["llm_supported_facts"] = item.get("supported_facts", [])
        kept_candidates = [item["candidate"] for item in selected]
        filtered = self._regroup(kept_candidates)
        debug = {
            "enabled": True,
            "model": model,
            "thinking": thinking,
            "candidate_count": len(candidates),
            "kept_count": len(kept_candidates),
            "min_score": threshold,
            "top_k": limit,
            "fallback": fallback,
            "selected": [
                {
                    "id": item["candidate"].candidate_id,
                    "doc_id": item["candidate"].doc_id,
                    "filter_score": item["filter_score"],
                    "reason": item.get("reason", ""),
                    "supported_facts": item.get("supported_facts", []),
                }
                for item in selected
            ],
        }
        return filtered, debug

    def _flatten_evidences(self, evidences: list[dict[str, Any]], *, max_chars: int) -> list[CandidateChunk]:
        out: list[CandidateChunk] = []
        for doc_rank, evidence in enumerate(evidences or [], start=1):
            if not isinstance(evidence, dict):
                continue
            citation = evidence.get("citation") if isinstance(evidence.get("citation"), dict) else {}
            doc_id = str(evidence.get("doc_id") or "").strip() or f"doc_{doc_rank}"
            chunks = evidence.get("chunks") if isinstance(evidence.get("chunks"), list) else []
            for chunk_rank, chunk in enumerate(chunks, start=1):
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text") or "").strip()
                if not text:
                    continue
                out.append(
                    CandidateChunk(
                        candidate_id=f"c{len(out) + 1}",
                        doc_rank=doc_rank,
                        chunk_rank=chunk_rank,
                        doc_id=doc_id,
                        citation=dict(citation),
                        evidence=evidence,
                        chunk=chunk,
                        original_score=_safe_score(chunk.get("score"), _safe_score(evidence.get("score"))),
                        text=text[: max(128, int(max_chars))],
                    )
                )
        return out

    def _score_candidates(
        self,
        *,
        question: str,
        candidates: list[CandidateChunk],
        batch_size: int,
        model: str,
        thinking: bool,
    ) -> list[dict[str, Any]]:
        scored_by_id: dict[str, dict[str, Any]] = {}
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            prompt = self._build_prompt(question=question, candidates=batch)
            raw = self.llm.chat_with_model(
                model=model,
                query=prompt,
                system_prompt="Return strict JSON only.",
                temperature=0.0,
                thinking=thinking,
                max_tokens=1800,
            )
            data = _extract_json(raw)
            items = data.get("items") if isinstance(data.get("items"), list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("id") or "").strip()
                if not cid:
                    continue
                scored_by_id[cid] = item

        out: list[dict[str, Any]] = []
        for cand in candidates:
            item = scored_by_id.get(cand.candidate_id, {})
            score = max(0.0, min(10.0, _safe_score(item.get("score"), 0.0)))
            useful = bool(item.get("useful", score >= 6.0))
            facts = item.get("supported_facts") if isinstance(item.get("supported_facts"), list) else []
            out.append(
                {
                    "candidate": cand,
                    "filter_score": score,
                    "useful": useful,
                    "reason": str(item.get("reason") or ""),
                    "supported_facts": [str(x) for x in facts[:5] if str(x).strip()],
                }
            )
        return out

    def _build_prompt(self, *, question: str, candidates: list[CandidateChunk]) -> str:
        items = []
        for cand in candidates:
            title = cand.citation.get("title") or cand.citation.get("file_name") or ""
            doi = cand.citation.get("doi") or ""
            year = cand.citation.get("year") or ""
            items.append(
                {
                    "id": cand.candidate_id,
                    "doc_id": cand.doc_id,
                    "title": title,
                    "year": year,
                    "doi": doi,
                    "text": cand.text,
                }
            )
        return (
            "You are an evidence gate for petroleum-domain question answering.\n"
            "Score each candidate chunk for whether it directly supports answering the question.\n"
            "Use 0-10 scores:\n"
            "- 9-10: contains explicit facts, values, method details, trends, or contrasts needed by the question.\n"
            "- 6-8: clearly useful, but may cover only part of the answer.\n"
            "- 3-5: same topic but mostly background or incomplete.\n"
            "- 0-2: unrelated or not useful.\n"
            "Do not reward broad topical similarity without answer-specific facts.\n"
            "Return strict JSON only with this schema:\n"
            '{"items":[{"id":"c1","score":8,"useful":true,"supported_facts":["..."],"reason":"..."}]}\n\n'
            f"Question:\n{question}\n\n"
            f"Candidate chunks:\n{json.dumps(items, ensure_ascii=False)}"
        )

    def _regroup(self, candidates: list[CandidateChunk]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for cand in candidates:
            if cand.doc_id not in grouped:
                evidence = dict(cand.evidence)
                evidence["chunks"] = []
                evidence["llm_filter_score"] = 0.0
                grouped[cand.doc_id] = evidence
                order.append(cand.doc_id)
            chunk = dict(cand.chunk)
            chunk["llm_filter_candidate_id"] = cand.candidate_id
            grouped[cand.doc_id]["chunks"].append(chunk)
            grouped[cand.doc_id]["llm_filter_score"] = max(
                _safe_score(grouped[cand.doc_id].get("llm_filter_score")),
                _safe_score(chunk.get("llm_filter_score"), cand.original_score),
            )
        return [grouped[doc_id] for doc_id in order]
