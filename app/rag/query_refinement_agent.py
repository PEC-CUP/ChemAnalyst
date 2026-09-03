from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from app.config import get_settings


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


def _looks_like_junk(q: str) -> bool:
    q2 = (q or "").strip()
    if not q2:
        return True
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", q2)
    alnum_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/.]{2,}", q2)
    return (not cjk_tokens) and (len(alnum_tokens) < 2)


@dataclass(frozen=True, slots=True)
class RefinementResult:
    queries: list[str]
    raw: dict


class QueryRefinementAgent:
    """LLM-driven multi-query generator.

    This agent is intentionally separated from the retrieval backend:
    - It can be driven by evaluation feedback (missing_points).
    - It logs/returns its own JSON output for reproducibility.
    - Retrieval remains responsible only for ranking/aggregation, not rewriting.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.active_llm_api_key or "MISSING_API_KEY",
            base_url=settings.active_llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )
        self._model = settings.active_llm_model

    def refine(
        self,
        *,
        question: str,
        missing_points: list[str] | None = None,
        max_queries: int = 4,
    ) -> RefinementResult:
        q = (question or "").strip()
        max_q = max(1, min(int(max_queries), 8))
        missing = [str(x).strip() for x in (missing_points or []) if str(x).strip()]

        sys = (
            "You are a query refinement agent for a RAG system.\n"
            "You must output STRICT JSON only."
        )
        user = (
            "Task: generate multiple diverse retrieval queries to maximize recall.\n"
            "Return STRICT JSON ONLY:\n"
            '{ "queries": ["...","..."] }\n\n'
            f"Rules:\n"
            f"- Provide 2 to {max_q} queries.\n"
            "- Each query <= 140 characters.\n"
            "- Queries must be diverse, not just splitting the sentence.\n"
            "- Include at least 1 keyword-style query (good for BM25).\n"
            "- Include at least 1 full-sentence question (good for embeddings).\n"
            "- Preserve key domain terms/abbreviations.\n"
            "- Do not output junk tokens or single letters.\n\n"
            f"User question:\n{q}\n"
        )
        if missing:
            user += "\nEvaluation missing_points (use these to focus some queries):\n" + "\n".join(
                f"- {m}" for m in missing[:8]
            )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        payload = _extract_json((content or "").strip())

        queries = payload.get("queries")
        if not isinstance(queries, list):
            # Fail-soft: return the original question as the only query.
            return RefinementResult(queries=[q] if q else [], raw={"error": "invalid_json", "content": content})

        out: list[str] = []
        seen = set()
        for item in queries:
            if not isinstance(item, str):
                continue
            cand = item.strip()
            if not cand:
                continue
            if len(cand) > 160:
                cand = cand[:160].rstrip()
            if _looks_like_junk(cand):
                continue
            key = cand.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
            if len(out) >= max_q:
                break

        if not out and q:
            out = [q]
        return RefinementResult(queries=out, raw=payload)

