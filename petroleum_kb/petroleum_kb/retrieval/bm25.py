from __future__ import annotations

import hashlib
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """
    Simple bilingual tokenizer:
    - English/number tokens: words
    - CJK: split into single characters (cheap fallback)
    """

    if not text:
        return []
    tokens: list[str] = []
    # English-like tokens
    tokens.extend([t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-_/]+", text)])
    # CJK characters
    tokens.extend([c for c in re.findall(r"[\\u4e00-\\u9fff]", text)])
    return tokens


@dataclass(slots=True)
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    doc_lens: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    df: dict[str, int] = field(default_factory=dict)
    tf: list[dict[str, int]] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fields are initialized by dataclass; keep this hook for forward-compat.
        return None

    @staticmethod
    def fingerprint(doc_ids: list[str]) -> str:
        """Stable fingerprint to detect stale persisted BM25 index."""

        h = hashlib.sha256()
        for cid in doc_ids:
            h.update(cid.encode("utf-8", errors="ignore"))
            h.update(b"\n")
        return h.hexdigest()

    def build(self, *, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        self.doc_ids = list(doc_ids)
        self.tf = []
        self.df = {}
        self.doc_lens = []

        for text in texts:
            terms = _tokenize(text)
            freqs: dict[str, int] = {}
            for t in terms:
                freqs[t] = freqs.get(t, 0) + 1
            self.tf.append(freqs)
            self.doc_lens.append(len(terms))
            for term in freqs.keys():
                self.df[term] = self.df.get(term, 0) + 1

        n_docs = max(len(self.doc_ids), 1)
        self.avgdl = float(sum(self.doc_lens)) / float(n_docs)

        # Okapi BM25 IDF with +1 to keep it non-negative.
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        return self

    def save(self, path: Path, *, fingerprint: str | None = None) -> None:
        payload = {
            "version": 1,
            "fingerprint": fingerprint,
            "index": self,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=4)

    @classmethod
    def load(cls, path: Path) -> tuple["BM25Index", str | None]:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict) and isinstance(payload.get("index"), BM25Index):
            return payload["index"], payload.get("fingerprint")
        if isinstance(payload, BM25Index):
            return payload, None
        raise ValueError("Invalid BM25 pickle payload")

    def search(self, query: str, *, top_n: int = 30) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        scores: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self.doc_ids):
            freqs = self.tf[idx]
            dl = self.doc_lens[idx] or 1
            score = 0.0
            for term in q_terms:
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (float(dl) / (self.avgdl or 1.0)))
                score += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
            if score > 0:
                scores.append((doc_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores.keys()}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
