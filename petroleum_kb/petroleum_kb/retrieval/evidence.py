from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ChunkMetadata, Citation, DocumentEvidence, SearchResult


def _doc_key(md: ChunkMetadata) -> str:
    # Prefer DOI, then title+year, then file_name+source.
    if md.doi:
        return f"doi:{md.doi}".strip()
    if md.title:
        y = md.year or ""
        return f"title:{md.title}::{y}".strip()
    return f"file:{md.file_name}::{md.source}".strip()


class EvidenceAggregator:
    def __init__(self, *, chunks_per_doc: int = 3) -> None:
        self.chunks_per_doc = int(chunks_per_doc)

    def aggregate(self, results: list[SearchResult], *, top_docs: int = 5) -> list[DocumentEvidence]:
        groups: dict[str, list[SearchResult]] = {}
        for r in results:
            key = _doc_key(r.metadata)
            groups.setdefault(key, []).append(r)

        evidences: list[DocumentEvidence] = []
        for doc_id, rs in groups.items():
            rs.sort(key=lambda x: x.final_score, reverse=True)
            md = rs[0].metadata
            citation = Citation(
                title=md.title,
                authors=list(md.authors or []),
                year=md.year,
                journal=md.journal,
                doi=md.doi,
                file_name=md.file_name,
                source=md.source,
            )
            evidences.append(
                DocumentEvidence(
                    doc_id=doc_id,
                    citation=citation,
                    score=float(rs[0].final_score),
                    chunks=rs[: self.chunks_per_doc],
                )
            )

        evidences.sort(key=lambda x: x.score, reverse=True)
        return evidences[: int(top_docs)]
