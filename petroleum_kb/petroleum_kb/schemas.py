from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LoadedDocument:
    text: str
    source: str
    file_name: str
    page: int | None = None
    section: str | None = None
    source_name: str | None = None
    source_group_hint: str | None = None
    task_group_hint: str | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    abstract: str | None = None
    document_type: str | None = None


@dataclass(slots=True)
class SourceRegistration:
    name: str
    path: str
    enabled: bool = True
    source_group_hint: str | None = None
    task_group_hint: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkMetadata:
    source: str
    file_name: str
    chunk_id: int
    source_group: str
    task_group: str
    entity_tags: list[str]
    method_tags: list[str]
    evidence_type: str
    confidence: float
    page: int | None = None
    section: str | None = None
    source_name: str | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    abstract: str | None = None
    document_type: str | None = None
    chunk_mode: str | None = None
    chunk_token_size: int | None = None
    chunk_overlap: int | None = None
    sentence_start: int | None = None
    sentence_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    semantic_score: float
    tag_score: float
    source_score: float
    final_score: float
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "semantic_score": self.semantic_score,
            "tag_score": self.tag_score,
            "source_score": self.source_score,
            "final_score": self.final_score,
            "debug": self.debug,
        }


@dataclass(slots=True)
class Citation:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    file_name: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentEvidence:
    doc_id: str
    citation: Citation
    score: float
    chunks: list[SearchResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "citation": self.citation.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
        }
