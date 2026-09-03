from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..schemas import LoadedDocument


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def load_pdf_with_grobid(
    path: Path,
    *,
    source_name: str | None,
    source_group_hint: str | None,
    task_group_hint: str | None,
    grobid_url: str,
    timeout_seconds: int = 300,
) -> list[LoadedDocument]:
    """Parse a scholarly PDF through a running GROBID service.

    GROBID is intentionally optional. If the service or the Python HTTP client
    is unavailable, callers should fallback to the lightweight PDF extractor.
    """

    tei_xml = _request_grobid(path, grobid_url, timeout_seconds=timeout_seconds)
    root = ET.fromstring(tei_xml)
    metadata = _extract_metadata(root)
    sections = _extract_sections(root)
    if not sections:
        body_text = _join_text(root.findall(".//tei:body//tei:p", TEI_NS))
        sections = [("body", body_text)] if body_text else []

    docs: list[LoadedDocument] = []
    for idx, (section_name, text) in enumerate(sections):
        cleaned = _normalize_space(text)
        if not cleaned:
            continue
        docs.append(
            LoadedDocument(
                text=cleaned,
                source=str(path),
                file_name=path.name,
                section=section_name or f"section_{idx}",
                source_name=source_name,
                source_group_hint=source_group_hint,
                task_group_hint=task_group_hint,
                title=metadata.get("title"),
                authors=metadata.get("authors", []),
                year=metadata.get("year"),
                journal=metadata.get("journal"),
                doi=metadata.get("doi"),
                abstract=metadata.get("abstract"),
                document_type="article",
            )
        )
    return docs


def _request_grobid(path: Path, grobid_url: str, *, timeout_seconds: int = 300) -> str:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("requests is required for GROBID parsing") from exc

    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    with path.open("rb") as fh:
        last_exc: Exception | None = None
        # Some papers can be slow. Retry a couple times before giving up.
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    endpoint,
                    files={"input": (path.name, fh, "application/pdf")},
                    data={"consolidateHeader": "1", "consolidateCitations": "1"},
                    timeout=(10, int(timeout_seconds)),
                )
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_exc = exc
                if attempt >= 3:
                    break
                # reset file handle to allow re-upload
                try:
                    fh.seek(0)
                except Exception:
                    pass
        raise RuntimeError(f"GROBID request failed after retries: {last_exc}") from last_exc


def _extract_metadata(root: ET.Element) -> dict:
    title = _first_text(root, ".//tei:titleStmt/tei:title")
    abstract = _join_text(root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS))
    doi = _first_text(root, ".//tei:idno[@type='DOI']") or _first_text(root, ".//tei:idno[@type='doi']")
    journal = _first_text(root, ".//tei:monogr/tei:title")
    year = _extract_year(root)
    authors = []
    for author in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
        name = _normalize_space(" ".join(author.itertext()))
        if name and name not in authors:
            authors.append(name)
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "doi": doi,
        "abstract": _normalize_space(abstract),
    }


def _extract_sections(root: ET.Element) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for div_idx, div in enumerate(root.findall(".//tei:body/tei:div", TEI_NS), start=1):
        head = _first_text(div, "tei:head") or f"section_{div_idx}"
        text = _join_text(div.findall(".//tei:p", TEI_NS))
        if text:
            sections.append((_normalize_space(head), text))
    return sections


def _extract_year(root: ET.Element) -> int | None:
    for date in root.findall(".//tei:publicationStmt/tei:date", TEI_NS) + root.findall(".//tei:imprint/tei:date", TEI_NS):
        raw = date.attrib.get("when") or _normalize_space(" ".join(date.itertext()))
        match = re.search(r"(19|20)\d{2}", raw or "")
        if match:
            return int(match.group(0))
    return None


def _first_text(root: ET.Element, path: str) -> str | None:
    item = root.find(path, TEI_NS)
    if item is None:
        return None
    text = _normalize_space(" ".join(item.itertext()))
    return text or None


def _join_text(elements: list[ET.Element]) -> str:
    return "\n".join(_normalize_space(" ".join(element.itertext())) for element in elements)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
