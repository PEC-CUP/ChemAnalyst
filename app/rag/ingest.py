import hashlib
import json
from pathlib import Path

from app.config import get_settings
from app.rag.chunking import chunk_text_by_paragraph
TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
from app.utils.logger import setup_logger


logger = setup_logger(__name__)
KNOWLEDGE_FILE_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES


def embed_text(text: str, dim: int):
    """
    Simple hash-based embedding for v1.
    Replace this function with a stronger embedding model later.
    """
    import numpy as np

    vector = np.zeros(dim, dtype="float32")
    tokens = [token.strip().lower() for token in text.split() if token.strip()]

    if not tokens:
        return vector

    for token in tokens:
        stable_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vector[stable_hash % dim] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _read_knowledge_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")

    if suffix in PDF_SUFFIXES:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pymupdf.") from exc

        doc = fitz.open(path)
        pages = []
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {idx}]\n{text}")
        return "\n\n".join(pages).strip()

    if suffix in DOCX_SUFFIXES:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX ingestion requires python-docx.") from exc

        doc = Document(path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    raise RuntimeError(f"Unsupported source file type: {suffix}")


def collect_documents(knowledge_dirs: list[Path]) -> list[dict]:
    documents: list[dict] = []
    seen: set[str] = set()

    for root_dir in knowledge_dirs:
        if not root_dir.exists():
            logger.warning("Knowledge source directory does not exist: %s", root_dir)
            continue
        if not root_dir.is_dir():
            logger.warning("Knowledge source path is not a directory: %s", root_dir)
            continue

        for path in root_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in KNOWLEDGE_FILE_SUFFIXES:
                continue

            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                content = _read_knowledge_file(path)
            except Exception as exc:
                logger.warning("Skip knowledge file %s: %s", path, exc)
                continue

            if not content.strip():
                logger.warning("Skip empty knowledge file: %s", path)
                continue

            documents.append({"source": str(path), "content": content})
    return documents


def ingest_knowledge() -> dict:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("RAG ingestion requires FAISS. Install dependencies with `pip install -r requirements.txt`.") from exc

    settings = get_settings()
    source_dirs = settings.knowledge_source_dirs()
    documents = collect_documents(source_dirs)

    if not documents:
        message = "No source documents were ingested. Supported source files are txt, md, json, pdf, and docx."
        logger.warning(message)
        return {"status": "warning", "message": message, "chunks": 0}

    chunks: list[str] = []
    metadata: list[dict] = []

    for doc in documents:
        for idx, chunk in enumerate(chunk_text_by_paragraph(doc["content"], settings.chunk_max_chars)):
            chunks.append(chunk)
            metadata.append({"source": doc["source"], "chunk_id": idx, "text": chunk})

    vectors = np.vstack([embed_text(chunk, settings.vector_dim) for chunk in chunks]).astype("float32")
    index = faiss.IndexFlatIP(settings.vector_dim)
    index.add(vectors)

    faiss.write_index(index, str(settings.vectorstore_dir / "knowledge.index"))
    (settings.vectorstore_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Knowledge ingested: docs=%s chunks=%s", len(documents), len(chunks))
    return {
        "status": "success",
        "message": "Ingest completed.",
        "source_dirs": [str(path) for path in source_dirs],
        "documents": len(documents),
        "chunks": len(chunks),
    }


if __name__ == "__main__":
    result = ingest_knowledge()
    print(json.dumps(result, ensure_ascii=False, indent=2))

