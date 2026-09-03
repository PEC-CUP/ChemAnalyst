from __future__ import annotations

import re

from ..schemas import LoadedDocument
from ..utils import normalize_text


def chunk_document(
    document: LoadedDocument,
    chunk_size: int,
    chunk_overlap: int,
    *,
    chunk_mode: str = "char",
    chunk_token_size: int = 384,
    chunk_sentence_overlap: int = 2,
    scientific_sentence_model: str = "en_core_web_sm",
    strict_scientific_sentence: bool = False,
) -> list[dict]:
    """
    Split a document into chunks.

    Supported modes:
    - char: deterministic legacy character windows.
    - sentence: lightweight multilingual sentence windows.
    - scientific_sentence: spaCy/scispaCy sentence windows with regex fallback.
    """

    text = normalize_text(document.text)
    if not text:
        return []
    mode = (chunk_mode or "char").strip().lower()
    if mode in {"sentence", "scientific_sentence"}:
        return _sentence_chunks(
            document=document,
            text=text,
            mode=mode,
            token_size=chunk_token_size,
            sentence_overlap=chunk_sentence_overlap,
            scientific_sentence_model=scientific_sentence_model,
            strict_scientific_sentence=strict_scientific_sentence,
        )
    return _char_chunks(document, text, chunk_size, chunk_overlap)


def _char_chunks(document: LoadedDocument, text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    chunks: list[dict] = []
    start = 0
    local_id = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end]
        chunks.append(
            {
                "text": chunk_text,
                "chunk_index": local_id,
                "page": document.page,
                "section": document.section,
                "source": document.source,
                "file_name": document.file_name,
                "chunk_mode": "char",
                "chunk_token_size": None,
                "chunk_overlap": chunk_overlap,
                "sentence_start": None,
                "sentence_end": None,
            }
        )
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
        local_id += 1
    return chunks


def _sentence_chunks(
    *,
    document: LoadedDocument,
    text: str,
    mode: str,
    token_size: int,
    sentence_overlap: int,
    scientific_sentence_model: str,
    strict_scientific_sentence: bool,
) -> list[dict]:
    sentences = _split_sentences(text, mode, scientific_sentence_model, strict_scientific_sentence=strict_scientific_sentence)
    if not sentences:
        return _char_chunks(document, text, 900, 150)

    chunks: list[dict] = []
    local_id = 0
    start = 0
    max_tokens = max(64, token_size)
    overlap = max(0, sentence_overlap)
    while start < len(sentences):
        token_count = 0
        end = start
        parts: list[str] = []
        while end < len(sentences):
            sentence = sentences[end]
            sentence_tokens = _count_tokens(sentence)
            if parts and token_count + sentence_tokens > max_tokens:
                break
            parts.append(sentence)
            token_count += sentence_tokens
            end += 1
        if not parts:
            parts.append(sentences[start])
            end = start + 1
        chunks.append(
            {
                "text": " ".join(parts).strip(),
                "chunk_index": local_id,
                "page": document.page,
                "section": document.section,
                "source": document.source,
                "file_name": document.file_name,
                "chunk_mode": mode,
                "chunk_token_size": max_tokens,
                "chunk_overlap": overlap,
                "sentence_start": start,
                "sentence_end": end - 1,
            }
        )
        if end >= len(sentences):
            break
        start = max(start + 1, end - overlap)
        local_id += 1
    return chunks


def _split_sentences(text: str, mode: str, scientific_sentence_model: str, *, strict_scientific_sentence: bool) -> list[str]:
    if mode == "scientific_sentence":
        if strict_scientific_sentence:
            return _split_with_spacy_strict(text, scientific_sentence_model)
        sentences = _split_with_spacy(text, scientific_sentence_model)
        if sentences:
            return sentences
        return _split_with_regex(text)
    return _split_with_regex(text)


def _split_with_spacy_strict(text: str, model_name: str) -> list[str]:
    try:
        import spacy  # type: ignore
    except Exception as exc:
        raise RuntimeError("strict_scientific_sentence enabled but spaCy is not installed.") from exc
    try:
        nlp = spacy.load(model_name, disable=["ner", "tagger", "lemmatizer"])
    except Exception as exc:
        raise RuntimeError(f"strict_scientific_sentence enabled but cannot load spaCy model: {model_name!r}") from exc
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        raise RuntimeError("strict_scientific_sentence enabled but sentence splitting returned empty.")
    return sentences


def _split_with_spacy(text: str, model_name: str) -> list[str]:
    try:
        import spacy  # type: ignore

        nlp = spacy.load(model_name, disable=["ner", "tagger", "lemmatizer"])
        if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        return []


def _split_with_regex(text: str) -> list[str]:
    # Keeps Chinese/English sentence boundaries without adding a hard dependency.
    pieces = re.split(r"(top<=[。！？!top])\s+|(top<=[.!top])\s+(top=[A-Z0-9])|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _count_tokens(text: str) -> int:
    # Lightweight proxy: English words/numbers plus individual CJK characters.
    english_tokens = re.findall(r"[A-Za-z0-9]+(top:[-_][A-Za-z0-9]+)*", text)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", text)
    return len(english_tokens) + len(cjk_tokens)
