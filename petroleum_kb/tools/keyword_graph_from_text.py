from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "if", "in", "into", "is",
    "it", "its", "may", "might", "of", "on", "or", "our", "should", "such", "than", "that",
    "the", "their", "there", "these", "this", "those", "to", "using", "used", "use", "via",
    "was", "were", "which", "while", "with", "within", "without", "would", "we", "they",
    "you", "your", "also", "however", "therefore", "thus", "during", "after", "before",
    "between", "over", "under", "toward", "towards", "because", "based", "study", "studies",
    "result", "results", "method", "methods", "analysis", "approach", "approaches",
    "paper", "work", "article", "review", "reviews",
}

DOMAIN_STOP = {
    "introduction", "conclusion", "discussion", "references", "supporting information",
    "supplementary information", "copyright", "author", "authors", "journal", "figure",
    "table", "tables", "fig", "eq", "equation",
}

GENERIC_SCIENCE_STOP = {
    "compounds", "compound", "oil", "oils", "mass", "molecular", "data", "sample", "samples",
    "ion", "ions", "number", "numbers", "species", "fraction", "fractions", "reaction", "reactions",
    "values", "different", "value", "obtained", "shown", "showed", "higher", "lower", "high", "low",
    "other", "more", "most", "each", "both", "only", "range", "class", "classes", "source", "sources",
    "chemical", "chemicals", "properties", "property", "content", "distribution", "process", "processes",
    "model", "models", "time", "information", "structures", "structure", "product", "products",
    "hydrocarbon", "hydrocarbons", "organic", "organics", "water", "weight", "relative", "separation",
    "conditions", "condition", "spectra", "spectrum", "molecules", "molecule", "formation", "abundance",
    "temperature", "carbon", "nitrogen", "sulfur", "crude", "petroleum", "gas",
}

WEAK_ACRONYM_STOP = {
    "ML", "LLM", "PDF", "LC", "HC", "OM", "IM", "CID", "ANN", "OPS", "SEC",
    "PET", "SOL", "NIR", "PCA", "RMSE", "RMSEP", "TOC", "THF", "DCM", "IMS",
    "TIMS", "NIST", "DESI", "SIMS", "GPC", "TGA", "SPE",
}

META_PHRASE_STOP = {
    "supplementary fig", "see figure", "recent years", "important role", "carried out",
    "final concentration", "base peak", "agilent technologies", "thermo scientific",
    "analytical chemistry", "energy fuels", "complex mixtures", "complex mixture",
    "wide variety", "large amount", "recent decades", "one type", "not suitable",
    "not expected", "not considered", "visual inspection", "parts per million",
    "internal standard", "air quality", "among others", "middle trace", "bottom trace",
    "fuels pubs",
}

DOMAIN_SIGNAL_WORDS = {
    "oil", "petroleum", "crude", "hydrocarbon", "hydrocarbons", "asphaltene", "asphaltenes",
    "aromatic", "aromatics", "sulfur", "sulfur-containing", "nitrogen", "oxygen",
    "naphthenic", "acid", "acids", "ionization", "chromatography", "spectrometry",
    "mass", "resonance", "diesel", "fuel", "fuels", "residue", "vacuum", "gas",
    "fraction", "fractions", "coking", "catalytic", "pyrolysis", "kinetic",
    "molecular", "double", "bond", "equivalent", "equivalents", "photoionization",
    "electrospray", "matrix", "field", "desorption", "hydrogenation", "hydrofining",
    "cracking", "biomarker", "biomarkers", "lumping", "reconstruction",
}

HIGH_VALUE_ACRONYMS = {
    "FTIR", "FTICR", "HRMS", "FCC", "petroleum fraction", "LCO", "PAH", "DOM", "TAN", "TOC",
    "HDS", "RICO", "CCS", "IMS", "TIMS", "MALDI", "ESI-MS", "GC-FID", "GC-FIMS",
    "LC-MS", "PLS", "PCA", "NMR",
}

CORE_DOMAIN_ROOTS = {
    "petroleum", "sulfur", "aromatic", "acid", "acids", "ionization", "chromatography",
    "spectrometry", "mass", "residue", "diesel", "fuel", "fuels", "cracking",
    "naphthenic", "asphaltene", "asphaltenes", "hydrogenation", "hydrofining",
}

GROUP_COLORS = {
    "ionization": "#3b82f6",
    "chromatography": "#f59e0b",
    "massspec": "#8b5cf6",
    "acids": "#ef4444",
    "process": "#10b981",
    "feedstocks": "#14b8a6",
    "modeling": "#f97316",
    "other": "#64748b",
}

ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(top:-[A-Z0-9]+)*\b")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")
PHRASE_RE = re.compile(r"\b(top:[A-Za-z][A-Za-z0-9\-]{1,}\s+){1,3}[A-Za-z][A-Za-z0-9\-]{1,}\b")
TITLE_ABSTRACT_PHRASE_RE = re.compile(
    r"\b(top:[A-Za-z][A-Za-z0-9\-]{2,}\s+){1,5}[A-Za-z][A-Za-z0-9\-]{2,}\b"
)
MIXED_TERM_TOKEN_RE = re.compile(r"[A-Za-z]{2,}(top:-[A-Za-z0-9]{1,})*|[A-Z]{2,}(top:-[A-Z0-9]+)*")


def normalize_keyword_variant(term: str) -> str:
    term = normalize_space(term)
    if not term:
        return ""

    lower = term.lower()

    exact_map = {
        "bond equivalent": "double bond equivalents (DBE)",
        "bond equivalents": "double bond equivalents (DBE)",
        "double bond equivalent": "double bond equivalents (DBE)",
        "double bond equivalents": "double bond equivalents (DBE)",
        "double bond equivalence": "double bond equivalents (DBE)",
        "double-double bond equivalents": "double bond equivalents (DBE)",
        "double bond equivalents dbe": "double bond equivalents (DBE)",
        "double double bond equivalents dbe": "double bond equivalents (DBE)",
        "dbe double bond": "double bond equivalents (DBE)",
        "electrospray ionization fourier transform": "ESI FT-ICR MS",
        "electrospray ionization fourier": "ESI FT-ICR MS",
        "chromatography-mass spectrometry": "gas chromatography-mass spectrometry",
        "appi ft-icrh": "APPI FT-ICR MS",
        "appi ft-icr": "APPI FT-ICR MS",
        "appi ft-icr ms": "APPI FT-ICR MS",
        "gc gc-tofms": "GC×GC-TOFMS",
        "gc×gc-tofms": "GC×GC-TOFMS",
        "gc-gc-tofms": "GC×GC-TOFMS",
        "esi ft-icr": "ESI FT-ICR MS",
        "esi ft-icr ms": "ESI FT-ICR MS",
        "ft-icr ms": "FT-ICR MS",
        "diesel fuels": "diesel fuel",
        "naphthenic acid": "naphthenic acids",
        "desorption electrospray": "desorption electrospray ionization",
        "field desorption": "field desorption ionization",
        "desorption ionization ldi": "laser desorption ionization",
        "laser desorption ionization ldi": "laser desorption ionization",
        "desorption ionization maldi": "matrix-assisted laser desorption ionization",
        "laser desorption ionization maldi": "matrix-assisted laser desorption ionization",
        "catalytic cracking fcc": "fluid catalytic cracking",
        "fluid catalytic cracking fcc": "fluid catalytic cracking",
    }
    if lower in exact_map:
        return exact_map[lower]

    lower = re.sub(r"\bdouble bond equivalent\b", "double bond equivalents", lower)
    lower = re.sub(r"\bbond equivalent(s)top\b", "double bond equivalents", lower)
    lower = re.sub(r"\bdouble bond equivalence\b", "double bond equivalents", lower)
    lower = re.sub(r"\bdouble[- ]double bond equivalents\b", "double bond equivalents", lower)
    lower = re.sub(r"\bdbe double bond\b", "double bond equivalents", lower)
    lower = re.sub(r"\belectrospray ionization fourier transform\b", "ESI FT-ICR MS", lower)
    lower = re.sub(r"\belectrospray ionization fourier\b", "ESI FT-ICR MS", lower)
    lower = re.sub(r"\bchromatography-mass spectrometry\b", "gas chromatography-mass spectrometry", lower)
    lower = re.sub(r"\bappi ft-icrh\b", "APPI FT-ICR MS", lower)
    lower = re.sub(r"\bappi ft-icr(top: ms)top\b", "APPI FT-ICR MS", lower)
    lower = re.sub(r"\besi ft-icr(top: ms)top\b", "ESI FT-ICR MS", lower)
    lower = re.sub(r"\bgc[× -]topgc-tofms\b", "GC×GC-TOFMS", lower)
    lower = re.sub(r"\bft-icr ms\b", "FT-ICR MS", lower)
    lower = re.sub(r"\bdiesel fuels\b", "diesel fuel", lower)
    lower = re.sub(r"\bnaphthenic acid\b", "naphthenic acids", lower)
    lower = re.sub(r"\bdesorption electrospray\b", "desorption electrospray ionization", lower)
    lower = re.sub(r"\bfield desorption\b", "field desorption ionization", lower)
    lower = re.sub(r"\bdesorption ionization ldi\b", "laser desorption ionization", lower)
    lower = re.sub(r"\blaser desorption ionization ldi\b", "laser desorption ionization", lower)
    lower = re.sub(r"\bdesorption ionization maldi\b", "matrix-assisted laser desorption ionization", lower)
    lower = re.sub(r"\blaser desorption ionization maldi\b", "matrix-assisted laser desorption ionization", lower)
    lower = re.sub(r"\bcatalytic cracking fcc\b", "fluid catalytic cracking", lower)
    lower = re.sub(r"\bfluid catalytic cracking fcc\b", "fluid catalytic cracking", lower)

    if lower in {"double bond equivalents", "double bond equivalents dbe", "double double bond equivalents dbe"}:
        return "double bond equivalents (DBE)"

    return lower


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract text-derived keywords and build a document-level keyword graph")
    parser.add_argument("--input", type=Path, required=True, help="Input chunks jsonl")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--top-doc-keywords", type=int, default=12, help="Top keywords to keep per document")
    parser.add_argument("--top-global-keywords", type=int, default=90, help="Top global keywords to keep for graph")
    parser.add_argument("--min-doc-freq", type=int, default=4, help="Minimum number of documents a keyword must appear in")
    parser.add_argument("--max-doc-freq-ratio", type=float, default=0.18, help="Drop overly common keywords above this document-frequency ratio")
    parser.add_argument("--min-edge-weight", type=int, default=3, help="Minimum document co-occurrence count for an edge")
    return parser


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def canonicalize_keyword(term: str) -> str:
    term = normalize_space(term.strip(" -_/.,:;()[]{}"))
    if not term:
        return ""
    term = normalize_keyword_variant(term)
    if term.isupper():
        return term
    return term.lower()


def canonicalize_mixed_phrase(term: str) -> str:
    parts = normalize_space(term.strip(" -_/.,:;()[]{}")).split()
    out = []
    for part in parts:
        if not part:
            continue
        if ACRONYM_RE.fullmatch(part):
            out.append(part.upper())
        else:
            out.append(part.lower())
    normalized = normalize_keyword_variant(" ".join(out))
    if normalized in {"APPI FT-ICR MS", "ESI FT-ICR MS", "GC×GC-TOFMS", "FT-ICR MS"}:
        return normalized
    return normalized.lower()


def looks_bad_term(term: str) -> bool:
    if not term:
        return True
    lower = term.lower()
    if lower in STOPWORDS or lower in DOMAIN_STOP:
        return True
    if lower in GENERIC_SCIENCE_STOP:
        return True
    parts = lower.split()
    if any(p in STOPWORDS for p in parts):
        return True
    if any(p in GENERIC_SCIENCE_STOP for p in parts):
        return True
    if lower in META_PHRASE_STOP:
        return True
    if len(parts) == 1 and len(lower) < 3:
        return True
    if len(parts) == 1 and lower.endswith(("tion", "sion", "ment", "ness", "ally")):
        return True
    if lower.isdigit():
        return True
    return False


def extract_candidates(text: str) -> list[str]:
    text = normalize_space(text)
    found: list[str] = []

    for m in ACRONYM_RE.finditer(text):
        found.append(m.group(0))

    for m in PHRASE_RE.finditer(text):
        phrase = canonicalize_keyword(m.group(0))
        words = phrase.split()
        if 2 <= len(words) <= 4 and not looks_bad_term(phrase):
            found.append(phrase)

    tokens = [canonicalize_keyword(m.group(0)) for m in TOKEN_RE.finditer(text)]
    for tok in tokens:
        if not looks_bad_term(tok):
            found.append(tok)

    return found


def extract_title_abstract_phrases(text: str) -> list[str]:
    text = normalize_space(text)
    found: list[str] = []
    for m in TITLE_ABSTRACT_PHRASE_RE.finditer(text):
        phrase = canonicalize_keyword(m.group(0))
        words = phrase.split()
        if 2 <= len(words) <= 6 and not looks_bad_term(phrase):
            signal_hits = sum(1 for w in words if w in DOMAIN_SIGNAL_WORDS)
            if sum(1 for w in words if w in STOPWORDS or w in GENERIC_SCIENCE_STOP) <= 1 and signal_hits >= 1:
                found.append(phrase)
    return found


def extract_mixed_domain_phrases(text: str) -> list[str]:
    text = normalize_space(text)
    raw_tokens = MIXED_TERM_TOKEN_RE.findall(text)
    found: list[str] = []
    n = len(raw_tokens)
    for i in range(n):
        for span in range(2, 6):
            j = i + span
            if j > n:
                break
            parts = raw_tokens[i:j]
            phrase = canonicalize_mixed_phrase(" ".join(parts))
            words = phrase.split()
            if len(words) < 2 or len(words) > 5:
                continue
            if looks_bad_term(phrase):
                continue
            signal_hits = sum(1 for w in words if w.lower() in DOMAIN_SIGNAL_WORDS)
            acronym_hits = sum(1 for w in words if ACRONYM_RE.fullmatch(w))
            if signal_hits + acronym_hits < 2:
                continue
            if sum(1 for w in words if w.lower() in STOPWORDS or w.lower() in GENERIC_SCIENCE_STOP) > 1:
                continue
            found.append(phrase)
    return found


def extract_high_value_acronyms(text: str) -> list[str]:
    text = normalize_space(text)
    found: list[str] = []
    for m in ACRONYM_RE.finditer(text):
        term = m.group(0).strip()
        if term in WEAK_ACRONYM_STOP:
            continue
        if term not in HIGH_VALUE_ACRONYMS and "-" not in term:
            continue
        if len(term) < 3 and "-" not in term:
            continue
        found.append(term)
    return found


def group_by_document(items: list[dict]) -> dict[tuple[str, str, str], dict]:
    docs: dict[tuple[str, str, str], dict] = {}
    for item in items:
        md = item.get("metadata") or {}
        title = str(md.get("title") or "").strip()
        doi = str(md.get("doi") or "").strip()
        source = str(md.get("source") or "").strip()
        key = (title, doi, source)
        row = docs.setdefault(
            key,
            {
                "title": title,
                "doi": doi,
                "journal": str(md.get("journal") or "").strip(),
                "year": md.get("year"),
                "source": source,
                "authors": md.get("authors") or [],
                "abstract": str(md.get("abstract") or "").strip(),
                "chunks": [],
            },
        )
        row["chunks"].append(str(item.get("text") or ""))
    return docs


def keyword_kind(term: str) -> str:
    if term.isupper() and len(term) >= 2:
        return "acronym"
    if " " in term:
        return "phrase"
    return "token"


def keyword_group(term: str) -> str:
    lower = term.lower()
    if any(
        token in lower
        for token in (
            "ionization",
            "electrospray",
            "photoionization",
            "desorption",
            "maldi",
            "appi",
            "esi",
            "field ionization",
            "field desorption",
        )
    ):
        return "ionization"
    if any(token in lower for token in ("chromatography", "gc", "lc", "hplc", "tofms", "fid", "gpc")):
        return "chromatography"
    if any(token in lower for token in ("ft-icr", "nmr", "ftir", "hrms", "mass spectrometry", "ccs")):
        return "massspec"
    if any(token in lower for token in ("acid", "acids", "naphthenic", "carboxylic", "formic")):
        return "acids"
    if any(
        token in lower
        for token in ("cracking", "pyrolysis", "hydrogenation", "hydrofining", "catalytic", "fcc", "hds")
    ):
        return "process"
    if any(
        token in lower
        for token in ("fuel", "fuels", "diesel", "petroleum fraction", "vacuum residue", "residue", "dom", "fossil fuels")
    ):
        return "feedstocks"
    if any(token in lower for token in ("pls", "pca", "reconstruction", "lumping", "model")):
        return "modeling"
    return "other"


def extract_document_keywords(
    docs: dict[tuple[str, str, str], dict],
    *,
    top_doc_keywords: int,
    min_doc_freq: int,
    max_doc_freq_ratio: float,
) -> tuple[list[dict], Counter[str], Counter[str]]:
    global_df: Counter[str] = Counter()
    per_doc_tf: dict[tuple[str, str, str], Counter[str]] = {}

    for key, row in docs.items():
        tf: Counter[str] = Counter()
        seen: set[str] = set()
        title = normalize_space(row["title"])
        abstract = normalize_space(row.get("abstract", ""))
        lead_text = normalize_space(" ".join([part for part in [title, abstract] if part]))

        for cand in extract_title_abstract_phrases(lead_text):
            tf[cand] += 6
            seen.add(cand)

        for cand in extract_mixed_domain_phrases(lead_text):
            tf[cand] += 7
            seen.add(cand)

        for cand in extract_high_value_acronyms(lead_text):
            tf[cand] += 2
            seen.add(cand)

        body_text = " ".join(row["chunks"])
        if len(body_text) > 120000:
            body_text = body_text[:120000]

        for cand in extract_title_abstract_phrases(body_text):
            tf[cand] += 2
            seen.add(cand)

        for cand in extract_mixed_domain_phrases(body_text):
            tf[cand] += 3
            seen.add(cand)

        for cand in extract_high_value_acronyms(body_text):
            tf[cand] += 1
            seen.add(cand)
        per_doc_tf[key] = tf
        for cand in seen:
            global_df[cand] += 1

    doc_rows: list[dict] = []
    global_score: Counter[str] = Counter()
    max_doc_freq = max(1, round(len(docs) * max_doc_freq_ratio))

    for key, row in docs.items():
        tf = per_doc_tf[key]
        scored: list[tuple[str, float]] = []
        for cand, count in tf.items():
            df = global_df[cand]
            if df < min_doc_freq:
                continue
            if df > max_doc_freq:
                continue
            kind = keyword_kind(cand)
            kind_bonus = 0.78 if kind == "acronym" else (1.45 if kind == "phrase" else 0.7)
            score = count * math.log(1 + len(docs) / df) * kind_bonus
            if kind == "phrase" and any(ch.isupper() for ch in cand):
                score *= 1.08
            scored.append((cand, score))
            global_score[cand] += score
        scored.sort(key=lambda x: (-x[1], x[0]))
        top_keywords = [cand for cand, _ in scored[:top_doc_keywords]]
        doc_rows.append(
            {
                "title": row["title"],
                "doi": row["doi"],
                "journal": row["journal"],
                "year": row["year"],
                "authors": "; ".join(str(a).strip() for a in (row.get("authors") or []) if str(a).strip()),
                "source": row["source"],
                "keyword_count": len(top_keywords),
                "keywords": top_keywords,
            }
        )

    return doc_rows, global_df, global_score


def _contains_core_root(term: str) -> bool:
    lower = term.lower()
    words = set(re.findall(r"[a-z]+", lower))
    return any(root in words for root in CORE_DOMAIN_ROOTS)


def build_graph(
    doc_rows: list[dict],
    global_df: Counter[str],
    global_score: Counter[str],
    *,
    top_global_keywords: int,
    min_edge_weight: int,
    mode: str,
) -> tuple[list[dict], list[dict]]:
    ranked_terms = []
    for kw, score in global_score.most_common():
        kind = keyword_kind(kw)
        if kind == "acronym" and kw not in HIGH_VALUE_ACRONYMS and "-" not in kw:
            continue
        if mode == "core" and not _contains_core_root(kw):
            continue
        ranked_terms.append((kw, score, kind))

    phrase_terms = [(kw, score) for kw, score, kind in ranked_terms if kind == "phrase"]
    acronym_terms = [(kw, score) for kw, score, kind in ranked_terms if kind == "acronym"]
    token_terms = [(kw, score) for kw, score, kind in ranked_terms if kind == "token"]

    if mode == "topic":
        phrase_limit = max(1, int(top_global_keywords * 0.75))
        acronym_limit = max(0, int(top_global_keywords * 0.20))
        token_limit = max(0, top_global_keywords - phrase_limit - acronym_limit)
    else:
        phrase_limit = max(1, int(top_global_keywords * 0.45))
        token_limit = max(1, int(top_global_keywords * 0.40))
        acronym_limit = max(0, top_global_keywords - phrase_limit - token_limit)

    kept = {kw for kw, _ in phrase_terms[:phrase_limit]}
    kept.update({kw for kw, _ in acronym_terms[:acronym_limit]})
    kept.update({kw for kw, _ in token_terms[:token_limit]})
    nodes = []
    for kw, score, _kind in ranked_terms[: max(top_global_keywords * 3, top_global_keywords)]:
        if kw not in kept:
            continue
        nodes.append(
            {
                "id": kw,
                "label": kw,
                "score": round(score, 4),
                "doc_freq": int(global_df[kw]),
            }
        )

    edge_counter: Counter[tuple[str, str]] = Counter()
    for row in doc_rows:
        kws = [kw for kw in row["keywords"] if kw in kept]
        kws = sorted(set(kws))
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                edge_counter[(kws[i], kws[j])] += 1

    edges = []
    for (a, b), w in edge_counter.most_common():
        if w < min_edge_weight:
            continue
        edges.append({"source": a, "target": b, "weight": int(w)})
    return nodes, edges


def write_graph_bundle(
    outdir: Path,
    *,
    doc_rows: list[dict],
    nodes: list[dict],
    edges: list[dict],
    input_path: str,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    node_ids = {node["id"] for node in nodes}
    nodes_out = []
    for node in nodes:
        enriched = dict(node)
        enriched["group"] = keyword_group(node["label"])
        enriched["color"] = GROUP_COLORS[enriched["group"]]
        nodes_out.append(enriched)

    docs_by_keyword: dict[str, list[dict]] = defaultdict(list)
    year_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in doc_rows:
        for kw in row["keywords"]:
            if kw not in node_ids:
                continue
            year_text = "" if row["year"] in (None, "", "None") else str(row["year"])
            docs_by_keyword[kw].append(
                {
                    "title": row["title"],
                    "doi": row["doi"],
                    "journal": row["journal"],
                    "year": year_text,
                    "authors": row.get("authors", ""),
                    "source": row["source"],
                }
            )
            year_counts[kw][year_text or "<missing>"] += 1

    for kw, items in docs_by_keyword.items():
        items.sort(key=lambda x: ((x.get("year") or "0000"), x.get("title") or ""), reverse=True)

    write_csv(
        outdir / "document_keywords.csv",
        [
            {
                "title": row["title"],
                "doi": row["doi"],
                "journal": row["journal"],
                "year": row["year"],
                "authors": row.get("authors", ""),
                "keyword_count": row["keyword_count"],
                "keywords": "; ".join(row["keywords"]),
                "source": row["source"],
            }
            for row in doc_rows
        ],
        ["title", "doi", "journal", "year", "authors", "keyword_count", "keywords", "source"],
    )
    write_csv(outdir / "keyword_nodes.csv", nodes_out, ["id", "label", "score", "doc_freq", "group", "color"])
    write_csv(outdir / "keyword_edges.csv", edges, ["source", "target", "weight"])

    graph_json = {
        "nodes": nodes_out,
        "edges": edges,
        "docs_by_keyword": docs_by_keyword,
        "year_counts": {
            kw: [{"year": year, "count": count} for year, count in sorted(counter.items())]
            for kw, counter in year_counts.items()
        },
    }
    (outdir / "keyword_graph.json").write_text(json.dumps(graph_json, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "documents": len(doc_rows),
        "graph_nodes": len(nodes_out),
        "graph_edges": len(edges),
        "input": input_path,
    }
    (outdir / "keyword_graph.html").write_text(
        render_html(nodes_out, edges, stats, docs_by_keyword, year_counts, doc_rows),
        encoding="utf-8",
    )
    return stats


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_html(
    nodes: list[dict],
    edges: list[dict],
    stats: dict[str, object],
    docs_by_keyword: dict[str, list[dict]],
    year_counts: dict[str, Counter[str]],
    doc_rows: list[dict],
) -> str:
    corpus_year_counts: Counter[str] = Counter()
    document_keyword_rows: list[dict] = []
    for row in doc_rows:
        year = str(row.get("year") or "")
        if year:
            corpus_year_counts[year] += 1
        document_keyword_rows.append(
            {
                "title": row.get("title", ""),
                "doi": row.get("doi", ""),
                "journal": row.get("journal", ""),
                "year": year,
                "authors": row.get("authors", ""),
                "keyword_count": row.get("keyword_count", 0),
                "keywords": row.get("keywords", []),
                "source": row.get("source", ""),
            }
        )
    document_keyword_rows.sort(
        key=lambda row: (str(row.get("year") or ""), str(row.get("title") or "")),
        reverse=True,
    )
    payload = json.dumps(
        {
            "nodes": nodes,
            "edges": edges,
            "stats": stats,
            "docs_by_keyword": docs_by_keyword,
            "year_counts": {
                kw: [{"year": year, "count": count} for year, count in sorted(counter.items())]
                for kw, counter in year_counts.items()
            },
            "corpus_year_counts": [
                {"year": year, "count": count}
                for year, count in sorted(corpus_year_counts.items())
            ],
            "document_keyword_rows": document_keyword_rows,
        },
        ensure_ascii=False,
    )
    help_text = json.dumps(
        "Click a node to inspect neighbors.\n"
        "Drag nodes to rearrange.\n"
        "Wheel to zoom. Drag blank space to pan.",
        ensure_ascii=False,
    )
    hint_text = html.escape(
        "Current graph uses text-derived keywords aggregated at document level, "
        "not chunk-level co-occurrence. This reduces the “hairball” effect and "
        "makes keyword relations easier to interpret."
    )
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Text-derived Keyword Graph</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f4f7fb; color: #16202a; }
    .wrap { padding: 18px; }
    .top { display: grid; grid-template-columns: 1fr 380px; gap: 14px; align-items: start; }
    .panel { background: #fff; border: 1px solid #d6deea; border-radius: 12px; padding: 12px 14px; }
    .meta { margin-bottom: 10px; font-size: 14px; color: #445; }
    .controls { display: flex; gap: 8px; margin-bottom: 10px; }
    .controls input { flex: 1; padding: 8px 10px; border: 1px solid #c7d3e1; border-radius: 8px; }
    .controls button { padding: 8px 12px; border: 1px solid #bdd0e3; background: #edf5fc; border-radius: 8px; cursor: pointer; }
    #graph { width: 100%; height: 82vh; min-height: 720px; background: #fff; border: 1px solid #d6deea; border-radius: 12px; touch-action: none; display: block; }
    .edge { stroke: #90a8bf; stroke-opacity: 0.22; }
    .edge.active { stroke: #4e79a7; stroke-opacity: 0.85; }
    .node { stroke: #ffffff; stroke-width: 1.2px; cursor: pointer; }
    .node.dimmed { opacity: 0.18; }
    .edge.dimmed { opacity: 0.06; }
    .node-label { font-size: 11px; fill: #17324d; pointer-events: none; opacity: 0; }
    .node-label.visible { opacity: 1; }
    .node-label.default-visible { opacity: 0.9; }
    .node-label.dimmed { opacity: 0.08; }
    .hint { font-size: 12px; color: #678; line-height: 1.5; }
    .detail h3 { margin: 0 0 8px; font-size: 15px; }
    .detail-box { white-space: normal; font-size: 12px; line-height: 1.45; margin: 0; }
    .detail-box h4 { margin: 12px 0 6px; font-size: 13px; }
    .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 8px 0 10px; }
    .stat-card { border: 1px solid #e3ebf5; border-radius: 10px; padding: 8px; background: #f8fbff; }
    .stat-value { font-weight: 700; font-size: 18px; color: #14324d; }
    .stat-label { color: #64748b; font-size: 11px; margin-top: 2px; }
    .method-list { margin: 6px 0 10px 16px; padding: 0; color: #334155; }
    .method-list li { margin: 4px 0; }
    .source-links { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }
    .source-links a { color: #1d4ed8; background: #eef6ff; border: 1px solid #dbeafe; border-radius: 999px; padding: 4px 8px; text-decoration: none; }
    .table-scroll { max-height: 220px; overflow: auto; border: 1px solid #e5edf6; border-radius: 8px; }
    .mini-table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .mini-table th, .mini-table td { border-bottom: 1px solid #e5edf6; padding: 4px 3px; text-align: left; vertical-align: top; }
    .mini-table th { color: #475569; font-weight: 700; background: #f8fbff; position: sticky; top: 0; }
    .year-bars { display: grid; gap: 6px; margin: 8px 0 10px; }
    .overview-year-bars { max-height: 190px; overflow-y: auto; padding-right: 4px; border: 1px solid #e5edf6; border-radius: 8px; padding: 6px; }
    .year-row { display: grid; grid-template-columns: 52px 1fr 40px; gap: 8px; align-items: center; }
    .year-bar { height: 8px; background: #dce8f5; border-radius: 999px; overflow: hidden; }
    .year-bar-fill { height: 100%; background: #4e79a7; border-radius: 999px; }
    .doc-list { display: grid; gap: 8px; max-height: 54vh; overflow: auto; padding-right: 4px; }
    .doc-item { border: 1px solid #e3ebf5; border-radius: 10px; padding: 8px 10px; background: #fafcff; }
    .doc-title { font-weight: 600; color: #14324d; }
    .doc-meta { color: #5b6d80; margin-top: 2px; }
    .doc-authors { color: #334155; margin-top: 4px; }
    .doc-doi a { color: #1d4ed8; text-decoration: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="meta">
          docs=__DOCS__ |
          nodes=__NODES__ |
          edges=__EDGES__
        </div>
        <div class="controls">
          <input id="search" placeholder="Search keyword..." />
          <button id="resetBtn">Reset</button>
        </div>
        <svg id="graph" viewBox="0 0 1600 980"></svg>
      </div>
      <div class="panel detail">
        <h3>Keyword Detail</h3>
        <div id="detailBox" class="detail-box">Loading graph...</div>
        <div class="hint" style="margin-top:10px;">
          __HINT_TEXT__
        </div>
      </div>
    </div>
  </div>
  <script>
    var payload = __PAYLOAD__;
    var defaultDetailText = __HELP_TEXT__;
    var svg = document.getElementById("graph");
    var searchInput = document.getElementById("search");
    var resetBtn = document.getElementById("resetBtn");
    var detailBox = document.getElementById("detailBox");
    var docsByKeyword = payload.docs_by_keyword || {};
    var yearCounts = payload.year_counts || {};
    var corpusYearCounts = payload.corpus_year_counts || [];
    var documentKeywordRows = payload.document_keyword_rows || [];
    var width = 1600, height = 980, cx = width / 2, cy = height / 2;
    var nodes = [];
    var edges = payload.edges || [];
    var i, j;
    var maxFreq = 1;
    var selectedNode = null;
    var transform = {x: 0, y: 0, k: 1};
    var draggingNode = null;
    var panning = false;
    var panStart = null;
    var defaultLabelCount = 14;

    for (i = 0; i < payload.nodes.length; i++) {
      var base = payload.nodes[i];
      var node = {
        id: base.id,
        label: base.label,
        score: base.score,
        doc_freq: base.doc_freq,
        group: base.group || "other",
        color: base.color || "#2a79b8",
        idx: i,
        vx: 0,
        vy: 0
      };
      nodes.push(node);
      if (node.doc_freq > maxFreq) maxFreq = node.doc_freq;
    }

    for (i = 0; i < nodes.length; i++) {
      var angle = (2 * Math.PI * i) / Math.max(1, nodes.length);
      var ring = 150 + (i % 8) * 22;
      nodes[i].x = cx + Math.cos(angle) * ring;
      nodes[i].y = cy + Math.sin(angle) * ring;
      nodes[i].r = 8 + 16 * Math.sqrt(nodes[i].doc_freq / maxFreq);
    }

    nodes.sort(function(a, b) { return b.doc_freq - a.doc_freq; });
    for (i = 0; i < nodes.length; i++) {
      nodes[i].showByDefault = i < defaultLabelCount;
    }
    nodes.sort(function(a, b) { return a.idx - b.idx; });

    var nodeMap = {};
    for (i = 0; i < nodes.length; i++) {
      nodeMap[nodes[i].id] = nodes[i];
    }
    for (i = 0; i < edges.length; i++) {
      edges[i].a = nodeMap[edges[i].source];
      edges[i].b = nodeMap[edges[i].target];
    }

    var adjacency = {};
    for (i = 0; i < nodes.length; i++) {
      adjacency[nodes[i].id] = {};
    }
    for (i = 0; i < edges.length; i++) {
      var e0 = edges[i];
      if (e0.a && e0.b) {
        adjacency[e0.a.id][e0.b.id] = true;
        adjacency[e0.b.id][e0.a.id] = true;
      }
    }

    var viewport = document.createElementNS("http://www.w3.org/2000/svg", "g");
    var edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    var nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    viewport.appendChild(edgeLayer);
    viewport.appendChild(nodeLayer);
    svg.appendChild(viewport);

    var edgeEls = [];
    var nodeEls = {};
    var labelEls = {};

    function applyTransform() {
      viewport.setAttribute("transform", "translate(" + transform.x + "," + transform.y + ") scale(" + transform.k + ")");
    }

    function esc(text) {
      return String(text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;");
    }

    function renderDetail(node) {
      if (!node) {
        detailBox.innerHTML = renderOverview();
        return;
      }
      var ids = [];
      for (var key in adjacency[node.id]) {
        if (adjacency[node.id].hasOwnProperty(key)) ids.push(key);
      }
      var nbrs = [];
      for (i = 0; i < ids.length; i++) {
        var n = nodeMap[ids[i]];
        if (n) nbrs.push(n);
      }
      nbrs.sort(function(a, b) { return b.doc_freq - a.doc_freq; });
      if (nbrs.length > 12) nbrs = nbrs.slice(0, 12);
      var docs = docsByKeyword[node.id] || [];
      var years = yearCounts[node.id] || [];
      var maxYearCount = 1;
      for (i = 0; i < years.length; i++) if (years[i].count > maxYearCount) maxYearCount = years[i].count;
      var yearHtml = "";
      for (i = 0; i < years.length; i++) {
        var yr = years[i];
        var pct = Math.max(6, Math.round((yr.count / maxYearCount) * 100));
        yearHtml += '<div class=\"year-row\"><div>' + esc(yr.year) + '</div><div class=\"year-bar\"><div class=\"year-bar-fill\" style=\"width:' + pct + '%\"></div></div><div>' + esc(yr.count) + '</div></div>';
      }
      var nbrHtml = "";
      for (i = 0; i < nbrs.length; i++) {
        nbrHtml += '<div>' + esc(nbrs[i].label) + ' (' + esc(nbrs[i].doc_freq) + ' docs)</div>';
      }
      var docHtml = "";
      for (i = 0; i < docs.length; i++) {
        var d = docs[i];
        var doiLink = d.doi ,  ('<div class=\"doc-doi\"><a href=\"https://doi.org/' + esc(d.doi) + '\" target=\"_blank\">' + esc(d.doi) + '</a></div>') : '';
        var authors = d.authors ,  ('<div class=\"doc-authors\">' + esc(d.authors) + '</div>') : '';
        docHtml += '<div class=\"doc-item\">' +
          '<div class=\"doc-title\">' + esc(d.title) + '</div>' +
          '<div class=\"doc-meta\">' + esc(d.year || '') + (d.journal ,  (' | ' + esc(d.journal)) : '') + '</div>' +
          authors + doiLink +
          '</div>';
      }
      detailBox.innerHTML =
        '<div><strong>keyword:</strong> ' + esc(node.label) + '</div>' +
        '<div><strong>group:</strong> ' + esc(node.group) + '</div>' +
        '<div><strong>document frequency:</strong> ' + esc(node.doc_freq) + '</div>' +
        '<div><strong>score:</strong> ' + esc(node.score) + '</div>' +
        '<h4>Year Distribution</h4>' +
        '<div class=\"year-bars\">' + (yearHtml || '<div>(none)</div>') + '</div>' +
        '<h4>Top Neighbors</h4>' +
        '<div>' + (nbrHtml || '<div>(none)</div>') + '</div>' +
        '<h4>Source Documents</h4>' +
        '<div class=\"doc-list\">' + (docHtml || '<div>(none)</div>') + '</div>';
    }

    function renderOverview() {
      var nodeList = nodes.slice().sort(function(a, b) { return b.doc_freq - a.doc_freq; }).slice(0, 12);
      var edgeList = edges.slice().sort(function(a, b) { return b.weight - a.weight; }).slice(0, 12);
      var docList = documentKeywordRows.slice(0, 10);
      var maxYearCount = 1;
      var k;
      for (k = 0; k < corpusYearCounts.length; k++) {
        if (corpusYearCounts[k].count > maxYearCount) maxYearCount = corpusYearCounts[k].count;
      }
      var yearHtml = "";
      for (k = 0; k < corpusYearCounts.length; k++) {
        var y = corpusYearCounts[k];
        var pct = Math.max(5, Math.round((y.count / maxYearCount) * 100));
        yearHtml += '<div class=\"year-row\"><div>' + esc(y.year) + '</div><div class=\"year-bar\"><div class=\"year-bar-fill\" style=\"width:' + pct + '%\"></div></div><div>' + esc(y.count) + '</div></div>';
      }
      var nodeRows = "";
      for (k = 0; k < nodeList.length; k++) {
        var n = nodeList[k];
        nodeRows += '<tr><td>' + esc(n.label) + '</td><td>' + esc(n.group) + '</td><td>' + esc(n.doc_freq) + '</td></tr>';
      }
      var edgeRows = "";
      for (k = 0; k < edgeList.length; k++) {
        var e = edgeList[k];
        edgeRows += '<tr><td>' + esc(e.source) + '</td><td>' + esc(e.target) + '</td><td>' + esc(e.weight) + '</td></tr>';
      }
      var docRows = "";
      for (k = 0; k < docList.length; k++) {
        var d = docList[k];
        var doi = d.doi ,  '<a href=\"https://doi.org/' + esc(d.doi) + '\" target=\"_blank\">' + esc(d.doi) + '</a>' : '';
        var kws = d.keywords && d.keywords.length ,  d.keywords.slice(0, 5).join('; ') : '';
        docRows += '<tr><td>' + esc(d.year || '') + '</td><td>' + esc(d.title || '') + '</td><td>' + esc(kws) + '</td><td>' + doi + '</td></tr>';
      }
      return '' +
        '<div class=\"stat-grid\">' +
        '<div class=\"stat-card\"><div class=\"stat-value\">' + esc(payload.stats.documents) + '</div><div class=\"stat-label\">source papers</div></div>' +
        '<div class=\"stat-card\"><div class=\"stat-value\">' + esc(payload.stats.graph_nodes) + '</div><div class=\"stat-label\">keyword nodes</div></div>' +
        '<div class=\"stat-card\"><div class=\"stat-value\">' + esc(payload.stats.graph_edges) + '</div><div class=\"stat-label\">co-occurrence edges</div></div>' +
        '</div>' +
        '<h4>Extraction and graph construction</h4>' +
        '<ul class=\"method-list\">' +
        '<li>Keywords are extracted from cleaned full-text chunks, with title and abstract terms prioritized and generic scientific terms filtered.</li>' +
        '<li>Term variants and high-value acronyms are normalized before scoring.</li>' +
        '<li>Nodes are document-level keywords retained by global score and document frequency.</li>' +
        '<li>Edges are document-level keyword co-occurrences; edge weight is the number of papers containing both terms.</li>' +
        '</ul>' +
        '<div class=\"source-links\">' +
        '<a href=\"keyword_nodes.csv\" target=\"_blank\">keyword_nodes.csv</a>' +
        '<a href=\"keyword_edges.csv\" target=\"_blank\">keyword_edges.csv</a>' +
        '<a href=\"document_keywords.csv\" target=\"_blank\">document_keywords.csv</a>' +
        '<a href=\"keyword_graph.json\" target=\"_blank\">keyword_graph.json</a>' +
        '</div>' +
        '<h4>Corpus year distribution</h4>' +
        '<div class=\"year-bars overview-year-bars\">' + (yearHtml || '<div>(none)</div>') + '</div>' +
        '<h4>Top keyword nodes</h4>' +
        '<div class=\"table-scroll\"><table class=\"mini-table\"><thead><tr><th>keyword</th><th>group</th><th>docs</th></tr></thead><tbody>' + nodeRows + '</tbody></table></div>' +
        '<h4>Top co-occurrence edges</h4>' +
        '<div class=\"table-scroll\"><table class=\"mini-table\"><thead><tr><th>source</th><th>target</th><th>weight</th></tr></thead><tbody>' + edgeRows + '</tbody></table></div>' +
        '<h4>Recent source documents</h4>' +
        '<div class=\"table-scroll\"><table class=\"mini-table\"><thead><tr><th>year</th><th>title</th><th>keywords</th><th>DOI</th></tr></thead><tbody>' + docRows + '</tbody></table></div>' +
        '<h4>Interaction</h4><div>' + esc(defaultDetailText).replace(/\\n/g, '<br>') + '</div>';
    }

    function hasClass(el, cls) {
      return (" " + el.getAttribute("class") + " ").indexOf(" " + cls + " ") >= 0;
    }
    function addClass(el, cls) {
      if (!hasClass(el, cls)) el.setAttribute("class", (el.getAttribute("class") + " " + cls).replace(/\\s+/g, " ").replace(/^\\s+|\\s+$/g, ""));
    }
    function removeClass(el, cls) {
      el.setAttribute("class", (" " + el.getAttribute("class") + " ").replace(" " + cls + " ", " ").replace(/\\s+/g, " ").replace(/^\\s+|\\s+$/g, ""));
    }

    function updateHighlight() {
      var selected = selectedNode ,  selectedNode.id : null;
      for (i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        var circle = nodeEls[node.id];
        var label = labelEls[node.id];
        var neighbor = selected && adjacency[selected] && adjacency[selected][node.id];
        var active = !selected || node.id === selected || neighbor;
        var showLabel = node.showByDefault || (!!selected && (node.id === selected || neighbor));
        if (active) {
          removeClass(circle, "dimmed");
        } else {
          addClass(circle, "dimmed");
        }
        removeClass(label, "visible");
        removeClass(label, "default-visible");
        removeClass(label, "dimmed");
        if (node.showByDefault) addClass(label, "default-visible");
        if (!!selected && (node.id === selected || neighbor)) addClass(label, "visible");
        if (!active) addClass(label, "dimmed");
        circle.setAttribute("fill", node.id === selected ,  "#e15759" : (neighbor ,  "#59a14f" : node.color));
      }
      for (i = 0; i < edgeEls.length; i++) {
        var item = edgeEls[i];
        var edge = item.edge;
        var el0 = item.el;
        var isActive = !selected || edge.source === selected || edge.target === selected;
        if (isActive) {
          removeClass(el0, "dimmed");
        } else {
          addClass(el0, "dimmed");
        }
        if (selected && isActive) addClass(el0, "active"); else removeClass(el0, "active");
      }
      renderDetail(selectedNode);
    }

    function runSimulation() {
      var minX = 120, maxX = width - 120, minY = 100, maxY = height - 100;
      for (var step = 0; step < 220; step++) {
        for (i = 0; i < nodes.length; i++) {
          for (j = i + 1; j < nodes.length; j++) {
            var a = nodes[i], b = nodes[j];
            var dx = b.x - a.x;
            var dy = b.y - a.y;
            var d2 = dx * dx + dy * dy + 0.01;
            var dist = Math.sqrt(d2);
            var force = 1800 / d2;
            var fx = (dx / dist) * force;
            var fy = (dy / dist) * force;
            a.vx -= fx; a.vy -= fy;
            b.vx += fx; b.vy += fy;
          }
        }
        for (i = 0; i < edges.length; i++) {
          var e1 = edges[i];
          if (!e1.a || !e1.b) continue;
          var dx2 = e1.b.x - e1.a.x;
          var dy2 = e1.b.y - e1.a.y;
          var dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) || 1;
          var desired = 62 + Math.min(140, e1.weight * 5);
          var kk = 0.0028 * e1.weight;
          var ff = (dist2 - desired) * kk;
          var fx2 = (dx2 / dist2) * ff;
          var fy2 = (dy2 / dist2) * ff;
          e1.a.vx += fx2; e1.a.vy += fy2;
          e1.b.vx -= fx2; e1.b.vy -= fy2;
        }
        for (i = 0; i < nodes.length; i++) {
          var nn = nodes[i];
          if (draggingNode && draggingNode.id === nn.id) continue;
          nn.vx += (cx - nn.x) * 0.0016;
          nn.vy += (cy - nn.y) * 0.0016;
          nn.vx *= 0.84;
          nn.vy *= 0.84;
          nn.x += nn.vx;
          nn.y += nn.vy;
          if (nn.x < minX) { nn.x = minX; nn.vx *= -0.25; }
          if (nn.x > maxX) { nn.x = maxX; nn.vx *= -0.25; }
          if (nn.y < minY) { nn.y = minY; nn.vy *= -0.25; }
          if (nn.y > maxY) { nn.y = maxY; nn.vy *= -0.25; }
        }
      }
    }

    function normalizeLayout() {
      if (!nodes.length) return;
      var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (i = 0; i < nodes.length; i++) {
        var n0 = nodes[i];
        if (n0.x < minX) minX = n0.x;
        if (n0.y < minY) minY = n0.y;
        if (n0.x > maxX) maxX = n0.x;
        if (n0.y > maxY) maxY = n0.y;
      }
      var boxW = Math.max(1, maxX - minX);
      var boxH = Math.max(1, maxY - minY);
      var marginX = 180;
      var marginY = 130;
      var scale = Math.min((width - marginX * 2) / boxW, (height - marginY * 2) / boxH);
      scale = Math.max(0.72, Math.min(1.45, scale));
      var centerX = (minX + maxX) / 2;
      var centerY = (minY + maxY) / 2;
      for (i = 0; i < nodes.length; i++) {
        var n1 = nodes[i];
        n1.x = cx + (n1.x - centerX) * scale;
        n1.y = cy + (n1.y - centerY) * scale;
      }
    }

    function redraw() {
      for (i = 0; i < edgeEls.length; i++) {
        var item2 = edgeEls[i];
        if (!item2.edge.a || !item2.edge.b) continue;
        item2.el.setAttribute("x1", item2.edge.a.x);
        item2.el.setAttribute("y1", item2.edge.a.y);
        item2.el.setAttribute("x2", item2.edge.b.x);
        item2.el.setAttribute("y2", item2.edge.b.y);
      }
      for (i = 0; i < nodes.length; i++) {
        var node2 = nodes[i];
        nodeEls[node2.id].setAttribute("cx", node2.x);
        nodeEls[node2.id].setAttribute("cy", node2.y);
        labelEls[node2.id].setAttribute("x", node2.x + node2.r + 4);
        labelEls[node2.id].setAttribute("y", node2.y + 4);
      }
    }

    for (i = 0; i < edges.length; i++) {
      if (!edges[i].a || !edges[i].b) continue;
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "edge");
      line.setAttribute("stroke-width", String(Math.max(1, Math.log(edges[i].weight + 1) / Math.log(2))));
      edgeLayer.appendChild(line);
      edgeEls.push({el: line, edge: edges[i]});
    }

    function toGraphPoint(clientX, clientY) {
      var pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      var inv = viewport.getScreenCTM().inverse();
      return pt.matrixTransform(inv);
    }

    for (i = 0; i < nodes.length; i++) {
      (function(n) {
        var c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("class", "node");
        c.setAttribute("fill", n.color);
        c.setAttribute("r", n.r);
        c.onmousedown = function(evt) {
          if (evt.stopPropagation) evt.stopPropagation();
          draggingNode = n;
        };
        c.onclick = function(evt) {
          if (evt.stopPropagation) evt.stopPropagation();
          selectedNode = (selectedNode && selectedNode.id === n.id) ,  null : n;
          updateHighlight();
        };
        var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("class", "node-label");
        t.setAttribute("text-anchor", "start");
        t.appendChild(document.createTextNode(n.label + " (" + n.doc_freq + ")"));
        nodeLayer.appendChild(c);
        nodeLayer.appendChild(t);
        nodeEls[n.id] = c;
        labelEls[n.id] = t;
      })(nodes[i]);
    }

    svg.onclick = function() {
      selectedNode = null;
      updateHighlight();
    };
    svg.onmousedown = function(evt) {
      if (evt.target !== svg) return;
      panning = true;
      panStart = {x: evt.clientX, y: evt.clientY, tx: transform.x, ty: transform.y};
    };
    svg.onmousemove = function(evt) {
      if (draggingNode) {
        var p = toGraphPoint(evt.clientX, evt.clientY);
        draggingNode.x = p.x;
        draggingNode.y = p.y;
        draggingNode.vx = 0;
        draggingNode.vy = 0;
        redraw();
        return;
      }
      if (!panning || !panStart) return;
      transform.x = panStart.tx + (evt.clientX - panStart.x);
      transform.y = panStart.ty + (evt.clientY - panStart.y);
      applyTransform();
    };
    svg.onmouseup = function() {
      draggingNode = null;
      panning = false;
      panStart = null;
    };
    svg.onmouseleave = function() {
      draggingNode = null;
      panning = false;
      panStart = null;
    };
    if (svg.addEventListener) {
      svg.addEventListener("wheel", function(evt) {
        evt.preventDefault();
        var scale = evt.deltaY < 0 ,  1.08 : 0.92;
        transform.k = Math.max(0.35, Math.min(3.5, transform.k * scale));
        applyTransform();
      }, false);
    }

    searchInput.oninput = function() {
      var q = searchInput.value.replace(/^\\s+|\\s+$/g, "").toLowerCase();
      if (!q) {
        selectedNode = null;
        updateHighlight();
        return;
      }
      var hit = null;
      for (i = 0; i < nodes.length; i++) {
        if (nodes[i].label.toLowerCase().indexOf(q) >= 0) {
          hit = nodes[i];
          break;
        }
      }
      selectedNode = hit;
      updateHighlight();
      if (hit) {
        transform.x = cx - hit.x * transform.k;
        transform.y = cy - hit.y * transform.k;
        applyTransform();
      }
    };
    resetBtn.onclick = function() {
      searchInput.value = "";
      selectedNode = null;
      transform = {x: 0, y: 0, k: 1};
      applyTransform();
      updateHighlight();
    };

    runSimulation();
    normalizeLayout();
    redraw();
    applyTransform();
    updateHighlight();
  </script>
</body>
</html>"""
    return (
        template
        .replace("__DOCS__", html.escape(str(stats.get("documents"))))
        .replace("__NODES__", html.escape(str(stats.get("graph_nodes"))))
        .replace("__EDGES__", html.escape(str(stats.get("graph_edges"))))
        .replace("__HELP_TEXT__", help_text)
        .replace("__HINT_TEXT__", hint_text)
        .replace("__PAYLOAD__", payload)
    )


def main() -> None:
    args = build_parser().parse_args()
    items = []
    with args.input.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    docs = group_by_document(items)
    doc_rows, global_df, global_score = extract_document_keywords(
        docs,
        top_doc_keywords=args.top_doc_keywords,
        min_doc_freq=args.min_doc_freq,
        max_doc_freq_ratio=args.max_doc_freq_ratio,
    )
    topic_nodes, topic_edges = build_graph(
        doc_rows,
        global_df,
        global_score,
        top_global_keywords=args.top_global_keywords,
        min_edge_weight=args.min_edge_weight,
        mode="topic",
    )
    core_nodes, core_edges = build_graph(
        doc_rows,
        global_df,
        global_score,
        top_global_keywords=max(30, min(args.top_global_keywords, 50)),
        min_edge_weight=max(args.min_edge_weight + 1, 3),
        mode="core",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    topic_stats = write_graph_bundle(
        args.outdir / "topic_keyword_graph",
        doc_rows=doc_rows,
        nodes=topic_nodes,
        edges=topic_edges,
        input_path=str(args.input),
    )
    core_stats = write_graph_bundle(
        args.outdir / "core_domain_term_graph",
        doc_rows=doc_rows,
        nodes=core_nodes,
        edges=core_edges,
        input_path=str(args.input),
    )

    summary = {
        "status": "success",
        "input": str(args.input),
        "outdir": str(args.outdir),
        "documents": len(doc_rows),
        "topic_graph": topic_stats,
        "core_graph": core_stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
