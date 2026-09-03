from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze chunk sources and build keyword co-occurrence graph")
    parser.add_argument("--input", type=Path, required=True, help="Input chunks.jsonl or chunks_clean.jsonl")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for reports and graph")
    parser.add_argument("--top-docs", type=int, default=100, help="How many top documents to keep in CSV preview")
    parser.add_argument("--min-edge-weight", type=int, default=5, help="Minimum co-occurrence count to keep a keyword edge")
    parser.add_argument("--max-nodes", type=int, default=120, help="Maximum keyword nodes to keep in graph")
    return parser


def _safe_str(value: object) -> str:
    return "" if value is None else str(value)


def load_chunks(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def summarize_documents(chunks: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], dict] = {}
    for item in chunks:
        md = item.get("metadata") or {}
        title = _safe_str(md.get("title")).strip()
        doi = _safe_str(md.get("doi")).strip()
        journal = _safe_str(md.get("journal")).strip()
        year = _safe_str(md.get("year")).strip()
        source = _safe_str(md.get("source")).strip()
        key = (title, doi, journal, year, source)
        row = grouped.setdefault(
            key,
            {
                "title": title,
                "doi": doi,
                "journal": journal,
                "year": year,
                "source": source,
                "chunk_count": 0,
                "entity_tags": Counter(),
                "method_tags": Counter(),
            },
        )
        row["chunk_count"] += 1
        for tag in md.get("entity_tags", []) or []:
            row["entity_tags"][str(tag)] += 1
        for tag in md.get("method_tags", []) or []:
            row["method_tags"][str(tag)] += 1

    result: list[dict] = []
    for row in grouped.values():
        result.append(
            {
                "title": row["title"],
                "doi": row["doi"],
                "journal": row["journal"],
                "year": row["year"],
                "source": row["source"],
                "chunk_count": row["chunk_count"],
                "top_entity_tags": [tag for tag, _ in row["entity_tags"].most_common(8)],
                "top_method_tags": [tag for tag, _ in row["method_tags"].most_common(8)],
            }
        )
    result.sort(key=lambda x: (-int(x["chunk_count"]), x["title"]))
    return result


def build_keyword_graph(chunks: list[dict], *, min_edge_weight: int, max_nodes: int) -> tuple[list[dict], list[dict]]:
    node_counter: Counter[str] = Counter()
    edge_counter: Counter[tuple[str, str]] = Counter()

    for item in chunks:
        md = item.get("metadata") or {}
        tags = []
        for tag in (md.get("entity_tags", []) or []):
            t = str(tag).strip()
            if t:
                tags.append(t)
        for tag in (md.get("method_tags", []) or []):
            t = str(tag).strip()
            if t:
                tags.append(t)
        tags = sorted(set(tags))
        for tag in tags:
            node_counter[tag] += 1
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                edge_counter[(tags[i], tags[j])] += 1

    kept_nodes = {tag for tag, _ in node_counter.most_common(max_nodes)}
    nodes = [
        {"id": tag, "label": tag, "weight": count}
        for tag, count in node_counter.most_common(max_nodes)
    ]

    edges = []
    for (a, b), w in edge_counter.most_common():
        if w < min_edge_weight:
            continue
        if a not in kept_nodes or b not in kept_nodes:
            continue
        edges.append({"source": a, "target": b, "weight": w})

    return nodes, edges


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_graph_html(nodes: list[dict], edges: list[dict], stats: dict[str, object]) -> str:
    payload = {
        "nodes": nodes,
        "edges": edges,
        "stats": stats,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Keyword Graph</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f7fb; color: #16202a; }}
    .wrap {{ padding: 20px; }}
    .meta {{ margin-bottom: 12px; font-size: 14px; color: #445; }}
    #graph {{ width: 100%; height: 78vh; background: #fff; border: 1px solid #d6deea; border-radius: 12px; }}
    .node-label {{ font-size: 12px; fill: #17324d; pointer-events: none; }}
    .edge {{ stroke: #9bb3c8; stroke-opacity: 0.45; }}
    .node {{ fill: #2c7fb8; stroke: #ffffff; stroke-width: 1.2px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="meta">
      chunks={html.escape(str(stats.get("chunks")))} |
      docs={html.escape(str(stats.get("docs")))} |
      graph_nodes={html.escape(str(stats.get("graph_nodes")))} |
      graph_edges={html.escape(str(stats.get("graph_edges")))}
    </div>
    <svg id="graph" viewBox="0 0 1400 900"></svg>
  </div>
  <script>
    const payload = {payload_json};
    const svg = document.getElementById("graph");
    const width = 1400;
    const height = 900;
    const cx = width / 2;
    const cy = height / 2;
    const nodes = payload.nodes.map((n, idx) => ({{...n, x: 0, y: 0, idx}}));
    const edges = payload.edges;
    const maxWeight = Math.max(1, ...nodes.map(n => n.weight));

    nodes.forEach((n, i) => {{
      const angle = (2 * Math.PI * i) / Math.max(1, nodes.length);
      const radius = 180 + (i % 11) * 24;
      n.x = cx + Math.cos(angle) * radius;
      n.y = cy + Math.sin(angle) * radius;
      n.r = 6 + 18 * Math.sqrt(n.weight / maxWeight);
    }});

    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    edges.forEach(e => {{
      e.a = nodeMap.get(e.source);
      e.b = nodeMap.get(e.target);
    }});

    const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    edges.forEach(e => {{
      if (!e.a || !e.b) return;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", "edge");
      line.setAttribute("x1", e.a.x);
      line.setAttribute("y1", e.a.y);
      line.setAttribute("x2", e.b.x);
      line.setAttribute("y2", e.b.y);
      line.setAttribute("stroke-width", String(Math.max(1, Math.log2(e.weight + 1))));
      edgeLayer.appendChild(line);
    }});

    nodes.forEach(n => {{
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("class", "node");
      c.setAttribute("cx", n.x);
      c.setAttribute("cy", n.y);
      c.setAttribute("r", n.r);
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "node-label");
      t.setAttribute("x", n.x + n.r + 4);
      t.setAttribute("y", n.y + 4);
      t.textContent = `${{n.label}} (${{n.weight}})`;
      g.appendChild(c);
      g.appendChild(t);
      nodeLayer.appendChild(g);
    }});
  </script>
</body>
</html>"""


def main() -> None:
    args = build_parser().parse_args()
    chunks = load_chunks(args.input)
    docs = summarize_documents(chunks)
    nodes, edges = build_keyword_graph(
        chunks,
        min_edge_weight=args.min_edge_weight,
        max_nodes=args.max_nodes,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    docs_csv_rows = []
    for row in docs[: args.top_docs]:
        docs_csv_rows.append(
            {
                "title": row["title"],
                "doi": row["doi"],
                "journal": row["journal"],
                "year": row["year"],
                "chunk_count": row["chunk_count"],
                "top_entity_tags": "; ".join(row["top_entity_tags"]),
                "top_method_tags": "; ".join(row["top_method_tags"]),
                "source": row["source"],
            }
        )
    write_csv(
        args.outdir / "document_chunk_stats.csv",
        docs_csv_rows,
        ["title", "doi", "journal", "year", "chunk_count", "top_entity_tags", "top_method_tags", "source"],
    )
    (args.outdir / "document_chunk_stats.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(args.outdir / "keyword_nodes.csv", nodes, ["id", "label", "weight"])
    write_csv(args.outdir / "keyword_edges.csv", edges, ["source", "target", "weight"])

    stats = {
        "chunks": len(chunks),
        "docs": len(docs),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "input": str(args.input),
    }
    (args.outdir / "keyword_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges, "stats": stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.outdir / "keyword_graph.html").write_text(
        render_graph_html(nodes, edges, stats),
        encoding="utf-8",
    )

    summary = {
        "status": "success",
        "input": str(args.input),
        "outdir": str(args.outdir),
        "chunks": len(chunks),
        "documents": len(docs),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "top_document": docs[0] if docs else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
