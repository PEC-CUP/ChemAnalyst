from __future__ import annotations

"""Runtime-authoritative taxonomy for the layered petroleum knowledge base.

This module defines the first-version controlled vocabulary used by:

- rule-based labeling
- metadata construction during ingest
- tag-aware retrieval and filtering

The current version intentionally keeps the taxonomy compact and editable.
Later versions can extend this file with:

- domain-specific aliases
- hierarchical parent/child tag mappings
- ontology IDs for knowledge graph construction
"""

SOURCE_GROUPS = [
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
]

TASK_GROUPS = [
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
]

EVIDENCE_TYPES = [
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
]

# Common petroleum-domain entities. Keys are normalized canonical labels and
# values are keyword/alias lists used by the rule labeler.
ENTITY_LEXICON = {
    "text": ["text", "crude oil", "feedstock"],
    "text": ["text", "petroleum fraction", "text", "text"],
    "text": ["text", "saturates", "text", "text"],
    "text": ["text", "aromatics", "text"],
    "text": ["text", "thiophenic", "sulfidic", "text"],
    "GC": ["gc", "text", "text", "text"],
    "GC×GC": ["gc×gc", "gcxgc", "text"],
    "FT-ICR MS": ["ft-icr", "orbitrap", "text"],
    "text": ["text", "density"],
    "API": ["api", "api gravity"],
    "DBE": ["dbe", "double bond equivalent"],
    "text": ["text", "carbon number"],
}

METHOD_LEXICON = {
    "GC": ["gc", "text", "text", "fid", "ms"],
    "GC×GC": ["gc×gc", "gcxgc", "text"],
    "FT-ICR MS": ["ft-icr", "orbitrap", "text"],
    "SARA": ["sara", "text", "text", "text", "text"],
    "NMR": ["nmr", "text"],
    "text": ["text", "text", "ea"],
}

# Path-level prior hints. These are intentionally weak priors and should be
# treated as anchors, not absolute truth.
PATH_GROUP_HINTS = {
    "papers": "text",
    "literature": "text",
    "standards": "text",
    "manuals": "text",
    "reports": "text",
    "notes": "text",
    "data": "text",
}

FILE_TASK_HINTS = {
    "gc": "text",
    "text": "text",
    "text": "text",
    "api": "text",
    "text": "text",
    "text": "text",
    "text": "text",
    "text": "text",
}
