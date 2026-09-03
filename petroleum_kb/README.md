# petroleum_kb Knowledge Base Subproject

`petroleum_kb` is the knowledge-base component used by ChemAnalyst. It converts petroleum-analysis documents into searchable chunks, labels them, builds a vector index, and exposes retrieval routines for the main application.

## Role in the System

- `petroleum_kb/` owns document ingestion, chunking, labeling, indexing, and retrieval.
- `app/rag/kb_backend.py` adapts the knowledge base for the application RAG routes.
- `app/agents/runtime_agent.py` and `app/agents/planner_orchestrator.py` consume retrieval results when a workflow needs literature-grounded evidence.

## Package Layout

- `main.py`: command-line entry point.
- `config.py`: knowledge-base configuration.
- `requirements.txt`: subproject dependencies.
- `petroleum_kb/io/`: document loaders, text chunking, and optional GROBID support.
- `petroleum_kb/labeling/`: taxonomy, rule labeling, and optional model/cluster labeling hooks.
- `petroleum_kb/indexing/`: embedding and vector-store utilities.
- `petroleum_kb/retrieval/`: search, evidence aggregation, and reranking.
- `petroleum_kb/pipeline/`: source registration, ingestion, and index-building workflows.
- `petroleum_kb/data/`: small demo/raw/processed/index files included for public testing.

## CLI

Install dependencies from the repository root:

```bash
python -m pip install -r petroleum_kb/requirements.txt
```

Common commands:

```bash
python petroleum_kb/main.py scan --source petroleum_kb/petroleum_kb/data/raw
python petroleum_kb/main.py ingest --source petroleum_kb/petroleum_kb/data/raw
python petroleum_kb/main.py build_index
python petroleum_kb/main.py search --query "DPF-MS fractionation principle"
python petroleum_kb/main.py search_docs --query "DPF-MS fractionation principle"
```

The CLI reads the repository-root `.env` when present. Public users should configure model access with the unified LLM variables described in the top-level README and `.env.example`.

## Offline Embedding Cache

For offline or restricted-network runs, configure relative cache paths in `.env`:

```text
PETROLEUM_KB_EMBEDDING_LOCAL_FILES_ONLY=true
HF_HOME=./.cache/huggingface
HF_HUB_CACHE=./.cache/huggingface/hub
SENTENCE_TRANSFORMERS_HOME=./.cache/sentence_transformers
TRANSFORMERS_CACHE=./.cache/huggingface/transformers
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

If the reranker is enabled, it can also be restricted to local files:

```text
PETROLEUM_KB_ENABLE_RERANKER=true
PETROLEUM_KB_RERANK_LOCAL_FILES_ONLY=true
```
