# ChemAnalyst

ChemAnalyst is an LLM-assisted workbench for petroleum analytical chemistry. It combines literature retrieval, historical experimental records, analytical-tool execution, and planner-guided reasoning in a single FastAPI application.

The public release is designed as a runnable and simplified version of the system. All LLM-backed functions use one unified model configuration, so users only need to configure one OpenAI-compatible endpoint and one model name before testing the workbenches.

## Main Capabilities

- **Unified LLM access**: configure one API key, base URL, and model name for all model-backed workflows.
- **Evidence-grounded RAG**: query a petroleum-domain knowledge base with naive, QA-oriented, and iterative-review RAG routes.
- **Knowledge-base construction**: ingest petroleum literature text, chunk documents, assign labels, build indexes, and prepare QA benchmark data.
- **Historical experimental database**: import petroleum-fraction records and retrieve neighbor-supported evidence for property inference.
- **Analytical tool onboarding**: package reviewed Python scripts as planner-visible tools and test them through a dedicated workbench.
- **Planner-routed analysis**: combine uploaded files, RAG evidence, experimental evidence, tool outputs, provenance, confidence, and limitations.

## Repository Layout

- `app/`: FastAPI backend, routing logic, LLM clients, planner/orchestration modules, RAG adapters, upload handlers, and web workbench pages.
- `petroleum_kb/`: petroleum knowledge-base ingestion, labeling, indexing, retrieval, RAG evaluation utilities, benchmark datasets, and demo KB assets.
- `data/experimental_db/`: public demo structure and the released petroleum-fraction historical experimental dataset used for database-guided property inference.
- `case_studies/`: representative machine-readable case records for historical-database inference, RAG method interpretation, integrated historical DB + RAG reasoning, and tool onboarding.
- `knowledge/`: released domain-knowledge artifacts, including the inspectable keyword graph generated from the cleaned literature corpus.
- `tools/`: offline utilities for sample workbook generation, evaluation preparation, and related reproducibility tasks.
- `tests/`: regression tests for routing, support-layer APIs, tool onboarding, RAG/evidence utilities, and experimental-database helpers.

Large private source corpora, upload caches, local runtime files, manuscript drafting materials, and intermediate plotting or exploratory outputs are not included in this public repository.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

After the server starts, open the main page:

```text
http://127.0.0.1:8000/
```

## Workbench Pages

- Main chat: `http://127.0.0.1:8000/`
- Support layer: `http://127.0.0.1:8000/support-layer`
- KB construction and QA generation: `http://127.0.0.1:8000/benchmark-workbench`
- RAG workbench: `http://127.0.0.1:8000/rag-workbench`
- Experimental DB workbench: `http://127.0.0.1:8000/experimental-db-workbench`
- Experimental DB sample manager: `http://127.0.0.1:8000/experimental-db-manager`
- Analytical tool onboarding: `http://127.0.0.1:8000/tool-onboarding`

## LLM API Configuration

ChemAnalyst uses one unified LLM configuration for all public workbench functions. Create `.env` from `.env.example` and set:

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com
LLM_MODEL=your_model_name_here
LLM_TIMEOUT_SECONDS=60
```

`LLM_BASE_URL` should point to an OpenAI-compatible chat-completions endpoint. `LLM_MODEL` is not hard-coded by ChemAnalyst; use the exact model identifier provided by your selected service.

The `.env` file is local and ignored by git, so API keys should be configured locally rather than committed to the repository.

## Historical Experimental Dataset

The processed petroleum-fraction historical experimental dataset used in the manuscript is released under `data/experimental_db/petroleum_fraction_dataset/`. It contains 68 sample workbooks split into 54 historical records and 14 hold-out query samples. Each workbook retains only the manuscript workflow inputs and targets: GC-FID, IR spectrum, density at 20 deg C, saturates, BP20, BP35, BP50, BP65, and BP80.

Reproducibility outputs for weighted historical-neighbor retrieval, ordinary KNN comparison, evidence centers, evidence ranges, and top-k sensitivity are provided under `data/reproducibility/historical_db/`.

## Knowledge Base and Benchmarks

The `petroleum_kb/` subproject can be used from the command line:

```powershell
python petroleum_kb/main.py scan --source petroleum_kb/petroleum_kb/data/raw
python petroleum_kb/main.py ingest --source petroleum_kb/petroleum_kb/data/raw
python petroleum_kb/main.py build_index
python petroleum_kb/main.py search --query "DPF-MS fractionation principle"
```

Released QA benchmark files are stored in `petroleum_kb/eval/datasets/` as JSONL files. They can be used for retrieval and answer-quality experiments with the evaluation scripts under `petroleum_kb/eval/`.

The complete keyword graph generated from the cleaned literature corpus is released at `knowledge/keyword_graph/topic_keyword_graph_demo.html` and can be opened directly in a browser.

## Development Checks

Useful lightweight checks before running or modifying the application:

```powershell
python -m py_compile app/main.py
python -m pytest -q
```

Some workflows require optional local assets, external model endpoints, or cached embedding/reranker models. See `.env.example` and the workbench manuals for the relevant runtime options.
