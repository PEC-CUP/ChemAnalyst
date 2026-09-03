# Evaluation

`petroleum_kb/eval/` contains the benchmark generation and evaluation pipeline for ChemAnalyst.

Current scope:

- compare naive RAG / QA-oriented RAG / iterative review RAG
- generate paper-level benchmark questions from the KB
- support benchmark styles:
  - `semantic`: literature-derived semantic QA
  - `specific_fact`: narrow evidence-local QA designed to be hard for closed-book LLMs
- support open short-answer and strict MCQ benchmark generation
- score retrieval, context quality, and answer quality

Milestone stability rule:

- As of 2026-06-04, the QA benchmark construction path and the RAG evaluation path are treated as milestone-stable paper baselines.
- Do not casually modify `build_dataset.py`, `dataset_generation_utils.py`, `run_eval.py`, RAG mode names, `/query` runtime behavior, judge prompts, evidence sufficiency scoring, or UI-to-backend benchmark parameters.
- Any future change to QA generation or RAG runtime must document the intended behavior change and rerun smoke tests on reviewed single-doc, reviewed two-doc, specific-fact open, and specific-fact MCQ benchmarks.

RAG mode contract used by `run_eval.py`:

- `naive`: direct retrieval and answer synthesis.
- QA-oriented RAG: broad retrieval, naive-vector evidence anchors, LLM evidence filtering/reranking, document aggregation, and final synthesis.
- Iterative review RAG: query refinement, retrieval, LLM evidence filtering, review generation, evidence sufficiency checking, supplementary retrieval, and review update.
- `standard`: removed from the formal evaluation interface.


## Benchmark generation modes

The benchmark generator now supports two QA-construction modes:

- `reviewed`
  - strong paper-level prompt
  - second-pass judge + rewrite
  - pass, quality, language, and duplicate gates must all pass before writing
- `naive_baseline`
  - weak QA-generation prompt
  - same judge schema for scoring
  - no rewrite before writing
  - no pass / quality / language / duplicate write gates

Why this split matters:

- `naive_baseline -> reviewed` measures the full benchmark-construction gain

## Mode flows

```text
Reviewed
  chunks.jsonl
    -> paper-level evidence assembly
    -> structured QA generation
    -> judge scoring
    -> judge rewrite
    -> pass/quality/language/duplicate gates
    -> persisted reviewed benchmark row

Naive baseline
  chunks.jsonl
    -> paper-level evidence assembly
    -> weak-prompt QA generation
    -> judge scoring
    -> no rewrite before write
    -> no write gates
    -> persisted naive row
```

## 1. Canonical scripts

- `petroleum_kb/eval/build_dataset.py`
  - canonical benchmark builder
- `petroleum_kb/eval/run_eval.py`
  - online evaluation runner against `/query`
- `petroleum_kb/eval/run_ragas_eval.py`
  - offline official RAGAS evaluation over saved `run_eval.py` JSONL results
- `petroleum_kb/eval/run_evidence_sufficiency_eval.py`
  - offline LLM evidence-sufficiency evaluation over saved `run_eval.py` JSON/JSONL results
- `petroleum_kb/eval/summarize_evidence_sufficiency.py`
  - recompute evidence-sufficiency summary from existing sufficiency JSONL without LLM calls
- `petroleum_kb/eval/run_closed_book_eval.py`
  - closed-book LLM baseline over the same benchmark dataset, without retrieval
- `petroleum_kb/eval/score_semantic_eval.py`
  - compact aggregator for the semantic benchmark
- `petroleum_kb/eval/jsonl_to_json.py`
  - convert JSONL results to formatted JSON

## 1.1 Unified LLM configuration for evaluation

The public ChemAnalyst release uses one OpenAI-compatible LLM configuration for answer generation, RAG synthesis, and optional offline judging. Set the model name exactly as required by your provider; the repository does not hard-code provider-specific model IDs.

```powershell
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=your_provider_model_name_here
```

Interpretation rules:

- Normal public use requires only `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
- Changing `--model` or `--judge-model` does not require rerunning `/query` RAG answers. It only requires rerunning the offline evaluator over saved result JSON/JSONL files.
- Some checked-in historical result directories retain the model name used during the original manuscript experiments. Those names are provenance labels for saved outputs, not default runtime configuration.

## 1.2 Integrated `run_eval.py` Evaluation

`run_eval.py --judge` now writes all standard ChemAnalyst evaluation fields in one pass:

- answer judge:
  - `judge.correctness_score`
  - `judge.relevance_score`
  - `judge.faithfulness_score`
  - `judge.key_point_coverage_score`
- semantic context judge:
  - `context_judge.context_relevance`
  - `context_judge.context_precision`
  - `context_judge.context_recall`
- evidence sufficiency judge:
  - `evidence_sufficiency.evidence_sufficiency_score`
  - `evidence_sufficiency.sufficient`
  - `evidence_sufficiency.pass_threshold`

The summary CSV includes:

- `{mode}_evidence_sufficiency_score`
- `{mode}_evidence_sufficient_pass`

The default evidence-sufficiency pass threshold is `0.6`. Override it with:

```powershell
--evidence-sufficiency-threshold 0.6
```

`evidence_sufficiency` uses a separate retrieved-evidence context budget from the shorter semantic context judge:

```powershell
--judge-context-max-chars 3000 `
--sufficiency-context-max-chars 20000
```

This prevents two-document evidence sufficiency from being underestimated because the second document was truncated by the shorter context-judge budget.

Formal ChemAnalyst RAG reporting should use this integrated evaluation path, together with hard retrieval metrics (`hit_at_k`, `recall_at_k`, `map`, `two_doc_coverage`, `two_doc_coverage_strict`). `run_evidence_sufficiency_eval.py` remains available for old saved result files or context-source ablations, but new formal RAG evaluations should use the integrated `run_eval.py --judge` path.

## 1.3 Official RAGAS Offline Evaluation

Status: exploratory only. RAGAS is kept as an optional reproducibility tool, but it is not used as the formal ChemAnalyst benchmark metric.

`run_ragas_eval.py` evaluates saved RAG outputs without calling `/query` again. It uses the official `ragas` package and maps ChemAnalyst fields to the standard single-turn RAGAS schema:

- `user_input` <- `question`
- `response` <- `answer`
- `retrieved_contexts` <- chunk texts from `raw_result.evidences` by default
- `reference` <- `answer_key` plus `key_points` unless `--no-key-points` is set

Install optional dependencies only when RAGAS evaluation is needed:

```powershell
python -m pip install -r petroleum_kb\eval\requirements-ragas.txt
```

Python 3.14 compatibility note: the script keeps the official RAGAS metric classes, but uses a sync-backed LangChain OpenAI client for evaluator LLM calls. This avoids known `asyncio` / `anyio` / `sniffio` failures in the OpenAI async HTTP transport on this environment while preserving official RAGAS scoring logic.

Smoke test on a small balanced subset:

```powershell
python -B petroleum_kb\eval\run_ragas_eval.py `
  --input <local_path_removed> `
  --out <local_path_removed> `
  --summary <local_path_removed> `
  --modes naive,qa_oriented,iterative_review `
  --model $env:LLM_MODEL `
  --metrics context_precision,context_recall,faithfulness `
  --limit-per-mode 5
```

The current `run_eval.py` `context_judge` fields are custom DeepSeek LLM scores. RAGAS outputs from `run_ragas_eval.py` should be reported separately as official RAGAS metrics.

Why RAGAS is not the formal metric here:

- The reviewed ChemAnalyst benchmark is scientific paper QA with explicit answer keys and key points, not generic open-domain RAG.
- Many valid answers can be supported by alternative retrieved papers or equivalent evidence groups; strict context-to-reference alignment can understate scientific sufficiency.
- Two-document questions often require synthesis across documents; RAGAS context metrics can penalize useful broad evidence as noise.
- RAGAS faithfulness is useful as a diagnostic, but it does not replace key-point coverage, answer correctness, hard evidence coverage, and evidence sufficiency.

Use RAGAS only for optional diagnostic analysis. Do not use it as the main table for `naive` vs `qa_oriented` vs `iterative_review`.

The script also writes RAG reporting columns:

- `rag_quality_correctness_pass_rate`
  - from existing `judge.correct`
- `rag_quality_answer_relevancy_pass_rate`
  - from existing `judge.relevant`
- `rag_quality_context_precision_score`
  - from official RAGAS context precision when requested
- `rag_quality_context_recall_score`
  - from official RAGAS context recall when requested
- `rag_quality_faithfulness_score`
  - from official RAGAS faithfulness when requested
- `rag_quality_context_relevance_score`
  - from existing `context_judge.context_relevance`, because context relevance is not one of the core official RAGAS metrics used here
- `rag_quality_answer_relevancy_score`
  - from official RAGAS response/answer relevancy when requested; otherwise from existing `judge.relevance_score`

`answer_relevancy` / `response_relevancy` requires an embedding backend in RAGAS. If no compatible embedding API is available, run the core context metrics first:

```powershell
--metrics context_precision,context_recall,faithfulness
```

## 1.4 Evidence Sufficiency Offline Evaluation

`run_evidence_sufficiency_eval.py` evaluates whether the retrieved evidence is sufficient to answer the scientific question. It does not require exact benchmark gold documents unless the question explicitly asks for those specific papers. This metric is useful when a RAG mode retrieves alternative but scientifically sufficient papers.

The pass column is computed from the continuous score:

```text
evidence_sufficient_pass = evidence_sufficiency_score >= threshold
```

The default threshold is `0.6`, which is intended as an evaluation-oriented sufficiency gate rather than a strict gold-document coverage gate. The original model boolean, when returned, is preserved as `model_sufficient`; the reported pass rate uses the numeric threshold.

Supported context sources:

- `evidences`
- `candidate_evidences`
- `accumulated_evidences`
- `filtered_evidences`
- `synthesis_evidences`
- `rounds`

Smoke test:

```powershell
python -B petroleum_kb\eval\run_evidence_sufficiency_eval.py `
  --input <local_path_removed> `
  --out <local_path_removed> `
  --summary <local_path_removed> `
  --modes naive,qa_oriented,iterative_review `
  --context-source evidences `
  --model $env:LLM_MODEL `
  --threshold 0.6 `
  --limit-per-mode 3 `
  --progress-every 1
```

Recompute an existing sufficiency summary at threshold `0.6` without calling the evaluator again:

```powershell
python -B petroleum_kb\eval\summarize_evidence_sufficiency.py `
  --input <local_path_removed> `
  --summary <local_path_removed> `
  --threshold 0.6
```

## 1.5 Closed-book LLM Baseline

`run_closed_book_eval.py` tests the base LLM without retrieval. This is the correct way to show whether RAG is necessary beyond the model's parametric knowledge.

Single-document sample:

```powershell
python -B petroleum_kb\eval\run_closed_book_eval.py `
  --dataset <local_path_removed> `
  --out <local_path_removed> `
  --summary <local_path_removed> `
  --model $env:LLM_MODEL `
  --judge `
  --judge-model $env:LLM_MODEL `
  --limit 10 `
  --progress-every 1
```

Full single-document run:

```powershell
python -B petroleum_kb\eval\run_closed_book_eval.py `
  --dataset <local_path_removed> `
  --out <local_path_removed> `
  --summary <local_path_removed> `
  --model $env:LLM_MODEL `
  --judge `
  --judge-model $env:LLM_MODEL `
  --resume `
  --progress-every 5
```

Full two-document run:

```powershell
python -B petroleum_kb\eval\run_closed_book_eval.py `
  --dataset <local_path_removed> `
  --out <local_path_removed> `
  --summary <local_path_removed> `
  --model $env:LLM_MODEL `
  --judge `
  --judge-model $env:LLM_MODEL `
  --resume `
  --progress-every 5
```

## 1.6 Paper result layout and interrupted runs

The formal paper result layout is documented in Section 6.1. The key rule is to keep reviewed QA, specific-fact open-answer QA, and specific-fact MCQ as separate benchmark families.

Interrupted RAG runs are documented in Section 6.2. `run_eval.py` does not implement `--resume`; the `--out` JSONL path is opened for writing, so rerunning the same command can overwrite an incomplete file.

Use these rules:

- `--retry-errors-from` is only for rows that already exist in a previous result file but contain request errors or judge errors.
- It does not recreate missing question-mode rows.
- For interrupted runs, identify missing `(question_id, rag_mode)` pairs, build a small temporary dataset for those question IDs, rerun each missing mode to a separate temporary output, then merge the original and replacement rows into a final complete JSONL.
- Keep `*.before_retry.jsonl` files only as temporary audit backups. They should not be treated as final paper results.

Completion target:

```text
single reviewed RAG run: 200 questions * 3 modes = 600 valid rows
two-doc reviewed RAG run: 200 questions * 3 modes = 600 valid rows
closed-book run: 200 questions * 1 mode = 200 valid rows
```

## 2. Dataset schema

Each line in the dataset is one question.

Minimal example:

```json
{
  "id": "q0001",
  "question": "How ... rows",
  "question_type": "two_doc",
  "gold_doc_ids": ["doi:...", "doi:..."],
  "answer_key": "Reference answer ..."
}
```

Extended example:

```json
{
  "id": "q0001",
  "benchmark_style": "semantic",
  "question": "How ... rows",
  "question_type": "two_doc",
  "gold_doc_ids": ["doi:a", "doi:b"],
  "equivalent_doc_groups": [
    {"gold_doc_id": "doi:a", "doc_ids": ["doi:a", "doi:a_alt1"]},
    {"gold_doc_id": "doi:b", "doc_ids": ["doi:b", "doi:b_alt1"]}
  ],
  "answer_key": "Reference answer ...",
  "key_points": ["point 1", "point 2", "point 3"]
}
```

Key fields:

- `benchmark_style`
  - `semantic`
- `question_type`
  - `single`, `two_doc`, `3_doc`, or `4_doc`
- `gold_doc_ids`
  - benchmark evidence set
- `equivalent_doc_groups`
  - soft evidence groups for relaxed coverage
- `answer_key`
  - reference answer
- `key_points`
  - atomic facts used by the judge

## 3. Benchmark style

### 3.1 `semantic`

Purpose:

- literature-derived semantic QA benchmark
- more suitable for evaluating end-to-end answer quality and context quality

Generation target:

- self-contained
- semantically clear
- less sensitive to exact wording
- more suitable for realistic paper QA

Recommended dataset filename:

- `petroleum_kb/eval/datasets/benchmark.semantic.paperlevel.en.jsonl`

### 3.2 `specific_fact`

`specific_fact` is a RAG benchmark mode for narrow, evidence-local questions.

Use it when the goal is to reduce closed-book answerability and force retrieval of concrete paper facts. Good questions ask for:

- exact thresholds, ranges, ratios, operating conditions, or parameter values
- named ionization modes, detectors, sample fractions, compound classes, or formula constraints
- specific method-selection facts supported by a paper
- short factual conclusions that are hard to infer from general model priors

Avoid broad prompts such as:

- "What are the advantages and challenges of ..."
- "How does this method contribute to petroleum analysistop"
- "Compare the overall implications of ..."

Supported answer formats:

- `--answer-format open`
  - short evidence-grounded answer
- `--answer-format mcq`
  - strict four-option multiple choice
  - the question text and `choices` field must contain the same A-D options
  - `correct_option` and `answer_key` must match the correct labeled option
  - the generator balances correct-answer positions and rejects inconsistent MCQ rows

## 4. Paper-level dataset generation

Questions are now generated from multiple chunks from the same paper, not from isolated excerpts.

Important current recommendation:

- For formal benchmark generation, prefer a cleaned processed file rather than the raw ingest output.
- Current recommended benchmark source:
  - `petroleum_kb/petroleum_kb/data/processed/chunks_clean.jsonl`
- Avoid mixing:
  - `chunks_demo.jsonl` for historical/demo experiments
  - raw `chunks.jsonl` for uncleaned exploratory runs
  - `chunks_clean.jsonl` for current formal benchmark construction

Benchmark construction supports two data-source modes:

- Existing chunks: start directly from a processed `chunks.jsonl`.
- PDF directory: parse local PDFs with GROBID or PyMuPDF, chunk the extracted text, then generate QA from the resulting `chunks.jsonl`.

For benchmark construction, chunk labels are not required. The PDF-directory mode can run ingest with `--label-mode none`; generated QA depends on document text, metadata, gold document IDs, and source chunk IDs rather than rule/LLM chunk labels.

Paper-level means that chunks are first grouped by DOI/title/file, then several chunks from the same paper are aggregated into one paper-level evidence block. Multi-document questions are multi-paper-level questions: a `2_doc`, `3_doc`, or `4_doc` row combines paper-level evidence blocks from two, three, or four separate papers.

Pipeline:

1. group chunks by DOI / document
2. build paper-level evidence from multiple chunks
3. let the LLM generate question + answer_key + key_points
4. let the LLM review the question for:
   - self-containedness
   - answerability
   - objective wording
   - specificity
   - ambiguity risk
   - evidence locality
   - reasoning type
   - difficulty
   - hallucination risk
5. rewrite or discard weak questions

This is implemented in:

- `petroleum_kb/eval/build_dataset.py`
- `petroleum_kb/eval/dataset_generation_utils.py`

Question-quality fields:

- `self_contained_score`: whether the question can be understood without hidden context
- `answerable_score`: whether the selected evidence is sufficient
- `objective_score`: whether the answer is evidence-based rather than opinion-based
- `specificity_score`: whether concrete methods, samples, phenomena, or targets are named
- `ambiguity_score`: risk of multiple plausible interpretations; lower is better
- `evidence_locality_score`: whether the answer is supported by selected papers rather than broad outside knowledge
- `reasoning_type`: expected reasoning pattern, e.g. fact lookup, method explanation, comparison, synthesis, causal reasoning, or property reasoning
- `difficulty`: easy / medium / hard label for stratified evaluation
- `hallucination_risk_score`: risk that weak wording or insufficient evidence invites unsupported generation; lower is better

### 4.1 Generate semantic benchmark

```powershell
python -B petroleum_kb/eval/build_dataset.py `
  --out petroleum_kb/eval/datasets/benchmark.semantic.paperlevel.en.jsonl `
  --chunks petroleum_kb/petroleum_kb/data/processed/chunks_clean.jsonl `
  --num 100 `
  --mode mixed `
  --doc-count-ratios '{\"1\":0.50,\"2\":0.30,\"3\":0.15,\"4\":0.05}' `
  --min-doc-chunks 3 `
  --doc-max-chunks 6 `
  --quality-min-score 0.7 `
  --benchmark-style semantic `
  --language en
```

### 4.2 Current reviewed benchmark recommendation

For current formal reviewed-paper benchmarks:

- use `chunks_clean.jsonl`
- prefer explicit `--chunks ...` instead of relying on whichever raw file happens to be present
- keep `*_demo` files out of the generation path

Example:

```powershell
python -B petroleum_kb/eval/build_dataset.py `
  --chunks <local_path_removed> `
  --out <local_path_removed> `
  --num 100 `
  --mode mixed `
  --benchmark-mode reviewed `
  --benchmark-style semantic `
  --language en `
  --two-doc-ratio 0.5 `
  --quality-min-score 0.7 `
  --seed 7 `
  --progress-every 5
```

### 4.3 Generate specific-fact benchmark questions

These commands generate RAG narrow factual QA from the current clean KB. Each output contains 100 questions with a 50/50 single-doc and two-doc composition.

Open short-answer specific-fact QA:

```powershell
python -B petroleum_kb\eval\build_dataset.py `
  --chunks <local_path_removed> `
  --out <local_path_removed> `
  --num 100 `
  --mode mixed `
  --doc-count-ratios '{\"1\":0.5,\"2\":0.5}' `
  --benchmark-mode reviewed `
  --benchmark-style specific_fact `
  --answer-format open `
  --language en `
  --two-doc-pairing semantic `
  --two-doc-vector-min-sim 0.58 `
  --quality-min-score 0.7 `
  --max-answer-chars 260 `
  --seed 17 `
  --progress-every 1
```

Strict MCQ specific-fact QA:

```powershell
python -B petroleum_kb\eval\build_dataset.py `
  --chunks <local_path_removed> `
  --out <local_path_removed> `
  --num 100 `
  --mode single `
  --benchmark-mode reviewed `
  --benchmark-style specific_fact `
  --answer-format mcq `
  --language en `
  --quality-min-score 0.7 `
  --max-answer-chars 220 `
  --seed 23 `
  --progress-every 1
```

If single-doc and two-doc sets should be generated as separate files, replace `--mode mixed --doc-count-ratios ...` with `--mode single` or `--mode two_doc`.

## 5. Run evaluation

Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The API does not need to be restarted before every evaluation run. Restart it only when server-side code, `.env`, index paths, or RAG configuration have changed. Dataset generation, result parsing, plotting, and README updates do not require restarting `uvicorn`.

### 5.1 Semantic benchmark

```powershell
python -B petroleum_kb/eval/run_eval.py `
  --dataset petroleum_kb/eval/datasets/benchmark.semantic.paperlevel.en.jsonl `
  --base-url http://127.0.0.1:8000 `
  --out petroleum_kb/eval/results.semantic.paperlevel.jsonl `
  --modes naive,qa_oriented,iterative_review `
  --retrieval-k 8 `
  --judge `
  --save-raw `
  --judge-context-max-chars 3000 `
  --evidence-sufficiency-threshold 0.6 `
  --progress-every 5 `
  --show-eta
```

## 6. Output files

Typical outputs:

- `results.xxx.jsonl`
  - line-oriented raw results
- `results.xxx.json`
  - formatted JSON for manual inspection
- `results.xxx.summary.csv`
  - summary table

## 6.1 Paper result folders

Formal paper outputs are currently organized under:

```text
petroleum_kb/eval/results_paper/
```

Reviewed QA benchmark folders:

```text
benchmark.reviewed-rag/
<historical-benchmark-directory>/
```

Specific-fact benchmark folders:

```text
specific_fact_open/
specific_fact_mcq/
```

Do not mix these families in one aggregate table:

- `benchmark.reviewed-*` is for reviewed single-document and two-document semantic QA
- `specific_fact_open` is for narrow open-answer fact lookup
- `specific_fact_mcq` is for narrow multiple-choice fact lookup

The reviewed closed-book vs RAG comparison should use only:

```text
closed-book:
  petroleum_kb/eval/results_paper/<historical-benchmark-directory>

RAG:
  petroleum_kb/eval/results_paper/benchmark.reviewed-rag
```

The generated reviewed-only comparison outputs are:

```text
petroleum_kb/eval/results_paper/<historical-benchmark-directory>/comparison/
petroleum_kb/eval/results_paper/<historical-benchmark-directory>/figures/
```

The comparison folder contains:

- `closed_vs_rag_overall_mean_sd.csv`
- `closed_vs_rag_by_run_summary.csv`
- `rag_gain_vs_closed_book.csv`
- `closed_vs_rag_threshold_pass_rates.csv`
- `rag_only_coverage_curves.csv`
- `source_files_manifest.csv`
- `closed_vs_rag_comparison_report.md`

The figures folder contains reviewed-only answer-quality, gain-over-closed-book, context/evidence, recall, strict-coverage, and MAP plots.

## 6.2 Completing interrupted RAG runs

`run_eval.py` does not provide a general `--resume` option. If a run is stopped before all questions and modes finish, do not rerun the same command into the same output file.

Use this safe pattern instead:

1. keep the partial `.jsonl`
2. detect missing `(question_id, rag_mode)` pairs
3. write one missing dataset per mode
4. rerun each missing dataset with a single mode
5. merge the partial and missing results into a `.complete.jsonl`

Example for a stopped `specific_fact_open` run:

```powershell
@'
import json
from pathlib import Path

dataset = Path(r"<local_path_removed>")
partial = Path(r"<local_path_removed>")
out_dir = Path(r"<local_path_removed>")
out_dir.mkdir(parents=True, exist_ok=True)

modes = ["naive", "qa_oriented", "iterative_review"]
questions = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]

done = set()
if partial.exists():
    for line in partial.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        qid = str(obj.get("id") or "").strip()
        mode = str(obj.get("rag_mode") or "").strip().lower()
        ok = obj.get("ok") is True
        has_judge = isinstance(obj.get("judge"), dict) and "error" not in obj.get("judge", {})
        has_context = isinstance(obj.get("context_judge"), dict) and "error" not in obj.get("context_judge", {})
        has_suff = isinstance(obj.get("evidence_sufficiency"), dict) and "error" not in obj.get("evidence_sufficiency", {})
        if qid and mode and ok and has_judge and has_context and has_suff:
            done.add((qid, mode))

for mode in modes:
    missing = [q for q in questions if (str(q.get("id") or "").strip(), mode) not in done]
    out = out_dir / f"missing.{mode}.jsonl"
    out.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in missing) + ("\n" if missing else ""), encoding="utf-8")
    print(mode, "missing_questions =", len(missing), "->", out)
'@ | python -X utf8 -
```

Then run the missing datasets with one `--modes` value at a time. This avoids duplicating completed `(id, mode)` rows and avoids overwriting partial results.

Convert JSONL to JSON:

```powershell
python -B petroleum_kb/eval/jsonl_to_json.py `
  --in petroleum_kb/eval/results.semantic.paperlevel.jsonl `
  --out petroleum_kb/eval/results.semantic.paperlevel.json
```

## 7. Metric groups

### 7.0 Metric key names in raw JSONL

Recent result JSONL files store hard-retrieval metrics with snake_case keys:

- `hit_at_k`
- `recall_at_k`
- `precision_at_k`
- `mrr`
- `map`

Older summaries and some helper scripts may still display these as:

- `hit@k`
- `recall@k`
- `precision@k`
- `MRR`
- `MAP`

When parsing partial JSONL files manually, treat these as the same metrics.

### 7.1 Hard retrieval

- `hit@k`
- `recall@k`
- `precision@k`
- `MRR`
- `MAP`
- `two_doc_coverage`
- `two_doc_coverage_strict`

Interpretation:

- `two_doc_coverage`
  - relaxed multi-document coverage
- `two_doc_coverage_strict`
  - exact-gold coverage

### 7.5 Interpretation note for reviewed single-doc subsets

Recent experiments showed an important pattern:

- on `reviewed` single-document subsets, naive RAG can legitimately outperform QA-oriented RAG
- this is not automatically evidence of a runtime regression
- the same pattern can still appear when using a broader retrieval-and-filtering RAG path

Why:

- `reviewed` generation removes many weak, vague, and non-self-contained questions before evaluation
- early partial files are often dominated by `single` questions before `two_doc` rows appear
- single-document reviewed QA is relatively friendly to short, direct, evidence-local answers, which benefits `naive`

Therefore:

- do not use reviewed single-doc-only partial runs as the main proof that QA-oriented RAG is broken
- judge current-vs-historical alignment mainly with:
  - semantic benchmark comparisons
  - reviewed `two_doc` subsets
  - multi-document evidence coverage and faithfulness

### 7.2 Raw judge

- `correctness_score`
- `relevance_score`
- `faithfulness_score`
- `key_point_coverage_score`
- `raw_correct_and_relevant_pass`
- `raw_faithful_pass`

### 7.3 Gated judge

- `gated_correct_and_relevant_pass`
- `gated_faithful_pass`

These apply evidence-aware gating on top of the raw answer judgment.

### 7.4 Semantic context

- `context_relevance`
- `context_precision`
- `context_recall`

## 8. KG-enhanced evaluation


- normalized petroleum terms
- resolved entities
- support paths
- graph visualization files

These are context-quality metrics, not final answer metrics.

## 8. Semantic benchmark aggregation

For a compact semantic-benchmark summary:

```powershell
python -B petroleum_kb/eval/score_semantic_eval.py `
  --results petroleum_kb/eval/results.semantic.paperlevel.jsonl
```

It reports:

- `answer_correctness_pass_rate`
- `answer_relevancy_pass_rate`
- `answer_correctness_and_relevancy_pass_rate`
- `faithfulness_score`
- `faithfulness_pass_rate`
- `key_point_coverage_score`
- `key_point_pass_rate`
- `context_precision`
- `context_recall`
- `context_relevance`

## 9. Reranker notes

If reranker is enabled, prefer local/offline loading:

```text
PETROLEUM_KB_ENABLE_RERANKER=true
PETROLEUM_KB_RERANK_LOCAL_FILES_ONLY=true
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Optional explicit local snapshot:

```text
PETROLEUM_KB_RERANK_MODEL=<local_path_removed>
```

Check reranker initialization failures:

```powershell
Select-String -Path petroleum_kb/eval/results.xxx.jsonl -Pattern '"rerank_error"' | Measure-Object
```

## 10. Recommended reporting

Do not report only one score. Report at least:

- hard retrieval
  - `recall@k`
  - `MAP`
  - `two_doc_coverage`
  - `two_doc_coverage_strict`
- semantic context
  - `context_relevance`
  - `context_precision`
  - `context_recall`
- answer quality
  - `correctness_score`
  - `faithfulness_score`
  - `key_point_coverage_score`
  - `raw_correct_and_relevant_pass`
- more evidence-constrained answer quality
  - `gated_correct_and_relevant_pass`
  - `gated_faithful_pass`
