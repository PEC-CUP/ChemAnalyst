# Historical Crude-Oil Database Utilities

This module implements a no-supervised-training workflow for crude-oil GC history analysis.

It currently serves two linked goals:

- historical-database-based property reasoning
- template-guided, LLM-assisted peak calibration under RT drift

## Current workflow

1. Ingest historical sample workbooks from a directory such as `examples/crude_oil_db/source_workbooks`
2. Normalize:
   - `raw data`
   - `peak area`
   - `property.xlsx`
3. Remove the solvent peak around `6.23 min`
4. Build sample-level database features from:
   - normalized raw chromatogram shape
   - integrated peak-area profile
   - anchor-candidate regions around:
     - `C17 / Pr`
     - `C18 / Ph`
5. Run leave-one-out nearest-neighbor property inference
6. Generate LLM-ready reasoning cases
7. Optionally run LLM reasoning over:
   - property-inference cases
   - loose peak-matching cases under RT drift

## Design constraints

- No supervised regression/classification model is required
- New historical samples only need to be added to the database
- Absolute chromatogram intensities are not compared across samples
- RT drift is allowed; anchor-pair regions are preserved for reasoning support
- The reference template is retained as a primary reference, especially for:
  - light-end peak identification
  - regular n-alkane sequence reasoning
  - anchor-pair calibration

## Current limitations: database-grounded property reasoning

The current property-inference path should be interpreted as an evidence-guided
prototype, not as an independent LLM property predictor. The numerical baseline
is still mainly produced by historical-neighbor retrieval and property
aggregation, while the LLM reads the structured evidence, explains whether the
baseline is plausible, applies only conservative corrections, and reports
uncertainty.

Key limitations to keep explicit:

- Neighbor weighting assumes that the selected chromatographic feature space is
  genuinely property-relevant; if this feature space is weak, the estimate is
  dominated by the wrong neighbors.
- A single similarity space may not be appropriate for all targets. Density,
  light-end boiling points, mid-cut boiling points, and heavy-end boiling points
  may require different property-specific evidence profiles.
- `property_feature_profile` improves prompt grounding, but it is still a
  hand-designed rule layer. Much of the current correction logic could be
  reproduced by a deterministic script.
- The LLM is currently closer to an evidence-constrained adjudicator than a
  free property predictor. Evaluation must therefore report both the neighbor
  baseline and the LLM-adjusted output.
- Future work should test whether LLM reasoning adds value beyond KNN by using
  property-specific retrieval, stronger feature-property rationales, ablation
  experiments, and external samples with measured ground truth.

## Expected source layout

- `1#.xlsx` ... `67#.xlsx`
- `property.xlsx`

Each sample workbook is expected to contain:

- `raw data`
- `peak area`
- `property` (optional placeholder in the workbook; the main property source is `property.xlsx`)

## Outputs

- `data/crude_oil_db/normalized/chromatogram/*.raw.csv`
- `data/crude_oil_db/normalized/peak_table/*.peak.csv`
- `data/crude_oil_db/normalized/properties/*.property.json`
- `data/crude_oil_db/features/*.features.json`
- `data/crude_oil_db/registry/sample_registry.csv`
- `data/crude_oil_db/indexes/leave_one_out_results.json`
- `data/crude_oil_db/indexes/pilot_summary.json`
- `data/crude_oil_db/indexes/llm_reasoning_cases.jsonl`
- `data/crude_oil_db/indexes/llm_property_reasoning_cases.jsonl`
- `data/crude_oil_db/indexes/llm_peak_match_cases.jsonl`
- `data/crude_oil_db/indexes/llm_property_reasoning_outputs.json`
- `data/crude_oil_db/indexes/llm_peak_match_outputs.json`

## Example

```powershell
python -m app.tools.crude_oil_db.pipeline `
  --source-dir examples\crude_oil_db\source_workbooks `
  --out-root data\crude_oil_db `
  --mode leave_one_out `
  --k 5 `
  --test-limit 9
```

Then run LLM reasoning on the prepared cases:

```powershell
python -m app.tools.crude_oil_db.pipeline `
  --source-dir examples\crude_oil_db\source_workbooks `
  --out-root data\crude_oil_db `
  --mode llm_property_reasoning `
  --max-cases 5

python -m app.tools.crude_oil_db.pipeline `
  --source-dir examples\crude_oil_db\source_workbooks `
  --out-root data\crude_oil_db `
  --mode llm_peak_match_reasoning `
  --max-cases 5
```

## Reasoning support

The generated reasoning cases include explicit expert priors for LLM use:

- solvent peak exclusion near `6.23 min`
- RT drift tolerance
- `C17 / Pr` and `C18 / Ph` paired-peak anchor logic
- regular n-alkane progression as a secondary cue
- template-guided calibration rather than template abandonment
- migration detection and explicit uncertainty reporting
- property inference from retrieved historical neighbors rather than trained models
