# Historical DB Reproducibility Data

This directory contains machine-readable outputs generated from `data/experimental_db/petroleum_fraction_dataset` for the historical-database-guided property inference experiments.

The source dataset contains 68 petroleum-fraction samples split into 54 historical records and 14 hold-out query samples. Only the manuscript target properties are evaluated: `density_20c`, `saturates`, `BP_20`, `BP_35`, `BP_50`, `BP_65`, and `BP_80`.

## Main files

- `top_k_2/`, `top_k_4/`, `top_k_6/`, `top_k_10/`: complete per-run outputs from `tools/evaluate_petroleum_fraction_experimental_db.py`.
- `retrieval_metrics_by_topk.csv`: combined MAPE, error, bias, and evidence-range coverage metrics.
- `test_predictions_by_topk.csv`: evidence-center estimates and measured hold-out values.
- `neighbor_evidence_by_topk.csv`: ranked historical-neighbor evidence used to build each estimate.
- `figure5_retrieval_summary.csv`: compact summary for weighted retrieval versus ordinary KNN.
- `database_evidence_ranges_table_s1.csv`: evidence centers and ranges for manuscript/SI table inspection.

Regenerate with:

```powershell
python tools/evaluate_petroleum_fraction_experimental_db.py --output-dir data/reproducibility/historical_db/top_k_2 --top-k 2
python tools/evaluate_petroleum_fraction_experimental_db.py --reuse-tables-dir data/reproducibility/historical_db/top_k_2 --output-dir data/reproducibility/historical_db/top_k_4 --top-k 4
python tools/evaluate_petroleum_fraction_experimental_db.py --reuse-tables-dir data/reproducibility/historical_db/top_k_2 --output-dir data/reproducibility/historical_db/top_k_6 --top-k 6
python tools/evaluate_petroleum_fraction_experimental_db.py --reuse-tables-dir data/reproducibility/historical_db/top_k_2 --output-dir data/reproducibility/historical_db/top_k_10 --top-k 10
```
