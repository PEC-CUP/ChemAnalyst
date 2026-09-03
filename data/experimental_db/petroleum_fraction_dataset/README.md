# Petroleum Fraction Historical Experimental Dataset

This directory contains the processed sample-level dataset used for the historical-database-guided property inference experiments reported in the ChemAnalyst manuscript.

## Contents

- `train/`: 54 historical petroleum-fraction records used as the retrieval evidence pool.
- `test/`: 14 hold-out query records used for validation.
- `manifest.csv` and `manifest.json`: machine-readable sample inventory.
- `sample_split.json`: train/test split used in the manuscript.
- `target_property_table.csv`: measured target properties for all 68 samples.

Each sample workbook contains only the data used by the public manuscript workflow:

- `sample_information`
- `bulk_properties`
- `gc_fid`
- `ir_spectrum`

The retained target properties are `density_20c`, `saturates`, `BP_20`, `BP_35`, `BP_50`, `BP_65`, and `BP_80`. Non-used legacy sheets and properties, including GC-SCD, GC-NCD, sample relations, kinematic viscosity, wax content, elemental composition, and non-target boiling cut points, are intentionally excluded from this public release.
