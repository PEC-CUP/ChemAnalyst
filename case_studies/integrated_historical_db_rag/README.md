# Integrated Historical DB and RAG Case

This case corresponds to the manuscript demonstration in which an unknown petroleum-fraction query is first interpreted with historical Experimental DB evidence and then explained with literature RAG evidence.

Files:

- `query_petroleum_fraction_db_007_gc_ir_only.xlsx`: masked query workbook containing only sample metadata, GC-FID, and IR spectrum.
- `validation_targets.csv`: measured target values kept separate from the query workbook for post-run validation.
- `historical_db_evidence.json`: structured database evidence generated from historical-neighbor retrieval.
- `integrated_query_full_output.md`: cleaned integrated answer combining database evidence, literature evidence, limitations, and follow-up measurements.
- `input_query.txt`: natural-language query used for the integrated case.

The target properties are density at 20 deg C, BP20, BP35, BP50, BP65, BP80, and saturates. The query workbook intentionally excludes these target values.
