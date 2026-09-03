# Sample-centered Experimental DB

`Experimental DB Workbench` is the curated historical-sample entry point in ChemAnalyst. Chat uploads, temporary tool outputs, and ad hoc session files are not historical records by default. They become reusable evidence only after explicit import into this module.

## Released manuscript dataset

The manuscript petroleum-fraction dataset is provided in `petroleum_fraction_dataset/`. It is a compact public release containing 68 processed sample workbooks: 54 historical records and 14 hold-out query samples. The released workbooks keep only `sample_information`, `bulk_properties`, `gc_fid`, and `ir_spectrum`; non-used legacy sheets such as GC-SCD, GC-NCD, and sample relations are excluded.

Machine-readable reproducibility outputs derived from this dataset are stored in `data/reproducibility/historical_db/`.

## 1. Design goal

The module is designed as a sample-centered historical experimental database for petroleum analysis. It has three responsibilities:

- standardize heterogeneous laboratory records into one sample-centered structure;
- preserve provenance, attachments, and inter-sample relations instead of keeping only isolated property tables;
- transform imported records into retrieval-ready evidence for later property reasoning and multi-source comparison.

The core design decision is that `sample_information` is kept independent as sample identity/context, whereas `bulk_properties` and other `analyses` are both treated as analysis-layer evidence.

## 2. Record structure

The database is centered on `sample_id`, not on one instrument or one project. A sample record can include:

- `sample_information`: sample name, sample type, origin, refinery, project, and optional metadata;
- `bulk_properties`: elemental composition, SARA, density, viscosity, wax, API gravity, octane/cetane numbers, and `BP_5`-`BP_95`;
- `analyses`: IR, GC-FID, HTGC-FID, GC-NCD, GC-SCD, hydrocarbon types, molecular composition, and future sheet-defined analysis types;
- `relations`: fraction-of, product-of, feedstock-of, blend-of, repeat-of, and other sample links;
- `artifacts`: source workbook, chromatogram export, processed result file, or any extra bound evidence file.

## 3. Storage layout

- `samples/<sample_id>/sample.json`: full sample-centered record used as the canonical editable object.
- `registry/samples.json`: lightweight list registry for UI browsing.
- `registry/property_observations.jsonl`: property-level append log.
- `registry/analysis_records.jsonl`: analysis-level append log.
- `registry/sample_relations.jsonl`: relation-level append log.
- `registry/artifacts.jsonl`: artifact-level append log.
- `objects/<sample_id>/...`: copied workbook and artifact files.
- `catalog/experimental_samples.sqlite`: local searchable catalog used by the manager and evidence retrieval logic.
- `templates/template_definition.json`: active template definition used to parse future workbook imports.
- `templates/chemanalyst_experimental_sample_template.xlsx`: active downloadable template workbook.

## 4. Template governance

The workbook template is no longer treated as a fixed hard-coded file only. The active template is now governed by two synchronized resources:

- the downloadable Excel workbook;
- the parsed JSON template definition.

The Workbench supports a hidden `Template update` path for governance use. When a new template workbook is uploaded:

1. the workbook is stored as the new active template;
2. the sheet structure is parsed into `template_definition.json`;
3. future workbook imports are interpreted against that active definition.

This allows different laboratories to extend the schema by submitting revised Excel templates rather than modifying code for every new analysis sheet.

## 5. Workbook import contract

Download `GET /support-layer/experimental-db/template` from the Workbench. The built-in default template contains:

- `sample_information`: field/value metadata and required `sample_id`; optional `related_sample_id`, `relation_type`, and `relation_description`;
- `bulk_properties`: `property_name`, `value`, `unit`, `method`;
- `ir_spectrum`: `wave_number_cm-1`, `transmittance`;
- `gc_fid`, `htgc_fid`, `gc_ncd`, `gc_scd`: `rt`, `intensity`;
- `hydrocarbon_types`: flexible `component`, `content`, `unit`;
- `molecular_composition`: `molecule_family`, `molecular_formula`, `content`, `unit`, optional DBE and extra columns.

The analysis sheets are intentionally label-driven rather than column-count rigid. This keeps the import path compatible with:

- laboratories adding new hydrocarbon subclasses;
- molecular-composition tables with extra annotation columns;
- new instrument-specific sheets that later become new analysis channels.

## 6. Sample relations

Fractions should remain first-class samples. A diesel, gasoline, petroleum fraction, residue, blend, repeat, or processed product is not embedded inside the parent crude record. Instead, import it with its own `sample_id` and define the relation explicitly:

- fraction to parent crude: `related_sample_id=crude_001`, `relation_type=fraction_of`;
- parent crude to known child: `related_sample_id=diesel_001`, `relation_type=has_fraction`;
- other useful types: `product_of`, `feedstock_of`, `blend_of`, `derived_from`, `repeat_of`, `same_feedstock`.

The relation fields are optional. A standalone sample can leave them blank. JSON/API imports can still provide a `relations` array when a sample needs multiple links.

## 7. Manager and deletion behavior

`Experimental DB Sample Manager` is the visual management layer for imported records. It supports:

- sample browsing and filtering;
- record inspection by `sample_id`;
- jump from a sample row to the Workbench update panel;
- full sample deletion.

The current deletion path removes the whole sample-level record, including:

- `sample.json`;
- attached stored objects under `objects/<sample_id>/`;
- SQLite catalog rows;
- registry entries rebuilt from remaining samples.

Fine-grained deletion of selected analyses or properties is planned as a later extension.

## 8. Evidence usage

Imported records are not stored only for archival purposes. They are transformed into historical evidence through the Experimental DB evidence path.

The current evidence flow is:

1. choose a curated sample as query sample;
2. build a comparable feature profile from its imported analysis rows and bulk-property rows;
3. compare it with other historical samples in shared feature space;
4. rank nearest historical neighbors;
5. aggregate supported property ranges and weighted estimates from the retrieved neighborhood;
6. expose the result as a structured evidence bundle for downstream reasoning.

The Workbench now renders this evidence in manuscript-friendly form:

- query sample snapshot;
- historical neighbor ranking table;
- property evidence summary table;
- evidence statement;
- reasoning notes.

This module therefore serves both as a data-management subsystem and as a historical-evidence engine for no-pretrained-model property reasoning.
