# Analytical Tool Onboarding Runtime Test: Hydrocarbon Type Calculator

This file records the Hydrocarbon Type Calculator runtime test used as the Analytical Tool Onboarding example. It is reconstructed from the saved runtime artifacts in this run directory and corresponds to the Planner-routed test path.

## 1. Test Query Sent to Planner

Calculate hydrocarbon type composition from the uploaded workbook.

## 2. Planner Tool Selection and Input Request

The Planner selected the Hydrocarbon Type Calculator as the relevant analytical tool for the request.

Required input:

- File type: `.xlsx` Excel workbook.
- Required workbook sheets: `saturate_ms`, `aromatic_ms`, and `fractions`.
- The `saturate_ms` and `aromatic_ms` sheets contain mass-spectrometric ion or intensity tables for saturated and aromatic fractions.
- The `fractions` sheet provides the saturated and aromatic fraction mass percentages used to combine the two fraction-level results.

The input workbook used in this runtime test was:

`<local_path_removed>

## 3. Tool Execution Result

The uploaded workbook was routed to the registered provider:

- Capability id: `tool.hydrocarbon_type_calculator`
- Description: Calculate hydrocarbon type composition from saturated/aromatic MS tables and fraction mass percentages.
- Execution status: `success`
- Runtime message: `generic_python_function_completed`
- Structured evidence rows: 18
- Quality flags: none

The output workbook artifact was generated at:

`<local_path_removed>

## 4. Structured Tool Evidence Returned

| Hydrocarbon group | Result |
|---|---:|
| Paraffins | 36.2 wt% |
| Monocycloparaffins | 16.2 wt% |
| Dicycloparaffins | 13.0 wt% |
| Tricycloparaffins | 5.0 wt% |
| Total cycloparaffins | 34.2 wt% |
| Total saturates | 70.3 wt% |
| Alkylbenzenes | 8.9 wt% |
| Indanes or tetralins | 5.1 wt% |
| Indenes or CnH2n-10 | 4.1 wt% |
| Total monoaromatics | 18.1 wt% |
| Naphthalene | 0.4 wt% |
| Naphthalenes | 5.4 wt% |
| Acenaphthenes or CnH2n-14 | 2.6 wt% |
| Fluorenes or CnH2n-16 | 2.4 wt% |
| Total diaromatics | 10.8 wt% |
| Tricyclic aromatics | 0.8 wt% |
| Total aromatics | 29.7 wt% |
| Total weight | 100.0 wt% |

## 5. Evidence Bundle Entry

The tool output was returned as structured tool evidence and attached to the evidence bundle.

```json
{
  "schema_version": "hydrocarbon-type.v1",
  "capability_id": "tool.hydrocarbon_type_calculator",
  "evidence_type": "hydrocarbon_group_composition",
  "method": "hydrocarbon type composition",
  "status": "success",
  "quality_flags": [],
  "structured_rows": 18,
  "artifacts": {
    "output_excel": "<local_path_removed>/app/tools/document_uploads/orchestration_runs/hydrocarbon_type_planner_smoke_161605/tool.hydrocarbon_type_calculator/02231c9ad08b4e1bbd604dd6da3a49c8_input_template_hydrocarbon_type_1f839728.xlsx"
  }
}
```

## 6. Planner Readback

Planner smoke test completed. The uploaded workbook was routed to Hydrocarbon Type Calculator, and the structured tool evidence was attached to the evidence bundle.

## 7. Interpretation for Section 3.4

The runtime test demonstrates that a reviewed analytical script can be packaged as a ChemAnalyst capability, selected by the Planner, executed on an uploaded workbook, and returned as structured evidence with an output artifact. In this example, the sample is saturate-rich, with total saturates of 70.3 wt% and total aromatics of 29.7 wt%. The dominant saturated groups are paraffins and cycloparaffins, while monoaromatics and diaromatics account for most of the aromatic fraction. This structured composition evidence can be used directly in an integrated ChemAnalyst response or combined with historical database and literature evidence in subsequent reasoning.


