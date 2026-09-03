# Integrated Interpretation of Unknown Petroleum Fraction Sample (query_petroleum_fraction_db_007_gc_ir_only)

## Query

Please interpret these results for the uploaded unknown petroleum fraction sample. The sample workbook contains only GC-FID chromatogram and IR spectrum as analytical inputs. Use the historical experimental database evidence to infer density at 20 deg C, distillation distribution (BP20, BP35, BP50, BP65, and BP80), and saturates. Then use literature evidence to explain whether the inferred boiling range and composition are chemically consistent, discuss possible processing implications, and identify limitations or follow-up measurements.

## 1. Direct Tool Findings

The `database.experimental_history` capability successfully compared the uploaded GC-FID chromatogram and IR spectrum against the historical experimental database using saved PCA/PLS spectral models. The tool retrieved 10 nearest-neighbor historical samples and produced inverse-distance-weighted interpolation estimates for the target properties:

| Property | Weighted estimate | Evidence range | Support count |
|---|---:|---:|---:|
| Density at 20 deg C (g/cm3) | 0.8688 | 0.8348-0.9004 | 10 |
| BP20 (deg C) | 367.99 | 336.4-399.6 | 10 |
| BP35 (deg C) | 391.81 | 352.0-415.8 | 10 |
| BP50 (deg C) | 411.46 | 365.8-438.0 | 10 |
| BP65 (deg C) | 429.88 | 379.2-467.2 | 10 |
| BP80 (deg C) | 452.70 | 395.6-499.8 | 10 |
| Saturates (wt%) | 85.26 | 71.55-96.27 | 10 |

The nearest supporting sample was `petroleum_fraction_db_009` with a distance of 0.279951 and 66 shared features. This neighbor has a density of 0.8498 g/cm3, BP20-BP80 spanning 358.4-434.0 deg C, and saturates at 88.65 wt%. These values are evidence-based estimates from historical-neighbor comparison, not direct measurements or final labels.

## 2. Experimental Database Evidence, Inference, and Limitations

The experimental database provides meaningful neighbor-based support for this query. All 10 retrieved neighbors are petroleum fraction samples with measured values for the seven requested target properties. The retrieval uses 66 shared PCA/PLS-derived features from the GC-FID and infrared channels, so the similarity comparison is based on analytical information available in the uploaded workbook rather than on the masked target properties.

The retrieved property distributions are chemically informative. Density ranges from 0.8348 to 0.9004 g/cm3. BP50 ranges from 365.8 to 438.0 deg C. Saturates range from 71.55 to 96.27 wt%. The nearest neighbor, `petroleum_fraction_db_009`, shows a relatively narrow BP20-BP80 span of 75.6 deg C and high saturates content of 88.65 wt%, suggesting a relatively paraffinic fraction with a moderate boiling range.

Several limitations remain. First, the estimates are interpolated from historical neighbors and are not direct measurements. Second, some evidence ranges remain broad, especially for saturates and the higher boiling-point cut points, indicating uncertainty in the neighborhood. Third, GC-FID and infrared latent features reduce high-dimensional analytical data to compact feature representations, but this reduction cannot fully preserve all compositional details. Fourth, the historical database contains a limited number of curated samples, so the evidence is strongest when the query sample falls inside the represented sample space.

## 3. KB-Supported Interpretation and Processing Suggestions

The inferred boiling range and composition are chemically plausible for a saturate-rich petroleum fraction. The estimated density at 20 deg C is 0.8688 g/cm3, while the inferred BP20-BP80 range extends from approximately 368 to 453 deg C. This boiling profile is consistent with a middle-to-heavy petroleum fraction rather than a light naphtha-like material. The high estimated saturates content of 85.26 wt% is also consistent with a paraffinic or naphthenic-rich fraction.

The literature evidence supports this interpretation. GC-FID-based compositional models have been used to relate chromatographic distributions to density and boiling behavior [DOC 1]. Boiling-point modeling and pseudo-component representation are commonly used to convert compositional or chromatographic information into distillation-related descriptors [DOC 2, DOC 3]. GC-based methods are also widely used for petroleum fraction characterization, although heavier or more complex fractions may require more advanced separation or mass spectrometric methods to resolve overlapping components [DOC 4, DOC 5].

The database evidence and literature evidence are consistent with each other. The high saturates content and moderate density suggest a petroleum fraction that may be suitable for hydroprocessing or catalytic-cracking-related evaluation. The relatively narrow inferred boiling interval may also indicate a more defined cut than a broad resid-like feed. However, processing suitability cannot be determined from GC-FID and infrared data alone. Viscosity, sulfur and nitrogen content, metals, Conradson carbon residue, and detailed SARA or molecular composition would be needed for a stronger processing recommendation.

## 4. Integrated Conclusion

The uploaded GC-FID and infrared data support a database-assisted interpretation of the unknown petroleum fraction. The historical experimental database gives a density estimate of 0.8688 g/cm3 at 20 deg C, a distillation profile from BP20 = 367.99 deg C to BP80 = 452.70 deg C, and a saturates estimate of 85.26 wt%. These estimates are supported by 10 historical neighbors and are accompanied by explicit evidence ranges.

The combined evidence indicates a saturate-rich petroleum fraction with a moderate boiling range. The inferred values are chemically consistent with a paraffinic or naphthenic-rich petroleum fraction and are compatible with further evaluation for hydroprocessing, catalytic cracking, or related refinery processing routes. The conclusion is constrained by the historical-neighbor evidence and literature context, rather than being generated from the LLM alone.

Recommended follow-up measurements include simulated distillation or ASTM distillation to confirm the boiling curve, density measurement to validate the predicted density, SARA or detailed hydrocarbon analysis to verify saturates content, viscosity measurement for processing assessment, and higher-resolution chromatographic or mass spectrometric analysis if detailed molecular composition is required.

## Literature Evidence Returned by RAG

| ID | Title | Year | DOI |
|---|---|---:|---|
| DOC 1 | Computer-Aided Gasoline Compositional Model Development Based on GC-FID Analysis | 2018 | 10.1021/acs.energyfuels.8b01953 |
| DOC 2 | Boiling point modeling for petroleum speciation data | 2024 | 10.1016/j.fuel.2023.130066 |
| DOC 3 | Incorporating numerical molecular characterization into pseudo-component representation of light to middle petroleum distillates | 2019 | 10.1016/j.cesx.2019.100029 |
| DOC 4 | Prediction of Molecular Weight By-Boiling-Point Distribution of Middle Distillates from Gas Chromatography-Field Ionization Mass Spectrometry | 2011 | 10.1021/ef101329y |
| DOC 5 | Test Method for Determination of Light Hydrocarbons in Stabilized Crude Oils by Gas Chromatography | 2023 | 10.1520/d7900-23 |

## Post-run Validation

The following measured values were stored separately and were not included in the query workbook or Planner input.

| Property | DB evidence center | Measured value | MAPE (%) |
|---|---:|---:|---:|
| density at 20 deg C | 0.8688 | 0.8713 | 0.29 |
| BP20 | 367.99 | 350.60 | 4.96 |
| BP35 | 391.81 | 378.20 | 3.60 |
| BP50 | 411.46 | 399.20 | 3.07 |
| BP65 | 429.88 | 419.00 | 2.60 |
| BP80 | 452.70 | 442.20 | 2.37 |
| saturates | 85.26 | 91.31 | 6.63 |
