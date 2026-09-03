# Appendix X. Full Experimental Database Evidence Test Case for Historical-Database-Based Property Inference

## Case selection

This case uses the hold-out query sample `petroleum_fraction_db_041` (`petroleum_fraction_60`). It is selected because LLM interpretation improves all seven target properties relative to the DB evidence center. Under GPT-4.1 with top-k = 10, the mean property MAPE decreases from 9.10% for the DB evidence center to 4.73% after LLM adjustment, corresponding to an absolute reduction of 4.37 percentage points and a relative reduction of 48.0%.

## Input query

- Query workbook: `petroleum_fraction_db_041__petroleum_fraction_60.xlsx`

- Query sample: `petroleum_fraction_db_041` (`petroleum_fraction_60`)

- Evidence channels: gc_fid, ir_spectrum

- Retrieved historical neighbors: top-k = 10

- LLM service: GPT-4.1


```text
Use IR and GC-FID to infer density, distillation distribution, and saturates for this petroleum fraction sample.
```

## Retrieved historical-neighbor evidence

| rank | sample_id | raw_sample_id | distance | shared_features | density_20c | BP_20 | BP_35 | BP_50 | BP_65 | BP_80 | saturates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | petroleum_fraction_db_035 | petroleum_fraction_46 | 0.287407 | 66 | 0.9004 | 378.4 | 411.2 | 438 | 467.2 | 499.8 | 71.5546 |
| 2 | petroleum_fraction_db_033 | petroleum_fraction_44 | 0.372195 | 66 | 0.9002 | 368 | 398.6 | 421.8 | 444.2 | 471 | 78.7296 |
| 3 | petroleum_fraction_db_009 | petroleum_fraction_11 | 0.445301 | 66 | 0.8498 | 358.4 | 378 | 395.6 | 412.8 | 434 | 88.6488 |
| 4 | petroleum_fraction_db_029 | petroleum_fraction_40 | 0.507368 | 66 | 0.8748 | 366.4 | 382.6 | 398.6 | 411.6 | 429 | 84.5994 |
| 5 | petroleum_fraction_db_063 | petroleum_fraction_87 | 0.509632 | 66 | 0.8783 | 343 | 375.6 | 404 | 429 | 456.4 | 86.5569 |
| 6 | petroleum_fraction_db_004 | petroleum_fraction_04 | 0.509831 | 66 | 0.9149 | 361.2 | 394.2 | 420.8 | 444.2 | 468.6 | 57.8355 |
| 7 | petroleum_fraction_db_010 | petroleum_fraction_13 | 0.576945 | 66 | 0.8619 | 336.4 | 352 | 365.8 | 379.2 | 395.6 | 88.0427 |
| 8 | petroleum_fraction_db_002 | petroleum_fraction_02 | 0.61161 | 66 | 0.8864 | 367.6 | 394.6 | 416.6 | 439.8 | 469.2 | 78.8644 |
| 9 | petroleum_fraction_db_066 | petroleum_fraction_90 | 0.635929 | 66 | 0.8712 | 399.6 | 415.8 | 427.2 | 438.2 | 450.8 | 90.2649 |
| 10 | petroleum_fraction_db_040 | petroleum_fraction_59 | 0.639381 | 66 | 0.8896 | 360.6 | 381.2 | 397.8 | 414.6 | 432.8 | 78.4085 |

## Property evidence and validation summary

| property | DB evidence center | evidence range | LLM estimate | measured value | DB APE (%) | LLM APE (%) | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| density_20c | 0.8844 | 0.8498-0.9149 | 0.9 | 0.905 | 2.28 | 0.55 | high |
| BP_20 | 364.72 | 336.4-399.6 | 375 | 406.6 | 10.3 | 7.77 | moderate |
| BP_35 | 390.18 | 352.0-415.8 | 405 | 436.4 | 10.59 | 7.2 | moderate |
| BP_50 | 411.24 | 365.8-438.0 | 430 | 459 | 10.41 | 6.32 | moderate |
| BP_65 | 431.7 | 379.2-467.2 | 450 | 480 | 10.06 | 6.25 | moderate |
| BP_80 | 455.42 | 395.6-499.8 | 485 | 501.4 | 9.17 | 3.27 | moderate |
| saturates | 79.56 | 57.84-90.26 | 73 | 71.74 | 10.9 | 1.76 | moderate |

## Complete LLM output

```json
{
  "sample_id": "petroleum_fraction_db_041",
  "model_reasoning_summary": "The adjusted estimates are based on close neighbor property values, with feature-weighted distances suggesting strongest similarity to samples petroleum_fraction_db_035 and petroleum_fraction_db_033. Evidence from both GC-FID and IR latent projections is considered via high feature weights for density and distillation points, while the broader neighbor set ensures reasonable uncertainty bounds. Density is tightly localized around 0.90, with compositional and cut-point consistencies across nearest samples. Distillation curve points and saturates are modulated toward nearest high-weight samples and along expected chemical/density/volatility relationships. All values are maintained strictly within the experimental evidence range.",
  "properties": [
    {
      "property_name": "density_20c",
      "adjusted_estimate": 0.9,
      "evidence_center": 0.8844,
      "evidence_range": [
        0.8498,
        0.9149
      ],
      "confidence": "high",
      "rationale": "Top two neighbors (petroleum_fraction_db_035: 0.9004, petroleum_fraction_db_033: 0.9002) are near center and have minimal feature distance; high feature weights from GC-FID and IR; projection sharply localizes estimate (~0.900) just above db center, consistent with distillation and saturates; uncertainty is low except for potential minor error due to compositional drift."
    },
    {
      "property_name": "BP_20",
      "adjusted_estimate": 375.0,
      "evidence_center": 364.72,
      "evidence_range": [
        336.4,
        399.6
      ],
      "confidence": "moderate",
      "rationale": "Nearest two neighbors show BP_20 of 378.4 and 368.0. Adjusted toward their mean (373.2) but kept slightly higher (375.0) to align with density and saturates (higher density/saturates often slightly increases cut points). The estimate remains safely within the experimental range."
    },
    {
      "property_name": "BP_35",
      "adjusted_estimate": 405.0,
      "evidence_center": 390.18,
      "evidence_range": [
        352.0,
        415.8
      ],
      "confidence": "moderate",
      "rationale": "BP_35 for top two neighbors is 411.2 and 398.6. Adjusted estimate (405.0) is interpolated above db center, consistent with higher density and nearest-neighbor compositional context. Remains within range and fits smooth distillation curve shape."
    },
    {
      "property_name": "BP_50",
      "adjusted_estimate": 430.0,
      "evidence_center": 411.24,
      "evidence_range": [
        365.8,
        438.0
      ],
      "confidence": "moderate",
      "rationale": "BP_50 is 438.0 and 421.8 for top two neighbors; their average is 429.9. With density and saturates correlating with higher boiling fractions, estimate is modestly above center (430.0), following curve progression and density relationship. There is ample range below the max evidence; curve monotonicity is preserved."
    },
    {
      "property_name": "BP_65",
      "adjusted_estimate": 450.0,
      "evidence_center": 431.7,
      "evidence_range": [
        379.2,
        467.2
      ],
      "confidence": "moderate",
      "rationale": "Top neighbors show BP_65 at 467.2 and 444.2. Taking density and curve shape, an adjusted value of 450.0 is justified—slightly skewed up versus center but aligned with neighboring data continuity. All values consistent with physical expectations and cut-point ordering."
    },
    {
      "property_name": "BP_80",
      "adjusted_estimate": 485.0,
      "evidence_center": 455.42,
      "evidence_range": [
        395.6,
        499.8
      ],
      "confidence": "moderate",
      "rationale": "Highest neighbor BP_80 at 499.8, with next in line at 471.0 and 499.8. Sample is among higher density, implying some enrichment of heavier fractions. 485.0 is above db center but below the max, matching the inferred distillation pattern and within possible petroleum fraction fraction range."
    },
    {
      "property_name": "saturates",
      "adjusted_estimate": 73.0,
      "evidence_center": 79.56,
      "evidence_range": [
        57.84,
        90.26
      ],
      "confidence": "moderate",
      "rationale": "The closest neighbors have saturates at 71.55 and 78.73, suggesting a value just below db center but well above the minimum. Slightly down-weighted due to sample's likely elevated aromatic/cyclic content as suggested by IR latent projections. Estimate remains firmly within support range and shows plausible compositional continuity."
    }
  ]
}
```

## Machine-readable full record

The complete evidence bundle and LLM output are stored in `appendix_historical_db_property_inference_case_petroleum_fraction_60_gpt41_k10.json`.
