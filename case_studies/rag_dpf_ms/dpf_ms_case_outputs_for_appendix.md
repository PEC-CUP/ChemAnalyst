# DPF-MS RAG Case Outputs

Question: Please provide a detailed analysis and introduction to the principles, application scenarios, relevant literature, and limitations of the Distillation Precipitation, Fractionation Mass Spectrometry (DPF-MS) method.

## Closed-book local chat

Model: configured-llm-model

Answer output:

# Distillation Precipitation, Fractionation Mass Spectrometry (DPF-MS): Principles, Applications, Literature, and Limitations

## 1. Introduction

Distillation Precipitation, Fractionation Mass Spectrometry (DPF-MS) is a hyphenated analytical technique developed primarily for the detailed characterization of complex hydrocarbon mixtures, particularly heavy petroleum fractions, bitumen, and crude oils. The method integrates a physical separation stage—combining distillation and precipitation—with mass spectrometric analysis to reduce sample complexity and enable molecular-level characterization of high-boiling, high-molecular-weight species that are otherwise intractable by direct MS analysis.

DPF-MS is not a single commercial instrument but rather a methodological workflow that can be assembled from modular components. It is most commonly encountered in petroleum geochemistry, heavy oil upgrading research, and refinery process development.

---

## 2. Principles of DPF-MS

### 2.1 Conceptual Basis

The central challenge in heavy petroleum analysis is the extreme complexity of the sample: thousands to tens of thousands of individual compounds spanning a wide range of boiling points, polarities, and molecular weights. Direct mass spectrometry (e.g., ESI, APPI, or FI) of whole crude or vacuum residue produces spectra that are convoluted and difficult to interpret. DPF-MS addresses this by performing **sequential physical fractionation** prior to MS analysis, thereby reducing ion suppression and enabling assignment of molecular features to specific boiling ranges and solubility classes.

### 2.2 Step 1 — Distillation

The sample is first subjected to distillation, typically using **simulated distillation (SimDist)** or **true boiling point (TBP) distillation** (ASTM D2892 or D5236 for vacuum residue). This separates the crude into narrow-boiling fractions (e.g., naphtha, kerosene, gas oil, vacuum gas oil, and vacuum residue). Each fraction is collected separately.

**Purpose:** Distillation provides a primary separation based on volatility, which correlates with molecular weight and carbon number for non-polar hydrocarbons. It also removes the light ends that would otherwise dominate the MS signal and suppress heavy components.

### 2.3 Step 2 — Precipitation

Each distillation cut (especially the heavier fractions) is then subjected to **solvent precipitation**. The most common protocol is **asphaltene precipitation** using n-alkane solvents (e.g., n-heptane or n-pentane) per ASTM D3279 or IP 143. The sample is mixed with a large excess of solvent; asphaltenes precipitate while maltenes remain soluble.

**Purpose:** Precipitation separates the sample into solubility classes (asphaltenes vs. maltenes, and optionally further sub-fractionation of maltenes into saturates, aromatics, and resins via SARA analysis). This is critical because asphaltenes are notoriously difficult to ionize and can cause severe ion suppression in MS.

### 2.4 Step 3 — Fractionation (Optional but Common)

The maltene fraction may be further separated by **liquid chromatography** (e.g., open-column or HPLC) into saturates, aromatics, and resins (SARA fractionation). This step is not strictly required for all DPF-MS workflows but is frequently included to further reduce complexity.

### 2.5 Step 4 — Mass Spectrometry

Each fraction is analyzed by mass spectrometry. The choice of ionization method depends on the fraction type:

- **Saturates:** Field ionization (FI) or electron ionization (EI) — FI is preferred for molecular ion preservation.
- **Aromatics:** Atmospheric pressure photoionization (APPI) or atmospheric pressure chemical ionization (APCI).
- **Resins and asphaltenes:** Electrospray ionization (ESI) for polar species, or laser desorption ionization (LDI) / matrix-assisted laser desorption ionization (MALDI) for high-mass species.

High-resolution MS (FT-ICR or Orbitrap) is typically employed to resolve isobaric species and enable elemental formula assignment.

### 2.6 Data Reconstruction

The mass spectra of individual fractions are then **recombined mathematically** (weighted by fraction yield) to reconstruct the full molecular composition of the original sample. This allows the analyst to map molecular features (e.g., heteroatom classes, aromaticity, carbon number distributions) onto the boiling-point and solubility dimensions.

---

## 3. Application Scenarios

### 3.1 Heavy Oil and Bitumen Characterization

DPF-MS is widely used to characterize the molecular composition of heavy oils and bitumen, where the combination of high viscosity, high heteroatom content, and broad boiling range makes direct analysis unreliable. The method provides:

- Carbon number and double-bond equivalent (DBE) distributions per boiling cut.
- Identification of sulfur, nitrogen, and oxygen compound classes.
- Asphaltene molecular weight distributions and aggregation behavior.

### 3.2 Refinery Process Development

In hydrocracking, visbreaking, and coking studies, DPF-MS is used to track compositional changes across process conditions. By comparing feed and product fractions, refiners can:

- Monitor conversion of heavy species into lighter products.
- Identify catalyst deactivation markers (e.g., accumulation of refractory polyaromatics).
- Optimize cut points and blending strategies.

### 3.3 Geochemical Correlation and Maturity Assessment

Petroleum geochemists use DPF-MS to correlate oils with source rocks by comparing biomarker distributions within specific boiling ranges. The fractionation step reduces matrix effects that would otherwise obscure trace biomarkers.

### 3.4 Environmental Forensics

In oil spill identification, DPF-MS can distinguish weathered vs. fresh oil by examining the boiling-range-specific depletion of light aromatics and the enrichment of heavy, recalcitrant species.

### 3.5 Quality Control of Synthetic Crudes

For synthetic crude oils derived from oil sands or coal-to-liquid processes, DPF-MS provides a compositional fingerprint that correlates with downstream refinery compatibility.

---

## 4. Relevant Literature

The following are representative publications that describe or apply DPF-MS or its constituent techniques:

1. **Rodgers, R. P., & Marshall, A. G. (2007).** "Petroleomics: Advanced characterization of petroleum-derived materials by Fourier transform ion cyclotron resonance mass spectrometry." *Energy & Fuels*, 21(3), 1122–1130. — Foundational review of high-resolution MS in petroleum analysis, including fractionation workflows.

2. **Cho, Y., et al. (2012).** "Application of atmospheric pressure photoionization hydrogen/deuterium exchange to the analysis of crude oils." *Energy & Fuels*, 26(5), 2558–2565. — Demonstrates APPI-based analysis of fractionated crude oils.

3. **Klein, G. C., et al. (2006).** "Mass spectral analysis of asphaltenes. I. Direct ESI FT-ICR MS of asphaltenes." *Energy & Fuels*, 20(5), 1965–1972. — Addresses asphaltene MS challenges and fractionation needs.

4. **Gaspar, A., et al. (2012).** "Comprehensive two-dimensional gas chromatography and mass spectrometry for the analysis of middle distillates." *Journal of Chromatography A*, 1240, 132–142. — While GC×GC-based, this illustrates the fractionation philosophy applied to lighter cuts.

5. **Purcell, J. M., et al. (2007).** "Sulfur speciation in petroleum: Atmospheric pressure photoionization mass analysis." *Energy & Fuels*, 21(5), 2869–2874. — Shows how fractionation improves sulfur compound identification.

6. **Mullins, O. C., et al. (2012).** *Asphaltenes, Heavy Oils, and Petroleomics.* Springer. — Comprehensive monograph covering precipitation and MS workflows.

7. **ASTM D2892** (TBP distillation) and **ASTM D5236** (vacuum potstill distillation) — Standard methods underpinning the distillation step.

---

## 5. Limitations of DPF-MS

### 5.1 Time and Throughput

DPF-MS is inherently **labor-intensive and slow**. TBP distillation of a crude can take 24–48 hours; precipitation and SARA fractionation add another 1–2 days. This makes the method unsuitable for high-throughput screening or real-time process monitoring.

### 5.2 Sample Loss and Artifact Formation

- **Volatile loss:** Light ends can be lost during solvent evaporation steps.
- **Thermal degradation:** High-boiling fractions may undergo cracking or polymerization during distillation, especially if vacuum is insufficient.
- **Precipitation artifacts:** The choice of solvent (n-pentane vs. n-heptane) changes the asphaltene yield and composition, leading to method-dependent results.

### 5.3 Incomplete Ionization Coverage

No single ionization method ionizes all compound classes equally. ESI is biased toward polar species; APPI favors aromatics; FI is limited to volatile, non-polar species. Consequently, some compound classes (e.g., high-molecular-weight saturates) may be under-represented or missed entirely.

### 5.4 Mass Range Limitations

FT-ICR and Orbitrap instruments have practical mass ranges (typically < 2,000 Da for routine analysis). Large asphaltene aggregates or high-mass resin species may exceed this range, requiring either fragmentation (which loses molecular information) or specialized ionization (e.g., MALDI), which introduces its own biases.

### 5.5 Data Complexity and Interpretation Burden

The volume of data generated is enormous. Assigning elemental formulas to every peak requires rigorous quality control (e.g., mass error thresholds, isotopic pattern matching). Reconstructing the whole-oil composition from fractions requires accurate yield measurements and careful weighting; small errors in fraction yields propagate into the final compositional model.

### 5.6 Reproducibility and Interlaboratory Variability

Because DPF-MS is a multi-step workflow with manual operations, interlaboratory reproducibility is often poor unless protocols are strictly standardized. Differences in distillation apparatus, solvent purity, precipitation time, and MS tuning can lead to significant discrepancies.

### 5.7 Not Truly "Comprehensive"

Despite the name, DPF-MS does not achieve complete molecular speciation. It provides **class-level and distribution-level** information rather than identification of every individual isomer. Isomeric compounds that co-elute in distillation and co-precipitate in the same solubility class remain unresolved.

---

## 6. Summary

DPF-MS is a powerful but resource-intensive approach for deep compositional analysis of heavy petroleum. Its strength lies in reducing sample complexity through orthogonal physical separations (volatility and solubility) before high-resolution mass spectrometry. It is best suited for research and development applications where depth of information outweighs throughput. Its principal weaknesses—time, labor, ionization bias, and reproducibility—must be carefully managed through rigorous protocol standardization and complementary analytical techniques (e.g., GC×GC, NMR) to obtain a balanced picture of the sample.

---

*Note: If you require specific numerical data, detailed experimental protocols, or a deeper dive into any single step (e.g., asphaltene precipitation kinetics or FT-ICR data processing), please specify, and I can elaborate further.*

## Naive RAG

Model: configured-llm-model

Retrieved / cited document candidates:
- 10.1016/j.fuel.2018.07.028; Molecular profiling of crude oil by using Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS); 2018
- 10.1021/acs.energyfuels.1c02355; Analytics Driving Kinetics: Advanced Mass Spectrometric Characterization of Petroleum Products; 2021
- 10.1016/j.trac.2025.118463; Toward accurate olefin quantification in plastic waste oils: Analytical strategies and future directions; 2025
- 10.1016/j.chroma.2005.05.106; Characterisation of middle-distillates by comprehensive two-dimensional gas chromatography (GC×GC): A powerful alternative for performing various standard analysis of middle-distillates; 2005
- 10.1016/j.fuel.2019.115852; Determination of the chemical compositions of heavy, medium, and light crude oils by using the Distillation, Precipitation, Fractionation Mass Spectrometry (DPF MS) method; 2019
- 10.1021/acs.energyfuels.2c00559; Characterization of the Compounds in Crude Oil That Preferentially Bind to Kaolinite; 2022
- 10.1021/acs.energyfuels.0c04286; Determination of the Chemical Compositions of Condensate-like Oils with Different API Gravities by Using the Distillation, Precipitation, Fractionation Mass Spectrometry (DPF MS) Method; 2021
- 10.1007/978-1-4684-6530-3_6; MOLECULAR REPRESENTATION OF COMPLEX HYDROCARBON FEEDSTOCKS T H R O U G H EFFICIENT CHARACTERIZATION AND STOCHASTIC ALGORITHMS; 1991
- 10.1021/ef1001502; Heavy Petroleum Composition. 2. Progression of the Boduszynski Model to the Limit of Distillation by Ultrahigh-Resolution FT-ICR Mass Spectrometry; 2010

Answer output:

# Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS): A Comprehensive Analysis

## Principles

DPF-MS is a separation and characterization method developed for molecular-level analysis of crude oil. The fundamental principle involves two sequential stages:

**1. Fractionation Stage:** Crude oil is separated into six distinct chemical classes based on volatility and solubility characteristics:
- **Fraction I (Distillate):** Volatile saturated hydrocarbons and light aromatic hydrocarbons
- **Fraction II (Asphaltenes):** Predominantly aromatic and heteroaromatic compounds
- **Fraction III (Heavy saturated hydrocarbons):** Cyclic, branched, and linear hydrocarbons
- **Fraction IV (Aromatic hydrocarbons):** Predominantly alkylaromatic compounds
- **Fraction V (Heteroaromatic compounds):** Mostly aromatic compounds containing heteroatoms
- **Fraction VI (Polar compounds):** Nonaromatic polar compounds

**2. Optimized Mass Spectrometry Stage:** Each fraction is analyzed using an ionization method specifically optimized for that fraction. The key innovation is the use of representative model compounds to tune ionization conditions so that all compounds within each fraction are ionized at approximately equal efficiency, generating only one type of ion (molecular ions, protonated molecules, or cations formed via hydride abstraction). This minimizes ionization bias and fragmentation, enabling semi-quantitative analysis.

**Instrumentation:** Two complementary high-resolution mass spectrometry platforms are employed:
- GC×GC/TOF MS with electron ionization (EI) for the volatile distillate (Fraction I)
- Atmospheric pressure chemical ionization (APCI) coupled with LQIT/orbitrap MS for nonvolatile fractions (II–VI), with different nebulization gases and solvent systems optimized per fraction

**Mass Balance Integration:** A distinguishing feature is the inclusion of mass balance calculations. Each fraction is weighed, and data from all fractions are consolidated into a single dataset, providing accurate overall crude oil characterization including average molecular weight, heteroatom content, ring and double bond equivalence (RDBE) values, and weight percentages of compound classes.

## Application Scenarios

1. **Crude Oil Molecular Profiling:** Determination of elemental compositions, molecular weights, and compound class distributions in whole crude oils and condensate-like oils of varying API gravities.

2. **Correlation Studies:** Establishing relationships between chemical composition and bulk physical properties such as API gravity, which is relevant for oil classification, quality assessment, and market pricing.

3. **Surface Interaction Studies:** The fractionation component has been adapted to investigate which crude oil compounds bind strongly to mineral surfaces (e.g., kaolinite), relevant to reservoir behavior and enhanced oil recovery.

4. **Semi-Quantitative Analysis:** Providing weight percentages of different compound classes, which is valuable for process design, feedstock evaluation, and quality control in the petrochemical industry.

## Relevant Literature

The method is documented across multiple publications from the same research group (references [18]–[22] in the provided context). Key studies include:

- Initial method development and validation papers describing the DPF-MS approach and optimized ionization methods for each fraction
- Application studies profiling five crude oils with different API gravities, correlating chemical composition with API gravity
- Proof-of-concept studies on condensate-like oils, including determination of average molecular weights and RDBE values
- Adaptation studies using the fractionation component to examine crude oil–mineral surface interactions
- Studies employing in-source collision-activated dissociation (ISCAD) to determine average alkyl chain lengths in aromatic compound classes

## Limitations

1. **Time Consumption:** The protocol is labor-intensive, requiring complete distillation separation, multiple extractions, and numerous mass spectrometry analyses per feedstock. This limits its suitability for routine refinery use.

2. **Semi-Quantitative Nature:** While ionization is optimized to reduce bias, the method provides semi-quantitative rather than fully quantitative results. Absolute concentrations are not directly determined.

3. **Complexity of Optimization:** Developing optimized ionization conditions for each fraction using model compounds requires significant expertise and method development effort.

4. **Applicability Range:** The method is best suited for lighter feedstocks. Heavier fractions present challenges due to lack of volatility and molecular complexity, which limit comprehensive molecular characterization.

5. **Instrumentation Requirements:** The method requires access to specialized high-resolution mass spectrometry instrumentation (GC×GC/TOF and LQIT/orbitrap), which may not be available in all laboratories.

6. **Data Interpretation Complexity:** Consolidating data from multiple fractions and different ionization methods into a coherent overall analysis requires sophisticated data processing and interpretation.

7. **Comparison with Conventional Methods:** While DPF-MS provides more detailed molecular information than conventional methods (e.g., GC-FID/TCD), it does not replace them for routine quantitative analysis where speed and simplicity are prioritized.

## QA-oriented RAG

Model: configured-llm-model

Retrieved / cited document candidates:
- 10.1016/j.fuel.2018.07.028; Molecular profiling of crude oil by using Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS); 2018
- 10.1021/acs.energyfuels.1c02355; Analytics Driving Kinetics: Advanced Mass Spectrometric Characterization of Petroleum Products; 2021
- 10.1016/j.trac.2025.118463; Toward accurate olefin quantification in plastic waste oils: Analytical strategies and future directions; 2025
- 10.1016/j.chroma.2005.05.106; Characterisation of middle-distillates by comprehensive two-dimensional gas chromatography (GC×GC): A powerful alternative for performing various standard analysis of middle-distillates; 2005
- 10.1016/j.fuel.2019.115852; Determination of the chemical compositions of heavy, medium, and light crude oils by using the Distillation, Precipitation, Fractionation Mass Spectrometry (DPF MS) method; 2019

Answer output:

# Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS): Principles, Applications, Literature, and Limitations

## Principles

DPF-MS is a separation and characterization method for crude oil that fractionates the sample into six major compound classes: volatile saturated hydrocarbons (distillate), asphaltenes, heavy saturated hydrocarbons, alkyl aromatic hydrocarbons, heteroaromatic compounds, and polar compounds [DOC 1][DOC 2]. The key novelty of the method is that optimized ionization and mass spectrometry conditions are developed separately for each fraction using representative model compounds, so that all compounds within a given fraction are ionized at approximately the same efficiency and generate only one type of ion (molecular ions, protonated molecules, or cations formed upon hydride abstraction) [DOC 1]. This minimizes ionization biases and fragmentation, making the method semi-quantitative [DOC 6].

Two mass spectrometry platforms are employed: GC×GC/(EI) TOF MS for the volatile distillate fraction, and positive-ion APCI coupled with LQIT/orbitrap (LTQ Orbitrap XL, resolution up to 100,000 at m/z 400) for the nonvolatile fractions [DOC 1][DOC 2]. For each nonvolatile fraction, the APCI parameters—including solvents, ionization reagents, tube lens voltages, sheath and auxiliary gases, and temperatures—are optimized individually [DOC 1]. After weighing and analyzing each fraction, mass balance considerations enable molecular profiling of the whole crude oil [DOC 1].

## Outputs and Capabilities

DPF-MS provides semi-quantitative molecular-level information, including elemental compositions, molecular weights, average molecular weight, ring and double bond equivalence (DBE) values, and percentage abundances of different compound classes [DOC 1][DOC 3]. In-source collision-activated dissociation (ISCAD) has been used to determine the average total number of carbon atoms in the alkyl chains of alkyl aromatic compounds [DOC 3].

## Application Scenarios

The method has been applied to:

- **Crude oils of varying API gravity**: DPF-MS was used to characterize light, medium, and heavy crude oils and to find correlations between chemical composition and API gravity [DOC 2].
- **Condensate-like oils**: The method was applied to five condensate-like oil samples of different API gravities, determining compound class weights, gravimetric wt %, average molecular weights, and DBE values. Correlations with API gravity were assigned when R² ≥ 0.9 [DOC 3].
- **Mineral–oil interaction studies**: The fractionation step of DPF-MS was used to investigate which crude oil compounds preferentially bind to kaolinite, a common clay mineral in oil reservoirs [DOC 6].

## Relevant Literature

Key publications include:

- Yerabolu et al., "Molecular Profiling of Crude Oil by Using Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS)," *Fuel*, 2018 [DOC 1][DOC 5].
- "Determination of the Chemical Compositions of Heavy, Medium, and Light Crude Oils by Using the DPF-MS Method," *Fuel*, 2019 [DOC 2].
- "Determination of the Chemical Compositions of Condensate-like Oils with Different API Gravities by Using the DPF-MS Method," *Energy & Fuels*, 2021 [DOC 3].
- "Characterization of the Compounds in Crude Oil That Preferentially Bind to Kaolinite," *Energy & Fuels*, 2022 [DOC 6].

## Limitations

The evidence provided does not explicitly enumerate limitations of the DPF-MS method. However, the following points can be inferred from the available material:

- The method is **semi-quantitative** rather than fully quantitative [DOC 1][DOC 6].
- The method requires **separate optimization of ionization and MS parameters for each fraction**, which is labor-intensive and requires representative model compounds [DOC 1][DOC 2].
- The evidence does not discuss specific limitations regarding sample size requirements, analysis time, or applicability to particular crude oil types beyond the samples tested.

## Iterative-review RAG

Model: configured-llm-model

Retrieved / cited document candidates:
- 10.1016/j.fuel.2018.07.028; ; 
- 10.1016/j.fuel.2018.07.028; Molecular profiling of crude oil by using Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS); 2018
- 10.1016/j.fuel.2019.115852; ; 
- 10.1016/j.fuel.2019.115852; Determination of the chemical compositions of heavy, medium, and light crude oils by using the Distillation, Precipitation, Fractionation Mass Spectrometry (DPF MS) method; 2019
- 10.1021/acs.energyfuels.0c04286; ; 
- 10.1021/acs.energyfuels.0c04286; Determination of the Chemical Compositions of Condensate-like Oils with Different API Gravities by Using the Distillation, Precipitation, Fractionation Mass Spectrometry (DPF MS) Method; 2021
- 10.1021/acs.energyfuels.2c00559; ; 
- 10.1021/acs.energyfuels.2c00559; Characterization of the Compounds in Crude Oil That Preferentially Bind to Kaolinite; 2022
- 10.1016/0016-0032(54)90925-4; ; 
- 10.1016/0016-0032(54)90925-4; Determination of Unsaturated Hydrocarbons by Low Voltage Mass Spectrometry; 1954

Answer output:

# Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS): Principles, Applications, and Limitations

## 1. Introduction and Principles

Distillation Precipitation Fractionation Mass Spectrometry (DPF-MS) is a separation and characterization method developed for the molecular-level analysis of crude oil [DOC 1]. The method addresses a fundamental challenge in petroleum analysis: crude oil is an extraordinarily complex mixture whose components exhibit widely differing ionization efficiencies, which complicates direct mass spectrometric characterization [DOC 1].

The core principle of DPF-MS is to reduce this complexity by separating crude oil into fractions containing compounds with similar chemical characteristics, followed by molecular-level characterization of each fraction using an optimized mass spectrometric method [DOC 1]. The novelty of the approach lies in the optimization of ionization and mass spectrometry conditions separately for each fraction, using representative model compounds, so that all compounds within a given fraction are ionized at approximately the same efficiency and yield stable ionized molecules [DOC 1]. This optimization makes the method semi-quantitative [DOC 1][DOC 5].

### 1.1 Fractionation Scheme

DPF-MS separates crude oil into six major compound classes [DOC 1][DOC 3]:

1. **Fraction I – Distillate (volatile hydrocarbons):** Contains predominantly volatile saturated hydrocarbons, along with light aromatic hydrocarbons [DOC 1]. Obtained by vacuum distillation at room temperature, with the receiving flask cooled using dry ice and acetone (< −70 °C) to collect and condense volatile compounds [DOC 3][DOC 4]. These compounds typically have boiling points lower than 170 °C [DOC 4].

2. **Fraction II – Asphaltenes:** Contains predominantly aromatic and heteroaromatic compounds [DOC 1]. Precipitated using n-hexane added in a 10:1 ratio to the sample after removal of volatile compounds, left undisturbed overnight, then filtered through a 0.45 μm Whatman PTFE membrane filter [DOC 4].

3. **Fraction III – Heavy saturated hydrocarbons:** Contains cyclic, branched, and linear hydrocarbons [DOC 1]. Obtained through chromatographic separation of the maltenes (the asphaltene-free residue) using a Combi-flash Rf 200 auto column system, followed by solid-phase extraction with a Si/CN-S-1.5 g cartridge to separate from alkyl aromatic hydrocarbons [DOC 3][DOC 4].

4. **Fraction IV – Aromatic hydrocarbons (alkyl aromatic hydrocarbons):** Contains predominantly alkylaromatic compounds [DOC 1]. Separated from heavy saturated hydrocarbons via solid-phase extraction [DOC 3][DOC 4].

5. **Fraction V – Heteroaromatic compounds:** Contains mostly aromatic compounds with heteroatoms [DOC 1]. Eluted with dichloromethane during chromatographic separation [DOC 3][DOC 4].

6. **Fraction VI – Polar compounds:** Contains nonaromatic polar compounds [DOC 1]. Eluted with isopropanol (2-propanol) during chromatographic separation [DOC 3][DOC 4].

### 1.2 Mass Spectrometric Analysis

Two complementary high-resolution mass spectrometry platforms are employed [DOC 1][DOC 3]:

- **GC×GC/(EI) TOF MS:** A high-resolution two-dimensional gas chromatograph coupled with electron ionization and a high-resolution time-of-flight mass spectrometer. This is used for the characterization of the volatile distillate (Fraction I) [DOC 1][DOC 3].

- **(APCI) LQIT/orbitrap:** A linear quadrupole ion trap coupled with a high-resolution orbitrap detector (LTQ Orbitrap XL, Thermo Fisher Scientific) with a maximum resolution of 100,000 at m/z 400. This is used for direct infusion positive ion mode atmospheric pressure chemical ionization (APCI) analysis of all non-distillate fractions [DOC 1][DOC 3].

For each nonvolatile fraction, optimized ionization methods—including specific solvents and ionization reagents, tube lens voltages, types of sheath and auxiliary gases, and temperatures—are selected to ionize all components with similar efficiency and to generate only one type of ion (molecular ions, protonated molecules, or cations formed upon hydride abstraction) [DOC 1]. The (+) APCI approach with different nebulization gases and solvent systems is used for each nonvolatile fraction [DOC 1].

### 1.3 Data Output

DPF-MS provides semi-quantitative molecular-level information, including [DOC 1]:

- Elemental compositions and molecular weights
- Accurate average molecular weight
- Ring and double bond equivalence (RDBE) values
- Percentage abundances of different compound classes

Following weighing and analysis of each fraction, consideration of mass balance facilitates the molecular profiling of the fractionated crude oil [DOC 1]. In addition, in-source collision-activated dissociation (ISCAD) has been utilized to determine the average total number of carbon atoms in the alkyl chains of compounds in the alkyl aromatic compound class, exploiting the fact that dealkylation is the prevailing fragmentation pathway for ions containing aromatic cores [DOC 4].

## 2. Application Scenarios

### 2.1 Molecular Profiling of Crude Oils with Varying API Gravity

A primary application of DPF-MS is the molecular profiling of crude oils with different API gravities. In one study, five crude oils of different API gravities were characterized, and correlations between chemical composition and API gravity were discussed—both for the crude oils as a whole and for individual compound classes [DOC 3]. In a subsequent proof-of-concept study, the method was applied to five condensate-like oil samples of different API gravities [DOC 4]. Notably, none of the condensate-like oils contained a detectable amount of asphaltenes, so only five compound classes were analyzed [DOC 4]. In that study, a linear correlation between chemical composition and API gravity was assigned only if the R² value (goodness of fit) was ≥ 0.9 [DOC 4].

### 2.2 Studying Compound–Mineral Interactions

The fractionation component of DPF-MS has been applied to investigate the types of compounds in crude oil that bind strongly to mineral surfaces. In one study, crude oil was mixed with kaolinite (a prototypical clay mineral commonly found in oil-producing reservoirs), and the mixture was centrifuged to generate nonbound and strongly bound oil components. The fractionation step of DPF-MS was then used to fractionate the components that were and were not strongly bound to kaolinite, with all fractions analyzed using the optimal ionization and MS methods developed for DPF-MS [DOC 5]. This application demonstrates the utility of DPF-MS beyond simple compositional analysis, extending to understanding interfacial phenomena relevant to reservoir behavior.

### 2.3 Improvement over SARA Fractionation

DPF-MS represents an improvement over traditional SARA (Saturates, Aromatics, Resins, Asphaltenes) fractionation. The SARA fractions remain complex mixtures of compounds rather than separate chemical classes—for example, the SARA "saturated" fraction contains a substantial amount of aromatic compounds [DOC 3]. By contrast, DPF-MS fractionates crude oil into six major compound classes with optimized ionization methods for each class, enabling more accurate determination of chemical compositions [DOC 3].

## 3. Relevant Literature and Methodological Context

The DPF-MS method was introduced in 2018 [DOC 1] and subsequently applied and refined in several studies:

- **2018 (Fuel):** Introduction of the DPF-MS method with detailed description of the six-fraction separation scheme and optimized ionization approaches [DOC 1].
- **2019 (Fuel):** Application to heavy, medium, and light crude oils with different API gravities, including detailed step-by-step fractionation procedures and correlations with API gravity [DOC 3].
- **2021 (Energy & Fuels):** Application to condensate-like oils with different API gravities, including the use of ISCAD to determine alkyl chain lengths and the establishment of an R² ≥ 0.9 threshold for assigning linear correlations [DOC 4].
- **2022 (Energy & Fuels):** Use of the fractionation component of DPF-MS to study compounds preferentially binding to kaolinite [DOC 5].

The method builds on a long history of combined separation–mass spectrometry approaches in petroleum analysis. For example, low-voltage mass spectrometry was used as early as 1954 for the determination of unsaturated hydrocarbons, where distillation and percolation of numerous distillation fractions were combined with mass spectrometric and infrared analyses [DOC 2]. That earlier approach required approximately 1.5 hours of instrument time versus at least 500 man-hours for the more complex conventional analysis, demonstrating the long-standing value of coupling separation with mass spectrometry for petroleum characterization [DOC 2]. However, the earlier method had serious limitations for determining aromatic molecular weight distributions in higher boiling ranges, which the low-voltage technique was designed to overcome [DOC 2]. DPF-MS extends this tradition with a more systematic fractionation scheme and fraction-specific ionization optimization.

## 4. Limitations

Several limitations of DPF-MS are evident from the supplied evidence:

1. **Semi-quantitative nature:** The method is described as semi-quantitative rather than fully quantitative. The semi-quantitative character arises because ionization optimization is performed using representative model compounds, which may not perfectly represent all components within a fraction [DOC 1][DOC 5].

2. **Complexity and time requirements:** The method involves multiple separation steps—vacuum distillation, asphaltene precipitation, chromatographic separation, and solid-phase extraction—each with specific equipment and conditions [DOC 3][DOC 4]. This multi-step procedure is inherently more labor-intensive than direct analysis methods.

3. **Sample-type limitations:** The method's applicability to samples lacking certain fractions is demonstrated by the condensate-like oil study, where no asphaltenes were detected [DOC 4]. While this did not prevent analysis, it highlights that the six-class scheme may not be universally applicable to all petroleum samples without modification.

4. **Correlation limitations:** In the condensate-like oil study, linear correlations between chemical composition and API gravity were assigned only when R² ≥ 0.9 [DOC 4], implying that many compositional parameters may not correlate strongly with API gravity, limiting the predictive utility of the method for bulk property estimation.

5. **Instrumentation requirements:** The method requires access to specialized high-resolution mass spectrometry instrumentation (GC×GC/TOF and LQIT/orbitrap) [DOC 1][DOC 3], which may not be available in all laboratories.

## 5. Research Gaps

Based on the supplied evidence, the following research gaps can be identified:

1. The evidence does not describe systematic validation of DPF-MS against independent quantitative methods (e.g., comparison with gravimetric or chromatographic reference methods) beyond the semi-quantitative framework described.

2. The application of DPF-MS to study compound–mineral interactions [DOC 5] suggests broader applicability, but the evidence does not indicate whether the method has been extended to other interfacial or process-related phenomena (e.g., emulsion stability, wettability, or enhanced oil recovery).

3. While correlations with API gravity have been explored [DOC 3][DOC 4], the evidence does not indicate whether DPF-MS compositional data have been correlated with other bulk properties (e.g., viscosity, density, or pour point).

4. The evidence does not address the reproducibility or inter-laboratory comparability of DPF-MS results, which would be important for establishing the method as a standard analytical tool.

## 6. Conclusion

DPF-MS is a sophisticated separation and characterization method that addresses the challenge of analyzing crude oil at the molecular level by combining systematic fractionation into six chemically similar compound classes with fraction-specific optimized ionization and high-resolution mass spectrometry. Its applications span molecular profiling of crude oils and condensate-like oils of varying API gravity, as well as studies of compound–mineral interactions. The method represents a significant improvement over SARA fractionation in terms of chemical class purity and ionization efficiency control. However, its semi-quantitative nature, multi-step complexity, and specialized instrumentation requirements constitute notable limitations, and further validation and broader application studies would strengthen its utility in petroleum science.
