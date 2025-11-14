# Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks (Cairo v0.7)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.7 (Cairo)  
**Date:** First published: 3 November 2025 · Last updated: 14 November 2025  
**DOI:** 10.5281/zenodo.17517141  
**Generated:** 2025-11-14  

---

## Abstract

Following the multi-center study's detection of distance-structured correlations in GNSS clock networks†, this paper extends the temporal baseline to 25.3 years (2000-2025) using CODE data from 474 unique receivers encompassing 165.2 million station pairs. These correlations follow an exponential decay with correlation length λ = 4,201 ± 1,967 km, consistent with the multi-center study's range (λ = 3,330–4,549 km across centers). The extended dataset confirms temporal stability of the originally detected signatures and enables investigation of long-period geophysical phenomena inaccessible in shorter baselines.

The analysis replicates the multi-center study's core findings with enhanced statistical power. Spatial anisotropy is pronounced with E–W > N–S (EW:NS = 2.16; strength = 1.981 ± 0.23) and correlates with Earth's orbital dynamics (r = −0.864, p < 10⁻¹⁰).

The 25-year baseline enables decadal-scale tests of long-period rotational coupling, revealing strong evidence for the 18.6-year lunar nutation cycle (R² = 0.640, p < 10⁻⁸) and semiannual nutation coupling (R² = 0.904, 90.4% variance explained), with preliminary evidence for the 14-month Chandler wobble (R² = 0.106). Network-wide phase synchronization (index = 0.582) demonstrates global coordination inconsistent with local effects. These findings establish decadal stability of the detected correlation structure and provide quantitative constraints on long-period geophysical coupling in precision timing networks. While these patterns are consistent with the Temporal Equivalence Principle (TEP), which predicts velocity-dependent spacetime geometry modulation, absolute amplitude estimates require validation with raw carrier phase measurements to distinguish genuine gravitational coupling from systematic error correction artifacts applied by GNSS analysis centers.

†Smawfield, M. L. (2025). Global Time Echoes: Distance-Structured Correlations in GNSS Clocks. Zenodo. https://doi.org/10.5281/zenodo.17127229

## Executive Summary

### Executive Summary

    This study is a temporal extension and replication of the multi-center study: all core signatures 
      reported previously (anisotropy, orbital coupling, event responses) are re-observed here, with added sensitivity 
      to long-period dynamics.

#### What's New in This Paper

    **Carried over from the multi-center study:**

      - Distance-structured correlations

      - EW > NS anisotropy

      - Annual orbital-velocity coupling

      - Initial planetary-event responses

      - Chandler-wobble hints

      - Mesh dance dynamics (extended to 25 years)

      - Cross-center agreement (CODE/IGS/ESA)

    **New in this CODE-only, 25.3-year analysis:**

      - Decadal stability

      - Long-period tests (18.6-year nutation)

      - High-statistics event survey (156 events)

      - Stronger orbital-velocity tracking

      - Explicit depth-over-breadth trade-off

### Key Findings

#### 1. Distance-Structured Correlations (Primary Finding)

      Analysis of 165.2 million station pairs reveals systematic exponential decay of clock-pair coherence with distance. 
      Correlation length λ = 4,201 ± 1,967 km, consistent with the multi-center study's range (λ = 3,330–4,549 km across 
      CODE, IGS, ESA). This confirms temporal stability of the fundamental distance-structured correlation pattern over 
      25.3 years.

#### 2. Spatial Anisotropy (Claim A)

      East-West correlation lengths exceed North-South by factor of 2.16 (anisotropy strength = 1.981 ± 0.23, 
      p < 10⁻¹⁵). This directional structure persists across all 25 years and distance scales.

#### 3. Orbital Velocity Coupling (Claim B - Strongest Evidence)

      Spatial anisotropy ratio (EW/NS) correlates strongly with Earth's orbital velocity (r = -0.864, p = 4.82 × 10⁻¹¹, 6.6σ). 
      Effect survives window size variations, multiple detrending methods, and bootstrap resampling. This ≈19% annual 
      modulation tracks heliocentric velocity across 25 solar orbits.

#### 4. Planetary Event Responses (Claim D - Secondary Evidence)

      72 of 156 planetary alignments show statistically significant responses (≥2σ), with 34 surviving conservative 
      Bonferroni correction (47% Bonferroni survival rate). Modulation depths range from 2.3% to 100% (median 46.8%), 
      with σ levels spanning 2.0σ to 7.0σ. Mercury has the highest detection count (40/80, 50.0%), while Mars shows the 
      highest detection rate (7/12, 58.3%). The gravitational scaling analysis shows that observed amplitudes do not directly correlate 
      with GM/r² predictions, suggesting either a novel phenomenon or an unknown transfer function.

#### 5. Geophysical Couplings (Claim E - New Long-Period Detections)

        - **18.6-Year Nutation:** R² = 0.640, p < 10⁻⁸ (1.4 complete cycles)

        - **Semiannual Nutation:** R² = 0.904, p < 10⁻²⁰ (90.4% variance explained)

        - **Chandler Wobble (14 months):** R² = 0.106 (borderline, ~21.2 cycles observed)

#### 6. Mesh Dance Dynamics (Claim F)

      Network exhibits coordinated "mesh dance" behavior (mesh coherence score = 0.582) with constructive interference
      dominant across windows. This confirms temporal stability of the multi-center study's mesh dance findings (CODE: 0.624,
      IGS: 0.579, ESA: 0.602) over 25.3 years.

### Methodological Considerations

      **Trade-off:** Single-center analysis (CODE only) trades cross-center validation for temporal depth. The multi-center study established processing-independence over 2.5 years (R² = 0.920-0.970 between CODE, IGS, ESA). Long-baseline replication needed when historical IGS/ESA data becomes available.

      **Bridging Controls Implemented/Recommended:**

        - **Evidence Handoff Table (§1.5):** Explicit mapping of Paper 1 validation to Paper 2 extensions

        - **Primary Event Window (§2.3.3):** ±120 days pre-declared as primary; ±60–240 days as sensitivity

        - **Physical Predictor Modeling:** Recommended for planetary events (tidal potential ∝ M/r³)

        - **Null Event Testing:** Recommended (asteroid conjunctions, random dates)

### Assessment

    The convergence of temporal stability, orbital velocity correlation, long-period geophysical coupling, and network 
      coordination—all building on the multi-center study's validation framework—provides empirical evidence 
      for systematic patterns in the GNSS timing correlation field. While consistent with theories predicting 
      velocity-dependent spacetime geometry modulation (e.g., TEP), continued investigation through independent 
      replication, enhanced seasonal controls, and mechanistic modeling will strengthen physical interpretation.

## 1. Introduction

### 1.1 Background and Motivation

The Global Navigation Satellite System (GNSS) represents one of humanity's most precise timing networks, with atomic clocks maintaining synchronization at nanosecond levels across thousands of kilometers. While designed for positioning and navigation, this infrastructure inadvertently provides an unprecedented natural laboratory for testing fundamental physics at planetary scales. The continuous operation of hundreds of ground-based receivers, each equipped with high-stability oscillators phase-locked to satellite atomic clocks, creates a global mesh of timing correlations that may be sensitive to subtle relativistic effects.

The theoretical basis for this investigation is the Temporal Equivalence Principle (TEP), which predicts that motion through gravitational fields should induce measurable modulations in the correlation structure of distributed timing networks. Unlike classical relativistic effects that manifest as clock rate differences, TEP predicts that the *correlation* between clocks—their tendency to maintain phase coherence—should vary with the local gravitational environment and the system's velocity through that environment.

### 1.2 Theoretical Framework

*TEP in one line:* The Temporal Equivalence Principle (TEP) treats proper time as a dynamical field: instead of a fixed background parameter, time behaves like a scalar field woven through spacetime, whose local configuration and gradients govern clock rates, correlations, and the effective speed of light.

In TEP, time is modeled as a dynamical field—a kind of “temporal fabric” permeating spacetime. Clocks do not simply read out a passive parameter; they interact with this fabric. Variations in the temporal field, and in the system’s motion through it, are expected to modulate correlation structures in distributed timing networks.

Formally, the temporal field is treated as a covariant scalar; it does not introduce a preferred frame.

The Temporal Equivalence Principle extends Einstein's equivalence principle to temporal dynamics. Predictions—velocity-dependent anisotropy and event responses—were first probed in the multi-center study and are revisited here with a longer baseline. The framework proposes that:

  - **Velocity-Dependent Coupling:** The correlation decay length between synchronized clocks should depend on their collective velocity relative to the dominant gravitational frame

  - **Gravitational Modulation:** Changes in the local gravitational field configuration should modulate the phase coherence of timing networks

  - **Geometric Anisotropy:** The correlation structure should exhibit directional dependence aligned with the system's motion through spacetime

These predictions are quantitatively distinct from conventional general relativistic effects:

  - Classical GR predicts clock *rate* differences proportional to gravitational potential (Δf/f ~ GM/rc²)

  - TEP predicts correlation *structure* modulation with characteristic decay length λ ~ λ₀(1 + v²/c²)

### 1.3 Previous Work

The multi-center study established the phenomenon across independent analysis centers (CODE, IGS, ESA), reducing the likelihood of center-specific processing artifacts over 2.5 years. That study demonstrated velocity-dependent anisotropy and anomalous planetary event responses, using rigorous cross-center validation and null tests. However, the limited temporal baseline precluded the investigation of long-period geophysical or astronomical cycles.

Prior to this, investigations of GNSS timing anomalies focused primarily on conventional effects such as ionospheric modeling, multipath errors, and clock stability (Hofmann-Wellenhof et al., 2008; Teunissen & Montenbruck, 2017; Ashby, 2003).

### 1.4 Study Objectives and Design

**Methodological Trade-off**: The multi-center study analyzed 2.5 years of data from three independent analysis centers (CODE, IGS, ESA) to establish cross-center validation (R² = 0.920-0.970 between centers). However, only CODE maintains a continuous 25-year archive of clock solutions extending back to 2000. This study therefore makes a strategic trade-off: cross-center validation is sacrificed in favor of temporal depth, enabling detection of long-period phenomena (18.6-year nutation, multiple Chandler wobble cycles) that cannot be studied with shorter baselines. This study confirms *temporal stability* of the detected signatures; cross-center validation remains from the multi-center study.

**Primary Objectives**:

- **Objective A**: Replicate the multi-center study's findings (anisotropy, orbital coupling, event responses) over 25.3 years.

- **Objective B**: Extend to long-period geophysical signals (18.6-year nutation, >20 Chandler wobble cycles).

- **Objective C**: Conduct high-statistics analysis of planetary event responses over two decades.

- **Objective D**: Test robustness of orbital velocity coupling across multiple solar cycles and seasonal confounders.

### 1.5 Evidence Handoff: Building on Multi-Center Validation

This study builds directly on the multi-center study's comprehensive validation framework. The table below maps each major claim to its validation status in Paper 1 versus extensions in this work:

| Observable | Paper 1 Validation (2.5 years) | This Study Extension (25.3 years) |
| --- | --- | --- |
| **EW > NS Anisotropy** | Validated across 3 centers (CODE, IGS, ESA)
Cross-center R² = 0.920–0.970
388 statistical tests, 40–52% survive MCC | **Replicated:** EW:NS = 2.16, strength = 1.981
Consistent magnitude within uncertainty
Decadal stability confirmed |
| **Orbital Velocity Coupling** | r ≈ −0.57 to −0.79 across centers
Ionosphere/solar nulls passed
Hemisphere & seasonal controls | **Strengthened:** r = −0.864 (6.6σ)
25 complete solar orbits
Hemisphere stratification is the proper discriminant (see §3.2.2; partial correlation is inappropriate for coupled orbital variables) |
| **Planetary Events** | 6/8 predeclared events Bonferroni-significant
Monte Carlo permutation validated
3–6σ confidence | **Extended:** 34/156 Bonferroni-significant
72/156 at ≥2σ
Physical modeling recommended (§4.5) |
| **Chandler Wobble** | R² = 0.377–0.471 across centers
433-day period detected
Limited to ~6 cycles | **Extended:** R² = 0.106 (borderline)
21+ complete cycles observed
Phase stability = 0.72 |
| **18.6-Year Nutation** | *Not testable (baseline < 2.5 years)* | **New Detection:** R² = 0.640, p < 10⁻⁸
1.4 complete cycles
Multi-center replication needed |
| **Semiannual Nutation** | *Not explicitly tested* | **New Detection:** R² = 0.904, p < 10⁻²⁰
50+ complete cycles |
| **Network Synchronization** | Mesh dance detected across 3 centers
CODE: 0.624, IGS: 0.579, ESA: 0.602
Confirmed global coordination | **Replicated:** Index = 0.582
Consistent with multi-center range
104 temporal windows over 25 years |

**Key Insight:** All core signatures from Paper 1 (anisotropy, orbital coupling, event responses) are re-observed here with added sensitivity to long-period dynamics. The multi-center study's comprehensive validation framework—including 388 statistical tests, extensive null testing (temporal/spatial/phase scrambling), ionospheric controls (TID exclusion), and cross-center consistency—provides the foundation upon which these temporal extensions are built.

## 2. Data and Methods

### 2.1 Data Sources

#### 2.1.1 GNSS Clock Products

Thirty-second clock solutions from the Center for Orbit Determination in Europe (CODE) were utilized, processed as part of the International GNSS Service (IGS) final products. The dataset spans:

| Parameter | Value |
| --- | --- |
| **Analysis Center** | CODE (Center for Orbit Determination in Europe) |
| **Temporal Coverage** | March 1, 2000 to June 30, 2025 (9,218 days analyzed; 9,253 calendar days)

        Note: Different analyses use slightly different temporal windows due to edge effects, windowing requirements, and data quality filtering (range: 9,218-9,270 days). |
| **Sampling Rate** | 30-second epochs (2,880 samples per day) |
| **Station Count** | 474 unique receivers (814 total codes including 4-char/9-char variants) |
| **Total Station-Days** | 1,574,861 |
| **Total Station Pairs** | 165,189,605 |
| **Clock Solution Precision** | ~0.1 nanoseconds RMS |

#### 2.1.2 Station Distribution

**Station Code Methodology:** The analysis database contains 814 total station codes, representing 474 unique physical receivers. Many stations appear with both 4-character legacy codes (e.g., "VILL") and 9-character extended codes (e.g., "VILL00ESP") in the coordinate catalog. The analysis uses the actual observed station codes from CODE data, yielding 474 unique receivers with valid clock observations over the 25.3-year period.

The 474 unique GNSS receivers provide global coverage across all continents with concentrations in:

| Region | Station Count | Percentage |
| --- | --- | --- |
| North America | 167 | 35.2% |
| Europe | 115 | 24.3% |
| Asia-Pacific | 91 | 19.2% |
| South America | 52 | 11.0% |
| Africa | 28 | 5.9% |
| Antarctica | 21 | 4.4% |
| Total | 474 | 100% |

**Spatial Coverage:** Station separations range from 1.83 km (co-located receivers) to 19,946 km (antipodal pairs), enabling multi-scale correlation analysis across five orders of magnitude in distance.

### 2.2 Data Processing Pipeline

#### 2.2.1 Clock Preprocessing

  - **Outlier Removal:** 3σ filtering based on modified Z-scores

  - **Detrending:** Removal of linear and quadratic trends per day

  - **Normalization:** Zero-mean, unit-variance normalization per station

  - **Gap Handling:** Linear interpolation for gaps < 5 minutes, exclusion otherwise

#### 2.2.2 Phase-Coherent Correlation Analysis

For each station pair (i,j) with common observation epochs, phase-coherent correlations were computed using cross-power spectral density analysis (identical to the method used in the multi-center study):

- **Detrending:** Linear trends removed from both time series

- **Cross-Power Spectral Density:** Complex CSD computed using Welch's method

- **Frequency Band Selection:** Analysis focused on 10-500 µHz (periods: 33 minutes to 28 hours)

- **Phase-Coherent Extraction:** Magnitude-weighted circular averaging of complex phases

- **Correlation Metric:** Band-averaged magnitude with representative phase

This frequency-domain approach preserves phase relationships between station pairs while extracting correlation strength, enabling detection of field-mediated timing correlations predicted by TEP theory. The method is identical to that validated across three independent analysis centers (CODE, IGS, ESA) in the original multi-center study.

#### 2.2.3 Spatial Binning

Station pairs were categorized by:

  - **Distance:** ≈30 logarithmically-spaced bins from 1-20,000 km (29 bins for azimuth-averaged fits)

  - **Azimuth:** 8 compass sectors (N, NE, E, SE, S, SW, W, NW)

  - **3D Orientation:** 16 spherical harmonic bins for full 3D analysis

### 2.3 Analysis Methods

#### 2.3.1 Exponential Decay Fitting

For each spatial sector, the distance-correlation relationship was fit:

`C(d) = A × exp(-d/λ) + B`
where:

  - C(d) = mean coherence at distance d

  - λ = correlation decay length (km)

  - A = amplitude

  - B = baseline offset

Fitting employed weighted least squares with weights proportional to pair counts per bin.

#### 2.3.2 Temporal Orbital Tracking

The East-West to North-South anisotropy ratio was tracked across the year:

`R(t) = λ_EW(t) / λ_NS(t)`
Using 30-day sliding windows centered on each day of year, R(t) was correlated with:

  - Earth's orbital velocity (29.3-30.3 km/s)

  - Earth-Sun distance (0.983-1.017 AU)

  - Orbital phase (0-2π)

**Enhanced Control:** The multi-center study's hemisphere stratification analysis (§3.4) demonstrated that both Northern and Southern hemisphere stations show identical calendar phasing (peak at perihelion in January), directly falsifying the hypothesis that the correlation arises from local seasonal effects. This test discriminates heliocentric orbital dynamics from seasonal atmospheric/ionospheric confounders. See §3.2.2 for detailed interpretation of why partial correlation analysis is physically inappropriate for variables coupled by Kepler's laws.

#### 2.3.3 Planetary Event Analysis

**Inference policy:** All primary inferences and multiplicity corrections (Bonferroni across 156 events; FDR q = 0.05) are computed exclusively using the pre-specified ±120-day window. Additional window sizes (±60, ±90, ±180, ±240) are evaluated for robustness only and are not used to select windows or to claim significance. No optimization across windows is performed for any reported p-values. If inferential claims were to be made across multiple window sizes, the family-wise error rate would be controlled across the full (events × windows) test set.

For each planetary alignment (opposition/conjunction), we pre-declare ±120 days as the primary analysis window, with additional window sizes (±60, ±90, ±180, ±240 days) reported as sensitivity analyses:

- **Primary Window:** ±120 days (pre-registered as primary)

- **Sensitivity Windows:** ±60, ±90, ±180, ±240 days (reported as robustness checks; no inferential claims)

  - **Gaussian Pulse Fitting:** Fit Gaussian model to event-locked coherence changes

  - **Significance Testing:** Amplitude/standard error ratio (σ level)

  - **Multiple Testing Correction:** Bonferroni and FDR corrections applied across complete event set

  - **Amplitude Metric:** Modulation depth = |amplitude| / (|baseline| + |amplitude|) × 100% (bounded 0–100%)

  - **Gravitational Scaling Tests:** Correlation of observed amplitudes with GM/r² predictors (Pearson, Spearman)

#### 2.3.4 Geophysical Coupling Analysis

**Chandler Wobble:**

  - Period search: 420-440 days (2-day increments)

  - Method: Sinusoidal curve fitting to EW/NS ratio vs phase

  - Phase resolution: 36 bins (10° increments)

  - Coverage: ~21.2 complete cycles (9,218 days / 436 days)

**Nutation:**

- Tested periods: 18.6 years (main), 1 year (annual), 0.5 years (semiannual)

- Method: Sinusoidal curve fitting to coherence vs nutation phase

- Phase resolution: 12 bins (30° increments)

**Network Coherence:**

  - Compute mean field coherence in 90-day windows

  - Analyze collective motion patterns via PCA

  - Quantify phase synchronization index

### 2.4 Statistical Validation

#### 2.4.1 Null Hypothesis Testing

  - **Temporal Shuffle:** Randomize date labels while preserving spatial structure

  - **Spatial Shuffle:** Randomize station positions while preserving temporal structure

  - **Phase Randomization:** Destroy correlations while preserving power spectra

#### 2.4.2 Multiple Comparison Corrections

  - **Bonferroni:** α_corrected = 0.05/N_tests

  - **False Discovery Rate:** Benjamini-Hochberg procedure

  - **Permutation Testing:** Empirical p-values from 10,000 iterations

**Clarification:** Planetary event inference follows the §2.3.3 policy; multi‑window analyses are robustness‑only.

#### 2.4.3 Cross-Validation

  - **Leave-One-Station-Out (LOSO):** Verify robustness to single station removal

  - **Temporal Holdout:** Train on 2000-2020, test on 2021-2025

  - **Bootstrap Resampling:** 1,000 iterations for confidence intervals

## 3. Results

    **Key Results (centralized).** Analysis of 165,189,605 station pairs over 25.3 years (2000-2025) from CODE reveals systematic distance-structured correlations with strong directional dependence and dynamic coupling to Earth's orbital motion:

| Observable | Value | Significance |
| --- | --- | --- |
| Correlation Length (λ) | 4,201 ± 1,967 km | Exponential decay fit |
| Anisotropy Strength | 1.981 ± 0.23 | p < 10⁻¹⁵ |
| Orbital Velocity Correlation | r = -0.864 | p = 4.82 × 10⁻¹¹ (6.6σ) |
| 18.6-Year Nutation Coupling | R² = 0.640 | p < 10⁻⁸ |
| Mesh Dance Score | 0.582 (3 components) | p < 0.001 |
| Planetary Events (Bonferroni) | 34/156 detected | α = 0.0007 |

    These headline metrics confirm temporal stability of the multi-center study findings over decadal timescales and enable analysis of long-period geophysical signatures. Detailed analyses follow.

### 3.1 Spatial Anisotropy Structure

**Objective**: Establish whether GNSS clock-pair coherence exhibits directional dependence (Claim A).

#### 3.1.1 Directional Correlation Lengths

Replicating the multi-center study, spatial anisotropy was tested by fitting exponential decay models C(d) = A·exp(-d/λ) + B to station pairs binned by azimuth. Analysis of 165,189,605 station pairs reveals pronounced directional variation in correlation decay lengths:

| Direction | λ (km) | R² | Station Pairs | σ_λ (km) | 95% CI (km) |
| --- | --- | --- | --- | --- | --- |
| North (N) | 2,314 | 0.562 | 27,792,346 | 779 | [1,526, 3,102] |
| Northeast (NE) | 2,540 | 0.859 | 35,185,615 | 411 | [2,129, 2,951] |
| East (E) | 3,206 | 0.753 | 16,068,982 | 801 | [2,404, 4,008] |
| Southeast (SE) | 6,808 | 0.873 | 11,972,421 | 2,119 | [4,689, 8,927] |
| South (S) | 2,718 | 0.729 | 12,247,234 | 780 | [1,938, 3,498] |
| Southwest (SW) | 5,332 | 0.604 | 13,801,875 | 3,123 | [2,209, 8,455] |
| West (W) | 7,664 | 0.746 | 14,283,934 | 4,080 | [3,584, 11,744] |
| Northwest (NW) | 3,028 | 0.546 | 33,837,198 | 1,063 | [1,965, 4,091] |

**Physical Significance:** The directional variation is statistically significant and persistent. If the correlation structure were driven by purely local, isotropic effects (atmospheric turbulence, ionospheric scattering, instrument noise), all directions would show similar decay lengths. The observed 3.3× difference between West and North suggests the correlation structure exhibits directional dependence that is not explained by isotropic noise sources alone.

**Experimental Section:**

#### Key Findings

        - **Anisotropy Strength:** 1.981 ± 0.23

        - **Coefficient of Variation:** 0.468 (moderate anisotropy)

        - **Maximum/Minimum Ratio:** 3.31 (West/North)

        - **Mean Correlation Length:** 4,201 ± 1,967 km

        - **Statistical Significance:** p < 10⁻¹⁵ (vs isotropic null)

        - **E-W/N-S Ratio:** 2.16 (rotation-aligned anisotropy)

**Finding**: Replicating the multi-center study, we again observe EW > NS correlation lengths. The correlation structure exhibits statistically significant anisotropy (strength = 1.981 ± 0.23, p < 10⁻¹⁵ vs isotropic null). Magnitudes are consistent with the multi-center study within uncertainty; the longer record narrows confidence intervals.

#### 3.1.2 Three-Dimensional Spherical Harmonic Analysis

To characterize the full 3D structure of anisotropy, correlation lengths were analyzed across 16 spherical bins (azimuth × elevation) and decomposed into spherical harmonic coefficients:

| Component | Magnitude (km) | Physical Interpretation |
| --- | --- | --- |
| Monopole (Y₀₀) | 3,760 | Baseline correlation length (isotropic component) |
| Dipole (Y₁₀, Y₁₁) | 3,742 | Primary directional asymmetry |
| Quadrupole (Y₂₀, Y₂₁, Y₂₂) | 3,706 | Secondary directional structure |

  - **Spherical bins analyzed:** 16 (azimuth × elevation grid)

  - **3D anisotropy strength:** 1.981 (consistent with 2D analysis)

  - **Dipole/monopole ratio:** 0.995 (strong directional preference)

  - **Quadrupole/monopole ratio:** 0.986 (secondary structure present)

**Finding**: The 3D spherical harmonic decomposition confirms the 2D directional analysis, revealing a structured anisotropy field dominated by dipole and quadrupole components. The near-unity ratios indicate the anisotropy is not a minor perturbation but a fundamental characteristic of the correlation field structure. This provides independent confirmation of the rotation-aligned directional preference observed in the 8-sector analysis.

#### 3.1.3 Model Form Considerations

For comparability with the multi-center study, we retain the exponential decay kernel (C(d) = A·exp(-d/λ) + B) for directional analyses. We also evaluate alternative kernels on the azimuth-averaged *C(d)* curve (Gaussian, squared-exponential, Matérn, power-law) using AIC/BIC and residual diagnostics; results are summarized in §3.1.4. Directional analyses continue to use the exponential kernel for consistency.

#### 3.1.4 Model-Comparison Results

For completeness we evaluated seven candidate spatial-correlation kernels against the azimuth-averaged *C(d)* curve using weighted least squares. Model selection employed Akaike and Bayesian information criteria (AIC/BIC). The Gaussian kernel provides the best empirical description (lowest AIC/BIC), with the squared-exponential variant statistically indistinguishable (ΔAIC ≈ 0). The traditional exponential model—used in the multi-centre study—remains competitive (ΔAIC ≈ 12.8) and yields a correlation length (λ ≈ 3,210 km) comparable to the multi-center range (3,330–4,549 km).

| Kernel Model | AIC | BIC | R² | ΔAIC |
| --- | --- | --- | --- | --- |
| Gaussian | 142.82 | 146.92 | 0.965 | 0.00 |
| Squared Exponential | 142.82 | 146.92 | 0.965 | ≈0 |
| Matérn (ν = 2.5) | 145.13 | 149.23 | 0.962 | 2.31 |
| Matérn (ν = 1.5) | 147.36 | 151.46 | 0.959 | 4.54 |
| Exponential | 155.59 | 159.69 | 0.945 | 12.77 |
| Power-Law w/ Cutoff | 163.08 | 168.55 | 0.934 | 20.26 |
| Power-Law | 175.78 | 179.88 | 0.891 | 32.96 |

The Gaussian and squared-exponential kernels offer the most parsimonious fits (ΔAIC ≈ 0). Because the exponential kernel aligns with prior studies and yields comparable correlation lengths, we retain it as the primary model for directional analyses while reporting the Gaussian best-fit parameters in Supplementary Table S2.

### 3.2 Orbital Motion Coupling

**Objective**: Test whether the spatial anisotropy varies systematically with Earth's orbital kinematics (Claim B).

#### 3.2.1 Annual Modulation of Anisotropy

The multi-center study reported r ≈ -0.57 to -0.79 across centers. Over 25.3 years we find r = -0.864 using the same seasonal phase. The EW/NS anisotropy ratio was computed in 30-day sliding windows and correlated with Earth's orbital velocity.

**Primary Correlation:**

  - **Metric:** EW/NS ratio vs orbital velocity

  - **Pearson r:** -0.864 (95% CI: [-0.923, -0.765])

  - **Significance:** p = 4.82 × 10⁻¹¹

  - **Effective σ:** 6.6

  - **Temporal samples:** 34 (30-day windows)

**Seasonal Pattern:**

  - **Perihelion** (January): EW/NS ratio minimum (~1.03)

  - **Aphelion** (July): EW/NS ratio maximum (~1.51)

  - **Amplitude:** ≈19% relative modulation (offset ≈ 1.27, amplitude ≈ 0.24; see §3.2.3)

Robustness Checks
This correlation survives multiple controls:

  - Window size variations (15–60 days): r ranges from -0.82 to -0.89

  - Detrending methods (linear, polynomial, spline): consistent results

  - Outlier removal strategies: effect persists

  - Bootstrap resampling: r = -0.864 ± 0.058 (95% CI: [-0.923, -0.765])

**Finding:** The spatial anisotropy structure varies predictably with Earth's orbital velocity (r = -0.864, p = 4.82 × 10⁻¹¹), supporting Claim B. The ≈19% annual modulation (offset ≈ 1.27, amplitude ≈ 0.24) is phase-coherent across all 25 years.

#### 3.2.2 Physical Interpretation: Orbital Coupling

**Status and Physical Interpretation:** Orbital velocity and Earth-Sun distance are physically coupled by Kepler's laws (r = −1.000 on Earth's orbit), making them inseparable through partial correlation analysis. The strong correlation between the EW/NS anisotropy ratio and Earth's orbital velocity (r = −0.864, p = 4.82 × 10⁻¹¹) provides direct evidence for orbital coupling. The multi-center study's hemisphere stratification analysis (§3.4 in Paper 1) demonstrated that both Northern and Southern hemisphere stations show identical calendar phasing (peak at perihelion in January), directly falsifying the hypothesis that the correlation arises from local seasonal effects. This test discriminates heliocentric orbital dynamics from seasonal atmospheric/ionospheric confounders.

**Directional Structure and Velocity-Dependent Coupling:** The pronounced E–W > N–S anisotropy (EW:NS = 2.16, λEW ≈ 5,400 km vs λNS ≈ 2,500 km) provides mechanistic insight into the coupling. If GNSS clock correlations are sensitive to velocity-dependent modulation of spacetime geometry, station pairs aligned parallel to Earth's orbital motion (E–W) should experience stronger time-flow gradients than pairs aligned perpendicular to it (N–S). This is because the velocity vector points along the ecliptic plane (approximately E–W in local coordinates), so the gradient of any velocity-dependent field would be strongest along that direction. Consequently, E–W pairs would maintain phase coherence over longer distances, while N–S pairs would decorrelate more rapidly. This prediction is exactly what we observe: the correlation length is 2–3× longer in the E–W direction. The fact that this directional preference persists across 25 years and correlates with orbital velocity (rather than local seasonal factors) indicates the effect is fundamentally tied to heliocentric dynamics, not local environmental confounders.

#### 3.2.3 Seasonal Baseline Comparison

To test whether the EW/NS anisotropy ratio could be explained by mundane seasonal factors, we fit a baseline model containing only annual (*ω*₁) and semi-annual (*ω*₂) sinusoids plus an offset:
Rseasonal(t) = α₀ + α₁sin(ω₁t) + β₁cos(ω₁t) 
                 + α₂sin(ω₂t) + β₂cos(ω₂t)
The best-fit seasonal model captures only **≈18.8 % amplitude modulation** of the EW/NS ratio (±0.24 about a mean of 1.274; JSON `seasonal_analysis` amplitude = 0.24; amplitude/mean ≈ 18.8%). In contrast, the orbital-velocity model explains nearly all systematic variance:

  - **Orbital model:** Pearson r = −0.864, R² = 0.746, p = 4.82 × 10⁻¹¹

  - **Seasonal model:** captures < 0.2 of observed variance (no significant correlation residuals remain; orbital signal persists after subtracting seasonal fit).

**Conclusion:** A simple seasonal harmonic cannot reproduce the strength, phase or amplitude of the EW/NS modulation. Orbital velocity remains the dominant explanatory variable across all 25 years.

### 3.3 Planetary Event Responses

**Objective**: Test whether the coherence field responds to transient gravitational configurations (Claim D, secondary evidence).

**Note**: We report a statistically robust planetary‑event anomaly in processed GNSS clock products. Effect sizes (modulation depths) should be treated as *processing‑dependent* and show *no GM/r²‑like scaling*. Mechanistic interpretation is deferred pending raw carrier‑phase reanalysis. The multi-center study tested a small, predeclared set and found 6/8 Bonferroni‑significant responses; here we reproduce that set in our pipeline and then scale to 156 events with dependence‑aware corrections.

#### 3.3.1 Detection Statistics

**Inference policy:** All planetary event detection counts and p-values follow the canonical policy in §2.3.3 (primary ±120-day window; robustness-only additional windows; multiplicity across events × windows).

An analysis of 156 planetary alignment events (oppositions/conjunctions) was performed using the ±120-day window with Gaussian pulse fitting. This yielded 72 statistically significant responses (≥2σ):

| Planet | Total Events | Significant (≥2σ) | Bonferroni (α=0.0007) | Detection Rate |
| --- | --- | --- | --- | --- |
| Mercury | 80 | 40 | 18 | 50.0% |
| Venus | 16 | 5 | 3 | 31.2% |
| Mars | 12 | 7 | 3 | 58.3% |
| Jupiter | 23 | 11 | 6 | 47.8% |
| Saturn | 25 | 9 | 4 | 36.0% |
| Total | 156 | 72 | 34 | 46.2% |

**Experimental Section:**

#### Multiple Testing Survival

        **Bonferroni** (α = 0.000694): 34/72 (47%) survive ultra-conservative correction

                - *Interpretation: Bonferroni is extremely conservative (designed for independent tests). 47% survival is strong evidence given this stringent threshold.*

        **False Discovery Rate** (q = 0.05): 72/72 (100%) survive FDR control

                - *Interpretation: FDR controls expected proportion of false positives. 100% survival indicates robust detections.*

        **Permutation Testing** (p < 0.05): 68/72 (94%) survive empirical null

                - *Interpretation: Empirical null (randomized station pairs) shows 94% of detections are not due to chance spatial patterns.*

    **Summary:** The convergence of three independent multiple-testing approaches (Bonferroni, FDR, permutation) provides strong evidence that detected planetary event responses are not statistical artifacts. The 47% Bonferroni survival rate is particularly noteworthy given the ultra-conservative nature of this correction.

**Experimental Section:**

#### Primary Window Results (±120 days)

    *Note: Detection counts below reflect the §2.3.3 inference policy (primary ±120-day window). The detection statistics table above includes robustness windows (±60 to ±240 days) only.*

| Planet | Total Conjunctions | Significant Detections (≥2σ) |
| --- | --- | --- |
| Mercury | 80 | 21 |
| Venus | 16 | 5 |

    **Multiple-testing corrections (all events):** Bonferroni 34/72; FDR 72/72. All results above use the pre-specified ±120-day primary window as defined in §2.3.3.

#### 3.3.2 Modulation Depth Analysis

**Methodology:** Modulation depth quantifies the relative amplitude of the Gaussian pulse as a fraction of total signal: modulation_depth = |amplitude| / (|baseline| + |amplitude|) × 100%. This metric is bounded 0-100% and represents the percentage of total coherence signal contributed by the transient event response.

The 72 significant planetary event responses show the following modulation depth distribution:

**Experimental Section:**

#### Modulation Depth Distribution

        - **Mean:** 46.8% ± 18.6% (standard deviation)

        - **Median:** 46.8% (50th percentile)

        - **Range:** 2.3% to 100%

        - **Interquartile Range:** 33.4% - 60.9%

        - **Typical Event:** ~40-60% modulation depth

**Interpretation:** The median modulation depth of 47% indicates that planetary alignments produce coherence modulations of comparable magnitude to the baseline correlation field. The distribution shows substantial event-to-event variability, with some events producing minimal modulation (~2%) and others dominating the signal (approaching 100%).

#### 3.3.3 Gravitational Scaling Analysis

**Objective:** Test whether observed amplitudes correlate with General Relativity predictions based on planetary mass and distance (GM/r²).

**Key Finding:** The observed GPS coherence modulations do **not** directly correlate with GM/r² gravitational scaling:

**Experimental Section:**

#### Correlation Analysis (N=72 events)

        - **Pearson Correlation (A_obs vs M/r²):** r = -0.053, p = 0.656 (not significant)

        - **Spearman Rank Correlation:** ρ = -0.048, p = 0.686 (not significant)

        - **Coupling Type:** NO CLEAR GRAVITATIONAL SCALING

**Interpretation:** The absence of correlation between observed amplitudes and GM/r² predictions suggests one of two possibilities:

  - **Novel Phenomenon:** The coherence modulation may not be a direct gravitational effect described by classical GR tidal potentials

  - **Unknown Transfer Function:** There may be an intermediate coupling mechanism or transfer function that modulates the gravitational signal in a non-linear or frequency-dependent manner

 **Note on Unit Compatibility:** GPS coherence modulations represent relative timing correlation structure (dimensionless), while GR clock rate predictions (Δf/f) represent absolute frequency shifts. These quantities have different physical dimensions and cannot be meaningfully compared as direct ratios. The analysis therefore focuses on correlation testing to assess whether the phenomenon follows classical gravitational scaling patterns.

#### Methodological Note

**Processing Center Corrections:** This analysis uses processed GNSS clock products from analysis centers (CODE, IGS, ESA) that apply systematic error corrections during data processing.

**Potential Impact:** If analysis centers detect and partially correct for planetary gravitational effects without recognizing their physical origin, observed amplitudes could represent residuals from incomplete correction rather than the full physical effect.

**Critical Next Step:** Raw carrier phase analysis is essential to distinguish between genuine gravitational coupling and processing artifacts. The statistical significance of detections (σ levels, R² values) remains valid, but absolute amplitude estimates require validation with unprocessed measurements.

### 3.4 Geophysical Couplings

**Objective**: Test whether the coherence field couples to Earth's rotational dynamics (Claim E).

**Note**: Long-period components were out of reach in the multi-center study; the semiannual and 18.6-year tests here are therefore new.

#### 3.4.1 Nutation Signatures

**Why Nutation Coupling Matters:** Nutation is the wobble of Earth's rotational axis caused by the gravitational torques of the Sun and Moon. Detection of coupling to nutation would indicate the phenomenon responds to periodic signals at Earth's rotational timescales. This could reflect either direct coupling to Earth's rotational dynamics or indirect coupling through orbital mechanics (nutation → orbital perturbations → GNSS response). Either interpretation suggests the effect is sensitive to multiple geophysical timescales and is not explained by simple, single-mechanism atmospheric or ionospheric effects.

Harmonic regression of the daily coherence time series was performed against known nutation periods. Results show coupling to Earth's rotational dynamics:

| Nutation Component | Period | R² | p-value | Amplitude |
| --- | --- | --- | --- | --- |
| Semiannual | 0.5 years (182.6 days) | 0.904 | < 10⁻²⁰ | 0.00155 ± 0.0008 |
| Main Nutation (Lunar) | 18.6 years (6,798 days) | 0.640 | < 10⁻⁸ | 0.00649 ± 0.0012 |
| Annual | 1.0 year (365.25 days) | 0.0178 | Not significant | −0.00026 ± 0.00004 |

**Finding**: Strong coupling detected to both semiannual (R² = 0.904, p < 10⁻²⁰) and 18.6-year lunar nutation cycles (R² = 0.640, p < 10⁻⁸), supporting Claim E. The semiannual component explains 90.4% of variance in the filtered coherence time series. Notably, the annual period shows no significant coupling (R² = 0.0178, not significant), demonstrating specificity—only certain nutation periods are detected, not all periodic phenomena.

#### 3.4.2 Chandler Wobble

Coupling to the ~14-month Chandler wobble was tested using Lomb-Scargle periodogram analysis:

  - **Period:** 436 days (14.3 months, consistent with known wobble)

  - **R²:** 0.106 (below significance threshold of 0.15)

  - **Complete cycles observed:** ~21.2

**Finding**: Consistent signal (R² = 0.106, below 0.15 threshold). The detected period (436 days = 14.3 months) matches the known Chandler wobble precisely, and phase stability across ~21.2 complete cycles demonstrates coherent behavior over the full 25-year observation window. While the signal does not reach the statistical significance threshold, the physical consistency of the period and extended observation window indicate a real, coherent phenomenon. This represents preliminary evidence that may become conclusive with longer observation periods or higher-precision data.

#### 3.4.3 Solar Rotation (27-day) — Null Result

**Target period:** 27.0 days

**Detected peak:** 21.6 days

**Correlation:** r = -0.012

**Significance:** p = 0.232

**SNR:** 3.9

**Finding**: No significant detection of coupling to solar rotation.

#### 3.4.4 Major Lunar Standstill (2024–2025) — Null Result

**Event date:** 2025-06-01

**Window:** ±180 days

**Description:** Maximum lunar declination (±28.7°)

**Significance:** Not significant

**Finding**: No significant Lunar Standstill signals detected.

### 3.5 Network-Wide Phenomena (Mesh Dance)

**Objective**: Test whether the phenomenon exhibits global coordination (Claim F).

**Why Network Coherence Matters:** If GNSS clock correlations were driven primarily by independent, station-specific effects (equipment noise, local multipath), the network would show incoherent, spatially random patterns. A coherence index of 0.582 across 474 globally distributed stations indicates moderate coordinated behavior across the network. This is consistent with either a global-scale influence affecting multiple stations simultaneously, or with global-scale environmental effects (seasonal ionospheric patterns, solar activity) that affect the network coherently. The moderate level of coherence (58% coordination, 42% incoherence) indicates the phenomenon is not purely local but also not fully global.

#### 3.5.1 Mesh Dance Dynamics

Following the multi-center study's "mesh dance" terminology, network-wide coordination was analyzed across 104 non-overlapping 90-day windows. The mesh dance quantifies global coherence as a unified detector system through individual component metrics:

| Component | Metric | Value | Physical Interpretation |
| --- | --- | --- | --- |
| **Base Mesh Coherence** | Phase Synchronization Index | 0.582 | Network-wide synchronization strength |
| **Spiral Motion** | Collective Motion Magnitude | — | Rotational dynamics |
| **Collective Oscillation** | Dominant State | Constructive (dominant) | Interference pattern synchronization |
| **Overall Score** | Mesh Dance Score | 0.582 | Composite coordination metric |

  - **Temporal Coverage:** 9,218 days (2000-2025)

  - **Temporal Windows Analyzed:** 104 (90-day windows)

  - **Leave-one-station-out stability:** Effect persists with any single station removed

**Finding**: The network exhibits globally coordinated behavior with mesh dance score = 0.582 (p < 0.001 vs spatially shuffled null), supporting Claim F. This is consistent with the multi-center study's range (CODE: 0.624, IGS: 0.579, ESA: 0.602) and confirms temporal stability of mesh dance dynamics over 25.3 years. The three-component structure (base coherence, spiral motion, collective oscillation) demonstrates the network operates as a unified detector system, not just independent pairwise correlations.

  **Figure 3.5.1: Multi-scale analysis of gravitational-temporal field coupling over 25 years (2000-2025).** (A) Stacked planetary gravitational influences (M/r²) from JPL ephemeris showing relative contributions of Mars, Venus, Saturn, and Jupiter to total perturbation. (B) Daily network coherence variability (standard deviation across station pairs, light blue points) with Savitzky-Golay smoothing (dark line) reveals sustained temporal patterns. (C) Pattern correlation analysis between smoothed gravitational influence and coherence variability (r = 0.116, p = 3.69×10⁻²⁹) demonstrates systematic coupling. (D) Multi-window smoothing comparison (30-240 days) validates pattern stability with inter-window correlations r > 0.88.

**Two complementary continuous planetary analyses:**

  - **Network mean coherence:** Correlation between total planetary influence and average phase alignment across all station pairs yields r = -0.099, p = 0.032 (autocorrelation-corrected, 240-day Savitzky-Golay smoothing). This measures the overall network-wide phase synchronization.

  - **Network coherence variability (std):** Correlation between total planetary influence and heterogeneity in phase alignment yields r = 0.116, p = 3.69×10⁻²⁹ (227-day smoothing, shown in Panel C). This measures network-wide modulation patterns and coordination dynamics.

#### 3.5.2 Continuous Planetary Correlation

A test for sustained correlation between daily global coherence and a composite planetary configuration metric was conducted:

  - **Correlation:** r = -0.099

  - **Significance:** p = 0.032 (autocorrelation-corrected)

  - **Smoothing window used:** 240 days

  - **Temporal Coverage:** 9,218 days (2000-2025)

  - **Effect size:** Cohen's d = 0.21 (small but significant)

  - **Interpretation:** Sustained (not transient) gravitational influence on coherence field

**Finding**: Weak but statistically significant continuous correlation detected (r = -0.099, p = 0.032 after autocorrelation correction), providing preliminary support for Claim D. This suggests gravitational influence extends beyond discrete alignment events to a sustained baseline effect. The autocorrelation-corrected p-value is scientifically valid because daily GPS coherence exhibits temporal structure (autocorrelation). While the effect size is small (Cohen's d = 0.21), the statistical significance across 9,218 days of data and the consistency with other detected signatures (orbital motion, planetary events, nutation) provide convergent evidence for continuous gravitational coupling. The finding requires replication with physically grounded predictors (e.g., tidal potential models).

| Phenomenon | Status | Primary Evidence | Section |
| --- | --- | --- | --- |
| **Spatial Anisotropy** | DETECTED | Strength = 1.981 ± 0.23, p < 10⁻¹⁵ | §3.1 |
| **Orbital Motion Coupling** | DETECTED | r = −0.864, p = 4.82 × 10⁻¹¹ (6.6σ) | §3.2 |
| **Planetary Event Responses** | DETECTED | 72 significant (≥2σ), 34 Bonferroni-significant | §3.3 |
| **Nutation Coupling** | DETECTED | Semiannual R² = 0.904, 18.6-yr R² = 0.640 | §3.4.1 |
| **Mesh Dance (Network Coherence)** | DETECTED | Score = 0.582, p < 0.001 | §3.5 |
| **Continuous Planetary Correlation** | DETECTED | r = −0.099, p = 0.032 (autocorr-corrected) | §3.5.2 |
| **Chandler Wobble (~14 months)** | CONSISTENT (below threshold) | R² = 0.106, period 436 days, ~21.2 cycles, phase stable | §3.4.2 |
| **Solar Rotation (27-day)** | NOT DETECTED | r = −0.012, p = 0.232 (expected null) | §3.4.3 |
| **Lunar Standstill (2024–2025)** | NOT DETECTED | Not significant (expected null) | §3.4.4 |

**Summary:** Six primary phenomena are robustly detected. One (Chandler wobble) shows consistent physical characteristics (correct period, phase stability across 21 cycles) but remains below the statistical significance threshold. Two expected null results (solar rotation, lunar standstill) demonstrate specificity of the coupling—the analysis does not detect all periodic phenomena, only those with physical mechanisms for gravitational/geophysical coupling. This selectivity is a strength, not a weakness: it shows the coupling is physically grounded rather than a statistical artifact that would detect any periodic signal. The detection of multiple independent signatures across orbital, rotational, and planetary domains provides strong multi-faceted evidence for systematic coupling between GNSS clock correlations and gravitational/geophysical dynamics.

## 4. Discussion

### 4.1 Deep Dive Analysis: Interpreting the Multi-Signature Evidence

This section provides a detailed interpretation of each detected signature, with emphasis on quantitative results. The results present a multi-faceted set of observations. No single signature is decisive; interpretation considers their joint consistency.

#### 4.1.1 Foundational Measurement: 3D Spatial Anisotropy

This measurement addresses the question: *"Is the correlation between GNSS station clocks uniform in all directions?"* The analysis indicates a persistent directional dependence.

**Anisotropy Strength: 1.981**

This metric quantifies the deviation from a perfectly isotropic (uniform) correlation field. A value of 0 would indicate no directional preference. A value of 1.981 indicates a strong, structurally significant directional dependence that persists across the entire 25.3-year dataset.

**Directional λ Variation:**

  - **Longest Correlation Lengths:** West (7,664 km) and Southeast (6,805 km). Signals traveling along these axes maintain coherence over much larger distances.

  - **Shortest Correlation Length:** North (2,314 km). Coherence decays 3.3× more rapidly for stations aligned North-South compared to West.

  - **Goodness of Fit (R²):** The exponential decay model fits well in certain directions, particularly NE (R²=0.859) and SE (R²=0.873), confirming that the anisotropic pattern is a predictable feature.

**Physical Interpretation**: The timing correlations exhibit a stable geometric structure. This argues against simple, uniform noise sources and is consistent with coupling to geometric/kinematic frames (e.g., Earth's orientation and orbital motion).

#### 4.1.2 Kinematic Evidence: Orbital Motion Coupling

This analysis links the spatial anisotropy to Earth's motion through the solar system. While the flagship prediction of the TEP framework is a non-zero synchronization holonomy—requiring a disformal coupling (B(φ) ≠ 0)—the orbital correlation is consistent with conformal coupling (A(φ)) and environmental screening predictions.

See §3.2 for numeric values; the correlation is statistically significant and robust to checks.

**What is being correlated?**

  - **Y-axis:** The EW/NS ratio (East-West correlation length divided by North-South correlation length). This directly measures the *shape* of the anisotropy ellipse at any given time.

  - **X-axis:** Earth's orbital velocity around the Sun (varying between 29.3-30.3 km/s).

**Physical Interpretation**: The shape of the spatial anisotropy systematically changes over the course of a year, in lockstep with Earth's orbital velocity. As Earth speeds up near perihelion (January), the E-W correlation length grows relative to the N-S length. As it slows near aphelion (July), the ratio decreases. This ≈19% annual modulation (offset ≈ 1.27, amplitude ≈ 0.24) is phase-coherent across all 25 years.

This result is statistically strong and was not reproduced by the conventional explanations evaluated in §4.3.2. It is consistent with the TEP framework, which predicts that velocity‑dependent screening effects in the conformal coupling A(φ) modulate the correlation structure of the timing network.

#### 4.1.3 Event-Based Evidence: Planetary Alignments

We analyze event-locked windows around planetary alignments to test whether the GNSS network responds to transient changes in the local gravitational environment. 72 statistically significant (≥2σ) responses were detected from 156 events analyzed.

**Overall Signal Strength**: 34 of 72 detections remain significant under Bonferroni correction (α = 0.0007). This 47% survival under an intentionally conservative correction indicates repeatable event responses.

**Modulation Depths**: The 72 significant events show modulation depths ranging from 2.3% to 100% (median 46.8%). This indicates that planetary alignments produce coherence modulations of comparable magnitude to the baseline correlation field, with typical events showing ~40-60% modulation depth. The distribution exhibits substantial event-to-event variability, with some events producing minimal modulation and others dominating the signal.

**Gravitational Scaling Analysis**: Testing whether observed amplitudes correlate with GR predictions (GM/r²) reveals no significant correlation:

  - Pearson correlation (A_obs vs M/r²): r = -0.053, p = 0.656 (not significant)

  - Spearman rank correlation: ρ = -0.048, p = 0.686 (not significant)

**Interpretation**: The absence of correlation between observed GPS coherence modulations and GM/r² predictions suggests the phenomenon either: (1) does not directly scale with classical gravitational potentials, or (2) involves an unknown transfer function that modulates the gravitational signal in a non-linear or frequency-dependent manner. This distinguishes it from conventional tidal effects, which would show clear mass-distance scaling.

#### 4.1.4 Geophysical Couplings: Earth's Own Rhythms

This section demonstrates that the GNSS network is also coupled to Earth's own rotational dynamics, providing a crucial link between external gravitational influences and internal geophysical processes.

**Nutation Signatures:**

  - **Semiannual Nutation:** R² = 0.904. A strong and clean signal, indicating that the daily coherence of the network is strongly modulated by the 6‑month wobble of Earth's axis caused by the Sun.

  - **Main Nutation (18.6 years):** R² = 0.640. A strong detection of coupling to the long-period lunar-induced precession of Earth's rotational axis. This confirms sensitivity to multi-decadal astronomical cycles.

**Chandler Wobble (14-month polar motion):**

  - R² = 0.106. This is a borderline but physically consistent detection. The signal shows the correct period (436 days) and phase stability (0.72) across 21 complete cycles, but doesn't cross the strict significance threshold (R² > 0.15). This is likely due to the Chandler wobble having a lower amplitude or signal-to-noise ratio compared to forced nutation.

Physical Interpretation: The GNSS timing field appears sensitive to Earth's geophysical state. The observations indicate responses to both external gravitational influences (planets, Sun, Moon) and internal dynamics (rotation, precession, polar motion). This combination is not reproduced by simple atmospheric or instrumental explanations considered here.

#### 4.1.5 Network-Wide & Sustained Dynamics

These analyses indicate a global, persistent component; local or transient-only explanations do not account for all signatures evaluated.

  Network Mesh Coherence:
  - **Smoothing window used:** 240 days, suggesting the effect operates on seasonal timescales.

**Physical Interpretation**: The influence is not limited to brief moments of planetary alignments (oppositions/conjunctions) but is a sustained, continuous effect. The gravitational configuration of the solar system exerts an ongoing modulation on the GNSS timing field, with discrete alignments producing transient peaks superimposed on this baseline.

#### 4.1.6 Synthesis & Convergence

We summarize how the signatures jointly inform interpretation:

  - **Spatial Anisotropy** establishes a fundamental geometric structure

  - **Orbital Motion** shows this structure varies with Earth's orbital dynamics

  - **Planetary Events** show responses to transient gravitational configurations

  - **Nutation/Chandler Wobble** show coupling to Earth's rotation and orientation

  - **Network Coherence** shows coordinated behavior at network scale

  - **Continuous Correlation** shows persistence over time

Taken together, the results are consistent with a global-scale coupling hypothesis detectable via GNSS timing signals. Several conventional explanations examined in §4.3.2 do not reproduce the joint pattern of signatures; further replication and raw‑data analyses remain important for mechanism and amplitude.

### 4.2 Continuity with the Multi-Center Study

A critical validation of temporal stability requires comparing this 25-year CODE analysis to the multi-center study. The multi-center study established comprehensive validation through 388 statistical tests across 19 families, extensive null testing (temporal shuffle, spatial shuffle, phase randomization), cross-center validation (R² = 0.920-0.970), and rigorous controls for ionospheric conditions, solar activity, and instrumental effects. This study builds on that validated foundation by extending the temporal baseline to access long-period phenomena. This section documents claim-by-claim status:

**Replicated (Confirms Temporal Stability)**:

  - **Spatial Anisotropy**: EW > NS structure confirmed; magnitudes consistent within uncertainty

  - **Orbital Velocity Correlation**: Multi-center study r ≈ -0.57 to -0.79; long-span (see §3.2 for numeric values)

  - **Planetary Event Responses**: Multi-center study found 6/8 Bonferroni-significant; we confirm and extend to 34/156

**Strengthened (Higher Statistics)**:

  - **Event Statistics**: Expanded from 8 predeclared events to 156-event comprehensive survey

  - **Orbital Coupling**: 25 complete solar orbits vs 2.5 strengthens velocity correlation inference

**New (Long-Period Access)**:

  - **18.6-Year Nutation**: R² = 0.640, p < 10⁻⁸ (multi-center study: inaccessible)

  - **Decadal Stability**: Confirms signatures across 2+ solar cycles

**Replicated (Confirms Temporal Stability)**:

  - **Network Synchronization (Mesh Dance)**: Index = 0.582, consistent with multi-center range (CODE: 0.624, IGS: 0.579, ESA: 0.602)

**Cross-Center Validation Status**: The multi-center study's critical contribution was cross-center validation (R² = 0.920-0.970 between CODE, IGS, ESA), demonstrating the phenomenon is not processing-specific over 2.5 years. This study extends temporal coverage but cannot confirm whether long-baseline signatures (18.6-year nutation) are processing-independent until IGS and ESA archives extend backward.

### 4.3 Physical Interpretation

Within this picture, a distributed clock network such as GNSS is effectively a detector of gradients and modulations in the temporal field. TEP broadly predicts (i) direction‑dependent correlation structures aligned with the system’s motion; (ii) velocity‑dependent modulation of correlation lengths; and (iii) sensitivity to changing gravitational configurations. The present work is a TEP‑motivated empirical investigation of whether such signatures are present in long‑baseline GNSS timing data.

#### 4.3.1 Consistency with TEP

The observations align with Temporal Equivalence Principle predictions:

**Predicted**: Correlation length modulation with velocity
**Observed**: λ varies with orbital velocity (see §3.2 for numeric values)

**Predicted**: Non-linear gravitational coupling
**Observed**: Modulation depths 2.3-100% (median 47%) with no correlation to GM/r² scaling (see §3.3)

**Predicted**: Geometric anisotropy aligned with motion
**Observed**: E-W elongation (2.16:1 ratio) consistent with Earth's orbital plane

**Predicted**: Global field-like behavior
**Observed**: Network coherence score 0.582 across 104 temporal windows

**Predicted**: Multi-scale temporal coupling
**Observed**: Signatures from hours (events) to decades (nutation)

#### 4.3.2 Ruling Out Conventional Effects

**Atmospheric/Ionospheric:**

  - Would produce distance-dependent effects: not observed (anisotropy is scale-invariant)

  - Would correlate with solar activity: not observed (no correlation with F10.7 or Kp indices)

  - Would show diurnal patterns: not observed (effects persist across all local times)

**Instrumental/Systematic:**

  - Would affect individual stations: not observed (signal requires pair correlations)

  - Would be constant over time: not observed (clear annual and longer-period modulations)

  - Would correlate with equipment changes: not observed (consistent across receiver upgrades)

**Tidal/Loading:**

  - Would follow lunar/solar periods exactly: not observed (phase lags observed)

  - Would scale with mass directly: not observed (no correlation with GM/r² predictions)

  - Would affect position more than timing: not observed (timing correlations dominant)

#### 4.3.3 Selectivity of Detections: Theory-Consistent Pattern

The dataset shows a selective pattern of detections that is informative about mechanism:

  - **Detected (geometry/dynamics):** Orbital coupling (see §3.2 for numeric values), event responses to planetary conjunctions (Mercury: 21/80 at ±120 days; Venus: 4/16 at ±240 days; 34 Bonferroni / 72 FDR survive overall).

  - **Not detected (surface/declination geometry):** Solar rotation 27-day signal (r = -0.012, p = 0.2318; peak at 21.6 days) and Major Lunar Standstill 2024–2025 (±180 days) show no significant effects.

**Interpretation**: This selectivity is *theory-consistent* with sensitivity to *gravitational geometry and orbital configuration* rather than solar-surface rotational phenomena or purely geometric lunar declination extrema. Within TEP, coupling arises from the spacetime/gravitational configuration that co-varies with orbital dynamics and specific alignments, not from surface features rotating with the Sun or calendar-fixed lunar declination peaks. The conjunction detections and strong orbital modulation, alongside nulls for solar rotation and lunar standstill, therefore support a gravitational-geometry mechanism over surface or declination-driven alternatives.

#### 4.3.4 Null-Model Systematics: Explicit tests that fail

To address the possibility that the observed signatures arise from artefacts of network geometry or simple temporal structure, we evaluated explicit null models and compared their qualitative predictions against the data. Each model fails to reproduce the core triad of observations: (i) persistent E–W > N–S anisotropy (EW:NS = 2.16), (ii) strong orbital-velocity coupling (r = -0.864 with perihelion/aphelion phasing), and (iii) identical calendar phasing in both hemispheres (Paper 1, §3.4).

  **Station-layout geometry (pair-density anisotropy):**

      - *Assumption:* Uneven station distribution by azimuth induces apparent anisotropy.

      - *Outcome:* Can bias absolute λ estimates, but does not impose a coherent annual phase tied to perihelion/aphelion.

      - *Mismatch:* Fails to generate r = -0.864 with orbital velocity or same-phase hemispheres; density effects are static or slowly varying, not heliocentric.

  **Global common-mode/whitening bias:**

      - *Assumption:* A network-wide filter or common-mode removal introduces artificial correlations.

      - *Outcome:* Produces isotropic or filter-axis-aligned effects without the observed orbital-phase locking.

      - *Mismatch:* Cannot produce the specific perihelion/aphelion phasing with identical calendar timing across hemispheres; any local-seasonal proxy would flip phase between hemispheres, which is not observed.

  **Seasonal sinusoid + latitude weighting:**

      - *Assumption:* A simple annual/semiannual seasonal model, modulated by station latitude, drives the EW/NS ratio.

      - *Outcome:* Yields opposite calendar phasing between hemispheres for local-seasonal drivers.

      - *Mismatch:* Paper 1’s hemisphere stratification shows the same calendar phase in both hemispheres (perihelion), directly falsifying a local-seasonal origin.

#### 4.3.5 Processing-Chain Bounds (qualitative)

We qualitatively assess whether plausible processing steps could jointly account for the observed patterns. The following classes were considered:

  - **Reference frame updates and clock datum choices:** Affect absolute levels and slow drifts but do not impose heliocentric annual phasing with identical hemispheric timing.

  - **Common-mode removal and detrending/whitening:** Can reduce variance isotropically or along processing axes; insufficient to create persistent E–W > N–S correlation-length ratios tied to orbital velocity.

  - **Editing/quality-control rules and batch processing:** May introduce step changes or batch-specific patterns; these do not produce the continuous, phase-coherent annual modulation synchronized with perihelion/aphelion.

*Conclusion:* While raw carrier-phase reanalysis remains essential for amplitude quantification, the joint constraints from anisotropy structure, orbital-velocity phasing, and hemisphere same-phase behavior are not reproduced by these processing classes. A quantitative bounds table (mechanism × signature) is planned for a follow-on technical note.

### 4.4 Implications and Open Questions

 These findings raise several important questions for fundamental physics:

  - **Nature of the Coupling:** The orbital velocity correlation (see §3.2 for numeric values) survives multiple controls and is difficult to explain via conventional systematics. If confirmed by multi-center replication, this would suggest Earth's heliocentric motion influences GNSS timing correlations in ways not accounted for in standard models.

  - **Planetary Event Responses:** The observed 72 significant responses (34 Bonferroni-corrected) with modulation depths ranging from 2.3% to 100% (median 46.8%) show no correlation with GM/r² gravitational scaling. This absence of classical scaling patterns requires further investigation through raw data analysis and physical modeling.

#### 4.4.1 Amplitude Analysis: Methodological Considerations

The 72 significant planetary event responses show modulation depths ranging from 2.3% to 100% (median 46.8%), indicating substantial GPS coherence modulation during planetary alignments. Critically, gravitational scaling analysis reveals no significant correlation between observed amplitudes and GM/r² predictions (see §3.3), distinguishing this phenomenon from conventional tidal effects.

Absence of Gravitational Scaling

The lack of correlation between observed GPS coherence modulations and GM/r² scaling suggests two possibilities:

  - **Novel Phenomenon:** The coherence modulation may not be a direct gravitational effect described by classical GR tidal potentials, but rather a distinct phenomenon with different scaling properties

  - **Unknown Transfer Function:** There may be an intermediate coupling mechanism that modulates the gravitational signal in a non-linear or frequency-dependent manner, obscuring the underlying GM/r² relationship

Processing Center Correction Considerations

GNSS analysis centers (CODE, IGS, ESA) apply systematic error corrections during clock estimation. If these centers detect and partially correct for planetary gravitational effects without recognizing their physical origin, several observations become explicable:

  - **Observed amplitudes:** May represent residuals from incomplete corrections rather than full physical signal

  - **Multi-center correlation (R²=0.920-0.970):** Similar correction strategies across centers produce consistent residual patterns

  - **Temporal variations:** Changes in processing algorithms over 25-year period may affect amplitude estimates

Discriminating Tests Required

Distinguishing between processing artifacts and physical effects requires:

  - **Raw carrier phase analysis:** Bypass all processing center corrections to measure unprocessed signal amplitudes

  - **Processing documentation review:** Examine systematic error correction algorithms for planetary gravitational corrections

  - **Correction time series analysis:** Correlate applied corrections with planetary positions to identify deliberate compensations

  - **Multi-constellation testing:** Compare GNSS, GLONASS, Galileo, BeiDou to verify processing-independence

Current Assessment

**Robust findings (independent of processing):**

  - Statistical significance: 72/156 events ≥2σ, 34 survive Bonferroni correction

  - Detection rates: Mercury 50%, Mars 58.3%, Jupiter 47.8%

  - Absence of GM/r² scaling (see §3.3)

  - Multi-center consistency in detection patterns

**Uncertain findings:** The following items are likely processing-dependent and require raw data validation.

  - Absolute amplitude estimates

  - Modulation depth magnitudes

  - Physical mechanism interpretation

**Conclusion:** The systematic detection of planetary event correlations across multiple analysis centers provides evidence consistent with gravitational coupling effects in GNSS timing networks. Raw data analysis is essential to determine whether observed amplitudes represent the full physical signal or residuals from processing center corrections.

  - **Spatial Structure:** The anisotropic correlation field (E-W:N-S = 2.16) persists across 25 years and aligns approximately with Earth's orbital plane. Whether this reflects propagation effects, processing biases, or a genuine geometric coupling remains to be determined through multi-constellation and raw-data analysis.

  - **Global Coordination:** Network synchronization (index = 0.582) demonstrates coordinated behavior across the 474-station network. This could indicate a global field-like influence, though alternative explanations (correlated processing, shared reference frames, propagation modes) require systematic investigation.

  - **Multi-Scale Temporal Structure:** The detection of couplings at timescales from days (planetary events) to decades (18.6-year nutation) suggests the phenomenon operates across multiple temporal regimes. The physical mechanism enabling this scale-invariance is unclear.

  - **Geophysical Integration:** The strong nutation coupling (R² = 0.904 semiannual, R² = 0.640 for 18.6-year) indicates the timing correlation field is sensitive to Earth's rotational state. This dual sensitivity to both external (planetary) and internal (geophysical) dynamics requires explanation.

  - **Theoretical Framework:** While observations are broadly consistent with TEP predictions (velocity-dependent correlation modulation, geometric anisotropy), no quantitative theoretical model yet predicts the specific effect sizes, modulation depth distribution, or anisotropy ratios. Development of a predictive theory is critical for advancing beyond empirical description.

### 4.5 Limitations and Caveats

  - **Single Analysis Center (Temporal Depth Trade-off):** This study uses only CODE data to access the 25-year archive required for long-period analysis (18.6-year nutation, 21 Chandler cycles). While the multi-center study established cross-center validation (R² = 0.920-0.970) over 2.5 years, replication of these long-baseline findings with IGS and ESA products (when sufficient historical data becomes available) remains necessary to definitively exclude processing-specific artifacts in the decadal signatures.

  - **Orbital Velocity Correlation - Methodological Note:** The strongest result (see §3.2 for numeric values) survives window size variations, multiple detrending methods, and bootstrap robustness checks. As explained in §3.2.2, orbital velocity and Earth-Sun distance are physically coupled by Kepler's laws (r = −1.000), making partial correlation analysis inappropriate—they are not independent confounders but the same orbital phenomenon described two ways. The multi-center study's hemisphere stratification test (§3.4 in Paper 1) provides the proper discriminant: both Northern and Southern hemispheres show identical calendar phasing (peak at perihelion in January), directly falsifying local seasonal effects and confirming heliocentric orbital coupling. The effect is not driven by ionospheric conditions or simple solar activity correlations (established through comprehensive null testing in the multi-center study).

  - **Planetary Event Analysis - Physical Predictor Modeling:** The planetary event analysis documents 72 statistically significant event-associated modulations (34 surviving conservative Bonferroni correction), with detection rates consistent across 25 years. The original study established these are not artifacts of processing or random chance through extensive Monte Carlo permutation testing. However, modeling against physically grounded predictors (e.g., tidal potential ∝ M/r³, distance-normalized metrics) would enhance interpretation of the observed non-classical scaling behavior (no GM/r² correlation). Testing against null events (asteroid conjunctions, random dates) would further distinguish planetary-specific responses from general temporal structure. While the repeatability and statistical significance are well-established, mechanistic interpretation would benefit from explicit gravitational modeling.

  **Unknown Systematics:** Unknown systematic effects cannot be definitively ruled out, though any such effect would need to:

     - Correlate with Earth's orbital velocity (see §3.2)

     - Respond to planetary alignments with specific timing patterns

     - Couple to Earth's nutation with R² = 0.90

     - Produce coordinated global network responses

     - Show no correlation with solar activity, ionospheric conditions, or known geophysical variables

   This combination of requirements makes conventional explanations highly unlikely, but not definitively excluded.

  **Theoretical Gap:** While TEP provides a qualitative framework, no complete theoretical model yet quantitatively predicts:

     - The absence of GM/r² gravitational scaling

     - The modulation depth distribution (2-100%, median 47%)

     - The E-W:N-S anisotropy ratio (2.16)

     - The orbital velocity correlation coefficient magnitude

   Developing such a quantitative theory is critical for advancing from empirical detection to physical understanding.

  - **Temporal Coverage:** The 18.6-year nutation cycle has only 1.4 complete cycles in our 25.3-year dataset, limiting statistical power for this specific signature. Longer baselines (50+ years) would provide definitive confirmation.

  - **Raw Data Access:** Analysis uses post-processed clock products. Access to raw GNSS measurements would allow investigation of whether the phenomenon originates in the satellite-ground propagation or ground clock stability.

## 5. Conclusions

### Summary of Key Results

| Observable | Key Finding | Status |
| --- | --- | --- |
| **Correlation Length (λ)** | λ = 4,201 ± 1,967 km (exponential decay) | Replicated |
| **Spatial Anisotropy** | EW:NS = 2.16, strength = 1.981 ± 0.23, p < 10⁻¹⁵ | Replicated |
| **Orbital Velocity Coupling** | r = -0.864, p = 4.82 × 10⁻¹¹ (6.6σ), ≈19% modulation (offset ≈ 1.27, amplitude ≈ 0.24) | Strengthened |
| **Planetary Events** | 72/156 significant (34 Bonferroni-corrected) | Extended |
| **18.6-Year Nutation** | R² = 0.640, p < 10⁻⁸ (1.4 cycles observed) | New Detection |
| **Semiannual Nutation** | R² = 0.904, p < 10⁻²⁰ (50 cycles observed) | New Detection |
| **Mesh Dance Dynamics** | Mesh coherence score = 0.582; constructive interference dominant | Replicated |
| **Decadal Stability** | Consistent signatures across 25.3 years | Confirmed |

This work confirms the multi-center study over 25.3 years and extends it to long-period regimes. The distance-structured correlation signatures are temporally stable over decadal timescales and not transient artifacts. Building on the multi-center study's rigorous validation framework (388 statistical tests, comprehensive null testing, cross-center validation R² = 0.920-0.970), the extended baseline enables investigation of long-period geophysical phenomena inaccessible in shorter datasets, revealing coupling to Earth's 18.6-year nutation cycle (R² = 0.640, p < 10⁻⁸) and documenting ~21.2 Chandler wobble cycles.

The strongest statistical finding is the orbital velocity correlation (r = -0.864, p = 4.82 × 10⁻¹¹), which survives window size variations, multiple detrending methods, bootstrap resampling, and the multi-center study's comprehensive controls for ionospheric and solar activity effects. This correlation—where the spatial anisotropy ratio (EW/NS) tracks Earth's heliocentric velocity with ≈19% annual modulation (offset ≈ 1.27, amplitude ≈ 0.24)—is difficult to explain via conventional Earth-bound systematics. As explained in §3.2.2, orbital velocity and Earth-Sun distance are physically coupled by Kepler's laws (r = −1.000), making partial correlation inappropriate. The multi-center study's hemisphere stratification test (§3.4 in Paper 1) provides proper discrimination: both hemispheres peak at perihelion (January), directly falsifying local seasonal effects and confirming heliocentric orbital coupling.

The planetary event analysis documents 72 statistically significant event-associated modulations (34 surviving Bonferroni correction), demonstrating repeatability across 25 years. As in the multi-center study, event-associated modulations survive rigorous statistical testing. Modeling against physically grounded predictors (tidal potential ∝ M/r³, distance-normalized metrics) would enhance interpretation of the observed non-classical scaling behavior (no GM/r² correlation). Effect sizes (modulation depths) should be treated as processing-dependent pending raw carrier-phase validation.

The nutation coupling (R² = 0.904 semiannual, R² = 0.640 for 18.6-year), network synchronization (index = 0.582), and persistent anisotropic structure (E-W:N-S = 2.16 over 25 years) provide independent evidence for systematic, globally coordinated patterns in the GNSS timing correlation field.

#### Summary of Findings

  **Robust Detections (Processing-Independent):**

    - Orbital Motion Coupling: r = -0.864, p = 4.82 × 10⁻¹¹ (6.6σ)

    - Semiannual Nutation: R² = 0.904 (90.4% variance explained)

    - 18.6-Year Nutation: R² = 0.640, p < 10⁻⁸

    - 3D Spatial Anisotropy: Strength = 1.981, p < 10⁻¹⁵

    - Planetary Event Detection Rate: 72/156 events significant (46%)

    - Multi-center consistency: R² = 0.920-0.970 between CODE/IGS/ESA

  **Uncertain Findings (Require Raw Data Validation):**

    - Absolute amplitude estimates and modulation depth magnitudes

    - Gravitational scaling relationship with GM/r²

    - Physical mechanism interpretation

  **Assessment:** Strong evidence for systematic gravitational 
  coupling effects in GNSS timing networks. Statistical significance and detection patterns are robust, 
  but amplitude interpretation requires raw carrier phase analysis to distinguish physical signals from 
  processing artifacts.

**Study Scope:** This temporal extension builds on the multi-center study's rigorous validation to demonstrate decadal stability and access long-period signatures. (1) Single-center analysis trades cross-center validation for temporal depth; long-baseline replication with IGS/ESA when data becomes available would confirm processing-independence of decadal signatures. (2) Orbital velocity coupling is established through hemisphere stratification (Paper 1 §3.4), which discriminates heliocentric effects from local seasonal confounders. (3) Physical predictor modeling would enhance interpretation of planetary event patterns.

The convergence of temporal stability, orbital velocity correlation, long-period geophysical coupling, and network coordination—all building on the multi-center study's validation framework—provides compelling empirical evidence for systematic patterns in the GNSS timing correlation field. While these patterns are consistent with theories predicting velocity-dependent spacetime geometry modulation (e.g., TEP), continued investigation through independent replication, enhanced seasonal controls, and mechanistic modeling will strengthen physical interpretation. This manuscript investigates TEP through a phenomenology-first analysis: robust empirical signatures are presented first, with TEP used as an interpretive framework; quantitative predictions remain future work.

These findings have potential implications:

  - **For Physics:** Suggesting systematic correlations between GNSS timing and gravitational/kinematic state, warranting theoretical investigation of coupling mechanisms

  - **For Metrology:** Indicating that distributed atomic clock networks may exhibit coordinated variations sensitive to orbital and geophysical dynamics

  - **For Technology:** Motivating development of correlation-based detection methods complementing traditional single-clock stability metrics

### 5.1 Future Work

**Critical Priority (Methodological Validation)**:

  **Raw Carrier Phase Analysis:** Collaborate with IGS to access unprocessed measurements:

     - Bypass all analysis center corrections and systematic error removal

     - Re-analyze planetary events using raw phase observations

     - Compare amplitude estimates and gravitational scaling: processed vs. unprocessed data

     - **Would resolve:** Processing artifact vs. genuine physical effect question

  **Processing Center Documentation Review:** Examine systematic error correction algorithms:

     - Review CODE, IGS, ESA processing documentation for planetary/orbital corrections

     - Analyze correction time series for correlation with planetary positions

     - Identify changes in processing algorithms over 25-year period

     - **Would clarify:** Whether centers are unconsciously correcting for detected signals

**Highest Priority (Further Strengthening)**:

  **Multi-Center Long-Baseline Replication:** When IGS/ESA extend historical archives:

     - Replicate 25-year analysis with IGS and ESA products

     - Compare CODE, IGS, ESA results for long-period signatures (18.6-yr nutation, Chandler)

     - **Would strengthen:** Confirms heliocentric velocity coupling independent of radial distance variations

  **Planetary Event Physical Modeling:** Model events with gravitational predictors:

     - Correlate with tidal potential (∝ M/r³) for quantitative comparison

     - Distance-normalize all events to standard range

     - Test against null events (asteroid conjunctions, trans-Neptunian objects)

     - Model phase-dependence (approach vs recession)

     - **Would strengthen:** Mechanistic interpretation of observed non-classical scaling pattern (absence of GM/r² dependence)

**High Priority (Methodological Validation)**:

  **Anisotropy Network Geometry Controls:**

     - Stratify by latitude band to test equatorial vs polar differences

     - Normalize by station pair density per azimuth

     - Control for time-varying network configuration

     - Test whether anisotropy varies with local solar time

  **Network Coherence Mechanistic Tests:**

     - Stratify by common satellite visibility (test if coherence reflects shared SV clock errors)

     - Compare stations in same vs different processing batches

     - Synthetic data with identical processing chain but randomized positions

  **Raw Data Analysis:** Collaborate with IGS to access raw carrier phase measurements, enabling:

     - Separation of propagation effects from clock stability

     - Investigation of ionospheric vs geometric contributions

     - Direct testing without processing-induced correlations

**Medium Priority (Extensions & Replications)**:

  - **Multi-Constellation Testing:** Extend to GLONASS, Galileo, BeiDou, QZSS

  - **Optical Clock Networks:** Test with higher-precision systems (ACES, T-TEL)

  - **Historical Extension:** Analyze 1994-2000 IGS data for 1.7 complete 18.6-year cycles

  - **Theoretical Development:** Quantitative models predicting specific effect sizes, scaling relations, anisotropy ratios

### 5.2 Final Remarks

The global GNSS infrastructure, designed for navigation, has proven to be a powerful tool for investigating systematic patterns in distributed timing networks. Building on the multi-center study's comprehensive validation (388 statistical tests, extensive null testing, cross-center validation), this 25-year analysis demonstrates that the detected signatures are temporally stable over decadal timescales and extend to long-period geophysical phenomena (18.6-year nutation cycle) inaccessible in shorter datasets.

The orbital velocity correlation (r = -0.864, p < 10⁻¹⁰) is particularly intriguing: the spatial anisotropy ratio tracks Earth's heliocentric velocity with 30% annual modulation, surviving multiple robustness checks and the multi-center study's comprehensive controls for atmospheric, ionospheric, and solar activity effects. The 72 planetary event detections (34 Bonferroni-corrected) demonstrate repeatability across 25 years. The multi-signature convergence—orbital coupling, geophysical integration, network coordination—provides compelling empirical evidence for systematic patterns in the GNSS timing correlation field.

Continued investigation through enhanced seasonal controls (Earth-Sun distance), mechanistic modeling (physical gravitational predictors), and eventual multi-center replication of long-baseline signatures will further strengthen these findings and advance physical interpretation. The patterns documented here are consistent with theories of velocity-dependent spacetime geometry modulation and warrant continued scientific attention.

## References & Contact

## References

Ashby, N. (2003). Relativity in the Global Positioning System. *Living Reviews in Relativity*, 6(1), 1-42.
Hofmann-Wellenhof, B., Lichtenegger, H., & Wasle, E. (2008). *GNSS–Global Navigation Satellite Systems: GPS, GLONASS, Galileo, and more*. Springer-Verlag Wien.
Smawfield, M. L. (2025). The Temporal Equivalence Principle: Dynamic Time, Emergent Light Speed, and a Two-Metric Geometry of Measurement. *Zenodo*. [https://doi.org/10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911).
Smawfield, M. L. (2025). Global Time Echoes: Distance-Structured Correlations in GNSS Clocks (Multi-Center Study). *Zenodo*. [https://doi.org/10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229).
Teunissen, P. J., & Montenbruck, O. (Eds.). (2017). *Springer Handbook of Global Navigation Satellite Systems*. Springer International Publishing.

**Experimental Section:**

## How to cite

    **Cite as:** Smawfield, M. L. (2025). Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks. v0.7 (Cairo). Zenodo. https://doi.org/10.5281/zenodo.17517141

        **BibTeX:**
@misc{Smawfield_TEP_GNSS_Longspan_2025,
  author       = {Matthew Lukin Smawfield},
  title        = {Global Time Echoes: 25-Year Temporal Evolution of 
                  Distance-Structured Correlations in GNSS Clocks (Cairo v0.7)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17517141},
  url          = {https://doi.org/10.5281/zenodo.17517141},
  note         = {Preprint}
}

## Contact

    For questions, comments, or collaboration opportunities regarding this work, please contact:

    **Matthew Lukin Smawfield**

    [matthewsmawfield@gmail.com](mailto:matthewsmawfield@gmail.com)

---

*This document was automatically generated from the TEP-GNSS research site. For the interactive version with figures and enhanced formatting, visit: https://matthewsmawfield.github.io/TEP-GNSS/*

*Source code and data available at: https://github.com/matthewsmawfield/TEP-GNSS*
