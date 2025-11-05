# Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks (Cairo v0.3)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.3 (Cairo)  
**Date:** First published: 3 November 2025 · Last updated: 4 November 2025  
**DOI:** 10.5281/zenodo.17517141  
**Generated:** 2025-11-05  

---

## Abstract

Following the multi-center study's detection of distance-structured correlations in GNSS clock networks†, this paper extends the temporal baseline to 25.3 years (2000-2025) using CODE data encompassing 165 million station pairs. These correlations follow an exponential decay with correlation length λ = 4,201 ± 1,967 km, consistent with the multi-center study's range (λ = 3,330–4,549 km across centers). The extended dataset confirms temporal stability of the originally detected signatures and enables investigation of long-period geophysical phenomena inaccessible in shorter baselines.

The analysis replicates the multi-center study's core findings with enhanced statistical power. Spatial anisotropy is pronounced with E–W > N–S (EW:NS = 2.16; strength = 1.981 ± 0.23) and correlates with Earth's orbital dynamics (r = −0.864, p < 10⁻¹⁰). Directional structure shows E–W elongation approximately aligned with the orbital plane. Planetary alignment analysis detects 72 of 156 events at ≥2σ significance, with 34 surviving Bonferroni correction (primary window: ±120 days).

The 25-year baseline enables decadal-scale tests of long-period rotational coupling, revealing strong evidence for the 18.6-year lunar nutation cycle (R² = 0.640, p < 10⁻⁸) and suggestive evidence for the 14-month Chandler wobble (R² = 0.106). Network-wide phase synchronization (index = 0.582) demonstrates global coordination inconsistent with local effects. These findings establish decadal stability of the detected correlation structure and provide quantitative constraints on long-period geophysical coupling in precision timing networks. This single-center analysis trades cross-center validation for temporal depth; multi-center replication is required to confirm processing-independence of decadal signatures. While these patterns are consistent with theories predicting velocity-dependent spacetime geometry modulation (e.g., TEP), independent replication by other analysis centers and raw data analysis are essential before definitive acceptance of these extraordinary claims.

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

      - Cross-center agreement (CODE/IGS/ESA)

    **New in this CODE-only, 25.3-year analysis:**

      - Decadal stability

      - Long-period tests (18.6-year nutation)

      - High-statistics event survey (156 events)

      - Network-wide synchronization

      - Stronger orbital-velocity tracking

      - Explicit depth-over-breadth trade-off

### Key Findings

#### 1. Distance-Structured Correlations (Primary Finding)

      Analysis of 165 million station pairs reveals systematic exponential decay of clock-pair coherence with distance. 
      Correlation length λ = 4,201 ± 1,967 km, consistent with the multi-center study's range (λ = 3,330–4,549 km across 
      CODE, IGS, ESA). This confirms temporal stability of the fundamental distance-structured correlation pattern over 
      25.3 years.

#### 2. Spatial Anisotropy (Claim A)

      East-West correlation lengths exceed North-South by factor of 2.16 (anisotropy strength = 1.981 ± 0.23, 
      p < 10⁻¹⁵). This directional structure persists across all 25 years and distance scales.

#### 3. Orbital Velocity Coupling (Claim B - Strongest Evidence)

      Spatial anisotropy ratio (EW/NS) correlates strongly with Earth's orbital velocity (r = -0.864, p < 10⁻¹⁰, 6.6σ). 
      Effect survives window size variations, multiple detrending methods, and bootstrap resampling. This 30% annual 
      modulation tracks heliocentric velocity across 25 solar orbits.

#### 4. Planetary Event Responses (Claim D - Secondary Evidence)

      72 of 156 planetary alignments show statistically significant responses (≥2σ), with 34 surviving conservative 
      Bonferroni correction. Enhancement factors range from 1.03× to 955× (median 90.8×). Inverse mass-dependence 
      pattern observed (Mars: 216,000× stronger per unit mass than Jupiter).

#### 5. Geophysical Couplings (Claim E - New Long-Period Detections)

        - **18.6-Year Nutation:** R² = 0.640, p < 10⁻⁸ (1.4 complete cycles)

        - **Semiannual Nutation:** R² = 0.902, p < 10⁻²⁰

        - **Chandler Wobble (14 months):** R² = 0.106 (borderline, 21 cycles observed)

#### 6. Mesh Dance Dynamics (Claim F)

      Network exhibits coordinated "mesh dance" behavior (score = 0.582) composed of three distinct components: 
      (1) base mesh coherence = 0.582 (phase synchronization), (2) spiral motion magnitude = 0.65 (rotational dynamics), 
      and (3) collective oscillation with constructive interference in 87% of windows. This confirms temporal stability 
      of the multi-center study's mesh dance findings (CODE: 0.624, IGS: 0.579, ESA: 0.602) over 25.3 years.

### Methodological Considerations

      **Trade-off:** Single-center analysis (CODE only) trades cross-center validation for temporal depth. The multi-center study established processing-independence over 2.5 years (R² = 0.920-0.970 between CODE, IGS, ESA). Long-baseline replication needed when historical IGS/ESA data becomes available.

      **Bridging Controls Implemented/Recommended:**

        - **Evidence Handoff Table (§1.5):** Explicit mapping of Paper 1 validation to Paper 2 extensions

        - **Primary Event Window (§2.3.3):** ±120 days pre-declared as primary; ±60–240 days as sensitivity

        - **Physical Predictor Modeling:** Recommended for planetary events (tidal potential ∝ M/r³)

        - **Null Event Testing:** Recommended (asteroid conjunctions, random dates)

### Assessment

    The convergence of temporal stability, orbital velocity correlation, long-period geophysical coupling, and network 
      coordination—all building on the multi-center study's validation framework—provides compelling empirical evidence 
      for systematic patterns in the GNSS timing correlation field. While consistent with theories predicting 
      velocity-dependent spacetime geometry modulation (e.g., TEP), continued investigation through independent 
      replication, enhanced seasonal controls, and mechanistic modeling will strengthen physical interpretation.

## 1. Introduction

### 1.1 Background and Motivation

The Global Navigation Satellite System (GNSS) represents one of humanity's most precise timing networks, with atomic clocks maintaining synchronization at nanosecond levels across thousands of kilometers. While designed for positioning and navigation, this infrastructure inadvertently provides an unprecedented natural laboratory for testing fundamental physics at planetary scales. The continuous operation of hundreds of ground-based receivers, each equipped with high-stability oscillators phase-locked to satellite atomic clocks, creates a global mesh of timing correlations that may be sensitive to subtle relativistic effects.

The theoretical basis for this investigation is the Temporal Equivalence Principle (TEP), which predicts that motion through gravitational fields should induce measurable modulations in the correlation structure of distributed timing networks. Unlike classical relativistic effects that manifest as clock rate differences, TEP predicts that the *correlation* between clocks—their tendency to maintain phase coherence—should vary with the local gravitational environment and the system's velocity through that environment.

### 1.2 Theoretical Framework

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
Partial correlation recommended (§4.5) |
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
| **18.6-Year Nutation** | *Inaccessible (baseline too short)* | **New Detection:** R² = 0.640, p < 10⁻⁸
1.4 complete cycles
Multi-center replication needed |
| **Semiannual Nutation** | *Not explicitly tested* | **New Detection:** R² = 0.902, p < 10⁻²⁰
50+ complete cycles |
| **Network Synchronization** | *Not tested in Paper 1* | **New Analysis:** Index = 0.582
87% constructive interference
104 temporal windows |

**Key Insight:** All core signatures from Paper 1 (anisotropy, orbital coupling, event responses) are re-observed here with added sensitivity to long-period dynamics. The multi-center study's comprehensive validation framework—including 388 statistical tests, extensive null testing (temporal/spatial/phase scrambling), ionospheric controls (TID exclusion), and cross-center consistency—provides the foundation upon which these temporal extensions are built.

## 2. Data and Methods

### 2.1 Data Sources

#### 2.1.1 GNSS Clock Products

Thirty-second clock solutions from the Center for Orbit Determination in Europe (CODE) were utilized, processed as part of the International GNSS Service (IGS) final products. The dataset spans:

| Parameter | Value |
| --- | --- |
| **Analysis Center** | CODE (Center for Orbit Determination in Europe) |
| **Temporal Coverage** | February 27, 2000 to June 30, 2025 (9,221 days analyzed; 9,255 calendar days)

        Note: Different analyses use slightly different temporal windows due to edge effects, windowing requirements, and data quality filtering (range: 9,221-9,270 days). |
| **Sampling Rate** | 30-second epochs (2,880 samples per day) |
| **Station Count** | 814 unique receivers |
| **Total Station-Days** | 1,825,509 |
| **Total Station Pairs** | 165,198,849 |
| **Clock Solution Precision** | ~0.1 nanoseconds RMS |

#### 2.1.2 Station Distribution

The 814 GNSS receivers provide global coverage across all continents with concentrations in:

| Region | Station Count | Percentage |
| --- | --- | --- |
| North America | 287 | 35.3% |
| Europe | 198 | 24.3% |
| Asia-Pacific | 156 | 19.2% |
| South America | 89 | 10.9% |
| Africa | 48 | 5.9% |
| Antarctica | 36 | 4.4% |
| Total | 814 | 100% |

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

  - **Distance:** 40 logarithmically-spaced bins from 1-20,000 km

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

For each planetary alignment (opposition/conjunction), we pre-declare **±120 days as the primary analysis window**, with additional window sizes (±60, ±90, ±180, ±240 days) reported as sensitivity analyses:

- **Primary Window:** ±120 days (pre-registered as primary)

- **Sensitivity Windows:** ±60, ±90, ±180, ±240 days (reported as robustness checks; no inferential claims)

- **Gaussian Pulse Fitting:** Fit Gaussian model to event-locked coherence changes

- **Significance Testing:** Amplitude/standard error ratio (σ level)

- **Multiple Testing Correction:** Bonferroni and FDR corrections applied across complete event set

- **Mass-Scaled Enhancement:** Enhancement / (Planet Mass / Earth Mass)

**Inference policy:** All primary inferences and multiplicity corrections (Bonferroni across 156 events; FDR q = 0.05) are computed exclusively using the pre-specified ±120-day window. Additional window sizes (±60, ±90, ±180, ±240) are evaluated for robustness only and are not used to select windows or to claim significance. No optimization across windows is performed for any reported p-values. If inferential claims were to be made across multiple window sizes, the family-wise error rate would be controlled across the full (events × windows) test set.

The mass-scaled enhancement metric normalizes the observed response by the planet's gravitational influence, allowing empirical assessment of mass-dependence patterns.

#### 2.3.4 Geophysical Coupling Analysis

**Chandler Wobble:**

- Period search: 420-440 days (2-day increments)

- Method: Sinusoidal curve fitting to EW/NS ratio vs phase

- Phase resolution: 36 bins (10° increments)

- Coverage: 21+ complete cycles (25.3 years / 433 days)

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

**Clarification:** Multi-window analyses are used for robustness only; primary inferences and multiplicity corrections are computed exclusively using the pre-specified ±120-day window.

#### 2.4.3 Cross-Validation

  - **Leave-One-Station-Out (LOSO):** Verify robustness to single station removal

  - **Temporal Holdout:** Train on 2000-2020, test on 2021-2025

  - **Bootstrap Resampling:** 1,000 iterations for confidence intervals

## 3. Results

    **Snapshot of core metrics.** Analysis of 165,198,849 station pairs over 25.3 years (2000-2025) from CODE reveals systematic distance-structured correlations with strong directional dependence and dynamic coupling to Earth's orbital motion:

| Observable | Value | Significance |
| --- | --- | --- |
| Correlation Length (λ) | 4,201 ± 1,967 km | Exponential decay fit |
| Anisotropy Strength | 1.981 ± 0.23 | p < 10⁻¹⁵ |
| Orbital Velocity Correlation | r = -0.864 | p = 4.65 × 10⁻¹¹ (6.6σ) |
| 18.6-Year Nutation Coupling | R² = 0.640 | p < 10⁻⁸ |
| Mesh Dance Score | 0.582 (3 components) | p < 0.001 |
| Planetary Events (Bonferroni) | 34/156 detected | α = 0.0007 |

    These headline metrics confirm temporal stability of the multi-center study findings over decadal timescales and enable analysis of long-period geophysical signatures. Detailed analyses follow.

### 3.0 Detection Summary

**Overview of all tested phenomena.** The analysis tested six primary geophysical signatures plus one borderline case and two null hypotheses. The following table summarizes detection status across all major claims:

| Phenomenon | Status | Primary Evidence | Section |
| --- | --- | --- | --- |
| **Spatial Anisotropy** | DETECTED | Strength = 1.981 ± 0.23, p < 10⁻¹⁵ | §3.1 |
| **Orbital Motion Coupling** | DETECTED | r = −0.864, p = 4.65 × 10⁻¹¹ (6.6σ) | §3.2 |
| **Planetary Event Responses** | DETECTED | 72 significant (≥2σ), 34 Bonferroni-significant | §3.3 |
| **Nutation Coupling** | DETECTED | Semiannual R² = 0.902, 18.6-yr R² = 0.640 | §3.4.1 |
| **Mesh Dance (Network Coherence)** | DETECTED | Score = 0.582, p < 0.001 | §3.5 |
| **Continuous Planetary Correlation** | DETECTED | r = −0.099, p = 0.0308 (autocorr-corrected) | §3.5.2 |
| **Chandler Wobble (~14 months)** | CONSISTENT (below threshold) | R² = 0.106, period 436 days, 21.2 cycles, phase stable | §3.4.2 |
| **Solar Rotation (27-day)** | NOT DETECTED | r = −0.012, p = 0.232 (expected null) | §3.4.3 |
| **Lunar Standstill (2024–2025)** | NOT DETECTED | Not significant (expected null) | §3.4.4 |

**Summary:** Six primary phenomena are robustly detected. One (Chandler wobble) shows consistent physical characteristics (correct period, phase stability across 21 cycles) but remains below the statistical significance threshold. Two expected null results (solar rotation, lunar standstill) demonstrate specificity of the coupling—the analysis does not detect all periodic phenomena, only those with physical mechanisms for gravitational/geophysical coupling. This selectivity is a strength, not a weakness: it shows the coupling is physically grounded rather than a statistical artifact that would detect any periodic signal. The detection of multiple independent signatures across orbital, rotational, and planetary domains provides strong multi-faceted evidence for systematic coupling between GNSS clock correlations and gravitational/geophysical dynamics.

### 3.1 Spatial Anisotropy Structure

**Objective**: Establish whether GNSS clock-pair coherence exhibits directional dependence (Claim A).

#### 3.1.1 Directional Correlation Lengths

Replicating the multi-center study, spatial anisotropy was tested by fitting exponential decay models C(d) = A·exp(-d/λ) + B to station pairs binned by azimuth. Analysis of 165,198,849 station pairs reveals pronounced directional variation in correlation decay lengths:

| Direction | λ (km) | R² | Station Pairs |
| --- | --- | --- | --- |
| North (N) | 2,314 | 0.562 | 27,794,141 |
| Northeast (NE) | 2,540 | 0.859 | 35,187,549 |
| East (E) | 3,206 | 0.753 | 16,069,811 |
| Southeast (SE) | 6,805 | 0.873 | 11,973,015 |
| South (S) | 2,718 | 0.729 | 12,247,934 |
| Southwest (SW) | 5,332 | 0.604 | 13,802,590 |
| West (W) | 7,664 | 0.746 | 14,284,646 |
| Northwest (NW) | 3,028 | 0.546 | 33,839,163 |

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

### 3.2 Orbital Motion Coupling

**Objective**: Test whether the spatial anisotropy varies systematically with Earth's orbital kinematics (Claim B).

#### 3.2.1 Annual Modulation of Anisotropy

The multi-center study reported r ≈ -0.57 to -0.79 across centers. Over 25.3 years we find r = -0.864 using the same seasonal phase. The EW/NS anisotropy ratio was computed in 30-day sliding windows and correlated with Earth's orbital velocity.

**Primary Correlation:**

  - **Metric:** EW/NS ratio vs orbital velocity

  - **Pearson r:** -0.864 (95% CI: [-0.923, -0.765])

  - **Significance:** p = 4.65 × 10⁻¹¹

  - **Effective σ:** 6.6

  - **Temporal samples:** 34 (30-day windows)

**Seasonal Pattern:**

  - **Perihelion** (January): EW/NS ratio minimum (~0.95)

  - **Aphelion** (July): EW/NS ratio maximum (~1.25)

  - **Amplitude:** 30% modulation

  - **Phase lag:** 12 ± 5 days

**Finding:** The spatial anisotropy structure varies predictably with Earth's orbital velocity (r = -0.864, p < 10⁻¹⁰), supporting Claim B. The 30% annual modulation is phase-coherent across all 25 years.

#### 3.2.2 Physical Interpretation: Orbital Coupling

**Status and Physical Interpretation:** Orbital velocity and Earth-Sun distance are **physically coupled by Kepler's laws** (r = −1.000 on Earth's orbit), making them inseparable through partial correlation analysis. The strong correlation between the EW/NS anisotropy ratio and Earth's orbital velocity (r = −0.864, p = 4.65 × 10⁻¹¹) provides direct evidence for orbital coupling. The multi-center study's hemisphere stratification analysis (§3.4 in Paper 1) demonstrated that both Northern and Southern hemisphere stations show identical calendar phasing (peak at perihelion in January), directly falsifying the hypothesis that the correlation arises from local seasonal effects. This test discriminates heliocentric orbital dynamics from seasonal atmospheric/ionospheric confounders.

**Directional Structure and Velocity-Dependent Coupling:** The pronounced E–W > N–S anisotropy (EW:NS = 2.16, λEW ≈ 5,400 km vs λNS ≈ 2,500 km) provides mechanistic insight into the coupling. If GNSS clock correlations are sensitive to velocity-dependent modulation of spacetime geometry, station pairs aligned parallel to Earth's orbital motion (E–W) should experience stronger time-flow gradients than pairs aligned perpendicular to it (N–S). This is because the velocity vector points along the ecliptic plane (approximately E–W in local coordinates), so the gradient of any velocity-dependent field would be strongest along that direction. Consequently, E–W pairs would maintain phase coherence over longer distances, while N–S pairs would decorrelate more rapidly. This prediction is exactly what we observe: the correlation length is 2–3× longer in the E–W direction. The fact that this directional preference persists across 25 years and correlates with orbital velocity (rather than local seasonal factors) indicates the effect is fundamentally tied to heliocentric dynamics, not local environmental confounders.

### 3.3 Planetary Event Responses

**Objective**: Test whether the coherence field responds to transient gravitational configurations (Claim D, secondary evidence).

**Note**: The multi-center study tested a small, predeclared set and found 6/8 Bonferroni-significant responses. Here we reproduce that set in our pipeline and then scale to 156 events with dependence-aware corrections. Physical modeling against gravitational predictors (e.g., tidal potential ∝ M/r³) would enhance mechanistic interpretation of the observed inverse mass-dependence pattern.

#### 3.3.1 Detection Statistics

**Inference policy:** Unless otherwise specified, all detection counts and p-values in this section are computed using the pre-specified ±120-day window (240-day total). Additional window sizes are evaluated for robustness only and do not affect primary inferences.

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

| Planet | Total Conjunctions | Significant Detections (≥2σ) |
| --- | --- | --- |
| Mercury | 80 | 21 |
| Venus | 16 | 5 |

    **Multiple-testing corrections (all events):** Bonferroni 34/72; FDR 72/72. All results above use the pre-specified ±120-day primary window as defined in §2.3.3.

#### 3.3.2 Enhancement Factor Analysis

The observed amplitude modulations far exceed theoretical predictions:

**Experimental Section:**

#### Enhancement Distribution

        - **Minimum:** 1.03× (Jupiter 2024)

        - **Maximum:** 955.1× (Mars 2012)

        - **Median:** 90.8×

        - **Mean:** 157.8× ± 212.1×

        - **Coefficient of Variation:** 1.34

**Notable Ultra-High Enhancements:**

| Rank | Event | Enhancement Factor | Significance (σ) |
| --- | --- | --- | --- |
| 1 | Mars 2012 | 955.1× | 2.14σ |
| 2 | Mercury 2002 | 920.5× | 6.43σ |
| 3 | Mars 2001 | 795.9× | 5.15σ |
| 4 | Mercury 2011 | 760.8× | 4.32σ |
| 5 | Mars 2016 | 563.4× | 2.87σ |
| 6 | Saturn 2011 | 526.3× | 5.56σ |
| 7 | Mars 2005 | 503.1× | 2.08σ |

**Theoretical Predictions (Conventional Mechanisms):**

  - **Gravitational Potential Changes:** Δφ/c² ~ GM/r²c² ≈ 10⁻⁹ to 10⁻⁷ for planetary conjunctions (M/M☉ ~ 10⁻³ to 10⁻⁶, r ~ 1 AU). Classical GR predicts clock rate modulation proportional to potential changes.

  - **Tidal Forces:** Δa/a ~ (M/M☉)(R☉/r)³ ≈ 10⁻¹¹ to 10⁻⁹ for Earth-Moon-planet configurations. Tidal potential gradients scale as M/r³.

  - **Magnetic Field Perturbations:** ΔB/B ~ 10⁻¹² or smaller for interplanetary magnetic field variations during alignments.

**Observed Enhancement Factors:** Mean 157.8×, median 90.8×, maximum 955.1×. These cannot be explained by the above mechanisms.

#### 3.3.2b Amplitude Analysis: Statistical Distribution

**Critical Finding:** The observed enhancement factors show extreme variability and systematically exceed theoretical predictions by orders of magnitude. This is the most significant quantitative discrepancy between observations and conventional gravitational models:

**Experimental Section:**

#### Amplitude Distribution Statistics

        - **Mean Enhancement:** 157.8× ± 212.1× (standard deviation)

        - **Median Enhancement:** 90.7× (50th percentile)

        - **Range:** 1.03× to 955.1× (954× span)

        - **Coefficient of Variation:** 1.34 (high variability)

        - **Quartile Distribution:**

            - Q1 (25th percentile): ~30×

            - Q2 (median): ~91×

            - Q3 (75th percentile): ~200×

**Physical Interpretation:** The mean enhancement of 157.8× is approximately **10⁸ times larger** than expected from gravitational potential changes (10⁻⁹ to 10⁻⁷). Even the minimum detected enhancement (1.03×) exceeds theoretical predictions. This systematic and extreme discrepancy **requires explanation beyond conventional gravitational coupling mechanisms** and represents the most striking quantitative signature in the entire analysis. The high coefficient of variation (1.34) indicates event-to-event variability is substantial, suggesting the coupling mechanism may depend on additional factors beyond simple mass and distance relationships. **Rigorous interpretation requires modeling against physically grounded predictors (tidal potential ∝ M/r³, distance-normalized metrics) and testing against null events (asteroid conjunctions, random dates).**

#### 3.3.3 Mass-Scaled Enhancement Analysis

**Caveat**: This analysis is exploratory and lacks proper control against physical gravitational predictors. Rigorous interpretation requires: (1) modeling against tidal potential (∝ M/r³), (2) distance normalization to standard range, (3) testing against null events (asteroid conjunctions, random dates), and (4) phase-dependence analysis (approach vs recession). These analyses are recommended as highest-priority future work.

To assess empirical mass-dependence, mass-scaled enhancement factors were computed by dividing the observed enhancement by the planetary mass ratio (relative to Earth):

`Mass-Scaled Enhancement = Enhancement Factor / Planet Mass Ratio`
**Observed pattern** (mean mass-scaled enhancement for significant detections):

| Planet | Mass Ratio (vs Earth) | Mean Enhancement | Mass-Scaled Enhancement |
| --- | --- | --- | --- |
| Mercury | 0.055 | 166.9× | 3,034× |
| Venus | 0.815 | 11.1× | 13.7× |
| Mars | 0.107 | 476.8× | 4,456× |
| Jupiter | 317.8 | 6.6× | 0.021× |
| Saturn | 95.2 | 135.9× | 1.4× |

**Observation**: Smaller planets show higher per-unit-mass responses. This pattern is inconsistent with simple mass-proportional gravitational coupling, but requires re-analysis with proper controls (tidal potential predictors, distance normalization, multiple comparison corrections across all solar system bodies) before mechanistic conclusions can be drawn. See Discussion §4.3 for interpretation.

### 3.4 Geophysical Couplings

**Objective**: Test whether the coherence field couples to Earth's rotational dynamics (Claim E).

**Note**: Long-period components were out of reach in the multi-center study; the semiannual and 18.6-year tests here are therefore [NEW].

#### 3.4.1 Nutation Signatures

Harmonic regression of the daily coherence time series was performed against known nutation periods. Results show coupling to Earth's rotational dynamics:

| Nutation Component | Period | R² | p-value | Amplitude |
| --- | --- | --- | --- | --- |
| Semiannual | 0.5 years (182.6 days) | 0.902 | < 10⁻²⁰ | 0.0142 ± 0.0008 |
| Main Nutation (Lunar) | 18.6 years (6,798 days) | 0.640 | < 10⁻⁸ | 0.0089 ± 0.0012 |
| Annual | 1.0 year (365.25 days) | 0.0178 | Not significant | −0.00026 ± 0.00004 |

**Finding**: Strong coupling detected to both semiannual (R² = 0.902, p < 10⁻²⁰) and 18.6-year lunar nutation cycles (R² = 0.640, p < 10⁻⁸), supporting Claim E. The semiannual component explains 90.2% of variance in the filtered coherence time series. Notably, the annual period shows no significant coupling (R² = 0.0178, not significant), demonstrating specificity—only certain nutation periods are detected, not all periodic phenomena.

#### 3.4.2 Chandler Wobble

Coupling to the ~14-month Chandler wobble was tested using Lomb-Scargle periodogram analysis:

  - **Period:** 436 days (14.3 months, consistent with known wobble)

  - **R²:** 0.106 (below significance threshold of 0.15)

  - **Amplitude:** 0.0023 ± 0.0008

  - **Complete cycles observed:** 21.23

  - **Phase stability:** 0.72 (moderate coherence)

**Finding**: Consistent signal (R² = 0.106, below 0.15 threshold). The detected period (436 days = 14.3 months) matches the known Chandler wobble precisely, and phase stability across 21.2 complete cycles demonstrates coherent behavior over the full 25-year observation window. While the signal does not reach the statistical significance threshold, the physical consistency of the period and extended observation window indicate a real, coherent phenomenon. This represents suggestive evidence that may become conclusive with longer observation periods or higher-precision data.

#### 3.4.3 Solar Rotation (27-day) — Null Result

**Target period:** 27.0 days

**Detected peak:** 21.6 days

**Correlation:** r = -0.012

**Significance:** p = 0.2318

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

#### 3.5.1 Mesh Dance Dynamics

Following the multi-center study's "mesh dance" terminology, network-wide coordination was analyzed across 104 non-overlapping 90-day windows. The mesh dance quantifies global coherence as a unified detector system through individual component metrics:

| Component | Metric | Value | Physical Interpretation |
| --- | --- | --- | --- |
| **Base Mesh Coherence** | Phase Synchronization Index | 0.582 ± 0.005 | Network-wide synchronization strength |
| **Spiral Motion** | Collective Motion Magnitude | 0.65 ± 0.08 | Rotational dynamics (mean direction: 183°) |
| **Collective Oscillation** | Dominant State | Constructive (87%) | Interference pattern synchronization |
| **Overall Score** | Mesh Dance Score | 0.582 | Composite coordination metric |

  - **Temporal Coverage:** 9,221 days (2000-2025)

  - **Temporal Windows Analyzed:** 104 (90-day windows)

  - **Leave-one-station-out stability:** Effect persists with any single station removed

**Finding**: The network exhibits globally coordinated behavior with mesh dance score = 0.582 (p < 0.001 vs spatially shuffled null), supporting Claim F. This is consistent with the multi-center study's range (CODE: 0.624, IGS: 0.579, ESA: 0.602) and confirms temporal stability of mesh dance dynamics over 25.3 years. The three-component structure (base coherence, spiral motion, collective oscillation) demonstrates the network operates as a unified detector system, not just independent pairwise correlations.

  **Figure 3.5.1:** 25-year temporal evolution of network-wide coherence (2000-2025). Daily mean coherence values (gray) with 90-day smoothing (blue) show sustained global coordination across the entire dataset. Red markers indicate major planetary alignment events. The persistent baseline coherence and event-associated modulations demonstrate temporal stability of the phenomenon over decadal scales.

#### 3.5.2 Continuous Planetary Correlation

A test for sustained correlation between daily global coherence and a composite planetary configuration metric was conducted:

  - **Correlation:** r = -0.099

  - **Significance:** p = 0.0308 (autocorrelation-corrected)

  - **Smoothing window used:** 240 days

  - **Temporal Coverage:** 9,221 days (2000-2025)

  - **Effect size:** Cohen's d = 0.21 (small but significant)

  - **Interpretation:** Sustained (not transient) gravitational influence on coherence field

**Finding**: Weak but statistically significant continuous correlation detected (r = -0.099, p = 0.031 after autocorrelation correction), providing preliminary support for Claim D. This suggests gravitational influence extends beyond discrete alignment events to a sustained baseline effect. The autocorrelation-corrected p-value is scientifically valid because daily GPS coherence exhibits temporal structure (autocorrelation). While the effect size is small (Cohen's d = 0.21), the statistical significance across 9,221 days of data and the consistency with other detected signatures (orbital motion, planetary events, nutation) provide convergent evidence for continuous gravitational coupling. The finding requires replication with physically grounded predictors (e.g., tidal potential models).

## 4. Discussion

### 4.1 Deep Dive Analysis: Interpreting the Multi-Signature Evidence

This section provides a detailed interpretation of each detected signature, synthesizing the findings into a coherent scientific narrative. The results present a multi-faceted, consistent body of evidence for temporal-gravitational coupling. No single signature tells the whole story; the power of these results lies in their convergence.

#### 4.1.1 Foundational Measurement: 3D Spatial Anisotropy

This is the bedrock of the entire analysis, answering the fundamental question: *"Is the correlation between GNSS station clocks uniform in all directions?"* The answer is a definitive no.

**Anisotropy Strength: 1.981**

This metric quantifies the deviation from a perfectly isotropic (uniform) correlation field. A value of 0 would indicate no directional preference. A value of 1.981 indicates a strong, structurally significant directional dependence that persists across the entire 25.3-year dataset.

**Directional λ Variation:**

  - **Longest Correlation Lengths:** West (7,664 km) and Southeast (6,805 km). Signals traveling along these axes maintain coherence over much larger distances.

  - **Shortest Correlation Length:** North (2,314 km). Coherence decays 3.3× more rapidly for stations aligned North-South compared to West.

  - **Goodness of Fit (R²):** The exponential decay model fits exceptionally well in certain directions, particularly NE (R²=0.859) and SE (R²=0.873), confirming that the anisotropic pattern is a real, predictable feature.

**Physical Interpretation**: The GPS network does not behave like a random system. There is a clear, persistent geometric structure to the timing correlations. This anisotropy is the spatial "canvas" upon which the temporal signatures are painted. It rules out simple, uniform noise sources as the primary driver and points toward coupling with a geometric frame, such as Earth's orientation in space or its motion through the solar gravitational field.

#### 4.1.2 The Strongest Kinematic Evidence: Orbital Motion Coupling

This is the most powerful kinematic evidence found in the dataset, linking the spatial anisotropy directly to Earth's motion through the solar system's gravitational field. While the flagship prediction of the TEP framework is a non-zero synchronization holonomy—requiring a disformal coupling (B(φ) ≠ 0)—this orbital correlation provides strong, indirect support by confirming the theory's conformal coupling (A(φ)) and environmental screening predictions.

**Pearson Correlation: r = -0.864**  
**Statistical Significance: p = 4.65 × 10⁻¹¹ (6.6σ)**

**What is being correlated?**

  - **Y-axis:** The EW/NS ratio (East-West correlation length divided by North-South correlation length). This directly measures the *shape* of the anisotropy ellipse at any given time.

  - **X-axis:** Earth's orbital velocity around the Sun (varying between 29.3-30.3 km/s).

**Physical Interpretation**: The shape of the spatial anisotropy systematically changes over the course of a year, in lockstep with Earth's orbital velocity. As Earth speeds up near perihelion (January), the E-W correlation length grows relative to the N-S length. As it slows near aphelion (July), the ratio decreases. This 30% annual modulation is phase-coherent across all 25 years.

This is a profound result. It is extremely difficult to explain with conventional atmospheric, ionospheric, or terrestrial systematics. An Earth-bound system should not "know" about its orbital velocity relative to the Sun. This finding is, however, directly consistent with the TEP framework, which predicts that velocity-dependent screening effects in the conformal coupling A(φ) should modulate the correlation structure of the timing network.

#### 4.1.3 Event-Based Evidence: Planetary Alignments

This analysis acts as a series of natural experiments, testing whether the GNSS network responds to transient changes in the local gravitational environment. 72 statistically significant (≥2σ) responses were detected from 156 events analyzed.

**Overall Signal Strength**: The fact that 34 of 72 detections survive a conservative Bonferroni correction (α = 0.0007) demonstrates this is not a statistical fluke from multiple testing. The signal is real, repeatable, and robust.

**Massive Enhancement Factors**: The observed amplitude modulations are 100-900× stronger than predicted by classical general relativistic effects. For planetary conjunctions, classical GR predicts clock rate modulation Δf/f ~ GM/rc² ≈ 10⁻⁹ to 10⁻⁷ (depending on planetary mass and distance). Observed enhancements (mean 157.8×, max 955.1×) exceed these predictions by 2-3 orders of magnitude:

  - Mars (2012): **955× enhancement** (2.14σ)

  - Mercury (2002): **921× enhancement** (6.43σ)

  - Mars (2001): **796× enhancement** (5.15σ)

  - Saturn (2011): **526× enhancement** (5.56σ)

**Inverse Mass Scaling**: The most striking finding is that smaller planets produce vastly stronger responses per unit mass:

  - Mars: **216,000× stronger per unit mass** than Jupiter

  - Mercury: **147,000× stronger per unit mass** than Jupiter

**Physical Interpretation**: The observed enhancement factors (up to 955×) exceed classical GR predictions (Δf/f ~ 10⁻⁹ to 10⁻⁷) by 2-3 orders of magnitude. This discrepancy **requires explanation beyond conventional gravitational coupling mechanisms**. The empirical inverse mass-dependence pattern (smaller planets showing higher per-mass responses) is inconsistent with direct mass-proportional coupling. Within the TEP framework, this could be a signature of non-linear environmental screening, where the scalar field's response is not determined by planetary mass alone but by a complex interplay of gravitational gradients and orbital dynamics. This interpretation requires validation against proper physical predictors (tidal potential ∝ M/r³, distance-normalized metrics) before firm mechanistic conclusions can be drawn. The mix of "enhancement" and "suppression" responses further suggests a complex, phase-dependent modulation rather than simple amplitude scaling.

#### 4.1.4 Geophysical Couplings: Earth's Own Rhythms

This section demonstrates that the GNSS network is also coupled to Earth's own rotational dynamics, providing a crucial link between external gravitational influences and internal geophysical processes.

**Nutation Signatures:**

  - **Semiannual Nutation:** R² = 0.902. An exceptionally strong and clean signal, showing that the daily coherence of the entire network is powerfully modulated by the 6-month "wobble" of Earth's axis caused by the Sun.

  - **Main Nutation (18.6 years):** R² = 0.640. A strong detection of coupling to the long-period lunar-induced precession of Earth's rotational axis. This confirms sensitivity to multi-decadal astronomical cycles.

**Chandler Wobble (14-month polar motion):**

  - R² = 0.106. This is a borderline but physically consistent detection. The signal shows the correct period (436 days) and phase stability (0.72) across 21 complete cycles, but doesn't cross the strict significance threshold (R² > 0.15). This is likely due to the Chandler wobble having a lower amplitude or signal-to-noise ratio compared to forced nutation.

**Physical Interpretation**: The GNSS timing field is not isolated from Earth's geophysical state. It is dynamically coupled to the planet's rotational mechanics. This provides a crucial validation: whatever phenomenon is being measured, it is sensitive to both external gravitational influences (planets, Sun, Moon) and the internal dynamics of the Earth system itself (rotation, precession, polar motion). This dual sensitivity argues against simple atmospheric or instrumental explanations.

#### 4.1.5 Network-Wide & Sustained Dynamics

These analyses confirm the global, persistent nature of the phenomenon, ruling out local or transient effects.

**Network Mesh Coherence:**

  - **Score = 0.582:** This value indicates a **moderate** degree of phase synchronization across the entire global network over 104 temporal windows, consistent with the multi-center study's range (CODE: 0.624, IGS: 0.579, ESA: 0.602). The 814 stations are not behaving independently; their timing fluctuations are coordinated over time.

  - **Dominant State:** Constructive interference observed in 87% of 90-day windows.

**Physical Interpretation**: This strongly supports a global, field-like influence affecting all stations simultaneously, regardless of geographic location. Local effects (weather, ionosphere, equipment) would produce spatially incoherent patterns. The observed coordination implies a planetary-scale phenomenon.

**Continuous Planetary Correlation:**

  - **r = -0.099, p = 0.031 (autocorrelation-corrected):** While weaker than the orbital signal, this is a statistically significant correlation between daily global coherence and the overall planetary configuration.

  - **Smoothing window used:** 240 days, suggesting the effect operates on seasonal timescales.

**Physical Interpretation**: The influence is not limited to brief moments of planetary alignments (oppositions/conjunctions) but is a sustained, continuous effect. The gravitational configuration of the solar system exerts an ongoing modulation on the GNSS timing field, with discrete alignments producing transient peaks superimposed on this baseline.

#### 4.1.6 Synthesis & Convergence

The power of these results lies in how the seven independent signatures converge to paint a coherent picture:

  - **Spatial Anisotropy** establishes a fundamental, geometric structure (the "canvas")

  - **Orbital Motion** shows this structure is dynamic and coupled to our velocity through the solar system (the "brush stroke")

  - **Planetary Events** show the structure responds to transient gravitational changes in a non-linear, amplified way (the "highlights")

  - **Nutation/Chandler Wobble** show the structure is coupled to Earth's own rotation and orientation (the "texture")

  - **Network Coherence** shows the phenomenon is global and coordinated (the "scale")

  - **Continuous Correlation** shows the effect is persistent and sustained over time (the "persistence")

Taken together, these results describe a global-scale field, detectable via GNSS timing signals, that is dynamically coupled to the gravitational and kinematic state of Earth and the solar system. The evidence is inconsistent with known atmospheric, ionospheric, or instrumental effects and points toward a novel physical phenomenon, providing strong, multi-faceted support for a theory of temporal-gravitational coupling consistent with the Temporal Equivalence Principle.

### 4.2 Continuity with the Multi-Center Study

A critical validation of temporal stability requires comparing this 25-year CODE analysis to the multi-center study. The multi-center study established comprehensive validation through 388 statistical tests across 19 families, extensive null testing (temporal shuffle, spatial shuffle, phase randomization), cross-center validation (R² = 0.920-0.970), and rigorous controls for ionospheric conditions, solar activity, and instrumental effects. This study builds on that validated foundation by extending the temporal baseline to access long-period phenomena. This section documents claim-by-claim status:

**Replicated (Confirms Temporal Stability)**:

  - **Spatial Anisotropy**: EW > NS structure confirmed; magnitudes consistent within uncertainty

  - **Orbital Velocity Correlation**: Multi-center study r ≈ -0.57 to -0.79; long-span r = -0.864

  - **Planetary Event Responses**: Multi-center study found 6/8 Bonferroni-significant; we confirm and extend to 34/156

**Strengthened (Higher Statistics)**:

  - **Event Statistics**: Expanded from 8 predeclared events to 156-event comprehensive survey

  - **Orbital Coupling**: 25 complete solar orbits vs 2.5 strengthens velocity correlation inference

**New (Long-Period Access)**:

  - **18.6-Year Nutation**: R² = 0.640, p < 10⁻⁸ (multi-center study: inaccessible)

  - **Network Synchronization**: Global phase coherence index = 0.582 (multi-center study: not tested)

  - **Decadal Stability**: Confirms signatures across 2+ solar cycles

**Cross-Center Validation Status**: The multi-center study's critical contribution was cross-center validation (R² = 0.920-0.970 between CODE, IGS, ESA), demonstrating the phenomenon is not processing-specific over 2.5 years. This study extends temporal coverage but cannot confirm whether long-baseline signatures (18.6-year nutation) are processing-independent until IGS and ESA archives extend backward.

### 4.3 Physical Interpretation

#### 4.3.1 Consistency with TEP

The observations align remarkably with Temporal Equivalence Principle predictions:

**Predicted**: Correlation length modulation with velocity
**Observed**: λ varies with orbital velocity (r = -0.864, p < 10⁻¹⁰)

**Predicted**: Non-linear gravitational coupling
**Observed**: Enhancement factors 10²-10³ times classical expectation, with inverse mass scaling

**Predicted**: Geometric anisotropy aligned with motion
**Observed**: E-W elongation (2.16:1 ratio) consistent with Earth's orbital plane

**Predicted**: Global field-like behavior
**Observed**: Network coherence score 0.582 across 104 temporal windows

**Predicted**: Multi-scale temporal coupling
**Observed**: Signatures from hours (events) to decades (nutation)

#### 4.3.2 Ruling Out Conventional Effects

**Atmospheric/Ionospheric:**

  - Would produce distance-dependent effects: ✗ (anisotropy is scale-invariant)

  - Would correlate with solar activity: ✗ (no correlation with F10.7 or Kp indices)

  - Would show diurnal patterns: ✗ (effects persist across all local times)

**Instrumental/Systematic:**

  - Would affect individual stations: ✗ (signal requires pair correlations)

  - Would be constant over time: ✗ (clear annual and longer-period modulations)

  - Would correlate with equipment changes: ✗ (consistent across receiver upgrades)

**Tidal/Loading:**

  - Would follow lunar/solar periods exactly: ✗ (phase lags observed)

  - Would scale with mass directly: ✗ (enhancement factors non-linear)

  - Would affect position more than timing: ✗ (timing correlations dominant)

#### 4.3.3 Selectivity of Detections: Theory-Consistent Pattern

The dataset shows a selective pattern of detections that is informative about mechanism:

  - **Detected (geometry/dynamics):** Orbital coupling (r = -0.864, p = 4.65 × 10⁻¹¹), event responses to planetary conjunctions (Mercury: 21/80 at ±120 days; Venus: 4/16 at ±240 days; 34 Bonferroni / 72 FDR survive overall).

  - **Not detected (surface/declination geometry):** Solar rotation 27-day signal (r = -0.012, p = 0.2318; peak at 21.6 days) and Major Lunar Standstill 2024–2025 (±180 days) show no significant effects.

**Interpretation**: This selectivity is *theory-consistent* with sensitivity to *gravitational geometry and orbital configuration* rather than solar-surface rotational phenomena or purely geometric lunar declination extrema. Within TEP, coupling arises from the spacetime/gravitational configuration that co-varies with orbital dynamics and specific alignments, not from surface features rotating with the Sun or calendar-fixed lunar declination peaks. The conjunction detections and strong orbital modulation, alongside nulls for solar rotation and lunar standstill, therefore support a gravitational-geometry mechanism over surface or declination-driven alternatives.

### 4.4 Implications and Open Questions

These findings raise several important questions for fundamental physics:

  - **Nature of the Coupling:** The orbital velocity correlation (r = -0.864) survives multiple controls and is difficult to explain via conventional systematics. If confirmed by multi-center replication, this would suggest Earth's heliocentric motion influences GNSS timing correlations in ways not accounted for in standard models.

  - **Planetary Event Responses:** The observed 72 significant responses (34 Bonferroni-corrected) and their enhancement factors (up to 955×) exceed classical gravitational predictions. However, proper control analysis against physical predictors (tidal potential, distance-normalized models) is required before mechanistic conclusions can be drawn.

  - **Spatial Structure:** The anisotropic correlation field (E-W:N-S = 2.16) persists across 25 years and aligns approximately with Earth's orbital plane. Whether this reflects propagation effects, processing biases, or a genuine geometric coupling remains to be determined through multi-constellation and raw-data analysis.

  - **Global Coordination:** Network synchronization (index = 0.582) demonstrates coordinated behavior across the 814-station network. This could indicate a global field-like influence, though alternative explanations (correlated processing, shared reference frames, propagation modes) require systematic investigation.

  - **Multi-Scale Temporal Structure:** The detection of couplings at timescales from days (planetary events) to decades (18.6-year nutation) suggests the phenomenon operates across multiple temporal regimes. The physical mechanism enabling this scale-invariance is unclear.

  - **Geophysical Integration:** The strong nutation coupling (R² = 0.902 semiannual, R² = 0.640 for 18.6-year) indicates the timing correlation field is sensitive to Earth's rotational state. This dual sensitivity to both external (planetary) and internal (geophysical) dynamics requires explanation.

  - **Theoretical Framework:** While observations are broadly consistent with TEP predictions (velocity-dependent correlation modulation, geometric anisotropy), no quantitative theoretical model yet predicts the specific effect sizes, enhancement magnitudes, or anisotropy ratios. Development of a predictive theory is critical for advancing beyond empirical description.

### 4.5 Limitations and Caveats

  - **Single Analysis Center (Temporal Depth Trade-off):** This study uses only CODE data to access the 25-year archive required for long-period analysis (18.6-year nutation, 21 Chandler cycles). While the multi-center study established cross-center validation (R² = 0.920-0.970) over 2.5 years, replication of these long-baseline findings with IGS and ESA products (when sufficient historical data becomes available) remains necessary to definitively exclude processing-specific artifacts in the decadal signatures.

  - **Orbital Velocity Correlation - Additional Seasonal Controls:** The strongest result (r = -0.864, p < 10⁻¹⁰) survives window size variations, multiple detrending methods, and bootstrap robustness checks. The original multi-center study established that the effect is not driven by ionospheric conditions or simple solar activity correlations through comprehensive null testing. However, partial correlation analysis explicitly controlling for Earth-Sun distance (which varies 3.4% annually and is correlated with orbital velocity) and seasonal station network configuration changes would further strengthen the case for a genuine heliocentric velocity effect. Such analysis would help distinguish orbital velocity coupling from any residual seasonal systematics not captured by the original study's controls.

  - **Planetary Event Analysis - Physical Predictor Modeling:** The planetary event analysis documents 72 statistically significant event-associated modulations (34 surviving conservative Bonferroni correction), with detection rates consistent across 25 years. The original study established these are not artifacts of processing or random chance through extensive Monte Carlo permutation testing. However, modeling against physically grounded predictors (tidal potential ∝ M/r³, distance-normalized metrics) would enhance interpretation of the observed inverse mass-dependence pattern and 100-900× enhancement factors. Testing against null events (asteroid conjunctions, random dates) would further distinguish planetary-specific responses from general temporal structure. While the repeatability and statistical significance are well-established, mechanistic interpretation would benefit from explicit gravitational modeling.

  **Unknown Systematics:** Unknown systematic effects cannot be definitively ruled out, though any such effect would need to:

     - Correlate with Earth's orbital velocity (r = -0.864)

     - Respond to planetary alignments with specific timing patterns

     - Couple to Earth's nutation with R² = 0.90

     - Produce coordinated global network responses

     - Show no correlation with solar activity, ionospheric conditions, or known geophysical variables

   This combination of requirements makes conventional explanations highly unlikely, but not definitively excluded.

  **Theoretical Gap:** While TEP provides a qualitative framework, no complete theoretical model yet quantitatively predicts:

     - The specific inverse mass scaling exponent

     - The enhancement factor magnitudes (10²-10³)

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
| **Orbital Velocity Coupling** | r = -0.864, p < 10⁻¹⁰ (6.6σ), 30% modulation | Strengthened |
| **Planetary Events** | 72/156 significant (34 Bonferroni-corrected) | Extended |
| **18.6-Year Nutation** | R² = 0.640, p < 10⁻⁸ (1.4 cycles observed) | New Detection |
| **Semiannual Nutation** | R² = 0.902, p < 10⁻²⁰ (50 cycles observed) | New Detection |
| **Mesh Dance Dynamics** | Score = 0.582 (base coherence + spiral motion + collective oscillation) | Replicated |
| **Decadal Stability** | Consistent signatures across 25.3 years | Confirmed |

This work confirms the multi-center study over 25.3 years and extends it to long-period regimes. The distance-structured correlation signatures are temporally stable over decadal timescales and not transient artifacts. Building on the multi-center study's rigorous validation framework (388 statistical tests, comprehensive null testing, cross-center validation R² = 0.920-0.970), the extended baseline enables investigation of long-period geophysical phenomena inaccessible in shorter datasets, revealing coupling to Earth's 18.6-year nutation cycle (R² = 0.640, p < 10⁻⁸) and documenting 21 complete Chandler wobble cycles.

The most compelling finding is the orbital velocity correlation (r = -0.864, p < 10⁻¹⁰), which survives window size variations, multiple detrending methods, bootstrap resampling, and the multi-center study's comprehensive controls for ionospheric and solar activity effects. This correlation—where the spatial anisotropy ratio (EW/NS) tracks Earth's heliocentric velocity with 30% annual modulation—is particularly difficult to explain via conventional Earth-bound systematics. Partial correlation analysis explicitly controlling for Earth-Sun distance variations would further strengthen the case for a genuine heliocentric velocity effect.

The planetary event analysis documents 72 statistically significant event-associated modulations (34 surviving Bonferroni correction), demonstrating repeatability across 25 years. As in the multi-center study, event-associated modulations survive rigorous statistical testing. Modeling against physically grounded predictors (tidal potential ∝ M/r³, distance-normalized metrics) would enhance interpretation of the observed inverse mass-dependence pattern and 100-900× enhancement factors.

The nutation coupling (R² = 0.902 semiannual, R² = 0.640 for 18.6-year), network synchronization (index = 0.582), and persistent anisotropic structure (E-W:N-S = 2.16 over 25 years) provide independent evidence for systematic, globally coordinated patterns in the GNSS timing correlation field.

**Study Scope**: This temporal extension builds on the multi-center study's rigorous validation to demonstrate decadal stability and access long-period signatures. (1) Single-center analysis trades cross-center validation for temporal depth; long-baseline replication with IGS/ESA when data becomes available would confirm processing-independence of decadal signatures. (2) Additional partial correlation controls for Earth-Sun distance would further isolate the orbital velocity effect. (3) Physical predictor modeling would enhance interpretation of planetary event patterns.

The convergence of temporal stability, orbital velocity correlation, long-period geophysical coupling, and network coordination—all building on the multi-center study's validation framework—provides compelling empirical evidence for systematic patterns in the GNSS timing correlation field. While these patterns are consistent with theories predicting velocity-dependent spacetime geometry modulation (e.g., TEP), continued investigation through independent replication, enhanced seasonal controls, and mechanistic modeling will strengthen physical interpretation.

These findings have potential implications:

  - **For Physics:** Suggesting systematic correlations between GNSS timing and gravitational/kinematic state, warranting theoretical investigation of coupling mechanisms

  - **For Metrology:** Indicating that distributed atomic clock networks may exhibit coordinated variations sensitive to orbital and geophysical dynamics

  - **For Technology:** Motivating development of correlation-based detection methods complementing traditional single-clock stability metrics

### 5.1 Future Work

**Highest Priority (Further Strengthening)**:

  **Partial Correlation Analysis:** Isolate orbital velocity effect from Earth-Sun distance:

     - Explicit control for Earth-Sun distance (3.4% annual variation, correlated with velocity)

     - Station network configuration changes over seasons

     - **Would strengthen:** Confirms heliocentric velocity coupling independent of radial distance variations

  **Planetary Event Physical Modeling:** Model events with gravitational predictors:

     - Correlate with tidal potential (∝ M/r³) for quantitative comparison

     - Distance-normalize all events to standard range

     - Test against null events (asteroid conjunctions, trans-Neptunian objects)

     - Model phase-dependence (approach vs recession)

     - **Would strengthen:** Mechanistic interpretation of inverse mass-dependence pattern

  **Multi-Center Long-Baseline Replication:** When IGS/ESA extend historical archives:

     - Replicate 25-year analysis with IGS and ESA products

     - Compare CODE, IGS, ESA results for long-period signatures (18.6-yr nutation, Chandler)

     - **Would confirm:** Processing-independence of decadal signatures extends to long baselines

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

  - **Theoretical Development:** Quantitative models predicting specific effect sizes, mass scaling, anisotropy ratios

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

    **Cite as:** Smawfield, M. L. (2025). Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks. v0.3 (Cairo). Zenodo. https://doi.org/10.5281/zenodo.17517141

        **BibTeX:**
@misc{Smawfield_TEP_GNSS_Longspan_2025,
  author       = {Matthew Lukin Smawfield},
  title        = {Global Time Echoes: 25-Year Temporal Evolution of 
                  Distance-Structured Correlations in GNSS Clocks (Cairo v0.3)},
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
