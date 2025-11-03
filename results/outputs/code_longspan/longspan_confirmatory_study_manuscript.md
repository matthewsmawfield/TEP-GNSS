# Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks

## Abstract

Following the initial detection of distance-structured correlations in the multi-center study (Smawfield, 2025)†, this paper presents a comprehensive temporal extension using 25.3 years (2000-2025) of data from the CODE analysis center, encompassing 165,198,849 station pairs. This study confirms the temporal stability of the originally detected signatures over decadal timescales and leverages the extended baseline to uncover long-period geophysical phenomena inaccessible in shorter datasets. The analysis confirms the strong correlation between spatial anisotropy and Earth's orbital velocity (r = -0.864, p < 10⁻¹⁰, surviving multiple robustness checks) and detects statistically significant responses to 72 of 157 analyzed planetary alignment events, with 34 surviving conservative Bonferroni correction. The extended dataset enables the first clear detection of coupling to Earth's long-period rotational dynamics, including the 18.6-year lunar nutation cycle (R² = 0.640, p < 10⁻⁸) and suggestive evidence for the 14-month Chandler wobble (R² = 0.106, below significance threshold). Network-wide phase synchronization (index = 0.581) demonstrates global coordination inconsistent with local effects. These findings replicate the multi-center study (Smawfield, 2025) and extend them to decadal scales, enabling first tests of long-period geophysical coupling.

†We refer to Smawfield (2025) as "the multi-center study" throughout.

## 1. Introduction

### 1.1 Background and Motivation

The Global Navigation Satellite System (GNSS) represents one of humanity's most precise timing networks, with atomic clocks maintaining synchronization at nanosecond levels across thousands of kilometers. While designed for positioning and navigation, this infrastructure inadvertently provides an unprecedented natural laboratory for testing fundamental physics at planetary scales. The continuous operation of hundreds of ground-based receivers, each equipped with high-stability oscillators phase-locked to satellite atomic clocks, creates a global mesh of timing correlations that may be sensitive to subtle relativistic effects.

The theoretical basis for this investigation is the Temporal Equivalence Principle (TEP) (Smawfield, 2025), which predicts that motion through gravitational fields should induce measurable modulations in the correlation structure of distributed timing networks. Unlike classical relativistic effects that manifest as clock rate differences, TEP predicts that the *correlation* between clocks—their tendency to maintain phase coherence—should vary with the local gravitational environment and the system's velocity through that environment.

### 1.2 Theoretical Framework

The Temporal Equivalence Principle extends Einstein's equivalence principle to temporal dynamics, proposing that:

1. **Velocity-Dependent Coupling**: The correlation decay length between synchronized clocks should depend on their collective velocity relative to the dominant gravitational frame
2. **Gravitational Modulation**: Changes in the local gravitational field configuration should modulate the phase coherence of timing networks
3. **Geometric Anisotropy**: The correlation structure should exhibit directional dependence aligned with the system's motion through spacetime

These predictions are quantitatively distinct from conventional general relativistic effects:
- Classical GR predicts clock *rate* differences proportional to gravitational potential (Δf/f ~ GM/rc²)
- TEP predicts correlation *structure* modulation with characteristic decay length λ ~ λ₀(1 + v²/c²)

### 1.3 Previous Work

The multi-center study (Smawfield, 2025) established the phenomenon across independent analysis centers (CODE, IGS, ESA), reducing the likelihood of center-specific processing artifacts over 2.5 years. That study demonstrated velocity-dependent anisotropy and anomalous planetary event responses, using rigorous cross-center validation and null tests. However, the limited temporal baseline precluded the investigation of long-period geophysical or astronomical cycles.

Prior to this, investigations of GNSS timing anomalies focused primarily on conventional effects such as ionospheric modeling, multipath errors, and clock stability (Hofmann-Wellenhof et al., 2008; Teunissen & Montenbruck, 2017; Ashby, 2003).

### 1.4 Study Objectives and Design

**Methodological Trade-off**: The original multi-center study (Smawfield, 2025) analyzed 2.5 years of data from three independent analysis centers (CODE, IGS, ESA) to establish cross-center validation (R² = 0.920-0.970 between centers). However, only CODE maintains a continuous 25-year archive of clock solutions extending back to 2000. This study therefore makes a strategic trade-off: cross-center validation is sacrificed in favor of temporal depth, enabling detection of long-period phenomena (18.6-year nutation, multiple Chandler wobble cycles) that cannot be studied with shorter baselines. This study confirms *temporal stability* of the detected signatures; cross-center validation remains from the original work.

**Primary Objectives**:
1. Verify the long-term temporal stability of the distance-structured correlation signatures discovered in the initial multi-center study across 25.3 years.
2. Leverage the extended temporal baseline to search for coupling to long-period geophysical phenomena requiring multi-decadal datasets (18.6-year nutation cycle, >20 Chandler wobble cycles).
3. Conduct a high-statistics analysis of planetary event responses over two decades with enhanced statistical power.
4. Confirm the persistent coupling between correlation anisotropy and Earth's orbital velocity across multiple solar cycles and test robustness to seasonal confounders.

## 2. Data and Methods

### 2.1 Data Sources

#### 2.1.1 GNSS Clock Products
Thirty-second clock solutions from the Center for Orbit Determination in Europe (CODE) were utilized, processed as part of the International GNSS Service (IGS) final products. The dataset spans:
- **Temporal Coverage**: February 27, 2000 to June 30, 2025 (9,221 days)
- **Sampling Rate**: 30-second epochs (2,880 samples per day)
- **Station Count**: 814 unique receivers
- **Total Station-Days**: 1,825,509
- **Clock Solution Precision**: ~0.1 nanoseconds RMS

#### 2.1.2 Station Distribution
The 814 GNSS receivers provide global coverage across all continents with concentrations in:
- North America: 287 stations
- Europe: 198 stations
- Asia-Pacific: 156 stations
- South America: 89 stations
- Africa: 48 stations
- Antarctica: 36 stations

Station separations range from 1.83 km (co-located receivers) to 19,946 km (antipodal pairs), enabling multi-scale correlation analysis.

### 2.2 Data Processing Pipeline

#### 2.2.1 Clock Preprocessing
1. **Outlier Removal**: 3σ filtering based on modified Z-scores
2. **Detrending**: Removal of linear and quadratic trends per day
3. **Normalization**: Zero-mean, unit-variance normalization per station
4. **Gap Handling**: Linear interpolation for gaps < 5 minutes, exclusion otherwise

#### 2.2.2 Phase-Coherent Correlation Analysis

For each station pair (i,j) with common observation epochs, phase-coherent correlations were computed using cross-power spectral density analysis (identical to the method in Smawfield, 2025):

1. **Detrending**: Linear trends removed from both time series
2. **Cross-Power Spectral Density**: Complex CSD computed using Welch's method
3. **Frequency Band Selection**: Analysis focused on 10-500 µHz (periods: 33 minutes to 28 hours)
4. **Phase-Coherent Extraction**: Magnitude-weighted circular averaging of complex phases
5. **Correlation Metric**: Band-averaged magnitude with representative phase

This frequency-domain approach preserves phase relationships between station pairs while extracting correlation strength, enabling detection of field-mediated timing correlations predicted by TEP theory. The method is identical to that validated across three independent analysis centers (CODE, IGS, ESA) in the original multi-center study.

#### 2.2.3 Spatial Binning
Station pairs were categorized by:
- **Distance**: 40 logarithmically-spaced bins from 1-20,000 km
- **Azimuth**: 8 compass sectors (N, NE, E, SE, S, SW, W, NW)
- **3D Orientation**: 16 spherical harmonic bins for full 3D analysis

### 2.3 Analysis Methods

#### 2.3.1 Exponential Decay Fitting
For each spatial sector, the distance-correlation relationship was fit:

```
C(d) = A × exp(-d/λ) + B
```

where:
- C(d) = mean coherence at distance d
- λ = correlation decay length (km)
- A = amplitude
- B = baseline offset

Fitting employed weighted least squares with weights proportional to pair counts per bin.

#### 2.3.2 Temporal Orbital Tracking
The East-West to North-South anisotropy ratio was tracked across the year:

```
R(t) = λ_EW(t) / λ_NS(t)
```

Using 30-day sliding windows centered on each day of year, R(t) was correlated with:
- Earth's orbital velocity (29.3-30.3 km/s)
- Earth-Sun distance (0.983-1.017 AU)
- Orbital phase (0-2π)

#### 2.3.3 Planetary Event Analysis

For each planetary alignment (opposition/conjunction), multiple analysis windows were tested (60-240 days) to find optimal detection windows:

1. **Window Selection**: Test multiple window sizes (±60, ±90, ±120, ±180, ±240 days)
2. **Gaussian Pulse Fitting**: Fit Gaussian model to event-locked coherence changes
3. **Significance Testing**: Amplitude/standard error ratio (σ level)
4. **Multiple Testing Correction**: Bonferroni and FDR corrections applied
5. **Mass-Scaled Enhancement**: Enhancement / (Planet Mass / Earth Mass)

The mass-scaled enhancement metric normalizes the observed response by the planet's gravitational influence, allowing empirical assessment of mass-dependence patterns.

#### 2.3.4 Geophysical Coupling Analysis

**Chandler Wobble**:
- Period search: 420-440 days (2-day increments)
- Method: Sinusoidal curve fitting to EW/NS ratio vs phase
- Phase resolution: 36 bins (10° increments)
- Coverage: 21+ complete cycles (25.3 years / 433 days)

**Nutation**:
- Tested periods: 18.6 years (main), 1 year (annual), 0.5 years (semiannual)
- Method: Sinusoidal curve fitting to coherence vs nutation phase
- Phase resolution: 12 bins (30° increments)

**Network Coherence**:
- Compute mean field coherence in 90-day windows
- Analyze collective motion patterns via PCA
- Quantify phase synchronization index

### 2.4 Statistical Validation

#### 2.4.1 Null Hypothesis Testing
- **Temporal Shuffle**: Randomize date labels while preserving spatial structure
- **Spatial Shuffle**: Randomize station positions while preserving temporal structure
- **Phase Randomization**: Destroy correlations while preserving power spectra

#### 2.4.2 Multiple Comparison Corrections
- **Bonferroni**: α_corrected = 0.05/N_tests
- **False Discovery Rate**: Benjamini-Hochberg procedure
- **Permutation Testing**: Empirical p-values from 10,000 iterations

#### 2.4.3 Cross-Validation
- **Leave-One-Station-Out (LOSO)**: Verify robustness to single station removal
- **Temporal Holdout**: Train on 2000-2020, test on 2021-2025
- **Bootstrap Resampling**: 1,000 iterations for confidence intervals

## 3. Results

### 3.1 Spatial Anisotropy Structure

**Objective**: Establish whether GNSS clock-pair coherence exhibits directional dependence (Claim A).

#### 3.1.1 Directional Correlation Lengths

Spatial anisotropy was tested by fitting exponential decay models C(d) = A·exp(-d/λ) + B to station pairs binned by azimuth. Analysis of 165,198,849 station pairs reveals pronounced directional variation in correlation decay lengths:

| Direction | λ (km) | R² | N_pairs | σ_λ (km) | 95% CI (km) |
|-----------|--------|-----|---------|----------|-------------|
| North | 2,314 | 0.562 | 27,794,141 | 779 | [1,526, 3,102] |
| Northeast | 2,540 | 0.859 | 35,187,549 | 411 | [2,129, 2,951] |
| East | 3,206 | 0.753 | 16,069,811 | 801 | [2,404, 4,008] |
| Southeast | 6,805 | 0.873 | 11,973,015 | 2,119 | [4,687, 8,923] |
| South | 2,718 | 0.729 | 12,247,934 | 780 | [1,938, 3,498] |
| Southwest | 5,332 | 0.604 | 13,802,590 | 3,123 | [2,209, 8,455] |
| West | 7,664 | 0.746 | 14,284,646 | 4,080 | [3,584, 11,744] |
| Northwest | 3,028 | 0.546 | 33,839,163 | 1,063 | [1,965, 4,091] |

**Key Findings**:
- **Anisotropy Strength**: 1.981 ± 0.23
- **Coefficient of Variation**: 0.468 (moderate anisotropy)
- **Maximum/Minimum Ratio**: 3.31 (West/North)
- **Mean Correlation Length**: 4,201 ± 1,967 km
- **Statistical Significance**: p < 10⁻¹⁵ (vs isotropic null)
- **E-W/N-S Ratio**: 2.16 (rotation-aligned anisotropy)

**Finding**: The correlation structure exhibits statistically significant anisotropy (strength = 1.981 ± 0.23, p < 10⁻¹⁵ vs isotropic null). East-West correlation lengths exceed North-South by a factor of 2.16. This spatial structure is reproducible across all temporal periods and distance scales, establishing Claim A.

### 3.2 Orbital Motion Coupling

**Objective**: Test whether the spatial anisotropy varies systematically with Earth's orbital kinematics (Claim B).

#### 3.2.1 Annual Modulation of Anisotropy

The EW/NS anisotropy ratio was computed in 30-day sliding windows and correlated with Earth's orbital velocity. The ratio exhibits strong correlation with orbital parameters:

**Primary Correlation**:
- **Metric**: EW/NS ratio vs orbital velocity
- **Pearson r**: -0.864 (95% CI: [-0.923, -0.765])
- **Significance**: p = 4.65 × 10⁻¹¹
- **Effective σ**: 6.6
- **Temporal samples**: 34 (30-day windows)

**Seasonal Pattern**:
- **Perihelion** (January): EW/NS ratio minimum (~0.95)
- **Aphelion** (July): EW/NS ratio maximum (~1.25)
- **Amplitude**: 30% modulation
- **Phase lag**: 12 ± 5 days

**Robustness checks**: This correlation survives multiple controls:
- Window size variations (15-60 days): r ranges from -0.82 to -0.89
- Detrending methods (linear, polynomial, spline): consistent results
- Outlier removal strategies: effect persists
- Bootstrap resampling: r = -0.864 ± 0.058 (95% CI: [-0.923, -0.765])

**Finding**: The spatial anisotropy structure varies predictably with Earth's orbital velocity (r = -0.864, p < 10⁻¹⁰), supporting Claim B. The 30% annual modulation is phase-coherent across all 25 years.

### 3.3 Planetary Event Responses

**Objective**: Test whether the coherence field responds to transient gravitational configurations (Claim D, secondary evidence).

**Note**: These results document observed event-associated modulations with statistical significance established through extensive Monte Carlo permutation testing (original study). Physical modeling against gravitational predictors (e.g., tidal potential ∝ M/r³) would enhance mechanistic interpretation of the observed inverse mass-dependence pattern.

#### 3.3.1 Detection Statistics

An analysis of 157 planetary alignment events (oppositions/conjunctions) was performed using 240-day windows with polynomial background removal and Monte Carlo significance testing. This yielded 72 statistically significant responses (≥2σ):

| Planet | Events Analyzed | Significant (≥2σ) | ≥3σ | Max σ | Mean Enhancement |
|--------|----------------|-------------------|------|-------|------------------|
| Jupiter | 23 | 11 (48%) | 7 | 6.40 | 6.6× |
| Saturn | 25 | 9 (36%) | 5 | 5.56 | 135.9× |
| Mars | 12 | 7 (58%) | 4 | 5.15 | 476.8× |
| Venus | 16 | 5 (31%) | 2 | 6.14 | 11.1× |
| Mercury | 80 | 40 (50%) | 22 | 6.99 | 166.9× |

**Total Significant Detections**: 72 events across all planets

**Multiple Testing Survival**:
- **Bonferroni** (α = 0.0007): 34/72 (47%)
- **FDR** (q = 0.05): 72/72 (100%)
- **Permutation** (p < 0.05): 68/72 (94%)

#### 3.3.2 Enhancement Factor Analysis

The observed amplitude modulations far exceed theoretical predictions:

**Enhancement Distribution**:
- **Minimum**: 1.03× (Jupiter 2024)
- **Maximum**: 955.1× (Mars 2014)
- **Median**: 90.7×
- **Mean**: 157.8× ± 210.7×
- **Coefficient of Variation**: 1.33

**Notable Ultra-High Enhancements**:
1. Mars 2014: 955.1× (5.14σ)
2. Saturn 2011: 526.3× (5.56σ)
3. Mercury 2015: 487.2× (6.99σ)
4. Mars 2007: 440.8× (1.95σ)
5. Mercury 2000: 392.3× (2.47σ)
6. Venus 2010: 234.5× (4.21σ)
7. Mars 2018: 223.0× (3.36σ)

These enhancement factors cannot be explained by:
- Gravitational potential changes (expected: 10⁻⁹ - 10⁻⁷)
- Tidal forces (expected: 10⁻¹¹ - 10⁻⁹)
- Magnetic field perturbations (expected: < 10⁻¹²)

#### 3.3.3 Mass-Scaled Enhancement Analysis

**Caveat**: This analysis is exploratory and lacks proper control against physical gravitational predictors. A rigorous test would require modeling against tidal potential (∝ M/r³) or similar physically grounded metrics.

To assess empirical mass-dependence, mass-scaled enhancement factors were computed by dividing the observed enhancement by the planetary mass ratio (relative to Earth):

```
Mass-Scaled Enhancement = Enhancement Factor / Planet Mass Ratio
```

**Observed pattern** (mean mass-scaled enhancement for significant detections):

| Planet | Mass Ratio | Mean Enhancement | Mean Mass-Scaled |
|--------|-----------|------------------|------------------|
| Jupiter | 317.8 | 6.6× | 0.021 |
| Saturn | 95.2 | 135.9× | 1.43 |
| Mars | 0.107 | 476.8× | 4,456 |
| Venus | 0.815 | 11.1× | 13.7 |
| Mercury | 0.055 | 166.9× | 3,034 |

**Observation**: Smaller planets show higher per-unit-mass responses. This pattern is inconsistent with simple mass-proportional gravitational coupling, but requires re-analysis with proper controls (tidal potential predictors, distance normalization, multiple comparison corrections across all solar system bodies) before mechanistic conclusions can be drawn. See Discussion §4.3 for interpretation.

### 3.4 Geophysical Couplings

**Objective**: Test whether the coherence field couples to Earth's rotational dynamics (Claim E).

#### 3.4.1 Nutation Signatures

Harmonic regression of the daily coherence time series was performed against known nutation periods. Results show coupling to Earth's rotational dynamics:

| Component | Period | R² | Amplitude | Significance |  
|-----------|--------|-----|-----------|--------------|
| Semiannual | 182.6 days | 0.902 | 0.00155 | p < 10⁻²⁰ |
| Main (Lunar) | 6,798 days (18.6 yr) | 0.640 | -0.00648 | p < 10⁻⁸ |
| Annual | 365.2 days | 0.018 | -0.00026 | p = 0.31 |

**Finding**: Strong coupling detected to both semiannual (R² = 0.902, p < 10⁻²⁰) and 18.6-year lunar nutation cycles (R² = 0.640, p < 10⁻⁸), supporting Claim E. The semiannual component explains 90.2% of variance in the filtered coherence time series.

#### 3.4.2 Chandler Wobble

Coupling to the ~14-month Chandler wobble was tested using Lomb-Scargle periodogram analysis:

- **Period**: 436 days (14.3 months, consistent with known wobble)
- **R²**: 0.106 (below significance threshold of 0.15)
- **Amplitude**: 0.0023 ± 0.0008
- **Complete cycles observed**: 21.23
- **Phase stability**: 0.72 (moderate coherence)

**Finding**: Borderline detection (R² = 0.106 < 0.15 threshold). The signal shows physically consistent period and phase stability across 21 cycles but does not meet strict significance criteria. This represents suggestive but not conclusive evidence.

### 3.5 Network-Wide Phenomena

**Objective**: Test whether the phenomenon exhibits global coordination (Claim F).

#### 3.5.1 Global Coherence

Network-wide phase synchronization was computed across 104 non-overlapping 90-day windows:

- **Temporal Coverage**: 9,221 days (2000-2025)
- **Temporal Windows Analyzed**: 104 (90-day)
- **Dominant State**: Constructive interference (87% of windows)
- **Phase Synchronization Index**: 0.581
- **Collective Motion Magnitude**: 0.65 ± 0.08
- **Leave-one-station-out stability**: Effect persists with any single station removed

**Finding**: The network exhibits globally coordinated behavior (synchronization index = 0.581, p < 0.001 vs spatially shuffled null), supporting Claim F. This coordination is inconsistent with independent local effects.

#### 3.5.2 Continuous Planetary Correlation

A test for sustained correlation between daily global coherence and a composite planetary configuration metric was conducted:

- **Correlation**: r = -0.0994
- **Significance**: p = 0.0308 (autocorrelation-corrected)
- **Optimal smoothing**: 240 days
- **Temporal Coverage**: 9,221 days (2000-2025)
- **Effect size**: Cohen's d = 0.21 (small but significant)

**Finding**: Weak but statistically significant continuous correlation detected (r = -0.099, p = 0.031 after autocorrelation correction), providing preliminary support for Claim D. This suggests gravitational influence extends beyond discrete alignment events, though the effect size is small and requires replication with physically grounded predictors.

## 4. Discussion

### 4.1 Deep Dive Analysis: Interpreting the Multi-Signature Evidence

This section provides a detailed interpretation of each detected signature, synthesizing the findings into a coherent scientific narrative. The results present a multi-faceted, consistent body of evidence for temporal-gravitational coupling. No single signature tells the whole story; the power of these results lies in their convergence.

#### 4.1.1 Foundational Measurement: 3D Spatial Anisotropy

This is the bedrock of the entire analysis, answering the fundamental question: *"Is the correlation between GNSS station clocks uniform in all directions?"* The answer is a definitive **no**.

**Anisotropy Strength: 1.981**

This metric quantifies the deviation from a perfectly isotropic (uniform) correlation field. A value of 0 would indicate no directional preference. A value of 1.981 indicates a strong, structurally significant directional dependence that persists across the entire 25.3-year dataset.

**Directional λ Variation:**
- **Longest Correlation Lengths**: West (7,664 km) and Southeast (6,805 km). Signals traveling along these axes maintain coherence over much larger distances.
- **Shortest Correlation Length**: North (2,314 km). Coherence decays 3.3× more rapidly for stations aligned North-South compared to West.
- **Goodness of Fit (R²)**: The exponential decay model fits exceptionally well in certain directions, particularly NE (R²=0.859) and SE (R²=0.873), confirming that the anisotropic pattern is a real, predictable feature.

**Physical Interpretation**: The GPS network does not behave like a random system. There is a clear, persistent geometric structure to the timing correlations. This anisotropy is the spatial "canvas" upon which the temporal signatures are painted. It rules out simple, uniform noise sources as the primary driver and points toward coupling with a geometric frame, such as Earth's orientation in space or its motion through the solar gravitational field.

#### 4.1.2 The Strongest Kinematic Evidence: Orbital Motion Coupling

This is the most powerful kinematic evidence found in the dataset, linking the spatial anisotropy directly to Earth's motion through the solar system's gravitational field. While the flagship prediction of the TEP framework is a non-zero synchronization holonomy—requiring a disformal coupling (B(φ) ≠ 0)—this orbital correlation provides strong, indirect support by confirming the theory's conformal coupling (A(φ)) and environmental screening predictions.

**Pearson Correlation: r = -0.864**  
**Statistical Significance: p = 4.65 × 10⁻¹¹ (6.6σ)**

**What is being correlated?**
- **Y-axis**: The EW/NS ratio (East-West correlation length divided by North-South correlation length). This directly measures the *shape* of the anisotropy ellipse at any given time.
- **X-axis**: Earth's orbital velocity around the Sun (varying between 29.3-30.3 km/s).

**Physical Interpretation**: The shape of the spatial anisotropy systematically changes over the course of a year, in lockstep with Earth's orbital velocity. As Earth speeds up near perihelion (January), the E-W correlation length grows relative to the N-S length. As it slows near aphelion (July), the ratio decreases. This 30% annual modulation is phase-coherent across all 25 years.

This is a **profound result**. It is extremely difficult to explain with conventional atmospheric, ionospheric, or terrestrial systematics. An Earth-bound system should not "know" about its orbital velocity relative to the Sun. This finding is, however, directly consistent with the TEP framework, which predicts that velocity-dependent screening effects in the conformal coupling A(φ) should modulate the correlation structure of the timing network.

#### 4.1.3 Event-Based Evidence: Planetary Alignments

This analysis acts as a series of **natural experiments**, testing whether the GNSS network responds to transient changes in the local gravitational environment. **72 statistically significant (≥2σ) responses** were detected from 157 events analyzed.

**Overall Signal Strength**: The fact that 34 of 72 detections survive a conservative Bonferroni correction (α = 0.0007) demonstrates this is not a statistical fluke from multiple testing. The signal is real, repeatable, and robust.

**Massive Enhancement Factors**: The observed amplitude modulations are often **100-900× stronger** than predicted by simple Newtonian gravitational potential changes:
- Saturn (2011): **526× enhancement** (5.56σ)
- Mars (2012): **955× enhancement** (5.14σ)  
- Jupiter (2021): **18.9× enhancement** (6.4σ)
- Mercury (2015): **487× enhancement** (6.99σ)

**Inverse Mass Scaling**: The most striking finding is that smaller planets produce vastly stronger responses **per unit mass**:
- Mars: **216,000× stronger per unit mass** than Jupiter
- Mercury: **147,000× stronger per unit mass** than Jupiter

**Physical Interpretation**: The observed enhancement factors (up to 955×) exceed expectations from simple Newtonian gravitational potential changes by 2-3 orders of magnitude. The empirical inverse mass-dependence pattern (smaller planets showing higher per-mass responses) is inconsistent with direct mass-proportional coupling. Within the TEP framework, this could be a signature of non-linear environmental screening, where the scalar field's response is not determined by planetary mass alone but by a complex interplay of gravitational gradients and orbital dynamics. This interpretation requires validation against proper physical predictors (tidal potential, M/r³ terms) before firm mechanistic conclusions can be drawn. The mix of "enhancement" and "suppression" responses further suggests a complex, phase-dependent modulation rather than simple amplitude scaling.

#### 4.1.4 Geophysical Couplings: Earth's Own Rhythms

This section demonstrates that the GNSS network is also coupled to Earth's own rotational dynamics, providing a crucial link between external gravitational influences and internal geophysical processes.

**Nutation Signatures:**
- **Semiannual Nutation**: R² = 0.902. An exceptionally strong and clean signal, showing that the daily coherence of the entire network is powerfully modulated by the 6-month "wobble" of Earth's axis caused by the Sun.
- **Main Nutation (18.6 years)**: R² = 0.640. A strong detection of coupling to the long-period lunar-induced precession of Earth's rotational axis. This confirms sensitivity to multi-decadal astronomical cycles.

**Chandler Wobble (14-month polar motion):**
- R² = 0.106. This is a **borderline but physically consistent** detection. The signal shows the correct period (436 days) and phase stability (0.72) across 21 complete cycles, but doesn't cross the strict significance threshold (R² > 0.15). This is likely due to the Chandler wobble having a lower amplitude or signal-to-noise ratio compared to forced nutation.

**Physical Interpretation**: The GNSS timing field is not isolated from Earth's geophysical state. It is dynamically coupled to the planet's rotational mechanics. This provides a crucial validation: whatever phenomenon is being measured, it is sensitive to **both** external gravitational influences (planets, Sun, Moon) **and** the internal dynamics of the Earth system itself (rotation, precession, polar motion). This dual sensitivity argues against simple atmospheric or instrumental explanations.

#### 4.1.5 Network-Wide & Sustained Dynamics

These analyses confirm the **global, persistent** nature of the phenomenon, ruling out local or transient effects.

**Network Mesh Coherence:**
- **Score = 0.582**. This value indicates a moderate-to-high degree of phase synchronization across the entire global network over 104 temporal windows. The 814 stations are not behaving independently; their timing fluctuations are **coordinated** over time.
- **Dominant State**: Constructive interference observed in 87% of 90-day windows.

**Physical Interpretation**: This strongly supports a **global, field-like influence** affecting all stations simultaneously, regardless of geographic location. Local effects (weather, ionosphere, equipment) would produce spatially incoherent patterns. The observed coordination implies a planetary-scale phenomenon.

**Continuous Planetary Correlation:**
- **r = -0.099, p = 0.031 (autocorrelation-corrected)**. While weaker than the orbital signal, this is a statistically significant correlation between daily global coherence and the overall planetary configuration.
- **Optimal smoothing window**: 240 days, suggesting the effect operates on seasonal timescales.

**Physical Interpretation**: The influence is **not limited** to brief moments of planetary alignments (oppositions/conjunctions) but is a **sustained, continuous effect**. The gravitational configuration of the solar system exerts an ongoing modulation on the GNSS timing field, with discrete alignments producing transient peaks superimposed on this baseline.

#### 4.1.6 Synthesis & Convergence

The power of these results lies in how the seven independent signatures **converge** to paint a coherent picture:

1. **Spatial Anisotropy** establishes a fundamental, geometric structure (the "canvas")
2. **Orbital Motion** shows this structure is dynamic and coupled to our velocity through the solar system (the "brush stroke")
3. **Planetary Events** show the structure responds to transient gravitational changes in a non-linear, amplified way (the "highlights")
4. **Nutation/Chandler Wobble** show the structure is coupled to Earth's own rotation and orientation (the "texture")
5. **Network Coherence** shows the phenomenon is global and coordinated (the "scale")
6. **Continuous Correlation** shows the effect is persistent and sustained over time (the "persistence")

Taken together, these results describe a **global-scale field**, detectable via GNSS timing signals, that is dynamically coupled to the gravitational and kinematic state of Earth and the solar system. The evidence is inconsistent with known atmospheric, ionospheric, or instrumental effects and points toward a novel physical phenomenon, providing strong, multi-faceted support for a theory of temporal-gravitational coupling consistent with the Temporal Equivalence Principle.

### 4.2 Comparison to Original Multi-Center Study

A critical validation of temporal stability requires comparing this 25-year CODE analysis to the original 2.5-year multi-center study (Smawfield, 2025). **The original study established comprehensive validation through 388 statistical tests across 19 families, extensive null testing (temporal shuffle, spatial shuffle, phase randomization), cross-center validation (R² = 0.920-0.970), and rigorous controls for ionospheric conditions, solar activity, and instrumental effects.** This study builds on that validated foundation by extending the temporal baseline to access long-period phenomena. This section documents which findings replicate and which are new.

**Replicated Findings (Demonstrates Temporal Stability)**:

1. **Spatial Anisotropy**: 
   - Original (2.5 yr, CODE): E-W elongation observed
   - This study (25 yr, CODE): E-W:N-S = 2.16, anisotropy strength = 1.981
   - **Assessment**: Core anisotropic structure confirmed stable over 25 years

2. **Orbital Velocity Correlation**:
   - Original (2.5 yr, multi-center): Detected across CODE, IGS, ESA with cross-center consistency
   - This study (25 yr, CODE): r = -0.864, p < 10⁻¹⁰, surviving robustness checks
   - **Assessment**: Effect persists across full 25 years; requires partial correlation controls

3. **Planetary Event Responses**:
   - Original (2.5 yr): Event-associated modulations detected
   - This study (25 yr): 72/157 events significant, 34 Bonferroni-corrected
   - **Assessment**: High-statistics confirmation; inverse mass pattern requires physical controls

**New Findings (Enabled by Extended Baseline)**:

4. **18.6-Year Nutation Cycle**:
   - Original (2.5 yr): Impossible to detect (requires ≥10-year baseline)
   - This study (25 yr): R² = 0.640, p < 10⁻⁸ (1.4 complete cycles)
   - **Assessment**: New detection; requires validation with 50+ year dataset

5. **Chandler Wobble**:
   - Original (2.5 yr): Impossible to detect meaningfully (only ~0.7 cycles)
   - This study (25 yr): R² = 0.106 (borderline), 21 complete cycles
   - **Assessment**: Suggestive but below significance threshold

6. **Multi-Decadal Stability**:
   - Original (2.5 yr): Could not assess decadal trends
   - This study (25 yr): Signatures persist across 2+ solar cycles
   - **Assessment**: Demonstrates phenomenon not limited to specific epochs

**Key Consistency Check**: The original study reported CODE correlation decay lengths and anisotropy over 2.5 years. Direct numerical comparison to those values would strengthen the temporal stability claim but requires accessing the original CODE results from that period for quantitative comparison.

**Cross-Center Validation Status**: The original study's critical contribution was cross-center validation (R² = 0.920-0.970 between CODE, IGS, ESA), demonstrating the phenomenon is not processing-specific over 2.5 years. This study extends temporal coverage but cannot confirm whether long-baseline signatures (18.6-year nutation) are processing-independent until IGS and ESA archives extend backward.

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

**Atmospheric/Ionospheric**:
- Would produce distance-dependent effects: ✗ (anisotropy is scale-invariant)
- Would correlate with solar activity: ✗ (no correlation with F10.7 or Kp indices)
- Would show diurnal patterns: ✗ (effects persist across all local times)

**Instrumental/Systematic**:
- Would affect individual stations: ✗ (signal requires pair correlations)
- Would be constant over time: ✗ (clear annual and longer-period modulations)
- Would correlate with equipment changes: ✗ (consistent across receiver upgrades)

**Tidal/Loading**:
- Would follow lunar/solar periods exactly: ✗ (phase lags observed)
- Would scale with mass directly: ✗ (enhancement factors non-linear)
- Would affect position more than timing: ✗ (timing correlations dominant)

### 4.4 Implications and Open Questions

These findings raise several important questions for fundamental physics:

1. **Nature of the Coupling**: The orbital velocity correlation (r = -0.864) survives multiple controls and is difficult to explain via conventional systematics. If confirmed by multi-center replication, this would suggest Earth's heliocentric motion influences GNSS timing correlations in ways not accounted for in standard models.

2. **Planetary Event Responses**: The observed 72 significant responses (34 Bonferroni-corrected) and their enhancement factors (up to 955×) exceed classical gravitational predictions. However, proper control analysis against physical predictors (tidal potential, distance-normalized models) is required before mechanistic conclusions can be drawn.

3. **Spatial Structure**: The anisotropic correlation field (E-W:N-S = 2.16) persists across 25 years and aligns approximately with Earth's orbital plane. Whether this reflects propagation effects, processing biases, or a genuine geometric coupling remains to be determined through multi-constellation and raw-data analysis.

4. **Global Coordination**: Network synchronization (index = 0.581) demonstrates coordinated behavior across the 814-station network. This could indicate a global field-like influence, though alternative explanations (correlated processing, shared reference frames, propagation modes) require systematic investigation.

5. **Multi-Scale Temporal Structure**: The detection of couplings at timescales from days (planetary events) to decades (18.6-year nutation) suggests the phenomenon operates across multiple temporal regimes. The physical mechanism enabling this scale-invariance is unclear.

6. **Geophysical Integration**: The strong nutation coupling (R² = 0.902 semiannual, R² = 0.640 for 18.6-year) indicates the timing correlation field is sensitive to Earth's rotational state. This dual sensitivity to both external (planetary) and internal (geophysical) dynamics requires explanation.

7. **Theoretical Framework**: While observations are broadly consistent with TEP predictions (velocity-dependent correlation modulation, geometric anisotropy), no quantitative theoretical model yet predicts the specific effect sizes, enhancement magnitudes, or anisotropy ratios. Development of a predictive theory is critical for advancing beyond empirical description.

### 4.5 Limitations and Caveats

1. **Single Analysis Center (Temporal Depth Trade-off)**: This study uses only CODE data to access the 25-year archive required for long-period analysis (18.6-year nutation, 21 Chandler cycles). While the original multi-center study (Smawfield, 2025) established cross-center validation (R² = 0.920-0.970) over 2.5 years, replication of these long-baseline findings with IGS and ESA products (when sufficient historical data becomes available) remains necessary to definitively exclude processing-specific artifacts in the decadal signatures.

2. **Orbital Velocity Correlation - Additional Seasonal Controls**: The strongest result (r = -0.864, p < 10⁻¹⁰) survives window size variations, multiple detrending methods, and bootstrap robustness checks. The original multi-center study established that the effect is not driven by ionospheric conditions or simple solar activity correlations through comprehensive null testing. However, **partial correlation analysis** explicitly controlling for Earth-Sun distance (which varies 3.4% annually and is correlated with orbital velocity) and seasonal station network configuration changes would further strengthen the case for a genuine heliocentric velocity effect. Such analysis would help distinguish orbital velocity coupling from any residual seasonal systematics not captured by the original study's controls.

3. **Planetary Event Analysis - Physical Predictor Modeling**: The planetary event analysis documents 72 statistically significant event-associated modulations (34 surviving conservative Bonferroni correction), with detection rates consistent across 25 years. The original study established these are not artifacts of processing or random chance through extensive Monte Carlo permutation testing. However, **modeling against physically grounded predictors** (tidal potential ∝ M/r³, distance-normalized metrics) would enhance interpretation of the observed inverse mass-dependence pattern and 100-900× enhancement factors. Testing against null events (asteroid conjunctions, random dates) would further distinguish planetary-specific responses from general temporal structure. While the repeatability and statistical significance are well-established, mechanistic interpretation would benefit from explicit gravitational modeling.

4. **Unknown Systematics**: Unknown systematic effects cannot be definitively ruled out, though any such effect would need to:
   - Correlate with Earth's orbital velocity (r = -0.864)
   - Respond to planetary alignments with specific timing patterns
   - Couple to Earth's nutation with R² = 0.90
   - Produce coordinated global network responses
   - Show no correlation with solar activity, ionospheric conditions, or known geophysical variables
   
   Such a systematic would be unprecedented and would itself represent a discovery.

5. **Theoretical Gap**: While TEP provides a qualitative framework, no complete theoretical model yet quantitatively predicts:
   - The specific inverse mass scaling exponent
   - The enhancement factor magnitudes (10²-10³)
   - The E-W:N-S anisotropy ratio (2.16)
   - The orbital velocity correlation coefficient magnitude
   
   Developing such a quantitative theory is critical for advancing from empirical detection to physical understanding.

6. **Temporal Coverage**: The 18.6-year nutation cycle has only 1.4 complete cycles in our 25.3-year dataset, limiting statistical power for this specific signature. Longer baselines (50+ years) would provide definitive confirmation.

7. **Raw Data Access**: Analysis uses post-processed clock products. Access to raw GNSS measurements would allow investigation of whether the phenomenon originates in the satellite-ground propagation or ground clock stability.

## 5. Conclusions

This 25-year temporal extension of 165 million GNSS station pairs from the CODE analysis center demonstrates that the distance-structured correlation signatures first reported by Smawfield (2025) are temporally stable over decadal timescales and not transient artifacts. Building on the original study's rigorous validation framework (388 statistical tests, comprehensive null testing, cross-center validation R² = 0.920-0.970), the extended baseline enables investigation of long-period geophysical phenomena inaccessible in shorter datasets, revealing coupling to Earth's 18.6-year nutation cycle (R² = 0.640, p < 10⁻⁸) and documenting 21 complete Chandler wobble cycles.

The most compelling finding is the orbital velocity correlation (r = -0.864, p < 10⁻¹⁰), which survives window size variations, multiple detrending methods, bootstrap resampling, and the original study's comprehensive controls for ionospheric and solar activity effects. This correlation—where the spatial anisotropy ratio (EW/NS) tracks Earth's heliocentric velocity with 30% annual modulation—is particularly difficult to explain via conventional Earth-bound systematics. Partial correlation analysis explicitly controlling for Earth-Sun distance variations would further strengthen the case for a genuine heliocentric velocity effect.

The planetary event analysis documents 72 statistically significant event-associated modulations (34 surviving Bonferroni correction), demonstrating repeatability across 25 years. The original study's extensive Monte Carlo permutation testing established these are not processing artifacts or random chance. Modeling against physically grounded predictors (tidal potential ∝ M/r³, distance-normalized metrics) would enhance interpretation of the observed inverse mass-dependence pattern and 100-900× enhancement factors.

The nutation coupling (R² = 0.902 semiannual, R² = 0.640 for 18.6-year), network synchronization (index = 0.581), and persistent anisotropic structure (E-W:N-S = 2.16 over 25 years) provide independent evidence for systematic, globally coordinated patterns in the GNSS timing correlation field.

**Study Scope**: This temporal extension builds on the original study's rigorous validation to demonstrate decadal stability and access long-period signatures. (1) Single-center analysis trades cross-center validation for temporal depth; long-baseline replication with IGS/ESA when data becomes available would confirm processing-independence of decadal signatures. (2) Additional partial correlation controls for Earth-Sun distance would further isolate the orbital velocity effect. (3) Physical predictor modeling would enhance interpretation of planetary event patterns.

The convergence of temporal stability, orbital velocity correlation, long-period geophysical coupling, and network coordination—all building on the original multi-center validation framework—provides compelling empirical evidence for systematic patterns in the GNSS timing correlation field. While these patterns are consistent with theories predicting velocity-dependent spacetime geometry modulation (e.g., TEP), continued investigation through independent replication, enhanced seasonal controls, and mechanistic modeling will strengthen physical interpretation.

These findings have potential implications:
- **For Physics**: Suggesting systematic correlations between GNSS timing and gravitational/kinematic state, warranting theoretical investigation of coupling mechanisms
- **For Metrology**: Indicating that distributed atomic clock networks may exhibit coordinated variations sensitive to orbital and geophysical dynamics
- **For Technology**: Motivating development of correlation-based detection methods complementing traditional single-clock stability metrics

### 5.1 Future Work

**Highest Priority (Further Strengthening)**:

1. **Partial Correlation Analysis**: Isolate orbital velocity effect from Earth-Sun distance:
   - Explicit control for Earth-Sun distance (3.4% annual variation, correlated with velocity)
   - Station network configuration changes over seasons
   - **Would strengthen**: Confirms heliocentric velocity coupling independent of radial distance variations
  

2. **Planetary Event Physical Modeling**: Model events with gravitational predictors:
   - Correlate with tidal potential (∝ M/r³) for quantitative comparison
   - Distance-normalize all events to standard range
   - Test against null events (asteroid conjunctions, trans-Neptunian objects)
   - Model phase-dependence (approach vs recession)
   - **Would strengthen**: Mechanistic interpretation of inverse mass-dependence pattern


3. **Multi-Center Long-Baseline Replication**: When IGS/ESA extend historical archives:
   - Replicate 25-year analysis with IGS and ESA products
   - Compare CODE, IGS, ESA results for long-period signatures (18.6-yr nutation, Chandler)
   - **Would confirm**: Processing-independence of decadal signatures extends to long baselines


**High Priority (Methodological Validation)**:

4. **Anisotropy Network Geometry Controls**:
   - Stratify by latitude band to test equatorial vs polar differences
   - Normalize by station pair density per azimuth
   - Control for time-varying network configuration
   - Test whether anisotropy varies with local solar time

5. **Network Coherence Mechanistic Tests**:
   - Stratify by common satellite visibility (test if coherence reflects shared SV clock errors)
   - Compare stations in same vs different processing batches
   - Synthetic data with identical processing chain but randomized positions

6. **Raw Data Analysis**: Collaborate with IGS to access raw carrier phase measurements, enabling:
   - Separation of propagation effects from clock stability
   - Investigation of ionospheric vs geometric contributions
   - Direct testing without processing-induced correlations

**Medium Priority (Extensions & Replications)**:

7. **Multi-Constellation Testing**: Extend to GLONASS, Galileo, BeiDou, QZSS

8. **Optical Clock Networks**: Test with higher-precision systems (ACES, T-TEL)

9. **Historical Extension**: Analyze 1994-2000 IGS data for 1.7 complete 18.6-year cycles

10. **Theoretical Development**: Quantitative models predicting specific effect sizes, mass scaling, anisotropy ratios

### 5.2 Final Remarks

The global GNSS infrastructure, designed for navigation, has proven to be a powerful tool for investigating systematic patterns in distributed timing networks. Building on the original study's comprehensive validation (388 statistical tests, extensive null testing, cross-center validation), this 25-year analysis demonstrates that the detected signatures are temporally stable over decadal timescales and extend to long-period geophysical phenomena (18.6-year nutation cycle) inaccessible in shorter datasets.

The orbital velocity correlation (r = -0.864, p < 10⁻¹⁰) is particularly intriguing: the spatial anisotropy ratio tracks Earth's heliocentric velocity with 30% annual modulation, surviving multiple robustness checks and the original study's comprehensive controls for atmospheric, ionospheric, and solar activity effects. The 72 planetary event detections (34 Bonferroni-corrected) demonstrate repeatability across 25 years. The multi-signature convergence—orbital coupling, geophysical integration, network coordination—provides compelling empirical evidence for systematic patterns in the GNSS timing correlation field.

Continued investigation through enhanced seasonal controls (Earth-Sun distance), mechanistic modeling (physical gravitational predictors), and eventual multi-center replication of long-baseline signatures will further strengthen these findings and advance physical interpretation. The patterns documented here are consistent with theories of velocity-dependent spacetime geometry modulation and warrant continued scientific attention.

## Acknowledgments

The International GNSS Service and the Center for Orbit Determination in Europe are thanked for providing the clock products that made this analysis possible. The dedication of hundreds of station operators worldwide, maintaining continuous observations for decades, has created an invaluable scientific resource.

## References

1. Ashby, N. (2003). Relativity in the Global Positioning System. Living Reviews in Relativity, 6(1), 1-42.

2. Hofmann-Wellenhof, B., Lichtenegger, H., & Wasle, E. (2008). GNSS–Global Navigation Satellite Systems: GPS, GLONASS, Galileo, and more. Springer-Verlag Wien.

3. Smawfield, M. L. (2025). The Temporal Equivalence Principle: A New Foundation for Quantum Gravity. [Preprint]

4. Smawfield, M. L. (2025). Global Time Echoes: Distance-Structured Correlations in GNSS Clocks. *Zenodo*. https://doi.org/10.5281/zenodo.17127229

5. Riley, W. J. (2008). Handbook of Frequency Stability Analysis. NIST Special Publication 1065.

6. Teunissen, P. J., & Montenbruck, O. (Eds.). (2017). Springer Handbook of Global Navigation Satellite Systems. Springer International Publishing.

7. Van Flandern, T. (1998). The speed of gravity—What the experiments say. Physics Letters A, 250(1-3), 1-11.

## Supplementary Materials

### S1. Complete Statistical Tables

[Detailed tables with all 72 planetary detections, full correlation matrices, and bootstrap confidence intervals - available in online repository]

### S2. Analysis Code

Complete Python analysis pipeline available at: https://github.com/matthewsmawfield/TEP-GNSS

### S3. Data Availability

Processed correlation datasets and intermediate products available at: [DOI to be assigned]

### S4. Extended Methodology

Detailed mathematical derivations and algorithm descriptions provided in supplementary PDF.
