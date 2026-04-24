# Global Time Echoes: Distance-Structured Correlations in GNSS Clocks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17127229.svg)](https://doi.org/10.5281/zenodo.17127229)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![Global Time Echoes](site/public/og-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.24 (Jaipur)  
**Date:** 24 April 2026  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229)  
**Website:** [https://mlsmawfield.com/tep/gnss-i/](https://mlsmawfield.com/tep/gnss-i/)

## Abstract

Phase-coherent spectral analysis of 62.7 million station-pair measurements from 364 GNSS stations (2023–2025) reveals systematic distance-structured correlations in clock networks. These correlations follow an exponential decay with a median correlation length λ = 3,330–4,549 km (95% CIs: CODE 1,198–5,918 km; IGS 3,197–4,871 km; ESA 2,532–3,984 km) and show strong goodness-of-fit when evaluated on distance-binned means across three independent analysis centres (R² = 0.920–0.970; fits are to bin means, not raw pairs). Cross-center validation, consistent across 12 frequency bands and confirmed through multiple binning schemes and null hypothesis testing, demonstrates these patterns represent genuine physical correlations rather than systematic artifacts. The patterns also show dependencies on station elevation and geomagnetic latitude, consistent with theoretical frameworks involving screened scalar fields.

The correlations demonstrate systematic coupling with Earth's orbital motion (r = -0.571 to -0.793 across centers), planetary gravitational influences (6 Bonferroni-significant events), Chandler wobble modulation (R² = 0.377–0.471), and systematic diurnal temporal variations with synchronized early morning coherence peaks (Local Solar Time). Comprehensive validation demonstrates 24-61× signal enhancement over randomized controls (z = 15.8-31.9 across 180 null test iterations), with FDR-BH: 203/388 tests (52.3%), Hierarchical EB: 154/388 (39.7%), and Bonferroni: 155/388 (40.0%) surviving multiple-comparison correction across 19 independent validation families. TID exclusion analysis shows 21–23% signal improvement when excluding high-ionosphere periods—the ionosphere suppresses rather than creates the correlation.

The investigation was structured to test predictions from the Temporal Equivalence Principle (TEP) framework, which suggested a correlation length (λ) of 1,000–10,000 km. The full analysis yielded λ = 3,330–4,549 km, a result consistent with this expectation which motivated tests of derived predictions (diurnal, eclipse, and orbital signatures). While multi-center consistency and extensive validation provide a strong basis for these findings, alternative explanations involving sophisticated systematics cannot be fully excluded. Therefore, definitive physical interpretation awaits critical next steps: raw-data analysis, multi-constellation testing, and independent replication.

## Related Papers

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | **TEP-GNSS** (This repo) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Validation of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Validation of the Temporal Equivalence Principle | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales | [10.5281/zenodo.18064366](https://doi.org/10.5281/zenodo.18064366) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: A Runaway Black Hole as a Gravitational Soliton | [10.5281/zenodo.18059251](https://doi.org/10.5281/zenodo.18059251) |
| **Paper 8** | [TEP-SLR](https://github.com/matthewsmawfield/TEP-SLR) | Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging | [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109761](https://doi.org/10.5281/zenodo.18109761) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | The Temporal Equivalence Principle: Density-Dependent Screening in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102062](https://doi.org/10.5281/zenodo.19102062) |

## Theoretical Foundation

The Temporal Equivalence Principle (TEP) represents an extension of General Relativity, where proper time is treated as a dynamical scalar field. The theoretical framework employs a two-metric geometric structure where matter fields couple to an effective causal metric:

$$\tilde{g}_{\mu\nu} = A(\phi) g_{\mu\nu} + B(\phi) \nabla_\mu\phi \nabla_\nu\phi$$

with a universal conformal coupling $A(\phi) = \exp(2\beta\phi/M_{Pl})$.

**Core Prediction:** Precision timing networks exhibit distance-structured correlations following exponential decay:
$$C(r) = A\cdot\exp(-r/\lambda) + C_0$$
with characteristic correlation lengths $\lambda = 1,000-10,000$ km for screened scalar field configurations.

**Key Consequence:** Clock synchronization procedures exhibit non-integrable properties, yielding measurable synchronization holonomy in closed-loop time transfer protocols.

## Key Findings

Analysis of 62.7 million station-pair measurements across three independent centers (CODE, IGS, ESA) reveals exponential correlation decay with λ = 3,330–4,549 km (R² = 0.92–0.97). The signal couples to Earth's orbital velocity (r = −0.571 to −0.793), Chandler wobble (R² = 0.38–0.47), and planetary events (6 Bonferroni-significant). Validation demonstrates 24–61× signal enhancement over null tests. Cross-center consistency (CV = 12.9%) and raw RINEX confirmation exclude processing artifacts.

---

## Cloud-Based Analysis Execution (Recommended)

**For optimal computational performance and scientific output quality:**

### Cloud Deployment Options

#### GCP High-CPU Analysis (Recommended for Large-Scale)

Professional cloud deployment optimized for Google Cloud Platform high-CPU instances:

**Quick Start:**
```bash
# 1. Set your GCP instance details
export GCP_PROJECT_ID=your-project-id
export GCP_ZONE=us-central1-c  
export GCP_INSTANCE_NAME=your-instance-name

# 2. Deploy and run the complete pipeline
./run_tep_gcp_high_cpu.sh

# 3. Monitor progress (in another terminal)
gcloud compute ssh $GCP_INSTANCE_NAME --zone=$GCP_ZONE --command='cd /mnt/data && tail -f full_pipeline.log'

# 4. Download results when complete
./download_gcp_results.sh
```

**Recommended Instance Type:**
- `n2-highcpu-96`: 96 vCPUs, 96 GB RAM (Maximum performance - recommended)

**What the Pipeline Does:**
- **Automated Setup**: Installs all dependencies (Python packages, system libraries)
- **Complete Analysis**: Runs Steps 1-4 (Data acquisition → Core analysis → Validation → Advanced analysis)
- **Full Date Range**: Analyzes 912 days (2023-01-01 to 2025-06-30)
- **High Performance**: Optimized for 96 vCPUs with parallel processing
- **Comprehensive Output**: Generates 57+ JSON results + 20+ figures + 25+ logs
- **Background Execution**: Runs continuously with detailed logging
- **Easy Download**: Simple script to get all results locally

**Prerequisites:**
- [Google Cloud Platform](https://cloud.google.com/) account with billing enabled
- `gcloud` CLI installed and authenticated
- High-CPU instance created and running

#### Local Pipeline Execution

For development and targeted analysis:

**Complete Pipeline Scripts:**
```bash
# Full pipeline execution (Steps 1.0-4.8)
python scripts/clean_run_full_pipeline.py

# Data acquisition and validation (Steps 1.0-1.2)
python scripts/clean_run_step1_2.py

# Data acquisition only (Step 1.0-1.1)
python scripts/clean_run_step1.py

# Validation suite (Steps 3.0-3.7)
python scripts/clean_run_step3.py

# Validation and advanced analysis (Steps 3.0-4.8)
python scripts/clean_run_step3_4.py

# Advanced analysis only (Steps 4.0-4.8)
python scripts/clean_run_step4.py

# Core analysis only (Step 2.0-2.2)
python scripts/run_step2_only.py
```

**Individual Step Execution:**
```bash
# Core geospatial analysis
python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py

# Advanced gravitational-temporal field analysis
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_4_gravitational_temporal_field_analysis.py
```

**Analysis Components:**
- **Step 2.2:** Comprehensive geospatial temporal analysis including orbital tracking, Chandler wobble detection, and lunar standstill correlations
- **Step 4.4:** Gravitational-temporal field correlation analysis with Earth motion energy hierarchy validation

**Computational Parameters:**
- **Temporal coverage:** Full 2.5-year dataset (2023-2025) with 62.73M station pair measurements across 364 ground stations
- **Expected duration:** 20-60 minutes per major step (total pipeline: ~8-12 hours for complete Steps 1.0-4.8)
- **Requirements:** Local Python environment with scientific computing libraries

**Analytical Advantages:**
- Eliminates local computational infrastructure requirements
- Ensures consistent scientific output formatting across execution environments
- Implements robust error handling with automatic retry mechanisms
- Provides persistent results storage through Google Drive integration
- Optimized for high-performance cloud computing resources

## Summary of Key Results and Findings

### Primary Results Table

| Metric | Value | Uncertainty | Significance |
|--------|-------|-------------|--------------|
| **Dataset Size** | 62.7 million station pairs | — | 364 unique stations |
| **Temporal Coverage** | 912 days | 2023-01-01 to 2025-06-30 | 2.5 years |
| **Correlation Length (λ)** | 3,330–4,549 km | CODE: 1,198–5,918 km (95% CI) | Matches TEP prediction (1,000–10,000 km) |
| **Exponential Fit Quality (R²)** | 0.920–0.970 | Across 3 centers | On distance-bin means (Neff ≈ 25–28) |
| **Inter-Center Consistency** | CV = 12.9% | CODE, IGS, ESA | Processing-independent |

### Multi-Center Validation

| Analysis Center | Station Pairs | Correlation Length (λ) | R² |
|-----------------|---------------|------------------------|-----|
| **CODE** | 39.0 million | 4,201 ± 1,967 km | 0.970 |
| **IGS Combined** | 12.9 million | 3,330 km | 0.948 |
| **ESA** | 10.8 million | 4,549 km | 0.920 |

### Geophysical and Astronomical Signatures

| Signature | Correlation/Value | p-value | Status |
|-----------|-------------------|---------|--------|
| **Orbital Velocity Coupling** | r = −0.571 to −0.793 | p < 10⁻⁷ | Detected |
| **Chandler Wobble Modulation** | R² = 0.377–0.471 | p < 10⁻⁵ | Detected |
| **Planetary Events** | 6 Bonferroni-significant | — | Detected |
| **Diurnal Temporal Variations** | Synchronized AM peaks | p < 10⁻⁵ | Detected |

### Validation Statistics

| Test | Result | Interpretation |
|------|--------|----------------|
| **Null Test Enhancement** | 24–61× | Signal vs randomized controls |
| **FDR-BH Correction** | 203/388 tests (52.3%) | Survive multiple comparison |
| **Bonferroni Correction** | 155/388 tests (40.0%) | Survive strictest correction |
| **TID Exclusion Improvement** | 21–23% | Signal improves when excluding high-ionosphere periods |

### Key Interpretation

These findings reveal a previously undetected structure in global atomic clock networks: timing correlations that decay exponentially with distance at planetary scales. The consistency across three independent analysis centers (CODE, IGS, ESA) rules out center-specific processing artifacts. The correlation length of ~4,000 km—comparable to Earth's radius—matches TEP's a priori prediction for a conformal time-field coupled to gravitational potential gradients. The detection of orbital velocity coupling and planetary event responses suggests the signal is coupled to Earth's motion through the solar system, consistent with a preferred cosmological reference frame (CMB rest frame).

---

## Principal Results

Analysis of 62.7 million station-pair measurements across 364 total unique stations (selected from 767 total cataloged stations) reveals significant distance-structured correlations consistent with Temporal Equivalence Principle predictions:

![Distance-structured correlations in GNSS precision timing networks](site/public/figures/figure_1_TEP_site_themed.png)

**Correlation Structure:**
- **Characteristic lengths:** $\lambda = 3,330-4,549$ km across independent analysis centers (CV = 12.9% inter-center variation)
- **Statistical robustness:** Strong exponential model fits ($R^2 = 0.920$–$0.970$ on distance-bin means, Neff ≈ 25–28 bins)
- **Theoretical alignment:** Results within predicted range [1,000–10,000 km], established before data analysis

### Validation Framework
- **Multi-center consistency:** Comprehensive null hypothesis testing consistent with genuine physical signal (24–61× signal enhancement over randomized controls)
- **Circular statistics:** Phase Locking Values (PLV) range 0.1–0.4 with Rayleigh test significance $p < 10^{-5}$
- **Cross-validation:** LOSO/LODO procedures confirming robustness across temporal and spatial sampling

### Key Methodological Notes

- R² values are computed on distance-bin means (Neff ≈ 25–28 bins), not individual station pairs—standard practice in spatial correlation analysis
- Tidal frequency enhancement is a TEP prediction (gravitational forcing modulates φ field), not contamination; post-tidal band shows R² = 0.946
- Signal strengthens by 21-23% when high-ionosphere days are excluded, demonstrating the ionosphere suppresses rather than creates the correlation
- Validation framework includes 11 independent criteria with null tests showing 24-61× signal enhancement over randomized controls

## Experimental Setup and Configuration

### System Requirements
- **Python:** Version 3.10 or higher
- **Network connectivity:** Required for acquisition of GNSS precision clock products
- **Storage allocation:** Approximately 10 GB for complete analysis pipeline execution
- **Cloud platform access:** Google Cloud Platform account required for high-performance computing deployment

### Installation Procedure
```bash
# Clone repository
git clone https://github.com/matthewsmawfield/TEP-GNSS.git
cd TEP-GNSS

# Install computational dependencies
pip install -r requirements/requirements.txt

# Configure computational environment (high-performance deployment)
cp env.example .env.local
# Configure .env.local with appropriate cloud platform credentials

# Validate installation integrity
python scripts/steps/step_1_data_acquisition/step_1_0_provenance_snapshot.py
```

### Configuration Management
The analysis framework employs environment variables for computational configuration management. Detailed setup instructions are provided in [SETUP_GUIDE.md](SETUP_GUIDE.md).

**Security Protocol:** All cloud platform credentials are managed exclusively through environment variables. No authentication credentials are stored within the repository structure.

### Computational Dependencies
**Core Scientific Libraries:**
- numpy, pandas, scipy, matplotlib

**Geospatial Analysis:**
- cartopy, pyproj

**Advanced Statistical Methods:**
- scikit-learn, statsmodels, PyWavelets

**Specialized Geophysical Calculations:**
- pyIGRF (geomagnetic field modeling)

## Analysis Pipeline Execution

### Complete Experimental Protocol

#### Primary Analysis Sequence (Steps 1.0-4.1)
```bash
# Step 1.0: Data provenance and integrity verification
python scripts/steps/step_1_data_acquisition/step_1_0_provenance_snapshot.py

# Step 1.1: GNSS precision clock data acquisition
python scripts/steps/step_1_data_acquisition/step_1_1_tep_data_acquisition.py

# Step 1.2: Coordinate validation and comprehensive audit framework
python scripts/steps/step_1_data_acquisition/step_1_2_tep_coordinate_validation.py

Establishes coordinate system integrity through comprehensive audit procedures. Validates ECEF coordinate data quality, performs integrated station identification audit with spatial analysis, determines authoritative station catalog for correlation analysis, and generates comprehensive validation summary with data-driven metadata. Ensures coordinate data integrity and establishes definitive station catalog for subsequent correlation analysis.

# Step 2.0: Temporal Equivalence Principle correlation analysis (Primary signal detection) ~3-4 hours*
python scripts/steps/step_2_core_analysis/step_2_0_tep_correlation_analysis.py

Implements core TEP signal detection methodology using phase-coherent cross-spectral density analysis. Computes complex cross-spectral density between all station pairs within the 10-500 µHz frequency band, extracts phase-coherent correlations using cos(phase(CSD)), and fits exponential decay models to correlation-distance relationships. Employs band-limited analytical approach preserving essential phase information for TEP signal detection.

# Step 2.1: Data quality validation and transparency framework
python scripts/steps/step_2_core_analysis/step_2_1_data_quality_validation.py

Comprehensive data quality assessment and transparency analysis. Processes quality-filtered correlation data from Step 2.0 with geospatial enrichments (azimuth, local time differences), performs extensive validation including station coverage analysis, temporal discontinuity detection, duplicate identification, outlier validation, boundary phase clustering analysis, and inter-analysis center comparison. Generates comprehensive transparency reports with identified anomalies and analytical recommendations to ensure scientific rigor.

# Step 2.2: Geospatial-temporal correlation analysis
python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py

Comprehensive geospatial and temporal analysis framework including astronomical event correlations, orbital mechanics, anisotropy analysis, spherical harmonics, and advanced temporal field studies. Examines correlations with planetary positions, lunar standstill periods, solar eclipse events, and Earth's orbital motion to validate TEP predictions across multiple temporal and spatial scales.

# Step 3.0: Cross-validation framework
python scripts/steps/step_3_validation_suite/step_3_0_tep_cross_validation_suite.py

Comprehensive validation framework implementing block-wise (monthly/spatial), Leave-One-Station-Out (LOSO), Leave-One-Day-Out (LODO), and block bootstrap analyses. Provides rigorous validation of TEP correlation parameters through multiple complementary statistical approaches to ensure analytical robustness.

# Step 3.2: Null hypothesis validation framework
python scripts/steps/step_3_validation_suite/step_3_2_tep_null_tests.py

# Step 4.0: Advanced analytical procedures
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_0_tep_advanced_analysis.py

# Step 4.1: Scientific visualization generation
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_1_tep_visualization.py
```

#### Extended Analysis Protocol (Steps 4.2-4.8)
```bash
# Step 4.2: Synthesis visualization generation
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_2_tep_synthesis_figure.py

# Step 4.3: High-resolution astronomical event analysis
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_3_high_resolution_astronomical_events.py

# Step 3.3: Methodology validation framework
python scripts/steps/step_3_validation_suite/step_3_3_methodology_validation.py

# Step 3.4: Geographic bias characterization and validation
python scripts/steps/step_3_validation_suite/step_3_4_geographic_bias_validation.py

# Step 3.5: Realistic ionospheric validation procedures
python scripts/steps/step_3_validation_suite/step_3_5_realistic_ionospheric_validation.py

# Step 3.6: Control band analysis (Frequency specificity validation)
python scripts/steps/step_3_validation_suite/step_3_6_control_band_analysis.py

Validates frequency specificity of TEP correlations through analysis of theoretically unmotivated control band (1000-2000 µHz) where no signal is predicted. Implements identical phase-coherent analysis methodology as Step 2.0 but within higher frequency range dominated by white noise processes. Expected outcome: $R^2 \approx 0.05$ in control band versus $R^2 \approx 0.85$ in TEP band (10-500 µHz), demonstrating that observed correlations are not broadband statistical artifacts. Addresses multiple testing concerns and "look-elsewhere effect" criticisms.

# Step 3.7: Bootstrap convergence validation
python scripts/steps/step_3_validation_suite/step_3_7_bootstrap_convergence_validation.py

Validates bootstrap convergence and stability through systematic analysis of bootstrap iteration requirements. Assesses convergence behavior of correlation parameter estimates (λ, A, C₀) across varying bootstrap sample sizes, determines minimum iteration requirements for stable confidence intervals, and validates bootstrap assumption adherence. Ensures robust statistical inference and optimal computational efficiency in bootstrap procedures.

# Step 4.4: Gravitational-temporal field coupling analysis
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_4_gravitational_temporal_field_analysis.py

# Step 4.5: Comprehensive diurnal and seasonal analysis
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_5_comprehensive_diurnal_analysis.py

# Step 4.6: Traveling Ionospheric Disturbance exclusion analysis
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_6_tid_exclusion_analysis.py

# Step 4.7: Multiple comparison correction framework (Final validation)
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_7_multiple_comparison_corrections.py

Systematic application of multiple comparison correction procedures including Bonferroni, False Discovery Rate (FDR), and Family-wise Error Rate corrections to all statistical tests performed across Steps 2.0-4.8. Ensures robust control of Type I error inflation across the complete analysis pipeline. Must be executed AFTER Step 4.0 completion (requires model comparison results for comprehensive correction).

# Step 4.8: Multiband visualization and analysis
python scripts/steps/step_4_advanced_analysis_and_visualization/step_4_8_multiband_visualization.py

Comprehensive multiband frequency analysis and visualization framework. Analyzes correlation patterns across multiple frequency bands to validate frequency-specific TEP predictions, generates comparative visualizations of amplitude decay and correlation lengths across frequency ranges, and provides spectral analysis overview with post-tidal emphasis. Demonstrates frequency-dependent behavior consistent with TEP theoretical framework.
```

### Analytical Configuration

#### v0.24 Configuration Framework (Jaipur Release - Established Methodology)

**Core Analysis Parameters:**
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `TEP_USE_PHASE_BAND` | 1 | Band-limited phase analysis methodology (v0.6 implementation) |
| `TEP_COHERENCY_F1` | $1 \times 10^{-5}$ | Lower frequency boundary (10 µHz) |
| `TEP_COHERENCY_F2` | $5 \times 10^{-4}$ | Upper frequency boundary (500 µHz) |
| `TEP_BINS` | 40 | Distance binning structure for correlation analysis |

**Computational Processing Parameters:**
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `TEP_PROCESS_ALL_CENTERS` | 1 | Process all analysis centers (CODE, IGS, ESA) |
| `TEP_WORKERS` | 14 | Number of parallel processing workers |
| `TEP_MEMORY_LIMIT_GB` | 8.0 | Memory allocation limit in GB |
| `TEP_BOOTSTRAP_ITER` | 5000 | Bootstrap iteration count for statistical validation |
| `TEP_NULL_ITERATIONS` | 500 | Null test scrambling iterations |

**Temporal Analysis Parameters:**
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `TEP_DATE_START` | 2023-01-01 | Analysis temporal window start date |
| `TEP_DATE_END` | 2025-06-30 | Analysis temporal window end date |
| `TEP_MIN_EPOCHS` | 20 | Minimum observation epochs per station |


**Statistical Validation Parameters:**
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `TEP_MIN_BIN_COUNT` | 50 | Minimum station pairs per distance bin |
| `TEP_MIN_BINS_FOR_FIT` | 5 | Minimum bins required for exponential fit |
| `TEP_CORRELATION_LENGTH_INITIAL_GUESS` | 3000 | Initial guess for correlation length (km) |

### Configuration Override

All parameters can be overridden via environment variables. For detailed configuration management, see [SETUP_GUIDE.md](SETUP_GUIDE.md) and `scripts/utils/config.py`.

## Scientific Validation Framework

### Comprehensive Validation Suite

The TEP-GNSS framework implements a rigorous multi-tier validation approach:

**Statistical Validation:**
- **Bootstrap Analysis:** 5,000+ iterations with confidence interval estimation
- **Cross-Validation:** Leave-One-Station-Out (LOSO) and Leave-One-Day-Out (LODO) procedures
- **Null Hypothesis Testing:** 500+ scrambling iterations demonstrating 24-61× signal enhancement

**Methodological Validation:**
- **Multi-Center Consistency:** Independent validation across CODE, IGS, and ESA analysis centers
- **Frequency Specificity:** Control band analysis demonstrating signal specificity to 10-500 µHz range
- **Geographic Bias Assessment:** Systematic evaluation of spatial sampling effects

**Physical Validation:**
- **Astronomical Correlations:** Coherent coupling to Earth's orbital dynamics and planetary events
- **Temporal Consistency:** Seasonal correlation patterns and diurnal variations
- **Ionospheric Exclusion:** TID analysis demonstrating signal independence from ionospheric effects

## Data Sources and Attribution

### Primary Data Sources

This analysis uses publicly available GNSS clock products from the International GNSS Service (IGS) and its analysis centers. All data sources are freely available for scientific research under IGS Terms of Use.

#### GNSS Clock Products

**Paper 1 (Multi-Center Analysis):**
- **CODE (Center for Orbit Determination in Europe)**
  - Provider: Astronomical Institute, University of Bern (AIUB)
  - Source: http://ftp.aiub.unibe.ch/CODE/
  - Coverage: January 1, 2023 – June 30, 2025 (912 days)
  - Station Pairs: 39.0 million measurements
  - Citation: Steigenberger et al. (2021), Johnston et al. (2017)

- **IGS Combined Products**
  - Provider: International GNSS Service (multi-center weighted combination)
  - Source: https://igs.bkg.bund.de/root_ftp/IGS/products/
  - Coverage: January 1, 2023 – June 30, 2025 (912 days)
  - Station Pairs: 12.9 million measurements
  - Citation: Johnston et al. (2017)

- **ESA (European Space Agency)**
  - Provider: ESA Navigation Support Office
  - Source: http://navigation-office.esa.int/products/gnss-products/
  - Coverage: January 1, 2023 – June 30, 2025 (912 days)
  - Station Pairs: 10.8 million measurements
  - Citation: Fernández (2016), Johnston et al. (2017)

#### Station Coordinates

- **Source:** IGS Network Metadata (ITRF2014/ITRF2020)
- **Access:** https://files.igs.org/pub/station/general/IGSNetworkWithFormer.json
- **Format:** JSON with Cartesian coordinates (X, Y, Z)
- **License:** Freely available under IGS Terms of Use
- **Citation:** Johnston et al. (2017)

### Data Format and Access

- **Clock Products:** RINEX 3 CLK format (compressed: .gz or .Z)
- **Temporal Resolution:** 30-second epochs
- **Quality Control:** Comprehensive filtering and validation (see Step 2.1)
- **Terms of Use:** [IGS Data and Product Disclaimer](https://igs.org/wp-content/uploads/2020/09/IGS-Data-and-Product-Disclaimer-and-Terms-of-Use-200805.pdf)

### Required Citations

When using this analysis framework or data, please cite:

1. **This Work:**
   - Paper 1: Smawfield (2025), DOI: 10.5281/zenodo.17127229
   - Paper 2: Smawfield (2025), DOI: 10.5281/zenodo.17517141

2. **Data Providers:**
   - IGS: Johnston et al. (2017), DOI: 10.1007/978-3-319-42928-1_33
   - CODE: Steigenberger et al. (2021), DOI: 10.1007/s00190-021-01487-8
   - ESA: Fernández (2016), ION GNSS+ 2016
   - JPL Ephemeris: Folkner et al. (2014), IPN Progress Report 42-196
   - Astropy: Astropy Collaboration (2013, 2022)

3. **Software Dependencies:**
   - NumPy, SciPy, Pandas, Matplotlib

### Key References

- **Johnston, G., Riddell, A., & Hausler, G. (2017).** The International GNSS Service. In *Springer Handbook of Global Navigation Satellite Systems* (pp. 967-982). DOI: 10.1007/978-3-319-42928-1_33

- **Steigenberger, P., Montenbruck, O., Dach, R., et al. (2021).** CODE reprocessing 1995-2020: improved GPS orbits and clocks. *Journal of Geodesy*, 95, 65. DOI: 10.1007/s00190-021-01487-8

- **Folkner, W. M., Williams, J. G., Boggs, D. H., Park, R. S., & Kuchynka, P. (2014).** The Planetary and Lunar Ephemerides DE430 and DE431. *IPN Progress Report 42-196*, JPL.

- **Astropy Collaboration (2022).** The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the Latest Major Release (v5.0) of the Core Package. *The Astrophysical Journal*, 935(2), 167. DOI: 10.3847/1538-4357/ac7c74

- **Astropy Collaboration (2013).** Astropy: A community Python package for astronomy. *Astronomy & Astrophysics*, 558, A33. DOI: 10.1051/0004-6361/201322068

### Data Availability Statement

All GNSS clock products and station coordinates used in this analysis are publicly available from the International GNSS Service and its contributing analysis centers. The complete analysis pipeline, including data acquisition scripts, is available in this repository under the MIT License. Processed results and supplementary materials are archived on Zenodo with persistent DOIs.

**Cross-Center Validation:** Three independent analysis centers (CODE, IGS Combined, ESA) demonstrate processing-independence with R² = 0.920-0.970 consistency across centers.

## Data Products and Outputs

### Primary Results Structure

```
results/
├── outputs/           # 57+ JSON result files
│   ├── step_2_0_correlation_*.json           # Core correlation analysis
│   ├── step_3_*_validation.json              # Validation suite results
│   ├── step_4_*_advanced_analysis.json       # Advanced analysis outputs
│   └── meta_analysis_comprehensive.json      # Comprehensive meta-analysis
├── figures/          # 20+ publication-quality figures
│   ├── figure_1_TEP_site_themed.png         # Primary correlation figure
│   ├── step_4_2_tep_synthesis_figure.png    # Synthesis visualization
│   └── step_4_4_comprehensive_*.png         # Gravitational analysis
└── tmp/              # Intermediate processing files
    └── streaming/    # TID/Hilbert analysis outputs
```

### Key Scientific Outputs

**Correlation Analysis:**
- Distance-structured correlation coefficients with exponential decay fits
- Characteristic correlation lengths (λ = 3,330-4,549 km) across analysis centers
- Statistical significance assessment with bootstrap confidence intervals

**Validation Results:**
- Multi-center consistency metrics (CV = 12.9% inter-center variation)
- Null test enhancement factors (24-61× signal over randomized controls)
- Cross-validation stability assessments ($R^2 = 0.920$-$0.970$)

**Advanced Analysis:**
- Gravitational-temporal field correlations ($r = -0.458$, $p < 10^{-48}$)
- Astronomical event coherence modulations (11 planetary events identified)
- Multiband frequency analysis validating TEP predictions

## Citation and Attribution

### Paper 1: Multi-Center Analysis

```bibtex
@article{smawfield2025globaltimeechoes,
  title={Global Time Echoes: Distance-Structured Correlations in GNSS Clocks (Jaipur v0.24)},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.17127229},
  url={https://doi.org/10.5281/zenodo.17127229},
  note={Preprint}
}
```

### Theoretical Framework Citation

```bibtex
@article{smawfield2025tep,
  title={Temporal Equivalence Principle: Theoretical Framework},
  author={Smawfield, Matthew Lukin},
  year={2025},
  doi={10.5281/zenodo.16921911},
  url={https://doi.org/10.5281/zenodo.16921911}
}
```

## License and Distribution

This repository is distributed under the **Creative Commons Attribution 4.0 International License (CC-BY-4.0)**. 

**You are free to:**
- **Share:** Copy and redistribute the material in any medium or format
- **Adapt:** Remix, transform, and build upon the material for any purpose, including commercially

**Under the following terms:**
- **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made

For complete license terms, see [LICENSE](LICENSE).

## Contact and Collaboration

**Author:** Matthew Lukin Smawfield  
**Email:** matthewsmawfield@gmail.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)

### Collaboration Invitation

This research presents findings that warrant independent investigation and collaborative validation. I welcome collaboration from researchers in:

- **GNSS/Geodesy:** Independent analysis of precision timing networks
- **Theoretical Physics:** Extensions and refinements of TEP framework  
- **Statistical Methods:** Advanced validation techniques and bias assessment
- **Astronomy/Geophysics:** Correlations with astronomical events and Earth dynamics
- **Metrology:** Atomic clock network analysis and time transfer protocols

For collaboration inquiries, technical discussions, or independent validation efforts, please contact:  
📧 **matthewsmawfield@gmail.com**

### Replication and Validation Encouraged

**Independent replication is essential for scientific progress.** These findings challenge conventional understanding and require rigorous independent validation. I strongly encourage researchers to:

- Replicate the analysis using independent methodologies and software implementations
- Challenge the methodology through alternative statistical approaches and bias assessments  
- Extend the dataset to different time periods, analysis centers, or GNSS constellations
- Test alternative explanations for the observed distance-structured correlations
- Propose novel validation approaches that could strengthen or refute the findings

This work is designed for reproducibility. All code, data processing steps, and analysis parameters are fully documented and publicly available. Scientific skepticism is welcomed and necessary—these findings have significant implications that require independent verification.

### Resources

**Repository:** [https://github.com/matthewsmawfield/TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS)  
**Website:** [https://matthewsmawfield.github.io/TEP-GNSS/](https://matthewsmawfield.github.io/TEP-GNSS/)  
**DOI:** [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229)  

---

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons and MIT licenses to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthewsmawfield@gmail.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
