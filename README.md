# TEP-GNSS: Global Time Echoes Analysis Package

**Author:** Matthew Lukin Smawfield  
**Version:** v0.2 (Jaipur)  
**Date:** September 17, 2025  
**DOI:** [10.5281/zenodo.17148714](https://doi.org/10.5281/zenodo.17148714)

## Overview

This repository contains a complete analysis package for testing Temporal Equivalence Principle (TEP) predictions using Global Navigation Satellite System (GNSS) ground station clock data. The analysis examines distance-structured correlations across three independent analysis centers: CODE, IGS, and ESA.

## Key Findings

Through analysis of 62.7 million station pair measurements from 529 global ground stations, we observe:

- **Correlation lengths**: λ = 3,299–3,818 km across all analysis centers (15.7% variation)
- **Statistical significance**: Strong exponential fits (R² = 0.915–0.964)
- **Theoretical consistency**: Results within predicted range [1,000–10,000 km]
- **Validation**: Comprehensive null tests confirm signal authenticity

## Installation

### Prerequisites
- Python 3.8+
- Internet connection for data download

### Setup
```bash
pip install -r requirements/requirements.txt
```

## Usage

### Complete Analysis Pipeline
```bash
# Step 1: Download GNSS clock data
python scripts/steps/step_1_tep_data_acquisition.py

# Step 2: Validate station coordinates  
python scripts/steps/step_2_tep_coordinate_validation.py

# Step 3: Correlation analysis
python scripts/steps/step_3_tep_correlation_analysis.py

# Step 4: Geospatial processing
python scripts/steps/step_4_aggregate_geospatial_data.py

# Step 5: Statistical validation
python scripts/steps/step_5_tep_statistical_validation.py

# Step 6: Null hypothesis testing
python scripts/steps/step_6_tep_null_tests.py

# Step 7: Advanced analysis
python scripts/steps/step_7_tep_advanced_analysis.py

# Step 8: Generate visualizations
python scripts/steps/step_8_tep_visualization.py
```

### Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TEP_PROCESS_ALL_CENTERS` | 1 | Process CODE, IGS, and ESA data |
| `TEP_WORKERS` | auto | Number of parallel workers |
| `TEP_BINS` | 40 | Distance bins for correlation analysis |
| `TEP_MAX_DISTANCE_KM` | 13000 | Maximum analysis distance |
| `TEP_USE_PHASE_BAND` | 0 | Use band-limited phase analysis (10-500 μHz) |
| `TEP_COHERENCY_F1` | 1e-5 | Lower frequency bound (Hz) for phase band |
| `TEP_COHERENCY_F2` | 5e-4 | Upper frequency bound (Hz) for phase band |

## Data Sources

- **CODE**: Center for Orbit Determination in Europe
- **IGS**: International GNSS Service  
- **ESA**: European Space Agency

All data sourced directly from official repositories. No synthetic or fallback data is used.

## Results

Main outputs are located in:
- `results/outputs/`: Analysis results in JSON format
- `results/figures/`: Generated visualizations
- `TEP-GNSS_manuscript_v0.2_Jaipur.md`: Comprehensive analysis report

## Scientific Background

This analysis implements **Clock Network Correlation Analysis**, a key experimental test from the Temporal Equivalence Principle (TEP) framework ([Smawfield, 2025](https://matthewsmawfield.github.io/TEP/); [DOI: 10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911)).

### Theoretical Foundation

The TEP proposes that gravitational fields couple directly to clock transition frequencies through a conformal factor A(φ) = exp(2βφ/M_Pl), where φ is a scalar time field. This coupling manifests as distance-structured correlations in precision timing networks, with correlation structure determined by screening properties of the underlying field.

### Experimental Design (TEP Section E)

**Objective**: Detect spatial correlations and environmental screening signatures in ground station clock frequency residuals consistent with screened scalar field coupling to transition frequencies.

**Phase I - Distance Correlation Analysis**:
- Analyze precision timing networks (GNSS ground stations) for distance-dependent correlations
- Apply phase-coherent cross-spectral analysis between station pairs  
- Bin pairs by 3D distance, fit exponential correlation model: C(r) = A·exp(-r/λ) + C₀
- Cross-validate across independent analysis centers to control systematics

**Theoretical Predictions**:
- Exponential decay with characteristic length λ ~ 1,000-10,000 km for viable screening parameters
- Multi-center consistency with <5% variation in fitted parameters

## Methodology

1. **Phase-coherent analysis**: Preserves complex cross-spectral density phase information
2. **Distance binning**: 40 logarithmic bins from 50 km to 13,000 km
3. **Exponential fitting**: Nonlinear least squares optimization
4. **Multi-center validation**: Independent analysis across three data products
5. **Statistical validation**: Comprehensive null tests and bootstrap confidence intervals

## Quality Assurance

- Multi-center consistency validation
- Comprehensive null testing (distance/phase/station scrambling)
- Bootstrap confidence intervals
- ECEF coordinate validation against ITRF2014
- Complete reproducibility with version control

## Citation

If you use this analysis package, please cite both the analysis and underlying theory:

**This Analysis:**
```bibtex
@misc{Smawfield_TEP_GNSS_2025,
  author = {Matthew Lukin Smawfield},
  title = {Global Time Echoes: Distance-Structured Correlations in GNSS 
           Clocks Across Independent Networks},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17148714},
  url = {https://doi.org/10.5281/zenodo.17148714}
}
```

**TEP Theory:**
```bibtex
@misc{Smawfield_TEP_2025,
  author = {Matthew Lukin Smawfield},
  title = {The Temporal Equivalence Principle: Dynamic Time, Emergent Light 
           Speed, and a Two-Metric Geometry of Measurement},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.16921911},
  url = {https://doi.org/10.5281/zenodo.16921911},
  note = {Cites the latest version}
}
```

## Documentation

- **Project website**: [https://matthewsmawfield.github.io/TEP-GNSS/](https://matthewsmawfield.github.io/TEP-GNSS/)
- **Full analysis report**: `TEP-GNSS_manuscript_v0.1_Jaipur.md`
- **Execution logs**: `logs/` directory

## License

This work is licensed under CC BY 4.0. See LICENSE file for details.

## Contact

For questions or collaboration opportunities:  
**Matthew Lukin Smawfield**  
matthewsmawfield@gmail.com