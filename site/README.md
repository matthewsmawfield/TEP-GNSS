# Global Time Echoes: Distance-Structured Correlations in GNSS Clocks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17127229.svg)](https://doi.org/10.5281/zenodo.17127229)

![TEP-GNSS Analysis Overview](./public/og-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.15 (Jaipur)  
**Date:** 7 October 2025  
**Status:** Preprint (Analysis Package)  
**DOI:** [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229)  
**Website:** [https://matthewsmawfield.github.io/TEP-GNSS/](https://matthewsmawfield.github.io/TEP-GNSS/)

## Abstract

We report distance-structured correlations in GNSS clock products testing predictions of the Temporal Equivalence Principle (TEP) theoretical framework. Using a phase-coherent analysis across IGS Combined, ESA Final, and CODE centers, we observe exponential correlation lengths of 3,330–4,549 km across 62.7M station-pair measurements, consistent with theory-predicted characteristic decay patterns. Primary pooled fits: R² = 0.92–0.97 (distance-bin means, N<sub>eff</sub> ≈ 25–28 bins used from 40 attempted). Sensitivity subsets (elevation/geomagnetic): R² = 0.70–0.91. Results are validated by comprehensive null tests (ΔR² = 0.85-0.93 separation from controls) and circular statistics (PLV 0.1–0.4, Rayleigh p < 1e-5). Bootstrap validation shows center-specific ranges: 3,685–5,413 km (CODE), 3,021–3,639 km (ESA Final), and 3,388–4,140 km (IGS Combined). Results are consistent with screened scalar-field models coupling to atomic transition frequencies and are robust across centers, geographies, and elevation ranges.

## Analysis Package

This repository contains the complete analysis pipeline for testing the Temporal Equivalence Principle theoretical predictions using GNSS atomic clock data:

- **Data processing**: Automated download and validation of IGS, ESA, and CODE clock products
- **Phase-coherent analysis**: Advanced cross-spectral density methods preserving phase information
- **Statistical validation**: Comprehensive null tests and circular statistics
- **Reproducible science**: Complete pipeline with checkpointing and error handling

## Key Results

- **Multi-center consistency**: λ = 3,330–4,549 km primary range with bootstrap validation: 3,685–5,413 km (CODE), 3,021–3,639 km (ESA Final), 3,388–4,140 km (IGS Combined)
- **Strong statistical fits**: R² = 0.920–0.970 for exponential correlation models
- **Theoretical compatibility**: Results within TEP-predicted range [1,000–10,000 km]
- **Validated methodology**: Real fits R² = 0.920–0.970 vs. null mean 0.034–0.056 (ΔR² = 0.864–0.946; permutation p < 1e-5)

## How to cite

**Main DOI (always latest version):** **10.5281/zenodo.17127229**

BibTeX:

```bibtex
@misc{Smawfield_TEP_GNSS_2025,
  author       = {Matthew Lukin Smawfield},
  title        = {Global Time Echoes: Distance-Structured Correlations in GNSS
                  Clocks (Jaipur v0.15)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17127229},
  url          = {https://doi.org/10.5281/zenodo.17127229},
  note         = {Preprint}
}
```

---

**Contact:** matthewsmawfield@gmail.com  
**Website:** https://matthewsmawfield.github.io/TEP-GNSS/  
**Zenodo:** https://doi.org/10.5281/zenodo.17127229