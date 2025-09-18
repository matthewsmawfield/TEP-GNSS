# Global Time Echoes: Distance-Structured Correlations in GNSS Clocks Across Independent Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17148714.svg)](https://doi.org/10.5281/zenodo.17148714)

![TEP-GNSS Analysis Overview](./og-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.2 (Jaipur)  
**Date:** 17 Sep 2025  
**Status:** Preprint (Analysis Package)  
**DOI:** [10.5281/zenodo.17148714](https://doi.org/10.5281/zenodo.17148714)  
**Website:** [https://matthewsmawfield.github.io/TEP-GNSS/](https://matthewsmawfield.github.io/TEP-GNSS/)

## Abstract

We report distance-structured correlations in GNSS clock products using a phase-coherent analysis across IGS Combined, ESA Final, and CODE centers. Across 62.7M station-pair measurements, exponential correlation lengths of 3,299–3,818 km are observed with strong fits (R² = 0.915–0.964), validated by comprehensive null tests (8.5–44× signal destruction) and circular statistics (PLV 0.1–0.4, Rayleigh p < 1e-5). Results are consistent with screened scalar-field models coupling to atomic transition frequencies and are robust across centers, geographies, and elevation ranges.

## Analysis Package

This repository contains the complete analysis pipeline for testing the Temporal Equivalence Principle using GNSS atomic clock data:

- **Data processing**: Automated download and validation of IGS, ESA, and CODE clock products
- **Phase-coherent analysis**: Advanced cross-spectral density methods preserving phase information
- **Statistical validation**: Comprehensive null tests and circular statistics
- **Reproducible science**: Complete pipeline with checkpointing and error handling

## Key Results

- **Multi-center consistency**: λ = 3,299–3,818 km (15.7% variation) across independent analysis centers
- **Strong statistical fits**: R² = 0.915–0.964 for exponential correlation models
- **Theoretical compatibility**: Results within TEP-predicted range [1,000–10,000 km]
- **Validated methodology**: Null tests confirm signal authenticity (8.5–44× destruction under scrambling)

## How to cite

**Main DOI (always latest version):** **10.5281/zenodo.17148714**

BibTeX:

```bibtex
@misc{Smawfield_TEP_GNSS_2025,
  author       = {Matthew Lukin Smawfield},
  title        = {Global Time Echoes: Distance-Structured Correlations in GNSS 
                  Clocks Across Independent Networks (Jaipur v0.2)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17148714},
  url          = {https://doi.org/10.5281/zenodo.17148714},
  note         = {Preprint}
}
```

---

**Contact:** matthewsmawfield@gmail.com  
**Website:** https://matthewsmawfield.github.io/TEP-GNSS/  
**Zenodo:** https://doi.org/10.5281/zenodo.17148714