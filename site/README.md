# Global Time Echoes: Distance-Structured Correlations in GNSS Clocks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17127229.svg)](https://doi.org/10.5281/zenodo.17127229)

![TEP-GNSS Analysis Overview](./public/og-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.25 (Jaipur)  
**Date:** 29 April 2026  
**Status:** Preprint (Analysis Package)  

**DOI:** [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229)  
**Website:** [https://matthewsmawfield.github.io/TEP-GNSS/](https://matthewsmawfield.github.io/TEP-GNSS/)

## Abstract

Phase-coherent spectral analysis of 62.7 million station-pair measurements from 364 GNSS stations (2023-2025) reveals systematic distance-structured correlations in clock networks. These correlations follow an exponential decay with a median correlation length λ = 3,330–4,549 km (95% CIs: CODE 1,198–5,918 km; IGS 3,197–4,871 km; ESA 2,532–3,984 km) and show strong goodness-of-fit when evaluated on distance-binned means across three independent analysis centres (R² = 0.920–0.970; fits are to bin means, not raw pairs). Cross-center validation, consistent across 12 frequency bands and confirmed through multiple binning schemes and null hypothesis testing, demonstrates these patterns represent genuine physical correlations rather than systematic artifacts. The patterns also show dependencies on station elevation and geomagnetic latitude, consistent with theoretical frameworks involving screened scalar fields via continuous Temporal Topology.

The primary inference rests on cross-centre distance-structured covariance and λ<sub>T</sub>; planetary, Chandler, diurnal, and geomagnetic signatures are treated as secondary or exploratory consistency tests. The correlations demonstrate systematic coupling with Earth's orbital motion (r = -0.571 to -0.793 across centers), planetary gravitational influences (6 Bonferroni-significant events), Chandler wobble modulation (R² = 0.377–0.471), and systematic diurnal temporal variations with synchronized early morning coherence peaks (Local Solar Time). Comprehensive validation demonstrates 24-61× signal enhancement over randomized controls (z = 15.8-31.9 across 180 null test iterations), with FDR-BH: 203/388 tests (52.3%), Hierarchical EB: 154/388 (39.7%), and Bonferroni: 155/388 (40.0%) surviving multiple-comparison correction across 19 independent validation families. TID exclusion analysis shows 21-23% signal improvement when excluding high-ionosphere periods—the ionosphere suppresses rather than creates the correlation.

The investigation was structured to test predictions from the Temporal Equivalence Principle (TEP) framework, which suggested a correlation length (λ) of 1,000–10,000 km. The full analysis yielded λ = 3,330–4,549 km, a result consistent with this expectation which motivated tests of derived predictions (diurnal, eclipse, and orbital signatures). While multi-center consistency and extensive validation provide a strong basis for these findings, alternative explanations involving sophisticated systematics cannot be fully excluded. Therefore, definitive physical interpretation awaits critical next steps: raw-data analysis, multi-constellation testing, and independent replication. A companion 25-year confirmatory analysis using CODE data is presented at [TEP-GNSS-II](https://matthewsmawfield.github.io/TEP-GNSS-II/).

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
- **Validated methodology**: Real fits R² = 0.920–0.970 vs. null mean 0.015–0.040 (ΔR² = 0.89-0.95; z = 15.8-31.9, all p < 0.05; 24-61× signal-to-null ratios across 180 scrambling iterations)

## How to cite

**Main DOI (always latest version):** **10.5281/zenodo.17127229**

BibTeX:

```bibtex
@misc{Smawfield_TEP_GNSS_2025,
  author       = {Matthew Lukin Smawfield},
  title        = {Global Time Echoes: Distance-Structured Correlations in GNSS
                  Clocks (Jaipur v0.25)},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17127229},
  url          = {https://doi.org/10.5281/zenodo.17127229},
  note         = {Preprint}
}
```

---

**Contact:** matthew@mlsmawfield.com  
**Website:** https://matthewsmawfield.github.io/TEP-GNSS/  
**Zenodo:** https://doi.org/10.5281/zenodo.17127229