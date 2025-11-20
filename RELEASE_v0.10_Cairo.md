# Global Time Echoes: 25-Year Temporal Evolution (v0.10 Cairo)

## Release Summary

**Paper 2: 25-Year Confirmatory Analysis** - This release presents the temporal extension of the TEP-GNSS analysis framework, extending the validation period from 2.5 years to 25.3 years using CODE analysis center data.

**DOI:** [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141)  
**Publication Date:** 20 November 2025  
**Website:** [https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/](https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/)

## Key Scientific Findings

### Temporal Stability Confirmation
- **Decadal Validation**: Original TEP signatures confirmed stable over 25-year timescale
- **Orbital Coupling**: Enhanced detection of Earth's orbital velocity correlation (r = -0.864, p = 0.0013)
- **Consistent Characteristic Lengths**: Correlation decay patterns remain consistent with theoretical predictions

### Long-Period Geophysical Signatures
- **18.6-Year Lunar Nutation**: Clear detection with R² = 0.641, p < 10⁻⁸
- **Chandler Wobble**: Confirmed with extended temporal baseline
- **Seasonal Patterns**: Robust annual modulation effects identified

### Enhanced Planetary Event Detection
- **67 Total Events**: Significant increase from 11 events in original analysis
- **31 Bonferroni-Significant**: Events surviving multiple comparison correction
- **Mass-Scaling Analysis**: No significant correlation with planetary masses (r = -0.053)

## Dataset Specifications

| Metric | Value |
|--------|-------|
| **Temporal Coverage** | 25.3 years (2000-2025) |
| **Station Pairs** | 165.2 million |
| **Unique Stations** | 474 (CODE analysis center) |
| **Daily Coverage** | 9,218 days |
| **Analysis Center** | CODE (extended temporal baseline) |

## Repository Structure

This release implements a multi-paper repository structure:

```
TEP-GNSS/
├── site/                    # Paper 1: Multi-Center Analysis (v0.21)
├── site-code-longspan/      # Paper 2: 25-Year Extension (v0.10)
├── deploy-all.sh           # Combined deployment script
├── final-dist/             # Temporary combined build output
└── scripts/code_longspan/  # Paper 2 analysis scripts
```

## Technical Improvements

### Enhanced Analysis Pipeline
- **Extended Data Processing**: Optimized for 25-year temporal coverage
- **Improved Statistical Power**: Enhanced detection capabilities for long-period phenomena
- **Comprehensive Validation**: Extended temporal validation framework

### Multi-Paper Deployment
- **Independent Sites**: Each paper maintains independent build and deployment
- **Combined Deployment**: `deploy-all.sh` script for simultaneous deployment
- **Cross-Reference Navigation**: Seamless navigation between papers

## Installation and Usage

### Quick Start
```bash
# Clone repository
git clone https://github.com/matthewsmawfield/TEP-GNSS.git
cd TEP-GNSS

# Paper 1 (Multi-Center Analysis)
cd site
npm install
npm run build
npm run dev

# Paper 2 (25-Year Extension)
cd ../site-code-longspan
npm install
npm run build
npm run dev
```

### Combined Deployment
```bash
# Deploy both papers simultaneously
./deploy-all.sh
```

### Analysis Scripts
```bash
# Paper 2 longspan analysis
python scripts/code_longspan/step_2_2_code_longspan.py
```

## Scientific Validation

### Statistical Framework
- **Bootstrap Analysis**: 5,000+ iterations with confidence intervals
- **Multiple Comparison Corrections**: Bonferroni and FDR procedures
- **Cross-Validation**: Temporal and spatial validation procedures

### Evidence Assessment
- **STRONG EVIDENCE**: 5/8 categories detected for temporal-gravitational coupling
- **Primary Signatures**: Orbital motion, 3D anisotropy, mesh coherence
- **Secondary Signatures**: Nutation cycles, continuous planetary correlation

## Files and Resources

### Core Analysis Files
- `scripts/code_longspan/step_2_2_code_longspan.py` - Main analysis script
- `site-code-longspan/` - Paper 2 website and documentation
- `results/outputs/code_longspan/` - Analysis results and JSON outputs

### Documentation
- `README.md` - Updated with both papers information
- `MULTI_PAPER_STRUCTURE.md` - Repository structure documentation
- `manuscript-code-longspan.md` - Complete manuscript markdown

### Deployment
- `deploy-all.sh` - Combined deployment script
- `site-code-longspan/build.js` - Build configuration for Paper 2

## Citation

```bibtex
@article{smawfield2025globaltimeechoes25year,
  title={Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks (Cairo v0.10)},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.17517141},
  url={https://doi.org/10.5281/zenodo.17517141},
  note={Preprint}
}
```

## Related Releases

- **[v0.21 (Jaipur)](https://github.com/matthewsmawfield/TEP-GNSS/releases/tag/v0.21)** - Paper 1: Multi-Center Analysis
- **[v0.10 (Cairo)](https://github.com/matthewsmawfield/TEP-GNSS/releases/tag/v0.10)** - Paper 2: 25-Year Extension (current)

## License

This release is distributed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0).

## Acknowledgments

This research extends the foundational TEP-GNSS framework with enhanced temporal validation. The 25-year baseline provides unprecedented statistical power for detecting long-period geophysical signatures and confirming the temporal stability of distance-structured correlations in GNSS networks.

---

**Paper 1 Website:** https://matthewsmawfield.github.io/TEP-GNSS/  
**Paper 2 Website:** https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/  
**Repository:** https://github.com/matthewsmawfield/TEP-GNSS/
