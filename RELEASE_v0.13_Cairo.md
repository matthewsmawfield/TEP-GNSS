# v0.13 (Cairo): Monte Carlo validation and statistical rigor enhancements

## Summary

This release represents a major strengthening of the statistical validation for Paper 2 (Global Time Echoes: 25-Year Temporal Evolution). The orbital velocity correlation now benefits from 5 million Monte Carlo surrogate tests, yielding the strongest statistical evidence (5.1σ, p < 2×10⁻⁷) for temporal-gravitational coupling in the entire dataset.

## Key Changes

### 🔬 Monte Carlo Validation
- **Added 5M Monte Carlo surrogate validation** for orbital correlation (p < 2×10⁻⁷, 5.1σ)
- 0 out of 5,000,000 surrogates exceeded the observed correlation
- Confirms this is the strongest statistical signature in the dataset

### 📊 Enhanced Orbital Correlation
- **Updated orbital correlation from r = -0.864 to r = -0.888** with enhanced significance
- 95% CI: [-0.94, -0.81]
- ≈19% annual modulation refers to dimensionless anisotropy ratio (λ_EW/λ_NS), not clock frequency changes

### 🪐 Refined Planetary Event Analysis
- **Updated to 56 significant events** (≥2σ threshold)
- 25 events survive Bonferroni correction (ultra-conservative)
- Per-planet statistics:
  - Mercury: 34/80 detections (42.5%)
  - Jupiter: 8/23 detections (34.8%)
  - Saturn: 7/25 detections (28.0%)
  - Mars: 4/14 detections (28.6%)
  - Venus: 3/14 detections (21.4%)

### 📈 Statistical Framework Improvements
- Enhanced statistical reporting in `step_2_2_code_longspan.py`
- Added per-planet statistics table in Executive Summary
- Quantified convergence argument with joint probabilities

### 📝 Manuscript Clarity
- Clarified the "19% modulation" in abstract (geometric anisotropy ratio, not frequency)
- Added "Addressing the Convenient Shield Critique" callout box in §4.1.3
- Explained least-squares processing filter effects across independent software packages
- Enhanced physical interpretations throughout

### 🔄 Site Updates
- Updated all `site-code-longspan` components with improved scientific framing
- Consistency updates to Paper 1 including references
- Regenerated PDF to v0.13 Cairo edition
- Updated `zenodo.txt` with r = -0.888

## Scientific Impact

**Multi-Signature Convergence**: Even considering only the two strongest independent signatures (orbital + semiannual nutation) and assuming they share common physical drivers, the joint probability is p ≈ 2×10⁻²⁷ (>10σ). Combined with:
- 25-year temporal stability (this study)
- Paper 1's multi-center validation (R² = 0.92-0.97)
- Successful falsification tests

This provides strong empirical evidence consistent with systematic gravitational coupling in GNSS timing networks.

## Falsification Scorecard

| **Test** | **Prediction** | **Result** | **Status** |
|----------|---------------|-----------|------------|
| Orbital correlation | Detected | r = -0.888, 5.1σ | ✅ CONFIRMED |
| Semiannual nutation | Detected | R² = 0.904, p < 10⁻²⁰ | ✅ CONFIRMED |
| 18.6-year nutation | Detected | R² = 0.641, p < 10⁻⁸ | ✅ CONFIRMED |
| Planetary events | Detected | 56 significant (25 Bonferroni) | ✅ CONFIRMED |
| Solar rotation (27d) | Expected null | No detection | ✅ CONFIRMED |
| Lunar standstill | Expected null | No detection | ✅ CONFIRMED |

## Dataset Specifications

- **Analysis Center**: CODE (Center for Orbit Determination in Europe)
- **Time Span**: 25.3 years (2000-01-01 to 2025-04-14)
- **Total Pairs**: 165,198,849 station pairs
- **Unique Stations**: 814 GNSS stations worldwide
- **Temporal Resolution**: 30-second clock products

## Version Information

- **Version**: v0.13
- **Codename**: Cairo
- **Date**: 2025-11-23
- **Previous Version**: v0.9 (Cairo)
- **DOI**: [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141)

## Resources

- 🌐 **Website**: https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/
- 📄 **PDF**: [Smawfield_2025_GlobalTimeEchoes_25Year_v0.13_Cairo.pdf](https://zenodo.org/records/17660901/files/Smawfield_2025_GlobalTimeEchoes_25Year_v0.13_Cairo.pdf?download=1)
- 💻 **Code**: https://github.com/matthewsmawfield/TEP-GNSS
- 📚 **Theory Paper**: https://matthewsmawfield.github.io/TEP/ | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911)
- 📊 **Paper 1 (Multi-Center)**: https://matthewsmawfield.github.io/TEP-GNSS/

## Citation

```bibtex
@misc{smawfield2025globaltimeechoes,
  author = {Smawfield, Matthew Lukin},
  title = {Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks},
  year = {2025},
  version = {v0.13 (Cairo)},
  doi = {10.5281/zenodo.17517141},
  url = {https://matthewsmawfield.github.io/TEP-GNSS/code-longspan/}
}
```

## Next Steps

As outlined in the manuscript, critical next steps include:
1. Independent replication by other research groups
2. Raw carrier-phase data analysis (unprocessed GNSS measurements)
3. Multi-constellation testing (GLONASS, Galileo, BeiDou)
4. Extension to optical clocks for enhanced precision
5. Systematic investigation of remaining alternative hypotheses
