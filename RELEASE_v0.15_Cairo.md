# Release Notes: v0.15 (Cairo)
**Date:** 25 November 2025  
**Paper:** Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks

## Summary
This release enhances manuscript clarity and adds additional validation tests, including ecliptic plane controls that strengthen the CMB frame identification.

## Key Enhancements

### Null Hypothesis Testing
Added `step_2_6_null_control.py` for systematic null hypothesis testing.

### Ecliptic Control Tests
Added ecliptic-plane control directions (RA=90°/270°, Dec=0°) to discriminate CMB alignment from generic ecliptic detection. CMB explains 136× more variance than ecliptic controls, confirming specificity of the reference frame identification.

### Statistical Clarifications
- Explicit Bonferroni correction calculations for planetary event analysis
- Clarified detection rates and survival statistics
- Updated ionospheric control values (r=0.12-0.13, p>0.29)

### Seasonal Confound Analysis
Strengthened discrimination with three independent lines of evidence: CMB frame geometry, ionospheric null controls, and Monte Carlo phase validation.

### Processing Filter Framework
Enhanced discussion of why absence of GM/r² scaling is an expected feature of processed GNSS data, with falsifiable predictions for raw carrier-phase analysis.

### Editorial
- Standardised British spelling throughout
- Consistent colour scheme for callout boxes
- Minor corrections to section numbering

## Files Modified
- Manuscript components (sections 2-4)
- Analysis scripts (`step_2_5_dual_motion_geometry.py`)
- Version metadata files

---

*Full changelog available in git history.*
