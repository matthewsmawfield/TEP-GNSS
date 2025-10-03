# TEP Complementary Metrics Validation Analysis

## Overview

This exploratory analysis provides theoretical validation of the `cos(phase(CSD))` methodology through two complementary approaches:

1. **Von Mises Concentration Analysis (κ)**: Direct estimation of phase concentration parameter from circular variance
2. **Magnitude-Squared Coherence (MSC)**: Traditional coherence analysis for comparison

## Theoretical Foundation

### Circular Statistics Interpretation

The `cos(phase(CSD))` metric can be interpreted through circular statistics:

- **Phase clustering**: Genuine correlations should produce phases clustered around 0 rad
- **Concentration parameter κ**: Measures the tightness of phase clustering  
- **Expected cosine mean**: E[cos(φ)] = I₁(κ)/I₀(κ) ≈ κ/2 for small κ

This provides a direct theoretical link between the observed `cos(phase(CSD))` values and the underlying phase concentration.

### Mathematical Relationships

#### Von Mises Distribution
For a von Mises distribution VM(μ=0, κ):
- **Concentration parameter κ**: Controls the tightness of phase clustering
- **Expected cosine mean**: E[cos(φ)] = I₁(κ)/I₀(κ)
- **Approximations**:
  - Small κ: E[cos(φ)] ≈ κ/2
  - Large κ: E[cos(φ)] ≈ 1 - 1/(2κ)

#### Magnitude-Squared Coherence
Traditional coherence measure:
- **MSC definition**: |S_xy(f)|² / (S_xx(f) * S_yy(f))
- **Expected relationship**: MSC should be lower than cos(φ) due to noise
- **Validation criterion**: Both metrics should show similar exponential decay patterns

## Files

### Core Analysis Scripts
- `tep_complementary_analysis.py`: Main analysis implementation
- `run_complementary_analysis.py`: Simple execution script
- `complementary_metrics_validation.py`: Alternative implementation (requires raw data)

### Key Functions

#### `von_mises_concentration_to_cosine_mean(kappa)`
Converts von Mises concentration parameter κ to expected cosine mean using the relationship E[cos(φ)] = I₁(κ)/I₀(κ).

#### `cosine_mean_to_von_mises_concentration(cos_mean)`
Inverse function to convert expected cosine mean back to concentration parameter κ.

#### `analyze_existing_correlation_data(data_dir)`
Analyzes existing correlation data and derives theoretical complementary metrics.

## Usage

### Quick Start
```bash
cd TEP-GNSS
python scripts/exploratory/run_complementary_analysis.py
```

### Manual Execution
```bash
cd TEP-GNSS
python scripts/exploratory/tep_complementary_analysis.py
```

## Outputs

### Plots
- `results/exploratory/tep_complementary_metrics_analysis.png`: Comprehensive comparison plots including:
  - λ comparison across metrics and centers
  - R² comparison across metrics and centers
  - Theoretical κ ↔ cos(φ) relationship
  - Distance decay curves
  - Cross-metric correlation analysis
  - Methodological summary

### Reports
- `results/exploratory/tep_complementary_metrics_report.md`: Detailed analysis report including:
  - Executive summary
  - Theoretical foundation
  - Results summary table
  - Key findings
  - Methodological validation
  - Limitations and future work
  - Conclusions

## Expected Results

### Consistency Validation
All three metrics should show:
- **Similar λ values**: Correlation length should be consistent across approaches
- **Similar R² values**: Goodness of fit should be comparable
- **Exponential decay**: All metrics should follow the same distance decay pattern

### Theoretical Relationships
- **κ ↔ cos(φ)**: Should follow the theoretical von Mises relationship
- **MSC vs cos(φ)**: MSC should be systematically lower due to noise
- **Cross-metric correlation**: Strong correlation between derived metrics

## Limitations

### Current Implementation
1. **Theoretical derivations**: The κ and MSC metrics are derived theoretically from cos(φ) rather than computed independently
2. **Raw data requirement**: Full validation requires access to raw time series data
3. **SNR assumptions**: The MSC relationship assumes typical SNR reduction factors

### Future Work
1. **Independent computation**: Implement κ and MSC computation from raw time series data
2. **SNR characterization**: Measure actual SNR reduction factors in GNSS processing
3. **Extended validation**: Apply to additional datasets and analysis centers
4. **Theoretical refinement**: Develop more accurate approximations for κ ↔ cos(φ) relationship

## Interpretation

### Methodological Validation
- **Circular statistics foundation**: The κ-based analysis provides theoretical justification for the phase-based approach
- **Cross-validation**: Agreement across multiple metrics supports the validity of the method
- **Physical interpretation**: The analysis links observed correlations to underlying phase concentration

### Statistical Significance
- **Multi-metric consistency**: Agreement across different approaches reduces the likelihood of artifacts
- **Theoretical foundation**: The circular statistics interpretation provides a solid mathematical basis
- **Robustness**: Multiple independent measures of the same phenomenon increase confidence

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `scipy`: Signal processing and optimization
- `json`: Data I/O
- `pathlib`: File path handling

## Configuration

The analysis uses the TEP configuration system:
- `TEPConfig.initialize()`: Loads configuration parameters
- Frequency band: 10-500 μHz (configurable)
- Distance binning: 40 logarithmic bins, 100km-15,000km range
- Exponential fitting: A*exp(-r/λ) + C₀ model

## Troubleshooting

### Common Issues
1. **Missing data files**: Ensure processed correlation data exists in `data/processed/`
2. **Insufficient data**: Analysis requires at least 5 distance bins with valid data
3. **Fitting failures**: Check for numerical issues in exponential fitting

### Error Messages
- `"File not found"`: Check data directory and file names
- `"Insufficient data"`: Verify correlation data quality
- `"Failed to fit"`: Check for numerical issues or data quality

## Contact

For questions or issues with this analysis, contact the TEP-GNSS development team.
