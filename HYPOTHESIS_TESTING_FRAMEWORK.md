# Hypothesis Testing Framework for Multi-Band Frequency Analysis

## Date: 2025-10-03
## Purpose: Transparent Data-Driven Hypothesis Testing

## Overview

The control band analysis now implements a **transparent hypothesis testing framework** that presents objective metrics for evaluating three competing physical explanations for observed GNSS timing correlations.

**Critical principle**: The analysis presents FACTS, not interpretations. All hypothesis test results are objective metrics that enable independent scientific evaluation.

## Three Physical Hypotheses

### Hypothesis A: Tidal-Dominated TEP Coupling

**Physical Basis**:
- Moon/Sun gravitational forces create large, periodic deformations in Earth's atmosphere and crust
- IF φ-field couples universally to matter
- THEN these tidal deformations should strongly modulate the φ-field
- GNSS clocks embedded in this environment should show **enhanced correlations at tidal frequencies**

**Expected Pattern**:
```
Tidal Bands (10-15, 20-30 μHz):    R² > 0.97  (STRONGEST - tidal forcing maximal)
Broad TEP Band (10-500 μHz):       R² ~ 0.90-0.95  (diluted by non-tidal frequencies)
Control Bands (>1000 μHz):         R² < 0.3  (no physical coupling)
```

**Key Prediction**: Narrow tidal bands should be STRONGER than broad TEP band because:
- Tidal frequencies have maximum gravitational forcing
- Broad TEP band averages over tidal + non-tidal contributions
- If tidal coupling dominates, signal should concentrate at tidal frequencies

### Hypothesis B: Broadband TEP with Systematic Effects

**Physical Basis**:
- TEP signal distributed across broad frequency range (10-500 μHz)
- Tidal coupling is NOT the dominant mechanism
- Systematic instrumental effects contribute across all frequencies
- Narrow tidal bands sample only a subset of the total signal

**Expected Pattern**:
```
Broad TEP Band (10-500 μHz):       R² ~ 0.95-0.97  (integrates full signal)
Tidal Bands (10-15, 20-30 μHz):    R² ~ 0.90-0.97  (narrow frequency subset)
Control Bands (>1000 μHz):         R² ~ 0.5-0.7  (systematic effects only)
```

**Key Prediction**: Broad TEP band shows comparable or stronger signal because:
- Signal distributed across 490 μHz range
- Tidal bands (5-10 μHz) capture only narrow slice
- Control bands show moderate correlations from systematics

### Hypothesis C: Systematic Effects Dominate

**Physical Basis**:
- Correlations arise primarily from instrumental or processing artifacts
- True physical coupling is weak or absent
- Systematic effects don't depend on specific frequency bands

**Expected Pattern**:
```
All Bands:                          R² ~ 0.8-0.9  (similar across all frequencies)
Minimal Differentiation:            Range < 0.2  (no frequency structure)
```

**Key Prediction**: All bands show similar correlations regardless of physical significance.

## Objective Metrics Computed

### Primary Metrics

**Raw R² Values**:
- Direct correlation strength for each band
- No normalization or adjustment
- Transparent measure of empirical correlation

**Bandwidth (μHz)**:
- Documents frequency range of each band
- TEP: 490 μHz (10-500 μHz)
- Tidal Diurnal: 5 μHz (10-15 μHz)
- Tidal Semidiurnal: 10 μHz (20-30 μHz)
- Control: 1000 μHz (1000-2000 μHz)

**Bandwidth-Normalized R²**:
- R² divided by bandwidth in μHz
- Assesses "signal density" per unit frequency
- Addresses question: Do narrow bands show stronger signal per μHz?

### Hypothesis A Metrics

```json
{
  "tidal_exceeds_tep": boolean,
  "tidal_minus_tep": float,  // Δ in R²
  "tidal_vs_tep_ratio": float,  // Ratio of R² values
  "pattern_consistent": boolean  // Matches A's expected pattern
}
```

### Hypothesis B Metrics

```json
{
  "tep_exceeds_or_equals_tidal": boolean,
  "tep_minus_tidal": float,  // Δ in R²
  "control_moderate": boolean,  // 0.5 ≤ R² ≤ 0.7
  "pattern_consistent": boolean  // Matches B's expected pattern
}
```

### Hypothesis C Metrics

```json
{
  "tep_tidal_difference": float,  // |TEP - Tidal|
  "tep_control_difference": float,  // |TEP - Control|
  "r_squared_range": float,  // max - min across bands
  "all_bands_similar": boolean,  // Differences < threshold
  "pattern_consistent": boolean  // Matches C's expected pattern
}
```

## What the Metrics Tell You

### Example Interpretation (Objective Facts Only)

**Observed Data**:
```
TEP Band:            R² = 0.966
Tidal Semidiurnal:   R² = 0.970
Tidal Diurnal:       R² = 0.952
Control:             R² = 0.627
```

**Hypothesis A Metrics**:
```
tidal_exceeds_tep: TRUE  (0.970 > 0.966)
tidal_minus_tep: +0.004
pattern_consistent: FALSE  (control R² too high: 0.627 > 0.3)
```
→ Tidal bands slightly exceed TEP, BUT control band shows substantial correlation inconsistent with pure tidal coupling.

**Hypothesis B Metrics**:
```
tep_exceeds_or_equals_tidal: TRUE  (0.966 ≥ 0.952, though semidiurnal slightly higher)
tep_minus_tidal: -0.004 (vs max tidal)
control_moderate: TRUE  (0.5 ≤ 0.627 ≤ 0.7)
pattern_consistent: TRUE
```
→ Broad TEP band comparable to tidal, control moderate - consistent with B.

**Hypothesis C Metrics**:
```
tep_tidal_difference: 0.004
tep_control_difference: 0.339
r_squared_range: 0.343  (0.970 - 0.627)
all_bands_similar: FALSE
```
→ TEP and tidal very similar, but control substantially weaker - NOT consistent with uniform systematics.

## Bandwidth Normalization Insight

**Question**: Should we normalize by bandwidth?

**Considerations**:

**For normalization**:
- Tidal bands have 50-100× narrower bandwidth than TEP band
- "Signal per μHz" might be more meaningful comparison
- Addresses whether narrow bands show concentrated signal

**Against normalization**:
- TEP theory might predict broadband coupling across full 10-500 μHz range
- Integrating over bandwidth could be physically meaningful
- Narrow bands might not independently test different physics

**Analysis provides both**:
- Raw R²: Direct empirical correlation
- Normalized R²: Signal density per μHz
- Let the physics guide interpretation

## Output Locations

### JSON Output

Main results file: `results/outputs/step_3_6_multiband_{ac}.json`

Contains:
```json
{
  "physical_hypotheses": {
    "hypothesis_a_tidal_dominated": {
      "description": "...",
      "prediction": "...",
      "physical_basis": "..."
    },
    // ... B and C
  },
  "hypothesis_test_results": {
    "raw_r_squared": {...},
    "bandwidth_microhz": {...},
    "bandwidth_normalized_r_squared": {...},
    "hypothesis_a_tidal_dominated": {...},
    "hypothesis_b_broadband_tep": {...},
    "hypothesis_c_systematics_dominate": {...}
  }
}
```

### Console Output

During analysis, prints:
```
HYPOTHESIS TEST RESULTS:
  (Objective metrics - no interpretation provided)

  Hypothesis A (Tidal-Dominated Coupling):
    Prediction: Tidal > TEP > Control
    Tidal exceeds TEP: True/False
    Tidal - TEP: +X.XXXX
    Pattern matches prediction: True/False

  Hypothesis B (Broadband TEP + Systematics):
    Prediction: TEP ≥ Tidal, Control moderate
    TEP ≥ Tidal: True/False
    TEP - Tidal: +X.XXXX
    Pattern matches prediction: True/False

  Hypothesis C (Systematic Effects Dominate):
    Prediction: All bands similar
    R² range across bands: X.XXXX
    All bands similar: True/False
    Pattern matches prediction: True/False

  Bandwidth Normalization:
    (R² per μHz - assesses signal density)
    tep_band: R²=X.XXXX → X.XXXXXX per μHz
    tidal_diurnal: R²=X.XXXX → X.XXXXXX per μHz
    ...
```

## Critical Insight from Your Analysis

**Your Physical Reasoning**:
> If gravity affects tides universally, and TEP couples to matter, then tidal deformations should strongly modulate φ-field. Tidal bands SHOULD be stronger than TEP band.

**Framework Response**:
The hypothesis testing framework now:
1. **Explicitly states** this physical prediction (Hypothesis A)
2. **Computes objective metrics** to test it
3. **Presents facts without interpretation**
4. **Enables you to evaluate** whether data supports this physics

**Current IGS Results Suggest**:
```
Tidal Semidiurnal: R² = 0.970
TEP Band:          R² = 0.966
Control:           R² = 0.627
```
- Tidal bands ARE slightly stronger (supports Hypothesis A)
- BUT control band shows substantial correlation (challenges pure tidal interpretation)
- TEP band very close to tidal (also consistent with Hypothesis B)

**Pattern is ambiguous** - requires careful physical reasoning to interpret.

## How to Use This Framework

1. **Run analysis**: `python step_3_6_control_band_analysis.py igs_combined`

2. **Examine JSON output**: Load `step_3_6_multiband_*.json`

3. **Check hypothesis metrics**: Look at `hypothesis_test_results` section

4. **Consider bandwidth effects**: Compare raw vs normalized R²

5. **Evaluate pattern consistency**: Does data match any hypothesis?

6. **Form independent conclusions**: Framework provides facts, you interpret

## Transparency Principles

1. **No hidden assumptions**: All metrics explicitly documented
2. **No premature interpretation**: Presents facts, not conclusions
3. **Multiple perspectives**: Tests competing hypotheses objectively
4. **Bandwidth transparency**: Shows both raw and normalized metrics
5. **Pattern matching**: Boolean flags for objective pattern consistency

## Scientific Value

This framework enables:

**Rigorous Hypothesis Testing**:
- Multiple competing explanations tested simultaneously
- Objective metrics for each prediction
- Transparent comparison of data vs expectations

**Independent Evaluation**:
- All data available for your own interpretation
- No AI-imposed conclusions
- Complete methodology documentation

**Publication-Ready Analysis**:
- Presents data scientifically
- Documents competing hypotheses
- Shows which patterns are/aren't consistent with observations

## Next Steps

1. **Examine current results**: Do they support A, B, C, or something else?

2. **Consider additional bands**: Would intermediate frequencies help distinguish hypotheses?

3. **Physical interpretation**: What does YOUR physical reasoning say about the pattern?

4. **Alternative hypotheses**: Are there other physical scenarios to test?

5. **Systematic modeling**: Can we quantify and subtract systematic contributions?

The framework is a **tool for transparent science**, not an oracle providing answers. You bring the physics, it provides the metrics.

