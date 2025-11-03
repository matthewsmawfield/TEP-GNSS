#!/usr/bin/env python3
"""
Check the actual precision of the mean residual to verify it's not exactly zero.
"""

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
checkpoint_file = ROOT / "results/tmp/code_longspan/phase_stream_code.npz"

# Load checkpoint
data = np.load(checkpoint_file, allow_pickle=True)
agg_sum_coh = data['agg_sum_coh']
agg_sum_coh_sq = data['agg_sum_coh_sq']
agg_sum_dist = data['agg_sum_dist']
agg_count = data['agg_count']

# Compute bin statistics
num_bins = len(agg_count)
max_distance = 20000
edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)

valid_bins = agg_count > 0
mean_coh = np.zeros(num_bins)
mean_dist = np.zeros(num_bins)

mean_coh[valid_bins] = agg_sum_coh[valid_bins] / agg_count[valid_bins]
mean_dist[valid_bins] = agg_sum_dist[valid_bins] / agg_count[valid_bins]

# Filter for fitting
min_bin_count = 100
fit_mask = agg_count >= min_bin_count
distances_fit = mean_dist[fit_mask]
coherences_fit = mean_coh[fit_mask]
weights_fit = agg_count[fit_mask]

# Correlation model
def correlation_model(r, amplitude, lambda_km, offset):
    return amplitude * np.exp(-r / lambda_km) + offset

# Fit
p0 = [0.1, 5000, 0.0]
bounds = ([0, 100, -1], [1, 20000, 1])
popt, pcov = curve_fit(
    correlation_model,
    distances_fit,
    coherences_fit,
    p0=p0,
    bounds=bounds,
    sigma=1/np.sqrt(weights_fit),
    absolute_sigma=False,
    maxfev=10000
)

# Compute residuals
residuals = coherences_fit - correlation_model(distances_fit, *popt)

# Compute mean with full precision
mean_residual_unweighted = np.mean(residuals)
mean_residual_weighted = np.average(residuals, weights=weights_fit)

print("=" * 80)
print("RESIDUAL PRECISION CHECK")
print("=" * 80)
print(f"\nMean residual (unweighted):     {mean_residual_unweighted:.15e}")
print(f"Mean residual (weighted):       {mean_residual_weighted:.15e}")
print(f"\nRMS residual:                   {np.sqrt(np.average(residuals**2, weights=weights_fit)):.15e}")
print(f"Standard deviation:             {np.std(residuals):.15e}")
print(f"\nMin residual:                   {residuals.min():.6f}")
print(f"Max residual:                   {residuals.max():.6f}")
print(f"\nNumber of positive residuals:   {(residuals > 0).sum()}")
print(f"Number of negative residuals:   {(residuals < 0).sum()}")
print(f"Number of zero residuals:       {(residuals == 0).sum()}")

# Check if mean is truly zero or just very small
if abs(mean_residual_weighted) < 1e-10:
    print(f"\n✅ Mean residual is effectively zero (< 10⁻¹⁰)")
    print(f"   This is EXPECTED from least-squares fitting with an offset parameter.")
else:
    print(f"\n⚠️  Mean residual is {mean_residual_weighted:.3e}")
    print(f"   This might indicate a systematic bias.")

print("\n" + "=" * 80)
