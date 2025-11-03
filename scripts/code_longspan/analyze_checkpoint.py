#!/usr/bin/env python3
"""
Quick analysis of checkpoint data to see preliminary results
without waiting for the full analysis to complete.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import json

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
checkpoint_file = ROOT / "results/tmp/code_longspan/phase_stream_code.npz"

print("=" * 80)
print("PRELIMINARY ANALYSIS - Checkpoint Data")
print("=" * 80)

# Load checkpoint
print(f"\nLoading checkpoint: {checkpoint_file}")
data = np.load(checkpoint_file, allow_pickle=True)

# Extract data
agg_sum_coh = data['agg_sum_coh']
agg_sum_coh_sq = data['agg_sum_coh_sq']
agg_sum_dist = data['agg_sum_dist']
agg_count = data['agg_count']
processed_files = data['processed_files']
successful_files = int(data['successful_files'])
total_pairs_kept = int(data['total_pairs_kept'])

print(f"\n📊 Processing Status:")
print(f"   Files processed: {len(processed_files)}")
print(f"   Successful files: {successful_files}")
print(f"   Total pairs: {total_pairs_kept:,}")

# Compute bin statistics
num_bins = len(agg_count)
max_distance = 20000  # km
edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
bin_centers = np.sqrt(edges[:-1] * edges[1:])

# Calculate mean coherence and std per bin
valid_bins = agg_count > 0
mean_coh = np.zeros(num_bins)
std_coh = np.zeros(num_bins)
mean_dist = np.zeros(num_bins)

mean_coh[valid_bins] = agg_sum_coh[valid_bins] / agg_count[valid_bins]
mean_dist[valid_bins] = agg_sum_dist[valid_bins] / agg_count[valid_bins]

# Compute standard deviation
variance = np.zeros(num_bins)
variance[valid_bins] = (agg_sum_coh_sq[valid_bins] / agg_count[valid_bins]) - mean_coh[valid_bins]**2
variance[variance < 0] = 0  # Handle numerical errors
std_coh = np.sqrt(variance)

print(f"\n📈 Binning Statistics:")
print(f"   Total bins: {num_bins}")
print(f"   Bins with data: {valid_bins.sum()}")
print(f"   Distance range: {edges[0]:.1f} - {edges[-1]:.1f} km")

# Filter bins for fitting (require minimum count)
min_bin_count = 100
fit_mask = agg_count >= min_bin_count
distances_fit = mean_dist[fit_mask]
coherences_fit = mean_coh[fit_mask]
weights_fit = agg_count[fit_mask]

print(f"\n🔬 Fitting Statistics:")
print(f"   Bins used for fitting: {fit_mask.sum()}")
print(f"   Minimum pairs per bin: {min_bin_count}")
print(f"   Total pairs in fit: {weights_fit.sum():,}")

# Exponential correlation model
def correlation_model(r, amplitude, lambda_km, offset):
    return amplitude * np.exp(-r / lambda_km) + offset

# Fit the model
if len(distances_fit) >= 3:
    try:
        # Initial guess
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
        
        amplitude, lambda_km, offset = popt
        perr = np.sqrt(np.diag(pcov))
        
        print(f"\n✨ PRELIMINARY TEP RESULTS:")
        print(f"   Amplitude (A):        {amplitude:.4f} ± {perr[0]:.4f}")
        print(f"   Screening length (λ): {lambda_km:.1f} ± {perr[1]:.1f} km")
        print(f"   Offset (C₀):          {offset:.4f} ± {perr[2]:.4f}")

        # Compute R²
        residuals = coherences_fit - correlation_model(distances_fit, *popt)
        ss_res = np.sum(weights_fit * residuals**2)
        ss_tot = np.sum(weights_fit * (coherences_fit - np.average(coherences_fit, weights=weights_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"   R² (goodness of fit): {r_squared:.4f}")

        # Additional detailed statistics
        print(f"\n📊 DETAILED STATISTICS:")
        print(f"   Mean correlation:     {np.average(coherences_fit, weights=weights_fit):.4f}")
        print(f"   Max correlation:      {coherences_fit.max():.4f} at {distances_fit[coherences_fit.argmax()]:.0f} km")
        print(f"   Min correlation:      {coherences_fit.min():.4f} at {distances_fit[coherences_fit.argmin()]:.0f} km")
        print(f"   Correlation range:    {coherences_fit.max() - coherences_fit.min():.4f}")

        # Signal-to-noise analysis
        signal_strength = amplitude / perr[0]
        print(f"\n🔊 SIGNAL QUALITY:")
        print(f"   Amplitude SNR:        {signal_strength:.1f}σ")
        print(f"   Lambda precision:     {(perr[1]/lambda_km)*100:.1f}%")
        print(f"   Relative error:       {(perr[0]/amplitude)*100:.1f}% (amplitude)")

        # Distance bin analysis
        print(f"\n📏 DISTANCE BIN ANALYSIS:")
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            if agg_count[i] > 0:
                print(f"   Bin {i:2d}: {low:7.0f}-{high:7.0f} km | {agg_count[i]:8,} pairs | μ={mean_coh[i]:7.4f} σ={std_coh[i]:7.4f}")

        # Correlation decay analysis
        print(f"\n📉 CORRELATION DECAY:")
        # Calculate correlation at key distances
        for dist in [100, 500, 1000, 2000, 5000, 10000, 15000]:
            corr = correlation_model(dist, *popt)
            print(f"   At {dist:5,} km: {corr:7.4f}")

        # Estimate e-folding distance (where correlation drops to 1/e of amplitude)
        e_fold_corr = amplitude / np.e + offset
        print(f"\n   e-folding correlation: {e_fold_corr:.4f}")
        print(f"   e-folding distance:    {lambda_km:.0f} km (by definition)")

        # Statistical significance
        chi_squared = ss_res / (len(distances_fit) - 3)  # 3 parameters
        print(f"\n📈 STATISTICAL SIGNIFICANCE:")
        print(f"   χ² per DOF:           {chi_squared:.4f}")
        print(f"   Degrees of freedom:   {len(distances_fit) - 3}")
        print(f"   Weighted residuals:   {np.sqrt(ss_res):.4f}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Correlation vs Distance
        ax1.errorbar(
            bin_centers[valid_bins],
            mean_coh[valid_bins],
            yerr=std_coh[valid_bins] / np.sqrt(agg_count[valid_bins]),
            fmt='o',
            alpha=0.6,
            markersize=4,
            label='Binned data'
        )
        
        # Plot fit
        r_smooth = np.logspace(np.log10(50), np.log10(max_distance), 200)
        ax1.plot(r_smooth, correlation_model(r_smooth, *popt), 'r-', linewidth=2, 
                label=f'Fit: λ = {lambda_km:.0f} km')
        
        ax1.set_xlabel('Distance (km)', fontsize=12)
        ax1.set_ylabel('Phase-Coherent Correlation', fontsize=12)
        ax1.set_xscale('log')
        ax1.set_title(f'Preliminary TEP Analysis ({successful_files} files)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Pairs per bin
        ax2.bar(range(num_bins), agg_count, alpha=0.7, color='steelblue')
        ax2.axhline(y=min_bin_count, color='r', linestyle='--', label=f'Min threshold ({min_bin_count})')
        ax2.set_xlabel('Bin Index', fontsize=12)
        ax2.set_ylabel('Number of Pairs', fontsize=12)
        ax2.set_title('Pairs per Distance Bin', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_dir = ROOT / "results/figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "preliminary_tep_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 Figure saved: {output_file}")
        
        # Compute additional metrics
        print(f"\n🎯 TEP THEORY COMPARISON:")
        print(f"   Expected λ range:     1,000 - 10,000 km (TEP theory)")
        print(f"   Measured λ:           {lambda_km:.0f} ± {perr[1]:.0f} km")
        print(f"   Within theory range:  {'✅ YES' if 1000 <= lambda_km <= 10000 else '❌ NO'}")
        
        # Compute correlation at Earth's radius
        earth_radius = 6371  # km
        corr_at_earth_radius = correlation_model(earth_radius, *popt)
        print(f"\n🌍 EARTH-SCALE CORRELATIONS:")
        print(f"   Earth radius:         {earth_radius} km")
        print(f"   Correlation at R⊕:    {corr_at_earth_radius:.4f}")
        print(f"   λ/R⊕ ratio:           {lambda_km/earth_radius:.2f}")
        
        # Analyze residuals
        print(f"\n📊 RESIDUAL ANALYSIS:")
        mean_residual = np.average(residuals, weights=weights_fit)
        std_residual = np.sqrt(np.average(residuals**2, weights=weights_fit))
        max_residual = np.abs(residuals).max()
        max_residual_dist = distances_fit[np.abs(residuals).argmax()]
        print(f"   Mean residual:        {mean_residual:.6f}")
        print(f"   RMS residual:         {std_residual:.6f}")
        print(f"   Max residual:         {max_residual:.6f} at {max_residual_dist:.0f} km")
        
        # Compute correlation at specific percentiles
        print(f"\n📐 CORRELATION PERCENTILES:")
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            dist_p = np.percentile(distances_fit, p)
            corr_p = correlation_model(dist_p, *popt)
            print(f"   {p:2d}th percentile:     {dist_p:7.0f} km → correlation = {corr_p:.4f}")
        
        # Estimate where correlation crosses zero
        if offset < 0:
            zero_crossing = -lambda_km * np.log(-offset / amplitude)
            if zero_crossing > 0:
                print(f"\n🎯 ZERO-CROSSING ANALYSIS:")
                print(f"   Correlation = 0 at:   {zero_crossing:.0f} km")
                print(f"   (Beyond this, correlation becomes negative)")
        
        # Compute effective range (where correlation drops below 1%)
        threshold_corr = 0.01
        if amplitude > threshold_corr:
            effective_range = -lambda_km * np.log((threshold_corr - offset) / amplitude)
            print(f"\n📏 EFFECTIVE CORRELATION RANGE:")
            print(f"   Correlation > 1% up to: {effective_range:.0f} km")
            print(f"   (Practical detection limit)")
        
        # Progress estimation
        completion_pct = (successful_files / 9221) * 100
        print(f"\n⏱️  PROGRESS ESTIMATION:")
        print(f"   Completion:           {completion_pct:.1f}%")
        print(f"   Files remaining:      {9221 - successful_files:,}")
        print(f"   Expected final pairs: ~{int(total_pairs_kept * 9221 / successful_files):,}")
        
        # Save preliminary results
        results = {
            'files_processed': len(processed_files),
            'successful_files': successful_files,
            'total_pairs': int(total_pairs_kept),
            'bins_with_data': int(valid_bins.sum()),
            'bins_used_for_fit': int(fit_mask.sum()),
            'completion_percentage': float(completion_pct),
            'fit_parameters': {
                'amplitude': float(amplitude),
                'amplitude_error': float(perr[0]),
                'lambda_km': float(lambda_km),
                'lambda_error': float(perr[1]),
                'offset': float(offset),
                'offset_error': float(perr[2]),
                'r_squared': float(r_squared),
                'chi_squared_per_dof': float(chi_squared),
                'signal_to_noise': float(signal_strength)
            },
            'bin_statistics': {
                'mean_correlation': float(np.average(coherences_fit, weights=weights_fit)),
                'max_correlation': float(coherences_fit.max()),
                'min_correlation': float(coherences_fit.min()),
                'correlation_range': float(coherences_fit.max() - coherences_fit.min())
            }
        }
        
        results_file = ROOT / "results/outputs/preliminary_tep_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved: {results_file}")
        
        print("\n" + "=" * 80)
        print("✅ Preliminary analysis complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error fitting model: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n⚠️  Not enough bins for fitting (need at least 3, have {len(distances_fit)})")

print("\nNote: This is preliminary data. Final results will be more accurate.")
