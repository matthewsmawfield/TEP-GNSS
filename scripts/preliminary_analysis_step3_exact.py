#!/usr/bin/env python3
"""
TEP GNSS Analysis - Step 3 Exact Preliminary Analysis
====================================================

Analyzes partial Step 3 data from results/tmp directory using EXACTLY
the same methodology as Step 3 main analysis for direct comparison.

Usage: python scripts/preliminary_analysis_step3_exact.py [analysis_center]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.special import kv
import matplotlib.pyplot as plt

# Anchor to package root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the exact model functions from Step 3
def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model: C(r) = A*exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def gaussian_model(r, amplitude, lambda_km, offset):
    """Gaussian model: C(r) = A*exp(-(r/λ)²) + C₀"""
    return amplitude * np.exp(-(r / lambda_km)**2) + offset

def squared_exponential_model(r, amplitude, lambda_km, offset):
    """Squared exponential model: C(r) = A*exp(-0.5*(r/λ)²) + C₀"""
    return amplitude * np.exp(-0.5 * (r / lambda_km)**2) + offset

def power_law_model(r, amplitude, gamma, offset):
    """Power law model: C(r) = A*r^(-γ) + C₀"""
    return amplitude * np.power(r, -gamma) + offset

def power_law_with_cutoff_model(r, amplitude, gamma, cutoff, offset):
    """Power law with cutoff: C(r) = A*r^(-γ)*exp(-r/λ_c) + C₀"""
    return amplitude * np.power(r, -gamma) * np.exp(-r / cutoff) + offset

def matern_model(r, amplitude, lambda_km, offset):
    """Matérn correlation function with ν=1.5"""
    return matern_general_model(r, amplitude, lambda_km, offset, nu=1.5)

def matern_general_model(r, amplitude, lambda_km, offset, nu=1.5):
    """General Matérn correlation function"""
    sqrt_2nu_r_over_l = np.sqrt(2 * nu) * r / lambda_km
    sqrt_2nu_r_over_l = np.maximum(sqrt_2nu_r_over_l, 1e-10)  # Avoid division by zero
    
    # Use the modified Bessel function of the second kind
    bessel_term = kv(nu, sqrt_2nu_r_over_l)
    
    # Handle the case where r=0
    correlation = np.where(
        r == 0,
        1.0,
        (2**(1-nu) / np.math.gamma(nu)) * (sqrt_2nu_r_over_l)**nu * bessel_term
    )
    
    return amplitude * correlation + offset

def fit_model_with_aic_bic(distances, coherences, weights, model_func, p0, bounds, name):
    """Fit model and compute AIC/BIC - exact copy from Step 3"""
    try:
        # Weighted least squares fitting
        sigma = 1.0 / np.sqrt(weights)  # Convert weights to uncertainties
        popt, pcov = curve_fit(
            model_func, distances, coherences,
            p0=p0, bounds=bounds, sigma=sigma,
            absolute_sigma=False, maxfev=5000
        )
        
        # Predictions and residuals
        y_pred = model_func(distances, *popt)
        residuals = coherences - y_pred
        
        # R-squared
        ss_res = np.sum(weights * residuals**2)
        ss_tot = np.sum(weights * (coherences - np.average(coherences, weights=weights))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # AIC and BIC
        n = len(distances)
        k = len(popt)
        mse = ss_res / n
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)
        
        return {
            'name': name,
            'success': True,
            'params': popt,
            'covariance': pcov,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'mse': mse,
            'residuals': residuals,
            'predictions': y_pred
        }
    except Exception as e:
        return {
            'name': name,
            'success': False,
            'error': str(e),
            'r_squared': 0,
            'aic': np.inf,
            'bic': np.inf
        }

def analyze_preliminary_data_step3_exact(ac='code'):
    """Analyze preliminary pair data using EXACT Step 3 methodology"""
    print(f"\n{'='*60}")
    print(f"STEP 3 EXACT PRELIMINARY ANALYSIS - {ac.upper()}")
    print(f"{'='*60}")
    
    # Load pair data from tmp directory
    pair_dir = ROOT / 'results' / 'tmp'
    if not pair_dir.exists():
        print(f"ERROR: No tmp directory found at {pair_dir}")
        return None
    
    # Find all pair files for this analysis center
    all_pair_files = list(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
    if not all_pair_files:
        print(f"ERROR: No pair files found for {ac}")
        return None
    
    # Filter out empty or incomplete files
    pair_files = []
    for f in all_pair_files:
        try:
            if f.stat().st_size > 100:
                pd.read_csv(f, nrows=1)
                pair_files.append(f)
        except:
            continue
    
    print(f"Found {len(pair_files)} valid pair files to analyze (out of {len(all_pair_files)} total)")
    
    # Step 3 binning configuration (EXACT match)
    num_bins = int(os.getenv('TEP_BINS', '40'))
    max_distance = float(os.getenv('TEP_MAX_DISTANCE_KM', '13000'))
    min_bin_count = int(os.getenv('TEP_MIN_BIN_COUNT', '200'))
    
    # EXACT Step 3 binning: logarithmic from 50 to max_distance
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    print(f"Using Step 3 binning: {num_bins} bins from 50 to {max_distance} km")
    print(f"Minimum {min_bin_count} pairs required per bin")
    
    # Initialize aggregation arrays (EXACT Step 3 method)
    agg_sum_coh = np.zeros(num_bins)
    agg_sum_coh_sq = np.zeros(num_bins)
    agg_sum_dist = np.zeros(num_bins)
    agg_count = np.zeros(num_bins, dtype=int)
    
    # Load and aggregate data using EXACT Step 3 method
    total_pairs = 0
    successful_files = 0
    
    for i, pair_file in enumerate(pair_files):
        try:
            df = pd.read_csv(pair_file)
            if len(df) == 0:
                continue
                
            # Drop invalid data
            df = df.dropna(subset=['dist_km', 'plateau_phase']).copy()
            if len(df) == 0:
                continue
            
            # Compute coherence from phase (EXACT Step 3 method)
            df['coherence'] = np.cos(df['plateau_phase'])
            
            # Bin distances (EXACT Step 3 method)
            df['dist_bin'] = pd.cut(df['dist_km'], bins=edges, labels=False, include_lowest=True)
            
            # Aggregate by bin (EXACT Step 3 aggregation)
            for bin_idx in range(num_bins):
                bin_mask = df['dist_bin'] == bin_idx
                if bin_mask.sum() > 0:
                    bin_data = df[bin_mask]
                    agg_sum_coh[bin_idx] += bin_data['coherence'].sum()
                    agg_sum_coh_sq[bin_idx] += (bin_data['coherence'] ** 2).sum()
                    agg_sum_dist[bin_idx] += bin_data['dist_km'].sum()
                    agg_count[bin_idx] += len(bin_data)
            
            total_pairs += len(df)
            successful_files += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(pair_files)} files ({total_pairs:,} pairs)")
                
        except Exception as e:
            print(f"  WARNING: Failed to load {pair_file.name}: {e}")
            continue
    
    print(f"Total pairs processed: {total_pairs:,} from {successful_files} files")
    
    # Compute bin statistics (EXACT Step 3 method)
    mean_coherence = np.zeros(num_bins)
    std_coherence = np.zeros(num_bins)
    mean_distance = np.zeros(num_bins)
    
    for i in range(num_bins):
        if agg_count[i] > 0:
            mean_coherence[i] = agg_sum_coh[i] / agg_count[i]
            mean_distance[i] = agg_sum_dist[i] / agg_count[i]
            # Standard deviation: sqrt(E[X²] - E[X]²)
            if agg_count[i] > 1:
                variance = (agg_sum_coh_sq[i] / agg_count[i]) - mean_coherence[i]**2
                std_coherence[i] = np.sqrt(max(0, variance))
    
    # Extract data for fitting (EXACT Step 3 filtering)
    distances = []
    coherences = []
    weights = []
    
    print(f"\nPhase coherence vs distance (bins with ≥{min_bin_count} pairs):")
    print("Distance (km) | Mean Coherence | Count")
    print("----------------------------------------")
    
    for i in range(num_bins):
        if agg_count[i] >= min_bin_count:  # EXACT Step 3 filtering
            dist = mean_distance[i]
            coh = mean_coherence[i]
            count = agg_count[i]
            
            distances.append(dist)
            coherences.append(coh)
            weights.append(count)
            print(f"{dist:8.1f} | {coh:12.6f} | {count:6.0f}")
    
    if len(distances) < 5:
        print(f"ERROR: Insufficient bins for fitting ({len(distances)} < 5)")
        return None
    
    distances = np.array(distances)
    coherences = np.array(coherences)
    weights = np.array(weights)
    
    print(f"\nUsing {len(distances)} bins for model fitting")
    
    # EXACT Step 3 model comparison
    c_range = coherences.max() - coherences.min()
    
    models_to_fit = [
        {
            'func': correlation_model,
            'name': 'Exponential',
            'p0': [c_range, 3000, coherences.min()],
            'bounds': ([1e-10, 100, -1], [5, 20000, 1])
        },
        {
            'func': gaussian_model,
            'name': 'Gaussian',
            'p0': [c_range, 3000, coherences.min()],
            'bounds': ([1e-10, 100, -1], [5, 20000, 1])
        },
        {
            'func': squared_exponential_model,
            'name': 'Squared Exponential',
            'p0': [c_range, 3000, coherences.min()],
            'bounds': ([1e-10, 100, -1], [5, 20000, 1])
        },
        {
            'func': power_law_model,
            'name': 'Power Law',
            'p0': [c_range, 5, coherences.min()],
            'bounds': ([1e-10, 0.1, -1], [5, 10, 1])
        },
        {
            'func': power_law_with_cutoff_model,
            'name': 'Power Law w/ Cutoff',
            'p0': [c_range, 1.0, 5000, coherences.min()],
            'bounds': ([1e-10, 0.1, 1000, -1], [5, 10, 20000, 1])
        },
        {
            'func': matern_model,
            'name': 'Matérn (ν=1.5)',
            'p0': [c_range, 3000, coherences.min()],
            'bounds': ([1e-10, 100, -1], [5, 20000, 1])
        },
        {
            'func': lambda r, amp, l, off: matern_general_model(r, amp, l, off, nu=2.5),
            'name': 'Matérn (ν=2.5)',
            'p0': [c_range, 3000, coherences.min()],
            'bounds': ([1e-10, 100, -1], [5, 20000, 1])
        }
    ]
    
    # Fit all models (EXACT Step 3 method)
    model_results = []
    for model_def in models_to_fit:
        result = fit_model_with_aic_bic(
            distances, coherences, weights,
            model_def['func'], model_def['p0'], model_def['bounds'], model_def['name']
        )
        model_results.append(result)
    
    # Find best model by AIC (EXACT Step 3 method)
    successful_models = [r for r in model_results if r['success']]
    if not successful_models:
        print("ERROR: All model fits failed")
        return None
    
    best_model = min(successful_models, key=lambda x: x['aic'])
    
    # Display model comparison (EXACT Step 3 format)
    print("\nModel Comparison Results:")
    print("Model           | AIC      | BIC      | R²     | ΔAIC")
    print("----------------|----------|----------|--------|--------")
    for result in sorted(successful_models, key=lambda x: x['aic']):
        delta_aic = result['aic'] - best_model['aic']
        print(f"{result['name']:15s} | {result['aic']:8.2f} | {result['bic']:8.2f} | {result['r_squared']:6.3f} | {delta_aic:6.2f}")
    
    # Get exponential model results for TEP comparison
    exp_result = next((r for r in model_results if r['name'] == 'Exponential' and r['success']), None)
    if not exp_result:
        print("ERROR: Exponential model failed")
        return None
    
    # Extract results (EXACT Step 3 format)
    best_amplitude, best_lambda_km, best_offset = best_model['params']
    best_param_errors = np.sqrt(np.diag(best_model['covariance']))
    best_r_squared = best_model['r_squared']
    
    exp_amplitude, exp_lambda_km, exp_offset = exp_result['params']
    exp_param_errors = np.sqrt(np.diag(exp_result['covariance']))
    exp_r_squared = exp_result['r_squared']
    
    # Display results (EXACT Step 3 format)
    print(f"\nBEST MODEL FIT RESULTS:")
    print(f"  Best Model: {best_model['name']} (AIC winner)")
    print(f"  Amplitude (A): {best_amplitude:.6f} ± {best_param_errors[0]:.6f}")
    print(f"  Correlation Length (λ): {best_lambda_km:.1f} ± {best_param_errors[1]:.1f} km")
    print(f"  Offset (C₀): {best_offset:.6f} ± {best_param_errors[2]:.6f}")
    print(f"  R-squared: {best_r_squared:.4f}")
    
    if best_model['name'] != 'Exponential':
        print(f"\nEXPONENTIAL MODEL (TEP) RESULTS:")
        print(f"  Amplitude (A): {exp_amplitude:.6f} ± {exp_param_errors[0]:.6f}")
        print(f"  Correlation Length (λ): {exp_lambda_km:.1f} ± {exp_param_errors[1]:.1f} km")
        print(f"  Offset (C₀): {exp_offset:.6f} ± {exp_param_errors[2]:.6f}")
        print(f"  R-squared: {exp_r_squared:.4f}")
    
    # TEP assessment (EXACT Step 3 logic)
    tep_consistent = (1000 < exp_lambda_km < 10000) and (exp_r_squared > 0.3)
    signal_strength = 'Strong' if exp_r_squared > 0.5 else 'Moderate' if exp_r_squared > 0.3 else 'Weak'
    
    if tep_consistent:
        print(f"\n✅ TEP-consistent signal detected")
        print(f"  Exponential model: λ = {exp_lambda_km:.0f} km is in TEP range")
        print(f"  R² = {exp_r_squared:.3f} indicates {signal_strength.lower()} correlation structure")
        print(f"  Phase-coherent analysis supports TEP predictions")
        if best_model['name'] != 'Exponential':
            print(f"  Note: {best_model['name']} model fits better (ΔAIC = {best_model['aic'] - exp_result['aic']:.2f})")
    else:
        print(f"\n⚠️  Signal detected but not clearly TEP-consistent")
    
    # Create results structure (EXACT Step 3 format)
    results = {
        'analysis_center': ac.upper(),
        'method': {
            'type': 'phase_alignment_index',
            'formula': 'cos(phase(CSD))'
        },
        'data_summary': {
            'total_pairs': int(total_pairs),
            'files_processed': int(successful_files),
            'files_total': len(pair_files),
            'bins_used': len(distances),
            'distance_range_km': [float(distances.min()), float(distances.max())],
            'coherence_range': [float(coherences.min()), float(coherences.max())],
            'mean_coherence': float(coherences.mean())
        },
        'model_comparison': {
            'models_tested': [r['name'] for r in successful_models],
            'best_model_aic': best_model['name'],
            'model_results': [
                {
                    'name': r['name'],
                    'aic': float(r['aic']),
                    'bic': float(r['bic']),
                    'r_squared': float(r['r_squared']),
                    'delta_aic': float(r['aic'] - best_model['aic'])
                } for r in successful_models
            ]
        },
        'best_fit': {
            'model_name': best_model['name'],
            'amplitude': float(best_amplitude),
            'amplitude_error': float(best_param_errors[0]),
            'lambda_km': float(best_lambda_km),
            'lambda_error': float(best_param_errors[1]),
            'offset': float(best_offset),
            'offset_error': float(best_param_errors[2]),
            'r_squared': float(best_r_squared),
            'n_bins': len(distances)
        },
        'exponential_fit': {
            'model': 'C(r) = A * exp(-r/lambda) + C0',
            'amplitude': float(exp_amplitude),
            'amplitude_error': float(exp_param_errors[0]),
            'lambda_km': float(exp_lambda_km),
            'lambda_error': float(exp_param_errors[1]),
            'offset': float(exp_offset),
            'offset_error': float(exp_param_errors[2]),
            'r_squared': float(exp_r_squared),
            'n_bins': len(distances)
        },
        'tep_interpretation': {
            'tep_consistent': bool(tep_consistent),
            'correlation_length_assessment': 'TEP-consistent' if 1000 < exp_lambda_km < 10000 else 'Outside TEP range',
            'signal_strength': signal_strength,
            'best_model_vs_exponential': f'Best model: {best_model["name"]} (ΔAIC = {best_model["aic"] - exp_result["aic"]:.2f})'
        }
    }
    
    # Save results
    output_file = ROOT / f"results/preliminary_step3_exact_{ac}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Create plot matching Step 3 style
    plt.figure(figsize=(12, 8))
    
    # Plot data with error bars
    plt.errorbar(distances, coherences, yerr=std_coherence[agg_count >= min_bin_count]/np.sqrt(weights),
                fmt='o', alpha=0.7, capsize=3, label='Binned Data')
    
    # Plot best fit
    x_fit = np.logspace(np.log10(distances.min()), np.log10(distances.max()), 1000)
    if best_model['name'] == 'Exponential':
        y_fit = correlation_model(x_fit, *best_model['params'])
    else:
        y_fit = best_model['func'](x_fit, *best_model['params'])
    
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
            label=f'{best_model["name"]}: λ={best_lambda_km:.0f}km, R²={best_r_squared:.3f}')
    
    # Plot exponential fit if different
    if best_model['name'] != 'Exponential':
        y_exp = correlation_model(x_fit, *exp_result['params'])
        plt.plot(x_fit, y_exp, 'g--', linewidth=2,
                label=f'Exponential: λ={exp_lambda_km:.0f}km, R²={exp_r_squared:.3f}')
    
    plt.xscale('log')
    plt.xlabel('Distance (km)')
    plt.ylabel('Phase Coherence')
    plt.title(f'Step 3 Exact Preliminary Analysis - {ac.upper()}\n'
             f'({successful_files} files, {total_pairs:,} pairs, {len(distances)} bins)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = ROOT / f"results/preliminary_step3_exact_plot_{ac}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    return results

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Step 3 Exact Preliminary Analysis')
    parser.add_argument('ac', nargs='?', default='code', 
                       choices=['code', 'igs_combined', 'esa_final'],
                       help='Analysis center to analyze')
    args = parser.parse_args()
    
    results = analyze_preliminary_data_step3_exact(args.ac)
    if results:
        print(f"\n{'='*60}")
        print("STEP 3 EXACT PRELIMINARY ANALYSIS COMPLETE")
        print(f"{'='*60}")
    else:
        print("STEP 3 EXACT PRELIMINARY ANALYSIS FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main()
