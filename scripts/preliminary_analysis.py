#!/usr/bin/env python3
"""
TEP GNSS Analysis - Preliminary Analysis Script
==============================================

Analyzes partial Step 3 data from results/tmp directory to provide
quick preliminary results while main processing continues.

Usage: python scripts/preliminary_analysis.py [analysis_center]
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Anchor to package root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def exponential_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model: C(r) = A*exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def analyze_preliminary_data(ac='code'):
    """Analyze preliminary pair data for given analysis center"""
    print(f"\n{'='*60}")
    print(f"PRELIMINARY TEP ANALYSIS - {ac.upper()}")
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
            if f.stat().st_size > 100:  # At least 100 bytes (header + some data)
                # Quick check if file is readable
                pd.read_csv(f, nrows=1)
                pair_files.append(f)
        except:
            continue
    
    print(f"Found {len(pair_files)} valid pair files to analyze (out of {len(all_pair_files)} total)")
    
    # Load and combine all pair data
    all_pairs = []
    total_pairs = 0
    
    for i, pair_file in enumerate(pair_files):
        try:
            df = pd.read_csv(pair_file)
            if len(df) > 0:
                all_pairs.append(df)
                total_pairs += len(df)
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i+1}/{len(pair_files)} files ({total_pairs:,} pairs)")
        except Exception as e:
            print(f"  WARNING: Failed to load {pair_file.name}: {e}")
            continue
    
    if not all_pairs:
        print("ERROR: No valid pair data found")
        return None
    
    # Combine all data
    df_all = pd.concat(all_pairs, ignore_index=True)
    df_all = df_all.dropna(subset=['dist_km', 'plateau_phase']).copy()
    
    print(f"Total pairs: {len(df_all):,}")
    print(f"Distance range: {df_all['dist_km'].min():.1f} - {df_all['dist_km'].max():.1f} km")
    
    # Compute coherence from phase
    df_all['coherence'] = np.cos(df_all['plateau_phase'])
    
    # Set up distance binning
    num_bins = 40
    max_distance = 13000
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    
    # Bin the data
    df_all['dist_bin'] = pd.cut(df_all['dist_km'], bins=edges)
    binned = df_all.groupby('dist_bin', observed=True).agg({
        'coherence': ['mean', 'std', 'count'],
        'dist_km': 'mean'
    }).reset_index()
    
    # Flatten column names
    binned.columns = ['dist_bin', 'coherence_mean', 'coherence_std', 'count', 'dist_mean']
    binned = binned.dropna()
    
    # Filter bins with sufficient data
    min_count = 200
    valid_bins = binned[binned['count'] >= min_count].copy()
    
    print(f"Valid bins: {len(valid_bins)}/{len(binned)} (≥{min_count} pairs each)")
    
    if len(valid_bins) < 5:
        print("WARNING: Too few valid bins for reliable fitting")
        return None
    
    # Fit exponential model
    try:
        # Initial guess
        p0 = [0.1, 3000, 0.0]  # amplitude, lambda_km, offset
        
        # Fit with weights
        weights = np.sqrt(valid_bins['count'])
        popt, pcov = curve_fit(
            exponential_model, 
            valid_bins['dist_mean'], 
            valid_bins['coherence_mean'],
            p0=p0,
            sigma=valid_bins['coherence_std'] / weights,
            absolute_sigma=False,
            maxfev=5000
        )
        
        amplitude, lambda_km, offset = popt
        
        # Calculate R²
        y_pred = exponential_model(valid_bins['dist_mean'], *popt)
        ss_res = np.sum((valid_bins['coherence_mean'] - y_pred) ** 2)
        ss_tot = np.sum((valid_bins['coherence_mean'] - np.mean(valid_bins['coherence_mean'])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        
        print(f"\nPRELIMINARY RESULTS:")
        print(f"  Amplitude: {amplitude:.6f} ± {param_errors[0]:.6f}")
        print(f"  Lambda: {lambda_km:.1f} ± {param_errors[1]:.1f} km")
        print(f"  Offset: {offset:.6f} ± {param_errors[2]:.6f}")
        print(f"  R²: {r_squared:.4f}")
        
        # TEP assessment
        tep_consistent = (1000 <= lambda_km <= 10000) and (r_squared > 0.3)
        print(f"  TEP Consistent: {'YES' if tep_consistent else 'NO'}")
        
        results = {
            'analysis_center': ac,
            'files_processed': len(pair_files),
            'total_pairs': int(len(df_all)),
            'valid_bins': int(len(valid_bins)),
            'exponential_fit': {
                'amplitude': float(amplitude),
                'lambda_km': float(lambda_km),
                'offset': float(offset),
                'amplitude_error': float(param_errors[0]),
                'lambda_error': float(param_errors[1]),
                'offset_error': float(param_errors[2]),
                'r_squared': float(r_squared)
            },
            'tep_assessment': {
                'is_consistent': bool(tep_consistent),
                'lambda_in_range': bool(1000 <= lambda_km <= 10000),
                'r_squared_sufficient': bool(r_squared > 0.3)
            },
            'data_summary': {
                'distance_range_km': [float(df_all['dist_km'].min()), float(df_all['dist_km'].max())],
                'coherence_range': [float(df_all['coherence'].min()), float(df_all['coherence'].max())],
                'mean_coherence': float(df_all['coherence'].mean())
            }
        }
        
        # Save results
        output_file = ROOT / f"results/preliminary_analysis_{ac}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Create quick plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(valid_bins['dist_mean'], valid_bins['coherence_mean'], 
                    yerr=valid_bins['coherence_std']/np.sqrt(valid_bins['count']),
                    fmt='o', alpha=0.7, label='Data')
        
        # Plot fit
        x_fit = np.logspace(np.log10(50), np.log10(max_distance), 1000)
        y_fit = exponential_model(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label=f'Fit: λ={lambda_km:.0f}km, R²={r_squared:.3f}')
        
        plt.xscale('log')
        plt.xlabel('Distance (km)')
        plt.ylabel('Phase Coherence')
        plt.title(f'Preliminary TEP Analysis - {ac.upper()} ({len(pair_files)} files)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = ROOT / f"results/preliminary_plot_{ac}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: Failed to fit model: {e}")
        return None

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Preliminary TEP Analysis')
    parser.add_argument('ac', nargs='?', default='code', 
                       choices=['code', 'igs_combined', 'esa_final'],
                       help='Analysis center to analyze')
    args = parser.parse_args()
    
    results = analyze_preliminary_data(args.ac)
    if results:
        print(f"\n{'='*60}")
        print("PRELIMINARY ANALYSIS COMPLETE")
        print(f"{'='*60}")
    else:
        print("PRELIMINARY ANALYSIS FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main()
