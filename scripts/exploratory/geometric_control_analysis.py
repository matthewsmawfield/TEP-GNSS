#!/usr/bin/env python3
"""
Geometric Control Analysis for TEP Signal Validation
===================================================

CRITICAL VALIDATION TEST: Can network geometry alone produce spurious TEP-like correlations?

This exploratory analysis addresses a fundamental concern: whether the bell-shaped 
distribution of pairwise distances between GNSS stations could create artificial 
correlation patterns that masquerade as the Temporal Equivalence Principle (TEP) signal.

Methodology:
1. Use identical station network geometry as real TEP analysis
2. Generate synthetic coherence data with NO physical distance correlations
3. Apply identical logarithmic binning and exponential fitting procedures
4. Compare results to real TEP findings

Expected Results:
- If methodology is robust: synthetic data should show R² < 0.1, random λ values
- If methodology is biased: synthetic data might show spurious correlations similar to TEP

This test is essential for validating the scientific integrity of TEP findings.

Author: Matthew Lukin Smawfield  
Date: September 2025
Purpose: Exploratory validation of TEP methodology against geometric bias
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import json
from datetime import datetime

# Anchor to package root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import safe_csv_read, safe_json_read, safe_json_write

def print_status(text: str, status: str = "INFO"):
    """Print verbose status message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefixes = {"INFO": "[INFO]", "SUCCESS": "[SUCCESS]", "WARNING": "[WARNING]", 
                "ERROR": "[ERROR]", "PROCESS": "[PROCESSING]"}
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model: C(r) = A * exp(-r/λ) + C0"""
    return A * np.exp(-r / lambda_km) + C0

def generate_synthetic_coherence_data(station_distances_file, n_synthetic_datasets=5):
    """
    Generate synthetic coherence data with no distance correlations.
    
    Uses real station network geometry but replaces coherence values with:
    1. Pure random noise (uniform distribution)
    2. Gaussian noise around zero mean
    3. Structured noise that mimics measurement characteristics but has no distance dependence
    
    Args:
        station_distances_file: Path to step_8_station_distances.csv
        n_synthetic_datasets: Number of different synthetic datasets to generate
    
    Returns:
        List of synthetic datasets with identical distance structure but uncorrelated coherence
    """
    print_status("Generating synthetic coherence data for geometric control test", "INFO")
    
    # Load real station distance matrix
    distances_df = safe_csv_read(station_distances_file)
    print_status(f"Loaded {len(distances_df)} station pairs from real network", "INFO")
    
    synthetic_datasets = []
    
    for dataset_id in range(n_synthetic_datasets):
        synthetic_df = distances_df.copy()
        
        # Generate different types of synthetic coherence data
        if dataset_id == 0:
            # Pure uniform random noise [-1, 1] (typical coherence range)
            synthetic_df['coherence'] = np.random.uniform(-1, 1, len(synthetic_df))
            dataset_name = "uniform_random"
            
        elif dataset_id == 1:
            # Gaussian noise around zero with realistic standard deviation
            synthetic_df['coherence'] = np.random.normal(0, 0.3, len(synthetic_df))
            # Clip to realistic coherence range
            synthetic_df['coherence'] = np.clip(synthetic_df['coherence'], -1, 1)
            dataset_name = "gaussian_noise"
            
        elif dataset_id == 2:
            # Structured noise that decreases with sample size (mimics measurement uncertainty)
            # but has NO distance dependence
            n_pairs = len(synthetic_df)
            base_coherence = np.random.normal(0, 0.2, n_pairs)
            # Add measurement noise that scales with sqrt(N) but is distance-independent
            measurement_noise = np.random.normal(0, 0.1/np.sqrt(np.random.randint(10, 1000, n_pairs)))
            synthetic_df['coherence'] = base_coherence + measurement_noise
            synthetic_df['coherence'] = np.clip(synthetic_df['coherence'], -1, 1)
            dataset_name = "structured_noise"
            
        elif dataset_id == 3:
            # Biased random walk (to test if any systematic drift could create spurious correlations)
            n_pairs = len(synthetic_df)
            coherence_walk = np.cumsum(np.random.normal(0, 0.01, n_pairs))
            # Normalize to coherence range
            coherence_walk = (coherence_walk - coherence_walk.mean()) / coherence_walk.std() * 0.3
            synthetic_df['coherence'] = np.clip(coherence_walk, -1, 1)
            dataset_name = "random_walk"
            
        else:
            # Distance-ANTI-correlated data (negative exponential) to test fitting robustness
            distances = synthetic_df['distance_km'].values
            # Create anti-correlation that gets stronger with distance
            anti_corr = -0.1 * np.exp(-distances / 5000) + np.random.normal(0, 0.2, len(distances))
            synthetic_df['coherence'] = np.clip(anti_corr, -1, 1)
            dataset_name = "anti_correlated"
        
        synthetic_df['dataset_name'] = dataset_name
        synthetic_df['dataset_id'] = dataset_id
        synthetic_datasets.append(synthetic_df)
        
        print_status(f"Generated synthetic dataset {dataset_id}: {dataset_name}", "SUCCESS")
    
    return synthetic_datasets

def apply_tep_binning_and_fitting(df, dataset_name):
    """
    Apply identical binning and fitting methodology as used in real TEP analysis.
    
    This uses the exact same logarithmic binning and exponential fitting procedures
    as step_3_tep_correlation_analysis.py to ensure fair comparison.
    """
    print_status(f"Applying TEP methodology to {dataset_name}", "INFO")
    
    # Use identical configuration as real TEP analysis
    num_bins = TEPConfig.get_int('TEP_BINS', default=30)
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM', default=13000)
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT', default=100)
    
    # Identical logarithmic binning
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    # Bin the synthetic data
    df = df.copy()
    df['dist_bin'] = pd.cut(df['distance_km'], bins=edges, right=False)
    
    # Aggregate by bins (identical to real analysis)
    binned_stats = df.groupby('dist_bin', observed=True).agg({
        'distance_km': 'mean',
        'coherence': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    binned_stats.columns = ['dist_bin', 'mean_distance_km', 'mean_coherence', 'std_coherence', 'pair_count']
    
    # Filter bins with sufficient data (identical criterion)
    binned_stats = binned_stats[binned_stats['pair_count'] >= min_bin_count]
    
    if len(binned_stats) < 5:
        print_status(f"Insufficient bins for fitting in {dataset_name}", "WARNING")
        return None
    
    # Extract data for fitting
    distances = binned_stats['mean_distance_km'].values
    coherences = binned_stats['mean_coherence'].values
    weights = binned_stats['pair_count'].values
    
    # Apply identical exponential fitting with same bounds as real analysis
    try:
        # Use identical bounds and fitting procedure
        bounds = ([0.01, 100, -1], [2, 20000, 1])
        popt, pcov = curve_fit(exponential_model, distances, coherences, 
                              sigma=1/np.sqrt(weights),  # Weight by sqrt(N) as in real analysis
                              bounds=bounds, maxfev=5000)
        
        A, lambda_km, C0 = popt
        param_errors = np.sqrt(np.diag(pcov))
        
        # Calculate R-squared
        y_pred = exponential_model(distances, A, lambda_km, C0)
        ss_res = np.sum((coherences - y_pred) ** 2)
        ss_tot = np.sum((coherences - np.mean(coherences)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results = {
            'dataset_name': dataset_name,
            'exponential_fit': {
                'amplitude': float(A),
                'amplitude_error': float(param_errors[0]),
                'lambda_km': float(lambda_km),
                'lambda_error': float(param_errors[1]),
                'offset': float(C0),
                'offset_error': float(param_errors[2]),
                'r_squared': float(r_squared)
            },
            'binned_data': {
                'distances': distances.tolist(),
                'coherences': coherences.tolist(),
                'weights': weights.tolist()
            },
            'n_bins_used': len(distances),
            'total_pairs': int(df['pair_count'].sum()) if 'pair_count' in df.columns else len(df)
        }
        
        print_status(f"{dataset_name}: λ = {lambda_km:.0f} km, R² = {r_squared:.3f}", "INFO")
        return results
        
    except Exception as e:
        print_status(f"Fitting failed for {dataset_name}: {e}", "WARNING")
        return None

def create_comparison_visualization(synthetic_results, real_tep_results, output_dir):
    """
    Create visualization comparing synthetic control results to real TEP findings.
    """
    print_status("Creating geometric control comparison visualization", "INFO")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Geometric Control Analysis: Synthetic vs Real TEP Results', fontsize=16, fontweight='bold')
    
    # Plot synthetic results
    for i, result in enumerate(synthetic_results):
        if result is None:
            continue
            
        row = i // 3
        col = i % 3
        if row >= 2:
            break
            
        ax = axes[row, col]
        
        # Plot binned data
        distances = np.array(result['binned_data']['distances'])
        coherences = np.array(result['binned_data']['coherences'])
        
        ax.scatter(distances, coherences, alpha=0.6, s=30, color='red', label='Synthetic Data')
        
        # Plot fit
        fit_params = result['exponential_fit']
        x_fit = np.linspace(100, 5000, 100)
        y_fit = exponential_model(x_fit, fit_params['amplitude'], 
                                 fit_params['lambda_km'], fit_params['offset'])
        ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
               label=f"λ = {fit_params['lambda_km']:.0f} km\nR² = {fit_params['r_squared']:.3f}")
        
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Coherence')
        ax.set_title(f"Synthetic: {result['dataset_name']}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add real TEP results for comparison if available
    if real_tep_results:
        # Plot real results in remaining subplot
        ax = axes[1, 2]
        
        # This would plot real TEP data - placeholder for now
        ax.text(0.5, 0.5, 'Real TEP Results\n(to be loaded)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Real TEP Signal')
    
    plt.tight_layout()
    
    output_file = output_dir / 'geometric_control_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved comparison visualization: {output_file}", "SUCCESS")
    return str(output_file)

def generate_control_report(synthetic_results, output_dir):
    """
    Generate comprehensive report on geometric control analysis results.
    """
    print_status("Generating geometric control analysis report", "INFO")
    
    report = {
        'analysis_type': 'geometric_control_validation',
        'purpose': 'Test whether network geometry alone can produce spurious TEP-like correlations',
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'description': 'Applied identical TEP binning and fitting to synthetic data with no distance correlations',
            'binning': 'Logarithmic binning from 50-13000 km',
            'fitting': 'Exponential decay model with identical bounds as real analysis',
            'synthetic_datasets': len(synthetic_results)
        },
        'results': {
            'synthetic_fits': []
        }
    }
    
    lambda_values = []
    r_squared_values = []
    
    for result in synthetic_results:
        if result is None:
            continue
            
        fit_params = result['exponential_fit']
        lambda_values.append(fit_params['lambda_km'])
        r_squared_values.append(fit_params['r_squared'])
        
        report['results']['synthetic_fits'].append({
            'dataset_name': result['dataset_name'],
            'lambda_km': fit_params['lambda_km'],
            'lambda_error': fit_params['lambda_error'],
            'r_squared': fit_params['r_squared'],
            'amplitude': fit_params['amplitude'],
            'offset': fit_params['offset']
        })
    
    # Statistical summary
    if lambda_values:
        report['results']['statistical_summary'] = {
            'lambda_mean': float(np.mean(lambda_values)),
            'lambda_std': float(np.std(lambda_values)),
            'lambda_range': [float(np.min(lambda_values)), float(np.max(lambda_values))],
            'r_squared_mean': float(np.mean(r_squared_values)),
            'r_squared_max': float(np.max(r_squared_values)),
            'n_high_r_squared': int(np.sum(np.array(r_squared_values) > 0.3))  # TEP threshold
        }
    
    # Interpretation
    max_r_squared = max(r_squared_values) if r_squared_values else 0
    
    if max_r_squared < 0.1:
        interpretation = "METHODOLOGY VALIDATED: No spurious correlations from network geometry"
        confidence = "HIGH"
    elif max_r_squared < 0.3:
        interpretation = "METHODOLOGY LIKELY VALID: Weak spurious correlations possible but below TEP threshold"
        confidence = "MEDIUM"
    else:
        interpretation = "METHODOLOGY CONCERN: Network geometry may produce spurious correlations"
        confidence = "LOW"
    
    report['interpretation'] = {
        'conclusion': interpretation,
        'confidence': confidence,
        'max_spurious_r_squared': float(max_r_squared),
        'tep_r_squared_threshold': 0.3,
        'recommendation': 'Compare with real TEP R² values (typically > 0.8) for final validation'
    }
    
    # Save report
    report_file = output_dir / 'geometric_control_analysis_report.json'
    safe_json_write(report, report_file)
    
    print_status(f"Saved control analysis report: {report_file}", "SUCCESS")
    return report

def main():
    """
    Main function to run geometric control analysis.
    """
    print("="*80)
    print("GEOMETRIC CONTROL ANALYSIS FOR TEP VALIDATION")
    print("="*80)
    print("Testing whether network geometry alone can create spurious TEP-like correlations")
    print()
    
    # Setup paths
    root_dir = ROOT
    data_dir = root_dir / 'data'
    output_dir = root_dir / 'results/exploratory'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load station distance matrix
    distances_file = data_dir / 'processed/step_8_station_distances.csv'
    
    if not distances_file.exists():
        print_status(f"Station distances file not found: {distances_file}", "ERROR")
        print_status("Please run step_8_tep_visualization.py first to generate station distances", "ERROR")
        return
    
    try:
        # Generate synthetic datasets
        print_status("Phase 1: Generating synthetic coherence data", "PROCESS")
        synthetic_datasets = generate_synthetic_coherence_data(distances_file, n_synthetic_datasets=5)
        
        # Apply TEP methodology to each synthetic dataset
        print_status("Phase 2: Applying TEP methodology to synthetic data", "PROCESS")
        synthetic_results = []
        
        for dataset in synthetic_datasets:
            result = apply_tep_binning_and_fitting(dataset, dataset['dataset_name'].iloc[0])
            if result:
                synthetic_results.append(result)
        
        if not synthetic_results:
            print_status("No successful fits on synthetic data", "ERROR")
            return
        
        # Load real TEP results for comparison (optional)
        real_tep_results = None
        try:
            real_results_file = root_dir / 'results/outputs/step_3_correlation_code.json'
            if real_results_file.exists():
                real_tep_results = safe_json_read(real_results_file)
                print_status("Loaded real TEP results for comparison", "INFO")
        except Exception as e:
            print_status(f"Could not load real TEP results: {e}", "WARNING")
        
        # Create visualization
        print_status("Phase 3: Creating comparison visualization", "PROCESS")
        viz_file = create_comparison_visualization(synthetic_results, real_tep_results, output_dir)
        
        # Generate comprehensive report
        print_status("Phase 4: Generating analysis report", "PROCESS")
        report = generate_control_report(synthetic_results, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("GEOMETRIC CONTROL ANALYSIS SUMMARY")
        print("="*60)
        
        lambda_values = [r['exponential_fit']['lambda_km'] for r in synthetic_results]
        r_squared_values = [r['exponential_fit']['r_squared'] for r in synthetic_results]
        
        print(f"Synthetic datasets analyzed: {len(synthetic_results)}")
        print(f"Correlation lengths (λ): {np.min(lambda_values):.0f} - {np.max(lambda_values):.0f} km")
        print(f"R² values: {np.min(r_squared_values):.3f} - {np.max(r_squared_values):.3f}")
        print(f"Maximum spurious R²: {np.max(r_squared_values):.3f}")
        print()
        print(f"Interpretation: {report['interpretation']['conclusion']}")
        print(f"Confidence: {report['interpretation']['confidence']}")
        print()
        print(f"Report saved: {output_dir / 'geometric_control_analysis_report.json'}")
        print(f"Visualization: {viz_file}")
        
    except Exception as e:
        print_status(f"Geometric control analysis failed: {e}", "ERROR")
        raise

if __name__ == "__main__":
    main()
