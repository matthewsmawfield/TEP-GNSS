#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 3.1: Robust Block Bootstrap Validation
==============================================================

Performs comprehensive block bootstrap validation by resampling entire stations
or entire days from the processed pair data to create new bootstrap datasets.
Each bootstrap sample undergoes the full correlation analysis to assess the
robustness of TEP parameters against spatial and temporal dependencies.

This implementation addresses the highest level of statistical scrutiny by
directly testing how sensitive the primary TEP correlation parameters (λ, A, C₀)
are to the specific composition of stations and days in the original dataset.

Key Analyses:
1. Station Block Bootstrap - resample entire stations with replacement
2. Day Block Bootstrap - resample entire days with replacement  
3. Hybrid Block Bootstrap - combined station-day resampling
4. Parameter stability assessment across bootstrap samples
5. Confidence interval estimation via bootstrap percentiles
6. Outlier detection and robustness metrics

METHODOLOGY: For each bootstrap iteration:
1. Resample fundamental units (stations/days) with replacement
2. Extract all pairs involving the resampled units
3. Re-run correlation analysis (binning + exponential fitting)
4. Store fitted parameters (λ, A, C₀, R²)
5. Compute bootstrap statistics and confidence intervals

This provides the most rigorous test of TEP parameter stability against
the inherent spatial and temporal structure of the GNSS station network.

Requirements: Step 2.0 complete (Core TEP Correlation Analysis)
Inputs:
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0)
  - results/tmp/step_2_0_pairs_*.csv (from Step 2.0, if `TEP_WRITE_PAIR_LEVEL=1`)
  - data/coordinates/step_1_1_station_coords_global.csv (station metadata, from Step 1.1)
Outputs:
  - results/outputs/step_3_1_robust_block_bootstrap_{ac}.json
Next: Step 3.2 (Null Tests)

Environment Variables:
  - TEP_STATION_BOOTSTRAP_SAMPLES: Number of station bootstrap samples (default: 500)
  - TEP_DAY_BOOTSTRAP_SAMPLES: Number of day bootstrap samples (default: 300)  
  - TEP_HYBRID_BOOTSTRAP_SAMPLES: Number of hybrid bootstrap samples (default: 200)
  - TEP_BOOTSTRAP_MIN_STATIONS: Minimum stations per bootstrap sample (default: 100)
  - TEP_BOOTSTRAP_MIN_DAYS: Minimum days per bootstrap sample (default: 100)
  - TEP_BOOTSTRAP_CONFIDENCE_LEVEL: Confidence level for intervals (default: 0.95)

Author: Matthew Lukin Smawfield
Date: September 2025
Theory: Temporal Equivalence Principle (TEP)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit
import gc
import psutil
from datetime import datetime

# Add the project root to Python path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import project utilities
from scripts.utils.config import TEPConfig
from scripts.utils.logger import print_status, setup_logging
from scripts.utils.exceptions import TEPFileError, TEPDataError

def check_memory_usage():
    """Monitor memory usage and warn if approaching limits."""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    
    print_status(f"Memory usage: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)", "MEMORY")
    
    memory_limit_gb = TEPConfig.get_float('TEP_MEMORY_LIMIT_GB', 8.0)
    if used_gb > memory_limit_gb:
        print_status(f"Memory usage ({used_gb:.1f} GB) exceeds limit ({memory_limit_gb:.1f} GB)", "WARNING")
        return False
    return True

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def safe_csv_read(file_path):
    """Safely read CSV with error handling."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print_status(f"Error reading {file_path}: {e}", "WARNING")
        return None

def load_station_coordinates():
    """Load station coordinates for metadata."""
    coord_file = ROOT / "data/coordinates/step_1_1_station_coords_global.csv"
    if not coord_file.exists():
        raise TEPFileError(f"Station coordinates file not found: {coord_file} (Ensure Step 1.1 is complete)")
    
    try:
        coords_df = safe_csv_read(coord_file)
        if coords_df is None or len(coords_df) == 0:
            raise TEPDataError("Station coordinates file is empty or unreadable")
        return coords_df
    except Exception as e:
        raise TEPDataError(f"Failed to load station coordinates: {e}")

def load_complete_pair_dataset(ac: str) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center.
    Optimized for memory efficiency with chunked loading.
    """
    print_status(f"Loading complete pair dataset for {ac}...", "PROCESS")
    
    pairs_dir = ROOT / "results/tmp"
    if not pairs_dir.exists():
        raise TEPFileError(f"Pairs directory not found: {pairs_dir} (Ensure Step 2.0 is complete and TEP_WRITE_PAIR_LEVEL is set)")
    
    # Find all pair files for this analysis center
    pair_files = list(pairs_dir.glob(f"step_2_0_pairs_{ac}_*.csv")) # Updated from step_3_pairs
    if not pair_files:
        raise TEPFileError(f"No pair files found for analysis center: {ac} (Ensure Step 2.0 is complete)")
    
    print_status(f"Found {len(pair_files)} pair files for {ac}", "INFO")
    
    # Load files in chunks to manage memory
    dataframes = []
    for i, file_path in enumerate(pair_files):
        if i % 50 == 0:  # Progress logging
            print_status(f"Loading file {i+1}/{len(pair_files)}: {file_path.name}", "PROCESS")
        
        try:
            df_chunk = safe_csv_read(file_path)
            if df_chunk is not None and len(df_chunk) > 0:
                # Ensure required columns exist
                required_cols = ['station_i', 'station_j', 'date', 'dist_km', 'plateau_phase']
                if all(col in df_chunk.columns for col in required_cols):
                    # Convert plateau_phase to coherence using cos() for compatibility
                    df_chunk['coherence'] = np.cos(df_chunk['plateau_phase'])
                    dataframes.append(df_chunk)
                else:
                    print_status(f"Skipping {file_path.name}: missing required columns", "WARNING")
        except Exception as e:
            print_status(f"Error loading {file_path.name}: {e}", "WARNING")
            continue
    
    if not dataframes:
        raise TEPDataError(f"No valid pair data loaded for {ac} (Ensure Step 2.0 is complete)")
    
    # Concatenate all chunks
    print_status(f"Concatenating {len(dataframes)} data chunks...", "PROCESS")
    complete_df = pd.concat(dataframes, ignore_index=True)
    
    # Clean up memory
    del dataframes
    gc.collect()
    
    print_status(f"Loaded complete dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")
    check_memory_usage()
    
    return complete_df

def create_station_bootstrap_sample(complete_df: pd.DataFrame, 
                                  stations_to_sample: List[str], 
                                  bootstrap_id: int) -> pd.DataFrame:
    """
    Create a bootstrap sample by resampling entire stations with replacement.
    
    Args:
        complete_df: Complete pair dataset
        stations_to_sample: List of unique stations available for sampling
        bootstrap_id: Bootstrap iteration ID for reproducibility
        
    Returns:
        Bootstrap sample DataFrame containing pairs from resampled stations
    """
    min_stations = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_STATIONS', 100)
    n_stations_to_sample = max(min_stations, len(stations_to_sample))
    
    # Set seed for reproducibility
    np.random.seed(42 + bootstrap_id)
    
    # Sample stations with replacement
    sampled_stations = np.random.choice(stations_to_sample, 
                                       size=n_stations_to_sample, 
                                       replace=True)
    
    # Extract all pairs involving the sampled stations
    station_mask = (complete_df['station_i'].isin(sampled_stations) | 
                   complete_df['station_j'].isin(sampled_stations))
    
    bootstrap_sample = complete_df[station_mask].copy()
    
    return bootstrap_sample

def create_day_bootstrap_sample(complete_df: pd.DataFrame, 
                               days_to_sample: List[str], 
                               bootstrap_id: int) -> pd.DataFrame:
    """
    Create a bootstrap sample by resampling entire days with replacement.
    
    Args:
        complete_df: Complete pair dataset
        days_to_sample: List of unique days available for sampling
        bootstrap_id: Bootstrap iteration ID for reproducibility
        
    Returns:
        Bootstrap sample DataFrame containing pairs from resampled days
    """
    min_days = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_DAYS', 100)
    n_days_to_sample = max(min_days, len(days_to_sample))
    
    # Set seed for reproducibility
    np.random.seed(42 + bootstrap_id)
    
    # Sample days with replacement
    sampled_days = np.random.choice(days_to_sample, 
                                   size=n_days_to_sample, 
                                   replace=True)
    
    # Extract all pairs from the sampled days
    day_mask = complete_df['date'].isin(sampled_days)
    bootstrap_sample = complete_df[day_mask].copy()
    
    return bootstrap_sample

def create_hybrid_bootstrap_sample(complete_df: pd.DataFrame, 
                                 stations_to_sample: List[str],
                                 days_to_sample: List[str], 
                                 bootstrap_id: int) -> pd.DataFrame:
    """
    Create a hybrid bootstrap sample by resampling both stations and days.
    
    Args:
        complete_df: Complete pair dataset
        stations_to_sample: List of unique stations available for sampling
        days_to_sample: List of unique days available for sampling
        bootstrap_id: Bootstrap iteration ID for reproducibility
        
    Returns:
        Bootstrap sample DataFrame containing pairs from resampled stations and days
    """
    min_stations = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_STATIONS', 100)
    min_days = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_DAYS', 100)
    
    n_stations_to_sample = max(min_stations, len(stations_to_sample) // 2)
    n_days_to_sample = max(min_days, len(days_to_sample) // 2)
    
    # Set seed for reproducibility
    np.random.seed(42 + bootstrap_id)
    
    # Sample both stations and days with replacement
    sampled_stations = np.random.choice(stations_to_sample, 
                                       size=n_stations_to_sample, 
                                       replace=True)
    sampled_days = np.random.choice(days_to_sample, 
                                   size=n_days_to_sample, 
                                   replace=True)
    
    # Extract pairs involving sampled stations AND sampled days
    station_mask = (complete_df['station_i'].isin(sampled_stations) | 
                   complete_df['station_j'].isin(sampled_stations))
    day_mask = complete_df['date'].isin(sampled_days)
    
    hybrid_mask = station_mask & day_mask
    bootstrap_sample = complete_df[hybrid_mask].copy()
    
    return bootstrap_sample

def fit_correlation_model_bootstrap(bootstrap_sample: pd.DataFrame) -> Tuple[Optional[np.ndarray], bool, Dict]:
    """
    Fit exponential correlation model on a bootstrap sample.
    
    Args:
        bootstrap_sample: Bootstrap sample DataFrame
        
    Returns:
        Tuple of (fitted_params, success_flag, diagnostics)
    """
    try:
        # Analysis parameters
        num_bins = TEPConfig.get_int('TEP_BINS', 50)
        max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM', 20000)
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT', 10)
        
        # Create logarithmic distance bins
        edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
        
        # Bin the bootstrap data
        bootstrap_sample = bootstrap_sample.copy()
        bootstrap_sample['dist_bin'] = pd.cut(bootstrap_sample['dist_km'], bins=edges, right=False)
        
        # Aggregate by distance bins
        binned = bootstrap_sample.groupby('dist_bin', observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        
        if len(binned) < 5:  # Need enough bins for stable fit
            return None, False, {'error': 'insufficient_bins', 'n_bins': len(binned)}
        
        distances = binned['mean_dist'].values
        coherences = binned['mean_coh'].values
        weights = binned['count'].values
        
        # Initial parameter estimates
        c_range = coherences.max() - coherences.min()
        p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS', 4000), coherences.min()]
        
        # Fit exponential model with robust bounds
        popt, pcov = curve_fit(
            correlation_model, distances, coherences,
            p0=p0, sigma=1.0/np.sqrt(weights),
            bounds=([1e-10, 100, -1], [2, 20000, 1]),
            maxfev=5000
        )
        
        # Calculate R²
        predicted = correlation_model(distances, *popt)
        ss_res = np.sum((coherences - predicted) ** 2)
        ss_tot = np.sum((coherences - np.mean(coherences)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Diagnostics
        diagnostics = {
            'n_pairs': len(bootstrap_sample),
            'n_bins': len(binned),
            'r_squared': float(r_squared),
            'amplitude': float(popt[0]),
            'lambda_km': float(popt[1]),
            'offset': float(popt[2]),
            'param_errors': [float(np.sqrt(pcov[i, i])) for i in range(3)]
        }
        
        return popt, True, diagnostics
        
    except Exception as e:
        return None, False, {'error': str(e), 'n_pairs': len(bootstrap_sample) if 'bootstrap_sample' in locals() else 0}

def run_station_block_bootstrap(complete_df: pd.DataFrame) -> Dict:
    """
    Perform station block bootstrap analysis.
    
    Args:
        complete_df: Complete pair dataset
        
    Returns:
        Dict containing bootstrap results and statistics
    """
    print_status("Starting Station Block Bootstrap Analysis...", "PROCESS")
    
    # Get unique stations
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    n_bootstrap_samples = TEPConfig.get_int('TEP_STATION_BOOTSTRAP_SAMPLES', 500)
    
    print_status(f"Running {n_bootstrap_samples} station bootstrap samples from {len(unique_stations)} unique stations", "INFO")
    
    bootstrap_results = []
    lambda_estimates = []
    
    for i in range(n_bootstrap_samples):
        if (i + 1) % 50 == 0:
            progress_pct = (i + 1) / n_bootstrap_samples * 100
            print_status(f"Station bootstrap progress: {i+1}/{n_bootstrap_samples} ({progress_pct:.1f}%)", "PROCESS")
        
        # Create bootstrap sample
        bootstrap_sample = create_station_bootstrap_sample(complete_df, unique_stations, i)
        
        if len(bootstrap_sample) < 1000:  # Skip samples that are too small
            continue
        
        # Fit correlation model
        fitted_params, fit_success, diagnostics = fit_correlation_model_bootstrap(bootstrap_sample)
        
        if fit_success:
            lambda_estimates.append(fitted_params[1])  # Store lambda
            bootstrap_results.append({
                'bootstrap_id': i,
                'n_pairs': diagnostics['n_pairs'],
                'n_bins': diagnostics['n_bins'],
                'lambda_km': diagnostics['lambda_km'],
                'amplitude': diagnostics['amplitude'],
                'offset': diagnostics['offset'],
                'r_squared': diagnostics['r_squared']
            })
    
    if not lambda_estimates:
        return {'success': False, 'error': 'No successful bootstrap fits'}
    
    # Compute bootstrap statistics
    confidence_level = TEPConfig.get_float('TEP_BOOTSTRAP_CONFIDENCE_LEVEL', 0.95)
    alpha = 1 - confidence_level
    
    results = {
        'success': True,
        'method': 'station_block_bootstrap',
        'n_bootstrap_samples': n_bootstrap_samples,
        'n_successful_fits': len(lambda_estimates),
        'success_rate': len(lambda_estimates) / n_bootstrap_samples,
        'lambda_statistics': {
            'mean': float(np.mean(lambda_estimates)),
            'std': float(np.std(lambda_estimates)),
            'min': float(np.min(lambda_estimates)),
            'max': float(np.max(lambda_estimates)),
            'median': float(np.median(lambda_estimates)),
            'confidence_interval': [
                float(np.percentile(lambda_estimates, 100 * alpha / 2)),
                float(np.percentile(lambda_estimates, 100 * (1 - alpha / 2)))
            ],
            'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
        },
        'bootstrap_samples': bootstrap_results,
        'lambda_values': [float(x) for x in lambda_estimates]
    }
    
    lambda_mean = results['lambda_statistics']['mean']
    lambda_std = results['lambda_statistics']['std']
    cv = results['lambda_statistics']['coefficient_of_variation']
    
    print_status(f"Station Block Bootstrap complete: λ = {lambda_mean:.1f} ± {lambda_std:.1f} km (CV = {cv:.3f})", "SUCCESS")
    return results

def run_day_block_bootstrap(complete_df: pd.DataFrame) -> Dict:
    """
    Perform day block bootstrap analysis.
    
    Args:
        complete_df: Complete pair dataset
        
    Returns:
        Dict containing bootstrap results and statistics
    """
    print_status("Starting Day Block Bootstrap Analysis...", "PROCESS")
    
    # Get unique days
    unique_days = complete_df['date'].unique()
    n_bootstrap_samples = TEPConfig.get_int('TEP_DAY_BOOTSTRAP_SAMPLES', 300)
    
    print_status(f"Running {n_bootstrap_samples} day bootstrap samples from {len(unique_days)} unique days", "INFO")
    
    bootstrap_results = []
    lambda_estimates = []
    
    for i in range(n_bootstrap_samples):
        if (i + 1) % 30 == 0:
            progress_pct = (i + 1) / n_bootstrap_samples * 100
            print_status(f"Day bootstrap progress: {i+1}/{n_bootstrap_samples} ({progress_pct:.1f}%)", "PROCESS")
        
        # Create bootstrap sample
        bootstrap_sample = create_day_bootstrap_sample(complete_df, unique_days, i)
        
        if len(bootstrap_sample) < 1000:  # Skip samples that are too small
            continue
        
        # Fit correlation model
        fitted_params, fit_success, diagnostics = fit_correlation_model_bootstrap(bootstrap_sample)
        
        if fit_success:
            lambda_estimates.append(fitted_params[1])  # Store lambda
            bootstrap_results.append({
                'bootstrap_id': i,
                'n_pairs': diagnostics['n_pairs'],
                'n_bins': diagnostics['n_bins'],
                'lambda_km': diagnostics['lambda_km'],
                'amplitude': diagnostics['amplitude'],
                'offset': diagnostics['offset'],
                'r_squared': diagnostics['r_squared']
            })
    
    if not lambda_estimates:
        return {'success': False, 'error': 'No successful bootstrap fits'}
    
    # Compute bootstrap statistics
    confidence_level = TEPConfig.get_float('TEP_BOOTSTRAP_CONFIDENCE_LEVEL', 0.95)
    alpha = 1 - confidence_level
    
    results = {
        'success': True,
        'method': 'day_block_bootstrap',
        'n_bootstrap_samples': n_bootstrap_samples,
        'n_successful_fits': len(lambda_estimates),
        'success_rate': len(lambda_estimates) / n_bootstrap_samples,
        'lambda_statistics': {
            'mean': float(np.mean(lambda_estimates)),
            'std': float(np.std(lambda_estimates)),
            'min': float(np.min(lambda_estimates)),
            'max': float(np.max(lambda_estimates)),
            'median': float(np.median(lambda_estimates)),
            'confidence_interval': [
                float(np.percentile(lambda_estimates, 100 * alpha / 2)),
                float(np.percentile(lambda_estimates, 100 * (1 - alpha / 2)))
            ],
            'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
        },
        'bootstrap_samples': bootstrap_results,
        'lambda_values': [float(x) for x in lambda_estimates]
    }
    
    lambda_mean = results['lambda_statistics']['mean']
    lambda_std = results['lambda_statistics']['std']
    cv = results['lambda_statistics']['coefficient_of_variation']
    
    print_status(f"Day Block Bootstrap complete: λ = {lambda_mean:.1f} ± {lambda_std:.1f} km (CV = {cv:.3f})", "SUCCESS")
    return results

def run_hybrid_block_bootstrap(complete_df: pd.DataFrame) -> Dict:
    """
    Perform hybrid block bootstrap analysis (stations + days).
    
    Args:
        complete_df: Complete pair dataset
        
    Returns:
        Dict containing bootstrap results and statistics
    """
    print_status("Starting Hybrid Block Bootstrap Analysis...", "PROCESS")
    
    # Get unique stations and days
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    unique_days = complete_df['date'].unique()
    n_bootstrap_samples = TEPConfig.get_int('TEP_HYBRID_BOOTSTRAP_SAMPLES', 200)
    
    print_status(f"Running {n_bootstrap_samples} hybrid bootstrap samples from {len(unique_stations)} stations and {len(unique_days)} days", "INFO")
    
    bootstrap_results = []
    lambda_estimates = []
    
    for i in range(n_bootstrap_samples):
        if (i + 1) % 20 == 0:
            progress_pct = (i + 1) / n_bootstrap_samples * 100
            print_status(f"Hybrid bootstrap progress: {i+1}/{n_bootstrap_samples} ({progress_pct:.1f}%)", "PROCESS")
        
        # Create bootstrap sample
        bootstrap_sample = create_hybrid_bootstrap_sample(complete_df, unique_stations, unique_days, i)
        
        if len(bootstrap_sample) < 500:  # Skip samples that are too small
            continue
        
        # Fit correlation model
        fitted_params, fit_success, diagnostics = fit_correlation_model_bootstrap(bootstrap_sample)
        
        if fit_success:
            lambda_estimates.append(fitted_params[1])  # Store lambda
            bootstrap_results.append({
                'bootstrap_id': i,
                'n_pairs': diagnostics['n_pairs'],
                'n_bins': diagnostics['n_bins'],
                'lambda_km': diagnostics['lambda_km'],
                'amplitude': diagnostics['amplitude'],
                'offset': diagnostics['offset'],
                'r_squared': diagnostics['r_squared']
            })
    
    if not lambda_estimates:
        return {'success': False, 'error': 'No successful bootstrap fits'}
    
    # Compute bootstrap statistics
    confidence_level = TEPConfig.get_float('TEP_BOOTSTRAP_CONFIDENCE_LEVEL', 0.95)
    alpha = 1 - confidence_level
    
    results = {
        'success': True,
        'method': 'hybrid_block_bootstrap',
        'n_bootstrap_samples': n_bootstrap_samples,
        'n_successful_fits': len(lambda_estimates),
        'success_rate': len(lambda_estimates) / n_bootstrap_samples,
        'lambda_statistics': {
            'mean': float(np.mean(lambda_estimates)),
            'std': float(np.std(lambda_estimates)),
            'min': float(np.min(lambda_estimates)),
            'max': float(np.max(lambda_estimates)),
            'median': float(np.median(lambda_estimates)),
            'confidence_interval': [
                float(np.percentile(lambda_estimates, 100 * alpha / 2)),
                float(np.percentile(lambda_estimates, 100 * (1 - alpha / 2)))
            ],
            'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
        },
        'bootstrap_samples': bootstrap_results,
        'lambda_values': [float(x) for x in lambda_estimates]
    }
    
    lambda_mean = results['lambda_statistics']['mean']
    lambda_std = results['lambda_statistics']['std']
    cv = results['lambda_statistics']['coefficient_of_variation']
    
    print_status(f"Hybrid Block Bootstrap complete: λ = {lambda_mean:.1f} ± {lambda_std:.1f} km (CV = {cv:.3f})", "SUCCESS")
    return results

def assess_bootstrap_consistency(station_results: Dict, day_results: Dict, hybrid_results: Dict) -> Dict:
    """
    Assess consistency between different bootstrap methods.
    
    Args:
        station_results: Results from station block bootstrap
        day_results: Results from day block bootstrap  
        hybrid_results: Results from hybrid block bootstrap
        
    Returns:
        Dict containing consistency assessment
    """
    print_status("Assessing bootstrap method consistency...", "PROCESS")
    
    methods = {}
    if station_results.get('success', False):
        methods['station'] = station_results['lambda_statistics']
    if day_results.get('success', False):
        methods['day'] = day_results['lambda_statistics']
    if hybrid_results.get('success', False):
        methods['hybrid'] = hybrid_results['lambda_statistics']
    
    if len(methods) < 2:
        return {'consistency_assessment': 'insufficient_methods', 'n_methods': len(methods)}
    
    # Extract lambda means and confidence intervals
    lambda_means = [methods[method]['mean'] for method in methods]
    lambda_stds = [methods[method]['std'] for method in methods]
    
    # Overall statistics
    overall_mean = np.mean(lambda_means)
    overall_std = np.std(lambda_means)
    overall_cv = overall_std / overall_mean if overall_mean > 0 else 0
    
    # Check if confidence intervals overlap
    overlaps = {}
    method_names = list(methods.keys())
    for i, method_a in enumerate(method_names):
        for method_b in method_names[i+1:]:
            ci_a = methods[method_a]['confidence_interval']
            ci_b = methods[method_b]['confidence_interval']
            
            # Check for overlap
            overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
            overlaps[f"{method_a}_vs_{method_b}"] = overlap
    
    # Consistency assessment
    all_overlap = all(overlaps.values()) if overlaps else False
    consistency_level = "high" if overall_cv < 0.05 and all_overlap else \
                      "moderate" if overall_cv < 0.10 and all_overlap else "low"
    
    consistency_results = {
        'consistency_assessment': consistency_level,
        'n_methods': len(methods),
        'lambda_cross_method': {
            'mean': float(overall_mean),
            'std': float(overall_std),
            'coefficient_of_variation': float(overall_cv),
            'min': float(min(lambda_means)),
            'max': float(max(lambda_means)),
            'range': float(max(lambda_means) - min(lambda_means))
        },
        'confidence_interval_overlaps': overlaps,
        'all_intervals_overlap': all_overlap,
        'method_details': methods
    }
    
    print_status(f"Bootstrap consistency: {consistency_level} (CV = {overall_cv:.3f})", "INFO")
    return consistency_results

def run_robust_block_bootstrap_analysis(ac: str) -> Dict:
    """
    Main function to run comprehensive robust block bootstrap analysis.
    
    Args:
        ac: Analysis center identifier
        
    Returns:
        Dict containing comprehensive bootstrap results
    """
    print_status(f"Starting robust block bootstrap analysis for {ac}", "PROCESS")
    
    try:
        # Load complete dataset
        complete_df = load_complete_pair_dataset(ac)
        
        # Initialize results
        results = {
            'analysis_center': ac,
            'dataset_info': {
                'n_total_pairs': len(complete_df),
                'n_unique_stations': len(pd.unique(complete_df[['station_i', 'station_j']].values.ravel())),
                'n_unique_days': len(complete_df['date'].unique()),
                'date_range': [
                    complete_df['date'].min(),
                    complete_df['date'].max()
                ]
            }
        }
        
        # Run station block bootstrap
        print_status("=" * 60, "INFO")
        station_results = run_station_block_bootstrap(complete_df)
        results['station_block_bootstrap'] = station_results
        
        # Run day block bootstrap  
        print_status("=" * 60, "INFO")
        day_results = run_day_block_bootstrap(complete_df)
        results['day_block_bootstrap'] = day_results
        
        # Run hybrid block bootstrap
        print_status("=" * 60, "INFO")
        hybrid_results = run_hybrid_block_bootstrap(complete_df)
        results['hybrid_block_bootstrap'] = hybrid_results
        
        # Assess consistency between methods
        print_status("=" * 60, "INFO")
        consistency_results = assess_bootstrap_consistency(station_results, day_results, hybrid_results)
        results['consistency_analysis'] = consistency_results
        
        # Overall success assessment
        successful_methods = []
        if station_results.get('success', False):
            successful_methods.append('station')
        if day_results.get('success', False):
            successful_methods.append('day')
        if hybrid_results.get('success', False):
            successful_methods.append('hybrid')
        
        results['summary'] = {
            'successful_methods': successful_methods,
            'n_successful_methods': len(successful_methods),
            'overall_success': len(successful_methods) >= 2,
            'consistency_level': consistency_results.get('consistency_assessment', 'unknown')
        }
        
        print_status(f"Robust block bootstrap analysis complete for {ac}", "SUCCESS")
        print_status(f"Successful methods: {successful_methods}", "INFO")
        
        return results
        
    except Exception as e:
        print_status(f"Robust block bootstrap analysis failed for {ac}: {e}", "ERROR")
        return {
            'analysis_center': ac,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def safe_json_write(data: Dict, file_path: Path) -> bool:
    """Safely write JSON with error handling."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print_status(f"Error writing JSON to {file_path}: {e}", "ERROR")
        return False

def main():
    """Main execution function."""
    start_time = time.time()
    
    print_status("=" * 80, "INFO")
    print_status("TEP GNSS Analysis Package v0.13 - STEP 3.1: Robust Block Bootstrap", "TITLE")
    print_status("=" * 80, "INFO")
    
    # Configuration summary
    print_status("Configuration:", "INFO")
    print_status(f"  Station bootstrap samples: {TEPConfig.get_int('TEP_STATION_BOOTSTRAP_SAMPLES', 500)}", "INFO")
    print_status(f"  Day bootstrap samples: {TEPConfig.get_int('TEP_DAY_BOOTSTRAP_SAMPLES', 300)}", "INFO")
    print_status(f"  Hybrid bootstrap samples: {TEPConfig.get_int('TEP_HYBRID_BOOTSTRAP_SAMPLES', 200)}", "INFO")
    print_status(f"  Minimum stations per sample: {TEPConfig.get_int('TEP_BOOTSTRAP_MIN_STATIONS', 100)}", "INFO")
    print_status(f"  Minimum days per sample: {TEPConfig.get_int('TEP_BOOTSTRAP_MIN_DAYS', 100)}", "INFO")
    print_status(f"  Confidence level: {TEPConfig.get_float('TEP_BOOTSTRAP_CONFIDENCE_LEVEL', 0.95)}", "INFO")
    print_status(f"  Memory limit: {TEPConfig.get_float('TEP_MEMORY_LIMIT_GB', 8.0)} GB", "INFO")
    
    # Analysis centers to process
    analysis_centers = ['code', 'esa_final', 'igs_combined']
    print_status(f"Processing {len(analysis_centers)} analysis centers: {', '.join(analysis_centers)}", "INFO")
    
    # Process each analysis center
    for ac in analysis_centers:
        print_status("", "INFO")
        print_status(f"Processing analysis center: {ac.upper()}", "PROCESS")
        print_status("-" * 50, "INFO")
        
        # Run bootstrap analysis
        results = run_robust_block_bootstrap_analysis(ac)
        
        # Save results
        output_file = ROOT / "results/outputs" / f"step_3_1_robust_block_bootstrap_{ac}.json" # Updated from step_5_6_robust_block_bootstrap
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        success = safe_json_write(results, output_file)
        if success:
            print_status(f"Results saved to: {output_file}", "SUCCESS")
        else:
            print_status(f"Failed to save results for {ac}", "ERROR")
        
        # Memory cleanup
        gc.collect()
        check_memory_usage()
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_status("=" * 80, "INFO")
    print_status(f"Robust block bootstrap analysis completed in {elapsed_time:.1f} seconds", "SUCCESS")
    print_status("All bootstrap validation results saved to results/outputs/", "INFO")
    print_status("=" * 80, "INFO")
    
    return True

if __name__ == "__main__":
    try:
        setup_logging()
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("Analysis interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"Fatal error: {e}", "ERROR")
        sys.exit(1)
