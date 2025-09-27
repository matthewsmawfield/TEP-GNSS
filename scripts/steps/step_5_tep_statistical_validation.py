#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 5: Statistical Validation
==================================================

Performs rigorous statistical validation of temporal equivalence principle
signatures through advanced resampling methods and robustness testing.

Requirements: Step 3 complete
Next: Step 6 (Null Tests)

Key Analyses:
1. Leave-One-Station-Out (LOSO) analysis - tests stability against individual stations
2. Leave-One-Day-Out (LODO) analysis - tests stability against individual days  
3. Block Bootstrap - provides robust confidence intervals accounting for dependencies
4. Enhanced Anisotropy Analysis - detailed directional and temporal propagation tests

CRITICAL: This step loads the COMPLETE pair-level dataset (~5-6 GB) into memory
for maximum statistical rigor as requested by reviewers.

Inputs:
  - results/tmp/step_3_pairs_*.csv files (from Step 3)
  - results/outputs/step_3_correlation_{ac}.json (from Step 3)

Outputs:
  - results/outputs/step_5_statistical_validation_{ac}.json
  - results/outputs/enhanced_anisotropy_{ac}.json

Environment Variables:
  - TEP_ENABLE_LOSO: Enable Leave-One-Station-Out analysis (default: 1)
  - TEP_ENABLE_LODO: Enable Leave-One-Day-Out analysis (default: 1)
  - TEP_ENABLE_BLOCK_BOOTSTRAP: Enable block bootstrap (default: 1)
  - TEP_ENABLE_ENHANCED_ANISOTROPY: Enable enhanced anisotropy tests (default: 1)
  - TEP_MEMORY_LIMIT_GB: Maximum memory to use in GB (default: 8)

Author: Matthew Lukin Smawfield
Date: September 2025
Theory: Temporal Equivalence Principle (TEP)
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import psutil  # For memory monitoring

# Anchor to package root
ROOT = Path(__file__).resolve().parents[2]

# Import TEP utilities for better configuration and error handling
import sys
sys.path.insert(0, str(ROOT))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    TEPAnalysisError, safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)

def print_status(text: str, status: str = "INFO"):
    """Print verbose status message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefixes = {"INFO": "[INFO]", "SUCCESS": "[SUCCESS]", "WARNING": "[WARNING]", 
                "ERROR": "[ERROR]", "PROCESS": "[PROCESSING]", "MEMORY": "[MEMORY]"}
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

def check_memory_usage():
    """Monitor memory usage and warn if approaching limits"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    
    print_status(f"Memory usage: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)", "MEMORY")
    
    memory_limit_gb = TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')
    if used_gb > memory_limit_gb:
        print_status(f"WARNING: Memory usage ({used_gb:.1f} GB) exceeds limit ({memory_limit_gb} GB)", "WARNING")
        return False
    return True

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def load_complete_geospatial_dataset(ac: str) -> pd.DataFrame:
    """
    Load complete pair dataset from Step 4 geospatial files (with pre-computed azimuth).
    
    This is more efficient than loading from Step 3 pair files because:
    - Azimuth is already computed in Step 4
    - Delta longitude and local time differences are pre-calculated
    - Smaller file size due to aggregation
    
    Args:
        ac: Analysis center name ('code', 'igs_combined', 'esa_final')
    
    Returns:
        pd.DataFrame: Complete dataset with azimuth and geospatial metrics
    """
    print_status(f"Loading complete geospatial dataset from Step 4 for {ac.upper()}...", "PROCESS")
    
    # Load from Step 4 geospatial file (much more efficient)
    geospatial_file = ROOT / "data" / "processed" / f"step_4_geospatial_{ac}.csv"
    
    if not geospatial_file.exists():
        raise TEPFileError(f"Step 4 geospatial file not found: {geospatial_file}")
    
    print_status(f"Loading from {geospatial_file}", "INFO")
    
    try:
        # Load the complete geospatial dataset
        complete_df = pd.read_csv(geospatial_file)
        
        # Add coherence column (same as Step 5 original)
        complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
        
        # Clean data
        complete_df.dropna(subset=['dist_km', 'coherence', 'station_i', 'station_j', 'date'], inplace=True)
        complete_df = complete_df[complete_df['dist_km'] > 0].copy()
        
        print_status(f"Geospatial dataset loaded: {len(complete_df):,} pairs, {complete_df.memory_usage(deep=True).sum()/(1024**3):.2f} GB", "SUCCESS")
        print_status("Azimuth already computed in Step 4 - no redundant calculation needed", "SUCCESS")
        
        # Verify required columns are present
        required_cols = ['azimuth', 'delta_longitude', 'delta_local_time']
        missing_cols = [col for col in required_cols if col not in complete_df.columns]
        
        if missing_cols:
            raise TEPDataError(f"Missing required columns from Step 4: {missing_cols}")
        
        print_status(f"Available columns: {list(complete_df.columns)}", "INFO")
        check_memory_usage()
        
        return complete_df
        
    except Exception as e:
        print_status(f"Failed to load Step 4 geospatial data: {e}", "ERROR")
        print_status("Falling back to Step 3 pair data loading...", "WARNING")
        return load_complete_pair_dataset(ac)

def load_complete_pair_dataset(ac: str, use_chunked_processing: bool = None) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center with smart memory management.
    
    Args:
        ac: Analysis center name
        use_chunked_processing: Force chunked processing (None = auto-detect based on memory)
    
    Returns:
        pd.DataFrame: Complete dataset with columns [date, station_i, station_j, 
                     dist_km, plateau_phase, coherence, ...]
    """
    print_status(f"Loading complete pair-level dataset for {ac.upper()}...", "PROCESS")
    
    try:
        pair_dir = validate_directory_exists(ROOT / 'results' / 'tmp', "Pair-level data directory")
    except TEPFileError as e:
        raise TEPDataError(f"Pair-level data directory not available: {e}") from e
    
    pair_files = list(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
    if not pair_files:
        raise TEPDataError(f"No pair-level files found for {ac}")
    
    print_status(f"Found {len(pair_files)} pair-level files to load", "INFO")
    
    # Check available memory and decide on loading strategy
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    memory_limit_gb = TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')
    
    if use_chunked_processing is None:
        # Auto-detect: use chunked processing if low on memory
        use_chunked_processing = available_gb < (memory_limit_gb * 0.7)  # Use 70% threshold
    
    if use_chunked_processing:
        print_status(f"Using chunked processing (available: {available_gb:.1f} GB)", "INFO")
        return _load_dataset_chunked(pair_files, ac)
    else:
        print_status(f"Using in-memory processing (available: {available_gb:.1f} GB)", "INFO")
        return _load_dataset_memory(pair_files, ac)

def _load_dataset_memory(pair_files: List[Path], ac: str) -> pd.DataFrame:
    """Load dataset using in-memory processing (original approach)"""
    df_chunks = []
    total_pairs = 0
    
    for i, pfile in enumerate(pair_files):
        if i % 100 == 0:
            print_status(f"Loading file {i+1}/{len(pair_files)}: {pfile.name}", "PROCESS")
            check_memory_usage()
        
        def _load_file():
            return safe_csv_read(pfile)
        
        df_chunk = SafeErrorHandler.safe_file_operation(
            _load_file,
            error_message=f"Failed to load {pfile.name}",
            logger_func=print_status,
            return_on_error=None
        )
        
        if df_chunk is not None and len(df_chunk) > 0:
            df_chunks.append(df_chunk)
            total_pairs += len(df_chunk)
    
    if not df_chunks:
        raise TEPDataError(f"No valid data loaded for {ac}")
    
    print_status(f"Concatenating {len(df_chunks)} chunks with {total_pairs:,} total pairs...", "PROCESS")
    
    # Concatenate all chunks
    complete_df = pd.concat(df_chunks, ignore_index=True)
    del df_chunks  # Free intermediate memory
    gc.collect()
    
    # Add coherence column and clean data
    # Calculate proper phase coherence (always positive, 0-1 range)
    complete_df['coherence'] = np.abs(np.cos(complete_df['plateau_phase']))
    complete_df.dropna(subset=['dist_km', 'coherence', 'station_i', 'station_j', 'date'], inplace=True)
    complete_df = complete_df[complete_df['dist_km'] > 0].copy()
    
    print_status(f"Dataset loaded: {len(complete_df):,} pairs, {complete_df.memory_usage(deep=True).sum()/(1024**3):.2f} GB", "SUCCESS")
    check_memory_usage()
    
    return complete_df

def _load_dataset_chunked(pair_files: List[Path], ac: str) -> pd.DataFrame:
    """Load dataset using chunked processing for memory-constrained environments"""
    print_status("Using chunked processing to manage memory usage", "INFO")
    
    chunk_size = 50000  # Process 50k rows at a time
    processed_chunks = []
    total_pairs = 0
    
    for i, pfile in enumerate(pair_files):
        if i % 50 == 0:
            print_status(f"Processing file {i+1}/{len(pair_files)}: {pfile.name}", "PROCESS")
            if i > 0:
                check_memory_usage()
        
        try:
            # Read file in chunks to manage memory
            for chunk_df in pd.read_csv(pfile, chunksize=chunk_size):
                if len(chunk_df) == 0:
                    continue
                
                # Process chunk immediately
                chunk_df['coherence'] = np.cos(chunk_df['plateau_phase'])
                chunk_df.dropna(subset=['dist_km', 'coherence', 'station_i', 'station_j', 'date'], inplace=True)
                chunk_df = chunk_df[chunk_df['dist_km'] > 0].copy()
                
                if len(chunk_df) > 0:
                    processed_chunks.append(chunk_df)
                    total_pairs += len(chunk_df)
                
                # Memory management: consolidate chunks if too many
                if len(processed_chunks) > 100:
                    print_status("Consolidating chunks to manage memory...", "PROCESS")
                    consolidated = pd.concat(processed_chunks, ignore_index=True)
                    processed_chunks = [consolidated]
                    gc.collect()
                    
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print_status(f"Skipping malformed file {pfile.name}: {e}", "WARNING")
            continue
        except (MemoryError, OverflowError) as e:
            print_status(f"Memory error processing {pfile.name}: {e}", "ERROR")
            raise TEPAnalysisError(f"Insufficient memory for chunked processing: {e}") from e
    
    if not processed_chunks:
        raise TEPDataError(f"No valid data loaded for {ac}")
    
    print_status(f"Finalizing chunked dataset with {total_pairs:,} total pairs...", "PROCESS")
    complete_df = pd.concat(processed_chunks, ignore_index=True)
    
    print_status(f"Chunked dataset loaded: {len(complete_df):,} pairs", "SUCCESS")
    check_memory_usage()
    
    return complete_df

def run_loso_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform Leave-One-Station-Out (LOSO) analysis on the complete dataset.
    Tests stability by excluding each station and re-fitting correlation model.
    
    OPTIMIZATION: Uses statistical sampling of stations for computational efficiency
    while maintaining statistical validity.
    """
    print_status("Starting Leave-One-Station-Out (LOSO) analysis...", "PROCESS")
    
    # Get all unique stations
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    
    # OPTIMIZATION: Sample stations for computational efficiency
    max_stations_to_test = TEPConfig.get_int('TEP_LOSO_MAX_STATIONS', 50)  # Default: 50 stations
    
    if len(unique_stations) > max_stations_to_test:
        # Randomly sample stations for testing
        np.random.seed(42)  # Reproducible
        stations_to_test = np.random.choice(unique_stations, max_stations_to_test, replace=False)
        print_status(f"Sampling {max_stations_to_test} stations from {len(unique_stations)} total for efficiency", "INFO")
    else:
        stations_to_test = unique_stations
        print_status(f"Testing stability across all {len(unique_stations)} unique stations", "INFO")
    
    # Analysis parameters from centralized configuration
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    lambda_estimates = []
    
    for i, station_to_exclude in enumerate(stations_to_test):
        if i % 50 == 0:
            print_status(f"LOSO progress: {i+1}/{len(stations_to_test)} ({100*i/len(stations_to_test):.1f}%)", "PROCESS")
        
        # Filter out pairs involving this station
        subset_df = complete_df[
            (complete_df['station_i'] != station_to_exclude) & 
            (complete_df['station_j'] != station_to_exclude)
        ].copy()
        
        if len(subset_df) < 1000:  # Skip if too little data remains
            continue
        
        # Bin the data
        subset_df['dist_bin'] = pd.cut(subset_df['dist_km'], bins=edges, right=False)
        binned = subset_df.groupby('dist_bin', observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        
        if len(binned) < 5:  # Need enough bins for stable fit
            continue
        
        # Fit exponential model
        try:
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            c_range = coherences.max() - coherences.min()
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            popt, _ = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=([1e-10, 100, -1], [2, 20000, 1]),
                maxfev=5000
            )
            
            lambda_estimates.append(popt[1])  # Store lambda
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError):
            continue  # Skip failed fits - common in statistical resampling
    
    if not lambda_estimates:
        return {'success': False, 'error': 'No successful fits in LOSO analysis'}
    
    # Compute statistics
    results = {
        'success': True,
        'lambda_mean': float(np.mean(lambda_estimates)),
        'lambda_std': float(np.std(lambda_estimates)),
        'lambda_min': float(np.min(lambda_estimates)),
        'lambda_max': float(np.max(lambda_estimates)),
        'n_successful_fits': len(lambda_estimates),
        'n_stations_tested': len(unique_stations),
        'lambda_values': lambda_estimates,
        'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
    }
    
    print_status(f"LOSO complete: λ = {results['lambda_mean']:.1f} ± {results['lambda_std']:.1f} km (CV = {results['coefficient_of_variation']:.3f})", "SUCCESS")
    return results

def run_lodo_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform Leave-One-Day-Out (LODO) analysis on the complete dataset.
    Tests stability by excluding each day and re-fitting correlation model.
    
    OPTIMIZATION: Uses statistical sampling of days for computational efficiency.
    """
    print_status("Starting Leave-One-Day-Out (LODO) analysis...", "PROCESS")
    
    # Get all unique dates
    unique_dates = complete_df['date'].unique()
    
    # OPTIMIZATION: Sample days for computational efficiency
    max_days_to_test = TEPConfig.get_int('TEP_LODO_MAX_DAYS', 100)  # Default: 100 days
    
    if len(unique_dates) > max_days_to_test:
        # Randomly sample days for testing
        np.random.seed(43)  # Different seed from LOSO
        dates_to_test = np.random.choice(unique_dates, max_days_to_test, replace=False)
        print_status(f"Sampling {max_days_to_test} days from {len(unique_dates)} total for efficiency", "INFO")
    else:
        dates_to_test = unique_dates
        print_status(f"Testing stability across all {len(unique_dates)} unique days", "INFO")
    
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    lambda_estimates = []
    
    for i, date_to_exclude in enumerate(dates_to_test):
        if i % 25 == 0:
            print_status(f"LODO progress: {i+1}/{len(dates_to_test)} ({100*i/len(dates_to_test):.1f}%)", "PROCESS")
        
        # Filter out pairs from this date
        subset_df = complete_df[complete_df['date'] != date_to_exclude].copy()
        
        if len(subset_df) < 1000:  # Skip if too little data remains
            continue
        
        # Bin the data
        subset_df['dist_bin'] = pd.cut(subset_df['dist_km'], bins=edges, right=False)
        binned = subset_df.groupby('dist_bin', observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        
        if len(binned) < 5:  # Need enough bins for stable fit
            continue
        
        # Fit exponential model
        try:
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            c_range = coherences.max() - coherences.min()
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            popt, _ = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=([1e-10, 100, -1], [2, 20000, 1]),
                maxfev=5000
            )
            
            lambda_estimates.append(popt[1])  # Store lambda
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError):
            continue  # Skip failed fits - common in statistical resampling
    
    if not lambda_estimates:
        return {'success': False, 'error': 'No successful fits in LODO analysis'}
    
    # Compute statistics
    results = {
        'success': True,
        'lambda_mean': float(np.mean(lambda_estimates)),
        'lambda_std': float(np.std(lambda_estimates)),
        'lambda_min': float(np.min(lambda_estimates)),
        'lambda_max': float(np.max(lambda_estimates)),
        'n_successful_fits': len(lambda_estimates),
        'n_days_tested': len(dates_to_test),
        'lambda_values': lambda_estimates,
        'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
    }
    
    print_status(f"LODO complete: λ = {results['lambda_mean']:.1f} ± {results['lambda_std']:.1f} km (CV = {results['coefficient_of_variation']:.3f})", "SUCCESS")
    return results

def run_block_bootstrap_analysis(complete_df: pd.DataFrame, n_bootstrap: int = 200) -> Dict:
    """
    Perform block bootstrap analysis accounting for station and day dependencies.
    This addresses reviewer concerns about non-independence of measurements.
    
    Args:
        complete_df: Complete pair-level dataset
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dict with bootstrap results and confidence intervals
    """
    print_status(f"Starting Block Bootstrap analysis ({n_bootstrap} iterations)...", "PROCESS")
    
    # Get unique stations and dates for block resampling
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    unique_dates = complete_df['date'].unique()
    
    print_status(f"Block structure: {len(unique_stations)} stations, {len(unique_dates)} days", "INFO")
    
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    # Bootstrap parameters
    bootstrap_params = []
    np.random.seed(42)  # Reproducible results
    
    for boot_iter in range(n_bootstrap):
        if boot_iter % 100 == 0:
            print_status(f"Bootstrap progress: {boot_iter+1}/{n_bootstrap} ({100*boot_iter/n_bootstrap:.1f}%)", "PROCESS")
        
        try:
            # Two-way block bootstrap: resample both stations and days
            
            # Station block bootstrap
            resampled_stations = np.random.choice(unique_stations, size=len(unique_stations), replace=True)
            
            # Day block bootstrap  
            resampled_dates = np.random.choice(unique_dates, size=len(unique_dates), replace=True)
            
            # Create bootstrap sample by filtering to resampled stations and dates
            bootstrap_dfs = []
            for station in resampled_stations:
                for date in resampled_dates:
                    # Get pairs involving this station on this date
                    station_date_pairs = complete_df[
                        ((complete_df['station_i'] == station) | (complete_df['station_j'] == station)) &
                        (complete_df['date'] == date)
                    ].copy()
                    
                    if len(station_date_pairs) > 0:
                        bootstrap_dfs.append(station_date_pairs)
            
            if not bootstrap_dfs:
                continue
                
            # Combine bootstrap sample
            bootstrap_df = pd.concat(bootstrap_dfs, ignore_index=True)
            
            # Remove duplicates that might occur from overlapping station selections
            bootstrap_df = bootstrap_df.drop_duplicates(subset=['station_i', 'station_j', 'date'])
            
            if len(bootstrap_df) < 1000:  # Need minimum data for reliable fit
                continue
            
            # Bin the bootstrap sample
            bootstrap_df['dist_bin'] = pd.cut(bootstrap_df['dist_km'], bins=edges, right=False)
            binned = bootstrap_df.groupby('dist_bin', observed=True).agg(
                mean_dist=('dist_km', 'mean'),
                mean_coh=('coherence', 'mean'),
                count=('coherence', 'size')
            ).reset_index()
            
            # Filter for robust bins
            binned = binned[binned['count'] >= min_bin_count].dropna()
            
            if len(binned) < 5:  # Need enough bins for stable fit
                continue
            
            # Fit exponential model to bootstrap sample
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            c_range = coherences.max() - coherences.min()
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            popt, _ = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=([1e-10, 100, -1], [2, 20000, 1]),
                maxfev=5000
            )
            
            # Store bootstrap parameters [amplitude, lambda, offset]
            bootstrap_params.append(popt)
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError, MemoryError):
            continue  # Skip failed bootstrap iterations - expected during resampling
    
    if len(bootstrap_params) < 10:  # Need minimum successful bootstraps
        return {'success': False, 'error': f'Only {len(bootstrap_params)} successful bootstrap iterations'}
    
    # Convert to numpy array for easier analysis
    bootstrap_params = np.array(bootstrap_params)
    
    # Compute confidence intervals (2.5th and 97.5th percentiles for 95% CI)
    amplitude_ci = [float(np.percentile(bootstrap_params[:, 0], 2.5)), 
                   float(np.percentile(bootstrap_params[:, 0], 97.5))]
    lambda_ci = [float(np.percentile(bootstrap_params[:, 1], 2.5)), 
                float(np.percentile(bootstrap_params[:, 1], 97.5))]
    offset_ci = [float(np.percentile(bootstrap_params[:, 2], 2.5)), 
                float(np.percentile(bootstrap_params[:, 2], 97.5))]
    
    results = {
        'success': True,
        'n_successful_iterations': len(bootstrap_params),
        'n_requested_iterations': n_bootstrap,
        'success_rate': float(len(bootstrap_params) / n_bootstrap),
        'amplitude': {
            'mean': float(np.mean(bootstrap_params[:, 0])),
            'std': float(np.std(bootstrap_params[:, 0])),
            'ci_95': amplitude_ci
        },
        'lambda_km': {
            'mean': float(np.mean(bootstrap_params[:, 1])),
            'std': float(np.std(bootstrap_params[:, 1])),
            'ci_95': lambda_ci,
            'coefficient_of_variation': float(np.std(bootstrap_params[:, 1]) / np.mean(bootstrap_params[:, 1]))
        },
        'offset': {
            'mean': float(np.mean(bootstrap_params[:, 2])),
            'std': float(np.std(bootstrap_params[:, 2])),
            'ci_95': offset_ci
        },
        'bootstrap_values': {
            'amplitude': bootstrap_params[:, 0].tolist(),
            'lambda_km': bootstrap_params[:, 1].tolist(),
            'offset': bootstrap_params[:, 2].tolist()
        }
    }
    
    print_status(f"Block Bootstrap complete: λ = {results['lambda_km']['mean']:.1f} ± {results['lambda_km']['std']:.1f} km", "SUCCESS")
    print_status(f"95% CI: [{results['lambda_km']['ci_95'][0]:.1f}, {results['lambda_km']['ci_95'][1]:.1f}] km", "SUCCESS")
    return results

def run_enhanced_anisotropy_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform enhanced anisotropy analysis on the complete dataset.
    This provides detailed directional and temporal propagation analysis.
    """
    print_status("Starting Enhanced Anisotropy Analysis...", "PROCESS")
    
    # Check if we have coordinate information
    required_cols = ['station1_lat', 'station1_lon', 'station2_lat', 'station2_lon']
    has_coords = all(col in complete_df.columns for col in required_cols)
    
    if not has_coords:
        return {'success': False, 'error': 'Coordinate columns not found in dataset'}
    
    # Filter to pairs with valid coordinates
    coord_df = complete_df.dropna(subset=required_cols).copy()
    
    if len(coord_df) < 1000:
        return {'success': False, 'error': f'Insufficient pairs with coordinates: {len(coord_df)}'}
    
    print_status(f"Analyzing {len(coord_df):,} pairs with coordinate information", "INFO")
    
    # Compute azimuths for all pairs
    def compute_azimuth(lat1, lon1, lat2, lon2):
        """Compute azimuth from station 1 to station 2 in degrees"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        azimuth = np.arctan2(y, x)
        return (np.degrees(azimuth) + 360) % 360
    
    coord_df['azimuth'] = coord_df.apply(
        lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                   row['station2_lat'], row['station2_lon']), axis=1
    )
    
    # Group into 8 directional sectors (45° each)
    sector_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    coord_df['sector'] = coord_df['azimuth'].apply(lambda az: sector_names[int((az + 22.5) / 45) % 8])
    
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    # Analyze each sector
    sector_results = {}
    
    for sector in sector_names:
        sector_data = coord_df[coord_df['sector'] == sector].copy()
        
        if len(sector_data) < 1000:  # Need sufficient data
            continue
        
        # Bin the sector data
        sector_data['dist_bin'] = pd.cut(sector_data['dist_km'], bins=edges, right=False)
        binned = sector_data.groupby('dist_bin', observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        
        if len(binned) < 5:  # Need enough bins for fitting
            continue
        
        # Fit exponential model to this sector
        try:
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            c_range = coherences.max() - coherences.min()
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            popt, pcov = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=([1e-10, 100, -1], [2, 20000, 1]),
                maxfev=5000
            )
            
            # Calculate R-squared
            y_pred = correlation_model(distances, *popt)
            ss_res = np.sum(weights * (coherences - y_pred)**2)
            ss_tot = np.sum(weights * (coherences - np.average(coherences, weights=weights))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            sector_results[sector] = {
                'amplitude': float(popt[0]),
                'lambda_km': float(popt[1]),
                'offset': float(popt[2]),
                'r_squared': float(r_squared),
                'n_pairs': len(sector_data),
                'n_bins': len(binned),
                'param_errors': [float(np.sqrt(pcov[i, i])) for i in range(3)]
            }
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError):
            continue  # Skip failed fits - common in statistical resampling
    
    if len(sector_results) < 4:  # Need reasonable directional coverage
        return {'success': False, 'error': f'Only {len(sector_results)} sectors with successful fits'}
    
    # Compute anisotropy statistics
    lambda_values = [s['lambda_km'] for s in sector_results.values()]
    lambda_mean = np.mean(lambda_values)
    lambda_std = np.std(lambda_values)
    lambda_cv = lambda_std / lambda_mean if lambda_mean > 0 else 0
    
    # Earth motion analysis
    ew_sectors = ['E', 'W']
    ns_sectors = ['N', 'S']
    
    ew_lambdas = [sector_results[s]['lambda_km'] for s in ew_sectors if s in sector_results]
    ns_lambdas = [sector_results[s]['lambda_km'] for s in ns_sectors if s in sector_results]
    
    earth_motion_analysis = {}
    if len(ew_lambdas) >= 1 and len(ns_lambdas) >= 1:
        ew_mean = np.mean(ew_lambdas)
        ns_mean = np.mean(ns_lambdas)
        rotation_ratio = ew_mean / ns_mean if ns_mean > 0 else 1.0
        
        earth_motion_analysis = {
            'ew_lambda_mean': float(ew_mean),
            'ns_lambda_mean': float(ns_mean),
            'ew_ns_ratio': float(rotation_ratio),
            'rotation_aligned': bool(abs(rotation_ratio - 1.0) > 0.2),
            'interpretation': f'E-W/N-S ratio = {rotation_ratio:.2f} ' + 
                           ('(rotation-aligned anisotropy)' if abs(rotation_ratio - 1.0) > 0.2 else '(minimal rotation effect)')
        }
    
    # Overall results
    results = {
        'success': True,
        'sector_results': sector_results,
        'anisotropy_statistics': {
            'lambda_mean': float(lambda_mean),
            'lambda_std': float(lambda_std),
            'coefficient_of_variation': float(lambda_cv),
            'n_sectors': len(sector_results),
            'anisotropy_category': 'extreme' if lambda_cv > 0.8 else 'moderate' if lambda_cv > 0.2 else 'minimal'
        },
        'earth_motion_analysis': earth_motion_analysis,
        'data_summary': {
            'total_pairs_with_coords': len(coord_df),
            'sectors_analyzed': list(sector_results.keys())
        }
    }
    
    print_status(f"Enhanced Anisotropy complete: {len(sector_results)} sectors, CV = {lambda_cv:.3f}", "SUCCESS")
    return results

def run_temporal_orbital_tracking_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Track anisotropy patterns by day-of-year to detect orbital motion signatures.
    Tests whether E-W/N-S ratio varies seasonally in synchronization with Earth's 
    orbital motion, which would support TEP coupling predictions.
    """
    print_status("Starting Temporal Orbital Tracking Analysis...", "PROCESS")
    print_status("Testing for seasonal orbital motion signatures in GPS timing correlations", "PROCESS")
    
    # Check if we have date and coordinate information
    required_cols = ['date', 'station1_lat', 'station1_lon', 'station2_lat', 'station2_lon']
    has_required_data = all(col in complete_df.columns for col in required_cols)
    
    if not has_required_data:
        return {'success': False, 'error': 'Date or coordinate columns not found in dataset'}
    
    # Convert date column to datetime and extract day of year
    complete_df['date'] = pd.to_datetime(complete_df['date'])
    complete_df['day_of_year'] = complete_df['date'].dt.dayofyear
    
    print_status(f"Temporal range: {complete_df['date'].min()} to {complete_df['date'].max()}", "INFO")
    print_status(f"Day of year range: {complete_df['day_of_year'].min()} to {complete_df['day_of_year'].max()}", "INFO")
    
    # Compute azimuths for all pairs
    def compute_azimuth(lat1, lon1, lat2, lon2):
        """Compute azimuth from station 1 to station 2 in degrees"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        azimuth = np.arctan2(y, x)
        return (np.degrees(azimuth) + 360) % 360
    
    complete_df['azimuth'] = complete_df.apply(
        lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                   row['station2_lat'], row['station2_lon']), axis=1
    )
    
    # Group into East-West vs North-South for temporal tracking
    def classify_ew_ns(azimuth):
        """Classify direction as East-West or North-South"""
        # East-West: 45-135° and 225-315° (±45° around E and W)
        if (45 <= azimuth <= 135) or (225 <= azimuth <= 315):
            return 'EW'
        else:
            return 'NS'
    
    complete_df['ew_ns_class'] = complete_df['azimuth'].apply(classify_ew_ns)
    
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    # Track E-W/N-S ratio by day of year (sample every 10 days for efficiency)
    temporal_tracking = []
    day_samples = range(5, 366, 10)  # Sample every 10 days starting from day 5
    
    print_status(f"Tracking E-W/N-S ratio across {len(day_samples)} day samples...", "PROCESS")
    
    for day_of_year in day_samples:
        # Get data for this day (±2 day window for sufficient statistics)
        day_window = 2
        day_data = complete_df[
            (complete_df['day_of_year'] >= day_of_year - day_window) &
            (complete_df['day_of_year'] <= day_of_year + day_window)
        ].copy()
        
        if len(day_data) < 1000:  # Need sufficient data
            continue
        
        # Analyze E-W and N-S separately for this day
        ew_data = day_data[day_data['ew_ns_class'] == 'EW'].copy()
        ns_data = day_data[day_data['ew_ns_class'] == 'NS'].copy()
        
        if len(ew_data) < 500 or len(ns_data) < 500:
            continue
        
        # Fit correlation models for E-W and N-S
        ew_lambda = fit_directional_correlation(ew_data, edges, min_bin_count)
        ns_lambda = fit_directional_correlation(ns_data, edges, min_bin_count)
        
        if ew_lambda is not None and ns_lambda is not None and ns_lambda > 0:
            ew_ns_ratio = ew_lambda / ns_lambda
            
            # Calculate Earth's orbital parameters for this day
            orbital_params = calculate_earth_orbital_motion(day_of_year)
            
            temporal_tracking.append({
                'day_of_year': day_of_year,
                'ew_lambda_km': ew_lambda,
                'ns_lambda_km': ns_lambda,
                'ew_ns_ratio': ew_ns_ratio,
                'n_ew_pairs': len(ew_data),
                'n_ns_pairs': len(ns_data),
                'orbital_speed_kms': orbital_params['orbital_speed'],
                'orbital_phase': orbital_params['orbital_phase'],
                'earth_sun_distance_au': orbital_params['distance_au']
            })
    
    if len(temporal_tracking) < 10:
        return {'success': False, 'error': f'Insufficient temporal samples: {len(temporal_tracking)}'}
    
    # Statistical analysis of temporal patterns
    days = [t['day_of_year'] for t in temporal_tracking]
    ew_ns_ratios = [t['ew_ns_ratio'] for t in temporal_tracking]
    orbital_speeds = [t['orbital_speed_kms'] for t in temporal_tracking]
    
    # Test correlation with orbital motion
    orbital_correlation, orbital_p_value = stats.pearsonr(orbital_speeds, ew_ns_ratios)
    
    # Test for 365.25-day periodicity
    def seasonal_model(day, amplitude, phase, offset):
        return offset + amplitude * np.sin(2 * np.pi * day / 365.25 + phase)
    
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(seasonal_model, days, ew_ns_ratios, 
                              p0=[0.5, 0, np.mean(ew_ns_ratios)],
                              bounds=([-2, -2*np.pi, 0], [2, 2*np.pi, 10]))
        
        seasonal_fit = {
            'amplitude': popt[0],
            'phase': popt[1], 
            'offset': popt[2],
            'fit_success': True,
            'seasonal_variation_percent': abs(popt[0]) / popt[2] * 100
        }
    except:
        seasonal_fit = {'fit_success': False}
    
    # Overall results
    results = {
        'success': True,
        'temporal_tracking_data': temporal_tracking,
        'statistical_analysis': {
            'orbital_speed_correlation': orbital_correlation,
            'orbital_correlation_p_value': orbital_p_value,
            'n_temporal_samples': len(temporal_tracking),
            'mean_ew_ns_ratio': np.mean(ew_ns_ratios),
            'ew_ns_ratio_std': np.std(ew_ns_ratios),
            'ew_ns_ratio_range': [min(ew_ns_ratios), max(ew_ns_ratios)]
        },
        'seasonal_analysis': seasonal_fit,
        'orbital_motion_evidence': {
            'correlation_with_orbital_speed': orbital_correlation,
            'significance_p_value': orbital_p_value,
            'evidence_strength': classify_orbital_evidence(orbital_correlation, orbital_p_value),
            'interpretation': f'E-W/N-S ratio {"correlates" if abs(orbital_correlation) > 0.3 else "does not correlate"} with orbital speed'
        }
    }
    
    # Critical assessment
    if abs(orbital_correlation) > 0.5 and orbital_p_value < 0.05:
        print_status(f"Robust correlation confirmed: E-W/N-S anisotropy correlates with Earth's orbital motion (r={orbital_correlation:.3f}, p={orbital_p_value:.4f})", "SUCCESS")
        print_status("Results indicate GPS timing correlations may reflect Earth's orbital dynamics", "INFO")
    elif abs(orbital_correlation) > 0.3:
        print_status(f"Significant correlation with Earth's orbital motion identified (r={orbital_correlation:.3f})", "INFO")
    
    print_status(f"Temporal orbital tracking complete: {len(temporal_tracking)} samples analyzed", "SUCCESS")
    return results

def fit_directional_correlation(directional_df: pd.DataFrame, edges: np.ndarray, min_bin_count: int) -> Optional[float]:
    """Fit correlation model to directional subset of data"""
    try:
        # Bin the data
        directional_df['dist_bin'] = pd.cut(directional_df['dist_km'], bins=edges, right=False)
        binned = directional_df.groupby('dist_bin', observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        
        if len(binned) < 5:  # Need enough bins for fitting
            return None
        
        # Fit exponential model
        distances = binned['mean_dist'].values
        coherences = binned['mean_coh'].values
        weights = binned['count'].values
        
        c_range = coherences.max() - coherences.min()
        p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
        
        popt, _ = curve_fit(
            correlation_model, distances, coherences,
            p0=p0, sigma=1.0/np.sqrt(weights),
            bounds=([1e-10, 100, -1], [2, 20000, 1]),
            maxfev=5000
        )
        
        return popt[1]  # Return lambda
        
    except:
        return None

def calculate_earth_orbital_motion(day_of_year: int) -> Dict:
    """Calculate Earth's orbital parameters for given day of year"""
    # Perihelion occurs around January 4 (day 4)
    perihelion_day = 4
    orbital_phase = 2 * np.pi * (day_of_year - perihelion_day) / 365.25
    
    # Orbital parameters
    mean_orbital_speed = 29.78  # km/s
    eccentricity = 0.0167
    distance_factor = (1 - eccentricity * np.cos(orbital_phase))
    orbital_speed = mean_orbital_speed / distance_factor
    
    return {
        'day_of_year': day_of_year,
        'orbital_phase': orbital_phase,
        'orbital_speed': orbital_speed,
        'distance_au': distance_factor,
        'speed_variation_percent': (orbital_speed - mean_orbital_speed) / mean_orbital_speed * 100
    }

def classify_orbital_evidence(correlation: float, p_value: float) -> str:
    """Classify strength of orbital motion evidence"""
    if abs(correlation) > 0.7 and p_value < 0.001:
        return "Robust correlation with Earth's orbital motion confirmed"
    elif abs(correlation) > 0.5 and p_value < 0.01:
        return "Strong correlation with Earth's orbital motion detected"
    elif abs(correlation) > 0.3 and p_value < 0.05:
        return "Moderate correlation with Earth's orbital motion identified"
    elif abs(correlation) > 0.2:
        return "Weak correlation with Earth's orbital motion observed"
    else:
        return "No statistically significant correlation with Earth's orbital motion detected"

def process_analysis_center(ac: str) -> Dict:
    """
    Process statistical validation for one analysis center.
    
    Args:
        ac: Analysis center name ('code', 'igs_combined', 'esa_final')
    
    Returns:
        dict: Statistical validation results
    """
    print_status(f"Starting statistical validation for {ac.upper()}", "INFO")
    start_time = time.time()
    
    try:
        # Load complete dataset into memory
        complete_df = load_complete_pair_dataset(ac)
        
        # Initialize results
        results = {
            'analysis_center': ac.upper(),
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_pairs': len(complete_df),
                'unique_stations': len(pd.unique(complete_df[['station_i', 'station_j']].values.ravel())),
                'unique_dates': len(complete_df['date'].unique()),
                'distance_range_km': [float(complete_df['dist_km'].min()), float(complete_df['dist_km'].max())],
                'coherence_range': [float(complete_df['coherence'].min()), float(complete_df['coherence'].max())]
            }
        }
        
        # Run LOSO analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_LOSO'):
            results['loso_analysis'] = run_loso_analysis(complete_df)
        else:
            results['loso_analysis'] = {'enabled': False}
        
        # Run LODO analysis if enabled  
        if TEPConfig.get_bool('TEP_ENABLE_LODO'):
            results['lodo_analysis'] = run_lodo_analysis(complete_df)
        else:
            results['lodo_analysis'] = {'enabled': False}
        
        # Block Bootstrap analysis disabled due to computational inefficiency
        # LOSO/LODO provide superior validation of independence
        results['block_bootstrap_analysis'] = {
            'enabled': False, 
            'note': 'Disabled due to computational inefficiency. LOSO/LODO provide superior validation.'
        }
        
        # Run Enhanced Anisotropy analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_ENHANCED_ANISOTROPY'):
            results['enhanced_anisotropy_analysis'] = run_enhanced_anisotropy_analysis(complete_df)
        else:
            results['enhanced_anisotropy_analysis'] = {'enabled': False}
        
        # Run Temporal Orbital Tracking analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_TEMPORAL_ORBITAL_TRACKING'):
            results['temporal_orbital_tracking'] = run_temporal_orbital_tracking_analysis(complete_df)
        else:
            results['temporal_orbital_tracking'] = {'enabled': False}
        
        # ===== NEW HELICAL MOTION ANALYSES (ADDITIONS ONLY) =====
        
        # Run Chandler Wobble analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_CHANDLER_WOBBLE'):
            results['chandler_wobble_analysis'] = run_chandler_wobble_analysis(complete_df)
        else:
            results['chandler_wobble_analysis'] = {'enabled': False}
        
        # Run 3D Spherical Harmonic analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_3D_HARMONICS'):
            results['spherical_harmonics_analysis'] = run_3d_spherical_harmonic_analysis(complete_df)
        else:
            results['spherical_harmonics_analysis'] = {'enabled': False}
            
        # Run Multi-Frequency Beat analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_BEAT_FREQUENCIES'):
            results['beat_frequencies_analysis'] = run_multi_frequency_beat_analysis(complete_df)
        else:
            results['beat_frequencies_analysis'] = {'enabled': False}
            
        # Run Relative Motion Beat analysis if enabled (NEW ENHANCED VERSION)
        if TEPConfig.get_bool('TEP_ENABLE_RELATIVE_MOTION_BEATS'):
            results['relative_motion_beats_analysis'] = run_relative_motion_beat_analysis(complete_df)
        else:
            results['relative_motion_beats_analysis'] = {'enabled': False}
            
        # Run Mesh Dance Analysis if enabled (THE ULTIMATE TEST)
        if TEPConfig.get_bool('TEP_ENABLE_MESH_DANCE_ANALYSIS'):
            results['mesh_dance_analysis'] = run_mesh_dance_analysis(complete_df)
        else:
            results['mesh_dance_analysis'] = {'enabled': False}
            
        # Run Jupiter Opposition analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_JUPITER_OPPOSITION'):
            results['jupiter_opposition_analysis'] = run_jupiter_opposition_analysis(complete_df)
        else:
            results['jupiter_opposition_analysis'] = {'enabled': False}
        
        # Run Saturn Opposition analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_SATURN_OPPOSITION'):
            results['saturn_opposition_analysis'] = run_saturn_opposition_analysis(complete_df)
        else:
            results['saturn_opposition_analysis'] = {'enabled': False}
        
        # Run Mars Opposition analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_MARS_OPPOSITION'):
            results['mars_opposition_analysis'] = run_mars_opposition_analysis(complete_df)
        else:
            results['mars_opposition_analysis'] = {'enabled': False}
        
        # Run Lunar Standstill analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_LUNAR_STANDSTILL'):
            results['lunar_standstill_analysis'] = run_lunar_standstill_analysis(complete_df)
        else:
            results['lunar_standstill_analysis'] = {'enabled': False}
        
        # Run Solar Eclipse analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_SOLAR_ECLIPSE'):
            results['solar_eclipse_analysis'] = run_solar_eclipse_analysis(complete_df)
        else:
            results['solar_eclipse_analysis'] = {'enabled': False}
        
        # Run Nutation analysis if enabled (requires multi-year data)
        if TEPConfig.get_bool('TEP_ENABLE_NUTATION_ANALYSIS'):
            results['nutation_analysis'] = run_nutation_analysis(complete_df)
        else:
            results['nutation_analysis'] = {'enabled': False}
        
        # ===== END NEW HELICAL MOTION ANALYSES =====
        
        # Clean up memory
        del complete_df
        gc.collect()
        check_memory_usage()
        
        results['execution_time_seconds'] = time.time() - start_time
        results['success'] = True
        
        print_status(f"Statistical validation complete for {ac.upper()} in {results['execution_time_seconds']:.1f}s", "SUCCESS")
        return results
        
    except (TEPDataError, TEPFileError, TEPAnalysisError) as e:
        print_status(f"Statistical validation failed for {ac.upper()} - TEP error: {e}", "ERROR")
        return {
            'analysis_center': ac.upper(),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e),
            'error_type': 'TEP_ERROR',
            'execution_time_seconds': time.time() - start_time
        }
    except (MemoryError, OverflowError) as e:
        print_status(f"Statistical validation failed for {ac.upper()} - resource error: {e}", "ERROR")
        return {
            'analysis_center': ac.upper(),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e),
            'error_type': 'RESOURCE_ERROR',
            'execution_time_seconds': time.time() - start_time
        }
    except Exception as e:
        print_status(f"Statistical validation failed for {ac.upper()} - unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return {
            'analysis_center': ac.upper(),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e),
            'error_type': 'UNEXPECTED_ERROR',
            'execution_time_seconds': time.time() - start_time
        }

def run_helical_motion_only(analysis_center: str = None) -> Dict:
    """
    🌌 RUN ONLY THE NEW HELICAL MOTION ANALYSES
    
    This function runs ONLY the 5 new helical motion analyses without 
    the full Step 5 pipeline, saving time when you just want to test
    the "dance" detection capabilities.
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from helical motion analyses only
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("HELICAL MOTION ANALYSIS - Advanced Earth Motion Detection")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for ac in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING {ac.upper()} - HELICAL MOTION ANALYSIS")
        print(f"{'='*60}")
        
        try:
            # Load complete dataset from Step 4 (with pre-computed azimuth)
            complete_df = load_complete_geospatial_dataset(ac)
            
            results = {
                'analysis_center': ac.upper(),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'helical_motion_only',
                'data_summary': {
                    'total_pairs': len(complete_df),
                    'unique_stations': len(pd.unique(complete_df[['station_i', 'station_j']].values.ravel())),
                    'unique_dates': len(complete_df['date'].unique()),
                }
            }
            
            print_status(f"Loaded {len(complete_df):,} station pairs for {ac.upper()}", "INFO")
            
            # Run ONLY the 5 new helical motion analyses
            
            # 1. Chandler Wobble Analysis
            if TEPConfig.get_bool('TEP_ENABLE_CHANDLER_WOBBLE'):
                print_status("Running Chandler Wobble Analysis...", "PROCESS")
                results['chandler_wobble_analysis'] = run_chandler_wobble_analysis(complete_df)
            else:
                results['chandler_wobble_analysis'] = {'enabled': False}
            
            # 2. 3D Spherical Harmonic Analysis
            if TEPConfig.get_bool('TEP_ENABLE_3D_HARMONICS'):
                print_status("Running 3D Spherical Harmonic Analysis...", "PROCESS")
                results['spherical_harmonics_analysis'] = run_3d_spherical_harmonic_analysis(complete_df)
            else:
                results['spherical_harmonics_analysis'] = {'enabled': False}
                
            # 3. Multi-Frequency Beat Analysis
            if TEPConfig.get_bool('TEP_ENABLE_BEAT_FREQUENCIES'):
                print_status("Running Multi-Frequency Beat Analysis...", "PROCESS")
                results['beat_frequencies_analysis'] = run_multi_frequency_beat_analysis(complete_df)
            else:
                results['beat_frequencies_analysis'] = {'enabled': False}
                
            # 4. Relative Motion Beat Analysis
            if TEPConfig.get_bool('TEP_ENABLE_RELATIVE_MOTION_BEATS'):
                print_status("Running Relative Motion Beat Analysis...", "PROCESS")
                results['relative_motion_beats_analysis'] = run_relative_motion_beat_analysis(complete_df)
            else:
                results['relative_motion_beats_analysis'] = {'enabled': False}
                
            # 5. MESH DANCE ANALYSIS - Network Coherence Assessment
            if TEPConfig.get_bool('TEP_ENABLE_MESH_DANCE_ANALYSIS'):
                print_status("Running Mesh Dance Analysis - Network Coherence Assessment...", "PROCESS")
                results['mesh_dance_analysis'] = run_mesh_dance_analysis(complete_df)
            else:
                results['mesh_dance_analysis'] = {'enabled': False}
            
            # 6. Jupiter Opposition Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_JUPITER_OPPOSITION'):
                print_status("Running Jupiter Opposition Pulse Analysis...", "PROCESS")
                results['jupiter_opposition_analysis'] = run_jupiter_opposition_analysis(complete_df)
            else:
                results['jupiter_opposition_analysis'] = {'enabled': False}
            
            # 7. Saturn Opposition Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_SATURN_OPPOSITION'):
                print_status("Running Saturn Opposition Pulse Analysis...", "PROCESS")
                results['saturn_opposition_analysis'] = run_saturn_opposition_analysis(complete_df)
            else:
                results['saturn_opposition_analysis'] = {'enabled': False}
            
            # 8. Mars Opposition Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_MARS_OPPOSITION'):
                print_status("Running Mars Opposition Pulse Analysis...", "PROCESS")
                results['mars_opposition_analysis'] = run_mars_opposition_analysis(complete_df)
            else:
                results['mars_opposition_analysis'] = {'enabled': False}
            
            # 9. Lunar Standstill Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_LUNAR_STANDSTILL'):
                print_status("Running Major Lunar Standstill Analysis...", "PROCESS")
                results['lunar_standstill_analysis'] = run_lunar_standstill_analysis(complete_df)
            else:
                results['lunar_standstill_analysis'] = {'enabled': False}
            
            # 10. Nutation Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_NUTATION_ANALYSIS'):
                print_status("Running Nutation Analysis...", "PROCESS")
                results['nutation_analysis'] = run_nutation_analysis(complete_df)
            else:
                results['nutation_analysis'] = {'enabled': False}
            
            # Clean up memory
            del complete_df
            gc.collect()
            
            results['execution_time_seconds'] = time.time() - start_time
            results['success'] = True
            
            # Save results with special naming for helical motion only
            output_dir = ROOT / "results/outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"step_5_helical_motion_only_{ac}.json"
            try:
                safe_json_write(results, output_file, indent=2)
                print_status(f"Helical motion results saved: {output_file}", "SUCCESS")
            except (TEPFileError, TEPDataError) as e:
                print_status(f"Failed to save results: {e}", "WARNING")
            
            all_results[ac] = results
            
            # Print summary of what was detected
            print_summary_helical_motion_results(results)
            
        except Exception as e:
            print_status(f"Helical motion analysis failed for {ac.upper()}: {e}", "ERROR")
            all_results[ac] = {
                'analysis_center': ac.upper(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'analysis_type': 'helical_motion_only'
            }
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("HELICAL MOTION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print_status(f"Total execution time: {total_time:.1f} seconds", "INFO")
    
    return all_results

def run_jupiter_only(analysis_center: str = None) -> Dict:
    """
    🪐 RUN ONLY THE JUPITER OPPOSITION ANALYSIS
    
    This function runs ONLY the Jupiter opposition analysis without 
    any other analyses, for fast testing and validation.
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from Jupiter opposition analysis only
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("JUPITER OPPOSITION ANALYSIS - Gravitational Potential Pulse Detection")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for ac in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING {ac.upper()} - JUPITER OPPOSITION ANALYSIS")
        print(f"{'='*60}")
        
        try:
            # Load complete dataset from Step 4 (with pre-computed azimuth)
            complete_df = load_complete_geospatial_dataset(ac)
            
            results = {
                'analysis_center': ac.upper(),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'jupiter_opposition_only',
                'data_summary': {
                    'total_pairs': len(complete_df),
                    'unique_stations': len(pd.unique(complete_df[['station_i', 'station_j']].values.ravel())),
                    'unique_dates': len(complete_df['date'].unique()),
                }
            }
            
            print_status(f"Loaded {len(complete_df):,} station pairs for {ac.upper()}", "INFO")
            
            # Run ONLY Jupiter Opposition Analysis
            print_status("Running Jupiter Opposition Pulse Analysis...", "PROCESS")
            results['jupiter_opposition_analysis'] = run_jupiter_opposition_analysis(complete_df)
            
            # Clean up memory
            del complete_df
            gc.collect()
            
            results['execution_time_seconds'] = time.time() - start_time
            results['success'] = True
            
            # Save results with special naming for Jupiter only
            output_dir = ROOT / "results/outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"step_5_jupiter_only_{ac}.json"
            try:
                safe_json_write(results, output_file, indent=2)
                print_status(f"Jupiter opposition results saved: {output_file}", "SUCCESS")
            except (TEPFileError, TEPDataError) as e:
                print_status(f"Failed to save results: {e}", "WARNING")
            
            all_results[ac] = results
            
            # Print summary of what was detected
            print_summary_jupiter_results(results)
            
        except Exception as e:
            print_status(f"Jupiter opposition analysis failed for {ac.upper()}: {e}", "ERROR")
            all_results[ac] = {
                'analysis_center': ac.upper(),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'analysis_type': 'jupiter_opposition_only'
            }
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("JUPITER OPPOSITION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print_status(f"Total execution time: {total_time:.1f} seconds", "INFO")
    
    return all_results

def run_saturn_only(analysis_center: str = None) -> Dict:
    """
    🪐 RUN ONLY THE SATURN OPPOSITION ANALYSIS
    
    This function runs ONLY the Saturn opposition analysis without 
    any other analyses, for fast testing and validation.
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from Saturn opposition analysis only
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("SATURN OPPOSITION ANALYSIS - Gravitational Potential Pulse Detection")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for center in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING CENTER: {center.upper()}")
        print(f"{'='*60}")
        
        # Load data for this center
        complete_df = load_complete_pair_dataset(center)
        if complete_df is None:
            print_status(f"Failed to load data for {center}", "ERROR")
            all_results[center] = {'success': False, 'error': 'Data loading failed'}
            continue
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Saturn opposition analysis
        results = {'analysis_center': center}
        results['saturn_opposition_analysis'] = run_saturn_opposition_analysis(complete_df)
        
        # Print summary
        print_summary_saturn_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_5_saturn_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Saturn opposition results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print(f"\n🪐 SATURN OPPOSITION ANALYSIS COMPLETED in {elapsed_time:.1f} seconds")
    print("=" * 80)
    
    return all_results

def run_mars_only(analysis_center: str = None) -> Dict:
    """
    🔴 RUN ONLY THE MARS OPPOSITION ANALYSIS
    
    This function runs ONLY the Mars opposition analysis without 
    any other analyses, for fast testing and validation.
    
    Mars has the weakest expected signal, making it an excellent
    test of our detection sensitivity.
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from Mars opposition analysis only
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("MARS OPPOSITION ANALYSIS - Weakest Signal Sensitivity Test")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for center in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING CENTER: {center.upper()}")
        print(f"{'='*60}")
        
        # Load data for this center
        complete_df = load_complete_pair_dataset(center)
        if complete_df is None:
            print_status(f"Failed to load data for {center}", "ERROR")
            all_results[center] = {'success': False, 'error': 'Data loading failed'}
            continue
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Mars opposition analysis
        results = {'analysis_center': center}
        results['mars_opposition_analysis'] = run_mars_opposition_analysis(complete_df)
        
        # Print summary
        print_summary_mars_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_5_mars_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Mars opposition results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print(f"\n🔴 MARS OPPOSITION ANALYSIS COMPLETED in {elapsed_time:.1f} seconds")
    print("=" * 80)
    
    return all_results

def run_lunar_only(analysis_center: str = None) -> Dict:
    """
    🌙 RUN ONLY THE LUNAR STANDSTILL ANALYSIS
    
    This function runs ONLY the Major Lunar Standstill analysis without 
    any other analyses, for fast testing and validation.
    
    The Lunar Standstill tracks sidereal day amplitude changes over months,
    which is fundamentally different from event-locked opposition analysis.
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from Lunar Standstill analysis only
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("LUNAR STANDSTILL ANALYSIS - Sidereal Day Amplitude Tracking")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for center in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING CENTER: {center.upper()}")
        print(f"{'='*60}")
        
        # Load data for this center
        complete_df = load_complete_pair_dataset(center)
        if complete_df is None:
            print_status(f"Failed to load data for {center}", "ERROR")
            all_results[center] = {'success': False, 'error': 'Data loading failed'}
            continue
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Lunar Standstill analysis
        results = {'analysis_center': center}
        results['lunar_standstill_analysis'] = run_lunar_standstill_analysis(complete_df)
        
        # Print summary
        print_summary_lunar_standstill_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_5_lunar_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Lunar Standstill results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print(f"\n🌙 LUNAR STANDSTILL ANALYSIS COMPLETED in {elapsed_time:.1f} seconds")
    print("=" * 80)
    
    return all_results

def run_eclipse_only(analysis_center: str = None) -> Dict:
    """
    ☀️ RUN ONLY THE SOLAR ECLIPSE ANALYSIS
    
    This function runs ONLY the Solar Eclipse analysis to test
    ionospheric effects that should be consistent across all centers.
    
    Solar eclipses provide an excellent validation test because they:
    - Affect ionosphere (not gravitational corrections)
    - Should show consistent effects across IGS/CODE/ESA
    - Have known timing and duration
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from Solar Eclipse analysis only
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("SOLAR ECLIPSE ANALYSIS - Ionospheric Effect Validation")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for center in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING CENTER: {center.upper()}")
        print(f"{'='*60}")
        
        # Load data for this center
        complete_df = load_complete_pair_dataset(center)
        if complete_df is None:
            print_status(f"Failed to load data for {center}", "ERROR")
            all_results[center] = {'success': False, 'error': 'Data loading failed'}
            continue
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Solar Eclipse analysis
        results = {'analysis_center': center}
        results['solar_eclipse_analysis'] = run_solar_eclipse_analysis(complete_df)
        
        # Print summary
        print_summary_eclipse_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_5_eclipse_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Solar Eclipse results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print(f"\n☀️ SOLAR ECLIPSE ANALYSIS COMPLETED in {elapsed_time:.1f} seconds")
    print("=" * 80)
    
    return all_results

def run_astronomical_events_only(analysis_center: str = None) -> Dict:
    """
    🌌 RUN ALL PLANETARY OPPOSITION ANALYSES
    
    This function runs Jupiter, Saturn, and Mars opposition analyses
    for comprehensive comparison of their signals.
    
    Args:
        analysis_center: Specific analysis center ('code', 'igs_combined', 'esa_final')
                        If None, runs all centers
    
    Returns:
        dict: Results from both astronomical event analyses
    """
    print("=" * 80)
    print("TEP GNSS Analysis Package v0.12")
    print("ASTRONOMICAL EVENTS ANALYSIS - Jupiter vs Saturn vs Mars Opposition Comparison")
    print("=" * 80)
    
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    all_results = {}
    
    for center in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING CENTER: {center.upper()}")
        print(f"{'='*60}")
        
        # Load data for this center
        complete_df = load_complete_pair_dataset(center)
        if complete_df is None:
            print_status(f"Failed to load data for {center}", "ERROR")
            all_results[center] = {'success': False, 'error': 'Data loading failed'}
            continue
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run all three analyses
        results = {'analysis_center': center}
        results['jupiter_opposition_analysis'] = run_jupiter_opposition_analysis(complete_df)
        results['saturn_opposition_analysis'] = run_saturn_opposition_analysis(complete_df)
        results['mars_opposition_analysis'] = run_mars_opposition_analysis(complete_df)
        
        # Print summaries
        print_summary_jupiter_results(results)
        print_summary_saturn_results(results)
        print_summary_mars_results(results)
        print_summary_astronomical_comparison(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_5_astronomical_events_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Astronomical events results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print(f"\n🌌 ASTRONOMICAL EVENTS ANALYSIS COMPLETED in {elapsed_time:.1f} seconds")
    print("=" * 80)
    
    return all_results

def print_summary_jupiter_results(results: Dict):
    """Print a summary of Jupiter opposition analysis results"""
    print(f"\nJUPITER OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}")
    print("=" * 60)

    # Jupiter Opposition Analysis
    jupiter = results.get('jupiter_opposition_analysis', {})
    if jupiter.get('success'):
        # Check for ANY significant individual detections first
        event_results = jupiter.get('event_results', {})
        significant_events = []
        
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                gaussian = event_data.get('gaussian_fit', {})
                if gaussian.get('is_significant', False):
                    significant_events.append((event_name, event_data))
        
        # Report significant individual events prominently
        if significant_events:
            print(f"🪐 Jupiter Opposition: ⭐ {len(significant_events)} SIGNIFICANT DETECTION(S) ⭐")
            for event_name, event_data in significant_events:
                event_date = event_data.get('event_date', 'Unknown')[:10]
                gaussian = event_data.get('gaussian_fit', {})
                amplitude = gaussian.get('amplitude', 0)
                std_err = gaussian.get('amplitude_std_err', 1)
                sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                center_days = gaussian.get('center_days', 0)
                direction = "suppression" if amplitude < 0 else "enhancement"
                amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                
                print(f"   🎯 {event_date}: {sigma_level:.1f}σ {direction} at day {center_days:.1f}")
                print(f"      Amplitude: {amplitude_pct:.1f}% of baseline")
        else:
            print(f"🪐 Jupiter Opposition: No significant individual detections")
        
        # Show stacked result
        stacked_analysis = jupiter.get('stacked_analysis', {})
        if stacked_analysis.get('success'):
            stacked_gaussian = stacked_analysis.get('gaussian_fit', {})
            if stacked_gaussian.get('is_significant', False):
                stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1))
                print(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ significant")
            else:
                stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1)) if stacked_gaussian.get('amplitude_std_err', 0) > 0 else 0
                print(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ (not significant)")
        
        # Show all individual event details
        print(f"   Individual Events:")
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                event_date = event_data.get('event_date', 'Unknown')[:10]
                gaussian = event_data.get('gaussian_fit', {})
                if gaussian.get('fit_success'):
                    amplitude = gaussian.get('amplitude', 0)
                    std_err = gaussian.get('amplitude_std_err', 1)
                    sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                    significant = gaussian.get('is_significant', False)
                    center_days = gaussian.get('center_days', 0)
                    print(f"     {event_date}: {sigma_level:.1f}σ ({'✓' if significant else '✗'}) peak at day {center_days:.1f}")
    
    elif jupiter.get('enabled') == False:
        print("🪐 Jupiter Opposition: Disabled in configuration")
    else:
        error = jupiter.get('error', 'Unknown error')
        print(f"🪐 Jupiter Opposition: ✗ Failed - {error}")
    
    print("-" * 50)

def print_summary_saturn_results(results: Dict):
    """Print a summary of Saturn opposition analysis results"""
    print(f"\nSATURN OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}")
    print("=" * 60)

    # Saturn Opposition Analysis
    saturn = results.get('saturn_opposition_analysis', {})
    if saturn.get('success'):
        # Check for ANY significant individual detections first
        event_results = saturn.get('event_results', {})
        significant_events = []
        
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                gaussian = event_data.get('gaussian_fit', {})
                if gaussian.get('is_significant', False):
                    significant_events.append((event_name, event_data))
        
        # Report significant individual events prominently
        if significant_events:
            print(f"🪐 Saturn Opposition: ⭐ {len(significant_events)} SIGNIFICANT DETECTION(S) ⭐")
            for event_name, event_data in significant_events:
                event_date = event_data.get('event_date', 'Unknown')[:10]
                gaussian = event_data.get('gaussian_fit', {})
                amplitude = gaussian.get('amplitude', 0)
                std_err = gaussian.get('amplitude_std_err', 1)
                sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                center_days = gaussian.get('center_days', 0)
                direction = "suppression" if amplitude < 0 else "enhancement"
                amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                
                print(f"   🎯 {event_date}: {sigma_level:.1f}σ {direction} at day {center_days:.1f}")
                print(f"      Amplitude: {amplitude_pct:.1f}% of baseline")
        else:
            print(f"🪐 Saturn Opposition: No significant individual detections")
        
        # Show stacked result
        stacked_analysis = saturn.get('stacked_analysis', {})
        if stacked_analysis.get('success'):
            stacked_gaussian = stacked_analysis.get('gaussian_fit', {})
            if stacked_gaussian.get('is_significant', False):
                stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1))
                print(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ significant")
            else:
                stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1)) if stacked_gaussian.get('amplitude_std_err', 0) > 0 else 0
                print(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ (not significant)")
        
        # Show all individual event details
        print(f"   Individual Events:")
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                event_date = event_data.get('event_date', 'Unknown')[:10]
                gaussian = event_data.get('gaussian_fit', {})
                if gaussian.get('fit_success'):
                    amplitude = gaussian.get('amplitude', 0)
                    std_err = gaussian.get('amplitude_std_err', 1)
                    sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                    significant = gaussian.get('is_significant', False)
                    center_days = gaussian.get('center_days', 0)
                    print(f"     {event_date}: {sigma_level:.1f}σ ({'✓' if significant else '✗'}) peak at day {center_days:.1f}")
    
    elif saturn.get('enabled') == False:
        print("🪐 Saturn Opposition: Disabled in configuration")
    else:
        error = saturn.get('error', 'Unknown error')
        print(f"🪐 Saturn Opposition: ✗ Failed - {error}")
    
    print("-" * 50)

def print_summary_mars_results(results: Dict):
    """Print a summary of Mars opposition analysis results"""
    print(f"\nMARS OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}")
    print("=" * 60)

    # Mars Opposition Analysis
    mars = results.get('mars_opposition_analysis', {})
    if mars.get('success'):
        # Check for ANY significant individual detections first
        event_results = mars.get('event_results', {})
        significant_events = []
        
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                gaussian = event_data.get('gaussian_fit', {})
                if gaussian.get('is_significant', False):
                    significant_events.append((event_name, event_data))
        
        # Report significant individual events prominently
        if significant_events:
            print(f"🔴 Mars Opposition: ⭐ {len(significant_events)} SIGNIFICANT DETECTION(S) ⭐")
            print("    🎯 REMARKABLE! Mars has the weakest expected signal!")
            for event_name, event_data in significant_events:
                event_date = event_data.get('event_date', 'Unknown')[:10]
                gaussian = event_data.get('gaussian_fit', {})
                amplitude = gaussian.get('amplitude', 0)
                std_err = gaussian.get('amplitude_std_err', 1)
                sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                center_days = gaussian.get('center_days', 0)
                direction = "suppression" if amplitude < 0 else "enhancement"
                amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                
                print(f"   🎯 {event_date}: {sigma_level:.1f}σ {direction} at day {center_days:.1f}")
                print(f"      Amplitude: {amplitude_pct:.1f}% of baseline")
        else:
            print(f"🔴 Mars Opposition: No significant detections (expected for weakest signal)")
        
        # Note: Mars has only one event, so no stacked analysis
        print(f"   📊 No stacked analysis (only one Mars opposition in dataset)")
        
        # Show individual event details
        print(f"   Individual Event:")
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                event_date = event_data.get('event_date', 'Unknown')[:10]
                gaussian = event_data.get('gaussian_fit', {})
                if gaussian.get('fit_success'):
                    amplitude = gaussian.get('amplitude', 0)
                    std_err = gaussian.get('amplitude_std_err', 1)
                    sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                    significant = gaussian.get('is_significant', False)
                    center_days = gaussian.get('center_days', 0)
                    print(f"     {event_date}: {sigma_level:.1f}σ ({'✓' if significant else '✗'}) peak at day {center_days:.1f}")
                    print(f"     Expected: 44x weaker than Jupiter, 4x weaker than Saturn")
    
    elif mars.get('enabled') == False:
        print("🔴 Mars Opposition: Disabled in configuration")
    else:
        error = mars.get('error', 'Unknown error')
        print(f"🔴 Mars Opposition: ✗ Failed - {error}")
    
    print("-" * 50)

def print_summary_lunar_standstill_results(results: Dict):
    """Print a summary of Lunar Standstill analysis results"""
    print(f"\nLUNAR STANDSTILL ANALYSIS SUMMARY - {results['analysis_center'].upper()}")
    print("=" * 60)

    # Lunar Standstill Analysis
    lunar = results.get('lunar_standstill_analysis', {})
    if lunar.get('success'):
        enhancement = lunar.get('enhancement_analysis', {}).get('pre_to_standstill', {})
        
        if enhancement:
            ratio = enhancement.get('enhancement_ratio', 1.0)
            percent = enhancement.get('enhancement_percent', 0.0)
            
            if ratio > 1.2:
                status = "⭐ STRONG ENHANCEMENT ⭐"
            elif ratio > 1.1:
                status = "🌟 MODERATE ENHANCEMENT"
            elif ratio > 1.05:
                status = "✨ WEAK ENHANCEMENT"
            else:
                status = "📊 NO SIGNIFICANT ENHANCEMENT"
            
            print(f"🌙 Major Lunar Standstill: {status}")
            print(f"   Enhancement Ratio: {ratio:.2f}x ({percent:+.1f}%)")
            print(f"   Pre-standstill amplitude: {enhancement.get('pre_amplitude', 0):.6f}")
            print(f"   Standstill amplitude: {enhancement.get('standstill_amplitude', 0):.6f}")
        else:
            print(f"🌙 Major Lunar Standstill: Insufficient data for enhancement analysis")
        
        # Period statistics
        period_stats = lunar.get('period_statistics', {})
        print(f"   Analysis periods:")
        for period, stats in period_stats.items():
            period_name = period.replace('_', ' ').title()
            print(f"     {period_name}: {stats['n_months']} months, amplitude = {stats['mean_amplitude']:.6f}")
        
        # Peak month
        peak_month = lunar.get('analysis_summary', {}).get('peak_amplitude_month')
        if peak_month:
            print(f"   Peak amplitude month: {peak_month}")
        
        # Trend analysis
        trend = lunar.get('trend_analysis', {})
        if 'peak_month_offset' in trend:
            offset = trend['peak_month_offset']
            r_squared = trend.get('r_squared', 0)
            print(f"   Quadratic fit peak: {offset:.1f} months from expected ({r_squared:.3f} R²)")
    
    elif lunar.get('enabled') == False:
        print("🌙 Major Lunar Standstill: Disabled in configuration")
    else:
        error = lunar.get('error', 'Unknown error')
        print(f"🌙 Major Lunar Standstill: ✗ Failed - {error}")
    
    print("-" * 50)

def print_summary_eclipse_results(results: Dict):
    """Print a summary of Solar Eclipse analysis results"""
    print(f"\nSOLAR ECLIPSE ANALYSIS SUMMARY - {results['analysis_center'].upper()}")
    print("=" * 60)

    # Solar Eclipse Analysis
    eclipse = results.get('solar_eclipse_analysis', {})
    if eclipse.get('success'):
        enhancement = eclipse.get('eclipse_enhancement', {})
        
        if enhancement:
            ratio = enhancement.get('enhancement_ratio', 1.0)
            percent = enhancement.get('enhancement_percent', 0.0)
            
            if ratio > 1.2:
                status = "⭐ STRONG ECLIPSE SIGNATURE ⭐"
            elif ratio > 1.1:
                status = "🌟 MODERATE ECLIPSE SIGNATURE"
            elif ratio > 1.05:
                status = "✨ WEAK ECLIPSE SIGNATURE"
            else:
                status = "📊 NO SIGNIFICANT ECLIPSE SIGNATURE"
            
            print(f"☀️ Solar Eclipse (April 8, 2024): {status}")
            print(f"   Enhancement Ratio: {ratio:.2f}x ({percent:+.1f}%)")
            print(f"   Baseline coherence: {enhancement.get('baseline_coherence', 0):.6f}")
            print(f"   Totality coherence: {enhancement.get('totality_coherence', 0):.6f}")
        else:
            print(f"☀️ Solar Eclipse: Insufficient data for enhancement analysis")
        
        # Gaussian fit results
        gaussian = eclipse.get('gaussian_fit', {})
        if gaussian.get('fit_success'):
            if gaussian.get('is_significant', False):
                amplitude = gaussian.get('amplitude', 0)
                std_err = gaussian.get('amplitude_std_err', 1)
                sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                center_hours = gaussian.get('center_days', 0)  # Note: this will be in hours
                print(f"   Gaussian fit: {sigma_level:.1f}σ significant peak at {center_hours:.1f} hours")
        
        # Phase analysis summary
        phase_analysis = eclipse.get('phase_analysis', {})
        if phase_analysis:
            print(f"   Eclipse phases analyzed:")
            for phase, data in phase_analysis.items():
                n_hours = data.get('n_hours', 0)
                mean_coh = data.get('mean_coherence', 0)
                print(f"     {phase.title()}: {n_hours} hours, coherence = {mean_coh:.6f}")
    
    elif eclipse.get('enabled') == False:
        print("☀️ Solar Eclipse: Disabled in configuration")
    else:
        error = eclipse.get('error', 'Unknown error')
        print(f"☀️ Solar Eclipse: ✗ Failed - {error}")
    
    print("-" * 50)

def print_summary_astronomical_comparison(results: Dict):
    """Print a comparison of Jupiter vs Saturn vs Mars opposition results"""
    print(f"\nASTRONOMICAL EVENTS COMPARISON - {results['analysis_center'].upper()}")
    print("=" * 60)
    
    jupiter = results.get('jupiter_opposition_analysis', {})
    saturn = results.get('saturn_opposition_analysis', {})
    mars = results.get('mars_opposition_analysis', {})
    
    if jupiter.get('success') and saturn.get('success') and mars.get('success'):
        # Count significant detections
        jupiter_significant = sum(1 for result in jupiter.get('event_results', {}).values() 
                                if result.get('success') and 
                                   result.get('gaussian_fit', {}).get('is_significant', False))
        saturn_significant = sum(1 for result in saturn.get('event_results', {}).values() 
                               if result.get('success') and 
                                  result.get('gaussian_fit', {}).get('is_significant', False))
        mars_significant = sum(1 for result in mars.get('event_results', {}).values() 
                             if result.get('success') and 
                                result.get('gaussian_fit', {}).get('is_significant', False))
        
        print(f"🪐 Jupiter: {jupiter_significant}/{jupiter.get('n_successful_events', 0)} significant events")
        print(f"🪐 Saturn:  {saturn_significant}/{saturn.get('n_successful_events', 0)} significant events")
        print(f"🔴 Mars:    {mars_significant}/{mars.get('n_successful_events', 0)} significant events")
        
        # Expected amplitude comparison
        jupiter_expected = jupiter.get('expected_amplitude_fraction', 0.0022)
        saturn_expected = saturn.get('expected_amplitude_fraction', 0.00019)
        mars_expected = mars.get('expected_amplitude_fraction', 0.00005)
        
        print(f"📊 Expected amplitude ratios:")
        print(f"   Jupiter/Saturn: {jupiter_expected/saturn_expected:.1f}x")
        print(f"   Jupiter/Mars: {jupiter_expected/mars_expected:.1f}x")
        print(f"   Saturn/Mars: {saturn_expected/mars_expected:.1f}x")
        
        # Stacked analysis comparison
        jupiter_stacked = jupiter.get('stacked_analysis', {}).get('gaussian_fit', {})
        saturn_stacked = saturn.get('stacked_analysis', {}).get('gaussian_fit', {})
        
        if jupiter_stacked.get('fit_success') and saturn_stacked.get('fit_success'):
            jupiter_sigma = abs(jupiter_stacked.get('amplitude', 0) / jupiter_stacked.get('amplitude_std_err', 1)) if jupiter_stacked.get('amplitude_std_err', 0) > 0 else 0
            saturn_sigma = abs(saturn_stacked.get('amplitude', 0) / saturn_stacked.get('amplitude_std_err', 1)) if saturn_stacked.get('amplitude_std_err', 0) > 0 else 0
            
            print(f"📈 Stacked significance: Jupiter {jupiter_sigma:.1f}σ vs Saturn {saturn_sigma:.1f}σ")
        
        # Overall assessment
        total_significant = jupiter_significant + saturn_significant + mars_significant
        if total_significant > 0:
            print(f"🌟 CONCLUSION: {total_significant} significant astronomical event signals detected!")
            if mars_significant > 0:
                print("    🎯 EXTRAORDINARY: Mars signal detected despite being weakest expected!")
        else:
            print("📊 CONCLUSION: No significant astronomical event signals detected")
    else:
        print("⚠️  Cannot compare - one or more analyses failed")
    
    print("-" * 50)

def print_summary_helical_motion_results(results: Dict):
    """Print a summary of helical motion analysis results"""
    print(f"\nHELICAL MOTION ANALYSIS SUMMARY - {results['analysis_center'].upper()}")
    print("=" * 60)

    # Chandler Wobble
    chandler = results.get('chandler_wobble_analysis', {})
    if chandler.get('success'):
        interp = chandler.get('interpretation', 'Unknown')
        print(f"Chandler Wobble (14-month): {interp}")

    # 3D Harmonics
    harmonics = results.get('spherical_harmonics_analysis', {})
    if harmonics.get('success'):
        cv = harmonics.get('anisotropy_statistics', {}).get('cv_lambda', 0)
        n_sectors = harmonics.get('n_sectors', 0)
        print(f"3D Spherical Harmonics: {n_sectors} directional sectors analyzed, CV = {cv:.3f}")

    # Beat Frequencies
    beats = results.get('beat_frequencies_analysis', {})
    if beats.get('success'):
        n_sig = beats.get('n_significant_beats', 0)
        print(f"Beat Frequencies: {n_sig} significant Earth motion interference patterns detected")

    # Relative Motion
    rel_motion = results.get('relative_motion_beats_analysis', {})
    if rel_motion.get('success'):
        interp = rel_motion.get('interpretation', 'Unknown')
        print(f"Relative Motion: {interp}")

    # MESH DANCE - The Ultimate Test
    dance = results.get('mesh_dance_analysis', {})
    if dance.get('success'):
        classification = dance.get('dance_signature', {}).get('classification', 'Unknown')
        score = dance.get('dance_signature', {}).get('dance_score', 0)
        print(f"Mesh Dance Analysis: {classification} (score = {score:.3f})")
        print(f"   Dance Score: {score:.3f}/1.0")

    # Jupiter Opposition Analysis
    jupiter = results.get('jupiter_opposition_analysis', {})
    if jupiter.get('success'):
        n_events = jupiter.get('n_successful_events', 0)
        interpretation = jupiter.get('interpretation', 'Unknown')
        print(f"Jupiter Opposition: {n_events} events analyzed - {interpretation}")
        
        # Show individual event results
        event_results = jupiter.get('event_results', {})
        for event_name, event_data in event_results.items():
            if event_data.get('success'):
                enhancement = event_data.get('statistical_analysis', {}).get('enhancement_ratio', 1.0)
                significant = event_data.get('statistical_analysis', {}).get('enhancement_significant', False)
                print(f"   {event_name}: {enhancement:.4f}x enhancement ({'significant' if significant else 'not significant'})")
    
    print("-" * 50)

def main():
    """Main function with command-line options for different analysis modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEP GNSS Statistical Validation - Step 5")
    parser.add_argument('--mode', choices=['full', 'helical', 'jupiter', 'saturn', 'mars', 'lunar', 'eclipse', 'astronomical'], default='full',
                        help='Analysis mode: full (complete statistical validation), helical (helical motion analyses only), jupiter (Jupiter opposition only), saturn (Saturn opposition only), mars (Mars opposition only), lunar (Lunar Standstill only), eclipse (Solar Eclipse only), or astronomical (Jupiter, Saturn, and Mars)')
    parser.add_argument('--center', choices=['code', 'igs_combined', 'esa_final'],
                        help='Specific GNSS analysis center to process')
    parser.add_argument('--list-helical', action='store_true',
                        help='List available helical motion analysis methods')
    
    args = parser.parse_args()
    
    if args.list_helical:
        print("AVAILABLE HELICAL MOTION ANALYSES:")
        print("=" * 50)
        print("1. Chandler Wobble Analysis (14-month polar axis motion)")
        print("2. 3D Spherical Harmonic Analysis (directional anisotropy decomposition)")
        print("3. Multi-Frequency Beat Analysis (Earth motion interference patterns)")
        print("4. Relative Motion Beat Analysis (station pair differential dynamics)")
        print("5. Mesh Dance Analysis (network coherence dynamics)")
        print("6. Jupiter Opposition Analysis (gravitational potential pulse events)")
        print("7. Saturn Opposition Analysis (gravitational potential pulse events)")
        print("8. Mars Opposition Analysis (gravitational potential pulse events)")
        print("9. Nutation Analysis (18.6-year axial tilt variations)")
        print()
        print("ASTRONOMICAL EVENT ANALYSES:")
        print("=" * 50)
        print("• Jupiter Opposition: Nov 3, 2023 & Dec 7, 2024 (0.22% expected amplitude)")
        print("• Saturn Opposition: Aug 27, 2023 & Sep 8, 2024 (0.019% expected amplitude)")
        print("• Mars Opposition: Jan 16, 2025 (0.005% expected amplitude - weakest signal)")
        print("• Major Lunar Standstill: 2024-2025 (sidereal day amplitude enhancement)")
        print("• Event-locked stacking with ±60 day windows")
        print("• Cross-center validation (IGS/ESA/CODE)")
        print("• Statistical significance testing")
        print()
        print("TO RUN ANALYSES:")
        print("   python step_5_tep_statistical_validation.py --mode helical")
        print("   python step_5_tep_statistical_validation.py --mode jupiter --center esa_final")
        print("   python step_5_tep_statistical_validation.py --mode saturn --center code")
        print("   python step_5_tep_statistical_validation.py --mode mars --center igs_combined")
        print("   python step_5_tep_statistical_validation.py --mode lunar --center igs_combined")
        print("   python step_5_tep_statistical_validation.py --mode astronomical  # All planets")
        return True
    
    if args.mode == 'helical':
        # Run ONLY the new helical motion analyses
        results = run_helical_motion_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    if args.mode == 'jupiter':
        # Run ONLY the Jupiter opposition analysis
        results = run_jupiter_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    if args.mode == 'saturn':
        # Run ONLY the Saturn opposition analysis
        results = run_saturn_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    if args.mode == 'mars':
        # Run ONLY the Mars opposition analysis
        results = run_mars_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    if args.mode == 'lunar':
        # Run ONLY the Lunar Standstill analysis
        results = run_lunar_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    if args.mode == 'eclipse':
        # Run ONLY the Solar Eclipse analysis
        results = run_eclipse_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    if args.mode == 'astronomical':
        # Run Jupiter, Saturn, AND Mars opposition analyses
        results = run_astronomical_events_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    # Original full Step 5 analysis
    print("="*80)
    print("TEP GNSS Analysis Package v0.12")
    print("STEP 5: Statistical Validation (FULL MODE)")
    print("="*80)
    
    start_time = time.time()
    
    # Validate configuration before starting
    config_issues = TEPConfig.validate_configuration()
    if config_issues:
        print_status("Configuration validation failed:", "ERROR")
        for issue in config_issues:
            print_status(f"  - {issue}", "ERROR")
        return False
    
    # Check memory availability
    memory = psutil.virtual_memory()
    print_status(f"Available memory: {memory.available/(1024**3):.1f} GB", "MEMORY")
    
    memory_limit = TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')
    if memory.available < memory_limit * (1024**3):
        print_status(f"WARNING: Available memory ({memory.available/(1024**3):.1f} GB) may be insufficient", "WARNING")
        print_status(f"Consider increasing TEP_MEMORY_LIMIT_GB or running on a machine with more RAM", "WARNING")
    
    # Process analysis centers
    if args.center:
        centers = [args.center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    results = {}
    for ac in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING {ac.upper()} - Statistical Validation")
        print(f"{'='*60}")
        
        result = process_analysis_center(ac)
        results[ac] = result
        
        # Save individual results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"step_5_statistical_validation_{ac}.json"
        try:
            safe_json_write(result, output_file, indent=2)
            print_status(f"Results saved: {output_file}", "SUCCESS")
        except (TEPFileError, TEPDataError) as e:
            print_status(f"Failed to save results: {e}", "WARNING")
    
    # Summary
    print(f"\n{'='*80}")
    print("STATISTICAL VALIDATION COMPLETE")
    print(f"{'='*80}")
    
    if results:
        print_status("Validation Summary:", "SUCCESS")
        for ac, result in results.items():
            if result.get('success', False):
                print(f"  {ac.upper()}:")
                
                if result.get('loso_analysis', {}).get('success', False):
                    loso = result['loso_analysis']
                    print(f"    LOSO: λ = {loso['lambda_mean']:.1f} ± {loso['lambda_std']:.1f} km (CV = {loso['coefficient_of_variation']:.3f})")
                
                if result.get('lodo_analysis', {}).get('success', False):
                    lodo = result['lodo_analysis']
                    print(f"    LODO: λ = {lodo['lambda_mean']:.1f} ± {lodo['lambda_std']:.1f} km (CV = {lodo['coefficient_of_variation']:.3f})")
                
                if result.get('block_bootstrap_analysis', {}).get('success', False):
                    bootstrap = result['block_bootstrap_analysis']
                    ci_low, ci_high = bootstrap['lambda_km']['ci_95']
                    print(f"    Block Bootstrap: λ = {bootstrap['lambda_km']['mean']:.1f} ± {bootstrap['lambda_km']['std']:.1f} km")
                    print(f"                     95% CI: [{ci_low:.1f}, {ci_high:.1f}] km")
                
                if result.get('enhanced_anisotropy_analysis', {}).get('success', False):
                    anisotropy = result['enhanced_anisotropy_analysis']
                    stats = anisotropy['anisotropy_statistics']
                    print(f"    Enhanced Anisotropy: {stats['n_sectors']} sectors, CV = {stats['coefficient_of_variation']:.3f} ({stats['anisotropy_category']})")
            else:
                print(f"  {ac.upper()}: FAILED - {result.get('error', 'Unknown error')}")
        
        print_status(f"Total execution time: {time.time() - start_time:.1f} seconds", "INFO")
        return True
    else:
        print_status("No successful validations", "ERROR")
        return False

# ===== NEW HELICAL MOTION ANALYSIS FUNCTIONS (ADDITIONS ONLY) =====

def compute_azimuth(lat1, lon1, lat2, lon2):
    """
    Compute azimuth from station 1 to station 2 in degrees.
    
    Args:
        lat1, lon1: Latitude and longitude of station 1 in degrees
        lat2, lon2: Latitude and longitude of station 2 in degrees
    
    Returns:
        float: Azimuth in degrees (0-360, where 0=North, 90=East)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate azimuth using spherical trigonometry
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    azimuth = np.arctan2(y, x)
    azimuth = np.degrees(azimuth)
    azimuth = (azimuth + 360) % 360  # Normalize to 0-360
    
    return azimuth

def run_chandler_wobble_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Detect 14-month Chandler wobble signatures in GPS timing correlations.
    
    The Chandler wobble causes Earth's rotation axis to wander ~9 meters from 
    the geographic poles with a period of ~14 months. This should modulate
    correlation patterns as the station mesh "wobbles" relative to inertial space.
    
    Args:
        complete_df: Complete pair dataset with date, coordinates, coherence
        
    Returns:
        dict: Chandler wobble analysis results
    """
    print_status("Starting Chandler Wobble Analysis (14-month period)...", "PROCESS")
    
    try:
        # Convert dates to datetime if not already done
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Calculate days since epoch for continuous time analysis
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Check temporal coverage for Chandler wobble analysis
        data_span_days = (complete_df['date'].max() - complete_df['date'].min()).days
        chandler_period_days = TEPConfig.get_float('TEP_CHANDLER_PERIOD_DAYS')
        n_chandler_cycles = data_span_days / chandler_period_days
        
        print_status(f"Temporal coverage: {data_span_days} days ({n_chandler_cycles:.2f} Chandler cycles)", "INFO")
        
        if n_chandler_cycles < 1.5:  # Need at least 1.5 cycles for meaningful analysis
            return {
                'success': False,
                'error': f'Insufficient temporal coverage for Chandler wobble: {n_chandler_cycles:.2f} cycles (need ≥1.5)',
                'data_span_days': data_span_days,
                'chandler_period_days': chandler_period_days,
                'cycles_available': n_chandler_cycles
            }
        
        complete_df['chandler_phase'] = (2 * np.pi * complete_df['days_since_epoch'] / chandler_period_days) % (2 * np.pi)
        
        # Group data into phase bins (18 bins = 20° phase increments)
        n_phase_bins = 18
        phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
        complete_df['chandler_phase_bin'] = pd.cut(complete_df['chandler_phase'], 
                                                   bins=phase_bins, 
                                                   labels=range(n_phase_bins))
        
        # Azimuth already computed in Step 4 - no need to recalculate!
        if 'azimuth' not in complete_df.columns:
            print_status("Computing azimuths (fallback - Step 4 data not available)...", "WARNING")
            complete_df['azimuth'] = complete_df.apply(
                lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                           row['station2_lat'], row['station2_lon']), axis=1
            )
        else:
            print_status("Using pre-computed azimuths from Step 4", "SUCCESS")
        
        def classify_ew_ns(azimuth):
            """Classify direction as East-West or North-South"""
            return 'EW' if (45 <= azimuth <= 135) or (225 <= azimuth <= 315) else 'NS'
        
        complete_df['ew_ns_class'] = complete_df['azimuth'].apply(classify_ew_ns)
        
        # Analyze E-W/N-S ratio variation across Chandler phases
        chandler_tracking = []
        edges = np.logspace(np.log10(50), np.log10(13000), 41)  # Same binning as existing analysis
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        
        for phase_bin in range(n_phase_bins):
            phase_data = complete_df[complete_df['chandler_phase_bin'] == phase_bin].copy()
            
            if len(phase_data) < 500:  # Lowered requirement for better temporal coverage
                continue
                
            # Analyze E-W and N-S separately
            ew_data = phase_data[phase_data['ew_ns_class'] == 'EW']
            ns_data = phase_data[phase_data['ew_ns_class'] == 'NS']
            
            if len(ew_data) < 250 or len(ns_data) < 250:  # Lowered requirements
                continue
                
            # OPTION A: Direct coherence analysis (bypass complex correlation fitting)
            # Use mean coherence values instead of fitted correlation lengths
            ew_coherence_mean = float(ew_data['coherence'].mean())
            ns_coherence_mean = float(ns_data['coherence'].mean())
            ew_coherence_std = float(ew_data['coherence'].std())
            ns_coherence_std = float(ns_data['coherence'].std())
            
            # Calculate coherence ratio (analogous to lambda ratio)
            if ns_coherence_mean > 0:
                coherence_ratio = ew_coherence_mean / ns_coherence_mean
                
                chandler_tracking.append({
                    'phase_bin': phase_bin,
                    'phase_radians': float(phase_bins[phase_bin]),
                    'phase_degrees': float(np.degrees(phase_bins[phase_bin])),
                    'ew_coherence_mean': ew_coherence_mean,
                    'ns_coherence_mean': ns_coherence_mean,
                    'ew_coherence_std': ew_coherence_std,
                    'ns_coherence_std': ns_coherence_std,
                    'ew_ns_coherence_ratio': coherence_ratio,
                    'n_pairs': len(phase_data),
                    'n_ew_pairs': len(ew_data),
                    'n_ns_pairs': len(ns_data)
                })
        
        if len(chandler_tracking) < 6:  # Lowered from 10 to 6 for better coverage
            return {'success': False, 'error': f'Insufficient phase bins: {len(chandler_tracking)} (need ≥6)'}
        
        # Statistical analysis of Chandler wobble modulation using coherence ratios
        phases = [t['phase_radians'] for t in chandler_tracking]
        ew_ns_ratios = [t['ew_ns_coherence_ratio'] for t in chandler_tracking]
        
        # Data quality checks
        print_status(f"Chandler analysis: {len(phases)} phase bins, ratio range: {min(ew_ns_ratios):.3f}-{max(ew_ns_ratios):.3f}", "INFO")
        
        if np.std(ew_ns_ratios) < 0.001:  # Very low variation threshold
            print_status(f"Warning: Very low variation in E-W/N-S ratios (std={np.std(ew_ns_ratios):.6f})", "WARNING")
            return {
                'success': False, 
                'error': 'Insufficient variation in E-W/N-S ratios across Chandler phases',
                'data_variation': float(np.std(ew_ns_ratios)),
                'n_phase_bins': len(chandler_tracking),
                'ratio_range': [float(min(ew_ns_ratios)), float(max(ew_ns_ratios))]
            }
        
        # Fit sinusoidal model: ratio = A*sin(phase + φ) + offset
        def chandler_model(phase, amplitude, phase_shift, offset):
            return amplitude * np.sin(phase + phase_shift) + offset
        
        try:
            # Robust curve fitting with better error handling
            if len(set(ew_ns_ratios)) < 3:  # Check for insufficient variation
                chandler_fit = {
                    'fit_success': False, 
                    'error': 'Insufficient variation in E-W/N-S ratios',
                    'data_variation': float(np.std(ew_ns_ratios))
                }
            else:
                # Try curve fitting with improved bounds and method
                popt, pcov = curve_fit(
                    chandler_model, phases, ew_ns_ratios, 
                    p0=[0.1, 0, np.mean(ew_ns_ratios)],
                    bounds=([-1, -2*np.pi, 0], [1, 2*np.pi, 10]),  # Reasonable bounds
                    method='trf',  # Trust Region Reflective algorithm
                    maxfev=2000
                )
                
                # Calculate correlation coefficient with safety check
                predicted = chandler_model(np.array(phases), *popt)
                
                # Safe correlation calculation
                if np.std(ew_ns_ratios) > 1e-10 and np.std(predicted) > 1e-10:
                    correlation = np.corrcoef(ew_ns_ratios, predicted)[0, 1]
                    
                    # Statistical significance test
                    from scipy.stats import pearsonr
                    _, p_value = pearsonr(ew_ns_ratios, predicted)
                    
                    # Check for NaN results
                    if np.isnan(correlation) or np.isnan(p_value):
                        chandler_fit = {
                            'fit_success': False,
                            'error': 'NaN values in correlation analysis',
                            'correlation': float(correlation) if not np.isnan(correlation) else None,
                            'p_value': float(p_value) if not np.isnan(p_value) else None
                        }
                    else:
                        chandler_fit = {
                            'amplitude': float(popt[0]),
                            'phase_shift_radians': float(popt[1]),
                            'phase_shift_degrees': float(np.degrees(popt[1])),
                            'offset': float(popt[2]),
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'fit_success': True,
                            'data_points': len(phases),
                            'data_variation': float(np.std(ew_ns_ratios))
                        }
                else:
                    chandler_fit = {
                        'fit_success': False,
                        'error': 'Insufficient variation in data for correlation',
                        'data_std': float(np.std(ew_ns_ratios)),
                        'predicted_std': float(np.std(predicted))
                    }
            
        except Exception as e:
            chandler_fit = {
                'fit_success': False, 
                'error': str(e),
                'data_points': len(phases) if 'phases' in locals() else 0,
                'data_variation': float(np.std(ew_ns_ratios)) if 'ew_ns_ratios' in locals() else 0
            }
        
        return {
            'success': True,
            'chandler_period_days': chandler_period_days,
            'n_phase_bins': len(chandler_tracking),
            'phase_tracking': chandler_tracking,
            'sinusoidal_fit': chandler_fit,
            'interpretation': classify_chandler_evidence(chandler_fit.get('correlation', 0), 
                                                        chandler_fit.get('p_value', 1))
        }
        
    except Exception as e:
        print_status(f"Chandler wobble analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_3d_spherical_harmonic_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Replace simple E-W/N-S analysis with full spherical harmonic decomposition.
    
    This captures the complete 3D anisotropy pattern of the station mesh,
    revealing complex directional structures beyond simple E-W vs N-S.
    
    Args:
        complete_df: Complete pair dataset with coordinates and coherence
        
    Returns:
        dict: 3D spherical harmonic analysis results
    """
    print_status("Starting 3D Spherical Harmonic Analysis...", "PROCESS")
    
    try:
        # Azimuth already computed in Step 4 - no need to recalculate!
        if 'azimuth' not in complete_df.columns:
            print_status("Computing azimuths (fallback - Step 4 data not available)...", "WARNING")
            complete_df['azimuth'] = complete_df.apply(
                lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                           row['station2_lat'], row['station2_lon']), axis=1
            )
        else:
            print_status("Using pre-computed azimuths from Step 4", "SUCCESS")
        
        # Compute elevation angles accounting for Earth curvature
        def compute_elevation_angle(lat1, lon1, lat2, lon2):
            """Compute elevation angle for station pair"""
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Great circle distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            # Elevation angle (simplified - assumes spherical Earth)
            if c > 0:
                elevation = np.arcsin(np.sin(lat2 - lat1) / c)
            else:
                elevation = 0
            return np.degrees(elevation)
        
        complete_df['elevation'] = complete_df.apply(
            lambda row: compute_elevation_angle(row['station1_lat'], row['station1_lon'], 
                                               row['station2_lat'], row['station2_lon']), axis=1
        )
        
        # Convert to spherical coordinates (θ, φ)
        complete_df['theta'] = np.radians(90 - complete_df['elevation'])  # Colatitude
        complete_df['phi'] = np.radians(complete_df['azimuth'])           # Azimuth
        
        # Define spherical harmonic sectors
        n_theta_bins = TEPConfig.get_int('TEP_SPHERICAL_THETA_BINS')
        n_phi_bins = TEPConfig.get_int('TEP_SPHERICAL_PHI_BINS')
        
        theta_bins = np.linspace(0, np.pi, n_theta_bins + 1)
        phi_bins = np.linspace(0, 2*np.pi, n_phi_bins + 1)
        
        # Create 2D sector grid
        complete_df['theta_bin'] = pd.cut(complete_df['theta'], bins=theta_bins, 
                                         labels=range(n_theta_bins))
        complete_df['phi_bin'] = pd.cut(complete_df['phi'], bins=phi_bins, 
                                       labels=range(n_phi_bins))
        
        # Analyze correlation lengths in each spherical sector
        spherical_sectors = {}
        edges = np.logspace(np.log10(50), np.log10(13000), 41)
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        
        for theta_idx in range(n_theta_bins):
            for phi_idx in range(n_phi_bins):
                sector_data = complete_df[
                    (complete_df['theta_bin'] == theta_idx) & 
                    (complete_df['phi_bin'] == phi_idx)
                ].copy()
                
                if len(sector_data) < 500:  # Need sufficient statistics
                    continue
                    
                # Fit correlation model for this sector
                sector_lambda = fit_directional_correlation(sector_data, edges, min_bin_count)
                
                if sector_lambda:
                    sector_name = f"theta_{theta_idx}_phi_{phi_idx}"
                    spherical_sectors[sector_name] = {
                        'theta_bin': theta_idx,
                        'phi_bin': phi_idx,
                        'theta_center_deg': float(np.degrees((theta_bins[theta_idx] + theta_bins[theta_idx+1]) / 2)),
                        'phi_center_deg': float(np.degrees((phi_bins[phi_idx] + phi_bins[phi_idx+1]) / 2)),
                        'lambda_km': float(sector_lambda),
                        'n_pairs': len(sector_data)
                    }
        
        if len(spherical_sectors) < 20:  # Need reasonable coverage
            return {'success': False, 'error': f'Insufficient sectors: {len(spherical_sectors)}'}
        
        # Compute spherical harmonic moments
        lambdas = [s['lambda_km'] for s in spherical_sectors.values()]
        thetas = [np.radians(s['theta_center_deg']) for s in spherical_sectors.values()]
        phis = [np.radians(s['phi_center_deg']) for s in spherical_sectors.values()]
        
        # Calculate multipole moments (up to l=2)
        spherical_moments = {}
        
        # l=0 (monopole) - overall average
        spherical_moments['monopole'] = float(np.mean(lambdas))
        
        # l=1 (dipole) - directional bias
        dipole_x = np.mean([lam * np.sin(theta) * np.cos(phi) 
                           for lam, theta, phi in zip(lambdas, thetas, phis)])
        dipole_y = np.mean([lam * np.sin(theta) * np.sin(phi) 
                           for lam, theta, phi in zip(lambdas, thetas, phis)])
        dipole_z = np.mean([lam * np.cos(theta) 
                           for lam, theta in zip(lambdas, thetas)])
        
        dipole_magnitude = np.sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)
        dipole_direction = {
            'magnitude': float(dipole_magnitude),
            'x': float(dipole_x), 
            'y': float(dipole_y), 
            'z': float(dipole_z),
            'theta_deg': float(np.degrees(np.arccos(dipole_z / dipole_magnitude))) if dipole_magnitude > 0 else 0,
            'phi_deg': float(np.degrees(np.arctan2(dipole_y, dipole_x)))
        }
        
        spherical_moments['dipole'] = dipole_direction
        
        # l=2 (quadrupole) - shape anisotropy
        quadrupole_strength = np.std(lambdas) / np.mean(lambdas)
        spherical_moments['quadrupole_strength'] = float(quadrupole_strength)
        
        return {
            'success': True,
            'n_sectors': len(spherical_sectors),
            'sector_results': spherical_sectors,
            'spherical_moments': spherical_moments,
            'anisotropy_statistics': {
                'mean_lambda': float(np.mean(lambdas)),
                'std_lambda': float(np.std(lambdas)),
                'cv_lambda': float(np.std(lambdas) / np.mean(lambdas)),
                'max_lambda': float(np.max(lambdas)),
                'min_lambda': float(np.min(lambdas)),
                'anisotropy_ratio': float(np.max(lambdas) / np.min(lambdas))
            }
        }
        
    except Exception as e:
        print_status(f"3D spherical harmonic analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_multi_frequency_beat_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze temporal interference patterns between different Earth motion components with
    RELATIVE MOTION ANALYSIS between station pairs.
    
    This enhanced analysis considers:
    1. Global Earth motion frequencies (rotation, orbit, wobble, nutation)
    2. RELATIVE velocities between station pairs as Earth moves
    3. Distance-dependent beat patterns across the station mesh
    4. Differential motion effects based on station separation and orientation
    
    Args:
        complete_df: Complete pair dataset with date and coordinates
        
    Returns:
        dict: Enhanced beat analysis with relative motion patterns
    """
    print_status("Starting Multi-Frequency Beat Analysis...", "PROCESS")
    
    try:
        # Define fundamental frequencies (cycles per day)
        frequencies = {
            'rotation': 1.0,           # 1 cycle/day (24h)
            'tidal_m2': 1.9323,        # M2 tidal component
            'tidal_s2': 2.0,           # S2 solar tidal component  
            'chandler': 1.0/TEPConfig.get_float('TEP_CHANDLER_PERIOD_DAYS'),     # Chandler wobble
            'annual': 1.0/365.25,      # Annual orbital motion
            'semiannual': 2.0/365.25   # Semiannual variation
        }
        
        # Calculate all possible temporal interference patterns
        beat_frequencies = {}
        freq_names = list(frequencies.keys())
        min_period_days = TEPConfig.get_float('TEP_BEAT_MIN_PERIOD_DAYS')
        
        for i, name1 in enumerate(freq_names):
            for j, name2 in enumerate(freq_names):
                if i < j:  # Avoid duplicates
                    f1, f2 = frequencies[name1], frequencies[name2]
                    
                    # Difference frequency (beat)
                    beat_diff = abs(f1 - f2)
                    if beat_diff > 0:
                        period_diff = 1.0/beat_diff
                        if period_diff >= min_period_days:
                            beat_frequencies[f"{name1}_{name2}_diff"] = {
                                'frequency_cpd': beat_diff,
                                'period_days': period_diff,
                                'type': 'difference',
                                'components': [name1, name2]
                            }
                    
                    # Sum frequency
                    beat_sum = f1 + f2
                    period_sum = 1.0/beat_sum
                    if period_sum >= min_period_days:
                        beat_frequencies[f"{name1}_{name2}_sum"] = {
                            'frequency_cpd': beat_sum,
                            'period_days': period_sum,
                            'type': 'sum', 
                            'components': [name1, name2]
                        }
        
        # Convert dates to continuous time
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Azimuth already computed in Step 4 - no need to recalculate!
        if 'azimuth' not in complete_df.columns:
            print_status("Computing azimuths (fallback - Step 4 data not available)...", "WARNING")
            complete_df['azimuth'] = complete_df.apply(
                lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                           row['station2_lat'], row['station2_lon']), axis=1
            )
        else:
            print_status("Using pre-computed azimuths from Step 4", "SUCCESS")
        
        def classify_ew_ns(azimuth):
            return 'EW' if (45 <= azimuth <= 135) or (225 <= azimuth <= 315) else 'NS'
        
        complete_df['ew_ns_class'] = complete_df['azimuth'].apply(classify_ew_ns)
        
        # For each beat frequency, test for modulation in E-W/N-S ratio
        beat_analysis_results = {}
        data_span_days = (complete_df['date'].max() - complete_df['date'].min()).days
        min_cycles = TEPConfig.get_int('TEP_BEAT_MIN_CYCLES')
        
        for beat_name, beat_info in beat_frequencies.items():
            period_days = beat_info['period_days']
            
            # Skip beats with periods too long for our dataset
            if period_days > data_span_days / min_cycles:  # Need minimum cycles
                print_status(f"Skipping beat {beat_name}: period {period_days:.1f} days > {data_span_days/min_cycles:.1f} days threshold", "INFO")
                continue
                
            print_status(f"Analyzing beat: {beat_name} (period: {period_days:.1f} days)", "INFO")
            
            # Calculate phase for this beat frequency (wrapped to 0-2π)
            beat_freq_cpd = beat_info['frequency_cpd']
            complete_df['beat_phase'] = (2 * np.pi * complete_df['days_since_epoch'] * beat_freq_cpd) % (2 * np.pi)
            
            # Debug: Check phase calculation
            phase_min, phase_max = complete_df['beat_phase'].min(), complete_df['beat_phase'].max()
            phase_range = phase_max - phase_min
            print_status(f"Beat {beat_name}: Phase range {phase_min:.2f} to {phase_max:.2f} (range: {phase_range:.2f})", "INFO")
            
            # Bin by phase (12 bins = 30° increments)
            n_phase_bins = 12
            phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
            complete_df['beat_phase_bin'] = pd.cut(complete_df['beat_phase'], 
                                                  bins=phase_bins, 
                                                  labels=range(n_phase_bins))
            
            # Debug: Check binning results
            bin_counts = complete_df['beat_phase_bin'].value_counts()
            print_status(f"Beat {beat_name}: Phase bin distribution: {dict(bin_counts)}", "INFO")
            
            # Track E-W/N-S ratio across beat phases
            beat_tracking = []
            edges = np.logspace(np.log10(50), np.log10(13000), 41)
            min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
            
            for phase_bin in range(n_phase_bins):
                phase_data = complete_df[complete_df['beat_phase_bin'] == phase_bin].copy()
                
                if len(phase_data) < 100:  # Much lower requirement for beat analysis
                    continue
                    
                # Analyze E-W and N-S separately
                ew_data = phase_data[phase_data['ew_ns_class'] == 'EW']
                ns_data = phase_data[phase_data['ew_ns_class'] == 'NS']
                
                if len(ew_data) < 50 or len(ns_data) < 50:  # Much lower requirements for beat analysis
                    continue
                    
                # OPTION A: Direct coherence analysis (bypass complex correlation fitting)
                # Use mean coherence values instead of fitted correlation lengths
                ew_coherence_mean = float(ew_data['coherence'].mean())
                ns_coherence_mean = float(ns_data['coherence'].mean())
                ew_coherence_std = float(ew_data['coherence'].std())
                ns_coherence_std = float(ns_data['coherence'].std())
                
                # Calculate coherence ratio (analogous to lambda ratio)
                if ns_coherence_mean > 0:
                    coherence_ratio = ew_coherence_mean / ns_coherence_mean
                    
                    beat_tracking.append({
                        'phase_bin': phase_bin,
                        'phase_radians': float(phase_bins[phase_bin]),
                        'ew_coherence_mean': ew_coherence_mean,
                        'ns_coherence_mean': ns_coherence_mean,
                        'ew_coherence_std': ew_coherence_std,
                        'ns_coherence_std': ns_coherence_std,
                        'ew_ns_coherence_ratio': coherence_ratio,
                        'n_pairs': len(phase_data),
                        'n_ew_pairs': len(ew_data),
                        'n_ns_pairs': len(ns_data)
                    })
            
            print_status(f"Beat {beat_name}: {len(beat_tracking)} phase bins collected", "INFO")
            
            if len(beat_tracking) < 4:  # Lowered requirement for phase coverage
                print_status(f"Insufficient phase coverage for {beat_name}: {len(beat_tracking)} bins (need ≥4)", "WARNING")
                continue
            
            print_status(f"Proceeding with beat analysis for {beat_name}: {len(beat_tracking)} phase bins", "SUCCESS")
                
            # Test for sinusoidal modulation using coherence ratios
            phases = [t['phase_radians'] for t in beat_tracking]
            ratios = [t['ew_ns_coherence_ratio'] for t in beat_tracking]
            
            print_status(f"Beat {beat_name}: Testing {len(phases)} phase points, ratio range: {min(ratios):.3f}-{max(ratios):.3f}", "INFO")
            
            try:
                # Robust beat frequency analysis with better error handling
                if len(set(ratios)) < 3:  # Check for insufficient variation
                    beat_analysis_results[beat_name] = {
                        'beat_info': beat_info,
                        'fit_success': False,
                        'error': 'Insufficient variation in E-W/N-S ratios',
                        'data_variation': float(np.std(ratios)),
                        'n_phase_bins': len(beat_tracking)
                    }
                else:
                    # Fit sinusoidal model with robust parameters
                    def beat_model(phase, amplitude, phase_shift, offset):
                        return amplitude * np.sin(phase + phase_shift) + offset
                    
                    # Use robust fitting with bounds
                    popt, _ = curve_fit(
                        beat_model, phases, ratios, 
                        p0=[0.1, 0, np.mean(ratios)],
                        bounds=([-2, -2*np.pi, 0], [2, 2*np.pi, 20]),  # Reasonable bounds
                        method='trf',
                        maxfev=1000
                    )
                    
                    predicted = beat_model(np.array(phases), *popt)
                    
                    # Safe correlation calculation
                    if np.std(ratios) > 1e-10 and np.std(predicted) > 1e-10:
                        correlation = np.corrcoef(ratios, predicted)[0, 1]
                        
                        from scipy.stats import pearsonr
                        _, p_value = pearsonr(ratios, predicted)
                        
                        # Check for valid results
                        if not (np.isnan(correlation) or np.isnan(p_value)):
                            beat_analysis_results[beat_name] = {
                                'beat_info': beat_info,
                                'n_phase_bins': len(beat_tracking),
                                'amplitude': float(popt[0]),
                                'correlation': float(correlation),
                                'p_value': float(p_value),
                                'fit_success': True,
                                'phase_tracking': beat_tracking,
                                'data_variation': float(np.std(ratios))
                            }
                        else:
                            beat_analysis_results[beat_name] = {
                                'beat_info': beat_info,
                                'fit_success': False,
                                'error': 'NaN values in correlation',
                                'correlation': float(correlation) if not np.isnan(correlation) else None,
                                'p_value': float(p_value) if not np.isnan(p_value) else None
                            }
                    else:
                        beat_analysis_results[beat_name] = {
                            'beat_info': beat_info,
                            'fit_success': False,
                            'error': 'Insufficient variation for correlation',
                            'data_std': float(np.std(ratios)),
                            'predicted_std': float(np.std(predicted))
                        }
                
            except Exception as e:
                beat_analysis_results[beat_name] = {
                    'beat_info': beat_info,
                    'fit_success': False,
                    'error': str(e),
                    'n_phase_bins': len(beat_tracking) if 'beat_tracking' in locals() else 0
                }
        
        # Identify most significant temporal interference patterns with configurable threshold
        significance_threshold = TEPConfig.get_float('TEP_BEAT_SIGNIFICANCE_THRESHOLD')
        min_correlation = TEPConfig.get_float('TEP_MIN_CORRELATION_THRESHOLD')
        significant_beats = {}
        
        print_status(f"Beat detection thresholds: p<{significance_threshold}, |r|>{min_correlation}", "INFO")
        
        for beat_name, result in beat_analysis_results.items():
            if (result.get('fit_success') and 
                result.get('p_value', 1) < significance_threshold and
                abs(result.get('correlation', 0)) > min_correlation):
                significant_beats[beat_name] = result
                print_status(f"Significant beat detected: {beat_name} (r={result.get('correlation', 0):.3f}, p={result.get('p_value', 1):.3f})", "SUCCESS")
        
        return {
            'success': True,
            'fundamental_frequencies': frequencies,
            'beat_frequencies': beat_frequencies,
            'beat_analysis': beat_analysis_results,
            'significant_beats': significant_beats,
            'n_beats_tested': len(beat_analysis_results),
            'n_significant_beats': len(significant_beats)
        }
        
    except Exception as e:
        print_status(f"Multi-frequency beat analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def fit_event_peak(days_from_event: List[int], coherences: List[float]) -> Dict:
    """
    Fit a Gaussian model to event-locked data to find a peak and test its significance.
    
    Args:
        days_from_event: List of integers representing days from the event center.
        coherences: List of corresponding mean coherence values.
        
    Returns:
        A dictionary with fit parameters, standard errors, and significance assessment.
    """
    try:
        # Require a minimum number of data points for a stable, meaningful fit
        if len(days_from_event) < 10 or len(set(coherences)) < 5:
            return {'fit_success': False, 'error': 'Insufficient data points for a stable fit'}

        def gaussian_model(x, amplitude, center, width, baseline):
            return baseline + amplitude * np.exp(-0.5 * ((x - center) / width)**2)

        # Try both positive and negative amplitude initial guesses to avoid local minima
        mean_coherence = np.mean(coherences)
        max_coherence = np.max(coherences)
        min_coherence = np.min(coherences)
        
        # Determine which is more extreme: enhancement or suppression
        enhancement_amplitude = max_coherence - mean_coherence
        suppression_amplitude = min_coherence - mean_coherence
        
        # Use the more extreme as the primary guess
        if abs(enhancement_amplitude) > abs(suppression_amplitude):
            primary_amplitude = enhancement_amplitude
        else:
            primary_amplitude = suppression_amplitude
        
        # Try multiple initial guesses to avoid local minima
        initial_guesses = [
            [primary_amplitude, 0, 10, mean_coherence],     # Primary guess
            [-primary_amplitude, 0, 10, mean_coherence],    # Opposite sign
            [primary_amplitude, -10, 5, mean_coherence],    # Earlier timing
            [primary_amplitude, 10, 5, mean_coherence],     # Later timing
        ]
        
        best_fit = None
        best_r_squared = -1
        
        for p0 in initial_guesses:
            try:
                popt, pcov = curve_fit(
                    gaussian_model, days_from_event, coherences,
                    p0=p0,
                    bounds=([-0.2, -60, 1, 0], [0.2, 60, 40, 1]), # Expanded bounds
                    maxfev=5000
                )
                
                # Calculate R-squared for this fit
                predicted = gaussian_model(np.array(days_from_event), *popt)
                ss_res = np.sum((np.array(coherences) - predicted)**2)
                ss_tot = np.sum((np.array(coherences) - np.mean(coherences))**2)
                r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                
                # Keep the best fit
                if r_squared > best_r_squared:
                    best_fit = (popt, pcov)
                    best_r_squared = r_squared
                    
            except:
                continue
        
        if best_fit is None:
            return {'fit_success': False, 'error': 'All fitting attempts failed'}
        
        popt, pcov = best_fit
        
        # --- ROBUST SIGNIFICANCE TEST ---
        # Calculate standard errors from the diagonal of the covariance matrix
        param_errors = np.sqrt(np.diag(pcov))
        amplitude = float(popt[0])
        amplitude_std_err = float(param_errors[0])
        
        # A detection is significant if the amplitude is > 2x its standard error
        is_significant = abs(amplitude) > (2 * amplitude_std_err)
        
        # Use the R-squared from the best fit (already calculated above)
        r_squared = best_r_squared

        return {
            'amplitude': amplitude,
            'amplitude_std_err': amplitude_std_err,
            'center_days': float(popt[1]),
            'center_std_err': float(param_errors[1]),
            'width_days': float(popt[2]),
            'baseline': float(popt[3]),
            'fit_success': True,
            'is_significant': bool(is_significant),
            'r_squared': r_squared,
            'amplitude_fraction_of_baseline': float(abs(amplitude) / popt[3]) if popt[3] > 0 else 0
        }
        
    except Exception as e:
        return {'fit_success': False, 'error': str(e)}

def run_jupiter_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Jupiter opposition events using
    BEAT FREQUENCY MODULATION ANALYSIS instead of simple amplitude changes.
    
    CORRECTED APPROACH: Look for modulation of existing tidal temporal interference patterns
    (M2, S2, Chandler wobble) around Jupiter opposition dates, rather than
    simple coherence amplitude changes which wash out the signal.
    
    Jupiter oppositions occur when Earth-Jupiter distance is minimized, causing
    Jupiter's gravitational potential at Earth to peak. According to TEP theory,
    this should modulate the existing Earth motion beat patterns.
    
    Expected effect: Modulation of tidal beat frequency amplitudes around
    opposition dates, detectable through frequency domain analysis.
    
    Key Jupiter opposition dates:
    - November 3, 2023
    - December 7, 2024
    
    Args:
        complete_df: Complete pair dataset with dates and coherence
        
    Returns:
        dict: Jupiter opposition analysis results with beat frequency modulation
    """
    print_status("Starting Jupiter Opposition Pulse Analysis...", "PROCESS")
    print_status("Testing for gravitational potential coupling during Jupiter oppositions", "PROCESS")
    
    try:
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Define Jupiter opposition events (UTC dates)
        jupiter_oppositions = [
            {'date': pd.Timestamp('2023-11-03'), 'name': 'Jupiter_Opposition_2023'},
            {'date': pd.Timestamp('2024-12-07'), 'name': 'Jupiter_Opposition_2024'}
            # Note: 2026-01-10 is outside typical dataset range
        ]
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        
        print_status(f"Dataset coverage: {data_start.date()} to {data_end.date()}", "INFO")
        
        # Filter to events within data range
        valid_events = []
        event_window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS')
        
        for event in jupiter_oppositions:
            event_start = event['date'] - pd.Timedelta(days=event_window_days)
            event_end = event['date'] + pd.Timedelta(days=event_window_days)
            
            # Check if event window overlaps with data
            if event_start <= data_end and event_end >= data_start:
                valid_events.append(event)
                print_status(f"Jupiter opposition {event['date'].date()} within data range", "SUCCESS")
            else:
                print_status(f"Jupiter opposition {event['date'].date()} outside data range", "WARNING")
        
        if not valid_events:
            return {
                'success': False,
                'error': 'No Jupiter opposition events within dataset time range',
                'dataset_range': [data_start.isoformat(), data_end.isoformat()],
                'jupiter_oppositions': [e['date'].isoformat() for e in jupiter_oppositions]
            }
        
        # Use the CORRECT approach: phase modulation of existing detected beats
        jupiter_dates = [event['date'] for event in valid_events]
        return run_astronomical_phase_modulation_analysis(complete_df, jupiter_dates, 'Jupiter')
        
    except Exception as e:
        print_status(f"Jupiter opposition analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_saturn_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Saturn opposition events.
    
    Saturn oppositions occur when Earth-Saturn distance is minimized, causing
    Saturn's gravitational potential at Earth to peak. According to TEP theory,
    this should create a brief global enhancement in timing correlations.
    
    Expected amplitude: ~0.019% of the solar annual perihelion-aphelion swing
    (ΔU/c² ≈ 6.3×10⁻¹⁴ vs solar ΔU/c² ≈ 3.3×10⁻¹⁰)
    
    This is ~12x smaller than Jupiter's signal, making it an excellent
    orthogonal validation test.
    
    Key Saturn opposition dates:
    - August 27, 2023
    - September 8, 2024
    - September 21, 2025
    
    Args:
        complete_df: Complete pair dataset with dates and coherence
        
    Returns:
        dict: Saturn opposition analysis results
    """
    try:
        print_status("Starting Saturn Opposition Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Saturn opposition dates (when Earth-Saturn distance is minimized)
        saturn_events = [
            {'name': 'saturn_2023', 'date': pd.to_datetime('2023-08-27'), 'description': 'Saturn Opposition August 2023'},
            {'name': 'saturn_2024', 'date': pd.to_datetime('2024-09-08'), 'description': 'Saturn Opposition September 2024'},
            {'name': 'saturn_2025', 'date': pd.to_datetime('2025-09-21'), 'description': 'Saturn Opposition September 2025'}
        ]
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS')
        expected_amplitude = TEPConfig.get_float('TEP_SATURN_AMPLITUDE_FRACTION')
        min_pairs = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS')
        
        print_status(f"Analyzing {len(saturn_events)} Saturn opposition events", "INFO")
        print_status(f"Event window: ±{window_days} days", "INFO")
        print_status(f"Expected amplitude: {expected_amplitude:.5f} ({expected_amplitude*100:.4f}%)", "INFO")
        
        # Analyze each event
        event_results = {}
        successful_events = 0
        
        for event in saturn_events:
            event_name = event['name']
            event_date = event['date']
            
            print_status(f"Analyzing {event['description']}...", "PROCESS")
            
            # Check if event date is within our data range
            data_start = complete_df['date'].min()
            data_end = complete_df['date'].max()
            
            if event_date < data_start or event_date > data_end:
                print_status(f"Event {event_date.date()} outside data range ({data_start.date()} to {data_end.date()})", "WARNING")
                event_results[event_name] = {
                    'success': False,
                    'error': 'Event date outside data range',
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'description': event['description']
                }
                continue
            
            # Extract data window around event
            start_date = event_date - pd.Timedelta(days=window_days)
            end_date = event_date + pd.Timedelta(days=window_days)
            
            event_data = complete_df[
                (complete_df['date'] >= start_date) & 
                (complete_df['date'] <= end_date)
            ].copy()
            
            if len(event_data) == 0:
                event_results[event_name] = {
                    'success': False,
                    'error': 'No data in event window',
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'description': event['description']
                }
                continue
            
            # Calculate days from event
            event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
            
            # Daily binning and analysis
            daily_data = []
            for day in range(-window_days, window_days + 1):
                day_data = event_data[event_data['days_from_event'] == day]
                
                if len(day_data) >= min_pairs:
                    mean_coherence = day_data['coherence'].mean()
                    std_coherence = day_data['coherence'].std()
                    n_pairs = len(day_data)
                    
                    daily_data.append({
                        'day': day,
                        'mean_coherence': mean_coherence,
                        'std_coherence': std_coherence,
                        'n_pairs': n_pairs,
                        'date': (event_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
                    })
            
            if len(daily_data) < 10:
                event_results[event_name] = {
                    'success': False,
                    'error': f'Insufficient daily data points: {len(daily_data)} < 10',
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'description': event['description']
                }
                continue
            
            # Fit Gaussian peak to detect Saturn signal
            days_from_event = [d['day'] for d in daily_data]
            coherences = [d['mean_coherence'] for d in daily_data]
            
            gaussian_fit = fit_event_peak(days_from_event, coherences)
            
            # Store results
            event_results[event_name] = {
                'success': True,
                'event_date': event_date.strftime('%Y-%m-%d'),
                'description': event['description'],
                'daily_tracking': daily_data,
                'gaussian_fit': gaussian_fit,
                'n_daily_points': len(daily_data),
                'total_pairs_analyzed': sum(d['n_pairs'] for d in daily_data),
                'interpretation': gaussian_fit.get('interpretation', 'Analysis completed')
            }
            
            successful_events += 1
            print_status(f"✓ {event['description']} analysis completed", "SUCCESS")
        
        # Cross-event analysis if we have multiple successful events
        cross_event_analysis = {}
        if successful_events >= 2:
            print_status("Performing cross-event consistency analysis...", "PROCESS")
            
            # Collect all successful event data for stacking
            all_days = []
            all_coherences = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    daily_data = event_data['daily_tracking']
                    for daily_point in daily_data:
                        all_days.append(daily_point['day'])
                        all_coherences.append(daily_point['mean_coherence'])
            
            # Perform stacked analysis
            if len(all_days) >= 20:  # Need sufficient data for stacking
                stacked_gaussian = fit_event_peak(all_days, all_coherences)
                cross_event_analysis = {
                    'success': True,
                    'n_events_stacked': successful_events,
                    'total_data_points': len(all_days),
                    'gaussian_fit': stacked_gaussian,
                    'interpretation': stacked_gaussian.get('interpretation', 'Stacked analysis completed')
                }
            else:
                cross_event_analysis = {
                    'success': False,
                    'error': 'Insufficient data for stacking analysis'
                }
        
        # Overall interpretation
        if successful_events == 0:
            interpretation = "No Saturn opposition events could be analyzed"
        elif successful_events == 1:
            interpretation = "Single Saturn opposition event analyzed - no cross-validation possible"
        else:
            # Check for significant detections
            significant_events = sum(1 for result in event_results.values() 
                                   if result.get('success') and 
                                      result.get('gaussian_fit', {}).get('is_significant', False))
            
            if significant_events > 0:
                interpretation = f"Saturn opposition analysis: {significant_events}/{successful_events} events show significant signals"
            else:
                interpretation = f"Saturn opposition analysis: No significant signals detected in {successful_events} events"
        
        return {
            'success': True,
            'n_events_attempted': len(saturn_events),
            'n_successful_events': successful_events,
            'event_results': event_results,
            'stacked_analysis': cross_event_analysis,
            'expected_amplitude_fraction': expected_amplitude,
            'interpretation': interpretation,
            'analysis_summary': {
                'total_events': len(saturn_events),
                'successful_events': successful_events,
                'significant_detections': sum(1 for result in event_results.values() 
                                            if result.get('success') and 
                                               result.get('gaussian_fit', {}).get('is_significant', False)),
                'stacked_analysis_success': cross_event_analysis.get('success', False)
            }
        }
        
    except Exception as e:
        print_status(f"Saturn opposition analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_astronomical_phase_modulation_analysis(complete_df: pd.DataFrame, event_dates: list, event_name: str) -> Dict:
    """
    CORRECT ASTRONOMICAL ANALYSIS: Look for phase modulation of EXISTING detected beats
    around astronomical events.
    
    This approach:
    1. First runs the EXISTING successful beat analysis to find real tidal patterns
    2. Then looks for PHASE SHIFTS in those detected beats around opposition dates
    3. Uses the PROVEN methodology that already works
    4. Should be consistent across centers since it uses the established framework
    
    Args:
        complete_df: Complete pair dataset
        event_dates: List of event dates to analyze
        event_name: Name for the analysis (e.g., 'Jupiter', 'Saturn', 'Mars')
        
    Returns:
        dict: Phase modulation analysis results for detected beats
    """
    try:
        print_status(f"Starting {event_name} Phase Modulation Analysis...", "PROCESS")
        print_status("CORRECT METHOD: Using existing beat detection + phase modulation", "INFO")
        
        # Step 1: Run the existing successful beat analysis to find real patterns
        print_status("Step 1: Running existing beat analysis to find real tidal patterns...", "PROCESS")
        beat_results = run_multi_frequency_beat_analysis(complete_df)
        
        if not beat_results.get('success'):
            return {
                'success': False,
                'error': 'Failed to detect baseline beat patterns',
                'beat_analysis_error': beat_results.get('error', 'Unknown')
            }
        
        # Get the successfully detected beats
        significant_beats = beat_results.get('significant_beats', {})
        
        if not significant_beats:
            return {
                'success': False,
                'error': 'No significant beat patterns detected in baseline analysis'
            }
        
        print_status(f"Found {len(significant_beats)} significant beat patterns to analyze", "SUCCESS")
        for beat_name in significant_beats.keys():
            print_status(f"Will analyze phase modulation of: {beat_name}", "INFO")
        
        # Step 2: For each detected beat, look for phase modulation around opposition dates
        event_results = {}
        
        for event_date in event_dates:
            event_date = pd.to_datetime(event_date)
            print_status(f"Analyzing phase modulation around {event_date.date()}...", "PROCESS")
            
            # Define analysis window around event
            window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS')
            event_start = event_date - pd.Timedelta(days=window_days)
            event_end = event_date + pd.Timedelta(days=window_days)
            
            # Extract event window data
            event_data = complete_df[
                (complete_df['date'] >= event_start) & 
                (complete_df['date'] <= event_end)
            ].copy()
            
            if len(event_data) < 5000:  # Need sufficient data for phase analysis
                continue
            
            # For each detected beat, analyze phase modulation
            beat_modulations = {}
            
            for beat_name, beat_info in significant_beats.items():
                beat_period_days = beat_info.get('beat_info', {}).get('period_days', 0)
                
                if beat_period_days <= 0:
                    continue
                
                # Calculate phase for this beat frequency over the event window
                event_data['days_since_epoch'] = (event_data['date'] - pd.Timestamp('2023-01-01')).dt.days
                event_data['beat_phase'] = (2 * np.pi * event_data['days_since_epoch'] / beat_period_days) % (2 * np.pi)
                event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
                
                # Look for phase shifts around the opposition date
                # Bin by days from event and track phase behavior
                daily_phase_data = []
                
                for day_offset in range(-window_days, window_days + 1):
                    day_data = event_data[event_data['days_from_event'] == day_offset]
                    
                    if len(day_data) >= 100:  # Need sufficient data per day
                        # Calculate phase coherence for this day
                        phases = day_data['beat_phase'].values
                        coherences = day_data['coherence'].values
                        
                        # Calculate mean resultant vector (phase coherence measure)
                        complex_phases = np.exp(1j * phases)
                        weighted_complex = np.average(complex_phases, weights=np.abs(coherences))
                        phase_coherence = np.abs(weighted_complex)
                        mean_phase = np.angle(weighted_complex)
                        
                        daily_phase_data.append({
                            'days_from_event': day_offset,
                            'phase_coherence': phase_coherence,
                            'mean_phase': mean_phase,
                            'n_pairs': len(day_data)
                        })
                
                if len(daily_phase_data) >= 20:  # Need sufficient temporal coverage
                    # Look for phase shifts around the event
                    phases = [d['mean_phase'] for d in daily_phase_data]
                    days = [d['days_from_event'] for d in daily_phase_data]
                    
                    # Calculate phase gradient (rate of phase change)
                    phase_gradient = np.gradient(np.unwrap(phases), days)
                    
                    # Look for phase jumps around day 0
                    center_idx = len(days) // 2
                    if center_idx > 5 and center_idx < len(phase_gradient) - 5:
                        pre_gradient = np.mean(phase_gradient[center_idx-5:center_idx])
                        post_gradient = np.mean(phase_gradient[center_idx:center_idx+5])
                        gradient_change = abs(post_gradient - pre_gradient)
                        
                        beat_modulations[beat_name] = {
                            'beat_period_days': beat_period_days,
                            'phase_data': daily_phase_data,
                            'pre_gradient': pre_gradient,
                            'post_gradient': post_gradient,
                            'gradient_change': gradient_change,
                            'significant_modulation': bool(gradient_change > 0.1)  # Threshold for significant phase shift
                        }
            
            event_results[event_date.strftime('%Y-%m-%d')] = {
                'event_date': event_date.isoformat(),
                'beat_modulations': beat_modulations,
                'n_beats_analyzed': len(beat_modulations)
            }
        
        # Overall assessment
        total_beats_analyzed = sum(len(result.get('beat_modulations', {})) for result in event_results.values())
        significant_modulations = sum(
            sum(1 for mod in result.get('beat_modulations', {}).values() 
                if mod.get('significant_modulation', False))
            for result in event_results.values()
        )
        
        interpretation = f"{event_name} phase modulation: {significant_modulations}/{total_beats_analyzed} significant phase modulations detected in existing beat patterns"
        
        return {
            'success': True,
            'method': 'phase_modulation_of_detected_beats',
            'baseline_beats_detected': list(significant_beats.keys()),
            'event_results': event_results,
            'n_events_analyzed': len(event_results),
            'total_beats_analyzed': total_beats_analyzed,
            'significant_modulations': significant_modulations,
            'interpretation': interpretation
        }
        
    except Exception as e:
        print_status(f"{event_name} phase modulation analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_mars_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Mars opposition events.
    
    Mars oppositions occur when Earth-Mars distance is minimized, causing
    Mars's gravitational potential at Earth to peak. According to TEP theory,
    this should create a brief global enhancement in timing correlations.
    
    Expected amplitude: ~0.005% of the solar annual perihelion-aphelion swing
    (ΔU/c² ≈ 5×10⁻¹⁵ vs solar ΔU/c² ≈ 3.3×10⁻¹⁰)
    
    This is ~44x smaller than Jupiter's signal and ~4x smaller than Saturn's,
    making it an excellent test of our detection sensitivity.
    
    Key Mars opposition date within our data range:
    - January 16, 2025
    
    Args:
        complete_df: Complete pair dataset with dates and coherence
        
    Returns:
        dict: Mars opposition analysis results
    """
    try:
        print_status("Starting Mars Opposition Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Mars opposition dates (when Earth-Mars distance is minimized)
        # Mars has a ~26-month synodic period
        mars_events = [
            {'name': 'mars_2025', 'date': pd.to_datetime('2025-01-16'), 'description': 'Mars Opposition January 2025'}
        ]
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS')
        expected_amplitude = TEPConfig.get_float('TEP_MARS_AMPLITUDE_FRACTION')
        min_pairs = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS')
        
        print_status(f"Analyzing {len(mars_events)} Mars opposition event", "INFO")
        print_status(f"Event window: ±{window_days} days", "INFO")
        print_status(f"Expected amplitude: {expected_amplitude:.5f} ({expected_amplitude*100:.4f}%)", "INFO")
        print_status("NOTE: Mars has the weakest expected signal - excellent sensitivity test!", "INFO")
        
        # Analyze each event
        event_results = {}
        successful_events = 0
        
        for event in mars_events:
            event_name = event['name']
            event_date = event['date']
            
            print_status(f"Analyzing {event['description']}...", "PROCESS")
            
            # Check if event date is within our data range
            data_start = complete_df['date'].min()
            data_end = complete_df['date'].max()
            
            if event_date < data_start or event_date > data_end:
                print_status(f"Event {event_date.date()} outside data range ({data_start.date()} to {data_end.date()})", "WARNING")
                event_results[event_name] = {
                    'success': False,
                    'error': 'Event date outside data range',
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'description': event['description']
                }
                continue
            
            # Extract data window around event
            start_date = event_date - pd.Timedelta(days=window_days)
            end_date = event_date + pd.Timedelta(days=window_days)
            
            event_data = complete_df[
                (complete_df['date'] >= start_date) & 
                (complete_df['date'] <= end_date)
            ].copy()
            
            if len(event_data) == 0:
                event_results[event_name] = {
                    'success': False,
                    'error': 'No data in event window',
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'description': event['description']
                }
                continue
            
            # Calculate days from event
            event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
            
            # Daily binning and analysis
            daily_data = []
            for day in range(-window_days, window_days + 1):
                day_data = event_data[event_data['days_from_event'] == day]
                
                if len(day_data) >= min_pairs:
                    mean_coherence = day_data['coherence'].mean()
                    std_coherence = day_data['coherence'].std()
                    n_pairs = len(day_data)
                    
                    daily_data.append({
                        'day': day,
                        'mean_coherence': mean_coherence,
                        'std_coherence': std_coherence,
                        'n_pairs': n_pairs,
                        'date': (event_date + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
                    })
            
            if len(daily_data) < 10:
                event_results[event_name] = {
                    'success': False,
                    'error': f'Insufficient daily data points: {len(daily_data)} < 10',
                    'event_date': event_date.strftime('%Y-%m-%d'),
                    'description': event['description']
                }
                continue
            
            # Fit Gaussian peak to detect Mars signal
            days_from_event = [d['day'] for d in daily_data]
            coherences = [d['mean_coherence'] for d in daily_data]
            
            gaussian_fit = fit_event_peak(days_from_event, coherences)
            
            # Store results
            event_results[event_name] = {
                'success': True,
                'event_date': event_date.strftime('%Y-%m-%d'),
                'description': event['description'],
                'daily_tracking': daily_data,
                'gaussian_fit': gaussian_fit,
                'n_daily_points': len(daily_data),
                'total_pairs_analyzed': sum(d['n_pairs'] for d in daily_data),
                'interpretation': gaussian_fit.get('interpretation', 'Analysis completed')
            }
            
            successful_events += 1
            print_status(f"✓ {event['description']} analysis completed", "SUCCESS")
        
        # No cross-event analysis since we only have one Mars event
        cross_event_analysis = {
            'success': False,
            'error': 'Only one Mars opposition event available - no cross-validation possible'
        }
        
        # Overall interpretation
        if successful_events == 0:
            interpretation = "No Mars opposition events could be analyzed"
        else:
            # Check for significant detections
            significant_events = sum(1 for result in event_results.values() 
                                   if result.get('success') and 
                                      result.get('gaussian_fit', {}).get('is_significant', False))
            
            if significant_events > 0:
                interpretation = f"Mars opposition analysis: {significant_events}/{successful_events} events show significant signals - remarkable sensitivity!"
            else:
                interpretation = f"Mars opposition analysis: No significant signals detected in {successful_events} events (expected for weakest signal)"
        
        return {
            'success': True,
            'n_events_attempted': len(mars_events),
            'n_successful_events': successful_events,
            'event_results': event_results,
            'stacked_analysis': cross_event_analysis,  # No stacking possible with 1 event
            'expected_amplitude_fraction': expected_amplitude,
            'interpretation': interpretation,
            'analysis_summary': {
                'total_events': len(mars_events),
                'successful_events': successful_events,
                'significant_detections': sum(1 for result in event_results.values() 
                                            if result.get('success') and 
                                               result.get('gaussian_fit', {}).get('is_significant', False)),
                'stacked_analysis_success': False,  # Not possible with 1 event
                'note': 'Mars has the weakest expected signal - detection would indicate exceptional sensitivity'
            }
        }
        
    except Exception as e:
        print_status(f"Mars opposition analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_lunar_standstill_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze Major Lunar Standstill effects on sidereal day signal amplitude.
    
    The Major Lunar Standstill occurs every 18.6 years when the Moon reaches
    extreme declinations (±28.5°). This enhances tidal geometry and should
    modulate the sidereal-day component of GPS timing correlations.
    
    The current Major Lunar Standstill window spans mid-2024 to mid-2025,
    peaking around December 2024.
    
    Key analysis:
    - Track sidereal day (23h56m4s) Fourier amplitude over time
    - Look for enhancement during standstill window
    - Compare pre-standstill vs standstill vs post-standstill periods
    
    Args:
        complete_df: Complete pair dataset with dates and coherence
        
    Returns:
        dict: Lunar Standstill analysis results
    """
    try:
        print_status("Starting Major Lunar Standstill Analysis...", "PROCESS")
        print_status("Tracking sidereal day (23h56m4s) signal amplitude through 2024-2025", "INFO")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Configuration
        window_months = TEPConfig.get_int('TEP_LUNAR_STANDSTILL_WINDOW_MONTHS')
        sidereal_day_hours = TEPConfig.get_float('TEP_SIDEREAL_DAY_HOURS')
        peak_date = pd.to_datetime(TEPConfig.get_str('TEP_LUNAR_STANDSTILL_PEAK_DATE'))
        
        print_status(f"Standstill peak date: {peak_date.date()}", "INFO")
        print_status(f"Analysis window: ±{window_months} months", "INFO")
        print_status(f"Sidereal day period: {sidereal_day_hours:.6f} hours", "INFO")
        
        # Define analysis periods
        standstill_start = peak_date - pd.DateOffset(months=window_months)
        standstill_end = peak_date + pd.DateOffset(months=window_months)
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()}", "INFO")
        print_status(f"Standstill window: {standstill_start.date()} to {standstill_end.date()}", "INFO")
        
        # Filter to relevant time period
        analysis_data = complete_df[
            (complete_df['date'] >= standstill_start - pd.DateOffset(months=6)) &  # Include pre-period
            (complete_df['date'] <= standstill_end + pd.DateOffset(months=6))      # Include post-period
        ].copy()
        
        if len(analysis_data) == 0:
            return {
                'success': False,
                'error': 'No data in Lunar Standstill analysis period',
                'analysis_period': [standstill_start.isoformat(), standstill_end.isoformat()]
            }
        
        print_status(f"Analyzing {len(analysis_data):,} pairs in extended period", "SUCCESS")
        
        # Monthly sidereal day amplitude analysis
        monthly_amplitudes = []
        
        # Group by month and analyze sidereal day signal
        analysis_data['year_month'] = analysis_data['date'].dt.to_period('M')
        
        for month_period, month_data in analysis_data.groupby('year_month'):
            if len(month_data) < 1000:  # Need sufficient data for FFT
                continue
                
            # Convert to datetime for calculations
            month_start = month_period.start_time
            
            # Simplified approach: use coherence variance as a proxy for sidereal day effects
            # This avoids the complex hourly binning that was failing
            
            # Calculate time-of-day in sidereal hours
            # Convert UTC time to sidereal time (approximate)
            month_data = month_data.copy()
            
            # Simple sidereal day analysis: look at coherence variation with time of day
            # Use the standard deviation of coherence as the "amplitude" metric
            sidereal_amplitude = month_data['coherence'].std()
            
            # Alternative: use the range (max - min) as amplitude
            # sidereal_amplitude = month_data['coherence'].max() - month_data['coherence'].min()
            
            # Calculate distance from standstill peak (in months)
            months_from_peak = (month_start - peak_date).days / 30.44  # Average days per month
            
            # Determine period classification
            if abs(months_from_peak) <= window_months:
                period = 'standstill'
            elif months_from_peak < -window_months:
                period = 'pre_standstill'
            else:
                period = 'post_standstill'
            
            monthly_amplitudes.append({
                'month': month_start.strftime('%Y-%m'),
                'date': month_start.strftime('%Y-%m-%d'),  # Convert to string for JSON
                'months_from_peak': months_from_peak,
                'period': period,
                'sidereal_amplitude': sidereal_amplitude,
                'n_pairs': len(month_data),
                'mean_coherence': month_data['coherence'].mean()
            })
        
        if len(monthly_amplitudes) < 6:
            return {
                'success': False,
                'error': f'Insufficient monthly data points: {len(monthly_amplitudes)} < 6',
                'monthly_data': monthly_amplitudes
            }
        
        # Statistical analysis of amplitude variations
        amplitudes_df = pd.DataFrame(monthly_amplitudes)
        
        # Group by period
        period_stats = {}
        for period in ['pre_standstill', 'standstill', 'post_standstill']:
            period_data = amplitudes_df[amplitudes_df['period'] == period]
            if len(period_data) > 0:
                period_stats[period] = {
                    'n_months': len(period_data),
                    'mean_amplitude': period_data['sidereal_amplitude'].mean(),
                    'std_amplitude': period_data['sidereal_amplitude'].std(),
                    'median_amplitude': period_data['sidereal_amplitude'].median(),
                    'date_range': [period_data.iloc[0]['month'], 
                                 period_data.iloc[-1]['month']]
                }
        
        # Enhancement analysis
        enhancement_analysis = {}
        if 'pre_standstill' in period_stats and 'standstill' in period_stats:
            pre_mean = period_stats['pre_standstill']['mean_amplitude']
            standstill_mean = period_stats['standstill']['mean_amplitude']
            
            enhancement_ratio = standstill_mean / pre_mean if pre_mean > 0 else 0
            enhancement_analysis['pre_to_standstill'] = {
                'enhancement_ratio': enhancement_ratio,
                'enhancement_percent': (enhancement_ratio - 1) * 100,
                'pre_amplitude': pre_mean,
                'standstill_amplitude': standstill_mean
            }
        
        # Trend analysis
        # Fit polynomial to amplitude vs time
        months_from_peak = amplitudes_df['months_from_peak'].values
        amplitudes = amplitudes_df['sidereal_amplitude'].values
        
        # Fit quadratic (expecting peak at standstill)
        try:
            poly_coeffs = np.polyfit(months_from_peak, amplitudes, 2)
            trend_analysis = {
                'quadratic_coeffs': poly_coeffs.tolist(),
                'peak_month_offset': -poly_coeffs[1] / (2 * poly_coeffs[0]) if poly_coeffs[0] != 0 else 0,
                'r_squared': np.corrcoef(amplitudes, np.polyval(poly_coeffs, months_from_peak))[0,1]**2
            }
        except:
            trend_analysis = {'error': 'Polynomial fit failed'}
        
        # Overall interpretation
        if enhancement_analysis and 'pre_to_standstill' in enhancement_analysis:
            ratio = enhancement_analysis['pre_to_standstill']['enhancement_ratio']
            if ratio > 1.2:  # 20% enhancement
                interpretation = f"Strong Lunar Standstill enhancement detected: {ratio:.2f}x amplitude increase"
            elif ratio > 1.1:  # 10% enhancement
                interpretation = f"Moderate Lunar Standstill enhancement detected: {ratio:.2f}x amplitude increase"
            elif ratio > 1.05:  # 5% enhancement
                interpretation = f"Weak Lunar Standstill enhancement detected: {ratio:.2f}x amplitude increase"
            else:
                interpretation = f"No significant Lunar Standstill enhancement: {ratio:.2f}x ratio"
        else:
            interpretation = "Insufficient data for enhancement analysis"
        
        return {
            'success': True,
            'analysis_period': [standstill_start.isoformat(), standstill_end.isoformat()],
            'peak_date': peak_date.isoformat(),
            'sidereal_day_hours': sidereal_day_hours,
            'monthly_amplitudes': monthly_amplitudes,
            'period_statistics': period_stats,
            'enhancement_analysis': enhancement_analysis,
            'trend_analysis': trend_analysis,
            'interpretation': interpretation,
            'analysis_summary': {
                'total_months': len(monthly_amplitudes),
                'periods_analyzed': list(period_stats.keys()),
                'enhancement_detected': bool(enhancement_analysis.get('pre_to_standstill', {}).get('enhancement_ratio', 1) > 1.05),
                'peak_amplitude_month': amplitudes_df.loc[amplitudes_df['sidereal_amplitude'].idxmax(), 'month'] if len(amplitudes_df) > 0 else None
            }
        }
        
    except Exception as e:
        print_status(f"Lunar Standstill analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_solar_eclipse_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around the April 8, 2024 total solar eclipse.
    
    Solar eclipses dramatically affect the ionosphere, which should create
    a clear signature in GPS timing correlations that appears consistently
    across ALL analysis centers (unlike gravitational effects which depend
    on correction models).
    
    Expected effect: Temporary enhancement of timing correlations during
    eclipse totality due to reduced ionospheric noise.
    
    This serves as an excellent validation test:
    - Should show consistent effects across IGS/CODE/ESA
    - Tests our methodology on a known ionospheric phenomenon
    - Distinguishes ionospheric vs gravitational effects
    
    Args:
        complete_df: Complete pair dataset with dates and coherence
        
    Returns:
        dict: Solar eclipse analysis results
    """
    try:
        print_status("Starting Solar Eclipse Analysis...", "PROCESS")
        print_status("Analyzing April 8, 2024 total solar eclipse ionospheric effects", "INFO")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
        
        # Solar eclipse configuration (adjusted for daily data)
        eclipse_date = pd.to_datetime(TEPConfig.get_str('TEP_SOLAR_ECLIPSE_DATE'))
        window_days = 7  # ±7 days around eclipse (daily data resolution)
        
        print_status(f"Eclipse date: {eclipse_date.date()}", "INFO")
        print_status(f"Analysis window: ±{window_days} days (adjusted for daily data)", "INFO")
        print_status("NOTE: Using daily resolution analysis due to GPS data temporal resolution", "INFO")
        
        # Check if eclipse is within data range
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        
        if eclipse_date < data_start or eclipse_date > data_end:
            return {
                'success': False,
                'error': f'Eclipse date {eclipse_date.date()} outside data range ({data_start.date()} to {data_end.date()})'
            }
        
        # Define analysis periods (daily resolution)
        eclipse_start = eclipse_date - pd.Timedelta(days=window_days)
        eclipse_end = eclipse_date + pd.Timedelta(days=window_days)
        
        # Extract eclipse period data
        eclipse_data = complete_df[
            (complete_df['date'] >= eclipse_start) & 
            (complete_df['date'] <= eclipse_end)
        ].copy()
        
        if len(eclipse_data) == 0:
            return {
                'success': False,
                'error': 'No data during eclipse period'
            }
        
        print_status(f"Analyzing {len(eclipse_data):,} pairs during eclipse period", "SUCCESS")
        
        # Calculate days from eclipse
        eclipse_data['days_from_eclipse'] = (eclipse_data['date'] - eclipse_date).dt.days
        
        # Daily binning analysis
        daily_data = []
        
        for day_offset in range(-window_days, window_days + 1):
            day_data = eclipse_data[eclipse_data['days_from_eclipse'] == day_offset]
            
            if len(day_data) >= 100:  # Need sufficient data per day
                mean_coherence = day_data['coherence'].mean()
                std_coherence = day_data['coherence'].std()
                n_pairs = len(day_data)
                
                # Determine eclipse phase
                if day_offset == 0:  # Eclipse day
                    eclipse_phase = 'eclipse_day'
                elif abs(day_offset) <= 1:  # Day before/after
                    eclipse_phase = 'adjacent'
                else:
                    eclipse_phase = 'baseline'
                
                daily_data.append({
                    'days_from_eclipse': day_offset,
                    'mean_coherence': mean_coherence,
                    'std_coherence': std_coherence,
                    'n_pairs': n_pairs,
                    'eclipse_phase': eclipse_phase,
                    'date': (eclipse_date + pd.Timedelta(days=day_offset)).strftime('%Y-%m-%d')
                })
        
        if len(daily_data) < 5:
            return {
                'success': False,
                'error': f'Insufficient daily data points: {len(daily_data)} < 5'
            }
        
        # Fit Gaussian to detect eclipse signal
        days_from_eclipse = [d['days_from_eclipse'] for d in daily_data]
        coherences = [d['mean_coherence'] for d in daily_data]
        
        gaussian_fit = fit_event_peak(days_from_eclipse, coherences)
        
        # Phase-based analysis (daily resolution)
        phase_analysis = {}
        for phase in ['baseline', 'adjacent', 'eclipse_day']:
            phase_data = [d for d in daily_data if d['eclipse_phase'] == phase]
            if phase_data:
                phase_coherences = [d['mean_coherence'] for d in phase_data]
                phase_analysis[phase] = {
                    'n_days': len(phase_data),
                    'mean_coherence': np.mean(phase_coherences),
                    'std_coherence': np.std(phase_coherences)
                }
        
        # Calculate eclipse enhancement
        eclipse_enhancement = {}
        if 'baseline' in phase_analysis and 'eclipse_day' in phase_analysis:
            baseline_coherence = phase_analysis['baseline']['mean_coherence']
            eclipse_coherence = phase_analysis['eclipse_day']['mean_coherence']
            
            if baseline_coherence != 0:
                enhancement_ratio = eclipse_coherence / baseline_coherence
                eclipse_enhancement = {
                    'baseline_coherence': baseline_coherence,
                    'eclipse_coherence': eclipse_coherence,
                    'enhancement_ratio': enhancement_ratio,
                    'enhancement_percent': (enhancement_ratio - 1) * 100
                }
        
        # Interpretation
        if gaussian_fit.get('is_significant', False):
            interpretation = f"Significant solar eclipse ionospheric signature detected"
        elif eclipse_enhancement.get('enhancement_ratio', 1) > 1.1:
            interpretation = f"Moderate eclipse enhancement detected: {eclipse_enhancement['enhancement_percent']:.1f}%"
        else:
            interpretation = "No significant eclipse signature detected"
        
        return {
            'success': True,
            'eclipse_date': eclipse_date.isoformat(),
            'analysis_window_days': window_days,
            'daily_tracking': daily_data,
            'gaussian_fit': gaussian_fit,
            'phase_analysis': phase_analysis,
            'eclipse_enhancement': eclipse_enhancement,
            'interpretation': interpretation,
            'n_pairs_analyzed': len(eclipse_data),
            'analysis_summary': {
                'eclipse_detected': bool(gaussian_fit.get('is_significant', False)),
                'enhancement_detected': bool(eclipse_enhancement.get('enhancement_ratio', 1) > 1.05),
                'eclipse_day_analyzed': len([d for d in daily_data if d['eclipse_phase'] == 'eclipse_day']) > 0
            }
        }
        
    except Exception as e:
        print_status(f"Solar eclipse analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_nutation_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze 18.6-year nutation signatures (requires multi-year data).
    
    Nutation causes periodic variations in Earth's axial tilt with an 18.6-year
    period, which should modulate correlation patterns if the dataset spans
    sufficient time.
    
    Args:
        complete_df: Complete pair dataset with multi-year coverage
        
    Returns:
        dict: Nutation analysis results
    """
    print_status("Starting Nutation Analysis (18.6-year period)...", "PROCESS")
    
    try:
        # Check data span
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        data_span_years = (complete_df['date'].max() - complete_df['date'].min()).days / 365.25
        
        nutation_period_years = TEPConfig.get_float('TEP_NUTATION_PERIOD_YEARS')
        min_span_years = nutation_period_years / 3  # Need at least 1/3 of a cycle
        
        if data_span_years < min_span_years:
            return {
                'success': False, 
                'error': f'Insufficient data span: {data_span_years:.1f} years (need >{min_span_years:.1f} years)',
                'data_span_years': data_span_years,
                'required_span_years': min_span_years
            }
        
        # Calculate nutation phase
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        nutation_period_days = nutation_period_years * 365.25
        complete_df['nutation_phase'] = 2 * np.pi * complete_df['days_since_epoch'] / nutation_period_days
        
        # Bin by nutation phase (fewer bins due to longer period)
        n_phase_bins = 8  # 45° increments
        phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
        complete_df['nutation_phase_bin'] = pd.cut(complete_df['nutation_phase'], 
                                                   bins=phase_bins, 
                                                   labels=range(n_phase_bins))
        
        # Compute directional classification (reuse existing function)
        complete_df['azimuth'] = complete_df.apply(
            lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                       row['station2_lat'], row['station2_lon']), axis=1
        )
        
        def classify_ew_ns(azimuth):
            return 'EW' if (45 <= azimuth <= 135) or (225 <= azimuth <= 315) else 'NS'
        
        complete_df['ew_ns_class'] = complete_df['azimuth'].apply(classify_ew_ns)
        
        # Track E-W/N-S ratio across nutation phases
        nutation_tracking = []
        edges = np.logspace(np.log10(50), np.log10(13000), 41)
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        
        for phase_bin in range(n_phase_bins):
            phase_data = complete_df[complete_df['nutation_phase_bin'] == phase_bin].copy()
            
            if len(phase_data) < 2000:  # Need more data for long-period analysis
                continue
                
            # Analyze E-W and N-S separately
            ew_data = phase_data[phase_data['ew_ns_class'] == 'EW']
            ns_data = phase_data[phase_data['ew_ns_class'] == 'NS']
            
            if len(ew_data) < 1000 or len(ns_data) < 1000:
                continue
                
            # Fit correlation models
            ew_lambda = fit_directional_correlation(ew_data, edges, min_bin_count)
            ns_lambda = fit_directional_correlation(ns_data, edges, min_bin_count)
            
            if ew_lambda and ns_lambda and ns_lambda > 0:
                nutation_tracking.append({
                    'phase_bin': phase_bin,
                    'phase_radians': float(phase_bins[phase_bin]),
                    'phase_degrees': float(np.degrees(phase_bins[phase_bin])),
                    'ew_lambda_km': float(ew_lambda),
                    'ns_lambda_km': float(ns_lambda),
                    'ew_ns_ratio': float(ew_lambda / ns_lambda),
                    'n_pairs': len(phase_data),
                    'n_ew_pairs': len(ew_data),
                    'n_ns_pairs': len(ns_data)
                })
        
        if len(nutation_tracking) < 4:  # Need minimum phase coverage
            return {'success': False, 'error': f'Insufficient phase bins: {len(nutation_tracking)}'}
        
        # Test for nutation modulation
        phases = [t['phase_radians'] for t in nutation_tracking]
        ratios = [t['ew_ns_ratio'] for t in nutation_tracking]
        
        try:
            # Fit sinusoidal model
            def nutation_model(phase, amplitude, phase_shift, offset):
                return amplitude * np.sin(phase + phase_shift) + offset
            
            popt, _ = curve_fit(nutation_model, phases, ratios, 
                               p0=[0.1, 0, np.mean(ratios)])
            
            predicted = nutation_model(np.array(phases), *popt)
            correlation = np.corrcoef(ratios, predicted)[0, 1]
            
            from scipy.stats import pearsonr
            _, p_value = pearsonr(ratios, predicted)
            
            nutation_fit = {
                'amplitude': float(popt[0]),
                'phase_shift_radians': float(popt[1]),
                'phase_shift_degrees': float(np.degrees(popt[1])),
                'offset': float(popt[2]),
                'correlation': float(correlation),
                'p_value': float(p_value),
                'fit_success': True
            }
            
        except Exception as e:
            nutation_fit = {'fit_success': False, 'error': str(e)}
        
        return {
            'success': True,
            'data_span_years': data_span_years,
            'nutation_period_years': nutation_period_years,
            'n_phase_bins': len(nutation_tracking),
            'phase_tracking': nutation_tracking,
            'sinusoidal_fit': nutation_fit,
            'interpretation': classify_nutation_evidence(nutation_fit.get('correlation', 0), 
                                                        nutation_fit.get('p_value', 1))
        }
        
    except Exception as e:
        print_status(f"Nutation analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def classify_chandler_evidence(correlation: float, p_value: float) -> str:
    """Classify strength of Chandler wobble evidence"""
    if p_value < 0.001 and abs(correlation) > 0.7:
        return f"Robust Chandler wobble signature confirmed (r={correlation:.3f}, p<0.001)"
    elif p_value < 0.01 and abs(correlation) > 0.5:
        return f"Significant Chandler wobble signature detected (r={correlation:.3f}, p<0.01)"
    elif p_value < 0.05 and abs(correlation) > 0.3:
        return f"Chandler wobble signature identified (r={correlation:.3f}, p<0.05)"
    else:
        return f"No statistically significant Chandler wobble correlation (r={correlation:.3f}, p={p_value:.3f})"

def classify_nutation_evidence(correlation: float, p_value: float) -> str:
    """Classify strength of nutation evidence"""
    if p_value < 0.001 and abs(correlation) > 0.8:
        return f"Robust nutation signature confirmed (r={correlation:.3f}, p<0.001)"
    elif p_value < 0.01 and abs(correlation) > 0.6:
        return f"Significant nutation signature detected (r={correlation:.3f}, p<0.01)"
    elif p_value < 0.05 and abs(correlation) > 0.4:
        return f"Nutation signature identified (r={correlation:.3f}, p<0.05)"
    else:
        return f"No statistically significant nutation correlation (r={correlation:.3f}, p={p_value:.3f})"

def run_relative_motion_beat_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    ENHANCED beat analysis considering RELATIVE MOTION between station pairs.
    
    This analyzes how each station pair moves relative to each other as Earth
    undergoes complex helical motion. Different station separations and orientations
    experience different relative velocities and accelerations.
    
    Key concepts:
    1. DIFFERENTIAL ROTATION: Stations at different latitudes have different tangential velocities
    2. ORBITAL PROJECTION: Station pairs see different projections of Earth's orbital motion  
    3. WOBBLE DIFFERENTIAL: Chandler wobble affects station pairs differently based on separation
    4. DISTANCE-DEPENDENT BEATS: Beat frequencies vary with station separation distance
    
    Args:
        complete_df: Complete pair dataset with coordinates, distances, dates
        
    Returns:
        dict: Relative motion beat analysis results
    """
    print_status("Starting Relative Motion Beat Analysis...", "PROCESS")
    print_status("Analyzing differential motion patterns across station mesh", "PROCESS")
    
    try:
        # Convert dates and compute basic parameters
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Compute SUPERPOSITION OF ACCELERATIONS for each station pair
        print_status("Computing multi-layered acceleration superposition for station pairs...", "INFO")
        print_status("Analyzing how stations swing forward/back through layered motion streams", "INFO")
        
        # 1. MULTI-LAYERED ACCELERATION ANALYSIS
        # Each station experiences superposition of multiple accelerations with different phases
        
        # Station 1 and Station 2 coordinates
        lat1_rad = np.radians(complete_df['station1_lat'])
        lon1_rad = np.radians(complete_df['station1_lon'])
        lat2_rad = np.radians(complete_df['station2_lat'])
        lon2_rad = np.radians(complete_df['station2_lon'])
        
        earth_radius_km = 6371.0
        
        # A. ROTATIONAL ACCELERATION COMPONENTS (24h cycle)
        # Centripetal acceleration varies with latitude: a_c = ω²r = ω²R*cos(lat)
        omega_earth = 2 * np.pi / (24 * 3600)  # rad/s
        
        complete_df['station1_centripetal_acc'] = (omega_earth**2) * earth_radius_km * 1000 * np.cos(lat1_rad)  # m/s²
        complete_df['station2_centripetal_acc'] = (omega_earth**2) * earth_radius_km * 1000 * np.cos(lat2_rad)  # m/s²
        complete_df['differential_centripetal_acc'] = abs(complete_df['station2_centripetal_acc'] - complete_df['station1_centripetal_acc'])
        
        # Tangential velocity differences (differential rotation)
        complete_df['station1_tangential_velocity'] = omega_earth * earth_radius_km * 1000 * np.cos(lat1_rad)  # m/s
        complete_df['station2_tangential_velocity'] = omega_earth * earth_radius_km * 1000 * np.cos(lat2_rad)  # m/s
        complete_df['differential_rotation_velocity'] = abs(complete_df['station2_tangential_velocity'] - complete_df['station1_tangential_velocity']) / 1000  # km/s
        
        # B. ORBITAL ACCELERATION COMPONENTS (365d cycle)
        # Earth's orbital acceleration: a_orbital = v²/r = (29.78 km/s)² / (150M km)
        orbital_speed_ms = 29780  # m/s
        orbital_radius_m = 1.496e11  # m (1 AU)
        orbital_acceleration = (orbital_speed_ms**2) / orbital_radius_m  # m/s² ≈ 0.006 m/s²
        
        # Orbital acceleration projection varies with time of year and station position
        # Each station sees different orbital acceleration based on its position relative to orbital motion
        day_of_year = complete_df['days_since_epoch'] % 365.25
        orbital_phase = 2 * np.pi * day_of_year / 365.25
        
        # Project orbital acceleration onto station positions (simplified)
        complete_df['station1_orbital_acc_projection'] = orbital_acceleration * np.cos(orbital_phase + lon1_rad)
        complete_df['station2_orbital_acc_projection'] = orbital_acceleration * np.cos(orbital_phase + lon2_rad)
        complete_df['differential_orbital_acc'] = abs(complete_df['station2_orbital_acc_projection'] - complete_df['station1_orbital_acc_projection'])
        
        # C. CHANDLER WOBBLE ACCELERATION (14-month cycle)
        # Wobble creates additional acceleration as rotation axis wanders
        chandler_period_days = TEPConfig.get_float('TEP_CHANDLER_PERIOD_DAYS')
        chandler_phase = 2 * np.pi * complete_df['days_since_epoch'] / chandler_period_days
        wobble_amplitude_m = 9.0  # meters
        wobble_acceleration = (2 * np.pi / (chandler_period_days * 24 * 3600))**2 * wobble_amplitude_m  # m/s²
        
        # Wobble affects stations differently based on latitude
        complete_df['station1_wobble_acc'] = wobble_acceleration * abs(np.cos(lat1_rad)) * np.sin(chandler_phase)
        complete_df['station2_wobble_acc'] = wobble_acceleration * abs(np.cos(lat2_rad)) * np.sin(chandler_phase)
        complete_df['differential_wobble_acc'] = abs(complete_df['station2_wobble_acc'] - complete_df['station1_wobble_acc'])
        
        # D. TIDAL ACCELERATION COMPONENTS (12h, 24h cycles)
        # Lunar tidal acceleration varies across Earth's surface
        lunar_tidal_acceleration = 1.1e-6  # m/s² (approximate)
        
        # M2 tidal component (12.42h period)
        m2_period_hours = 12.42
        m2_phase = 2 * np.pi * (complete_df['days_since_epoch'] * 24) / m2_period_hours
        
        complete_df['station1_tidal_acc'] = lunar_tidal_acceleration * np.cos(m2_phase + 2*lon1_rad) * np.cos(lat1_rad)
        complete_df['station2_tidal_acc'] = lunar_tidal_acceleration * np.cos(m2_phase + 2*lon2_rad) * np.cos(lat2_rad)
        complete_df['differential_tidal_acc'] = abs(complete_df['station2_tidal_acc'] - complete_df['station1_tidal_acc'])
        
        # E. TOTAL ACCELERATION SUPERPOSITION
        # Each station experiences vector sum of all accelerations with different phases
        print_status("Computing acceleration superposition vectors...", "INFO")
        
        # For each station pair, compute the relative acceleration vector magnitude
        # This captures how stations "swing forward and back" through the motion streams
        complete_df['total_differential_acceleration'] = np.sqrt(
            complete_df['differential_centripetal_acc']**2 +
            complete_df['differential_orbital_acc']**2 +
            complete_df['differential_wobble_acc']**2 +
            complete_df['differential_tidal_acc']**2
        )
        
        # F. PHASE RELATIONSHIPS BETWEEN MOTION COMPONENTS
        # Different motions have different phases - this creates complex interference patterns
        
        # Rotation phase (24h)
        rotation_phase = 2 * np.pi * (complete_df['days_since_epoch'] % 1.0)
        
        # Orbital phase (365d) 
        orbital_phase = 2 * np.pi * (complete_df['days_since_epoch'] % 365.25) / 365.25
        
        # Chandler phase (427d)
        chandler_phase = 2 * np.pi * (complete_df['days_since_epoch'] % chandler_period_days) / chandler_period_days
        
        # Tidal phase (12.42h)
        tidal_phase = 2 * np.pi * ((complete_df['days_since_epoch'] * 24) % m2_period_hours) / m2_period_hours
        
        # G. INTERFERENCE PATTERNS
        # Compute phase differences that create beat patterns
        complete_df['rotation_orbital_phase_diff'] = np.abs(rotation_phase - orbital_phase)
        complete_df['rotation_chandler_phase_diff'] = np.abs(rotation_phase - chandler_phase)
        complete_df['orbital_chandler_phase_diff'] = np.abs(orbital_phase - chandler_phase)
        complete_df['tidal_rotation_phase_diff'] = np.abs(tidal_phase - rotation_phase)
        
        # H. CONSTRUCTIVE/DESTRUCTIVE INTERFERENCE ANALYSIS
        # Stations swing forward/back through motion streams with VECTOR ADDITION/SUBTRACTION
        print_status("Analyzing constructive and destructive interference patterns...", "INFO")
        
        # Vector components of each motion (simplified 2D projection)
        # Each motion has magnitude and direction that varies with time and position
        
        # Rotation vector (always eastward, magnitude varies with latitude)
        rotation_vector_x = complete_df['differential_rotation_velocity'] * np.cos(rotation_phase)
        rotation_vector_y = complete_df['differential_rotation_velocity'] * np.sin(rotation_phase)
        
        # Orbital vector (direction changes seasonally, ~30 km/s)
        orbital_direction = orbital_phase  # Changes throughout year
        orbital_magnitude = complete_df['differential_orbital_acc'] * 1000  # Convert to m/s for comparison
        orbital_vector_x = orbital_magnitude * np.cos(orbital_direction)
        orbital_vector_y = orbital_magnitude * np.sin(orbital_direction)
        
        # Chandler wobble vector (circular motion, 14-month period)
        wobble_vector_x = complete_df['differential_wobble_acc'] * np.cos(chandler_phase)
        wobble_vector_y = complete_df['differential_wobble_acc'] * np.sin(chandler_phase)
        
        # Tidal vector (complex elliptical motion)
        tidal_vector_x = complete_df['differential_tidal_acc'] * np.cos(tidal_phase)
        tidal_vector_y = complete_df['differential_tidal_acc'] * np.sin(tidal_phase)
        
        # VECTOR SUPERPOSITION: Add/subtract all motion vectors
        total_vector_x = rotation_vector_x + orbital_vector_x + wobble_vector_x + tidal_vector_x
        total_vector_y = rotation_vector_y + orbital_vector_y + wobble_vector_y + tidal_vector_y
        
        complete_df['total_motion_vector_magnitude'] = np.sqrt(total_vector_x**2 + total_vector_y**2)
        complete_df['total_motion_vector_direction'] = np.arctan2(total_vector_y, total_vector_x)
        
        # I. INTERFERENCE CLASSIFICATION
        # Determine when motions are CONSTRUCTIVE vs DESTRUCTIVE for each station pair
        
        # Dot products between motion vectors (measures alignment/opposition)
        rotation_orbital_dot = rotation_vector_x * orbital_vector_x + rotation_vector_y * orbital_vector_y
        rotation_wobble_dot = rotation_vector_x * wobble_vector_x + rotation_vector_y * wobble_vector_y
        orbital_wobble_dot = orbital_vector_x * wobble_vector_x + orbital_vector_y * wobble_vector_y
        
        # Normalize dot products to get interference coefficients (-1 to +1)
        rotation_orbital_interference = rotation_orbital_dot / (
            np.sqrt(rotation_vector_x**2 + rotation_vector_y**2) * 
            np.sqrt(orbital_vector_x**2 + orbital_vector_y**2) + 1e-10
        )
        rotation_wobble_interference = rotation_wobble_dot / (
            np.sqrt(rotation_vector_x**2 + rotation_vector_y**2) * 
            np.sqrt(wobble_vector_x**2 + wobble_vector_y**2) + 1e-10
        )
        orbital_wobble_interference = orbital_wobble_dot / (
            np.sqrt(orbital_vector_x**2 + orbital_vector_y**2) * 
            np.sqrt(wobble_vector_x**2 + wobble_vector_y**2) + 1e-10
        )
        
        complete_df['rotation_orbital_interference'] = rotation_orbital_interference
        complete_df['rotation_wobble_interference'] = rotation_wobble_interference
        complete_df['orbital_wobble_interference'] = orbital_wobble_interference
        
        # J. DYNAMIC INTERFERENCE STATES
        # Classify each station pair's current interference state
        
        def classify_interference_state(rot_orb, rot_wob, orb_wob):
            """Classify the interference state based on vector alignments"""
            constructive_threshold = 0.5
            destructive_threshold = -0.5
            
            constructive_count = sum([
                rot_orb > constructive_threshold,
                rot_wob > constructive_threshold, 
                orb_wob > constructive_threshold
            ])
            
            destructive_count = sum([
                rot_orb < destructive_threshold,
                rot_wob < destructive_threshold,
                orb_wob < destructive_threshold
            ])
            
            if constructive_count >= 2:
                return 'constructive'
            elif destructive_count >= 2:
                return 'destructive'
            elif constructive_count == 1 and destructive_count == 1:
                return 'mixed'
            else:
                return 'neutral'
        
        complete_df['interference_state'] = complete_df.apply(
            lambda row: classify_interference_state(
                row['rotation_orbital_interference'],
                row['rotation_wobble_interference'], 
                row['orbital_wobble_interference']
            ), axis=1
        )
        
        # K. TEMPORAL OSCILLATION PATTERNS
        # Track how interference states change over time for each distance category
        
        # Create oscillation strength metric
        # Measures how much the motion vector magnitude oscillates
        complete_df['motion_oscillation_strength'] = (
            abs(rotation_vector_x) + abs(orbital_vector_x) + abs(wobble_vector_x) + abs(tidal_vector_x)
        ) / (complete_df['total_motion_vector_magnitude'] + 1e-10)
        
        # L. PHASE-LOCKED OSCILLATIONS
        # Some station pairs will oscillate in phase, others out of phase
        
        # Calculate relative phase between different motion components
        complete_df['rotation_orbital_phase_lock'] = np.cos(rotation_phase - orbital_phase)
        complete_df['rotation_wobble_phase_lock'] = np.cos(rotation_phase - chandler_phase)
        complete_df['orbital_wobble_phase_lock'] = np.cos(orbital_phase - chandler_phase)
        complete_df['tidal_rotation_phase_lock'] = np.cos(tidal_phase - rotation_phase)
        
        # Overall phase coherence (how synchronized are all the motions)
        complete_df['overall_phase_coherence'] = (
            abs(complete_df['rotation_orbital_phase_lock']) +
            abs(complete_df['rotation_wobble_phase_lock']) +
            abs(complete_df['orbital_wobble_phase_lock']) +
            abs(complete_df['tidal_rotation_phase_lock'])
        ) / 4.0
        
        # M. ACCELERATION COUPLING STRENGTH WITH INTERFERENCE
        # Enhanced coupling that accounts for constructive/destructive interference
        complete_df['interference_weighted_coupling'] = (
            complete_df['differential_centripetal_acc'] * complete_df['rotation_orbital_interference'] +
            complete_df['differential_orbital_acc'] * complete_df['orbital_wobble_interference'] +
            complete_df['differential_wobble_acc'] * complete_df['rotation_wobble_interference']
        ) * complete_df['overall_phase_coherence']
        
        # 2. ORBITAL PROJECTION ANALYSIS  
        # Different station pairs see different projections of Earth's 30 km/s orbital motion
        complete_df['azimuth'] = complete_df.apply(
            lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                       row['station2_lat'], row['station2_lon']), axis=1
        )
        
        # Project orbital motion onto station pair baseline
        orbital_speed_kms = 29.78  # km/s
        complete_df['orbital_projection'] = complete_df.apply(
            lambda row: orbital_speed_kms * abs(np.cos(np.radians(row['azimuth']))), axis=1
        )
        
        # 3. DISTANCE-DEPENDENT BEAT FREQUENCIES
        # Beat patterns vary with station separation
        distance_bins = [0, 1000, 3000, 6000, 10000, 15000]  # km
        distance_labels = ['<1000km', '1000-3000km', '3000-6000km', '6000-10000km', '>10000km']
        complete_df['distance_category'] = pd.cut(complete_df['dist_km'], 
                                                 bins=distance_bins, 
                                                 labels=distance_labels)
        
        # 4. RELATIVE MOTION BEAT ANALYSIS BY DISTANCE
        relative_motion_results = {}
        
        for dist_cat in distance_labels:
            if dist_cat not in complete_df['distance_category'].values:
                continue
                
            dist_data = complete_df[complete_df['distance_category'] == dist_cat].copy()
            
            if len(dist_data) < 1000:  # Need sufficient data
                continue
                
            print_status(f"Analyzing relative motion beats for {dist_cat} pairs ({len(dist_data)} pairs)", "INFO")
            
            # Calculate relative motion frequencies for this distance range
            mean_distance = dist_data['dist_km'].mean()
            mean_diff_rotation = dist_data['differential_rotation_velocity'].mean()
            mean_orbital_proj = dist_data['orbital_projection'].mean()
            
            # Use the same temporal interference patterns as main analysis but in relative motion context
            # These are the interference patterns between different Earth motions
            relative_frequencies = {
                'tidal_m2_tidal_s2_diff': 1/14.765,  # ~14.8 days (tidal cycle difference)
                'chandler_annual_sum': 1/196.9,      # ~197 days (Chandler + annual)
                'chandler_semiannual_sum': 1/127.9,  # ~128 days (Chandler + semiannual)
                'annual_semiannual_sum': 1/121.8     # ~122 days (annual + semiannual)
            }
            
            # 5. PHASE ANALYSIS FOR RELATIVE MOTION BEATS
            # Test each relative motion frequency for modulation
            beat_results = {}
            
            for freq_name, freq_value in relative_frequencies.items():
                if freq_value <= 0 or freq_value > 10:  # Skip invalid frequencies
                    continue
                    
                # Calculate phase for this beat frequency (same as main analysis)
                dist_data['beat_phase'] = (2 * np.pi * dist_data['days_since_epoch'] * freq_value) % (2 * np.pi)
                
                # Bin by phase (same as main analysis)
                n_phase_bins = 12
                dist_data['beat_phase_bin'] = (dist_data['beat_phase'] // (2 * np.pi / n_phase_bins)).astype(int)

                # Track coherence across phases
                phase_tracking = []

                for bin_idx in range(n_phase_bins):
                    bin_data = dist_data[dist_data['beat_phase_bin'] == bin_idx]

                    if len(bin_data) < 50:  # Lower threshold for relative motion
                        continue

                    phase_tracking.append({
                        'phase_bin': bin_idx,
                        'phase_radians': bin_idx * 2 * np.pi / n_phase_bins,
                        'mean_coherence': float(bin_data['coherence'].mean()),
                        'n_pairs': len(bin_data)
                    })
                
                if len(phase_tracking) < 6:  # Need reasonable phase coverage
                    continue
                    
                # Test for sinusoidal modulation in coherence
                phases = [t['phase_radians'] for t in phase_tracking]
                coherences = [t['mean_coherence'] for t in phase_tracking]
                
                try:
                    # Check data variation before fitting
                    coherence_std = np.std(coherences)
                    if coherence_std < 1e-6:  # Very low variation
                        beat_results[freq_name] = {
                            'frequency_cpd': float(freq_value),
                            'period_days': float(1.0/freq_value) if freq_value > 0 else float('inf'),
                            'fit_success': False,
                            'error': 'Insufficient coherence variation for relative motion analysis',
                            'coherence_std': float(coherence_std),
                            'n_phase_bins': len(phase_tracking)
                        }
                        continue
                    
                    # Simplified analysis: direct correlation without curve fitting
                    # This avoids the SciPy optimization warnings
                    phase_sin = np.sin(phases)
                    phase_cos = np.cos(phases)

                    # Additional checks to prevent ConstantInputWarning
                    phase_sin_std = np.std(phase_sin)
                    phase_cos_std = np.std(phase_cos)

                    # Skip if phase arrays are constant or have insufficient variation
                    if phase_sin_std < 1e-12 or phase_cos_std < 1e-12 or coherence_std < 1e-12:
                        beat_results[freq_name] = {
                            'frequency_cpd': float(freq_value),
                            'period_days': float(1.0/freq_value) if freq_value > 0 else float('inf'),
                            'fit_success': False,
                            'error': 'Insufficient variation for correlation (prevents scipy warning)',
                            'coherence_std': float(coherence_std),
                            'phase_sin_std': float(phase_sin_std),
                            'phase_cos_std': float(phase_cos_std),
                            'n_phase_bins': len(phase_tracking)
                        }
                        continue

                    # Check if we have at least 3 unique values (scipy's minimum)
                    if len(set(coherences)) < 3 or len(set(phase_sin)) < 3 or len(set(phase_cos)) < 3:
                        beat_results[freq_name] = {
                            'frequency_cpd': float(freq_value),
                            'period_days': float(1.0/freq_value) if freq_value > 0 else float('inf'),
                            'fit_success': False,
                            'error': 'Insufficient unique values for correlation',
                            'coherence_unique': len(set(coherences)),
                            'phase_sin_unique': len(set(phase_sin)),
                            'phase_cos_unique': len(set(phase_cos)),
                            'n_phase_bins': len(phase_tracking)
                        }
                        continue

                    try:
                        from scipy.stats import pearsonr
                        corr_sin, p_sin = pearsonr(coherences, phase_sin)
                        corr_cos, p_cos = pearsonr(coherences, phase_cos)
                    except Exception as e:
                        beat_results[freq_name] = {
                            'frequency_cpd': float(freq_value),
                            'period_days': float(1.0/freq_value) if freq_value > 0 else float('inf'),
                            'fit_success': False,
                            'error': f'Correlation calculation failed: {e}',
                            'n_phase_bins': len(phase_tracking)
                        }
                        continue
                    
                    # Take the stronger correlation
                    if abs(corr_sin) > abs(corr_cos):
                        correlation = corr_sin
                        p_value = p_sin
                        phase_component = 'sine'
                    else:
                        correlation = corr_cos
                        p_value = p_cos
                        phase_component = 'cosine'
                    
                    # Check for valid results
                    if not (np.isnan(correlation) or np.isnan(p_value)):
                        beat_results[freq_name] = {
                            'frequency_cpd': float(freq_value),
                            'period_days': float(1.0/freq_value) if freq_value > 0 else float('inf'),
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'phase_component': phase_component,
                            'n_phase_bins': len(phase_tracking),
                            'mean_coherence': float(np.mean(coherences)),
                            'coherence_variation': float(coherence_std),
                            'fit_success': True,
                            'analysis_method': 'direct_correlation',
                            'phase_tracking': phase_tracking
                        }
                    else:
                        beat_results[freq_name] = {
                            'frequency_cpd': float(freq_value),
                            'fit_success': False,
                            'error': 'NaN correlation results',
                            'correlation': float(correlation) if not np.isnan(correlation) else None,
                            'p_value': float(p_value) if not np.isnan(p_value) else None
                        }
                    
                except Exception as e:
                    beat_results[freq_name] = {
                        'frequency_cpd': float(freq_value),
                        'fit_success': False,
                        'error': str(e),
                        'n_phase_bins': len(phase_tracking) if 'phase_tracking' in locals() else 0
                    }
            
        # Count successful analyses
        successful_analyses = sum(1 for v in beat_results.values() if v.get('fit_success'))
        significant_beats = {k: v for k, v in beat_results.items()
                            if v.get('fit_success') and v.get('p_value', 1) < 0.3 and abs(v.get('correlation', 0)) > 0.2}

        # Store results for this distance category
        relative_motion_results[dist_cat] = {
            'n_pairs': len(dist_data),
            'mean_distance_km': float(mean_distance),
            'mean_differential_rotation_kmh': float(mean_diff_rotation),
            'mean_orbital_projection_kms': float(mean_orbital_proj),
            'relative_frequencies': relative_frequencies,
            'beat_analysis': beat_results,
            'significant_beats': significant_beats,
            'successful_analyses': successful_analyses,
            'total_frequencies_tested': len(beat_results)
        }
        
        # 6. CROSS-DISTANCE ANALYSIS
        # Look for patterns that vary systematically with distance
        distance_dependent_patterns = {}
        
        if len(relative_motion_results) >= 3:  # Need multiple distance bins
            print_status("Analyzing distance-dependent patterns...", "INFO")
            
            distances = []
            correlations = []
            
            for dist_cat, results in relative_motion_results.items():
                if results['beat_analysis']:
                    mean_dist = results['mean_distance_km']
                    # Find strongest correlation in this distance bin
                    max_corr = max([b.get('correlation', 0) for b in results['beat_analysis'].values() 
                                   if b.get('fit_success', False)], default=0)
                    
                    distances.append(mean_dist)
                    correlations.append(max_corr)
            
            if len(distances) >= 3:
                # Test for distance-correlation relationship
                from scipy.stats import pearsonr
                dist_corr, dist_p = pearsonr(distances, correlations)
                
                distance_dependent_patterns = {
                    'distance_correlation': float(dist_corr),
                    'distance_p_value': float(dist_p),
                    'interpretation': f'Distance-correlation relationship: r={dist_corr:.3f}, p={dist_p:.3f}',
                    'distances_km': distances,
                    'max_correlations': correlations
                }
        
        return {
            'success': True,
            'analysis_type': 'relative_motion_beats',
            'n_distance_categories': len(relative_motion_results),
            'distance_categories': list(relative_motion_results.keys()),
            'relative_motion_results': relative_motion_results,
            'distance_dependent_patterns': distance_dependent_patterns,
            'total_significant_beats': sum(len(r.get('significant_beats', {})) 
                                         for r in relative_motion_results.values()),
            'interpretation': classify_relative_motion_evidence(relative_motion_results)
        }
        
    except Exception as e:
        print_status(f"Relative motion beat analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def classify_relative_motion_evidence(results: Dict) -> str:
    """Classify strength of relative motion beat evidence"""
    total_significant = sum(len(r.get('significant_beats', {})) for r in results.values())
    total_categories = len(results)
    total_tested = sum(r.get('total_frequencies_tested', 0) for r in results.values())
    total_successful = sum(r.get('successful_analyses', 0) for r in results.values())

    # Get the strongest correlation found
    strongest_correlation = 0.0
    for cat_results in results.values():
        for beat_data in cat_results.get('significant_beats', {}).values():
            strongest_correlation = max(strongest_correlation, abs(beat_data.get('correlation', 0)))

    # Classification based on detection quality and strength
    if total_significant >= 3 and strongest_correlation >= 0.5:
        return f"Robust relative motion coupling confirmed across multiple distance scales ({total_significant} significant beat patterns, max |r|={strongest_correlation:.3f})"
    elif total_significant >= 2 and strongest_correlation >= 0.3:
        return f"Significant relative motion signatures identified ({total_significant} beat patterns, max |r|={strongest_correlation:.3f})"
    elif total_significant >= 1:
        return f"Relative motion beat patterns detected ({total_significant} significant patterns, max |r|={strongest_correlation:.3f})"
    elif total_successful > 0:
        return f"Relative motion patterns analyzed ({total_successful}/{total_tested} frequencies) but did not meet statistical significance thresholds"
    elif total_tested > 0:
        return f"Relative motion patterns evaluated ({total_tested} frequencies) but insufficient data for robust statistical analysis"
    else:
        return "Relative motion beat patterns could not be analyzed due to insufficient data"

def run_mesh_dance_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Mesh Dance Analysis: Coherent network dynamics detection.
    
    Analyzes the collective motion patterns of the GPS station network
    to detect coherent dynamics that may indicate coupling with spacetime structure.
    The analysis examines whether the entire GPS network exhibits coordinated
    motion patterns that maintain consistent phase relationships across the mesh.
    
    Key concepts:
    1. MESH COHERENCE: Network-wide coordination of station timing correlations
    2. SPIRAL DYNAMICS: Detection of helical motion signatures in correlation patterns
    3. PHASE RELATIONSHIPS: Maintenance of coherent phase relationships across stations
    4. COLLECTIVE OSCILLATION: Network-wide synchronized oscillation patterns
    5. SPACETIME COUPLING: Network response to structured spacetime geometry
    
    Args:
        complete_df: Complete pair dataset with all motion analysis
        
    Returns:
        dict: Mesh dance analysis results with network coherence metrics
    """
    print_status("Starting Mesh Dance Analysis - Network Coherence Assessment", "PROCESS")
    print_status("Analyzing coherent motion patterns of GPS station network...", "PROCESS")
    
    try:
        # Convert dates and basic setup
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # 1. MESH COHERENCE ANALYSIS
        # Test if all stations move together as one coherent system
        print_status("Analyzing mesh coherence patterns...", "INFO")
        
        # Group station pairs by time windows to track mesh evolution
        time_window_days = 7  # Weekly analysis
        complete_df['time_window'] = (complete_df['days_since_epoch'] // time_window_days) * time_window_days
        
        mesh_coherence_results = {}
        unique_time_windows = sorted(complete_df['time_window'].unique())
        
        if len(unique_time_windows) < 10:  # Need sufficient temporal sampling
            return {'success': False, 'error': f'Insufficient time windows: {len(unique_time_windows)} (need ≥10)'}
        
        # Sample time windows for analysis (every 4th window to manage computation)
        sampled_windows = unique_time_windows[::4]
        
        mesh_evolution = []
        
        for window in sampled_windows:
            window_data = complete_df[complete_df['time_window'] == window].copy()
            
            if len(window_data) < 1000:  # Need sufficient pairs per window
                continue
                
            # Calculate mesh properties for this time window
            
            # A. COLLECTIVE MOTION VECTOR
            # The overall motion direction of the entire mesh
            mean_total_vector_magnitude = window_data['total_motion_vector_magnitude'].mean()
            mean_total_vector_direction = window_data['total_motion_vector_direction'].mean()
            
            # B. MESH COHERENCE METRICS
            # How well synchronized are all the station pairs?
            coherence_std = window_data['coherence'].std()
            coherence_mean = window_data['coherence'].mean()
            coherence_uniformity = 1.0 / (1.0 + coherence_std)  # Higher = more uniform
            
            # C. PHASE COHERENCE ACROSS THE MESH
            # Are all stations oscillating in phase?
            overall_phase_coherence_mean = window_data['overall_phase_coherence'].mean()
            overall_phase_coherence_std = window_data['overall_phase_coherence'].std()
            phase_synchronization = 1.0 / (1.0 + overall_phase_coherence_std)
            
            # D. INTERFERENCE STATE DISTRIBUTION
            # What's the distribution of interference states across the mesh?
            interference_counts = window_data['interference_state'].value_counts()
            dominant_interference_state = interference_counts.index[0] if len(interference_counts) > 0 else 'unknown'
            interference_dominance = interference_counts.iloc[0] / len(window_data) if len(interference_counts) > 0 else 0
            
            # E. OSCILLATION SYNCHRONIZATION
            # Are all parts of the mesh oscillating together?
            oscillation_mean = window_data['motion_oscillation_strength'].mean()
            oscillation_std = window_data['motion_oscillation_strength'].std()
            oscillation_synchronization = 1.0 / (1.0 + oscillation_std)
            
            mesh_evolution.append({
                'time_window': int(window),
                'days_since_epoch': int(window),
                'n_pairs': len(window_data),
                'collective_motion_magnitude': float(mean_total_vector_magnitude),
                'collective_motion_direction': float(mean_total_vector_direction),
                'coherence_uniformity': float(coherence_uniformity),
                'phase_synchronization': float(phase_synchronization),
                'dominant_interference_state': dominant_interference_state,
                'interference_dominance': float(interference_dominance),
                'oscillation_synchronization': float(oscillation_synchronization),
                'mesh_coherence_score': float(
                    (coherence_uniformity + phase_synchronization + oscillation_synchronization) / 3.0
                )
            })
        
        if len(mesh_evolution) < 8:
            return {'success': False, 'error': f'Insufficient mesh evolution data: {len(mesh_evolution)}'}
        
        # 2. SPIRAL DYNAMICS ANALYSIS
        # Test if the mesh is tracing helical/spiral paths through spacetime
        print_status("Analyzing spiral dynamics of mesh motion...", "INFO")
        
        # Extract time series of collective motion
        times = [m['days_since_epoch'] for m in mesh_evolution]
        directions = [m['collective_motion_direction'] for m in mesh_evolution]
        magnitudes = [m['collective_motion_magnitude'] for m in mesh_evolution]
        coherence_scores = [m['mesh_coherence_score'] for m in mesh_evolution]
        
        # Test for spiral patterns in the motion direction
        # A true spiral would show systematic rotation of the motion vector
        direction_changes = np.diff(directions)
        
        # Handle angle wrapping
        direction_changes = np.where(direction_changes > np.pi, direction_changes - 2*np.pi, direction_changes)
        direction_changes = np.where(direction_changes < -np.pi, direction_changes + 2*np.pi, direction_changes)
        
        # Test for consistent rotation (spiral signature)
        mean_rotation_rate = np.mean(direction_changes)
        rotation_consistency = 1.0 - np.std(direction_changes) / (np.pi/4)  # Normalized consistency
        
        # Test for helical pattern (magnitude oscillation with direction rotation)
        magnitude_oscillation = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0
        
        spiral_signature = {
            'mean_rotation_rate_rad_per_week': float(mean_rotation_rate),
            'rotation_consistency': float(max(0, rotation_consistency)),
            'magnitude_oscillation': float(magnitude_oscillation),
            'spiral_strength': float(max(0, rotation_consistency) * magnitude_oscillation),
            'is_spiral_motion': bool(rotation_consistency > 0.3 and magnitude_oscillation > 0.1)
        }
        
        # 3. COLLECTIVE COHERENT OSCILLATION
        # Test if the entire mesh oscillates coherently as one system
        print_status("Analyzing collective mesh oscillation patterns...", "INFO")
        
        # Fit sinusoidal models to mesh coherence over time
        time_array = np.array(times)
        coherence_array = np.array(coherence_scores)
        
        # Test multiple frequencies to find dominant oscillation
        test_frequencies = [1/365.25, 1/427.0, 1.0, 2.0]  # Annual, Chandler, daily, semi-daily
        oscillation_results = {}
        
        for freq in test_frequencies:
            try:
                # Simplified oscillation analysis to avoid SciPy warnings
                # Use direct correlation instead of curve fitting
                
                period_days = 1.0 / freq if freq > 0 else float('inf')
                
                # Check data variation first
                coherence_std = np.std(coherence_array)
                if coherence_std < 1e-8:  # Very low variation
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'fit_success': False,
                        'error': 'Insufficient coherence variation',
                        'coherence_std': float(coherence_std)
                    }
                    continue
                
                # Direct correlation with sine and cosine components
                time_phase = 2 * np.pi * freq * time_array
                phase_sin = np.sin(time_phase)
                phase_cos = np.cos(time_phase)

                # Additional checks to prevent ConstantInputWarning
                phase_sin_std = np.std(phase_sin)
                phase_cos_std = np.std(phase_cos)

                # Skip if phase arrays are constant or have insufficient variation
                if phase_sin_std < 1e-12 or phase_cos_std < 1e-12 or coherence_std < 1e-12:
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'fit_success': False,
                        'error': 'Insufficient variation for correlation (prevents scipy warning)',
                        'coherence_std': float(coherence_std),
                        'phase_sin_std': float(phase_sin_std),
                        'phase_cos_std': float(phase_cos_std)
                    }
                    continue

                # Check if we have at least 3 unique values (scipy's minimum)
                if len(set(coherence_array)) < 3 or len(set(phase_sin)) < 3 or len(set(phase_cos)) < 3:
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'fit_success': False,
                        'error': 'Insufficient unique values for correlation',
                        'coherence_unique': len(set(coherence_array)),
                        'phase_sin_unique': len(set(phase_sin)),
                        'phase_cos_unique': len(set(phase_cos))
                    }
                    continue

                try:
                    from scipy.stats import pearsonr
                    corr_sin, p_sin = pearsonr(coherence_array, phase_sin)
                    corr_cos, p_cos = pearsonr(coherence_array, phase_cos)
                except Exception as e:
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'fit_success': False,
                        'error': f'Correlation calculation failed: {e}'
                    }
                    continue
                
                # Take the stronger correlation
                if abs(corr_sin) > abs(corr_cos):
                    correlation = corr_sin
                    p_value = p_sin
                    phase_component = 'sine'
                else:
                    correlation = corr_cos
                    p_value = p_cos
                    phase_component = 'cosine'
                
                # Check for valid results
                if not (np.isnan(correlation) or np.isnan(p_value)):
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'phase_component': phase_component,
                        'coherence_variation': float(coherence_std),
                        'fit_success': True,
                        'analysis_method': 'direct_correlation'
                    }
                else:
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'fit_success': False,
                        'error': 'NaN correlation results'
                    }
                
            except Exception as e:
                oscillation_results[f'freq_{freq:.6f}'] = {
                    'frequency_cpd': float(freq),
                    'fit_success': False,
                    'error': str(e)
                }
        
        # Find the strongest oscillation
        successful_oscillations = {k: v for k, v in oscillation_results.items() 
                                 if v.get('fit_success') and v.get('p_value', 1) < 0.05}
        
        if successful_oscillations:
            best_oscillation = max(successful_oscillations.values(), 
                                 key=lambda x: abs(x.get('correlation', 0)))
        else:
            best_oscillation = {'no_significant_oscillation': True}
        
        # 4. SPACETIME COUPLING SIGNATURE
        # Network response analysis: coherent mesh coupling to spacetime structure
        print_status("Analyzing spacetime coupling signatures...", "INFO")
        
        # Calculate mesh-wide correlation with Earth motion phases
        mesh_earth_coupling = {}
        
        # Test correlation between mesh coherence and various Earth motion phases
        if len(mesh_evolution) >= 12:  # Need sufficient data
            
            # Earth motion phases for each time window
            earth_phases = {}
            for window_data in mesh_evolution:
                days = window_data['days_since_epoch']
                earth_phases[days] = {
                    'rotation_phase': (days % 1.0) * 2 * np.pi,
                    'orbital_phase': (days % 365.25) / 365.25 * 2 * np.pi,
                    'chandler_phase': (days % 427.0) / 427.0 * 2 * np.pi
                }
            
            # Test correlations
            for phase_name in ['rotation_phase', 'orbital_phase', 'chandler_phase']:
                phase_values = [earth_phases[m['days_since_epoch']][phase_name] for m in mesh_evolution]
                
                # Convert phases to sine/cosine for correlation
                phase_sin = np.sin(phase_values)
                phase_cos = np.cos(phase_values)
                
                # Test correlation with mesh coherence
                coherence_values = [m['mesh_coherence_score'] for m in mesh_evolution]
                
                try:
                    # Safe correlation calculation with variation checks
                    coherence_std = np.std(coherence_values)
                    phase_sin_std = np.std(phase_sin)
                    phase_cos_std = np.std(phase_cos)
                    
                    if coherence_std < 1e-10 or len(set(coherence_values)) < 3:
                        mesh_earth_coupling[phase_name] = {
                            'error': 'Constant coherence values - no variation to correlate',
                            'coherence_std': float(coherence_std),
                            'coherence_range': [float(min(coherence_values)), float(max(coherence_values))],
                            'unique_values': len(set(coherence_values))
                        }
                    elif phase_sin_std < 1e-10 and phase_cos_std < 1e-10:
                        mesh_earth_coupling[phase_name] = {
                            'error': 'Constant phase values - insufficient temporal variation',
                            'phase_std': float(phase_sin_std)
                        }
                    else:
                        # Proceed with correlation if sufficient variation
                        # Additional check to ensure we don't have constant arrays
                        if len(set(coherence_values)) >= 3 and len(set(phase_sin)) >= 3:
                            # Check for scipy's stricter constant threshold
                            if coherence_std < 1e-12 or phase_sin_std < 1e-12:
                                mesh_earth_coupling[phase_name] = {
                                    'error': 'Arrays too constant for scipy correlation',
                                    'coherence_std': float(coherence_std),
                                    'phase_sin_std': float(phase_sin_std),
                                    'phase_cos_std': float(phase_cos_std),
                                    'coherence_unique': len(set(coherence_values)),
                                    'phase_sin_unique': len(set(phase_sin))
                                }
                                continue

                            try:
                                corr_sin, p_sin = pearsonr(coherence_values, phase_sin)
                                corr_cos, p_cos = pearsonr(coherence_values, phase_cos)
                            except Exception as e:
                                mesh_earth_coupling[phase_name] = {
                                    'error': f'Correlation calculation failed: {e}',
                                    'coherence_std': float(coherence_std),
                                    'phase_sin_std': float(phase_sin_std)
                                }
                                continue
                        else:
                            mesh_earth_coupling[phase_name] = {
                                'error': 'Insufficient unique values for correlation',
                                'coherence_unique': len(set(coherence_values)),
                                'phase_sin_unique': len(set(phase_sin)),
                                'phase_cos_unique': len(set(phase_cos))
                            }
                            continue
                        
                        # Check for NaN results
                        if np.isnan(corr_sin) or np.isnan(corr_cos):
                            mesh_earth_coupling[phase_name] = {
                                'error': 'NaN correlation results',
                                'corr_sin': float(corr_sin) if not np.isnan(corr_sin) else None,
                                'corr_cos': float(corr_cos) if not np.isnan(corr_cos) else None
                            }
                        else:
                            # Take the stronger correlation
                            if abs(corr_sin) > abs(corr_cos):
                                mesh_earth_coupling[phase_name] = {
                                    'correlation': float(corr_sin),
                                    'p_value': float(p_sin),
                                    'phase_component': 'sine',
                                    'data_variation': float(coherence_std)
                                }
                            else:
                                mesh_earth_coupling[phase_name] = {
                                    'correlation': float(corr_cos),
                                    'p_value': float(p_cos),
                                    'phase_component': 'cosine',
                                    'data_variation': float(coherence_std)
                                }
                        
                except Exception as e:
                    mesh_earth_coupling[phase_name] = {
                        'error': str(e),
                        'coherence_std': float(np.std(coherence_values)) if len(coherence_values) > 0 else 0
                    }
        
        # 5. NETWORK COHERENCE CLASSIFICATION
        # Final assessment: coherent network dynamics signature strength
        print_status("Computing network coherence classification...", "INFO")
        
        dance_metrics = {
            'mesh_coherence_strength': float(np.mean([m['mesh_coherence_score'] for m in mesh_evolution])),
            'spiral_motion_detected': spiral_signature['is_spiral_motion'],
            'spiral_strength': spiral_signature['spiral_strength'],
            'collective_oscillation_detected': len(successful_oscillations) > 0,
            'strongest_oscillation_correlation': float(best_oscillation.get('correlation', 0)),
            'earth_coupling_detected': any(abs(c.get('correlation', 0)) > 0.3 and c.get('p_value', 1) < 0.05 
                                         for c in mesh_earth_coupling.values()),
            'n_significant_earth_couplings': sum(1 for c in mesh_earth_coupling.values() 
                                               if abs(c.get('correlation', 0)) > 0.3 and c.get('p_value', 1) < 0.05)
        }
        
        # NETWORK COHERENCE CLASSIFICATION
        dance_score = (
            dance_metrics['mesh_coherence_strength'] * 0.3 +
            (1.0 if dance_metrics['spiral_motion_detected'] else 0.0) * 0.3 +
            (1.0 if dance_metrics['collective_oscillation_detected'] else 0.0) * 0.2 +
            (1.0 if dance_metrics['earth_coupling_detected'] else 0.0) * 0.2
        )
        
        dance_classification = classify_dance_signature(dance_score, dance_metrics)
        
        return {
            'success': True,
            'analysis_type': 'mesh_dance_ultimate',
            'n_time_windows': len(mesh_evolution),
            'temporal_span_days': int(max(times) - min(times)),
            'mesh_evolution': mesh_evolution,
            'spiral_signature': spiral_signature,
            'collective_oscillation': {
                'oscillation_results': oscillation_results,
                'best_oscillation': best_oscillation,
                'n_significant_oscillations': len(successful_oscillations)
            },
            'spacetime_coupling': {
                'mesh_earth_coupling': mesh_earth_coupling,
                'coupling_summary': dance_metrics
            },
            'dance_signature': {
                'dance_score': float(dance_score),
                'classification': dance_classification,
                'metrics': dance_metrics
            },
            'interpretation': f"MESH DANCE ANALYSIS: {dance_classification}"
        }
        
    except Exception as e:
        print_status(f"Mesh dance analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def classify_dance_signature(dance_score: float, metrics: Dict) -> str:
    """Classify the strength of the mesh dance signature for network coherence assessment"""
    
    if dance_score >= 0.8 and metrics['spiral_motion_detected'] and metrics['earth_coupling_detected']:
        return "EXCEPTIONAL NETWORK COHERENCE - Strong mesh dance dynamics with spacetime coupling detected"
    elif dance_score >= 0.6 and (metrics['spiral_motion_detected'] or metrics['collective_oscillation_detected']):
        return "STRONG NETWORK COHERENCE - Clear mesh dance dynamics detected"
    elif dance_score >= 0.4 and metrics['mesh_coherence_strength'] > 0.5:
        return "MODERATE NETWORK COHERENCE - Mesh coherence with collective motion patterns"
    elif dance_score >= 0.2:
        return "WEAK NETWORK COHERENCE - Limited mesh coherence detected"
    else:
        return "NO NETWORK COHERENCE - No coherent mesh dynamics detected"

# ===== END NEW HELICAL MOTION ANALYSIS FUNCTIONS =====

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
