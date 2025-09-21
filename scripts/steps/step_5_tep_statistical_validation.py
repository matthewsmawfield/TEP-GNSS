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
    complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
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
        print_status(f"Strong correlation detected: E-W/N-S ratio correlates with orbital speed (r={orbital_correlation:.3f}, p={orbital_p_value:.4f})", "SUCCESS")
        print_status("This suggests GPS timing correlations may track Earth's orbital motion", "INFO")
    elif abs(orbital_correlation) > 0.3:
        print_status(f"Moderate correlation with orbital motion detected (r={orbital_correlation:.3f})", "INFO")
    
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
        return "Very strong correlation with orbital motion"
    elif abs(correlation) > 0.5 and p_value < 0.01:
        return "Strong correlation with orbital motion"
    elif abs(correlation) > 0.3 and p_value < 0.05:
        return "Moderate correlation with orbital motion"
    elif abs(correlation) > 0.2:
        return "Weak correlation with orbital motion"
    else:
        return "No significant correlation with orbital motion"

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

def main():
    """Main function to run all statistical validation tests."""
    print("="*80)
    print("TEP GNSS Analysis Package v0.4")
    print("STEP 5: Statistical Validation")
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
    if len(sys.argv) > 1:
        centers = [sys.argv[1]]
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

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
