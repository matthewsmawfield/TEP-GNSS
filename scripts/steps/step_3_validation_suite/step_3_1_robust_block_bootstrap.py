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
Date: October 2025
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

# Ensure the directory containing 'scripts' is on the path
script_dir = Path(__file__).resolve()
project_root = script_dir.parents[3] # Point to project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Define PACKAGE_ROOT for consistent usage throughout the script
PACKAGE_ROOT = project_root

# Import project utilities
from scripts.utils.config import TEPConfig
from scripts.utils.logger import print_status, check_memory_usage, TEPLogger, set_step_logger # Import global functions

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_3_1_robust_block_bootstrap",
    level="DEBUG",
    log_file_path=Path(__file__).resolve().parents[3] / "logs" / "step_3_1_robust_block_bootstrap.log"
)

# Register step logger so print_status uses it
set_step_logger(step_logger)
from scripts.utils.exceptions import TEPFileError, TEPDataError, TEPAnalysisError, safe_csv_read, safe_json_write # Import TEPAnalysisError and safe_csv_read, safe_json_write
from scripts.utils.pid_manager import ensure_single_instance

# The global logger instance is handled by scripts.utils.logger
# No need for local logger initialization here

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def load_station_coordinates():
    """Load station coordinates for metadata."""
    coord_file = PACKAGE_ROOT / "data/coordinates/step_1_1_station_coords_global.csv"
    if not coord_file.exists():
        raise TEPFileError(f"Station coordinates file not found: {coord_file} (Ensure Step 1.1 is complete)")
    
    try:
        coords_df = safe_csv_read(coord_file)
        if coords_df is None or len(coords_df) == 0:
            raise TEPDataError("Station coordinates file is empty or unreadable")
        return coords_df
    except Exception as e:
        raise TEPDataError(f"Failed to load station coordinates: {e}")

def load_complete_pair_dataset_chunked(ac: str, chunk_size: int = None) -> pd.DataFrame:
    """
    Load complete pair dataset with chunking for large datasets.
    
    Args:
        ac: Analysis center name
        chunk_size: Size of chunks to process (None for auto-sizing)
        
    Returns:
        Complete pair dataset DataFrame
    """
    if chunk_size is None:
        # Auto-size chunks based on dataset size
        min_chunk = TEPConfig.get_int('TEP_MIN_CHUNK_SIZE', 25000)
        max_chunk = TEPConfig.get_int('TEP_MAX_CHUNK_SIZE', 100000)
        chunk_size = min_chunk  # Start with minimum chunk size
    
    print_status(f"Loading complete pair dataset for {ac} with chunking (chunk_size={chunk_size})...", "PROCESS")
    
    # Try to load from Step 2.1 geospatial data first (preferred)
    step_2_1_file = project_root / "data" / "processed" / f"step_2_1_geospatial_{ac}.csv"
    
    if step_2_1_file.exists():
        print_status(f"Using filtered Step 2.1 geospatial data: {step_2_1_file.name}", "INFO")
        
        # For large files, use chunking
        file_size_mb = step_2_1_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # If file is larger than 100MB, use chunking
            print_status(f"Large file detected ({file_size_mb:.1f} MB), using chunked loading...", "INFO")
            
            chunks = []
            for chunk in pd.read_csv(step_2_1_file, chunksize=chunk_size):
                # Create coherence column in each chunk if needed
                if 'coherence' not in chunk.columns and 'plateau_phase' in chunk.columns:
                    chunk['coherence'] = np.cos(chunk['plateau_phase'])
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    print_status(f"Loaded {len(chunks)} chunks...", "DEBUG")
            
            complete_df = pd.concat(chunks, ignore_index=True)
            del chunks  # Free memory
            gc.collect()
        else:
            complete_df = pd.read_csv(step_2_1_file)
            # Create coherence column if needed
            if 'coherence' not in complete_df.columns and 'plateau_phase' in complete_df.columns:
                complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
    else:
        # Fallback to original method
        return load_complete_pair_dataset(ac)
    
    print_status(f"Loaded Step 2.1 dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")
    print_status(f"Columns: {list(complete_df.columns)}", "DEBUG")
    
    # Validate and create coherence column if needed
    if 'coherence' not in complete_df.columns:
        if 'plateau_phase' in complete_df.columns:
            complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
            print_status("Created coherence column from plateau_phase", "INFO")
        else:
            raise TEPDataError("Neither 'coherence' nor 'plateau_phase' column found in dataset")
    
    # Now validate all required columns exist
    required_cols = ['station_i', 'station_j', 'date', 'dist_km', 'coherence']
    missing_cols = [col for col in required_cols if col not in complete_df.columns]
    if missing_cols:
        raise TEPDataError(f"Missing required columns: {missing_cols}")
    
    # Convert date column if needed
    if not pd.api.types.is_datetime64_any_dtype(complete_df['date']):
        try:
            complete_df['date'] = pd.to_datetime(complete_df['date'])
        except Exception as e:
            print_status(f"Warning: Could not convert date column: {e}", "WARNING")
    
    print_status(f"Date range: {complete_df['date'].min()} to {complete_df['date'].max()}", "DEBUG")
    return complete_df

def load_complete_pair_dataset(ac: str) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center.
    Uses consolidated data for consistency with main Step 2.0 analysis.
    Implements chunked loading for large datasets to manage memory.
    """
    print_status(f"Loading complete pair dataset for {ac}...", "PROCESS")
    
    # Prefer filtered Step 2.1 data to stay consistent with geospatial cleaning
    step21_file = PACKAGE_ROOT / "data" / "processed" / f"step_2_1_geospatial_{ac}.csv"

    if step21_file.exists():
        print_status(f"Using filtered Step 2.1 geospatial data: {step21_file.name}", "INFO")
        try:
            # Check file size to determine if chunked loading is needed
            file_size_mb = step21_file.stat().st_size / (1024 * 1024)
            chunk_size = TEPConfig.get_int('TEP_MAX_CHUNK_SIZE', 100000)
            
            if file_size_mb > 500:  # Large file - use chunked loading
                print_status(f"Large file detected ({file_size_mb:.1f} MB), using chunked loading...", "INFO")
                chunks = []
                for chunk in pd.read_csv(step21_file, chunksize=chunk_size, parse_dates=['date']):
                    chunks.append(chunk)
                    if len(chunks) % 10 == 0:
                        print_status(f"Loaded {len(chunks)} chunks...", "DEBUG")
                complete_df = pd.concat(chunks, ignore_index=True)
                del chunks  # Free memory
                gc.collect()
            else:
                complete_df = pd.read_csv(step21_file, parse_dates=['date'])
                
            if 'plateau_phase' in complete_df.columns and 'coherence' not in complete_df.columns:
                complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
            print_status(f"Loaded Step 2.1 dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")
            return complete_df
        except Exception as e:
            print_status(f"Failed to load Step 2.1 data: {e}", "WARNING")
            print_status("Falling back to Step 2.0 consolidated data...", "WARNING")

    # Use consolidated Step 2.0 data when Step 2.1 is unavailable
    consolidated_file = PACKAGE_ROOT / "results" / "outputs" / f"step_2_0_pairs_consolidated_{ac}.csv"
    
    if consolidated_file.exists():
        print_status(f"Using consolidated data: {consolidated_file.name}", "INFO")
        try:
            complete_df = pd.read_csv(consolidated_file, parse_dates=['date'])
            # Ensure coherence column exists
            if 'plateau_phase' in complete_df.columns and 'coherence' not in complete_df.columns:
                complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
            print_status(f"Loaded consolidated dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")
            return complete_df
        except Exception as e:
            print_status(f"Failed to load consolidated data: {e}", "WARNING")
            print_status("Falling back to individual pair files...", "INFO")
    else:
        print_status(f"Consolidated file not found: {consolidated_file.name}", "WARNING")
        print_status("Using individual pair files (WARNING: may not match main analysis data)", "WARNING")
    
    # Fallback to individual files if consolidated not available
    pairs_dir = PACKAGE_ROOT / "results/tmp"
    if not pairs_dir.exists():
        raise TEPFileError(f"Pairs directory not found: {pairs_dir} (Ensure Step 2.0 is complete and TEP_WRITE_PAIR_LEVEL is set)")
    
    # Find all pair files for this analysis center
    pair_files = list(pairs_dir.glob(f"step_2_0_pairs_{ac}_*.csv"))
    if not pair_files:
        raise TEPFileError(f"No pair files found for analysis center: {ac} (Ensure Step 2.0 is complete)")
    
    print_status(f"Found {len(pair_files)} pair files for {ac} (fallback mode)", "INFO")
    
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

def precompute_station_pairs(complete_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Pre-compute station pairs to avoid repeated calculations during bootstrap.
    Memory-efficient version that only stores indices instead of full data copies.
    
    Args:
        complete_df: Complete pair dataset
        
    Returns:
        Dict mapping station names to their pair indices (for memory efficiency)
    """
    print_status("Pre-computing station pairs for efficient bootstrap sampling...", "PROCESS")
    
    # Get unique stations
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    
    # Create a mapping of station to its pair indices (memory efficient)
    station_pairs = {}
    
    for i, station in enumerate(unique_stations):
        if i % 50 == 0:  # Progress reporting every 50 stations
            print_status(f"Pre-computing station {i+1}/{len(unique_stations)}: {station}", "PROCESS")
        
        # Find all pairs involving this station
        station_mask = (complete_df['station_i'] == station) | (complete_df['station_j'] == station)
        # Store indices instead of full data to save memory
        station_pairs[station] = complete_df[station_mask].index.tolist()
        
        # Periodic memory cleanup during pre-computation
        if (i + 1) % 100 == 0:
            cleanup_memory(force_gc=True, log_usage=False)
    
    print_status(f"Pre-computed pair indices for {len(station_pairs)} stations", "SUCCESS")
    return station_pairs

def create_station_bootstrap_sample_optimized(station_pairs: Dict[str, List[int]], 
                                            stations_to_sample: List[str], 
                                            bootstrap_id: int,
                                            complete_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a bootstrap sample using pre-computed station pairs for efficiency.
    
    Args:
        station_pairs: Pre-computed station pairs dictionary
        stations_to_sample: List of unique stations available for sampling
        bootstrap_id: Bootstrap iteration ID for reproducibility
        
    Returns:
        Bootstrap sample DataFrame containing pairs from resampled stations
    """
    min_stations = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_STATIONS', 100)
    available_stations = len(stations_to_sample)

    if available_stations < min_stations:
        raise TEPDataError(
            f"Insufficient unique stations for station bootstrap: required {min_stations}, available {available_stations}"
        )

    # Sample all available stations with replacement
    n_stations_to_sample = available_stations
    
    # Set seed for reproducibility
    np.random.seed(42 + bootstrap_id)
    
    # Sample stations with replacement
    sampled_stations = np.random.choice(stations_to_sample, 
                                       size=n_stations_to_sample, 
                                       replace=True)
    
    # Use pre-computed indices for efficient sampling
    bootstrap_indices = set()
    for station in sampled_stations:
        if station in station_pairs:
            bootstrap_indices.update(station_pairs[station])
    
    # Convert indices back to DataFrame
    if bootstrap_indices:
        bootstrap_sample = complete_df.loc[list(bootstrap_indices)].copy()
        return bootstrap_sample
    else:
        return pd.DataFrame()

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
    available_stations = len(stations_to_sample)

    if available_stations < min_stations:
        raise TEPDataError(
            f"Insufficient unique stations for station bootstrap: required {min_stations}, available {available_stations}"
        )

    # Sample all available stations with replacement
    n_stations_to_sample = available_stations
    
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
    available_days = len(days_to_sample)

    if available_days < min_days:
        raise TEPDataError(
            f"Insufficient unique days for day bootstrap: required {min_days}, available {available_days}"
        )

    n_days_to_sample = available_days
    
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

    available_stations = len(stations_to_sample)
    available_days = len(days_to_sample)

    if available_stations < min_stations:
        raise TEPDataError(
            f"Insufficient unique stations for hybrid bootstrap: required {min_stations}, available {available_stations}"
        )

    if available_days < min_days:
        raise TEPDataError(
            f"Insufficient unique days for hybrid bootstrap: required {min_days}, available {available_days}"
        )

    n_stations_to_sample = available_stations
    n_days_to_sample = available_days
    
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
        
        # Validate data before fitting
        if np.any(~np.isfinite(distances)) or np.any(~np.isfinite(coherences)):
            return None, False, {'error': 'invalid_data', 'has_nan': True}
        
        if len(np.unique(distances)) < 3:
            return None, False, {'error': 'insufficient_unique_distances', 'n_unique': len(np.unique(distances))}
        
        # Initial parameter estimates
        c_range = coherences.max() - coherences.min()
        if c_range <= 0:
            return None, False, {'error': 'no_coherence_variation', 'coherence_range': c_range}
        
        p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS', 4000), coherences.min()]
        
        try:
            # Fit exponential model with robust bounds
            popt, pcov = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=TEPConfig.get_adaptive_lambda_bounds(distances),
                maxfev=5000
            )
        except Exception as fit_error:
            return None, False, {'error': f'curve_fit_failed: {str(fit_error)}', 'p0': p0}
        
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

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        return rss_mb, vms_mb
    except Exception:
        return 0, 0

def cleanup_memory(force_gc=True, log_usage=True):
    """
    Aggressive memory cleanup between major operations.
    
    Args:
        force_gc: Whether to force garbage collection
        log_usage: Whether to log memory usage
    """
    if force_gc:
        # Multiple rounds of garbage collection
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        # Temporarily lower GC thresholds for more aggressive cleanup
        if hasattr(gc, 'set_threshold'):
            old_thresholds = gc.get_threshold()
            gc.set_threshold(50, 5, 5)  # More aggressive thresholds
            gc.collect()
            gc.set_threshold(*old_thresholds)
    
    if log_usage:
        rss_mb, vms_mb = get_memory_usage()
        print_status(f"Memory cleanup: RSS={rss_mb:.2f} MB, VMS={vms_mb:.2f} MB", "DEBUG")

def monitor_memory_usage(operation_name: str, threshold_mb: float = 2000):
    """
    Monitor memory usage and trigger cleanup if needed.
    
    Args:
        operation_name: Name of the operation for logging
        threshold_mb: Memory threshold in MB to trigger cleanup
    """
    rss_mb, vms_mb = get_memory_usage()
    
    if rss_mb > threshold_mb:
        print_status(f"High memory usage detected in {operation_name}: {rss_mb:.2f} MB", "WARNING")
        cleanup_memory(force_gc=True, log_usage=True)
        return True
    
    # Critical memory check - if we're approaching system limits, abort
    if rss_mb > 8000:  # 8GB limit to prevent system kill
        print_status(f"CRITICAL: Memory usage too high ({rss_mb:.2f} MB) - aborting to prevent system kill", "ERROR")
        raise MemoryError(f"Memory usage exceeded safe limit: {rss_mb:.2f} MB")
    
    return False

def process_bootstrap_sample_parallel(args):
    """
    Process a single bootstrap sample in parallel with memory management.
    NOTE: This function is currently disabled due to multiprocessing issues with large DataFrames.
    Use sequential processing instead.
    
    Args:
        args: Tuple of (bootstrap_id, complete_df_path, station_pairs, unique_stations, min_pairs)
        
    Returns:
        Tuple of (bootstrap_id, result_dict, success_flag)
    """
    bootstrap_id, complete_df_path, station_pairs, unique_stations, min_pairs = args
    
    try:
        # Load DataFrame in worker process to avoid serialization issues
        complete_df = pd.read_parquet(complete_df_path) if complete_df_path.endswith('.parquet') else pd.read_csv(complete_df_path)
        
        # Create bootstrap sample using standard method (optimized method has issues with multiprocessing)
        bootstrap_sample = create_station_bootstrap_sample(complete_df, unique_stations, bootstrap_id)
        
        if len(bootstrap_sample) < min_pairs:
            del bootstrap_sample, complete_df
            gc.collect()
            return bootstrap_id, None, False
        
        # Fit correlation model
        fitted_params, fit_success, diagnostics = fit_correlation_model_bootstrap(bootstrap_sample)
        
        # Clean up immediately after use
        del bootstrap_sample, complete_df
        gc.collect()
        
        if fit_success:
            result = {
                'bootstrap_id': bootstrap_id,
                'n_pairs': diagnostics['n_pairs'],
                'n_bins': diagnostics['n_bins'],
                'lambda_km': diagnostics['lambda_km'],
                'amplitude': diagnostics['amplitude'],
                'offset': diagnostics['offset'],
                'r_squared': diagnostics['r_squared']
            }
            return bootstrap_id, result, True
        else:
            return bootstrap_id, None, False
            
    except Exception as e:
        # Ensure cleanup even on error
        if 'bootstrap_sample' in locals():
            del bootstrap_sample
        if 'complete_df' in locals():
            del complete_df
        gc.collect()
        print_status(f"Error in bootstrap sample {bootstrap_id}: {e}", "ERROR")
        return bootstrap_id, None, False

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
    n_bootstrap_samples = TEPConfig.get_int('TEP_STATION_BOOTSTRAP_SAMPLES', 50)
    n_workers = TEPConfig.get_worker_count('TEP_WORKERS')
    
    print_status(f"Configuration: TEP_STATION_BOOTSTRAP_SAMPLES = {TEPConfig.get_int('TEP_STATION_BOOTSTRAP_SAMPLES', 50)}", "DEBUG")
    print_status(f"Running {n_bootstrap_samples} station bootstrap samples from {len(unique_stations)} unique stations", "INFO")
    print_status(f"Using {n_workers} parallel workers for bootstrap processing", "INFO")
    
    # Pre-compute station pairs for efficiency
    station_pairs = precompute_station_pairs(complete_df)
    
    # Memory cleanup after pre-computation
    cleanup_memory(force_gc=True, log_usage=True)
    
    # Use sequential processing (multiprocessing has issues with large DataFrames)
    min_pairs = 1000
    bootstrap_results = []
    lambda_estimates = []
    successful_samples = 0
    
    # Monitor initial memory usage
    monitor_memory_usage("Station Bootstrap Start")
    
    print_status(f"Processing {n_bootstrap_samples} bootstrap samples sequentially (multiprocessing disabled for large datasets)", "INFO")
    
    # Run a diagnostic check with the first few samples to catch issues early
    diagnostic_samples = min(5, n_bootstrap_samples)
    print_status(f"Running diagnostic check with first {diagnostic_samples} samples...", "INFO")
    
    diagnostic_errors = []
    for i in range(diagnostic_samples):
        try:
            bootstrap_sample = create_station_bootstrap_sample_optimized(station_pairs, unique_stations, i, complete_df)
            if len(bootstrap_sample) >= min_pairs:
                fitted_params, fit_success, diagnostics = fit_correlation_model_bootstrap(bootstrap_sample)
                if fit_success:
                    print_status(f"Diagnostic sample {i}: SUCCESS - λ={diagnostics['lambda_km']:.1f} km, R²={diagnostics['r_squared']:.3f}", "INFO")
                else:
                    error_msg = diagnostics.get('error', 'unknown_error')
                    diagnostic_errors.append(f"Sample {i}: {error_msg}")
                    print_status(f"Diagnostic sample {i}: FAILED - {error_msg}", "WARNING")
            else:
                diagnostic_errors.append(f"Sample {i}: insufficient_pairs ({len(bootstrap_sample)})")
            del bootstrap_sample
        except Exception as e:
            diagnostic_errors.append(f"Sample {i}: exception - {e}")
            print_status(f"Diagnostic sample {i}: ERROR - {e}", "ERROR")
    
    if len(diagnostic_errors) == diagnostic_samples:
        print_status("All diagnostic samples failed! Check data and configuration.", "ERROR")
        print_status(f"Diagnostic errors: {diagnostic_errors[:3]}", "ERROR")  # Show first 3 errors
    else:
        print_status(f"Diagnostic complete: {diagnostic_samples - len(diagnostic_errors)}/{diagnostic_samples} samples successful", "SUCCESS")
    
    # Sequential processing with regular progress updates and memory cleanup
    for i in range(n_bootstrap_samples):
        if (i + 1) % 10 == 0:  # More frequent progress updates
            progress_pct = (i + 1) / n_bootstrap_samples * 100
            print_status(f"Station bootstrap progress: {i+1}/{n_bootstrap_samples} ({progress_pct:.1f}%), {successful_samples} successful", "PROCESS")
        
        try:
            # Create bootstrap sample using optimized method
            bootstrap_sample = create_station_bootstrap_sample_optimized(station_pairs, unique_stations, i, complete_df)
            
            if len(bootstrap_sample) < min_pairs:  # Skip samples that are too small
                if (i + 1) % 50 == 0:  # Only log occasionally to avoid spam
                    print_status(f"Bootstrap sample {i}: too few pairs ({len(bootstrap_sample)} < {min_pairs})", "DEBUG")
                del bootstrap_sample
                continue
            
            # Fit correlation model
            fitted_params, fit_success, diagnostics = fit_correlation_model_bootstrap(bootstrap_sample)
            
            # Clean up bootstrap sample immediately
            del bootstrap_sample
            
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
                successful_samples += 1
            else:
                # Log fitting errors for debugging (but not too frequently)
                if (i + 1) % 50 == 0 or successful_samples < 5:
                    error_msg = diagnostics.get('error', 'unknown_error')
                    print_status(f"Bootstrap sample {i}: fit failed - {error_msg}", "DEBUG")
            
            # More frequent memory cleanup for large datasets
            if (i + 1) % 5 == 0:
                cleanup_memory(force_gc=True, log_usage=False)
                
        except Exception as e:
            print_status(f"Error in bootstrap sample {i}: {e}", "WARNING")
            # Ensure cleanup on error
            if 'bootstrap_sample' in locals():
                del bootstrap_sample
            continue
    
    # Final memory cleanup
    cleanup_memory(force_gc=True, log_usage=True)
    
    # Clean up pre-computed station pairs
    del station_pairs
    cleanup_memory(force_gc=True, log_usage=True)
    
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
    
    # Monitor memory before day bootstrap
    monitor_memory_usage("Day Bootstrap Start")
    
    # Get unique days
    unique_days = complete_df['date'].unique()
    n_bootstrap_samples = TEPConfig.get_int('TEP_DAY_BOOTSTRAP_SAMPLES', 300)
    min_days = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_DAYS', 100)
    
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

        unique_days_in_sample = bootstrap_sample['date'].nunique()
        if unique_days_in_sample < min_days:
            print_status(
                f"    Skipping bootstrap sample {i}: only {unique_days_in_sample} unique days (minimum {min_days})",
                "WARNING"
            )
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
    
    # Memory cleanup after day bootstrap
    cleanup_memory(force_gc=True, log_usage=True)
    
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
    
    # Monitor memory before hybrid bootstrap
    monitor_memory_usage("Hybrid Bootstrap Start")
    
    # Get unique stations and days
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    unique_days = complete_df['date'].unique()
    n_bootstrap_samples = TEPConfig.get_int('TEP_HYBRID_BOOTSTRAP_SAMPLES', 200)
    min_days = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_DAYS', 100)
    min_stations = TEPConfig.get_int('TEP_BOOTSTRAP_MIN_STATIONS', 100)
    
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

        unique_days_in_sample = bootstrap_sample['date'].nunique()
        unique_stations_in_sample = pd.unique(
            bootstrap_sample[['station_i', 'station_j']].values.ravel()
        )

        if unique_days_in_sample < min_days:
            print_status(
                f"    Skipping hybrid sample {i}: only {unique_days_in_sample} unique days (minimum {min_days})",
                "WARNING"
            )
            continue

        if len(unique_stations_in_sample) < min_stations:
            print_status(
                f"    Skipping hybrid sample {i}: only {len(unique_stations_in_sample)} stations (minimum {min_stations})",
                "WARNING"
            )
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
    
    # Memory cleanup after hybrid bootstrap
    cleanup_memory(force_gc=True, log_usage=True)
    
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
        # Monitor memory before loading
        monitor_memory_usage(f"Before loading {ac} dataset")
        
        # Load complete dataset with chunking for large datasets
        complete_df = load_complete_pair_dataset_chunked(ac)
        
        # Monitor memory after loading
        monitor_memory_usage(f"After loading {ac} dataset")
        
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
        
        # Memory cleanup between bootstrap methods
        cleanup_memory(force_gc=True, log_usage=True)
        monitor_memory_usage("After Station Bootstrap")
        
        # Run day block bootstrap  
        print_status("=" * 60, "INFO")
        day_results = run_day_block_bootstrap(complete_df)
        results['day_block_bootstrap'] = day_results
        
        # Memory cleanup between bootstrap methods
        cleanup_memory(force_gc=True, log_usage=True)
        monitor_memory_usage("After Day Bootstrap")
        
        # Run hybrid block bootstrap
        print_status("=" * 60, "INFO")
        hybrid_results = run_hybrid_block_bootstrap(complete_df)
        results['hybrid_block_bootstrap'] = hybrid_results
        
        # Final memory cleanup
        cleanup_memory(force_gc=True, log_usage=True)
        
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
        raise TEPAnalysisError(f"Robust block bootstrap analysis failed for {ac}: {e}") # Re-raise a specific exception

@ensure_single_instance
def main():
    """Main execution function."""
    start_time = time.time()
    
    print_status("=" * 80, "INFO")
    print_status("TEP GNSS Analysis Package v0.14 - STEP 3.1: Robust Block Bootstrap", "TITLE")
    print_status("=" * 80, "INFO")
    
    # Configuration summary
    print_status("Configuration:", "INFO")
    print_status(f"  Station bootstrap samples: {TEPConfig.get_int('TEP_STATION_BOOTSTRAP_SAMPLES', 50)}", "INFO")
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
    for i, ac in enumerate(analysis_centers):
        print_status("", "INFO")
        print_status(f"Processing analysis center: {ac.upper()}", "PROCESS")
        print_status("-" * 50, "INFO")
        
        # Monitor memory before processing each center
        monitor_memory_usage(f"Before processing {ac}")
        
        # Run bootstrap analysis
        results = run_robust_block_bootstrap_analysis(ac)
        
        # Save results
        output_file = PACKAGE_ROOT / "results/outputs" / f"step_3_1_robust_block_bootstrap_{ac}.json" # Updated from step_5_6_robust_block_bootstrap
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            safe_json_write(results, output_file)
            print_status(f"Results saved to: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results for {ac}: {e}", "ERROR")
        
        # Aggressive memory cleanup between analysis centers
        cleanup_memory(force_gc=True, log_usage=True)
        
        # Log memory usage after each center
        rss_mb, vms_mb = get_memory_usage()
        print_status(f"Memory usage after {ac}: RSS={rss_mb:.2f} MB, VMS={vms_mb:.2f} MB", "DEBUG")
        
        # Additional cleanup for large datasets (especially CODE)
        if ac == 'code':
            print_status("Performing additional memory cleanup for CODE dataset...", "INFO")
            cleanup_memory(force_gc=True, log_usage=True)
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_status("=" * 80, "INFO")
    print_status(f"Robust block bootstrap analysis completed in {elapsed_time:.1f} seconds", "SUCCESS")
    print_status("All bootstrap validation results saved to results/outputs/", "INFO")
    print_status("=" * 80, "INFO")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0)  # Always exit successfully to continue pipeline
    except KeyboardInterrupt:
        print_status("Analysis interrupted by user", "WARNING")
        sys.exit(0)  # Don't stop pipeline
    except Exception as e:
        print_status(f"Error: {e}", "ERROR")
        import traceback
        print_status(traceback.format_exc(), "DEBUG")
        sys.exit(0)  # Don't stop pipeline
