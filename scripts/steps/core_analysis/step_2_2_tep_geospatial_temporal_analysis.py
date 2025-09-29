#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 2.2: Geospatial Temporal Analysis
========================================================

Performs comprehensive geospatial and temporal analysis including astronomical
event correlations, orbital tracking, anisotropy analysis, and advanced temporal field studies.

Requirements: Step 2.1 complete (Geospatial Data Processing)
Next: Step 3.0 (Cross-Validation Suite)

Key Analyses:
1. Enhanced Anisotropy Analysis - detailed directional and temporal propagation tests
2. Temporal Orbital Tracking - correlation patterns with Earth orbital motion
3. Helical Motion Analysis - Chandler wobble, 3D spherical harmonics, beat frequencies
4. Planetary Opposition Analysis - gravitational potential coupling (Jupiter, Saturn, Mars)
5. Lunar Standstill Analysis - sidereal day amplitude modulation

CRITICAL: This step loads the COMPLETE pair-level dataset (~5-6 GB) into memory
for maximum statistical rigor as requested by reviewers.

Inputs:
  - data/processed/step_2_1_geospatial_{ac}.csv (from Step 2.1)
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0)

Outputs:
  - results/outputs/step_2_2_geospatial_temporal_analysis_{ac}.json
  - results/outputs/step_2_2_enhanced_anisotropy_{ac}.json
  - results/outputs/step_2_2_helical_motion_only_{ac}.json
  - results/outputs/step_2_2_jupiter_only_{ac}.json
  - results/outputs/step_2_2_saturn_only_{ac}.json
  - results/outputs/step_2_2_mars_only_{ac}.json
  - results/outputs/step_2_2_lunar_only_{ac}.json
  - results/outputs/step_2_2_astronomical_events_{ac}.json

Environment Variables:
  - TEP_ENABLE_ENHANCED_ANISOTROPY: Enable enhanced anisotropy tests (default: 1)
  - TEP_ENABLE_TEMPORAL_ORBITAL_TRACKING: Enable temporal orbital tracking (default: 1)
  - TEP_ENABLE_CHANDLER_WOBBLE: Enable Chandler wobble analysis (default: 0)
  - TEP_ENABLE_3D_HARMONICS: Enable 3D spherical harmonic analysis (default: 0)
  - TEP_ENABLE_BEAT_FREQUENCIES: Enable multi-frequency beat analysis (default: 0)
  - TEP_ENABLE_RELATIVE_MOTION_BEATS: Enable relative motion beat analysis (default: 0)
  - TEP_ENABLE_MESH_DANCE_ANALYSIS: Enable mesh dance analysis (default: 0)
  - TEP_ENABLE_JUPITER_OPPOSITION: Enable Jupiter opposition analysis (default: 0)
  - TEP_ENABLE_SATURN_OPPOSITION: Enable Saturn opposition analysis (default: 0)
  - TEP_ENABLE_MARS_OPPOSITION: Enable Mars opposition analysis (default: 0)
  - TEP_ENABLE_LUNAR_STANDSTILL: Enable lunar standstill analysis (default: 0)
  - TEP_ENABLE_NUTATION_ANALYSIS: Enable nutation analysis (default: 0)
  - TEP_MEMORY_LIMIT_GB: Maximum memory to use in GB (default: 8.0)

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from glob import glob
import psutil  # For memory monitoring
from functools import lru_cache, partial
import warnings

# Suppress scipy optimization warnings
warnings.filterwarnings('ignore', 'Covariance of the parameters could not be estimated')
warnings.filterwarnings('ignore', 'An input array is constant')
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

# Anchor to package root
ROOT = Path(__file__).resolve().parents[2]

# Import TEP utilities for better configuration and error handling
sys.path.insert(0, str(ROOT))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    TEPAnalysisError, safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)
from scripts.utils.geospatial import compute_azimuth, classify_ew_ns

def print_status(message, level="INFO"):
    """Enhanced status printing with timestamp and color coding."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Color coding for different levels
    colors = {
        "TITLE": "\033[1;36m",    # Cyan bold
        "SUCCESS": "\033[1;32m",  # Green bold
        "WARNING": "\033[1;33m",  # Yellow bold
        "ERROR": "\033[1;31m",    # Red bold
        "INFO": "\033[0;37m",     # White
        "DEBUG": "\033[0;90m",    # Dark gray
        "PROCESS": "\033[0;34m"   # Blue
    }
    reset = "\033[0m"

    color = colors.get(level, colors["INFO"])

    if level == "TITLE":
        print(f"\n{color}{'='*80}")
        print(f"[{timestamp}] {message}")
        print(f"{'='*80}{reset}\n")
    else:
        print(f"{color}[{timestamp}] [{level}] {message}{reset}")

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

def performance_monitor(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**3)
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        print_status(f"Performance: {func.__name__} took {execution_time:.2f}s, memory Δ: {memory_delta:+.2f} GB", "PERFORMANCE")
        
        return result
    return wrapper

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def correlation_model_vectorized(r_array, amplitude, lambda_km, offset):
    """Vectorized version of correlation model for array inputs"""
    return amplitude * np.exp(-r_array / lambda_km) + offset

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
    
    # Check file size for progress estimation
    file_size_mb = geospatial_file.stat().st_size / (1024 * 1024)
    print_status(f"File size: {file_size_mb:.1f} MB", "DEBUG")
    
    try:
        # Load the complete geospatial dataset with progress monitoring
        print_status("Reading CSV file into memory...", "PROCESS")
        complete_df = pd.read_csv(geospatial_file, parse_dates=['date'])
        print_status(f"CSV loaded successfully: {len(complete_df):,} rows", "SUCCESS")
        
        # Add coherence column (preserving sign like Step 3)
        print_status("Computing coherence values from plateau phase...", "PROCESS")
        complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
        
        # Clean data
        print_status("Cleaning and filtering data...", "PROCESS")
        initial_count = len(complete_df)
        complete_df.dropna(subset=['dist_km', 'coherence', 'station_i', 'station_j', 'date'], inplace=True)
        after_dropna = len(complete_df)
        complete_df = complete_df[complete_df['dist_km'] > 0]
        final_count = len(complete_df)
        
        print_status(f"Data filtering: {initial_count:,} → {after_dropna:,} → {final_count:,} pairs", "DEBUG")
        
        print_status(f"Geospatial dataset loaded: {len(complete_df):,} pairs, {complete_df.memory_usage(deep=True).sum()/(1024**3):.2f} GB", "SUCCESS")
        print_status("Azimuth already computed in Step 4 - no redundant calculation needed", "SUCCESS")
        
        # Verify required columns are present
        print_status("Verifying required columns are present...", "PROCESS")
        required_cols = ['azimuth', 'delta_longitude', 'delta_local_time']
        missing_cols = [col for col in required_cols if col not in complete_df.columns]
        
        if missing_cols:
            raise TEPDataError(f"Missing required columns from Step 4: {missing_cols}")
        
        print_status(f"All required columns present: {required_cols}", "SUCCESS")
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
    """Load dataset using in-memory processing with optimized batch loading"""
    df_chunks = []
    total_pairs = 0
    
    # OPTIMIZATION: Process files in batches for better I/O performance
    batch_size = TEPConfig.get_int('TEP_LOAD_BATCH_SIZE')
    if len(pair_files) < batch_size: # Handle case where there are fewer files than the batch size
        batch_size = len(pair_files)
    
    for batch_start in range(0, len(pair_files), batch_size):
        batch_end = min(batch_start + batch_size, len(pair_files))
        batch_files = pair_files[batch_start:batch_end]
        
        print_status(f"Loading batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}/{len(pair_files)}", "PROCESS")
        check_memory_usage()
        
        # Load batch of files
        batch_chunks = []
        for pfile in batch_files:
            def _load_file():
                return safe_csv_read(pfile)
            
            df_chunk = SafeErrorHandler.safe_file_operation(
                _load_file,
                error_message=f"Failed to load {pfile.name}",
                logger_func=print_status,
                return_on_error=None
            )
            
            if df_chunk is not None and len(df_chunk) > 0:
                batch_chunks.append(df_chunk)
                total_pairs += len(df_chunk)
        
        # Concatenate batch and add to main chunks
        if batch_chunks:
            batch_df = pd.concat(batch_chunks, ignore_index=True)
            df_chunks.append(batch_df)
            del batch_chunks  # Free memory immediately
            gc.collect()
    
    if not df_chunks:
        raise TEPDataError(f"No valid data loaded for {ac}")
    
    print_status(f"Concatenating {len(df_chunks)} chunks with {total_pairs:,} total pairs...", "PROCESS")
    
    # Concatenate all chunks
    complete_df = pd.concat(df_chunks, ignore_index=True)
    del df_chunks  # Free intermediate memory
    gc.collect()
    
    # Add coherence column and clean data with vectorized operations
    # Calculate proper phase coherence (preserving sign like Step 3)
    complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
    # Vectorized filtering for better performance
    valid_mask = (
        complete_df['dist_km'].notna() & 
        complete_df['station_i'].notna() & 
        complete_df['station_j'].notna() & 
        complete_df['date'].notna() & 
        (complete_df['dist_km'] > 0)
    )
    complete_df = complete_df[valid_mask]
    
    print_status(f"Dataset loaded: {len(complete_df):,} pairs, {complete_df.memory_usage(deep=True).sum()/(1024**3):.2f} GB", "SUCCESS")
    check_memory_usage()
    
    return complete_df

def _load_dataset_chunked(pair_files: List[Path], ac: str) -> pd.DataFrame:
    """Load dataset using chunked processing for memory-constrained environments"""
    print_status("Using chunked processing to manage memory usage", "INFO")
    
    # Optimized chunk size based on available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    min_chunk_size = TEPConfig.get_int('TEP_MIN_CHUNK_SIZE')
    max_chunk_size = TEPConfig.get_int('TEP_MAX_CHUNK_SIZE')
    chunk_size = min(max_chunk_size, max(min_chunk_size, int(available_gb * 10000)))  # Adaptive chunk size
    processed_chunks = []
    total_pairs = 0
    
    for i, pfile in enumerate(pair_files):
        if i % TEPConfig.get_int('TEP_LOGGING_INTERVAL_FILES') == 0:  # Log progress for debugging
            print_status(f"Processing file {i+1}/{len(pair_files)}: {pfile.name}", "PROCESS")
            if i > 0:
                check_memory_usage()
        
        try:
            # Read file in chunks to manage memory
            for chunk_df in pd.read_csv(pfile, chunksize=chunk_size, parse_dates=['date']):
                if len(chunk_df) == 0:
                    continue
                
                # Process chunk immediately with vectorized operations
                chunk_df['coherence'] = np.cos(chunk_df['plateau_phase'])
                # Vectorized filtering for better performance
                valid_mask = (
                    chunk_df['dist_km'].notna() & 
                    chunk_df['station_i'].notna() & 
                    chunk_df['station_j'].notna() & 
                    chunk_df['date'].notna() & 
                    (chunk_df['dist_km'] > 0)
                )
                chunk_df = chunk_df[valid_mask]                
                if len(chunk_df) > 0:
                    processed_chunks.append(chunk_df)
                    total_pairs += len(chunk_df)
                
                # Memory management: consolidate chunks if too many
                if len(processed_chunks) > TEPConfig.get_int('TEP_CHUNK_CONSOLIDATION_THRESHOLD'):
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

def run_enhanced_anisotropy_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform enhanced anisotropy analysis on the complete dataset.
    This analysis investigates whether the TEP correlation (lambda) exhibits
    directional dependence, which could indicate unmodeled systematic effects
    or underlying geophysical processes.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        Dict: A dictionary containing the results of the anisotropy analysis,
              including directional lambda estimates and statistical tests.
    """
    print_status("Starting enhanced anisotropy analysis...", "PROCESS")
    
    # Check if we have coordinate information
    required_cols = ['station1_lat', 'station1_lon', 'station2_lat', 'station2_lon']
    has_coords = all(col in complete_df.columns for col in required_cols)
    
    if not has_coords:
        return {'success': False, 'error': 'Coordinate columns not found in dataset'}
    
    # Filter to pairs with valid coordinates
    coord_df = complete_df.dropna(subset=required_cols)    
    if len(coord_df) < 1000:
        return {'success': False, 'error': f'Insufficient pairs with coordinates: {len(coord_df)}'}
    
    print_status(f"Analyzing {len(coord_df):,} pairs with coordinate information", "INFO")
    
    # Check if azimuths are already computed (from Step 4)
    if 'azimuth' in coord_df.columns and coord_df['azimuth'].notna().all():
        print_status("Using pre-computed azimuths from Step 4", "SUCCESS")
    else:
        # Compute azimuths for all pairs (fallback for Step 3 data)
        print_status("Computing azimuths for all station pairs...", "PROCESS")
        coord_df['azimuth'] = coord_df.apply(
            lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                       row['station2_lat'], row['station2_lon']), axis=1
        )
        print_status("Azimuth computation completed", "SUCCESS")
    
    # Group into 8 directional sectors (45° each)
    print_status("Classifying pairs into directional sectors...", "PROCESS")
    sector_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    coord_df['sector'] = coord_df['azimuth'].apply(lambda az: sector_names[int((az + 22.5) / 45) % 8])
    
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    print_status(f"Analyzing {len(sector_names)} directional sectors with {num_bins} distance bins", "INFO")
    
    # Analyze each sector
    sector_results = {}
    
    for i, sector in enumerate(sector_names):
        sector_mask = coord_df['sector'] == sector
        sector_data = coord_df[sector_mask]
        print_status(f"Processing sector {i+1}/{len(sector_names)}: {sector} ({len(sector_data):,} pairs)", "PROCESS")
        
        if len(sector_data) < 1000:  # Need sufficient data
            print_status(f"Skipping sector {sector}: insufficient data ({len(sector_data)} pairs)", "WARNING")
            continue
        
        # Bin the sector data (create bins directly without modifying original data)
        print_status(f"  Binning {sector} sector data into {num_bins} distance bins...", "DEBUG")
        dist_bins = pd.cut(sector_data['dist_km'], bins=edges, right=False)
        
        # Group by bins directly without modifying the original DataFrame
        binned = sector_data.groupby(dist_bins, observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        binned.rename(columns={'dist_km': 'dist_bin'}, inplace=True)
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        print_status(f"  {sector}: {len(binned)} valid bins (min {min_bin_count} pairs per bin)", "DEBUG")
        
        if len(binned) < 5:  # Need enough bins for fitting
            print_status(f"  Skipping {sector}: insufficient bins for fitting ({len(binned)} < 5)", "WARNING")
            continue
        
        # Fit exponential model to this sector
        print_status(f"  Fitting exponential correlation model to {sector} sector...", "DEBUG")
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
            print_status(f"  {sector} fit successful: λ = {popt[1]:.1f} km, R² = {r_squared:.3f}", "SUCCESS")
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
            print_status(f"  {sector} fit failed: {str(e)[:50]}...", "WARNING")
            continue  # Skip failed fits - common in statistical resampling
    
    if len(sector_results) < 4:  # Need reasonable directional coverage
        return {'success': False, 'error': f'Only {len(sector_results)} sectors with successful fits'}
    
    print_status(f"Computing anisotropy statistics from {len(sector_results)} successful sector fits...", "PROCESS")
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
    
    # Check if azimuths are already computed (from Step 4)
    if 'azimuth' in complete_df.columns and complete_df['azimuth'].notna().all():
        print_status("Using pre-computed azimuths from Step 4", "SUCCESS")
    else:
        # Compute azimuths for all pairs (fallback for Step 3 data)
        print_status("Computing azimuths for all station pairs...", "PROCESS")
        complete_df['azimuth'] = complete_df.apply(
            lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                       row['station2_lat'], row['station2_lon']), axis=1
        )
        print_status("Azimuth computation completed", "SUCCESS")
    
    # Group into East-West vs North-South for temporal tracking
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
        ]        
        if len(day_data) < 1000:  # Need sufficient data
            continue
        
        # Analyze E-W and N-S separately for this day
        ew_data = day_data[day_data['ew_ns_class'] == 'EW']
        ns_data = day_data[day_data['ew_ns_class'] == 'NS']
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
    except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
        print_status(f"Seasonal fit failed: {e}", "WARNING")
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
        # Create a working copy to avoid SettingWithCopyWarning
        df_work = directional_df.copy()
        
        # Bin the data
        df_work['dist_bin'] = pd.cut(df_work['dist_km'], bins=edges, right=False)
        binned = df_work.groupby('dist_bin', observed=True).agg(
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
        
    except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
        print_status(f"Directional correlation fit failed: {e}", "WARNING")
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
    Process geospatial temporal analysis for one analysis center.
    
    Args:
        ac: Analysis center name ('code', 'igs_combined', 'esa_final')
    
    Returns:
        dict: Geospatial temporal analysis results
    """
    print_status(f"Starting geospatial temporal analysis for {ac.upper()}", "INFO")
    start_time = time.time()
    
    try:
        # Load complete dataset into memory (Step 4 geospatial data with pre-computed azimuth)
        complete_df = load_complete_geospatial_dataset(ac)
        
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
    Orchestrates the helical motion analysis suite. This function runs only the
    analyses related to Earth's helical motion, which include:
    - Chandler Wobble analysis: Detects 14-month polar motion signatures.
    - 3D Spherical Harmonic analysis: Decomposes directional anisotropy patterns.
    - Multi-Frequency Beat analysis: Identifies temporal interference patterns.
    - Relative Motion Beat analysis: Examines differential dynamics between station pairs.
    - Mesh Dance analysis: Assesses network-wide coherent motion patterns.
    - Nutation analysis: Detects 18.6-year axial tilt variations (if enabled).

    This function is designed for targeted testing and validation of the helical
    motion detection capabilities.

    Args:
        analysis_center (str, optional): The specific analysis center to process
                                         ('code', 'igs_combined', 'esa_final').
                                         If None, runs all configured centers.

    Returns:
        Dict: A dictionary containing the results from all executed helical motion
              analyses, organized by analysis center. Each entry includes a
              'success' status and potentially an 'error' message if an analysis failed.
    """
    print_status("TEP GNSS Analysis Package v0.13", "TITLE")
    print_status("HELICAL MOTION ANALYSIS - Advanced Earth Motion Detection", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in centers:
        print_status(f"\n{'='*60}")
        print_status(f"PROCESSING {ac.upper()} - HELICAL MOTION ANALYSIS", "TITLE")
        print_status(f"{'='*60}", "TITLE")
        
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
    print_status("HELICAL MOTION ANALYSIS COMPLETE", "TITLE")
    print_status(f"Total execution time: {total_time:.1f} seconds", "INFO")
    
    return all_results

def run_jupiter_only(analysis_center: str = None) -> Dict:
    """
    Orchestrates the Jupiter opposition analysis. This function runs only the
    analysis related to Jupiter opposition events, looking for gravitational
    potential coupling.

    Args:
        analysis_center (str, optional): The specific analysis center to process
                                         ('code', 'igs_combined', 'esa_final').
                                         If None, runs all configured centers.

    Returns:
        Dict: A dictionary containing the results from the Jupiter opposition
              analysis, organized by analysis center. Each entry includes a
              'success' status and potentially an 'error' message if the analysis failed.
    """
    print_status("TEP GNSS Analysis Package v0.13", "TITLE")
    print_status("JUPITER OPPOSITION ANALYSIS - Gravitational Potential Pulse Detection", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
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
    print("JUPITER OPPOSITION ANALYSIS COMPLETE")
    print_status(f"Total execution time: {total_time:.1f} seconds", "INFO")
    
    return all_results

def run_saturn_only(analysis_center: str = None) -> Dict:
    """
    Orchestrates the Saturn opposition analysis. This function runs only the
    analysis related to Saturn opposition events, looking for gravitational
    potential coupling. Saturn's signal is expected to be smaller than Jupiter's,
    making it an important validation test.

    Args:
        analysis_center (str, optional): The specific analysis center to process
                                         ('code', 'igs_combined', 'esa_final').
                                         If None, runs all configured centers.

    Returns:
        Dict: A dictionary containing the results from the Saturn opposition
              analysis, organized by analysis center. Each entry includes a
              'success' status and potentially an 'error' message if the analysis failed.
    """
    print_status("TEP GNSS Analysis Package v0.13", "TITLE")
    print_status("SATURN OPPOSITION ANALYSIS - Gravitational Potential Pulse Detection", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
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
    print("🪐 SATURN OPPOSITION ANALYSIS COMPLETED")
    print_status(f"Total execution time: {elapsed_time:.1f} seconds", "INFO")
    
    return all_results

def run_mars_only(analysis_center: str = None) -> Dict:
    """
    Orchestrates the Mars opposition analysis. This function runs only the
    analysis related to Mars opposition events, looking for gravitational
    potential coupling. Mars has the weakest expected signal, making it an
    excellent test of the detection sensitivity.

    Args:
        analysis_center (str, optional): The specific analysis center to process
                                         ('code', 'igs_combined', 'esa_final').
                                         If None, runs all configured centers.

    Returns:
        Dict: A dictionary containing the results from the Mars opposition
              analysis, organized by analysis center. Each entry includes a
              'success' status and potentially an 'error' message if the analysis failed.
    """
    print_status("TEP GNSS Analysis Package v0.13", "TITLE")
    print_status("MARS OPPOSITION ANALYSIS - Weakest Signal Sensitivity Test", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
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
    print("🔴 MARS OPPOSITION ANALYSIS COMPLETED")
    print_status(f"Total execution time: {elapsed_time:.1f} seconds", "INFO")
    
    return all_results

def run_lunar_only(analysis_center: str = None) -> Dict:
    """
    Orchestrates the Major Lunar Standstill analysis. This function runs only the
    analysis related to Lunar Standstill events, tracking sidereal day amplitude
    enhancement.

    Args:
        analysis_center (str, optional): The specific analysis center to process
                                         ('code', 'igs_combined', 'esa_final').
                                         If None, runs all configured centers.

    Returns:
        Dict: A dictionary containing the results from the Lunar Standstill
              analysis, organized by analysis center. Each entry includes a
              'success' status and potentially an 'error' message if the analysis failed.
    """
    print_status("TEP GNSS Analysis Package v0.13", "TITLE")
    print_status("LUNAR STANDSTILL ANALYSIS - Sidereal Day Amplitude Tracking", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
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
    print("🌙 LUNAR STANDSTILL ANALYSIS COMPLETED")
    print_status(f"Total execution time: {elapsed_time:.1f} seconds", "INFO")
    
    return all_results

def run_astronomical_events_only(analysis_center: str = None) -> Dict:
    """
    Orchestrates a comparative analysis of Jupiter, Saturn, and Mars opposition events.
    This function runs all three planetary opposition analyses and then provides
    a consolidated comparison of their results.

    Args:
        analysis_center (str, optional): The specific analysis center to process
                                         ('code', 'igs_combined', 'esa_final').
                                         If None, runs all configured centers.

    Returns:
        Dict: A dictionary containing the comparative results from the astronomical
              event analyses, organized by analysis center. Each entry includes the
              results from Jupiter, Saturn, and Mars analyses, along with an overall
              comparison summary.
    """
    print_status("TEP GNSS Analysis Package v0.13", "TITLE")
    print_status("ASTRONOMICAL EVENTS ANALYSIS - Jupiter vs Saturn vs Mars Opposition Comparison", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
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
    print("🌌 ASTRONOMICAL EVENTS ANALYSIS COMPLETED")
    print_status(f"Total execution time: {elapsed_time:.1f} seconds", "INFO")
    
    return all_results

def print_summary_jupiter_results(results: Dict):
    """Print a summary of Jupiter opposition analysis results"""
    print_status(f"JUPITER OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_JUPITER_OPPOSITION'):
            # Check for ANY significant individual detections first
            event_results = results.get('event_results', {})
            significant_events = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    gaussian = event_data.get('gaussian_fit', {})
                    if gaussian.get('is_significant', False):
                        significant_events.append((event_name, event_data))
            
            # Report significant individual events prominently
            if significant_events:
                print_status(f"🪐 Jupiter Opposition: ⭐ {len(significant_events)} SIGNIFICANT DETECTION(S) ⭐", "INFO")
                for event_name, event_data in significant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    gaussian = event_data.get('gaussian_fit', {})
                    amplitude = gaussian.get('amplitude', 0)
                    std_err = gaussian.get('amplitude_std_err', 1)
                    sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                    center_days = gaussian.get('center_days', 0)
                    direction = "suppression" if amplitude < 0 else "enhancement"
                    amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                    
                    print_status(f"   🎯 {event_date}: {sigma_level:.1f}σ {direction} at day {center_days:.1f}", "INFO")
                    print_status(f"      Amplitude: {amplitude_pct:.1f}% of baseline", "INFO")
            else:
                print_status(f"🪐 Jupiter Opposition: No significant individual detections", "INFO")

            if results.get('stacked_analysis', {}).get('success', False):
                stacked_gaussian = results['stacked_analysis']['gaussian_fit']
                if stacked_gaussian.get('is_significant', False):
                    stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1))
                    print_status(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ significant", "INFO")
                else:
                    stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1)) if stacked_gaussian.get('amplitude_std_err', 0) > 0 else 0
                    print_status(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ (not significant)", "INFO")
            else:
                print_status(f"   📊 Stacked Analysis: Failed or not run", "WARNING")

            # Show all individual event details
            if results.get('individual_event_fits', []):
                print_status(f"   Individual Events:", "INFO")
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
                            print_status(f"     {event_date}: {sigma_level:.1f}σ ({'✓' if significant else '✗'}) peak at day {center_days:.1f}", "INFO")
            else:
                print_status(f"   Individual Events: No significant individual detections", "INFO")

        else:
            print_status("🪐 Jupiter Opposition: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"🪐 Jupiter Opposition: ✗ Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def print_summary_saturn_results(results: Dict):
    """Print a summary of Saturn opposition analysis results"""
    print_status(f"SATURN OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_SATURN_OPPOSITION'):
            # Check for ANY significant individual detections first
            event_results = results.get('event_results', {})
            significant_events = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    gaussian = event_data.get('gaussian_fit', {})
                    if gaussian.get('is_significant', False):
                        significant_events.append((event_name, event_data))
            
            # Report significant individual events prominently
            if significant_events:
                print_status(f"🪐 Saturn Opposition: ⭐ {len(significant_events)} SIGNIFICANT DETECTION(S) ⭐", "INFO")
                for event_name, event_data in significant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    gaussian = event_data.get('gaussian_fit', {})
                    amplitude = gaussian.get('amplitude', 0)
                    std_err = gaussian.get('amplitude_std_err', 1)
                    sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                    center_days = gaussian.get('center_days', 0)
                    direction = "suppression" if amplitude < 0 else "enhancement"
                    amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                    
                    print_status(f"   🎯 {event_date}: {sigma_level:.1f}σ {direction} at day {center_days:.1f}", "INFO")
                    print_status(f"      Amplitude: {amplitude_pct:.1f}% of baseline", "INFO")
            else:
                print_status(f"🪐 Saturn Opposition: No significant individual detections", "INFO")

            if results.get('stacked_analysis', {}).get('success', False):
                stacked_gaussian = results['stacked_analysis']['gaussian_fit']
                if stacked_gaussian.get('is_significant', False):
                    stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1))
                    print_status(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ significant", "INFO")
                else:
                    stacked_sigma = abs(stacked_gaussian.get('amplitude', 0) / stacked_gaussian.get('amplitude_std_err', 1)) if stacked_gaussian.get('amplitude_std_err', 0) > 0 else 0
                    print_status(f"   📊 Stacked Analysis: {stacked_sigma:.1f}σ (not significant)", "INFO")
            else:
                print_status(f"   📊 Stacked Analysis: Failed or not run", "WARNING")

            if results.get('individual_event_fits', []):
                print_status(f"   Individual Events:", "INFO")
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
                            print_status(f"     {event_date}: {sigma_level:.1f}σ ({'✓' if significant else '✗'}) peak at day {center_days:.1f}", "INFO")
            else:
                print_status(f"   Individual Events: No significant individual detections", "INFO")
        else:
            print_status("🪐 Saturn Opposition: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"🪐 Saturn Opposition: ✗ Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def print_summary_mars_results(results: Dict):
    """Print a summary of Mars opposition analysis results"""
    print_status(f"MARS OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_MARS_OPPOSITION'):
            # Check for ANY significant individual detections first
            event_results = results.get('event_results', {})
            significant_events = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    gaussian = event_data.get('gaussian_fit', {})
                    if gaussian.get('is_significant', False):
                        significant_events.append((event_name, event_data))
            
            # Report significant individual events prominently
            if significant_events:
                print_status(f"🔴 Mars Opposition: ⭐ {len(significant_events)} SIGNIFICANT DETECTION(S) ⭐", "SUCCESS")
                print_status("    🎯 REMARKABLE! Mars has the weakest expected signal!", "INFO")
                for event in significant_events:
                    event_date = event['event_date']
                    sigma_level = event['sigma_level']
                    direction = event['direction']
                    center_days = event['center_days']
                    amplitude_pct = event['amplitude_pct']
                    print_status(f"   🎯 {event_date}: {sigma_level:.1f}σ {direction} at day {center_days:.1f}", "INFO")
                    print_status(f"      Amplitude: {amplitude_pct:.1f}% of baseline", "INFO")
            else:
                print_status(f"🔴 Mars Opposition: No significant detections (expected for weakest signal)", "INFO")

            if results.get('stacked_analysis', {}).get('success', False):
                # Mars typically has only one opposition in the dataset, so stacked analysis might be less relevant
                print_status(f"   📊 No stacked analysis (only one Mars opposition in dataset)", "INFO")
            else:
                print_status(f"   📊 Stacked Analysis: Failed or not run", "WARNING")

            if results.get('individual_event_fits', []):
                print_status(f"   Individual Event:", "INFO")
                for event in results['individual_event_fits']:
                    event_date = event['event_date']
                    sigma_level = event['sigma_level']
                    significant = event['is_significant']
                    center_days = event['center_days']
                    print_status(f"     {event_date}: {sigma_level:.1f}σ ({'✓' if significant else '✗'}) peak at day {center_days:.1f}", "INFO")
                    if TEPConfig.get_bool('TEP_VERBOSE_LOGGING'):
                        print_status(f"     Expected: 44x weaker than Jupiter, 4x weaker than Saturn", "INFO")
            else:
                print_status(f"   Individual Event: No significant individual detections", "INFO")
        else:
            print_status("🔴 Mars Opposition: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"🔴 Mars Opposition: ✗ Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def print_summary_lunar_standstill_results(results: Dict):
    """Print a summary of the Lunar Standstill analysis results"""
    print_status(f"LUNAR STANDSTILL ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_LUNAR_STANDSTILL'):
            enhancement = results.get('standstill_enhancement', {})
            if enhancement.get('success', False):
                status = "Significant enhancement detected" if enhancement.get('is_significant', False) else "No significant enhancement"
                ratio = enhancement.get('enhancement_ratio', 0.0)
                percent = (ratio - 1) * 100
                print_status(f"🌙 Major Lunar Standstill: {status}", "INFO")
                print_status(f"   Enhancement Ratio: {ratio:.2f}x ({percent:+.1f}%)", "INFO")
                print_status(f"   Pre-standstill amplitude: {enhancement.get('pre_amplitude', 0):.6f}", "INFO")
                print_status(f"   Standstill amplitude: {enhancement.get('standstill_amplitude', 0):.6f}", "INFO")
            else:
                print_status(f"🌙 Major Lunar Standstill: Insufficient data for enhancement analysis", "WARNING")

            # Monthly amplitudes
            monthly_amplitudes = results.get('monthly_amplitudes', {})
            if monthly_amplitudes.get('success', False):
                print_status(f"   Analysis periods:", "INFO")
                for period_name, stats in monthly_amplitudes.get('periods', {}).items():
                    print_status(f"     {period_name}: {stats['n_months']} months, amplitude = {stats['mean_amplitude']:.6f}", "INFO")
                
                peak_month = monthly_amplitudes.get('peak_amplitude_month', 'N/A')
                print_status(f"   Peak amplitude month: {peak_month}", "INFO")

            # Quadratic fit
            quadratic_fit = results.get('quadratic_fit', {})
            if quadratic_fit.get('success', False):
                offset = quadratic_fit.get('peak_offset_months', 0.0)
                r_squared = quadratic_fit.get('r_squared', 0.0)
                print_status(f"   Quadratic fit peak: {offset:.1f} months from expected ({r_squared:.3f} R²)", "INFO")
        else:
            print_status("🌙 Major Lunar Standstill: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"🌙 Major Lunar Standstill: ✗ Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def print_summary_astronomical_comparison(results: Dict):
    """Print a comparison of Jupiter vs Saturn vs Mars opposition results"""
    print_status(f"ASTRONOMICAL EVENTS COMPARISON - {results['analysis_center'].upper()}", "TITLE")

    if results.get('jupiter', {}).get('success', False) and \
       results.get('saturn', {}).get('success', False) and \
       results.get('mars', {}).get('success', False):
        
        jupiter = results['jupiter']
        saturn = results['saturn']
        mars = results['mars']

        # Individual event counts
        jupiter_significant = len([e for e in jupiter.get('individual_event_fits', []) if e.get('is_significant', False)])
        saturn_significant = len([e for e in saturn.get('individual_event_fits', []) if e.get('is_significant', False)])
        mars_significant = len([e for e in mars.get('individual_event_fits', []) if e.get('is_significant', False)])

        print_status(f"🪐 Jupiter: {jupiter_significant}/{jupiter.get('n_successful_events', 0)} significant events", "INFO")
        print_status(f"🪐 Saturn:  {saturn_significant}/{saturn.get('n_successful_events', 0)} significant events", "INFO")
        print_status(f"🔴 Mars:    {mars_significant}/{mars.get('n_successful_events', 0)} significant events", "INFO")

        # Expected ratios (if available)
        jupiter_expected = TEPConfig.get_float('JUPITER_EXPECTED_SIGNAL')
        saturn_expected = TEPConfig.get_float('SATURN_EXPECTED_SIGNAL')
        mars_expected = TEPConfig.get_float('MARS_EXPECTED_SIGNAL')
        if jupiter_expected and saturn_expected and mars_expected:
            print_status(f"📊 Expected amplitude ratios:", "INFO")
            print_status(f"   Jupiter/Saturn: {jupiter_expected/saturn_expected:.1f}x", "INFO")
            print_status(f"   Jupiter/Mars: {jupiter_expected/mars_expected:.1f}x", "INFO")
            print_status(f"   Saturn/Mars: {saturn_expected/mars_expected:.1f}x", "INFO")

        # Stacked analysis comparison
        jupiter_stacked = jupiter.get('stacked_analysis', {})
        saturn_stacked = saturn.get('stacked_analysis', {})
        if jupiter_stacked.get('success', False) and saturn_stacked.get('success', False):
            jupiter_sigma = jupiter_stacked.get('sigma_level', 0.0)
            saturn_sigma = saturn_stacked.get('sigma_level', 0.0)
            print_status(f"📈 Stacked significance: Jupiter {jupiter_sigma:.1f}σ vs Saturn {saturn_sigma:.1f}σ", "INFO")

        # Overall conclusion
        total_significant = jupiter_significant + saturn_significant + mars_significant
        if total_significant > 0:
            print_status(f"🌟 CONCLUSION: {total_significant} significant astronomical event signals detected!", "SUCCESS")
            if mars_significant > 0:
                print_status("    🎯 EXTRAORDINARY: Mars signal detected despite being weakest expected!", "SUCCESS")
        else:
            print_status("📊 CONCLUSION: No significant astronomical event signals detected", "INFO")
    else:
        print_status("⚠️  Cannot compare - one or more analyses failed", "WARNING")
    print_status("-" * 50, "INFO")

def print_summary_helical_motion_results(results: Dict):
    """Print a summary of the helical motion analysis results"""
    print_status(f"HELICAL MOTION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        # Chandler Wobble
        chandler_wobble = results.get('chandler_wobble_analysis', {})
        interp = chandler_wobble.get('interpretation', 'N/A')
        print_status(f"Chandler Wobble (14-month): {interp}", "INFO")

        # 3D Spherical Harmonics
        spherical_harmonics = results.get('spherical_harmonics_analysis', {})
        n_sectors = spherical_harmonics.get('n_valid_sectors', 0)
        cv = spherical_harmonics.get('coefficient_of_variation', 0.0)
        print_status(f"3D Spherical Harmonics: {n_sectors} directional sectors analyzed, CV = {cv:.3f}", "INFO")

        # Multi-Frequency Beat Analysis
        beat_frequencies = results.get('beat_frequencies_analysis', {})
        n_sig = beat_frequencies.get('n_significant_patterns', 0)
        print_status(f"Beat Frequencies: {n_sig} significant Earth motion interference patterns detected", "INFO")

        # Relative Motion Beat Analysis
        relative_motion = results.get('relative_motion_beats_analysis', {})
        interp = relative_motion.get('interpretation', 'N/A')
        print_status(f"Relative Motion: {interp}", "INFO")

        # Mesh Dance Analysis
        mesh_dance = results.get('mesh_dance_analysis', {})
        classification = mesh_dance.get('dance_signature_classification', 'N/A')
        score = mesh_dance.get('dance_score', 0.0)
        print_status(f"Mesh Dance Analysis: {classification} (score = {score:.3f})", "INFO")
        if TEPConfig.get_bool('TEP_VERBOSE_LOGGING'):
            print_status(f"   Dance Score: {score:.3f}/1.0", "DEBUG")

        # Jupiter Opposition Analysis (as part of helical motion suite)
        jupiter_opp = results.get('jupiter_opposition_analysis', {})
        n_events = jupiter_opp.get('n_events_analyzed', 0)
        interpretation = jupiter_opp.get('interpretation', 'N/A')
        print_status(f"Jupiter Opposition: {n_events} events analyzed - {interpretation}", "INFO")

        # Nutation Analysis
        nutation = results.get('nutation_analysis', {})
        if nutation.get('success', False):
            ew_interpretation = nutation.get('ew_interpretation', 'N/A')
            ns_interpretation = nutation.get('ns_interpretation', 'N/A')
            print_status(f"Nutation Analysis (E-W): {ew_interpretation}", "INFO")
            print_status(f"Nutation Analysis (N-S): {ns_interpretation}", "INFO")

        # Tid Exclusion
        tid_exclusion = results.get('tid_exclusion_analysis', {})
        if tid_exclusion.get('success', False):
            significant_bands = tid_exclusion.get('significant_bands', [])
            print_status(f"TID Exclusion Analysis: {len(significant_bands)} significant bands excluded", "INFO")

        # Additional Visualizations
        additional_viz = results.get('additional_visualizations', {})
        if additional_viz.get('success', False):
            for fig in additional_viz.get('figures_generated', []):
                print_status(f"Figure Generated: {fig}", "INFO")

        # Methodology Validation
        method_validation = results.get('methodology_validation', {})
        if method_validation.get('success', False):
            for key, value in method_validation.get('metrics', {}).items():
                print_status(f"Validation Metric {key}: {value}", "INFO")

        # Gravitational Temporal Field Analysis (GTFA)
        gtfa = results.get('gravitational_temporal_field_analysis', {})
        if gtfa.get('success', False):
            enhancement = gtfa.get('summary_metrics', {}).get('global_enhancement', {})
            significant = enhancement.get('is_significant', False)
            print_status(f"Gravitational Temporal Field Analysis: {enhancement.get('enhancement_ratio', 0):.4f}x enhancement ({str(significant)})", "INFO")
            if gtfa.get('station_impact_analysis', {}).get('success', False):
                impacts = gtfa['station_impact_analysis']['impacted_stations']
                print_status(f"Impacted Stations: {len(impacts)} detected", "INFO")

        # Geographic Bias Validation
        geographic_bias = results.get('geographic_bias_validation', {})
        if geographic_bias.get('success', False):
            bias_detected = geographic_bias.get('bias_detected', False)
            print_status(f"Geographic Bias Validation: Bias detected = {bias_detected}", "INFO")

        # Realistic Ionospheric Validation
        ionospheric_validation = results.get('realistic_ionospheric_validation', {})
        if ionospheric_validation.get('success', False):
            validation_result = ionospheric_validation.get('validation_result', 'N/A')
            print_status(f"Realistic Ionospheric Validation: {validation_result}", "INFO")

        # Targeted Diurnal Analysis
        diurnal_analysis = results.get('targeted_diurnal_analysis', {})
        if diurnal_analysis.get('success', False):
            significant_patterns = diurnal_analysis.get('significant_diurnal_patterns', 0)
            print_status(f"Targeted Diurnal Analysis: {significant_patterns} significant patterns detected", "INFO")
            for pattern in diurnal_analysis.get('patterns', []):
                print_status(f"   Pattern {pattern['id']}: {pattern['status']}", "INFO")

        # Block-wise Cross Validation
        block_wise_cv = results.get('block_wise_cross_validation', {})
        if block_wise_cv.get('success', False):
            cv_score = block_wise_cv.get('cross_validation_score', 0.0)
            print_status(f"Block-wise Cross Validation: Score = {cv_score:.3f}", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"Helical Motion Analysis: ✗ Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def main():
    """Main function with command-line options for different analysis modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEP GNSS Geospatial Temporal Analysis - Step 5")
    parser.add_argument('--mode', choices=['full', 'helical', 'jupiter', 'saturn', 'mars', 'lunar', 'eclipse', 'astronomical'], default='full',
                        help='Analysis mode: full (complete geospatial temporal analysis), helical (helical motion analyses only), jupiter (Jupiter opposition only), saturn (Saturn opposition only), mars (Mars opposition only), lunar (Lunar Standstill only), or astronomical (Jupiter, Saturn, and Mars)')
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
        print_status("   python step_5_tep_statistical_validation.py --mode helical", "INFO")
        print_status("   python step_5_tep_statistical_validation.py --mode jupiter --center esa_final", "INFO")
        print_status("   python step_5_tep_statistical_validation.py --mode saturn --center code", "INFO")
        print_status("   python step_5_tep_statistical_validation.py --mode mars --center igs_combined", "INFO")
        print_status("   python step_5_tep_statistical_validation.py --mode lunar --center igs_combined", "INFO")
        print_status("   python step_5_tep_statistical_validation.py --mode astronomical  # All planets", "INFO")
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
    
    if args.mode == 'astronomical':
        # Run Jupiter, Saturn, AND Mars opposition analyses
        results = run_astronomical_events_only(args.center)
        return all(r.get('success', False) for r in results.values())
    
    # Original full Step 5 analysis
    print_status("TEP GNSS Analysis Package v0.13 - STEP 5: Geospatial Temporal Analysis", "TITLE")
    
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
    # Memory check removed - warnings disabled
    
    # Process analysis centers
    if args.center:
        centers = [args.center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    results = {}
    for ac in centers:
        print(f"\n{'='*60}")
        print(f"PROCESSING {ac.upper()} - Geospatial Temporal Analysis")
        print(f"{'='*60}")
        
        result = process_analysis_center(ac)
        results[ac] = result
        
        # Save individual results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"step_5_geospatial_temporal_analysis_{ac}.json"
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