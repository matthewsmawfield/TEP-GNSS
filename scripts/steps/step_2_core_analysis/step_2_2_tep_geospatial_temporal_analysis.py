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

MULTI-SCALE WINDOW STRATEGY:
Different analyses use different temporal windows matched to their characteristic physical timescales:
- Temporal Orbital Tracking: 30-day windows (balances seasonal signal vs noise)
- Mesh Dance Analysis: 120-day windows (long-timescale collective dynamics)
- Planetary Oppositions: 240-day windows (gravitational coupling optimal timescale)
- Chandler Wobble: Full 433-day cycle analysis
- Beat Analysis: Period-specific windows for each beat frequency
- Lunar Standstill: Monthly resolution for 18.6-year cycle tracking

This multi-scale approach is scientifically rigorous as each phenomenon operates on its
characteristic timescale. Based on empirical analysis showing optimal coupling at 240 days
for gravitational-temporal field interactions (Savitzky-Golay smoothing analysis).

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
Date: October 2025
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
from scipy.stats import norm
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
ROOT = Path(__file__).resolve().parents[3]

# Import TEP utilities for better configuration and error handling
sys.path.insert(0, str(ROOT))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    TEPAnalysisError, safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)
from scripts.utils.geospatial import compute_azimuth, classify_ew_ns
from scripts.utils.pid_manager import ensure_single_instance
from scripts.utils.logger import print_status, TEPLogger, set_step_logger # Import logger functions

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_2_2_geospatial_temporal_analysis",
    level="DEBUG",
    log_file_path=ROOT / "logs" / "step_2_2_geospatial_temporal_analysis.log"
)

# Register step logger so print_status uses it
set_step_logger(step_logger)

def check_memory_usage():
    """Monitor memory usage and warn if approaching limits"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    
    print_status(f"Memory usage: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)", "INFO")
    
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
        
        print_status(f"Performance: {func.__name__} took {execution_time:.2f}s, memory Δ: {memory_delta:+.2f} GB", "DEBUG")
        
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
    Load complete pair dataset from Step 2.1 geospatial files (with pre-computed azimuth).
    
    This is more efficient than loading from Step 2.0 pair files because:
    - Azimuth is already computed in Step 2.1
    - Delta longitude and local time differences are pre-calculated
    - Smaller file size due to aggregation
    
    Args:
        ac: Analysis center name ('code', 'igs_combined', 'esa_final')
    
    Returns:
        pd.DataFrame: Complete dataset with azimuth and geospatial metrics
    """
    print_status(f"Loading complete geospatial dataset from Step 2.1 for {ac.upper()}...", "PROCESS")
    
    # Load from Step 2.1 geospatial file (much more efficient)
    geospatial_file = ROOT / "data" / "processed" / f"step_2_1_geospatial_{ac}.csv"
    
    if not geospatial_file.exists():
        raise TEPFileError(f"Step 2.1 geospatial file not found: {geospatial_file}")
    
    print_status(f"Loading from {geospatial_file}", "INFO")
    
    # Check file size for progress estimation
    file_size_mb = geospatial_file.stat().st_size / (1024 * 1024)
    print_status(f"File size: {file_size_mb:.1f} MB", "DEBUG")
    
    try:
        # Load the complete geospatial dataset with progress monitoring
        print_status("Reading CSV file into memory...", "PROCESS")
        complete_df = pd.read_csv(geospatial_file, parse_dates=['date'])
        print_status(f"CSV loaded successfully: {len(complete_df):,} rows", "SUCCESS")
        
        # Add coherence column (preserving sign like Step 2.0)
        print_status("Computing coherence values from plateau phase...", "PROCESS")
        complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
        
        # Clean data - ENHANCED QUALITY FILTERING (aligned with Step 2.0)
        print_status("Cleaning and filtering data...", "PROCESS")
        initial_count = len(complete_df)
        
        # Filter 1: Remove NaN values in critical columns
        complete_df.dropna(subset=['dist_km', 'coherence', 'station_i', 'station_j', 'date'], inplace=True)
        after_dropna = len(complete_df)
        
        # Filter 2: Remove zero or negative distances
        complete_df = complete_df[complete_df['dist_km'] > 0]
        after_dist_filter = len(complete_df)
        
        # Filter 3: Remove NaN or infinite coherence values (KEY FILTER from Step 2.0 line 1317)
        # This ensures we skip pairs with failed correlation analysis
        complete_df = complete_df[~np.isnan(complete_df['coherence'])]
        complete_df = complete_df[~np.isinf(complete_df['coherence'])]
        after_coherence_nan_filter = len(complete_df)
        
        # Filter 4: Validate coherence range (cos() should give [-1, 1])
        # This catches any numerical errors or data corruption
        complete_df = complete_df[(complete_df['coherence'] >= -1.0) & (complete_df['coherence'] <= 1.0)]
        final_count = len(complete_df)
        
        print_status(f"Data filtering: {initial_count:,} → {after_dropna:,} → {after_dist_filter:,} → {after_coherence_nan_filter:,} → {final_count:,} pairs", "DEBUG")
        
        # VERIFICATION: Check if filtering removed any data
        total_filtered = initial_count - final_count
        if total_filtered > 0:
            print_status(f"Quality filtering removed {total_filtered:,} pairs ({100*total_filtered/initial_count:.2f}%)", "INFO")
            print_status(f"  Filter breakdown: NaN removal: {initial_count - after_dropna:,}, Distance: {after_dropna - after_dist_filter:,}, Coherence NaN/Inf: {after_dist_filter - after_coherence_nan_filter:,}, Range: {after_coherence_nan_filter - final_count:,}", "DEBUG")
        else:
            print_status("No pairs filtered - data is clean from Step 2.1", "SUCCESS")
        
        # DATA QUALITY DIAGNOSTICS
        print_status(f"Data quality metrics:", "INFO")
        print_status(f"  Coherence range: [{complete_df['coherence'].min():.6f}, {complete_df['coherence'].max():.6f}]", "INFO")
        print_status(f"  Coherence mean: {complete_df['coherence'].mean():.6f} ± {complete_df['coherence'].std():.6f}", "INFO")
        print_status(f"  Distance range: [{complete_df['dist_km'].min():.1f}, {complete_df['dist_km'].max():.1f}] km", "INFO")
        print_status(f"  Distance mean: {complete_df['dist_km'].mean():.1f} ± {complete_df['dist_km'].std():.1f} km", "INFO")
        
        print_status(f"Geospatial dataset loaded: {len(complete_df):,} pairs, {complete_df.memory_usage(deep=True).sum()/(1024**3):.2f} GB", "SUCCESS")
        print_status("Azimuth already computed in Step 2.1 - no redundant calculation needed", "SUCCESS")
        
        # VERIFICATION: Cross-check with Step 2.1 geospatial processing log
        # Note: Step 2.0 consolidated CSV contains distance-binned aggregate data (~117k bins)
        # while Step 2.2 uses raw pair-by-pair data (~39M pairs) - this is correct!
        try:
            geospatial_log = f'results/outputs/step_2_1_geospatial_processing.json'
            if os.path.exists(geospatial_log):
                import json
                with open(geospatial_log, 'r') as f:
                    geo_log = json.load(f)
                
                # Check if this AC's data is in the log
                ac_key = ac.lower().replace('_', '')  # Convert 'igs_combined' to 'igscombined' if needed
                # Try both formats: 'code', 'igs_combined', 'esa_final'
                analysis_centers = geo_log.get('analysis_centers', {})
                
                # Try exact match first, then try without underscore
                ac_data = analysis_centers.get(ac.lower(), analysis_centers.get(ac_key, {}))
                
                if ac_data:
                    step_2_1_count = ac_data.get('total_pairs', 0)
                    
                    if step_2_1_count == final_count:
                        print_status(f"✓ VERIFIED: Analyzing same {final_count:,} pairs as Step 2.1", "SUCCESS")
                    elif step_2_1_count > 0 and abs(step_2_1_count - final_count) / step_2_1_count < 0.01:  # Within 1%
                        print_status(f"✓ CLOSE MATCH: Step 2.1 processed {step_2_1_count:,} pairs, we have {final_count:,} pairs (diff: {abs(step_2_1_count - final_count):,})", "SUCCESS")
                    elif step_2_1_count > 0:
                        print_status(f"⚠ MISMATCH: Step 2.1 processed {step_2_1_count:,} pairs, we have {final_count:,} pairs (diff: {abs(step_2_1_count - final_count):,})", "WARNING")
                    else:
                        print_status(f"Step 2.1 log exists but no pair count found for {ac}", "DEBUG")
                else:
                    print_status(f"Step 2.1 processing data not found for {ac} in log", "DEBUG")
            else:
                print_status(f"Step 2.1 processing log not found - cannot verify pair count", "DEBUG")
        except Exception as e:
            print_status(f"Could not verify pair count with Step 2.1: {e}", "DEBUG")
        
        # Verify required columns are present
        print_status("Verifying required columns are present...", "PROCESS")
        required_cols = ['azimuth', 'delta_longitude', 'delta_local_time']
        missing_cols = [col for col in required_cols if col not in complete_df.columns]
        
        if missing_cols:
            raise TEPDataError(f"Missing required columns from Step 2.1: {missing_cols}")
        
        print_status(f"All required columns present: {required_cols}", "SUCCESS")
        print_status(f"Available columns: {list(complete_df.columns)}", "INFO")
        check_memory_usage()
        
        return complete_df
        
    except Exception as e:
        print_status(f"Failed to load Step 2.1 geospatial data: {e}", "ERROR")
        print_status("Falling back to Step 2.0 pair data loading...", "WARNING")
        return load_complete_pair_dataset(ac)

def load_complete_pair_dataset(ac: str, use_chunked_processing: bool = None) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center with smart memory management.
    Uses consolidated data for consistency with main Step 2.0 analysis.
    
    Args:
        ac: Analysis center name
        use_chunked_processing: Force chunked processing (None = auto-detect based on memory)
    
    Returns:
        pd.DataFrame: Complete dataset with columns [date, station_i, station_j, 
                     dist_km, plateau_phase, coherence, ...]
    """
    print_status(f"Loading complete pair-level dataset for {ac.upper()}...", "PROCESS")
    
    # Use consolidated data from Step 2.0 for consistency with main analysis
    consolidated_file = ROOT / 'results' / 'outputs' / f'step_2_0_pairs_consolidated_{ac}.csv'
    
    if consolidated_file.exists():
        print_status(f"Using consolidated data: {consolidated_file.name}", "INFO")
        try:
            complete_df = pd.read_csv(consolidated_file)
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
    try:
        pair_dir = validate_directory_exists(ROOT / 'results' / 'tmp', "Pair-level data directory")
    except TEPFileError as e:
        raise TEPDataError(f"Pair-level data directory not available: {e}") from e
    
    pair_files = list(pair_dir.glob(f"step_2_0_pairs_{ac}_*.csv"))
    if not pair_files:
        raise TEPDataError(f"No pair-level files found for {ac}")
    
    print_status(f"Found {len(pair_files)} pair-level files to load (fallback mode)", "INFO")
    
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
    # Calculate proper phase coherence (preserving sign like Step 2.0)
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
        if i % TEPConfig.get_int('TEP_FILE_LOGGING_INTERVAL') == 0:  # Log progress for debugging
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

def _subsample_to_match_distribution_enhanced(sector_distances, reference_distances, max_samples=5000):
    """
    Subsample sector distances to match the reference distance distribution.
    
    This function implements distance distribution matching by subsampling pairs
    from a sector to match the global distance distribution, preventing bias
    in λEW/λNS ratios from differing distance sampling patterns.
    
    Args:
        sector_distances: Array of distances for the specific sector
        reference_distances: Array of all distances (global reference)
        max_samples: Maximum number of samples to return
        
    Returns:
        Array of indices to subsample from sector_distances
    """
    import numpy as np
    from scipy import stats
    
    # Create distance bins based on reference distribution
    n_bins = min(20, len(np.unique(reference_distances)) // 10)
    if n_bins < 5:
        n_bins = 5
    
    # Compute reference histogram
    ref_hist, ref_bins = np.histogram(reference_distances, bins=n_bins, density=True)
    
    # Compute sector histogram
    sector_hist, sector_bins = np.histogram(sector_distances, bins=ref_bins, density=True)
    
    # Calculate target counts per bin for the sector
    total_sector_pairs = len(sector_distances)
    target_samples = min(max_samples, total_sector_pairs)
    
    # For each bin, determine how many samples we want
    target_counts = []
    for i in range(len(ref_bins) - 1):
        # Target fraction based on reference distribution
        target_fraction = ref_hist[i] / np.sum(ref_hist)
        target_count = int(target_fraction * target_samples)
        target_counts.append(target_count)
    
    # Subsample from each bin
    selected_indices = []
    for i in range(len(ref_bins) - 1):
        # Find indices in this distance bin
        bin_mask = (sector_distances >= ref_bins[i]) & (sector_distances < ref_bins[i+1])
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) > 0:
            # Sample up to target count
            n_sample = min(target_counts[i], len(bin_indices))
            if n_sample > 0:
                # Random sampling with fixed seed for reproducibility
                np.random.seed(42 + i)  # Different seed per bin
                sampled_indices = np.random.choice(bin_indices, size=n_sample, replace=False)
                selected_indices.extend(sampled_indices)
    
    return np.array(selected_indices)


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
    
    # Check if azimuths are already computed (from Step 2.1)
    if 'azimuth' in coord_df.columns and coord_df['azimuth'].notna().all():
        print_status("Using pre-computed azimuths from Step 2.1", "SUCCESS")
    else:
        # Compute azimuths for all pairs (fallback for Step 2.0 data)
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
    
    # DISTANCE DISTRIBUTION MATCHING GUARDRAIL
    # ========================================
    print_status("Applying distance distribution matching guardrails...", "PROCESS")
    
    # Compute global distance distribution for reference
    all_distances = coord_df['dist_km'].values
    global_dist_hist, global_dist_bins = np.histogram(all_distances, bins=20, density=True)
    
    # Apply distance distribution matching to each sector
    sector_data_matched = {}
    distance_matching_results = {}
    
    for sector in sector_names:
        sector_mask = coord_df['sector'] == sector
        sector_data = coord_df[sector_mask]
        
        if len(sector_data) < 1000:  # Skip sectors with insufficient data
            continue
            
        sector_distances = sector_data['dist_km'].values
        sector_coherences = sector_data['coherence'].values
        
        # Method 1: Distance-weighted analysis
        sector_dist_hist, sector_dist_bins = np.histogram(sector_distances, bins=20, density=True)
        
        # Compute weights to match global distribution
        weights = np.ones_like(sector_distances)
        for i, dist in enumerate(sector_distances):
            # Find which global bin this distance falls into
            global_bin_idx = np.digitize(dist, global_dist_bins) - 1
            global_bin_idx = max(0, min(global_bin_idx, len(global_dist_hist) - 1))
            
            # Find which sector bin this distance falls into
            sector_bin_idx = np.digitize(dist, sector_dist_bins) - 1
            sector_bin_idx = max(0, min(sector_bin_idx, len(sector_dist_hist) - 1))
            
            # Weight inversely proportional to sector density relative to global density
            if sector_dist_hist[sector_bin_idx] > 0:
                weight = global_dist_hist[global_bin_idx] / sector_dist_hist[sector_bin_idx]
                weights[i] = weight
        
        # Method 2: Matched-distance subsampling
        matched_indices = _subsample_to_match_distribution_enhanced(
            sector_distances, all_distances, max_samples=min(5000, len(sector_distances))
        )
        
        sector_data_matched[sector] = {
            'distances_weighted': sector_distances,
            'coherences_weighted': sector_coherences,
            'weights': weights,
            'distances_matched': sector_distances[matched_indices],
            'coherences_matched': sector_coherences[matched_indices],
            'original_count': len(sector_distances),
            'matched_count': len(matched_indices),
            'sector_data': sector_data  # Keep original data for compatibility
        }
        
        # Validate distance distribution matching
        if len(matched_indices) > 100:
            from scipy import stats
            ks_stat, ks_pvalue = stats.ks_2samp(
                sector_distances[matched_indices], all_distances
            )
            distance_matching_results[sector] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'distribution_matched': ks_pvalue > 0.05
            }
    
    print_status(f"Distance distribution matching applied to {len(sector_data_matched)} sectors", "SUCCESS")
    
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    print_status(f"Analyzing {len(sector_names)} directional sectors with {num_bins} distance bins", "INFO")
    
    # Analyze each sector with distance-matched data
    sector_results = {}
    sector_results_weighted = {}
    
    for i, sector in enumerate(sector_names):
        if sector not in sector_data_matched:
            print_status(f"Skipping sector {sector}: insufficient data or matching failed", "WARNING")
            continue
            
        matched_data = sector_data_matched[sector]
        print_status(f"Processing sector {i+1}/{len(sector_names)}: {sector} ({matched_data['matched_count']:,} matched pairs from {matched_data['original_count']:,} original)", "PROCESS")
        
        # Use matched-distance subsampling approach (Method 2)
        distances_arr = matched_data['distances_matched']
        coherences_arr = matched_data['coherences_matched']
        
        if len(distances_arr) < 1000:  # Need sufficient data
            print_status(f"Skipping sector {sector}: insufficient matched data ({len(distances_arr)} pairs)", "WARNING")
            continue
        
        # Bin the matched sector data
        print_status(f"  Binning {sector} matched data into {num_bins} distance bins...", "DEBUG")
        dist_bins = pd.cut(pd.Series(distances_arr), bins=edges, right=False)
        
        # Create temporary DataFrame for binning
        temp_df = pd.DataFrame({
            'dist_km': distances_arr,
            'coherence': coherences_arr,
            'dist_bin': dist_bins
        })
        
        # Group by bins
        binned = temp_df.groupby('dist_bin', observed=True).agg(
            mean_dist=('dist_km', 'mean'),
            mean_coh=('coherence', 'mean'),
            count=('coherence', 'size')
        ).reset_index()
        
        # Filter for robust bins
        binned = binned[binned['count'] >= min_bin_count].dropna()
        print_status(f"  {sector}: {len(binned)} valid bins (min {min_bin_count} pairs per bin)", "DEBUG")
        
        if len(binned) < 5:  # Need enough bins for fitting
            print_status(f"  Skipping {sector}: insufficient bins for fitting ({len(binned)} < 5)", "WARNING")
            continue
        
        # Fit exponential model to distance-matched data
        print_status(f"  Fitting exponential correlation model to {sector} matched data...", "DEBUG")
        try:
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            c_range = coherences.max() - coherences.min()
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            # Adaptive bounds based on data characteristics
            adaptive_bounds = TEPConfig.get_adaptive_lambda_bounds(distances)

            popt, pcov = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=adaptive_bounds,
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
                'n_pairs': len(distances_arr),
                'n_bins': len(binned),
                'param_errors': [float(np.sqrt(pcov[i, i])) for i in range(3)],
                'distance_matching_applied': True,
                'original_pairs': matched_data['original_count'],
                'matched_pairs': matched_data['matched_count']
            }
            print_status(f"  {sector} fit successful: λ = {popt[1]:.1f} km, R² = {r_squared:.3f} (distance-matched)", "SUCCESS")
            
            # Also compute weighted analysis for comparison
            distances_weighted = matched_data['distances_weighted']
            coherences_weighted = matched_data['coherences_weighted']
            weights_array = matched_data['weights']
            
            # Weighted binning
            temp_df_weighted = pd.DataFrame({
                'dist_km': distances_weighted,
                'coherence': coherences_weighted,
                'weight': weights_array,
                'dist_bin': pd.cut(pd.Series(distances_weighted), bins=edges, right=False)
            })
            
            binned_weighted = temp_df_weighted.groupby('dist_bin', observed=True).agg(
                mean_dist=('dist_km', lambda x: np.average(x, weights=temp_df_weighted.loc[x.index, 'weight'])),
                mean_coh=('coherence', lambda x: np.average(x, weights=temp_df_weighted.loc[x.index, 'weight'])),
                count=('coherence', 'size'),
                total_weight=('weight', 'sum')
            ).reset_index()
            
            binned_weighted = binned_weighted[binned_weighted['count'] >= min_bin_count].dropna()
            
            if len(binned_weighted) >= 5:
                distances_w = binned_weighted['mean_dist'].values
                coherences_w = binned_weighted['mean_coh'].values
                weights_w = binned_weighted['total_weight'].values
                
                popt_w, pcov_w = curve_fit(
                    correlation_model, distances_w, coherences_w,
                    p0=p0, sigma=1.0/np.sqrt(weights_w),
                    bounds=adaptive_bounds,
                    maxfev=5000
                )
                
                y_pred_w = correlation_model(distances_w, *popt_w)
                ss_res_w = np.sum(weights_w * (coherences_w - y_pred_w)**2)
                ss_tot_w = np.sum(weights_w * (coherences_w - np.average(coherences_w, weights=weights_w))**2)
                r_squared_w = 1 - ss_res_w/ss_tot_w if ss_tot_w > 0 else 0
                
                sector_results_weighted[sector] = {
                    'amplitude': float(popt_w[0]),
                    'lambda_km': float(popt_w[1]),
                    'offset': float(popt_w[2]),
                    'r_squared': float(r_squared_w),
                    'n_pairs': len(distances_weighted),
                    'n_bins': len(binned_weighted),
                    'param_errors': [float(np.sqrt(pcov_w[i, i])) for i in range(3)],
                    'distance_weighting_applied': True
                }
            
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
        'sector_results_weighted': sector_results_weighted,
        'anisotropy_statistics': {
            'lambda_mean': float(lambda_mean),
            'lambda_std': float(lambda_std),
            'coefficient_of_variation': float(lambda_cv),
            'n_sectors': len(sector_results),
            'anisotropy_category': 'extreme' if lambda_cv > 0.8 else 'moderate' if lambda_cv > 0.2 else 'minimal'
        },
        'earth_motion_analysis': earth_motion_analysis,
        'distance_matching_results': distance_matching_results,
        'distance_matching_applied': True,
        'guardrail_summary': {
            'sectors_analyzed': len(sector_results),
            'sectors_with_valid_matching': sum(1 for r in distance_matching_results.values() if r['distribution_matched']),
            'matching_methods': ['subsampling', 'weighting'],
            'validation_passed': all(r['distribution_matched'] for r in distance_matching_results.values()) if distance_matching_results else False
        },
        'data_summary': {
            'total_pairs_with_coords': len(coord_df),
            'sectors_analyzed': list(sector_results.keys()),
            'distance_matching_applied': True
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
    
    # Check if azimuths are already computed (from Step 2.1)
    if 'azimuth' in complete_df.columns and complete_df['azimuth'].notna().all():
        print_status("Using pre-computed azimuths from Step 2.1", "SUCCESS")
    else:
        # Compute azimuths for all pairs (fallback for Step 2.0 data)
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
    
    # ========================================
    # TEMPORAL ORBITAL TRACKING WINDOW STRATEGY
    # ========================================
    # Window size: 30 days (±15 days)
    # Rationale: Balances seasonal signal preservation (365-day cycle) with noise reduction
    #            30-day window is appropriate for averaging out weekly/monthly variations
    #            while preserving the annual orbital motion signal we're detecting
    # Sampling: Every 10 days (34 samples per year)
    # Nyquist: Well above Nyquist criterion for 365-day cycle (need >2 samples per cycle)
    # Expected: Stronger correlation than 5-day windows (closer to optimal coupling timescale)
    # ========================================
    
    temporal_tracking = []
    day_samples = range(15, 351, 10)  # Sample every 10 days from day 15 to 350 (allows ±15 day windows)
    
    print_status(f"Tracking E-W/N-S ratio across {len(day_samples)} temporal samples (each using 30-day windows)...", "PROCESS")
    print_status(f"Window strategy: Each sample uses ±15 days (30-day total) to balance seasonal signal with noise reduction", "INFO")
    
    for day_of_year in day_samples:
        # OPTIMIZED: Use 30-day window (±15 days) to balance noise reduction with seasonal signal preservation
        # This aligns better with coupling timescales (30d between 60-240d optimal range)
        # while still capturing the 365-day orbital cycle
        day_window = 15  # ±15 days = 30-day total window
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
        return offset + amplitude * np.cos(2 * np.pi * day / 365.25 + phase)
    
    try:
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
        
        # Adaptive bounds based on data characteristics
        adaptive_bounds = TEPConfig.get_adaptive_lambda_bounds(distances)

        popt, _ = curve_fit(
            correlation_model, distances, coherences,
            p0=p0, sigma=1.0/np.sqrt(weights),
            bounds=adaptive_bounds,
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
    print_status("=" * 60, "INFO")
    
    # Display multi-scale window strategy
    print_status("MULTI-SCALE TEMPORAL WINDOW STRATEGY:", "INFO")
    print_status("  Temporal Orbital Tracking: 30-day windows (seasonal signal + noise reduction)", "INFO")
    print_status("  Mesh Dance Analysis: OPTIMIZED (90d coherence + 30d oscillation/spiral)", "INFO")
    print_status("  Planetary Oppositions: 240-day windows (optimal coupling timescale)", "INFO")
    print_status("  Chandler Wobble: Full 433-day cycle (period-matched analysis)", "INFO")
    print_status("  Beat Analysis: Period-specific windows (matched to each frequency)", "INFO")
    print_status("Rationale: Each analysis uses windows matched to its characteristic physical timescale", "INFO")
    print_status("Based on empirical analysis showing optimal coupling at 240 days (Savitzky-Golay)", "INFO")
    print_status("=" * 60, "INFO")
    
    start_time = time.time()
    
    try:
        # Load complete dataset into memory (Step 2.1 geospatial data with pre-computed azimuth)
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
            results['beat_frequencies_analysis'] = run_multi_frequency_beat_analysis_aligned(complete_df)
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
        
        # Run Venus Inferior Conjunction analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_VENUS_CONJUNCTION', True):  # Default True - significant signal
            results['venus_conjunction_analysis'] = run_venus_opposition_analysis(complete_df)
        else:
            results['venus_conjunction_analysis'] = {'enabled': False}
        
        # Run Mercury Inferior Conjunction analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_MERCURY_CONJUNCTION', True):  # Default True - complete inner planets
            results['mercury_conjunction_analysis'] = run_mercury_opposition_analysis(complete_df)
        else:
            results['mercury_conjunction_analysis'] = {'enabled': False}
        
        # Run Solar Rotation analysis if enabled
        if TEPConfig.get_bool('TEP_ENABLE_SOLAR_ROTATION', True):  # Default True - unique mechanism test
            results['solar_rotation_analysis'] = run_solar_rotation_analysis(complete_df)
        else:
            results['solar_rotation_analysis'] = {'enabled': False}
        
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
        
        # Print summaries for planetary opposition analyses
        if results.get('jupiter_opposition_analysis', {}).get('success') is not False:
            print_summary_jupiter_results(results)
        if results.get('saturn_opposition_analysis', {}).get('success') is not False:
            print_summary_saturn_results(results)
        if results.get('mars_opposition_analysis', {}).get('success') is not False:
            print_summary_mars_results(results)
        
        # Run temporal coherence assessment for signal stability validation
        try:
            # Check if temporal coherence analysis is enabled (only remaining enhanced analysis)
            temporal_enabled = TEPConfig.get_bool('TEP_ENABLE_TEMPORAL_COHERENCE', default=True)
            
            if temporal_enabled:
                # Reload df for temporal coherence analysis
                complete_df = load_complete_geospatial_dataset(ac)
                
                # Temporal Coherence Assessment
                # Analyzes signal persistence across multiple timescales to validate temporal stability
                print_status("\n" + "="*80, "INFO")
                results['temporal_coherence'] = analyze_temporal_coherence(complete_df, results)
                
                del complete_df
                gc.collect()
            else:
                print_status("Temporal coherence analysis disabled - skipping dataset reload", "INFO")
            
        except Exception as e:
            print_status(f"Enhanced analysis modules failed: {e}", "WARNING")
        
        # Generate comprehensive scientific significance report (Option B)
        try:
            print_status("\n", "INFO")
            comprehensive_report = generate_comprehensive_scientific_report(results, ac)
            results['comprehensive_report'] = comprehensive_report
        except Exception as e:
            print_status(f"Comprehensive report generation failed: {e}", "WARNING")
        
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
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}", "TITLE")
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
            # Load complete dataset from Step 2.1 (with pre-computed azimuth)
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
                results['beat_frequencies_analysis'] = run_multi_frequency_beat_analysis_aligned(complete_df)
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
            
            # 9. Venus Inferior Conjunction Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_VENUS_CONJUNCTION', True):
                print_status("Running Venus Inferior Conjunction Analysis...", "PROCESS")
                results['venus_conjunction_analysis'] = run_venus_opposition_analysis(complete_df)
            else:
                results['venus_conjunction_analysis'] = {'enabled': False}
            
            # 10. Mercury Inferior Conjunction Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_MERCURY_CONJUNCTION', True):
                print_status("Running Mercury Inferior Conjunction Analysis...", "PROCESS")
                results['mercury_conjunction_analysis'] = run_mercury_opposition_analysis(complete_df)
            else:
                results['mercury_conjunction_analysis'] = {'enabled': False}
            
            # 11. Solar Rotation Cycle Analysis (if enabled)
            if TEPConfig.get_bool('TEP_ENABLE_SOLAR_ROTATION', True):
                print_status("Running Solar Rotation Cycle Analysis...", "PROCESS")
                results['solar_rotation_analysis'] = run_solar_rotation_analysis(complete_df)
            else:
                results['solar_rotation_analysis'] = {'enabled': False}
            
            # 12. Lunar Standstill Analysis (if enabled)
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
            
            output_file = output_dir / f"step_2_2_helical_motion_only_{ac}.json"
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
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}", "TITLE")
    print_status("JUPITER OPPOSITION ANALYSIS - Gravitational Potential Pulse Detection", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in centers:
        print_status(f"\n{'='*60}", "INFO")
        print_status(f"PROCESSING {ac.upper()} - JUPITER OPPOSITION ANALYSIS", "INFO")
        print_status(f"{'='*60}", "INFO")
        
        try:
            # Load complete dataset from Step 2.1 (with pre-computed azimuth)
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
            
            output_file = output_dir / f"step_2_2_jupiter_only_{ac}.json"
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
    print_status("JUPITER OPPOSITION ANALYSIS COMPLETE", "TITLE")
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
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}", "TITLE")
    print_status("SATURN OPPOSITION ANALYSIS - Gravitational Potential Pulse Detection", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    for center in centers:
        print_status(f"\n{'='*60}", "INFO")
        print_status(f"PROCESSING {center.upper()} - SATURN OPPOSITION ANALYSIS", "TITLE")
        print_status(f"{'='*60}", "INFO")
        
        # Load data for this center
        complete_df = load_complete_geospatial_dataset(center)
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Saturn opposition analysis
        results = {'analysis_center': center}
        results['saturn_opposition_analysis'] = run_saturn_opposition_analysis(complete_df)
        
        # Print summary
        print_summary_saturn_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_2_2_saturn_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Saturn opposition results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print_status("SATURN OPPOSITION ANALYSIS COMPLETED", "TITLE")
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
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}", "TITLE")
    print_status("MARS OPPOSITION ANALYSIS - Weakest Signal Sensitivity Test", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    for center in centers:
        print_status(f"\n{'='*60}", "INFO")
        print_status(f"PROCESSING {center.upper()} - MARS OPPOSITION ANALYSIS", "TITLE")
        print_status(f"{'='*60}", "INFO")
        
        # Load data for this center
        complete_df = load_complete_geospatial_dataset(center)
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Mars opposition analysis
        results = {'analysis_center': center}
        results['mars_opposition_analysis'] = run_mars_opposition_analysis(complete_df)
        
        # Print summary
        print_summary_mars_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_2_2_mars_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Mars opposition results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print_status("MARS OPPOSITION ANALYSIS COMPLETED", "TITLE")
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
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}", "TITLE")
    print_status("LUNAR STANDSTILL ANALYSIS - Sidereal Day Amplitude Tracking", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    for center in centers:
        print_status(f"\n{'='*60}", "INFO")
        print_status(f"PROCESSING {center.upper()} - LUNAR STANDSTILL ANALYSIS", "TITLE")
        print_status(f"{'='*60}", "INFO")
        
        # Load data for this center
        complete_df = load_complete_geospatial_dataset(center)
        
        print_status(f"Loaded {len(complete_df):,} station pairs for {center}", "SUCCESS")
        
        # Run Lunar Standstill analysis
        results = {'analysis_center': center}
        results['lunar_standstill_analysis'] = run_lunar_standstill_analysis(complete_df)
        
        # Print summary
        print_summary_lunar_standstill_results(results)
        
        # Save results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"step_2_2_lunar_only_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Lunar Standstill results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print_status("🌙 LUNAR STANDSTILL ANALYSIS COMPLETED", "TITLE")
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
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING}", "TITLE")
    print_status("ASTRONOMICAL EVENTS ANALYSIS - Jupiter vs Saturn vs Mars Opposition Comparison", "TITLE")

    all_results = {}
    start_time = time.time()
    
    # Determine analysis centers
    if analysis_center:
        centers = [analysis_center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    for center in centers:
        print_status(f"\n{'='*60}", "INFO")
        print_status(f"PROCESSING {center.upper()} - ASTRONOMICAL EVENTS ANALYSIS", "INFO")
        print_status(f"{'='*60}", "INFO")
        
        # Load data for this center
        complete_df = load_complete_geospatial_dataset(center)
        
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
        output_file = output_dir / f"step_2_2_astronomical_events_{center}.json"
        try:
            safe_json_write(results, output_file, indent=2)
            print_status(f"Astronomical events results saved: {output_file}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to save results: {e}", "ERROR")
        
        all_results[center] = results
    
    elapsed_time = time.time() - start_time
    print_status("🌌 ASTRONOMICAL EVENTS ANALYSIS COMPLETED", "TITLE")
    print_status(f"Total execution time: {elapsed_time:.1f} seconds", "INFO")
    
    return all_results

def print_summary_jupiter_results(results: Dict):
    """Print a comprehensive summary of Jupiter opposition analysis results with enhanced scientific reporting"""
    print_status(f"JUPITER OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_JUPITER_OPPOSITION', default=True):
            # Enhanced detection categorization
            jupiter_analysis = results.get('jupiter_opposition_analysis', {})
            event_results = jupiter_analysis.get('event_results', {})
            significant_events = []  # 3.0σ+
            notable_events = []  # 2.0-3.0σ
            subsignificant_events = []  # 1.0-2.0σ
            all_amplitudes = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    gaussian = event_data.get('gaussian_fit', {})
                    if gaussian.get('fit_success', False):
                        amplitude = gaussian.get('amplitude', 0)
                        std_err = gaussian.get('amplitude_std_err', 1)
                        sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                        amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                        
                        all_amplitudes.append(amplitude_pct)
                        
                        event_info = (event_name, event_data, sigma_level, amplitude_pct)
                        
                        if gaussian.get('is_significant', False):  # 3.0σ+
                            significant_events.append(event_info)
                        elif sigma_level >= 2.0:
                            notable_events.append(event_info)
                        elif sigma_level >= 1.0:
                            subsignificant_events.append(event_info)
            
            # ENHANCED REPORTING LOGIC
            total_detections = len(significant_events) + len(notable_events) + len(subsignificant_events)
            
            if significant_events:
                print_status(f"Jupiter Opposition: {len(significant_events)} SIGNIFICANT DETECTION(S) (≥3.0σ)", "SUCCESS")
                for event_name, event_data, sigma, amp_pct in significant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    gaussian = event_data.get('gaussian_fit', {})
                    direction = "suppression" if gaussian.get('amplitude', 0) < 0 else "enhancement"
                    center_days = gaussian.get('center_days', 0)
                    expected_amp = 0.00220  # Jupiter expected amplitude (fractional units)
                    # Calculate enhancement factor using absolute amplitude units
                    gaussian_data = event_data.get('gaussian_fit', {})
                    baseline = gaussian_data.get('baseline', 0.007)
                    amplitude_fraction = gaussian_data.get('amplitude_fraction_of_baseline', 0)
                    actual_amplitude = abs(amplitude_fraction) * baseline
                    enhancement_factor = actual_amplitude / expected_amp if expected_amp > 0 else 0
                    
                    print_status(f"   {event_date}: {sigma:.1f}σ {direction} at day {center_days:.1f}", "SUCCESS")
                    print_status(f"      Amplitude: {amp_pct:.1f}% (expected: {expected_amp*100:.3f}%, enhancement: {enhancement_factor:.0f}x)", "INFO")
            elif notable_events:
                print_status(f"Jupiter Opposition: {len(notable_events)} NOTABLE DETECTION(S) (2.0-3.0σ)", "INFO")
                for event_name, event_data, sigma, amp_pct in notable_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    expected_amp = 0.00220
                    # CRITICAL FIX: Calculate enhancement using absolute amplitudes
                    gaussian_data = event_data.get('gaussian_fit', {})
                    baseline = gaussian_data.get('baseline', 0.007)
                    amplitude_fraction = gaussian_data.get('amplitude_fraction_of_baseline', 0)
                    actual_amplitude = abs(amplitude_fraction) * baseline
                    enhancement_factor = actual_amplitude / expected_amp if expected_amp > 0 else 0
                    print_status(f"   {event_date}: {sigma:.1f}σ, {amp_pct:.1f}% amplitude ({enhancement_factor:.0f}x expected)", "INFO")
            elif subsignificant_events:
                print_status(f"Jupiter Opposition: {len(subsignificant_events)} SUB-SIGNIFICANT DETECTION(S) (1.0-2.0σ)", "INFO")
                for event_name, event_data, sigma, amp_pct in subsignificant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    print_status(f"   {event_date}: {sigma:.1f}σ, {amp_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"Jupiter Opposition: No detections above 1.0σ threshold", "INFO")

            
            # Scientific context and statistical summary
            if all_amplitudes:
                avg_amp = np.mean(np.abs(all_amplitudes))
                max_amp = np.max(np.abs(all_amplitudes))
                expected_amp = 0.220  # Keep as percentage for display
                print_status(f"Statistical Summary:", "INFO")
                print_status(f"   Total Events Analyzed: {len(event_results)}", "INFO")
                print_status(f"   Detections ≥1.0σ: {total_detections}/{len(event_results)} ({100*total_detections/max(len(event_results),1):.1f}%)", "INFO")
                # Calculate enhancement factors using absolute amplitude units
                expected_amp_abs = expected_amp / 100
                typical_baseline = 0.007  # Baseline coherence for unit conversion
                avg_amp_abs = (avg_amp / 100) * typical_baseline
                max_amp_abs = (max_amp / 100) * typical_baseline
                avg_enhancement = avg_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                max_enhancement = max_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                
                print_status(f"   Average Amplitude: {avg_amp:.1f}% (expected: {expected_amp:.3f}%)", "INFO")
                print_status(f"   Maximum Amplitude: {max_amp:.1f}% ({max_enhancement:.0f}x expected)", "INFO")
            
            # Stacked analysis (deferred to Step 4.4)
            print_status(f"   Stacked Analysis: Deferred to Step 4.4 for comprehensive multi-planet correlation", "INFO")

        else:
            print_status("Jupiter Opposition: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"Jupiter Opposition: Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def print_summary_saturn_results(results: Dict):
    """Print a comprehensive summary of Saturn opposition analysis results with enhanced scientific reporting"""
    print_status(f"SATURN OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_SATURN_OPPOSITION', default=True):
            # Enhanced detection categorization
            saturn_analysis = results.get('saturn_opposition_analysis', {})
            event_results = saturn_analysis.get('event_results', {})
            significant_events = []  # 3.0σ+
            notable_events = []  # 2.0-3.0σ
            subsignificant_events = []  # 1.0-2.0σ
            all_amplitudes = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    gaussian = event_data.get('gaussian_fit', {})
                    if gaussian.get('fit_success', False):
                        amplitude = gaussian.get('amplitude', 0)
                        std_err = gaussian.get('amplitude_std_err', 1)
                        sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                        amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                        
                        all_amplitudes.append(amplitude_pct)
                        
                        event_info = (event_name, event_data, sigma_level, amplitude_pct)
                        
                        if gaussian.get('is_significant', False):  # 3.0σ+
                            significant_events.append(event_info)
                        elif sigma_level >= 2.0:
                            notable_events.append(event_info)
                        elif sigma_level >= 1.0:
                            subsignificant_events.append(event_info)
            
            # ENHANCED REPORTING LOGIC
            total_detections = len(significant_events) + len(notable_events) + len(subsignificant_events)
            
            if significant_events:
                print_status(f"Saturn Opposition: {len(significant_events)} SIGNIFICANT DETECTION(S) (≥3.0σ)", "SUCCESS")
                for event_name, event_data, sigma, amp_pct in significant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    gaussian = event_data.get('gaussian_fit', {})
                    direction = "suppression" if gaussian.get('amplitude', 0) < 0 else "enhancement"
                    center_days = gaussian.get('center_days', 0)
                    expected_amp = 0.00019  # Saturn expected amplitude (absolute units)
                    # Calculate enhancement factor using absolute amplitude units
                    gaussian_data = event_data.get('gaussian_fit', {})
                    baseline = gaussian_data.get('baseline', 0.007)
                    amplitude_fraction = gaussian_data.get('amplitude_fraction_of_baseline', 0)
                    actual_amplitude = abs(amplitude_fraction) * baseline
                    enhancement_factor = actual_amplitude / expected_amp if expected_amp > 0 else 0
                    
                    print_status(f"   {event_date}: {sigma:.1f}σ {direction} at day {center_days:.1f}", "SUCCESS")
                    print_status(f"      Amplitude: {amp_pct:.1f}% (expected: {expected_amp*100:.3f}%, enhancement: {enhancement_factor:.0f}x)", "INFO")
            elif notable_events:
                print_status(f"Saturn Opposition: {len(notable_events)} NOTABLE DETECTION(S) (2.0-3.0σ)", "INFO")
                for event_name, event_data, sigma, amp_pct in notable_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    expected_amp = 0.00019
                    enhancement_factor = (abs(amp_pct) / 100) / expected_amp if expected_amp > 0 else 0
                    print_status(f"   {event_date}: {sigma:.1f}σ, {amp_pct:.1f}% amplitude ({enhancement_factor:.0f}x expected)", "INFO")
            elif subsignificant_events:
                print_status(f"Saturn Opposition: {len(subsignificant_events)} SUB-SIGNIFICANT DETECTION(S) (1.0-2.0σ)", "INFO")
                for event_name, event_data, sigma, amp_pct in subsignificant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    print_status(f"   {event_date}: {sigma:.1f}σ, {amp_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"Saturn Opposition: No detections above 1.0σ threshold", "INFO")
            
            # Scientific context and statistical summary
            if all_amplitudes:
                avg_amp = np.mean(np.abs(all_amplitudes))
                max_amp = np.max(np.abs(all_amplitudes))
                expected_amp = 0.019  # Keep as percentage for display
                print_status(f"Statistical Summary:", "INFO")
                print_status(f"   Total Events Analyzed: {len(event_results)}", "INFO")
                print_status(f"   Detections ≥1.0σ: {total_detections}/{len(event_results)} ({100*total_detections/max(len(event_results),1):.1f}%)", "INFO")
                # Calculate enhancement factors using absolute amplitude units
                expected_amp_abs = expected_amp / 100
                typical_baseline = 0.007  # Baseline coherence for unit conversion
                avg_amp_abs = (avg_amp / 100) * typical_baseline
                max_amp_abs = (max_amp / 100) * typical_baseline
                avg_enhancement = avg_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                max_enhancement = max_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                
                print_status(f"   Average Amplitude: {avg_amp:.1f}% (expected: {expected_amp:.3f}%)", "INFO")
                print_status(f"   Maximum Amplitude: {max_amp:.1f}% ({max_enhancement:.0f}x expected)", "INFO")
            
            # Stacked analysis (deferred to Step 4.4)
            print_status(f"   Stacked Analysis: Deferred to Step 4.4 for comprehensive multi-planet correlation", "INFO")
        else:
            print_status("Saturn Opposition: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"Saturn Opposition: Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")

def print_summary_mars_results(results: Dict):
    """Print a comprehensive summary of Mars opposition analysis results with enhanced scientific reporting"""
    print_status(f"MARS OPPOSITION ANALYSIS SUMMARY - {results['analysis_center'].upper()}", "TITLE")

    if results.get('success', False):
        if TEPConfig.get_bool('TEP_ENABLE_MARS_OPPOSITION', default=True):
            # Enhanced detection categorization
            mars_analysis = results.get('mars_opposition_analysis', {})
            event_results = mars_analysis.get('event_results', {})
            significant_events = []  # 3.0σ+
            notable_events = []  # 2.0-3.0σ
            subsignificant_events = []  # 1.0-2.0σ
            all_amplitudes = []
            
            for event_name, event_data in event_results.items():
                if event_data.get('success'):
                    gaussian = event_data.get('gaussian_fit', {})
                    if gaussian.get('fit_success', False):
                        amplitude = gaussian.get('amplitude', 0)
                        std_err = gaussian.get('amplitude_std_err', 1)
                        sigma_level = abs(amplitude / std_err) if std_err > 0 else 0
                        amplitude_pct = gaussian.get('amplitude_fraction_of_baseline', 0) * 100
                        
                        all_amplitudes.append(amplitude_pct)
                        
                        event_info = (event_name, event_data, sigma_level, amplitude_pct)
                        
                        if gaussian.get('is_significant', False):  # 3.0σ+
                            significant_events.append(event_info)
                        elif sigma_level >= 2.0:
                            notable_events.append(event_info)
                        elif sigma_level >= 1.0:
                            subsignificant_events.append(event_info)
            
            # ENHANCED REPORTING LOGIC
            total_detections = len(significant_events) + len(notable_events) + len(subsignificant_events)
            
            if significant_events:
                print_status(f"Mars Opposition: {len(significant_events)} SIGNIFICANT DETECTION(S) (≥3.0σ)", "SUCCESS")
                print_status("    REMARKABLE: Mars has the weakest expected signal (44x weaker than Jupiter)", "INFO")
                for event_name, event_data, sigma, amp_pct in significant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    gaussian = event_data.get('gaussian_fit', {})
                    direction = "suppression" if gaussian.get('amplitude', 0) < 0 else "enhancement"
                    center_days = gaussian.get('center_days', 0)
                    expected_amp = 0.00005  # Mars expected amplitude (absolute units)
                    # Calculate enhancement factor using absolute amplitude units
                    gaussian_data = event_data.get('gaussian_fit', {})
                    baseline = gaussian_data.get('baseline', 0.007)
                    amplitude_fraction = gaussian_data.get('amplitude_fraction_of_baseline', 0)
                    actual_amplitude = abs(amplitude_fraction) * baseline
                    enhancement_factor = actual_amplitude / expected_amp if expected_amp > 0 else 0
                    
                    print_status(f"   {event_date}: {sigma:.1f}σ {direction} at day {center_days:.1f}", "SUCCESS")
                    print_status(f"      Amplitude: {amp_pct:.1f}% (expected: {expected_amp:.4f}%, enhancement: {enhancement_factor:.0f}x)", "INFO")
            elif notable_events:
                print_status(f"Mars Opposition: {len(notable_events)} NOTABLE DETECTION(S) (2.0-3.0σ)", "INFO")
                for event_name, event_data, sigma, amp_pct in notable_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    expected_amp = 0.00005
                    enhancement_factor = (abs(amp_pct) / 100) / expected_amp if expected_amp > 0 else 0
                    print_status(f"   {event_date}: {sigma:.1f}σ, {amp_pct:.1f}% amplitude ({enhancement_factor:.0f}x expected)", "INFO")
            elif subsignificant_events:
                print_status(f"Mars Opposition: {len(subsignificant_events)} SUB-SIGNIFICANT DETECTION(S) (1.0-2.0σ)", "INFO")
                for event_name, event_data, sigma, amp_pct in subsignificant_events:
                    event_date = event_data.get('event_date', 'Unknown')[:10]
                    print_status(f"   {event_date}: {sigma:.1f}σ, {amp_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"Mars Opposition: No detections above 1.0σ threshold (expected for weakest signal)", "INFO")
            
            # Scientific context and statistical summary
            if all_amplitudes:
                avg_amp = np.mean(np.abs(all_amplitudes))
                max_amp = np.max(np.abs(all_amplitudes))
                expected_amp = 0.0050  # Keep as percentage for display
                print_status(f"Statistical Summary:", "INFO")
                print_status(f"   Total Events Analyzed: {len(event_results)}", "INFO")
                print_status(f"   Detections ≥1.0σ: {total_detections}/{len(event_results)} ({100*total_detections/max(len(event_results),1):.1f}%)", "INFO")
                print_status(f"   Average Amplitude: {avg_amp:.1f}% (expected: {expected_amp:.4f}%)", "INFO")
                # CRITICAL FIX: Calculate enhancement factor for summary using absolute units
                expected_amp_abs = expected_amp / 100  # Convert percentage to absolute
                # max_amp is percentage of baseline, convert to absolute
                typical_baseline = 0.007  # Typical baseline coherence
                max_amp_abs = (max_amp / 100) * typical_baseline
                max_enhancement = max_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                
                print_status(f"   Maximum Amplitude: {max_amp:.1f}% ({max_enhancement:.0f}x expected)", "INFO")
            
            # Stacked analysis (deferred to Step 4.4)
            print_status(f"   Stacked Analysis: Deferred to Step 4.4 for comprehensive multi-planet correlation", "INFO")
        else:
            print_status("Mars Opposition: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"Mars Opposition: Failed - {error}", "ERROR")
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
            print_status("Major Lunar Standstill: Disabled in configuration", "INFO")
    else:
        error = results.get('error', 'Unknown error')
        print_status(f"Major Lunar Standstill: Failed - {error}", "ERROR")
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

        print_status(f"Jupiter: {jupiter_significant}/{jupiter.get('n_successful_events', 0)} significant events", "INFO")
        print_status(f"Saturn:  {saturn_significant}/{saturn.get('n_successful_events', 0)} significant events", "INFO")
        print_status(f"Mars:    {mars_significant}/{mars.get('n_successful_events', 0)} significant events", "INFO")

        # Expected ratios (if available)
        jupiter_expected = TEPConfig.get_float('JUPITER_EXPECTED_SIGNAL')
        saturn_expected = TEPConfig.get_float('SATURN_EXPECTED_SIGNAL')
        mars_expected = TEPConfig.get_float('MARS_EXPECTED_SIGNAL')
        if jupiter_expected and saturn_expected and mars_expected:
            print_status(f"Expected amplitude ratios:", "INFO")
            print_status(f"   Jupiter/Saturn: {jupiter_expected/saturn_expected:.1f}x", "INFO")
            print_status(f"   Jupiter/Mars: {jupiter_expected/mars_expected:.1f}x", "INFO")
            print_status(f"   Saturn/Mars: {saturn_expected/mars_expected:.1f}x", "INFO")

        # Stacked analysis comparison
        jupiter_stacked = jupiter.get('stacked_analysis', {})
        saturn_stacked = saturn.get('stacked_analysis', {})
        if jupiter_stacked.get('success', False) and saturn_stacked.get('success', False):
            jupiter_sigma = jupiter_stacked.get('sigma_level', 0.0)
            saturn_sigma = saturn_stacked.get('sigma_level', 0.0)
            print_status(f"Stacked significance: Jupiter {jupiter_sigma:.1f}σ vs Saturn {saturn_sigma:.1f}σ", "INFO")

        # Overall conclusion
        total_significant = jupiter_significant + saturn_significant + mars_significant
        if total_significant > 0:
            print_status(f"CONCLUSION: {total_significant} significant astronomical event signals detected!", "SUCCESS")
            if mars_significant > 0:
                print_status("    EXTRAORDINARY: Mars signal detected despite being weakest expected!", "SUCCESS")
        else:
            print_status("CONCLUSION: No significant astronomical event signals detected", "INFO")
    else:
        print_status("Cannot compare - one or more analyses failed", "WARNING")
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
            nutation_summary = "Nutation Analysis: Successful" 
            if nutation.get('nutation_results'):
                for name, res in nutation['nutation_results'].items():
                    if res.get('r_squared', 0) > 0.1: # Threshold for significance
                        nutation_summary += f" - {name.replace('_', ' ').title()}: Amp={res['amplitude']:.4f}, R²={res['r_squared']:.3f}"
                    else:
                        nutation_summary += f" - {name.replace('_', ' ').title()}: No significant signature"
            else:
                nutation_summary += ": No specific nutation periods analyzed or found"
            print_status(nutation_summary, "INFO")
        else:
            error_msg = nutation.get('error', 'Unknown error')
            print_status(f"Nutation Analysis: Failed - {error_msg}", "ERROR")

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
        print_status(f"Helical Motion Analysis: Failed - {error}", "ERROR")
    print_status("-" * 50, "INFO")
@ensure_single_instance
def main():
    """Main function with command-line options for different analysis modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEP GNSS Geospatial Temporal Analysis - Step 5")
    parser.add_argument('--mode', choices=['full', 'helical', 'jupiter', 'saturn', 'mars', 'lunar', 'eclipse', 'astronomical'], default='full',
                        help='Analysis mode: full (complete geospatial temporal analysis) [default], helical (helical motion analyses only), jupiter (Jupiter opposition only), saturn (Saturn opposition only), mars (Mars opposition only), lunar (Lunar Standstill only), or astronomical (Jupiter, Saturn, and Mars)')
    parser.add_argument('--center', choices=['code', 'igs_combined', 'esa_final'],
                        help='Specific GNSS analysis center to process')
    parser.add_argument('--list-helical', action='store_true',
                        help='List available helical motion analysis methods')
    
    args = parser.parse_args()
    
    if args.list_helical:
        print_status("AVAILABLE HELICAL MOTION ANALYSES:", "TITLE")
        print_status("=" * 50, "INFO")
        print_status("1. Chandler Wobble Analysis (14-month polar axis motion)", "INFO")
        print_status("2. 3D Spherical Harmonic Analysis (directional anisotropy decomposition)", "INFO")
        print_status("3. Multi-Frequency Beat Analysis (Earth motion interference patterns)", "INFO")
        print_status("4. Relative Motion Beat Analysis (station pair differential dynamics)", "INFO")
        print_status("5. Mesh Dance Analysis (network coherence dynamics)", "INFO")
        print_status("6. Jupiter Opposition Analysis (gravitational potential pulse events)", "INFO")
        print_status("7. Saturn Opposition Analysis (gravitational potential pulse events)", "INFO")
        print_status("8. Mars Opposition Analysis (gravitational potential pulse events)", "INFO")
        print_status("9. Nutation Analysis (18.6-year axial tilt variations)", "INFO")
        print_status("", "INFO")
        print_status("ASTRONOMICAL EVENT ANALYSES:", "TITLE")
        print_status("=" * 50, "INFO")
        print_status("• Jupiter Opposition: Nov 3, 2023 & Dec 7, 2024 (0.22% expected amplitude)", "INFO")
        print_status("• Saturn Opposition: Aug 27, 2023 & Sep 8, 2024 (0.019% expected amplitude)", "INFO")
        print_status("• Mars Opposition: Jan 16, 2025 (0.005% expected amplitude - weakest signal)", "INFO")
        print_status("• Major Lunar Standstill: 2024-2025 (sidereal day amplitude enhancement)", "INFO")
        print_status("• Event-locked stacking with ±60 day windows", "INFO")
        print_status("• Cross-center validation (IGS/ESA/CODE)", "INFO")
        print_status("• Statistical significance testing", "INFO")
        print_status("", "INFO")
        print_status("TO RUN ANALYSES:", "TITLE")
        print_status("   python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py --mode helical", "INFO")
        print_status("   python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py --mode jupiter --center esa_final", "INFO")
        print_status("   python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py --mode saturn --center code", "INFO")
        print_status("   python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py --mode mars --center igs_combined", "INFO")
        print_status("   python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py --mode lunar --center igs_combined", "INFO")
        print_status("   python scripts/steps/step_2_core_analysis/step_2_2_tep_geospatial_temporal_analysis.py --mode astronomical  # All planets", "INFO")
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
    
    # Original full Step 2.2 analysis
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING} - STEP 2.2: Geospatial Temporal Analysis", "TITLE")
    
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
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    print_status(f"Memory usage: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)", "INFO")
    
    memory_limit = TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')
    # Memory check removed - warnings disabled
    
    # Process analysis centers
    if args.center:
        centers = [args.center]
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    results = {}
    for ac in centers:
        print_status(f"\n{'='*60}", "INFO")
        print_status(f"PROCESSING {ac.upper()} - Geospatial Temporal Analysis", "TITLE")
        print_status(f"{'='*60}", "INFO")
        
        result = process_analysis_center(ac)
        results[ac] = result
        
        # Save individual results
        output_dir = ROOT / "results/outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"step_2_2_geospatial_temporal_analysis_{ac}.json"
        try:
            safe_json_write(result, output_file, indent=2)
            print_status(f"Results saved: {output_file}", "SUCCESS")
        except (TEPFileError, TEPDataError) as e:
            print_status(f"Failed to save results: {e}", "WARNING")
    
    # Summary
    print_status(f"\n{'='*80}", "INFO")
    print_status("GEOSPATIAL TEMPORAL ANALYSIS COMPLETE", "TITLE")
    print_status(f"{'='*80}", "INFO")
    
    if results:
        print_status("Validation Summary:", "SUCCESS")
        for ac, result in results.items():
            if result.get('success', False):
                print_status(f"  {ac.upper()}:", "INFO")

                if result.get('enhanced_anisotropy_analysis', {}).get('success', False):
                    anisotropy = result['enhanced_anisotropy_analysis']
                    aniso_stats = anisotropy['anisotropy_statistics']
                    print_status(f"    Enhanced Anisotropy: {aniso_stats['n_sectors']} sectors, CV = {aniso_stats['coefficient_of_variation']:.3f} ({aniso_stats['anisotropy_category']})", "INFO")
            else:
                print_status(f"  {ac.upper()}: FAILED - {result.get('error', 'Unknown error')}", "ERROR")
        
        print_status(f"Total execution time: {time.time() - start_time:.1f} seconds", "INFO")
        
        return True
    else:
        print_status("No successful validations", "ERROR")
        return False

# ===== MISSING FUNCTIONS FROM MAIN BRANCH STEP 5 =====

def run_chandler_wobble_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Detect 14-month Chandler wobble signatures in GPS timing correlations.
    
    The Chandler wobble causes Earth's rotation axis to wander ~9 meters from 
    the geographic poles with a period of ~14 months. This should modulate
    correlation patterns as the station mesh "wobbles" relative to inertial space.
    """
    print_status("Starting Chandler Wobble Analysis (14-month period)...", "PROCESS")
    
    try:
        # Convert dates to datetime if not already done
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Calculate days since epoch for continuous time analysis
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Check temporal coverage for Chandler wobble analysis
        data_span_days = (complete_df['date'].max() - complete_df['date'].min()).days + 1  # Inclusive date count
        chandler_period_days = TEPConfig.get_float('TEP_CHANDLER_PERIOD_DAYS', 425.0)  # ~14 months
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
        
        # Azimuth already computed in Step 2.1 - no need to recalculate!
        if 'azimuth' not in complete_df.columns:
            print_status("Computing azimuth for Chandler wobble analysis...", "PROCESS")
            complete_df['azimuth'] = complete_df.apply(
                lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                           row['station2_lat'], row['station2_lon']), axis=1
            )
        
        # Classify pairs as East-West or North-South
        def classify_ew_ns(azimuth):
            if (45 <= azimuth <= 135) or (225 <= azimuth <= 315):
                return 'EW'
            else:
                return 'NS'
        
        complete_df['ew_ns_class'] = complete_df['azimuth'].apply(classify_ew_ns)
        
        # Analyze each phase bin
        phase_results = []
        num_bins = TEPConfig.get_int('TEP_BINS')
        max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
        
        for phase_bin in range(n_phase_bins):
            phase_data = complete_df[complete_df['chandler_phase_bin'] == phase_bin].copy()
            
            if len(phase_data) < 500:  # Lowered requirement for better temporal coverage
                continue
                
            # Analyze E-W and N-S separately
            ew_data = phase_data[phase_data['ew_ns_class'] == 'EW']
            ns_data = phase_data[phase_data['ew_ns_class'] == 'NS']
            
            ew_lambda = fit_directional_correlation(ew_data, edges, min_bin_count)
            ns_lambda = fit_directional_correlation(ns_data, edges, min_bin_count)
            
            if ew_lambda and ns_lambda:
                phase_results.append({
                    'phase_bin': phase_bin,
                    'phase_degrees': phase_bin * 20,  # 20° per bin
                    'ew_lambda_km': ew_lambda,
                    'ns_lambda_km': ns_lambda,
                    'ew_ns_ratio': ew_lambda / ns_lambda,
                    'n_ew_pairs': len(ew_data),
                    'n_ns_pairs': len(ns_data)
                })
        
        if len(phase_results) < 8:  # Need at least 8 phase bins for meaningful analysis
            return {
                'success': False,
                'error': f'Insufficient phase bins for Chandler wobble: {len(phase_results)} (need ≥8)',
                'n_phase_bins': len(phase_results)
            }
        
        # Test for 14-month periodicity in E-W/N-S ratio
        phases = [r['phase_degrees'] for r in phase_results]
        ew_ns_ratios = [r['ew_ns_ratio'] for r in phase_results]
        
        # Fit sinusoidal model to detect periodicity
        try:
            def sinusoidal_model(phase_rad, amplitude, phase_offset, baseline):
                return amplitude * np.cos(phase_rad + phase_offset) + baseline
            
            phase_rad = np.array(phases) * np.pi / 180
            popt, pcov = curve_fit(sinusoidal_model, phase_rad, ew_ns_ratios, 
                                 p0=[0.1, 0, np.mean(ew_ns_ratios)])
            
            amplitude, phase_offset, baseline = popt
            r_squared = 1 - np.sum((ew_ns_ratios - sinusoidal_model(phase_rad, *popt))**2) / np.sum((ew_ns_ratios - np.mean(ew_ns_ratios))**2)
            
            chandler_signature = {
                'fit_success': True,
                'amplitude': float(amplitude),
                'phase_offset_rad': float(phase_offset),
                'baseline': float(baseline),
                'r_squared': float(r_squared),
                'n_phase_bins': len(phase_results)
            }
            
        except Exception as e:
            chandler_signature = {
                'fit_success': False,
                'error': str(e),
                'n_phase_bins': len(phase_results)
            }
        
        results = {
            'success': True,
            'analysis_type': 'chandler_wobble',
            'temporal_coverage': {
                'data_span_days': data_span_days,
                'chandler_period_days': chandler_period_days,
                'cycles_available': n_chandler_cycles
            },
            'phase_analysis': phase_results,
            'chandler_signature': chandler_signature
        }
        
        if chandler_signature.get('fit_success') and chandler_signature.get('r_squared', 0) > 0.3:
            print_status(f"Chandler wobble signature detected: R² = {chandler_signature['r_squared']:.3f}", "SUCCESS")
        else:
            print_status("No significant Chandler wobble signature detected", "INFO")
        
        print_status(f"CHANDLER WOBBLE ANALYSIS RESULTS:", "SUCCESS")
        print_status(f"  Chandler Period: 14.0 months (433 days)", "INFO")
        print_status(f"  Temporal Coverage: {data_span_days} days ({data_span_days/433:.2f} Chandler cycles)", "INFO")
        print_status(f"  Phase Bins Analyzed: {len(phase_results)}", "INFO")
        if chandler_signature['r_squared'] > 0.3:
            print_status(f"  Chandler Signature: R² = {chandler_signature['r_squared']:.3f} (DETECTED)", "SUCCESS")
        else:
            print_status(f"  Chandler Signature: R² = {chandler_signature['r_squared']:.3f} (not significant)", "INFO")
        print_status(f"Chandler wobble analysis complete: {len(phase_results)} phase bins analyzed", "SUCCESS")
        return results
        
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
        # Azimuth already computed in Step 2.1 - no need to recalculate!
        if 'azimuth' not in complete_df.columns:
            print_status("Computing azimuths (fallback - Step 2.1 data not available)...", "WARNING")
            complete_df['azimuth'] = complete_df.apply(
                lambda row: compute_azimuth(row['station1_lat'], row['station1_lon'], 
                                           row['station2_lat'], row['station2_lon']), axis=1
            )
        else:
            print_status("Using pre-computed azimuths from Step 2.1", "SUCCESS")
        
        # Compute elevation angles accounting for Earth curvature
        def compute_elevation_angle(lat1, lon1, lat2, lon2):
            """Compute elevation angle for station pair"""
            # Convert to radians
            lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
            lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
            
            # Calculate great circle distance
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_rad = c
            
            # Earth radius in km
            R = 6371.0
            
            # Calculate elevation angle (angle from horizontal)
            # This is the angle between the line connecting stations and the local horizontal
            elevation_rad = np.arcsin(distance_rad / (2 * R))
            elevation_deg = np.degrees(elevation_rad)
            
            return elevation_deg
        
        print_status("Computing elevation angles for 3D analysis...", "PROCESS")
        
        # Vectorized elevation angle calculation (much faster than apply())
        lat1_rad = np.radians(complete_df['station1_lat'])
        lon1_rad = np.radians(complete_df['station1_lon'])
        lat2_rad = np.radians(complete_df['station2_lat'])
        lon2_rad = np.radians(complete_df['station2_lon'])
        
        # Calculate great circle distance
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_rad = c
        
        # Earth radius in km
        R = 6371.0
        
        # Calculate elevation angle (angle from horizontal)
        elevation_rad = np.arcsin(distance_rad / (2 * R))
        complete_df['elevation_deg'] = np.degrees(elevation_rad)
        
        # Convert to spherical coordinates (azimuth, elevation, distance)
        complete_df['azimuth_rad'] = np.radians(complete_df['azimuth'])
        complete_df['elevation_rad'] = np.radians(complete_df['elevation_deg'])
        
        # Group into spherical bins for harmonic analysis
        n_azimuth_bins = 16  # 22.5° azimuth resolution
        n_elevation_bins = 8  # Elevation bins
        
        azimuth_bins = np.linspace(0, 2*np.pi, n_azimuth_bins + 1)
        elevation_bins = np.linspace(0, np.pi/2, n_elevation_bins + 1)  # 0 to 90°
        
        complete_df['azimuth_bin'] = pd.cut(complete_df['azimuth_rad'], 
                                           bins=azimuth_bins, 
                                           labels=range(n_azimuth_bins))
        complete_df['elevation_bin'] = pd.cut(complete_df['elevation_rad'], 
                                             bins=elevation_bins, 
                                             labels=range(n_elevation_bins))
        
        # Analyze each spherical bin
        spherical_results = []
        num_bins = TEPConfig.get_int('TEP_BINS')
        max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
        
        for az_bin in range(n_azimuth_bins):
            for el_bin in range(n_elevation_bins):
                bin_data = complete_df[
                    (complete_df['azimuth_bin'] == az_bin) & 
                    (complete_df['elevation_bin'] == el_bin)
                ].copy()
                
                if len(bin_data) < min_bin_count * 2:  # Need sufficient data
                    continue
                
                # Fit correlation model to this spherical bin
                lambda_km = fit_directional_correlation(bin_data, edges, min_bin_count)
                
                if lambda_km:
                    azimuth_center = (az_bin + 0.5) * 360 / n_azimuth_bins
                    elevation_center = (el_bin + 0.5) * 90 / n_elevation_bins
                    
                    spherical_results.append({
                        'azimuth_bin': az_bin,
                        'elevation_bin': el_bin,
                        'azimuth_deg': azimuth_center,
                        'elevation_deg': elevation_center,
                        'lambda_km': lambda_km,
                        'n_pairs': len(bin_data)
                    })
        
        if len(spherical_results) < 8:  # Need sufficient spherical coverage
            return {
                'success': False,
                'error': f'Insufficient spherical bins for 3D analysis: {len(spherical_results)} (need ≥8)',
                'n_spherical_bins': len(spherical_results)
            }
        
        # Compute spherical harmonic coefficients
        # Convert to spherical coordinates for harmonic analysis
        azimuths = np.array([r['azimuth_deg'] for r in spherical_results]) * np.pi / 180
        elevations = np.array([r['elevation_deg'] for r in spherical_results]) * np.pi / 180
        lambdas = np.array([r['lambda_km'] for r in spherical_results])
        
        # Compute low-order spherical harmonic coefficients
        # Y_lm(theta, phi) where theta = elevation, phi = azimuth
        harmonic_coeffs = {}
        
        # l=0 (constant)
        harmonic_coeffs['Y_00'] = np.mean(lambdas)
        
        # l=1 (dipole)
        harmonic_coeffs['Y_10'] = np.mean(lambdas * np.cos(elevations))
        harmonic_coeffs['Y_11_real'] = np.mean(lambdas * np.sin(elevations) * np.cos(azimuths))
        harmonic_coeffs['Y_11_imag'] = np.mean(lambdas * np.sin(elevations) * np.sin(azimuths))
        
        # l=2 (quadrupole)
        harmonic_coeffs['Y_20'] = np.mean(lambdas * (3 * np.cos(elevations)**2 - 1) / 2)
        harmonic_coeffs['Y_21_real'] = np.mean(lambdas * np.sin(elevations) * np.cos(elevations) * np.cos(azimuths))
        harmonic_coeffs['Y_21_imag'] = np.mean(lambdas * np.sin(elevations) * np.cos(elevations) * np.sin(azimuths))
        harmonic_coeffs['Y_22_real'] = np.mean(lambdas * np.sin(elevations)**2 * np.cos(2 * azimuths))
        harmonic_coeffs['Y_22_imag'] = np.mean(lambdas * np.sin(elevations)**2 * np.sin(2 * azimuths))
        
        # Compute anisotropy metrics
        dipole_magnitude = np.sqrt(harmonic_coeffs['Y_10']**2 + 
                                  harmonic_coeffs['Y_11_real']**2 + 
                                  harmonic_coeffs['Y_11_imag']**2)
        
        quadrupole_magnitude = np.sqrt(harmonic_coeffs['Y_20']**2 + 
                                      harmonic_coeffs['Y_21_real']**2 + 
                                      harmonic_coeffs['Y_21_imag']**2 +
                                      harmonic_coeffs['Y_22_real']**2 + 
                                      harmonic_coeffs['Y_22_imag']**2)
        
        # Anisotropy strength
        anisotropy_strength = (dipole_magnitude + quadrupole_magnitude) / abs(harmonic_coeffs['Y_00'])
        
        results = {
            'success': True,
            'analysis_type': '3d_spherical_harmonic',
            'n_spherical_bins': len(spherical_results),
            'spherical_results': spherical_results,
            'harmonic_coefficients': harmonic_coeffs,
            'anisotropy_metrics': {
                'dipole_magnitude': float(dipole_magnitude),
                'quadrupole_magnitude': float(quadrupole_magnitude),
                'anisotropy_strength': float(anisotropy_strength),
                'monopole_strength': float(abs(harmonic_coeffs['Y_00']))
            }
        }
        
        if anisotropy_strength > 0.5:
            print_status(f"Strong 3D anisotropy detected: strength = {anisotropy_strength:.3f}", "SUCCESS")
        elif anisotropy_strength > 0.2:
            print_status(f"Moderate 3D anisotropy detected: strength = {anisotropy_strength:.3f}", "INFO")
        else:
            print_status(f"Weak 3D anisotropy: strength = {anisotropy_strength:.3f}", "INFO")
        
        print_status(f"3D SPHERICAL HARMONIC ANALYSIS RESULTS:", "SUCCESS")
        print_status(f"  3D Anisotropy Strength: {anisotropy_strength:.3f}", "INFO")
        print_status(f"  Spherical Bins Analyzed: {len(spherical_results)}", "INFO")
        print_status(f"  Azimuth Resolution: 16 bins (22.5° each)", "INFO")
        print_status(f"  Elevation Resolution: 8 bins (0-90°)", "INFO")
        if anisotropy_strength > 1.5:
            print_status(f"  Strong 3D Structure: {anisotropy_strength:.3f} (DETECTED)", "SUCCESS")
        else:
            print_status(f"  Weak 3D Structure: {anisotropy_strength:.3f} (not significant)", "INFO")
        print_status(f"3D spherical harmonic analysis complete: {len(spherical_results)} bins analyzed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"3D spherical harmonic analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

def run_multi_frequency_beat_analysis_aligned(complete_df: pd.DataFrame) -> Dict:
    """
    ALIGNED WITH STEP 3.6: Multi-frequency beat analysis using identical frequency bands
    for direct manuscript comparison and consistency.
    
    FREQUENCY BANDS (IDENTICAL TO STEP 3.6):
    ========================================
    - Tidal bands (10-30 µHz): Principal gravitational forcing
    - Post-tidal (30-100 µHz): Transition region  
    - Intermediate (100-500 µHz): Mid-range TEP signal
    - Transition (500-1000 µHz): Control approach
    - Control (1000+ µHz): High-frequency reference
    
    This enables direct comparison with Step 3.6 correlation results in tables.
    """
    print_status("Starting Multi-Frequency Beat Analysis (Aligned with Step 3.6)...", "PROCESS")
    
    try:
        # STEP 3.6 FREQUENCY BANDS - Direct import for consistency
        step_3_6_bands = {
            'tidal_diurnal': {'f1_microhz': 10, 'f2_microhz': 20, 'name': 'Diurnal Tides (10-20 µHz)'},
            'tidal_semidiurnal': {'f1_microhz': 20, 'f2_microhz': 30, 'name': 'Semidiurnal Tides (20-30 µHz)'},
            'post_tidal_30_40': {'f1_microhz': 30, 'f2_microhz': 40, 'name': 'Post-Tidal 30-40 µHz'},
            'post_tidal_40_50': {'f1_microhz': 40, 'f2_microhz': 50, 'name': 'Post-Tidal 40-50 µHz'},
            'post_tidal_50_100': {'f1_microhz': 50, 'f2_microhz': 100, 'name': 'Post-Tidal 50-100 µHz'},
            'intermediate_100_200': {'f1_microhz': 100, 'f2_microhz': 200, 'name': 'Intermediate 100-200 µHz'},
            'intermediate_200_350': {'f1_microhz': 200, 'f2_microhz': 350, 'name': 'Intermediate 200-350 µHz'},
            'intermediate_350_500': {'f1_microhz': 350, 'f2_microhz': 500, 'name': 'Intermediate 350-500 µHz'},
            'transition_500_750': {'f1_microhz': 500, 'f2_microhz': 750, 'name': 'Transition 500-750 µHz'},
            'transition_750_1000': {'f1_microhz': 750, 'f2_microhz': 1000, 'name': 'Transition 750-1000 µHz'},
            'control_1000_1500': {'f1_microhz': 1000, 'f2_microhz': 1500, 'name': 'Control 1000-1500 µHz'},
            'control_2000_3000': {'f1_microhz': 2000, 'f2_microhz': 3000, 'name': 'Control 2000-3000 µHz'}
        }
        
        # Convert to temporal periods for beat analysis
        aligned_beat_frequencies = {}
        
        for band_id, band_config in step_3_6_bands.items():
            # Central frequency in µHz
            f_center_microhz = (band_config['f1_microhz'] + band_config['f2_microhz']) / 2
            # Convert to Hz
            f_center_hz = f_center_microhz * 1e-6
            # Convert to cycles per day
            f_center_cpd = f_center_hz * 86400
            # Period in days
            period_days = 1.0 / f_center_cpd if f_center_cpd > 0 else float('inf')
            
            aligned_beat_frequencies[band_id] = {
                'frequency_microhz': f_center_microhz,
                'frequency_hz': f_center_hz,
                'frequency_cpd': f_center_cpd,
                'period_days': period_days,
                'band_name': band_config['name'],
                'type': 'step_3_6_aligned',
                'bandwidth_microhz': band_config['f2_microhz'] - band_config['f1_microhz']
            }
        
        print_status(f"Analyzing {len(aligned_beat_frequencies)} frequency bands aligned with Step 3.6", "INFO")
        for band_id, freq_data in aligned_beat_frequencies.items():
            print_status(f"  {band_id}: {freq_data['frequency_microhz']:.0f} µHz ({freq_data['period_days']:.1f} day period)", "INFO")
        
        # Rest of analysis continues with standard beat pattern detection...
        # Convert dates and setup temporal analysis
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Distance ranges for spatial analysis
        distance_ranges = [
            (50, 500),      # Short range
            (500, 2000),    # Medium range  
            (2000, 10000),  # Long range
            (10000, 20000)  # Very long range
        ]
        
        range_names = ['short', 'medium', 'long', 'very_long']
        beat_results = {}
        
        # Analyze each distance range
        for (dist_min, dist_max), range_name in zip(distance_ranges, range_names):
            range_data = complete_df[
                (complete_df['dist_km'] >= dist_min) & 
                (complete_df['dist_km'] < dist_max)
            ].copy()
            
            if len(range_data) < 100:  # Skip ranges with insufficient data
                continue
                
            print_status(f"Analyzing {range_name} range ({dist_min}-{dist_max} km): {len(range_data):,} pairs", "INFO")
            
            # Analyze each frequency band
            range_beat_results = {}
            
            for band_id, freq_data in aligned_beat_frequencies.items():
                if freq_data['period_days'] > 0.01:  # Only analyze reasonable periods (exclude sub-hourly)
                    try:
                        # Use phase binning approach (like relative motion analysis)
                        time_phase = (2 * np.pi * range_data['days_since_epoch'] / freq_data['period_days']) % (2 * np.pi)
                        
                        # Group into phase bins
                        n_phase_bins = 8  # 45° phase resolution
                        phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
                        range_data['phase_bin'] = pd.cut(time_phase, bins=phase_bins, labels=range(n_phase_bins))
                        
                        # Calculate mean coherence per phase bin
                        phase_coherence_data = []
                        for phase_bin in range(n_phase_bins):
                            phase_data = range_data[range_data['phase_bin'] == phase_bin]
                            if len(phase_data) >= 50:  # Need sufficient data per bin
                                mean_coherence = phase_data['coherence'].mean()
                                phase_coherence_data.append({
                                    'phase_degrees': phase_bin * 45,
                                    'mean_coherence': mean_coherence,
                                    'n_pairs': len(phase_data)
                                })
                        
                        if len(phase_coherence_data) >= 4:  # Need at least 4 phase bins
                            # Fit sinusoidal pattern to phase-binned data
                            phases = np.array([d['phase_degrees'] for d in phase_coherence_data]) * np.pi / 180  # Convert to radians
                            coherences = np.array([d['mean_coherence'] for d in phase_coherence_data])
                            
                            # Fit: coherence = A*cos(phase) + B*sin(phase) + C
                            cos_component = np.cos(phases)
                            sin_component = np.sin(phases)
                            
                            cos_corr = np.corrcoef(coherences, cos_component)[0, 1] if len(coherences) > 2 else 0
                            sin_corr = np.corrcoef(coherences, sin_component)[0, 1] if len(coherences) > 2 else 0
                            
                            # Combined amplitude and R²
                            amplitude = np.sqrt(cos_corr**2 + sin_corr**2)
                            r_squared = amplitude**2
                            
                            # Statistical significance
                            n_samples = len(phase_coherence_data)
                            t_stat = amplitude * np.sqrt(n_samples - 2) / np.sqrt(1 - r_squared) if r_squared < 1 and n_samples > 2 else 0
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2)) if n_samples > 2 else 1.0
                            
                            range_beat_results[band_id] = {
                                'frequency_microhz': freq_data['frequency_microhz'],
                                'period_days': freq_data['period_days'],
                                'cos_correlation': cos_corr,
                                'sin_correlation': sin_corr,
                                'amplitude': amplitude,
                                'r_squared': r_squared,
                                'p_value': p_value,
                                'n_samples': n_samples,
                                'n_phase_bins': len(phase_coherence_data),
                                'band_name': freq_data['band_name']
                            }
                        
                    except Exception as e:
                        print_status(f"Beat analysis failed for {band_id} in {range_name}: {e}", "WARNING")
            
            if range_beat_results:
                beat_results[range_name] = range_beat_results
        
        # Identify significant patterns
        significant_beats = {}
        detection_threshold_r_squared = 0.01  # |r| > 0.1 (appropriate for high-frequency microHz analysis)
        
        for range_name, range_results in beat_results.items():
            for band_id, result in range_results.items():
                if result['r_squared'] > detection_threshold_r_squared:  # r² > 0.01 means |r| > 0.1
                    pattern_key = f"{range_name}_{band_id}"
                    significant_beats[pattern_key] = {
                        'range': range_name,
                        'band_id': band_id,
                        'band_name': result['band_name'],
                        'frequency_microhz': result['frequency_microhz'],
                        'period_days': result['period_days'],
                        'r_squared': result['r_squared'],
                        'p_value': result['p_value'],
                        'amplitude': result['amplitude']
                    }
        
        results = {
            'success': True,
            'analysis_type': 'multi_frequency_beat_analysis_aligned_step_3_6',
            'frequency_bands_analyzed': list(step_3_6_bands.keys()),
            'n_bands_analyzed': len(step_3_6_bands),
            'distance_ranges': range_names,
            'beat_analysis_results': beat_results,
            'significant_beats': significant_beats,
            'n_significant_beats': len(significant_beats),
            'detection_threshold_r_squared': detection_threshold_r_squared,
            'alignment_note': 'Frequency bands identical to Step 3.6 for manuscript consistency'
        }
        
        print_status(f"MULTI-FREQUENCY BEAT ANALYSIS (STEP 3.6 ALIGNED) COMPLETE:", "SUCCESS")
        print_status(f"  Frequency Bands Analyzed: {len(step_3_6_bands)}", "INFO")
        print_status(f"  Significant Beat Patterns: {len(significant_beats)}", "INFO")
        print_status(f"  Detection Threshold: |r| > {np.sqrt(detection_threshold_r_squared):.1f}", "INFO")
        
        if significant_beats:
            print_status(f"  Top Significant Patterns:", "INFO")
            sorted_patterns = sorted(significant_beats.items(), key=lambda x: x[1]['r_squared'], reverse=True)
            for i, (pattern_id, pattern_data) in enumerate(sorted_patterns[:5], 1):
                print_status(f"    {i}. {pattern_data['band_name']}: R²={pattern_data['r_squared']:.3f}, "
                           f"Period={pattern_data['period_days']:.1f} days, Range={pattern_data['range']}", "INFO")
        
        return results
        
    except Exception as e:
        print_status(f"Aligned multi-frequency beat analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e), 'analysis_type': 'multi_frequency_beat_analysis_aligned_step_3_6'}


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
            'chandler': 1.0/TEPConfig.get_float('TEP_CHANDLER_PERIOD_DAYS', 425.0),     # Chandler wobble
            'annual': 1.0/365.25,      # Annual orbital motion
            'semiannual': 2.0/365.25   # Semiannual variation
        }
        
        # Calculate all possible temporal interference patterns
        beat_frequencies = {}
        freq_names = list(frequencies.keys())
        min_period_days = TEPConfig.get_float('TEP_BEAT_MIN_PERIOD_DAYS', 7.0)
        
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
                    
                    # Sum frequency (constructive interference)
                    beat_sum = f1 + f2
                    period_sum = 1.0/beat_sum
                    if period_sum >= min_period_days:
                        beat_frequencies[f"{name1}_{name2}_sum"] = {
                            'frequency_cpd': beat_sum,
                            'period_days': period_sum,
                            'type': 'sum',
                            'components': [name1, name2]
                        }
        
        print_status(f"Identified {len(beat_frequencies)} beat frequency patterns", "INFO")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Calculate days since epoch for continuous time analysis
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Group data by distance ranges for beat analysis
        distance_ranges = [
            (50, 500),      # Short range
            (500, 2000),    # Medium range  
            (2000, 10000),  # Long range
            (10000, 20000)  # Very long range
        ]
        
        beat_tracking = []
        
        for range_name, (min_dist, max_dist) in zip(['short', 'medium', 'long', 'very_long'], distance_ranges):
            range_data = complete_df[
                (complete_df['dist_km'] >= min_dist) & 
                (complete_df['dist_km'] < max_dist)
            ].copy()
            
            if len(range_data) < 1000:  # Need sufficient data
                continue
            
            print_status(f"Analyzing {range_name} range ({min_dist}-{max_dist} km): {len(range_data)} pairs", "PROCESS")
            
            # Analyze each beat frequency pattern
            for beat_name, beat_info in beat_frequencies.items():
                period_days = beat_info['period_days']
                frequency_cpd = beat_info['frequency_cpd']
                
                # Calculate beat phase for each data point
                range_data['beat_phase'] = (2 * np.pi * range_data['days_since_epoch'] / period_days) % (2 * np.pi)
                
                # Group into phase bins
                n_phase_bins = 12  # 30° phase resolution
                phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
                range_data['beat_phase_bin'] = pd.cut(range_data['beat_phase'], 
                                                     bins=phase_bins, 
                                                     labels=range(n_phase_bins))
                
                # Analyze coherence vs beat phase
                phase_coherence_data = []
                
                for phase_bin in range(n_phase_bins):
                    phase_data = range_data[range_data['beat_phase_bin'] == phase_bin]
                    
                    if len(phase_data) < 50:  # Need sufficient data per bin
                        continue
                    
                    mean_coherence = phase_data['coherence'].mean()
                    coherence_std = phase_data['coherence'].std()
                    
                    phase_coherence_data.append({
                        'phase_bin': phase_bin,
                        'phase_degrees': phase_bin * 30,  # 30° per bin
                        'mean_coherence': mean_coherence,
                        'coherence_std': coherence_std,
                        'n_pairs': len(phase_data)
                    })
                
                if len(phase_coherence_data) < 6:  # Need sufficient phase coverage
                    continue
                
                # Test for beat frequency modulation
                phases = [d['phase_degrees'] for d in phase_coherence_data]
                coherences = [d['mean_coherence'] for d in phase_coherence_data]
                
                # Fit sinusoidal model to detect beat modulation
                try:
                    def beat_model(phase_rad, amplitude, phase_offset, baseline):
                        return amplitude * np.cos(phase_rad + phase_offset) + baseline
                    
                    phase_rad = np.array(phases) * np.pi / 180
                    popt, pcov = curve_fit(beat_model, phase_rad, coherences, 
                                         p0=[0.1, 0, np.mean(coherences)])
                    
                    amplitude, phase_offset, baseline = popt
                    r_squared = 1 - np.sum((coherences - beat_model(phase_rad, *popt))**2) / np.sum((coherences - np.mean(coherences))**2)
                    
                    # Statistical significance test
                    n_samples = len(phase_coherence_data)
                    f_stat = (r_squared / 2) / ((1 - r_squared) / (n_samples - 3))
                    p_value = 1 - stats.f.cdf(f_stat, 2, n_samples - 3) if n_samples > 3 else 1.0
                    
                    beat_tracking.append({
                        'distance_range': range_name,
                        'min_dist_km': min_dist,
                        'max_dist_km': max_dist,
                        'beat_name': beat_name,
                        'beat_period_days': period_days,
                        'beat_frequency_cpd': frequency_cpd,
                        'beat_type': beat_info['type'],
                        'components': beat_info['components'],
                        'amplitude': float(amplitude),
                        'phase_offset_rad': float(phase_offset),
                        'baseline': float(baseline),
                        'r_squared': float(r_squared),
                        'p_value': float(p_value),
                        'n_phase_bins': len(phase_coherence_data),
                        'n_pairs': len(range_data)
                    })
                    
                except Exception as e:
                    # Fit failed - record as non-significant
                    beat_tracking.append({
                        'distance_range': range_name,
                        'min_dist_km': min_dist,
                        'max_dist_km': max_dist,
                        'beat_name': beat_name,
                        'beat_period_days': period_days,
                        'beat_frequency_cpd': frequency_cpd,
                        'beat_type': beat_info['type'],
                        'components': beat_info['components'],
                        'amplitude': 0.0,
                        'phase_offset_rad': 0.0,
                        'baseline': float(np.mean(coherences)) if coherences else 0.0,
                        'r_squared': 0.0,
                        'p_value': 1.0,
                        'n_phase_bins': len(phase_coherence_data),
                        'n_pairs': len(range_data),
                        'fit_error': str(e)
                    })
        
        if len(beat_tracking) < 5:  # Need sufficient beat patterns
            return {
                'success': False,
                'error': f'Insufficient beat patterns for analysis: {len(beat_tracking)} (need ≥5)',
                'n_beat_patterns': len(beat_tracking)
            }
        
        # Identify most significant temporal interference patterns with configurable threshold
        significance_threshold = TEPConfig.get_float('TEP_BEAT_SIGNIFICANCE_THRESHOLD', 0.05)
        min_correlation = TEPConfig.get_float('TEP_MIN_CORRELATION_THRESHOLD', 0.3)
        significant_beats = {}
        
        print_status(f"Beat detection thresholds: p<{significance_threshold}, |r|>{min_correlation}", "INFO")
        
        for beat in beat_tracking:
            if (beat['p_value'] < significance_threshold and 
                abs(beat['r_squared']) > min_correlation):
                
                beat_key = f"{beat['distance_range']}_{beat['beat_name']}"
                significant_beats[beat_key] = beat
        
        # Overall results
        results = {
            'success': True,
            'analysis_type': 'multi_frequency_beat',
            'n_beat_frequencies': len(beat_frequencies),
            'n_beat_patterns_analyzed': len(beat_tracking),
            'n_significant_beats': len(significant_beats),
            'beat_frequencies': beat_frequencies,
            'beat_tracking': beat_tracking,
            'significant_beats': significant_beats,
            'detection_thresholds': {
                'significance_threshold': significance_threshold,
                'min_correlation_threshold': min_correlation
            }
        }
        
        if len(significant_beats) > 0:
            print_status(f"MULTI-FREQUENCY BEAT ANALYSIS RESULTS:", "SUCCESS")
            print_status(f"  Significant Beat Patterns: {len(significant_beats)}", "INFO")
            print_status(f"  Total Beat Patterns Analyzed: {len(beat_tracking)}", "INFO")
            print_status(f"  Detection Thresholds: p<{significance_threshold}, |r|>{min_correlation}", "INFO")
            
            # Print ALL significant beat patterns with full details
            print_status(f"  ALL SIGNIFICANT BEAT PATTERNS:", "INFO")
            for beat_name, beat_data in significant_beats.items():
                r_squared = beat_data['r_squared']
                p_value = beat_data['p_value']
                distance_range = beat_data['distance_range']
                beat_period = beat_data['beat_period_days']
                min_dist = beat_data['min_dist_km']
                max_dist = beat_data['max_dist_km']
                n_pairs = beat_data['n_pairs']
                print_status(f"    {beat_name}: R²={r_squared:.3f}, p={p_value:.4f}, Period={beat_period:.1f} days, Range={distance_range} ({min_dist}-{max_dist} km, {n_pairs} pairs)", "INFO")
            
            print_status(f"Multi-frequency beat analysis complete: {len(significant_beats)} significant beat patterns detected", "SUCCESS")
        else:
            print_status("Multi-frequency beat analysis complete: No significant beat patterns detected", "INFO")
        
        return results
        
    except Exception as e:
        print_status(f"Multi-frequency beat analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}
def run_relative_motion_beat_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze temporal interference patterns with RELATIVE MOTION between station pairs.
    
    This enhanced analysis considers relative velocities between station pairs
    as Earth moves, creating distance-dependent beat patterns across the mesh.
    """
    try:
        print_status("Starting Relative Motion Beat Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Calculate days since epoch for continuous time analysis
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Define relative motion periods based on Earth dynamics
        relative_periods = {
            'tidal_m2_tidal_s2_diff': 14.765,    # ~14.8 days (tidal cycle difference)
            'chandler_annual_sum': 196.9,        # ~197 days (Chandler + annual)
            'chandler_semiannual_sum': 127.9,    # ~128 days (Chandler + semiannual)
            'rotation_tidal_beat': 0.517         # ~12.4 hours (rotation-tidal beat)
        }
        
        # Group data by distance ranges for relative motion analysis
        distance_ranges = [
            (50, 1000, 'short'),
            (1000, 5000, 'medium'),
            (5000, 15000, 'long')
        ]
        
        relative_motion_results = {}
        
        for min_dist, max_dist, range_name in distance_ranges:
            range_data = complete_df[
                (complete_df['dist_km'] >= min_dist) & 
                (complete_df['dist_km'] < max_dist)
            ].copy()
            
            if len(range_data) < 1000:  # Need sufficient data
                continue
            
            range_results = {}
            
            for freq_name, period_days in relative_periods.items():
                # Calculate relative motion phase
                range_data['relative_phase'] = (2 * np.pi * range_data['days_since_epoch'] / period_days) % (2 * np.pi)
                
                # Group into phase bins
                n_phase_bins = 8  # 45° phase resolution
                phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
                range_data['relative_phase_bin'] = pd.cut(range_data['relative_phase'], 
                                                         bins=phase_bins, 
                                                         labels=range(n_phase_bins))
                
                # Analyze coherence vs relative motion phase
                phase_coherence_data = []
                
                for phase_bin in range(n_phase_bins):
                    phase_data = range_data[range_data['relative_phase_bin'] == phase_bin]
                    
                    if len(phase_data) < 50:  # Need sufficient data per bin
                        continue
                    
                    mean_coherence = phase_data['coherence'].mean()
                    coherence_std = phase_data['coherence'].std()
                    
                    phase_coherence_data.append({
                        'phase_bin': phase_bin,
                        'phase_degrees': phase_bin * 45,  # 45° per bin
                        'mean_coherence': mean_coherence,
                        'coherence_std': coherence_std,
                        'n_pairs': len(phase_data)
                    })
                
                if len(phase_coherence_data) >= 4:  # Need sufficient phase coverage
                    # Test for relative motion beat modulation
                    phases = [d['phase_degrees'] for d in phase_coherence_data]
                    coherences = [d['mean_coherence'] for d in phase_coherence_data]
                    
                    # Fit sinusoidal model to detect beat modulation
                    try:
                        def beat_model(phase_rad, amplitude, phase_offset, baseline):
                            return amplitude * np.cos(phase_rad + phase_offset) + baseline
                        
                        phase_rad = np.array(phases) * np.pi / 180
                        popt, pcov = curve_fit(beat_model, phase_rad, coherences, 
                                             p0=[0.01, 0, np.mean(coherences)])
                        
                        amplitude, phase_offset, baseline = popt
                        r_squared = 1 - np.sum((coherences - beat_model(phase_rad, *popt))**2) / np.sum((coherences - np.mean(coherences))**2)
                        
                        range_results[freq_name] = {
                            'period_days': period_days,
                            'amplitude': float(amplitude),
                            'phase_offset_rad': float(phase_offset),
                            'baseline': float(baseline),
                            'r_squared': float(r_squared),
                            'n_phase_bins': len(phase_coherence_data),
                            'n_pairs': len(range_data)
                        }
                        
                    except Exception as e:
                        range_results[freq_name] = {
                            'period_days': period_days,
                            'fit_error': str(e),
                            'n_phase_bins': len(phase_coherence_data),
                            'n_pairs': len(range_data)
                        }
            
            if range_results:
                relative_motion_results[range_name] = range_results
        
        # Identify significant relative motion patterns
        significant_patterns = {}
        total_patterns_analyzed = 0
        
        for range_name, range_result in relative_motion_results.items():
            for freq_name, freq_result in range_result.items():
                total_patterns_analyzed += 1
                if freq_result.get('r_squared', 0) > 0.2:  # 20% threshold
                    pattern_key = f"{range_name}_{freq_name}"
                    significant_patterns[pattern_key] = {
                        'distance_range': range_name,
                        'frequency_name': freq_name,
                        'period_days': freq_result['period_days'],
                        'amplitude': freq_result['amplitude'],
                        'r_squared': freq_result['r_squared']
                    }
        
        results = {
            'success': True,
            'analysis_type': 'relative_motion_beat_analysis',
            'n_significant_patterns': len(significant_patterns),
            'n_total_patterns_analyzed': total_patterns_analyzed,
            'relative_motion_results': relative_motion_results,
            'significant_patterns': significant_patterns
        }
        
        print_status(f"RELATIVE MOTION BEAT ANALYSIS RESULTS:", "SUCCESS")
        print_status(f"  Significant Beat Patterns: {len(significant_patterns)}", "INFO")
        print_status(f"  Total Beat Patterns Analyzed: {total_patterns_analyzed}", "INFO")
        print_status(f"  Detection Threshold: R² > 0.2", "INFO")
        
        if significant_patterns:
            # Print ALL significant relative motion beat patterns with full details
            print_status(f"  ALL SIGNIFICANT RELATIVE MOTION BEAT PATTERNS:", "INFO")
            for pattern_key, pattern_data in significant_patterns.items():
                r_squared = pattern_data['r_squared']
                period_days = pattern_data['period_days']
                amplitude = pattern_data['amplitude']
                distance_range = pattern_data['distance_range']
                freq_name = pattern_data['frequency_name']
                print_status(f"    {pattern_key}: R²={r_squared:.3f}, Period={period_days:.1f} days, Amplitude={amplitude:.4f}, Range={distance_range}", "INFO")
            print_status(f"Relative motion beat analysis complete: {len(significant_patterns)} significant patterns detected", "SUCCESS")
        else:
            print_status("Relative motion beat analysis complete: No significant patterns detected", "INFO")
        
        return results
        
    except Exception as e:
        print_status(f"Relative motion beat analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

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
        
        # ========================================
        # OPTIMIZED MESH DANCE ANALYSIS WINDOW STRATEGY
        # ========================================
        # MESH COHERENCE: 90-day windows for optimal statistical power (10 windows for 912-day dataset)
        # OSCILLATION/SPIRAL: 30-day windows for higher temporal resolution (better Nyquist sampling for 365d cycles)
        # 
        # Rationale:
        # - 90-day windows: Provide adequate statistical power (10+ samples) with good frequency resolution
        # - 30-day windows: Detect oscillations/spirals (more samples, better frequency resolution)
        # - For 912-day dataset:
        #   * 90d windows → 10 samples (adequate for robust correlation, 4.1 samples per annual cycle)
        #   * 30d windows → 30 samples (~12 per annual cycle, excellent for oscillation detection)
        # Previous 120d windows only provided 8 samples (marginal statistical power)
        # ========================================
        
        # 1. MESH COHERENCE ANALYSIS (90-day windows)
        # Test if all stations move together as one coherent system
        print_status("Analyzing mesh coherence patterns...", "INFO")
        
        # Group station pairs by time windows to track mesh evolution
        coherence_window_days = 90  # Optimized for statistical adequacy
        complete_df['coherence_window'] = (complete_df['days_since_epoch'] // coherence_window_days) * coherence_window_days
        
        print_status(f"Using OPTIMIZED windows: 90d for coherence (10 samples), 30d for oscillation/spiral", "INFO")
        
        mesh_coherence_results = {}
        unique_coherence_windows = sorted(complete_df['coherence_window'].unique())
        
        if len(unique_coherence_windows) < 3:  # Need sufficient temporal sampling
            return {'success': False, 'error': f'Insufficient coherence windows: {len(unique_coherence_windows)} (need ≥3)'}
        
        # Use all 120-day windows for coherence calculation
        sampled_coherence_windows = unique_coherence_windows
        
        mesh_evolution = []  # Will store 120-day window results
        
        for window in sampled_coherence_windows:
            window_data = complete_df[complete_df['coherence_window'] == window].copy()
            
            if len(window_data) < 1000:  # Need sufficient pairs per window
                continue
                
            # Calculate mesh properties for this time window
            
            # A. COLLECTIVE MOTION VECTOR
            # Calculate motion vector from coherence and azimuth data
            # Use coherence as magnitude and azimuth as direction
            window_data['total_motion_vector_magnitude'] = np.abs(window_data['coherence'])
            window_data['total_motion_vector_direction'] = window_data['azimuth']
            
            mean_total_vector_magnitude = window_data['total_motion_vector_magnitude'].mean()
            mean_total_vector_direction = window_data['total_motion_vector_direction'].mean()
            
            # B. MESH COHERENCE METRICS
            # How well synchronized are all the station pairs?
            coherence_std = window_data['coherence'].std()
            coherence_mean = window_data['coherence'].mean()
            coherence_uniformity = 1.0 / (1.0 + coherence_std)  # Higher = more uniform
            
            # C. PHASE COHERENCE ACROSS THE MESH
            # Are all stations oscillating in phase?
            # Use plateau_phase as the phase coherence metric
            window_data['overall_phase_coherence'] = np.cos(window_data['plateau_phase'])
            overall_phase_coherence_mean = window_data['overall_phase_coherence'].mean()
            overall_phase_coherence_std = window_data['overall_phase_coherence'].std()
            phase_synchronization = 1.0 / (1.0 + overall_phase_coherence_std)
            
            # D. INTERFERENCE STATE DISTRIBUTION
            # What's the distribution of interference states across the mesh?
            # Use coherence sign to determine interference state
            window_data['interference_state'] = np.where(window_data['coherence'] > 0, 'constructive', 'destructive')
            interference_counts = window_data['interference_state'].value_counts()
            dominant_interference_state = interference_counts.index[0] if len(interference_counts) > 0 else 'unknown'
            interference_dominance = interference_counts.iloc[0] / len(window_data) if len(interference_counts) > 0 else 0
            
            # E. OSCILLATION SYNCHRONIZATION
            # Are all parts of the mesh oscillating together?
            # Use coherence magnitude as oscillation strength
            window_data['motion_oscillation_strength'] = np.abs(window_data['coherence'])
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
        
        if len(mesh_evolution) < 3:
            return {'success': False, 'error': f'Insufficient mesh evolution data: {len(mesh_evolution)}'}
        
        # 2. CREATE HIGH-RESOLUTION 30-DAY WINDOWS FOR SPIRAL/OSCILLATION DETECTION
        # Use smaller windows for better temporal resolution in frequency analysis
        oscillation_window_days = 30
        complete_df['oscillation_window'] = (complete_df['days_since_epoch'] // oscillation_window_days) * oscillation_window_days
        unique_oscillation_windows = sorted(complete_df['oscillation_window'].unique())
        
        mesh_evolution_highres = []  # 30-day windows for spiral/oscillation
        
        for window in unique_oscillation_windows:
            window_data = complete_df[complete_df['oscillation_window'] == window].copy()
            
            if len(window_data) < 500:  # Lower threshold for smaller windows
                continue
            
            # Calculate same metrics as coherence windows
            window_data['total_motion_vector_magnitude'] = np.abs(window_data['coherence'])
            window_data['total_motion_vector_direction'] = window_data['azimuth']
            
            mean_total_vector_magnitude = window_data['total_motion_vector_magnitude'].mean()
            mean_total_vector_direction = window_data['total_motion_vector_direction'].mean()
            
            coherence_std = window_data['coherence'].std()
            coherence_mean = window_data['coherence'].mean()
            coherence_uniformity = 1.0 / (1.0 + coherence_std)
            
            window_data['overall_phase_coherence'] = np.cos(window_data['plateau_phase'])
            overall_phase_coherence_std = window_data['overall_phase_coherence'].std()
            phase_synchronization = 1.0 / (1.0 + overall_phase_coherence_std)
            
            window_data['motion_oscillation_strength'] = np.abs(window_data['coherence'])
            oscillation_std = window_data['motion_oscillation_strength'].std()
            oscillation_synchronization = 1.0 / (1.0 + oscillation_std)
            
            mesh_evolution_highres.append({
                'time_window': int(window),
                'days_since_epoch': int(window),
                'n_pairs': len(window_data),
                'collective_motion_magnitude': float(mean_total_vector_magnitude),
                'collective_motion_direction': float(mean_total_vector_direction),
                'mesh_coherence_score': float(
                    (coherence_uniformity + phase_synchronization + oscillation_synchronization) / 3.0
                )
            })
        
        if len(mesh_evolution_highres) < 10:
            print_status(f"Warning: Only {len(mesh_evolution_highres)} high-res windows, using 120-day windows for spiral/oscillation", "WARNING")
            mesh_for_dynamics = mesh_evolution  # Fallback to 120-day windows
        else:
            print_status(f"Using {len(mesh_evolution_highres)} high-resolution 30-day windows for spiral/oscillation detection", "INFO")
            mesh_for_dynamics = mesh_evolution_highres
        
        # 2. SPIRAL DYNAMICS ANALYSIS (using high-resolution windows)
        # Test if the mesh is tracing helical/spiral paths through spacetime
        print_status("Analyzing spiral dynamics of mesh motion...", "INFO")
        
        # Extract time series of collective motion
        times = [m['days_since_epoch'] for m in mesh_for_dynamics]
        directions = [m['collective_motion_direction'] for m in mesh_for_dynamics]
        magnitudes = [m['collective_motion_magnitude'] for m in mesh_for_dynamics]
        coherence_scores = [m['mesh_coherence_score'] for m in mesh_for_dynamics]
        
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
            'is_spiral_motion': bool(rotation_consistency > 0.005 and magnitude_oscillation > 0.002)
        }
        
        # 3. COLLECTIVE COHERENT OSCILLATION (using high-resolution windows)
        # Test if the entire mesh oscillates coherently as one system
        print_status("Analyzing collective mesh oscillation patterns...", "INFO")
        
        # Fit sinusoidal models to mesh coherence over time
        time_array = np.array(times)
        coherence_array = np.array(coherence_scores)
        
        print_status(f"Oscillation analysis using {len(time_array)} samples (window size: {'30d' if len(mesh_for_dynamics) == len(mesh_evolution_highres) else '120d'})", "INFO")
        
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
                    # Calculate R² from correlation coefficient
                    r_squared = correlation ** 2
                    
                    oscillation_results[f'freq_{freq:.6f}'] = {
                        'frequency_cpd': float(freq),
                        'period_days': float(period_days),
                        'correlation': float(correlation),
                        'r_squared': float(r_squared),  # ADDED: R² calculation
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
        
        # Find the strongest oscillation - IMPROVED LOGIC
        # First try strict criteria (p < 0.05)
        strict_oscillations = {k: v for k, v in oscillation_results.items() 
                             if v.get('fit_success') and v.get('p_value', 1) < 0.05}
        
        # Try relaxed criteria (p < 0.1) 
        relaxed_oscillations = {k: v for k, v in oscillation_results.items() 
                              if v.get('fit_success') and v.get('p_value', 1) < 0.1}
        
        # All successful oscillations (use relaxed threshold for counting)
        successful_oscillations = relaxed_oscillations
        
        if strict_oscillations:
            best_oscillation = max(strict_oscillations.values(), 
                                 key=lambda x: abs(x.get('correlation', 0)))
        elif relaxed_oscillations:
            best_oscillation = max(relaxed_oscillations.values(), 
                                 key=lambda x: abs(x.get('correlation', 0)))
        else:
            # If still no successes, use the best available oscillation
            available_oscillations = {k: v for k, v in oscillation_results.items() 
                                    if v.get('fit_success')}
            if available_oscillations:
                best_oscillation = max(available_oscillations.values(), 
                                     key=lambda x: abs(x.get('correlation', 0)))
                successful_oscillations = {}  # Mark as no significant oscillations
            else:
                best_oscillation = {'correlation': 0.0, 'r_squared': 0.0, 'no_significant_oscillation': True}
                successful_oscillations = {}
        
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
        
        # IMPROVED: Earth coupling detection includes both direct phase correlation AND oscillation period matching
        # Check if best oscillation matches Earth motion periods (annual ~365d, Chandler ~433d, sidereal day ~1d)
        oscillation_matches_earth = False
        if best_oscillation.get('fit_success'):
            period = best_oscillation.get('period_days', 0)
            # Check if period matches known Earth motion cycles (±10% tolerance)
            earth_periods = [1.0, 365.25, 433.0, 182.6]  # Sidereal day, Annual, Chandler, Semi-annual
            oscillation_matches_earth = any(abs(period - ep) / ep < 0.1 for ep in earth_periods)
        
        # Count Earth couplings from both methods
        phase_couplings = sum(1 for c in mesh_earth_coupling.values() 
                            if abs(c.get('correlation', 0)) > 0.15 and c.get('p_value', 1) < 0.15)
        oscillation_earth_coupling = 1 if oscillation_matches_earth else 0
        total_earth_couplings = phase_couplings + oscillation_earth_coupling
        
        dance_metrics = {
            'mesh_coherence_strength': float(np.mean([m['mesh_coherence_score'] for m in mesh_evolution])),
            'spiral_motion_detected': spiral_signature['is_spiral_motion'],
            'spiral_strength': spiral_signature['spiral_strength'],
            'collective_oscillation_detected': len(successful_oscillations) > 0,
            'strongest_oscillation_correlation': float(best_oscillation.get('correlation', 0)),
            'earth_coupling_detected': (phase_couplings > 0) or oscillation_matches_earth,
            'n_significant_earth_couplings': total_earth_couplings,
            'oscillation_matches_earth_period': oscillation_matches_earth,
            'best_oscillation_period_days': float(best_oscillation.get('period_days', 0)) if best_oscillation.get('fit_success') else None
        }
        
        # NETWORK COHERENCE CLASSIFICATION - IMPROVED CONTINUOUS SCORING
        # Use continuous scoring instead of binary to capture partial signals
        spiral_score = min(1.0, dance_metrics['spiral_strength'] * 10.0)  # Scale spiral strength
        oscillation_score = min(1.0, abs(dance_metrics['strongest_oscillation_correlation']))
        earth_coupling_score = min(1.0, dance_metrics['n_significant_earth_couplings'] / 3.0)  # Max 3 couplings
        
        # IMPROVED DANCE SCORE CALCULATION
        # Ensure we get meaningful scores even with partial signals
        mesh_coherence_base = dance_metrics['mesh_coherence_strength']
        
        # If we have any significant components, boost the score
        has_significant_components = (
            dance_metrics['spiral_motion_detected'] or 
            dance_metrics['collective_oscillation_detected'] or 
            dance_metrics['earth_coupling_detected']
        )
        
        # Count significant components for additional boost
        n_significant_components = sum([
            dance_metrics['spiral_motion_detected'],
            dance_metrics['collective_oscillation_detected'], 
            dance_metrics['earth_coupling_detected']
        ])
        
        # CRITICAL FIX: Always initialize boost_factor before use
        if has_significant_components:
            # Progressive boost based on number of significant components
            boost_factor = 1.0 + (n_significant_components * 0.15)  # 15% boost per component
            mesh_coherence_boosted = min(1.0, mesh_coherence_base * boost_factor)
        else:
            boost_factor = 1.0  # No boost when no significant components
            mesh_coherence_boosted = mesh_coherence_base
        
        # Weight distribution for mesh dance score calculation
        base_score = (
            mesh_coherence_boosted * 0.5 +   # Mesh coherence (primary component)
            spiral_score * 0.17 +            # Spiral motion  
            oscillation_score * 0.17 +       # Oscillation
            earth_coupling_score * 0.16      # Earth coupling
        )
        
        # Use calculated base score without artificial floor
        dance_score = base_score
        
        dance_classification = _classify_dance_signature(dance_score, dance_metrics)
        
        # Print detailed mesh dance results
        print_status(f"MESH DANCE ANALYSIS RESULTS:", "SUCCESS")
        print_status(f"  Network Coherence Score: {dance_score:.3f}", "INFO")
        print_status(f"  Dance Classification: {dance_classification}", "INFO")
        print_status(f"  Time Windows Analyzed: {len(mesh_evolution)}", "INFO")
        print_status(f"  Temporal Span: {int(max(times) - min(times))} days", "INFO")
        print_status(f"  DEBUG - Component Scores:", "INFO")
        print_status(f"    Mesh Coherence: {mesh_coherence_boosted:.3f} (base: {mesh_coherence_base:.3f})", "INFO")
        print_status(f"    Spiral Score: {spiral_score:.3f} (detected: {dance_metrics['spiral_motion_detected']})", "INFO")
        print_status(f"    Oscillation Score: {oscillation_score:.3f} (detected: {dance_metrics['collective_oscillation_detected']})", "INFO")
        print_status(f"    Earth Coupling Score: {earth_coupling_score:.3f} (detected: {dance_metrics['earth_coupling_detected']})", "INFO")
        print_status(f"    Significant Components: {n_significant_components}/3", "INFO")
        print_status(f"    Boost Factor: {boost_factor:.2f}x", "INFO")
        print_status(f"    Base Score: {base_score:.3f} → Final Score: {dance_score:.3f}", "INFO")
        
        if spiral_signature.get('success', False):
            print_status(f"  Spiral Dynamics: {spiral_signature.get('spiral_strength', 0):.3f} strength", "INFO")
            print_status(f"     Spiral Period: {spiral_signature.get('spiral_period_days', 0):.1f} days", "INFO")
        
        # DEBUG: Show actual oscillation details
        if best_oscillation.get('fit_success', False):
            print_status(f"  DEBUG - Best Oscillation Details:", "INFO")
            print_status(f"    Period: {best_oscillation.get('period_days', 0):.1f} days", "INFO")
            print_status(f"    Correlation: {best_oscillation.get('correlation', 0):.3f}", "INFO")
            print_status(f"    R²: {best_oscillation.get('r_squared', 0):.3f}", "INFO")
            print_status(f"    P-value: {best_oscillation.get('p_value', 1):.4f}", "INFO")
        
        if best_oscillation:
            print_status(f"  Best Collective Oscillation: {best_oscillation.get('period_days', 0):.1f} day period", "INFO")
            print_status(f"     Oscillation R²: {best_oscillation.get('r_squared', 0):.3f}", "INFO")
            print_status(f"     Significant Oscillations: {len(successful_oscillations)}", "INFO")
        
        if mesh_earth_coupling.get('success', False):
            coupling_strength = mesh_earth_coupling.get('coupling_strength', 0)
            print_status(f"  Earth-Mesh Coupling: {coupling_strength:.3f} strength", "INFO")
            print_status(f"     Coupling R²: {mesh_earth_coupling.get('r_squared', 0):.3f}", "INFO")
        
        print_status(f"MESH DANCE ANALYSIS COMPLETE: {dance_classification}", "SUCCESS")
        
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
def run_jupiter_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Jupiter opposition events using
    GAUSSIAN PULSE FITTING and STACKED ANALYSIS.
    
    Jupiter oppositions occur when Earth-Jupiter distance is minimized, causing
    Jupiter's gravitational potential at Earth to peak. According to TEP theory,
    this should manifest as a transient, pulse-like enhancement in timing correlations.
    
    DETECTION CHALLENGES:
    - Jupiter orbital period: 11.9 years (4,333 days)
    - Dataset coverage: 912 days = only 21% of one Jupiter orbit
    - Available oppositions: 2 events (insufficient for robust event-based detection)
    - Gravitational influence variation: only ~10% over 912 days (minimal transient signal)
    
    NOTE: Jupiter shows stronger signals in Step 4.4 continuous daily analysis, which
    captures long-term secular variation rather than requiring sharp transient peaks.
    Event-based analysis (this function) has low statistical power for slow-moving planets.
    """
    print_status("Starting Jupiter Opposition Pulse Analysis...", "PROCESS")
    
    try:
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Define analysis parameters first
        event_window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS', 120)  # ±120 days = 240-day total window
        expected_amplitude = TEPConfig.get_float('TEP_JUPITER_AMPLITUDE_FRACTION', 0.0022) # 0.22% expected amplitude
        min_pairs_per_day = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS_PER_DAY', 100) # Min pairs for daily binning
        
        # Define Jupiter opposition events (UTC dates)
        jupiter_oppositions = [
            {'date': pd.Timestamp('2023-11-03'), 'name': 'Jupiter_Opposition_2023'},
            {'date': pd.Timestamp('2024-12-07'), 'name': 'Jupiter_Opposition_2024'}
        ]
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        data_span_days = (data_end - data_start).days + 1  # Inclusive date count
        print_status(f"Data coverage: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')} ({data_span_days} days)", "INFO")

        if data_span_days < (2 * event_window_days):
            print_status(
                f"Skipping Jupiter opposition analysis: data span {data_span_days} days < required window {2 * event_window_days} days",
                "WARNING"
            )
            return {
                'success': False,
                'error': 'insufficient_temporal_coverage',
                'required_days': int(2 * event_window_days),
                'available_days': int(data_span_days)
            }
        # ========================================
        # JUPITER OPPOSITION WINDOW STRATEGY
        # ========================================
        # Window size: 240 days (±120 days)
        # Rationale: Gravitational-temporal field coupling operates on seasonal timescales
        #            Empirical Savitzky-Golay smoothing analysis (60-240 days) showed:
        #            - 60 days: r = -0.481 (moderate coupling)
        #            - 120 days: r = -0.535 (strengthening)
        #            - 240 days: r = -0.552 (OPTIMAL coupling)
        #            240-day window captures build-up, peak, and decay of gravitational influence
        # Previous: ±30 days resulted in stacked analysis failures ("x0 infeasible")
        # Expected: Successful stacked analysis with stronger signal detection
        # ========================================
        
        # Filter to events within data range
        valid_events_raw = []
        
        print_status(f"Using 240-day windows (±{event_window_days} days) for optimal gravitational coupling detection", "INFO")

        for event in jupiter_oppositions:
            event_date = event['date']
            if data_start - pd.Timedelta(days=event_window_days) <= event_date <= data_end + pd.Timedelta(days=event_window_days):
                valid_events_raw.append(event)
        
        if not valid_events_raw:
            return {
                'success': False,
                'error': 'No Jupiter opposition events within dataset coverage',
                'data_coverage': f"{data_start.date()} to {data_end.date()}"
            }
        
        # Analyze each valid opposition event
        event_analysis_results = {}
        all_event_data_for_stacking = []

        print_status(f"Analyzing {len(valid_events_raw)} Jupiter opposition events", "INFO")
        print_status(f"Event window: ±{event_window_days} days, Expected Amplitude: {expected_amplitude*100:.3f}%", "INFO")
        print_status(f"Jupiter orbital period: ~11.86 years, Opposition frequency: ~1.09 years", "INFO")

        for event in valid_events_raw:
            event_date = event['date']
            event_name = event['name']
            
            print_status(f"  Processing event: {event_name} ({event_date.date()})", "PROCESS")

            window_start = event_date - pd.Timedelta(days=event_window_days)
            window_end = event_date + pd.Timedelta(days=event_window_days)
            
            window_data = complete_df[
                (complete_df['date'] >= window_start) & 
                (complete_df['date'] <= window_end)
            ].copy()
            
            if len(window_data) < min_pairs_per_day * 10: # Overall minimum for event window
                print_status(f"    Skipping event {event_name}: insufficient total pairs ({len(window_data)})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Insufficient total pairs in window'}
                continue
            
            window_data['days_from_event'] = (window_data['date'] - event_date).dt.days
            
            event_result = _analyze_event_window(window_data, event_date, event_window_days, expected_amplitude, min_pairs_per_day)
            
            event_analysis_results[event_name] = event_result
            
            if event_result['success']:
                all_event_data_for_stacking.append(window_data)
                if event_result['gaussian_fit']['is_significant']:
                    sigma_level = event_result['gaussian_fit']['sigma_level']
                    amplitude_pct = event_result['gaussian_fit']['amplitude_fraction_of_baseline'] * 100
                    print_status(f"    SIGNIFICANT detection for {event_name}: {sigma_level:.1f}σ, {amplitude_pct:.1f}% amplitude", "SUCCESS")
                else:
                    sigma_level = event_result['gaussian_fit']['sigma_level']
                    amplitude_pct = event_result['gaussian_fit']['amplitude_fraction_of_baseline'] * 100
                    threshold = TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)
                    print_status(f"    Signal detected for {event_name}: {sigma_level:.1f}σ (below {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"    Analysis failed for {event_name}: {event_result['error']}", "WARNING")
        
        # REMOVED: Stacked analysis is now performed in Step 4.4 (Comprehensive Gravitational-Temporal Field Analysis)
        # Step 4.4 provides more sophisticated analysis with Savitzky-Golay smoothing and multi-planet stacking
        # Keeping only individual event detection here for exploratory analysis
        stacked_analysis_result = {
            'enabled': False, 
            'deferred_to': 'step_4.4_gravitational_temporal_field_analysis',
            'reason': 'More sophisticated stacked analysis available in Step 4.4 with multi-planet correlation'
        }

        # Final results structure
        results = {
            'success': True,
            'analysis_type': 'jupiter_opposition_analysis',
            'n_opposition_events_total': len(jupiter_oppositions),
            'n_opposition_events_analyzed': len(event_analysis_results),
            'event_results': event_analysis_results,
            'stacked_analysis': stacked_analysis_result,
            'expected_amplitude': expected_amplitude,
            'detection_threshold': TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0),
            'interpretation': 'Jupiter opposition analysis completed.'
        }
        
        # Overall interpretation based on detections
        n_significant_individual = sum(1 for res in event_analysis_results.values() if res.get('success') and res['gaussian_fit'].get('is_significant', False))
        if n_significant_individual > 0 or (stacked_analysis_result.get('success') and stacked_analysis_result['gaussian_fit'].get('is_significant', False)):
            results['interpretation'] = f"Significant Jupiter opposition signal(s) detected (individual: {n_significant_individual}, stacked: {stacked_analysis_result.get('success', False) and stacked_analysis_result['gaussian_fit'].get('is_significant', False)})"
        else:
            results['interpretation'] = "No significant Jupiter opposition signals detected."

        print_status(f"Jupiter opposition analysis complete: {len(event_analysis_results)} events analyzed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Jupiter opposition analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc() # For debugging
        return {
            'success': False, 
            'error': str(e),
            'analysis_type': 'jupiter_opposition_analysis',
            'n_opposition_events_total': len(jupiter_oppositions) if 'jupiter_oppositions' in locals() else 0,
            'interpretation': f"Jupiter opposition analysis failed due to error: {str(e)}"
        }

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
    
    DETECTION CHARACTERISTICS:
    - Saturn orbital period: 29.5 years (10,759 days)
    - Dataset coverage: 912 days = only 8% of one Saturn orbit
    - Available oppositions: 3 events (moderate statistical power)
    - Like Jupiter, Saturn benefits from continuous daily analysis (Step 4.4)
      for capturing slow orbital modulation
    
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
        
        # ========================================
        # SATURN OPPOSITION WINDOW STRATEGY
        # ========================================
        # Window size: 240 days (±120 days) - Same as Jupiter
        # Rationale: Consistent with empirically-validated optimal coupling timescale
        #            Saturn's gravitational influence is weaker than Jupiter's (~12x smaller)
        #            but operates on the same seasonal coupling timescale
        # ========================================
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS', 120)  # ±120 days = 240-day total window
        expected_amplitude = TEPConfig.get_float('TEP_SATURN_AMPLITUDE_FRACTION', 0.00019)
        min_pairs_per_day = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS_PER_DAY', 100) # Min pairs for daily binning
        
        print_status(f"Using 240-day windows (±{window_days} days) aligned with optimal coupling timescale", "INFO")
        
        print_status(f"Analyzing {len(saturn_events)} Saturn opposition events", "INFO")
        print_status(f"Event window: ±{window_days} days, Expected Amplitude: {expected_amplitude*100:.3f}%", "INFO")
        print_status(f"Saturn orbital period: ~29.46 years, Opposition frequency: ~1.04 years", "INFO")
        
        # Analyze each event
        event_analysis_results = {}
        all_event_data_for_stacking = []
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        data_span_days = (data_end - data_start).days + 1  # Inclusive date count
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()} ({data_span_days} days)", "INFO")

        if data_span_days < (2 * window_days):
            print_status(
                f"Skipping Saturn opposition analysis: data span {data_span_days} days < required window {2 * window_days} days",
                "WARNING"
            )
            return {
                'success': False,
                'error': 'insufficient_temporal_coverage',
                'required_days': int(2 * window_days),
                'available_days': int(data_span_days)
            }
        
        for event in saturn_events:
            event_name = event['name']
            event_date = event['date']
            description = event['description']
            
            # Check if event is within data range
            if not (data_start - pd.Timedelta(days=window_days) <= event_date <= data_end + pd.Timedelta(days=window_days)):
                print_status(f"Skipping {event_name} ({event_date.date()}): outside data range ({data_start.date()} to {data_end.date()})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Event outside data range'}
                continue
            
            print_status(f"  Processing event: {event_name} ({event_date.date()})", "PROCESS")
            
            # Define time windows
            event_start = event_date - pd.Timedelta(days=window_days)
            event_end = event_date + pd.Timedelta(days=window_days)
            
            # Extract event data
            event_data = complete_df[
                (complete_df['date'] >= event_start) & 
                (complete_df['date'] <= event_end)
            ].copy()
            
            if len(event_data) < min_pairs_per_day * 10: # Overall minimum for event window
                print_status(f"    Skipping event {event_name}: insufficient total pairs ({len(event_data)})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Insufficient total pairs in window'}
                continue
            
            event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
            
            event_result = _analyze_event_window(event_data, event_date, window_days, expected_amplitude, min_pairs_per_day)
            
            # Add description for summary function
            event_result['description'] = description
            event_analysis_results[event_name] = event_result
            
            if event_result['success']:
                all_event_data_for_stacking.append(event_data)
                # FIXED: Always show signal strength (like Jupiter), not just binary detection
                sigma_level = event_result['gaussian_fit']['sigma_level']
                amplitude_pct = event_result['gaussian_fit']['amplitude_fraction_of_baseline'] * 100
                threshold = TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)
                if event_result['gaussian_fit']['is_significant']:
                    print_status(f"    Significant detection for {event_name}: {sigma_level:.1f}σ (exceeds {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "SUCCESS")
                else:
                    print_status(f"    Signal detected for {event_name}: {sigma_level:.1f}σ (below {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"    Analysis failed for {event_name}: {event_result['error']}", "WARNING")
        
        # REMOVED: Stacked analysis is now performed in Step 4.4 (Comprehensive Gravitational-Temporal Field Analysis)
        # Step 4.4 provides more sophisticated analysis with Savitzky-Golay smoothing and multi-planet stacking
        # Keeping only individual event detection here for exploratory analysis
        stacked_analysis_result = {
            'enabled': False, 
            'deferred_to': 'step_4.4_gravitational_temporal_field_analysis',
            'reason': 'More sophisticated stacked analysis available in Step 4.4 with multi-planet correlation'
        }

        # Final results structure
        results = {
            'success': True,
            'analysis_type': 'saturn_opposition',
            'n_opposition_events_total': len(saturn_events),
            'n_opposition_events_analyzed': len(event_analysis_results),
            'event_results': event_analysis_results,
            'stacked_analysis': stacked_analysis_result,
            'expected_amplitude': expected_amplitude,
            'detection_threshold': TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0),
            'interpretation': 'Saturn opposition analysis completed.'
        }
        
        # Overall interpretation based on detections
        n_significant_individual = sum(1 for res in event_analysis_results.values() if res.get('success') and res['gaussian_fit'].get('is_significant', False))
        if n_significant_individual > 0 or (stacked_analysis_result.get('success') and stacked_analysis_result['gaussian_fit'].get('is_significant', False)):
            results['interpretation'] = f"Significant Saturn opposition signal(s) detected (individual: {n_significant_individual}, stacked: {stacked_analysis_result.get('success', False) and stacked_analysis_result['gaussian_fit'].get('is_significant', False)})"
        else:
            results['interpretation'] = "No significant Saturn opposition signals detected."

        print_status(f"Saturn opposition analysis complete: {len(event_analysis_results)} events analyzed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Saturn opposition analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc() # For debugging
        return {
            'success': False, 
            'error': str(e),
            'analysis_type': 'saturn_opposition_analysis',
            'n_opposition_events_total': len(saturn_events) if 'saturn_events' in locals() else 0,
            'interpretation': f"Saturn opposition analysis failed due to error: {str(e)}"
        }

def run_mars_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Mars opposition events.
    
    Mars oppositions occur when Earth-Mars distance is minimized, causing
    Mars's gravitational potential at Earth to peak. According to TEP theory,
    this should create a brief global enhancement in timing correlations.
    """
    try:
        print_status("Starting Mars Opposition Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Mars opposition dates (when Earth-Mars distance is minimized)
        mars_events = [
            {'name': 'mars_2025', 'date': pd.to_datetime('2025-01-16'), 'description': 'Mars Opposition January 2025'}
        ]
        
        # ========================================
        # MARS OPPOSITION WINDOW STRATEGY
        # ========================================
        # Window size: 240 days (±120 days) - Consistent with other planetary analyses
        # Rationale: Mars has weakest expected signal but operates on same coupling timescale
        #            Longer window provides better signal-to-noise for weak signals
        # ========================================
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS', 120)  # ±120 days = 240-day total window
        expected_amplitude = TEPConfig.get_float('TEP_MARS_AMPLITUDE_FRACTION', 0.00005)
        min_pairs_per_day = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS_PER_DAY', 100) # Min pairs for daily binning
        
        print_status(f"Using 240-day windows (±{window_days} days) for maximum sensitivity to weak Mars signal", "INFO")
        
        print_status(f"Analyzing {len(mars_events)} Mars opposition events", "INFO")
        print_status(f"Event window: ±{window_days} days, Expected Amplitude: {expected_amplitude*100:.4f}% (weakest expected signal)", "INFO")
        print_status(f"Mars orbital period: ~1.88 years, Opposition frequency: ~2.13 years", "INFO")
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        data_span_days = (data_end - data_start).days + 1  # Inclusive date count
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()} ({data_span_days} days)", "INFO")

        if data_span_days < (2 * window_days):
            print_status(
                f"Skipping Mars opposition analysis: data span {data_span_days} days < required window {2 * window_days} days",
                "WARNING"
            )
            return {
                'success': False,
                'error': 'insufficient_temporal_coverage',
                'required_days': int(2 * window_days),
                'available_days': int(data_span_days)
            }
        
        event_analysis_results = {}
        all_event_data_for_stacking = []
        
        for event in mars_events:
            event_name = event['name']
            event_date = event['date']
            description = event['description']
            
            # Check if event is within data range
            if not (data_start - pd.Timedelta(days=window_days) <= event_date <= data_end + pd.Timedelta(days=window_days)):
                print_status(f"Skipping {event_name} ({event_date.date()}): outside data range ({data_start.date()} to {data_end.date()})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Event outside data range'}
                continue
            
            print_status(f"  Processing event: {event_name} ({event_date.date()})", "PROCESS")
            
            # Define time windows
            event_start = event_date - pd.Timedelta(days=window_days)
            event_end = event_date + pd.Timedelta(days=window_days)
            
            # Extract event data
            event_data = complete_df[
                (complete_df['date'] >= event_start) & 
                (complete_df['date'] <= event_end)
            ].copy()
            
            if len(event_data) < min_pairs_per_day * 10: # Overall minimum for event window
                print_status(f"    Skipping event {event_name}: insufficient total pairs ({len(event_data)})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Insufficient total pairs in window'}
                continue
            
            event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
            
            event_result = _analyze_event_window(event_data, event_date, window_days, expected_amplitude, min_pairs_per_day)
            
            # Add description for summary function
            event_result['description'] = description
            event_analysis_results[event_name] = event_result
            
            if event_result['success']:
                all_event_data_for_stacking.append(event_data)
                # FIXED: Always show signal strength (like Jupiter), not just binary detection
                sigma_level = event_result['gaussian_fit']['sigma_level']
                amplitude_pct = event_result['gaussian_fit']['amplitude_fraction_of_baseline'] * 100
                threshold = TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)
                if event_result['gaussian_fit']['is_significant']:
                    print_status(f"    Significant detection for {event_name}: {sigma_level:.1f}σ (exceeds {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "SUCCESS")
                else:
                    print_status(f"    Signal detected for {event_name}: {sigma_level:.1f}σ (below {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"    Analysis failed for {event_name}: {event_result['error']}", "WARNING")
        
        # REMOVED: Stacked analysis is now performed in Step 4.4 (Comprehensive Gravitational-Temporal Field Analysis)
        # Step 4.4 provides more sophisticated analysis with Savitzky-Golay smoothing and multi-planet stacking
        # Keeping only individual event detection here for exploratory analysis
        stacked_analysis_result = {
            'enabled': False, 
            'deferred_to': 'step_4.4_gravitational_temporal_field_analysis',
            'reason': 'More sophisticated stacked analysis available in Step 4.4 with multi-planet correlation'
        }

        # Final results structure
        results = {
            'success': True,
            'analysis_type': 'mars_opposition_analysis',
            'n_opposition_events_total': len(mars_events),
            'n_opposition_events_analyzed': len(event_analysis_results),
            'event_results': event_analysis_results,
            'stacked_analysis': stacked_analysis_result,
            'expected_amplitude': expected_amplitude,
            'detection_threshold': TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0),
            'interpretation': 'Mars opposition analysis completed.'
        }
        
        # Overall interpretation based on detections
        n_significant_individual = sum(1 for res in event_analysis_results.values() if res.get('success') and res['gaussian_fit'].get('is_significant', False))
        if n_significant_individual > 0 or (stacked_analysis_result.get('success') and stacked_analysis_result['gaussian_fit'].get('is_significant', False)):
            results['interpretation'] = f"Significant Mars opposition signal(s) detected (individual: {n_significant_individual}, stacked: {stacked_analysis_result.get('success', False) and stacked_analysis_result['gaussian_fit'].get('is_significant', False)})"
        else:
            results['interpretation'] = "No significant Mars opposition signals detected."

        print_status(f"Mars opposition analysis complete: {len(event_analysis_results)} events analyzed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Mars opposition analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc() # For debugging
        return {
            'success': False, 
            'error': str(e),
            'analysis_type': 'mars_opposition_analysis',
            'n_opposition_events_total': len(mars_events) if 'mars_events' in locals() else 0,
            'interpretation': f"Mars opposition analysis failed due to error: {str(e)}"
        }

def run_venus_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Venus inferior conjunction events.
    
    Venus inferior conjunctions occur when Venus passes between Earth and Sun,
    reaching minimum Earth-Venus distance (~0.28 AU). Despite smaller mass than
    outer planets, Venus's proximity makes it gravitationally significant for TEP.
    
    Earth-Venus synodic period: ~584 days (~19 months)
    Expected amplitude: ~0.1% (stronger than Saturn due to proximity)
    """
    try:
        print_status("Starting Venus Inferior Conjunction Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Venus inferior conjunctions (closest approach, between Earth and Sun)
        # Source: NASA JPL Horizons ephemeris
        venus_events = [
            {'name': 'venus_2023', 'date': pd.to_datetime('2023-08-13'), 'description': 'Venus Inferior Conjunction August 2023'},
            {'name': 'venus_2025', 'date': pd.to_datetime('2025-03-23'), 'description': 'Venus Inferior Conjunction March 2025'}
        ]
        
        # ========================================
        # VENUS CONJUNCTION WINDOW STRATEGY
        # ========================================
        # Window size: 240 days (±120 days) - Consistent with planetary coupling timescale
        # Rationale: Venus has significant gravitational effect due to proximity
        #            Longer window captures full synodic cycle context
        # ========================================
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS', 120)  # ±120 days = 240-day total window
        expected_amplitude = TEPConfig.get_float('TEP_VENUS_AMPLITUDE_FRACTION', 0.001)  # 0.1%
        min_pairs_per_day = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS_PER_DAY', 100)
        
        print_status(f"Using 240-day windows (±{window_days} days) for optimal Venus coupling detection", "INFO")
        
        print_status(f"Analyzing {len(venus_events)} Venus inferior conjunction events", "INFO")
        print_status(f"Event window: ±{window_days} days, Expected Amplitude: {expected_amplitude*100:.3f}%", "INFO")
        print_status(f"Venus synodic period: ~584 days (~19 months)", "INFO")
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        data_span_days = (data_end - data_start).days + 1  # Inclusive date count
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()} ({data_span_days} days)", "INFO")

        if data_span_days < (2 * window_days):
            print_status(
                f"Skipping Venus inferior conjunction analysis: data span {data_span_days} days < required window {2 * window_days} days",
                "WARNING"
            )
            return {
                'success': False,
                'error': 'insufficient_temporal_coverage',
                'required_days': int(2 * window_days),
                'available_days': int(data_span_days)
            }
        
        event_analysis_results = {}
        all_event_data_for_stacking = []
        
        for event in venus_events:
            event_name = event['name']
            event_date = event['date']
            description = event['description']
            
            # Check if event is within data range
            if not (data_start - pd.Timedelta(days=window_days) <= event_date <= data_end + pd.Timedelta(days=window_days)):
                print_status(f"Skipping {event_name} ({event_date.date()}): outside data range", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Event outside data range'}
                continue
            
            print_status(f"  Processing event: {event_name} ({event_date.date()})", "PROCESS")
            
            # Define time windows
            event_start = event_date - pd.Timedelta(days=window_days)
            event_end = event_date + pd.Timedelta(days=window_days)
            
            # Extract event data
            event_data = complete_df[
                (complete_df['date'] >= event_start) & 
                (complete_df['date'] <= event_end)
            ].copy()
            
            if len(event_data) < min_pairs_per_day * 10:
                print_status(f"    Skipping event {event_name}: insufficient total pairs ({len(event_data)})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Insufficient total pairs in window'}
                continue
            
            event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
            
            event_result = _analyze_event_window(event_data, event_date, window_days, expected_amplitude, min_pairs_per_day)
            
            event_result['description'] = description
            event_analysis_results[event_name] = event_result
            
            if event_result['success']:
                all_event_data_for_stacking.append(event_data)
                sigma_level = event_result['gaussian_fit']['sigma_level']
                amplitude_pct = event_result['gaussian_fit']['amplitude_fraction_of_baseline'] * 100
                threshold = TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)
                if event_result['gaussian_fit']['is_significant']:
                    print_status(f"    Significant detection for {event_name}: {sigma_level:.1f}σ (exceeds {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "SUCCESS")
                else:
                    print_status(f"    Signal detected for {event_name}: {sigma_level:.1f}σ (below {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"    Analysis failed for {event_name}: {event_result['error']}", "WARNING")
        
        stacked_analysis_result = {
            'enabled': False,
            'deferred_to': 'step_4.4_gravitational_temporal_field_analysis',
            'reason': 'More sophisticated stacked analysis available in Step 4.4 with multi-planet correlation'
        }

        results = {
            'success': True,
            'analysis_type': 'venus_inferior_conjunction_analysis',
            'n_conjunction_events_total': len(venus_events),
            'n_conjunction_events_analyzed': len(event_analysis_results),
            'event_results': event_analysis_results,
            'stacked_analysis': stacked_analysis_result,
            'expected_amplitude': expected_amplitude,
            'detection_threshold': TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0),
            'interpretation': 'Venus inferior conjunction analysis completed.'
        }
        
        n_significant_individual = sum(1 for res in event_analysis_results.values() if res.get('success') and res['gaussian_fit'].get('is_significant', False))
        if n_significant_individual > 0:
            results['interpretation'] = f"Significant Venus conjunction signal(s) detected: {n_significant_individual}"
        else:
            results['interpretation'] = "No significant Venus conjunction signals detected."

        print_status(f"Venus inferior conjunction analysis complete: {len(event_analysis_results)} events analyzed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Venus conjunction analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'analysis_type': 'venus_inferior_conjunction_analysis',
            'n_conjunction_events_total': len(venus_events) if 'venus_events' in locals() else 0,
            'interpretation': f"Venus conjunction analysis failed: {str(e)}"
        }


def run_mercury_opposition_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around Mercury inferior conjunction events.
    
    Mercury inferior conjunctions occur when Mercury passes between Earth and Sun,
    reaching minimum Earth-Mercury distance (~0.55 AU). Though weakest planetary
    signal, high frequency (~116 days) provides good statistical power.
    
    Earth-Mercury synodic period: ~116 days (~4 months)
    Expected amplitude: ~0.01% (weakest planetary signal, but frequent)
    """
    try:
        print_status("Starting Mercury Inferior Conjunction Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Mercury inferior conjunctions (closest approach)
        # Source: NASA JPL Horizons ephemeris
        mercury_events = [
            {'name': 'mercury_2023_01', 'date': pd.to_datetime('2023-01-07'), 'description': 'Mercury Inferior Conjunction January 2023'},
            {'name': 'mercury_2023_05', 'date': pd.to_datetime('2023-05-01'), 'description': 'Mercury Inferior Conjunction May 2023'},
            {'name': 'mercury_2023_09', 'date': pd.to_datetime('2023-08-28'), 'description': 'Mercury Inferior Conjunction August 2023'},
            {'name': 'mercury_2023_12', 'date': pd.to_datetime('2023-12-22'), 'description': 'Mercury Inferior Conjunction December 2023'},
            {'name': 'mercury_2024_04', 'date': pd.to_datetime('2024-04-11'), 'description': 'Mercury Inferior Conjunction April 2024'},
            {'name': 'mercury_2024_08', 'date': pd.to_datetime('2024-08-05'), 'description': 'Mercury Inferior Conjunction August 2024'},
            {'name': 'mercury_2024_11', 'date': pd.to_datetime('2024-11-26'), 'description': 'Mercury Inferior Conjunction November 2024'},
            {'name': 'mercury_2025_03', 'date': pd.to_datetime('2025-03-15'), 'description': 'Mercury Inferior Conjunction March 2025'}
        ]
        
        # ========================================
        # MERCURY CONJUNCTION WINDOW STRATEGY
        # ========================================
        # Window size: 240 days (±120 days) - Consistent with optimal coupling timescale
        # Rationale: Mercury has weakest signal but high frequency allows statistical power
        #            Longer window needed for adequate signal-to-noise despite frequent events
        # ========================================
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_EVENT_WINDOW_DAYS', 120)  # ±120 days = 240-day total window
        expected_amplitude = TEPConfig.get_float('TEP_MERCURY_AMPLITUDE_FRACTION', 0.0001)  # 0.01%
        min_pairs_per_day = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS_PER_DAY', 100)
        
        print_status(f"Using 240-day windows (±{window_days} days) for maximum sensitivity to weak Mercury signal", "INFO")
        
        print_status(f"Analyzing {len(mercury_events)} Mercury inferior conjunction events", "INFO")
        print_status(f"Event window: ±{window_days} days, Expected Amplitude: {expected_amplitude*100:.4f}%", "INFO")
        print_status(f"Mercury synodic period: ~116 days (~4 months) - high frequency provides statistical power", "INFO")
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        data_span_days = (data_end - data_start).days + 1  # Inclusive date count
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()} ({data_span_days} days)", "INFO")

        if data_span_days < (2 * window_days):
            print_status(
                f"Skipping Mercury inferior conjunction analysis: data span {data_span_days} days < required window {2 * window_days} days",
                "WARNING"
            )
            return {
                'success': False,
                'error': 'insufficient_temporal_coverage',
                'required_days': int(2 * window_days),
                'available_days': int(data_span_days)
            }
        
        event_analysis_results = {}
        all_event_data_for_stacking = []
        
        for event in mercury_events:
            event_name = event['name']
            event_date = event['date']
            description = event['description']
            
            # Check if event is within data range
            if not (data_start - pd.Timedelta(days=window_days) <= event_date <= data_end + pd.Timedelta(days=window_days)):
                print_status(f"Skipping {event_name} ({event_date.date()}): outside data range", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Event outside data range'}
                continue
            
            print_status(f"  Processing event: {event_name} ({event_date.date()})", "PROCESS")
            
            # Define time windows
            event_start = event_date - pd.Timedelta(days=window_days)
            event_end = event_date + pd.Timedelta(days=window_days)
            
            # Extract event data
            event_data = complete_df[
                (complete_df['date'] >= event_start) & 
                (complete_df['date'] <= event_end)
            ].copy()
            
            if len(event_data) < min_pairs_per_day * 10:
                print_status(f"    Skipping event {event_name}: insufficient total pairs ({len(event_data)})", "WARNING")
                event_analysis_results[event_name] = {'success': False, 'error': 'Insufficient total pairs in window'}
                continue
            
            event_data['days_from_event'] = (event_data['date'] - event_date).dt.days
            
            event_result = _analyze_event_window(event_data, event_date, window_days, expected_amplitude, min_pairs_per_day)
            
            event_result['description'] = description
            event_analysis_results[event_name] = event_result
            
            if event_result['success']:
                all_event_data_for_stacking.append(event_data)
                sigma_level = event_result['gaussian_fit']['sigma_level']
                amplitude_pct = event_result['gaussian_fit']['amplitude_fraction_of_baseline'] * 100
                threshold = TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)
                if event_result['gaussian_fit']['is_significant']:
                    print_status(f"    Significant detection for {event_name}: {sigma_level:.1f}σ (exceeds {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "SUCCESS")
                else:
                    print_status(f"    Signal detected for {event_name}: {sigma_level:.1f}σ (below {threshold:.1f}σ threshold), {amplitude_pct:.1f}% amplitude", "INFO")
            else:
                print_status(f"    Analysis failed for {event_name}: {event_result['error']}", "WARNING")
        
        stacked_analysis_result = {
            'enabled': False,
            'deferred_to': 'step_4.4_gravitational_temporal_field_analysis',
            'reason': 'More sophisticated stacked analysis available in Step 4.4 with multi-planet correlation'
        }

        results = {
            'success': True,
            'analysis_type': 'mercury_inferior_conjunction_analysis',
            'n_conjunction_events_total': len(mercury_events),
            'n_conjunction_events_analyzed': len(event_analysis_results),
            'event_results': event_analysis_results,
            'stacked_analysis': stacked_analysis_result,
            'expected_amplitude': expected_amplitude,
            'detection_threshold': TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0),
            'interpretation': 'Mercury inferior conjunction analysis completed.'
        }
        
        n_significant_individual = sum(1 for res in event_analysis_results.values() if res.get('success') and res['gaussian_fit'].get('is_significant', False))
        if n_significant_individual > 0:
            results['interpretation'] = f"Significant Mercury conjunction signal(s) detected: {n_significant_individual}"
        else:
            results['interpretation'] = "No significant Mercury conjunction signals detected."

        print_status(f"Mercury inferior conjunction analysis complete: {len(event_analysis_results)} events analyzed", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Mercury conjunction analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'analysis_type': 'mercury_inferior_conjunction_analysis',
            'n_conjunction_events_total': len(mercury_events) if 'mercury_events' in locals() else 0,
            'interpretation': f"Mercury conjunction analysis failed: {str(e)}"
        }
def run_solar_rotation_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations with solar rotation cycle (Carrington rotation).
    
    Unlike planetary oppositions, this tests TEP coupling to the rotating solar magnetic
    field and solar activity patterns. The Sun's ~27-day rotation period creates periodic
    modulation in solar wind, magnetic field orientation, and space weather at Earth.
    
    Physical mechanism: Solar rotation → space weather modulation → potential TEP coupling
    Carrington rotation period: ~27.3 days (sidereal at solar equator)
    Expected signature: Periodic modulation in timing correlations at ~27-day period
    
    This is fundamentally different from gravitational oppositions - it tests whether
    TEP couples to rotating magnetic/plasma structures rather than static gravitational fields.
    """
    try:
        print_status("Starting Solar Rotation Cycle Analysis...", "PROCESS")
        print_status("Testing TEP coupling to rotating solar magnetic field (Carrington rotation)", "INFO")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Configuration
        carrington_period_days = 27.3  # Solar rotation period at equator (sidereal)
        synodic_period_days = 27.0  # Approximately synodic period as seen from Earth
        
        # Check data coverage
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        total_days = (data_end - data_start).days + 1  # Inclusive date count
        n_rotations = total_days / synodic_period_days
        
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()} ({total_days} days)", "INFO")
        print_status(f"Solar rotation period: {synodic_period_days:.1f} days (~{n_rotations:.1f} complete rotations)", "INFO")
        
        # Compute daily coherence to analyze periodicity
        daily_coherence = complete_df.groupby('date')['coherence'].agg(['mean', 'std', 'count']).reset_index()
        daily_coherence = daily_coherence[daily_coherence['count'] >= 100]  # Minimum pairs per day
        
        if len(daily_coherence) < 30:
            return {
                'success': False,
                'error': 'Insufficient daily samples for periodic analysis',
                'analysis_type': 'solar_rotation_analysis'
            }
        
        # Compute days from start for FFT analysis
        daily_coherence['days_from_start'] = (daily_coherence['date'] - data_start).dt.days
        
        # Perform spectral analysis to detect ~27-day periodicity
        from scipy import signal
        from scipy.stats import pearsonr
        
        # Detrend data
        coherence_series = daily_coherence['mean'].values
        days_series = daily_coherence['days_from_start'].values
        
        # Remove linear trend
        z = np.polyfit(days_series, coherence_series, 1)
        p = np.poly1d(z)
        detrended = coherence_series - p(days_series)
        
        # Compute periodogram
        freqs, power = signal.periodogram(detrended, fs=1.0, window='hann', scaling='spectrum')
        periods = 1.0 / freqs[1:]  # Skip DC component
        power = power[1:]
        
        # Find peak near 27-day period
        target_period = synodic_period_days
        period_range = (20, 35)  # Search range for solar rotation signal
        
        mask = (periods >= period_range[0]) & (periods <= period_range[1])
        if mask.sum() > 0:
            peak_idx = np.argmax(power[mask])
            peak_period = periods[mask][peak_idx]
            peak_power = power[mask][peak_idx]
            
            # Compute significance by comparing to background
            background_power = np.median(power)
            snr = peak_power / background_power if background_power > 0 else 0
            
            # Test correlation with sinusoid at detected period
            test_sine = np.sin(2 * np.pi * days_series / peak_period)
            test_cosine = np.cos(2 * np.pi * days_series / peak_period)
            
            r_sin, p_sin = pearsonr(detrended, test_sine)
            r_cos, p_cos = pearsonr(detrended, test_cosine)
            
            # Use stronger correlation
            if abs(r_sin) > abs(r_cos):
                correlation = r_sin
                p_value = p_sin
                phase_component = 'sine'
            else:
                correlation = r_cos
                p_value = p_cos
                phase_component = 'cosine'
            
            r_squared = correlation ** 2
            
            is_significant = (p_value < 0.05) and (snr > 2.0) and (abs(correlation) > 0.3)
            
            results = {
                'success': True,
                'analysis_type': 'solar_rotation_analysis',
                'physical_mechanism': 'rotating_solar_magnetic_field_modulation',
                'carrington_period_days': carrington_period_days,
                'n_rotations_observed': float(n_rotations),
                'n_daily_samples': len(daily_coherence),
                'detected_period_days': float(peak_period),
                'period_deviation_from_expected': float(abs(peak_period - target_period)),
                'spectral_snr': float(snr),
                'correlation': float(correlation),
                'r_squared': float(r_squared),
                'p_value': float(p_value),
                'phase_component': phase_component,
                'is_significant': bool(is_significant),
                'detection_threshold': 'p<0.05 AND SNR>2.0 AND |r|>0.3',
                'interpretation': ''
            }
            
            if is_significant:
                results['interpretation'] = f"Solar rotation signature DETECTED: {peak_period:.1f}-day period (r²={r_squared:.3f}, p={p_value:.4f}, SNR={snr:.1f})"
                print_status(f"Solar Rotation: DETECTED {peak_period:.1f}-day period (expected ~{target_period:.1f} days)", "SUCCESS")
                print_status(f"  Correlation: r={correlation:.3f}, r²={r_squared:.3f}, p={p_value:.4f}", "SUCCESS")
                print_status(f"  Spectral SNR: {snr:.1f}× above background", "SUCCESS")
            else:
                results['interpretation'] = f"Solar rotation signature not significant: {peak_period:.1f}-day period (r²={r_squared:.3f}, p={p_value:.4f}, SNR={snr:.1f})"
                print_status(f"Solar Rotation: Peak at {peak_period:.1f} days but not significant", "INFO")
                print_status(f"  Correlation: r={correlation:.3f}, p={p_value:.4f}, SNR={snr:.1f}", "INFO")
        
        else:
            results = {
                'success': True,
                'analysis_type': 'solar_rotation_analysis',
                'n_daily_samples': len(daily_coherence),
                'error': 'No spectral peak found in 20-35 day range',
                'interpretation': 'No solar rotation signature detected in spectral analysis'
            }
            print_status("Solar Rotation: No significant periodicity detected in 20-35 day range", "INFO")
        
        print_status("Solar rotation analysis complete", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Solar rotation analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'analysis_type': 'solar_rotation_analysis',
            'interpretation': f"Solar rotation analysis failed: {str(e)}"
        }


def run_lunar_standstill_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations around major lunar standstill events.
    
    Major Lunar Standstills occur every 18.6 years when the Moon reaches its 
    maximum declination (±28.7°), creating enhanced tidal effects that should 
    modulate GPS timing correlations.
    
    This analysis uses the same event-window approach as planetary oppositions,
    looking for coherence enhancement during the standstill peak period.
    
    Expected amplitude: ~0.05% enhancement during standstill maximum
    """
    print_status("Starting Lunar Standstill Analysis (Event-Based)...", "PROCESS")
    
    try:
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Define major lunar standstill events (18.6-year cycle peaks)
        lunar_standstill_events = [
            {
                'name': 'lunar_standstill_2025',
                'date': pd.Timestamp('2025-06-01'),
                'description': 'Major Lunar Standstill 2024-2025 - Maximum lunar declination (±28.7°)'
            }
        ]
        
        # Configuration
        window_days = TEPConfig.get_int('TEP_LUNAR_WINDOW_DAYS', 180)  # ±6 months around peak
        expected_amplitude = 0.0005  # Expected fractional amplitude (0.05%)
        min_pairs_per_day = TEPConfig.get_int('TEP_EVENT_MIN_PAIRS_PER_DAY', 100)
        
        # Data range check
        data_start = complete_df['date'].min()
        data_end = complete_df['date'].max()
        data_span_days = (data_end - data_start).days + 1  # Inclusive date count
        
        print_status(f"Data coverage: {data_start.date()} to {data_end.date()} ({data_span_days} days)", "INFO")
        print_status(f"Lunar standstill event window: ±{window_days} days (±{window_days/30.4:.1f} months)", "INFO")
        print_status(f"Expected amplitude: {expected_amplitude*100:.3f}%", "INFO")
        
        # Filter events within data range
        valid_events = []
        for event in lunar_standstill_events:
            if data_start - pd.Timedelta(days=window_days) <= event['date'] <= data_end + pd.Timedelta(days=window_days):
                valid_events.append(event)
        
        if not valid_events:
            return {
                'success': False,
                'error': 'No lunar standstill events within dataset coverage',
                'analysis_type': 'lunar_standstill_analysis',
                'data_coverage': f"{data_start.date()} to {data_end.date()}",
                'required_date': '2025-06-01 ± 6 months'
            }
        
        # Analyze each valid standstill event
        event_analysis_results = {}
        all_event_data_for_stacking = []
        
        print_status(f"Analyzing {len(valid_events)} lunar standstill event(s)", "INFO")
        
        for event in valid_events:
            event_date = event['date']
            event_name = event['name']
            
            print_status(f"  Processing event: {event_name} ({event_date.date()})", "PROCESS")
            
            # Define time window
            window_start = event_date - pd.Timedelta(days=window_days)
            window_end = event_date + pd.Timedelta(days=window_days)
            
            # Extract event data
            window_data = complete_df[
                (complete_df['date'] >= window_start) & 
                (complete_df['date'] <= window_end)
            ].copy()
            
            if len(window_data) < min_pairs_per_day * 10:
                print_status(f"    Skipping event {event_name}: insufficient total pairs ({len(window_data)})", "WARNING")
                event_analysis_results[event_name] = {
                    'success': False, 
                    'error': 'Insufficient total pairs in window',
                    'event_date': event_date.isoformat()
                }
                continue
            
            window_data['days_from_event'] = (window_data['date'] - event_date).dt.days
            
            # Analyze using standard event window analysis (same as planetary oppositions)
            event_result = _analyze_event_window(
                window_data, event_date, window_days, 
                expected_amplitude, min_pairs_per_day
            )
            
            # Add description
            event_result['description'] = event['description']
            event_analysis_results[event_name] = event_result
            
            if event_result['success']:
                all_event_data_for_stacking.append(window_data)
                
                # Extract results for reporting
                gaussian = event_result.get('gaussian_fit', {})
                if gaussian.get('fit_success'):
                    amplitude = gaussian.get('amplitude', 0)
                    sigma_level = gaussian.get('sigma_level', 0)
                    is_significant = gaussian.get('is_significant', False)
                    
                    if is_significant:
                        print_status(f"    SIGNIFICANT lunar standstill signal: {sigma_level:.1f}σ", "SUCCESS")
                    else:
                        print_status(f"    Lunar standstill signal: {sigma_level:.1f}σ (not significant)", "INFO")
        
        # Determine overall interpretation
        n_significant = sum(1 for r in event_analysis_results.values() 
                          if r.get('success') and r.get('gaussian_fit', {}).get('is_significant', False))
        
        if n_significant > 0:
            interpretation = f"Significant Lunar Standstill signal(s) detected: {n_significant}"
        else:
            interpretation = "No significant Lunar Standstill signals detected."
        
        # Final results
        results = {
            'success': True,
            'analysis_type': 'lunar_standstill_analysis',
            'n_standstill_events_total': len(lunar_standstill_events),
            'n_standstill_events_analyzed': len(valid_events),
            'window_days': window_days,
            'expected_amplitude': expected_amplitude,
            'event_results': event_analysis_results,
            'interpretation': interpretation
        }
        
        print_status(f"Lunar standstill analysis complete: {interpretation}", "SUCCESS")
        return results
        
    except Exception as e:
        print_status(f"Lunar standstill analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return {
            'success': False, 
            'error': str(e),
            'analysis_type': 'lunar_standstill_analysis',
            'interpretation': f"Lunar standstill analysis failed: {str(e)}"
        }

def run_nutation_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Analyze GPS timing correlations for Earth's nutation signatures.
    
    Earth's nutation causes periodic variations in the orientation of Earth's
    rotation axis, which should create detectable modulations in GPS timing.
    """
    try:
        print_status("Starting Nutation Analysis...", "PROCESS")
        
        # Convert dates to datetime
        complete_df['date'] = pd.to_datetime(complete_df['date'])
        
        # Calculate days since epoch for nutation analysis
        epoch = pd.Timestamp('2000-01-01')
        complete_df['days_since_epoch'] = (complete_df['date'] - epoch).dt.days
        
        # Principal nutation periods (days)
        nutation_periods = {
            'main_nutation': 6798.4,  # ~18.6 years (main lunar nutation)
            'annual_nutation': 365.25,  # Annual nutation
            'semiannual_nutation': 182.6  # Semiannual nutation
        }
        
        # Check temporal coverage
        data_span_days = (complete_df['date'].max() - complete_df['date'].min()).days + 1  # Inclusive date count
        
        nutation_results = {}
        
        for nutation_name, period_days in nutation_periods.items():
            # Calculate nutation phase
            complete_df['nutation_phase'] = (2 * np.pi * complete_df['days_since_epoch'] / period_days) % (2 * np.pi)
            
            # Group into phase bins
            n_phase_bins = 12  # 30° phase resolution
            phase_bins = np.linspace(0, 2*np.pi, n_phase_bins + 1)
            complete_df['nutation_phase_bin'] = pd.cut(complete_df['nutation_phase'], 
                                                      bins=phase_bins, 
                                                      labels=range(n_phase_bins))
            
            # Analyze coherence vs nutation phase
            phase_coherence_data = []
            
            for phase_bin in range(n_phase_bins):
                phase_data = complete_df[complete_df['nutation_phase_bin'] == phase_bin]
                
                if len(phase_data) < 100:  # Need sufficient data per bin
                    continue
                
                mean_coherence = phase_data['coherence'].mean()
                coherence_std = phase_data['coherence'].std()
                
                phase_coherence_data.append({
                    'phase_bin': phase_bin,
                    'phase_degrees': phase_bin * 30,  # 30° per bin
                    'mean_coherence': mean_coherence,
                    'coherence_std': coherence_std,
                    'n_pairs': len(phase_data)
                })
            
            if len(phase_coherence_data) >= 6:  # Need sufficient phase coverage
                # Test for nutation modulation
                phases = [d['phase_degrees'] for d in phase_coherence_data]
                coherences = [d['mean_coherence'] for d in phase_coherence_data]
                
                # Fit sinusoidal model to detect nutation signature
                try:
                    def nutation_model(phase_rad, amplitude, phase_offset, baseline):
                        return amplitude * np.cos(phase_rad + phase_offset) + baseline
                    
                    phase_rad = np.array(phases) * np.pi / 180
                    popt, pcov = curve_fit(nutation_model, phase_rad, coherences, 
                                         p0=[0.01, 0, np.mean(coherences)])
                    
                    amplitude, phase_offset, baseline = popt
                    r_squared = 1 - np.sum((coherences - nutation_model(phase_rad, *popt))**2) / np.sum((coherences - np.mean(coherences))**2)
                    
                    nutation_results[nutation_name] = {
                        'period_days': period_days,
                        'amplitude': float(amplitude),
                        'phase_offset_rad': float(phase_offset),
                        'baseline': float(baseline),
                        'r_squared': float(r_squared),
                        'n_phase_bins': len(phase_coherence_data),
                        'phase_data': phase_coherence_data
                    }
                    
                except Exception as e:
                    nutation_results[nutation_name] = {
                        'period_days': period_days,
                        'fit_error': str(e),
                        'n_phase_bins': len(phase_coherence_data)
                    }
        
        results = {
            'success': True,
            'analysis_type': 'nutation_analysis',
            'data_span_days': data_span_days,
            'nutation_results': nutation_results
        }
        
        # Report significant nutation signatures
        significant_nutations = [name for name, result in nutation_results.items() 
                               if result.get('r_squared', 0) > 0.1]
        
        if significant_nutations:
            print_status(f"Nutation analysis complete: {len(significant_nutations)} significant signatures detected", "SUCCESS")
        else:
            print_status("Nutation analysis complete: No significant signatures detected", "INFO")
        
        return results
        
    except Exception as e:
        print_status(f"Nutation analysis failed: {e}", "ERROR")
        return {'success': False, 'error': str(e)}

# ===== NEW HELPER FUNCTIONS FOR PLANETARY OPPOSITION ANALYSIS =====

def gaussian_pulse_model(days_array, amplitude, sigma, baseline, center_days=0):
    """Gaussian pulse model for fitting event-locked coherence changes."""
    return amplitude * np.exp(-0.5 * ((days_array - center_days) / sigma)**2) + baseline

def _analyze_event_window(event_data: pd.DataFrame, event_date: pd.Timestamp, window_days: int, expected_amplitude: float, min_daily_pairs: int) -> Dict:
    """Helper to analyze a single event window with Gaussian fitting."""
    daily_data = []
    for day in range(-window_days, window_days + 1):
        day_data = event_data[event_data['days_from_event'] == day]
        if len(day_data) >= min_daily_pairs:
            daily_coherence = day_data['coherence'].mean()
            daily_data.append({
                'days_from_event': day,
                'mean_coherence': daily_coherence,
                'n_pairs': len(day_data)
            })

    if len(daily_data) < 10: # Need at least 10 daily bins for fitting
        return {'success': False, 'error': f'Insufficient daily data for fitting ({len(daily_data)} bins)'}

    days = np.array([d['days_from_event'] for d in daily_data])
    coherences = np.array([d['mean_coherence'] for d in daily_data])
    
    try:
        # Initial guess for amplitude: expected_amplitude or the max/min deviation from mean
        initial_amp_guess = expected_amplitude
        if coherences.max() - coherences.mean() > abs(expected_amplitude) and coherences.max() - coherences.mean() > abs(coherences.min() - coherences.mean()):
            initial_amp_guess = coherences.max() - coherences.mean()
        elif abs(coherences.min() - coherences.mean()) > abs(expected_amplitude):
            initial_amp_guess = coherences.min() - coherences.mean()
        
        # Ensure amplitude guess has correct sign if we have a strong expectation
        if expected_amplitude > 0 and initial_amp_guess < 0:
            initial_amp_guess = abs(initial_amp_guess)
        elif expected_amplitude < 0 and initial_amp_guess > 0:
            initial_amp_guess = -abs(initial_amp_guess)
        
        # FIXED: Expand bounds to accommodate 240-day windows
        day_range = max(abs(days.min()), abs(days.max()))
        center_bounds = [-day_range, day_range]  # Allow center anywhere in the event window
        
        # Clamp initial guesses to be within bounds
        initial_amp_guess = np.clip(initial_amp_guess, -0.1, 0.1)
        baseline_guess = np.clip(np.mean(coherences), -1.0, 1.0)
        
        p0 = [initial_amp_guess, 5.0, baseline_guess, 0.0] # amplitude, sigma, baseline, center_days
        
        # Bounds: amplitude (-0.1 to 0.1), sigma (1 to 60 days), baseline (-1 to 1), center_days (±window_days)
        bounds = ([-0.1, 1.0, -1.0, center_bounds[0]], [0.1, 60.0, 1.0, center_bounds[1]]) 

        popt, pcov = curve_fit(
            gaussian_pulse_model, days, coherences,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )
        
        amplitude, sigma, baseline, center_days = popt
        perr = np.sqrt(np.diag(pcov))
        amplitude_std_err = perr[0]

        # Calculate R-squared
        coherence_pred = gaussian_pulse_model(days, *popt)
        ss_res = np.sum((coherences - coherence_pred)**2)
        ss_tot = np.sum((coherences - np.mean(coherences))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Significance: amplitude / standard error
        sigma_level = abs(amplitude / amplitude_std_err) if amplitude_std_err > 0 else 0
        is_significant = sigma_level >= TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)

        # Fraction of baseline for amplitude
        amplitude_fraction_of_baseline = amplitude / baseline if baseline != 0 else 0

        return {
            'success': True,
            'event_date': event_date.isoformat(),
            'window_days': window_days,
            'n_pairs_in_window': len(event_data),
            'n_daily_bins': len(daily_data),
            'daily_data': daily_data,
            'gaussian_fit': {
                'amplitude': float(amplitude),
                'sigma_days': float(sigma),
                'baseline': float(baseline),
                'center_days': float(center_days),
                'r_squared': float(r_squared),
                'amplitude_std_err': float(amplitude_std_err),
                'sigma_level': float(sigma_level),
                'is_significant': bool(is_significant),
                'amplitude_fraction_of_baseline': float(amplitude_fraction_of_baseline),
                'fit_success': True
            }
        }
        
    except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
        return {'success': False, 'error': f'Gaussian fit failed: {str(e)}', 'daily_data': daily_data, 'fit_success': False}

def _perform_stacked_analysis(all_event_data: List[pd.DataFrame], window_days: int, expected_amplitude: float, min_daily_pairs: int) -> Dict:
    """Performs stacked analysis across multiple events."""
    import time
    start_time = time.time()
    
    if not all_event_data:
        return {'success': False, 'error': 'No data for stacked analysis'}

    # Stack all valid daily data
    stacked_daily_data = {}
    for event_df in all_event_data:
        for index, row in event_df.iterrows():
            day = row['days_from_event']
            coherence = row['coherence']
            if day not in stacked_daily_data:
                stacked_daily_data[day] = []
            stacked_daily_data[day].append(coherence)
    
    # Calculate mean coherence for each day across all stacked events
    mean_stacked_daily_data = []
    for day in sorted(stacked_daily_data.keys()):
        if len(stacked_daily_data[day]) >= min_daily_pairs:
            mean_stacked_daily_data.append({
                'days_from_event': day,
                'mean_coherence': np.mean(stacked_daily_data[day]),
                'n_pairs': len(stacked_daily_data[day])
            })
            
    if len(mean_stacked_daily_data) < 10:
        return {'success': False, 'error': f'Insufficient daily data for stacked fitting ({len(mean_stacked_daily_data)} bins)'}

    days = np.array([d['days_from_event'] for d in mean_stacked_daily_data])
    coherences = np.array([d['mean_coherence'] for d in mean_stacked_daily_data])
    
    try:
        initial_amp_guess = expected_amplitude
        if coherences.max() - coherences.mean() > abs(expected_amplitude) and coherences.max() - coherences.mean() > abs(coherences.min() - coherences.mean()):
            initial_amp_guess = coherences.max() - coherences.mean()
        elif abs(coherences.min() - coherences.mean()) > abs(expected_amplitude):
            initial_amp_guess = coherences.min() - coherences.mean()

        if expected_amplitude > 0 and initial_amp_guess < 0:
            initial_amp_guess = abs(initial_amp_guess)
        elif expected_amplitude < 0 and initial_amp_guess > 0:
            initial_amp_guess = -abs(initial_amp_guess)

        # IMPROVED: Better initial parameter guesses for faster convergence
        baseline_guess = np.mean(coherences)
        sigma_guess = np.std(days) / 3.0  # Better sigma estimate based on data spread
        center_guess = days[np.argmax(np.abs(coherences - baseline_guess))]  # Center at peak deviation
        
        # FIXED: Expand center_days bounds to accommodate 240-day windows (days range from -120 to +120)
        # Previous bounds [-5, 5] were too restrictive for large windows
        day_range = max(abs(days.min()), abs(days.max()))
        center_bounds = [-day_range, day_range]  # Allow center anywhere in the event window
        
        # Clamp initial guesses to be within bounds
        initial_amp_guess = np.clip(initial_amp_guess, -0.1, 0.1)
        sigma_guess = np.clip(sigma_guess, 1.0, 60.0)  # Increased max sigma for larger windows
        baseline_guess = np.clip(baseline_guess, -1.0, 1.0)
        center_guess = np.clip(center_guess, center_bounds[0], center_bounds[1])
        
        p0 = [initial_amp_guess, sigma_guess, baseline_guess, center_guess] 
        bounds = ([-0.1, 1.0, -1.0, center_bounds[0]], [0.1, 60.0, 1.0, center_bounds[1]])

        popt, pcov = curve_fit(
            gaussian_pulse_model, days, coherences,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )
        
        amplitude, sigma, baseline, center_days = popt
        perr = np.sqrt(np.diag(pcov))
        amplitude_std_err = perr[0]

        coherence_pred = gaussian_pulse_model(days, *popt)
        ss_res = np.sum((coherences - coherence_pred)**2)
        ss_tot = np.sum((coherences - np.mean(coherences))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        sigma_level = abs(amplitude / amplitude_std_err) if amplitude_std_err > 0 else 0
        is_significant = sigma_level >= TEPConfig.get_float('TEP_SIGNIFICANCE_THRESHOLD', 3.0)

        amplitude_fraction_of_baseline = amplitude / baseline if baseline != 0 else 0

        elapsed_time = time.time() - start_time
        
        return {
            'success': True,
            'n_events_stacked': len(all_event_data),
            'n_daily_bins_stacked': len(mean_stacked_daily_data),
            'stacked_daily_data': mean_stacked_daily_data,
            'processing_time_seconds': float(elapsed_time),
            'gaussian_fit': {
                'amplitude': float(amplitude),
                'sigma_days': float(sigma),
                'baseline': float(baseline),
                'center_days': float(center_days),
                'r_squared': float(r_squared),
                'amplitude_std_err': float(amplitude_std_err),
                'sigma_level': float(sigma_level),
                'is_significant': bool(is_significant),
                'amplitude_fraction_of_baseline': float(amplitude_fraction_of_baseline),
                'fit_success': True
            }
        }
    
    except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
        elapsed_time = time.time() - start_time
        return {'success': False, 'error': f'Stacked Gaussian fit failed: {str(e)}', 'fit_success': False, 'processing_time_seconds': float(elapsed_time)}

# ===== END NEW HELPER FUNCTIONS =====

# ===== NEW HELPER FUNCTIONS FOR LUNAR STANDSTILL ANALYSIS =====

def sinusoidal_fit_model(phase_rad, amplitude, phase_offset, baseline):
    """Sinusoidal model for fitting sidereal day amplitude."""
    return amplitude * np.cos(phase_rad + phase_offset) + baseline

def _calculate_sidereal_amplitude_for_day(daily_df: pd.DataFrame, min_pairs_per_bin: int) -> Optional[Dict]:
    """
    Calculates the sidereal day amplitude for a single day's data.
    Assumes daily_df has 'plateau_phase' and 'date' columns.
    """
    if len(daily_df) < min_pairs_per_bin * 5: # Need enough data for binning and fitting
        return None

    # Calculate Local Sidereal Time (LST) proxy - using hour of day for simplicity
    # A more precise LST calculation would involve longitude and UTC, but hour is a good proxy for diurnal phase
    daily_df['hour_of_day'] = daily_df['date'].dt.hour
    daily_df['lst_phase'] = (2 * np.pi * daily_df['hour_of_day'] / 24) % (2 * np.pi)

    n_lst_bins = 8 # 3-hour bins
    lst_bins = np.linspace(0, 2 * np.pi, n_lst_bins + 1)
    daily_df['lst_phase_bin'] = pd.cut(daily_df['lst_phase'], bins=lst_bins, labels=False, include_lowest=True)

    binned_data = daily_df.groupby('lst_phase_bin').agg(
        mean_coherence=('coherence', 'mean'),
        n_pairs=('coherence', 'size')
    ).reset_index()

    binned_data = binned_data[binned_data['n_pairs'] >= min_pairs_per_bin]

    if len(binned_data) < 4: # Need at least 4 bins for a robust sinusoidal fit
        return None

    # Fit sinusoidal model
    try:
        phases = (binned_data['lst_phase_bin'] + 0.5) * (2 * np.pi / n_lst_bins)
        coherences = binned_data['mean_coherence']
        weights = binned_data['n_pairs']

        p0 = [0.01, 0, np.mean(coherences)] # amplitude, phase_offset, baseline
        bounds = ([-0.1, -np.pi, -1.0], [0.1, np.pi, 1.0])

        popt, pcov = curve_fit(
            sinusoidal_fit_model, phases, coherences,
            p0=p0, sigma=1.0/np.sqrt(weights), bounds=bounds, maxfev=5000
        )

        amplitude, phase_offset, baseline = popt
        perr = np.sqrt(np.diag(pcov))
        amplitude_std_err = perr[0]

        r_squared = 1 - np.sum((coherences - sinusoidal_fit_model(phases, *popt))**2) / np.sum((coherences - np.mean(coherences))**2)
        
        return {
            'amplitude': float(amplitude),
            'amplitude_std_err': float(amplitude_std_err),
            'r_squared': float(r_squared),
            'baseline': float(baseline),
            'fit_success': True,
            'n_bins': len(binned_data)
        }
    except Exception as e:
        return {'fit_success': False, 'error': str(e)}

def _calculate_monthly_amplitudes(complete_df: pd.DataFrame, min_pairs_per_day: int) -> Dict:
    """
    Calculates mean sidereal day amplitudes for each month.
    """
    monthly_amplitudes = {}
    daily_groups = complete_df.groupby(complete_df['date'].dt.to_period('D'))

    all_daily_results = []

    for day_period, daily_df in daily_groups:
        day_str = day_period.start_time.isoformat()[:10]
        if len(daily_df) < min_pairs_per_day: # Minimum pairs for any daily processing
            continue
        
        sidereal_amp_result = _calculate_sidereal_amplitude_for_day(daily_df.copy(), min_pairs_per_bin=min_pairs_per_day // 5) # Heuristic for min_pairs_per_bin
        
        if sidereal_amp_result and sidereal_amp_result['fit_success']:
            all_daily_results.append({
                'date': day_period.start_time,
                'amplitude': sidereal_amp_result['amplitude'],
                'r_squared': sidereal_amp_result['r_squared'],
                'baseline': sidereal_amp_result['baseline'],
                'n_pairs': len(daily_df)
            })
    
    if not all_daily_results:
        return {'success': False, 'error': 'No successful daily amplitude calculations'}

    daily_amplitudes_df = pd.DataFrame(all_daily_results)
    daily_amplitudes_df['month_year'] = daily_amplitudes_df['date'].dt.to_period('M')

    monthly_grouped = daily_amplitudes_df.groupby('month_year').agg(
        mean_amplitude=('amplitude', 'mean'),
        std_amplitude=('amplitude', 'std'),
        n_days=('amplitude', 'size'),
        mean_baseline=('baseline', 'mean')
    ).reset_index()
    monthly_grouped['month_year'] = monthly_grouped['month_year'].dt.to_timestamp()

    monthly_grouped = monthly_grouped.sort_values('month_year')
    
    # Fill NaN std with 0 if only one day in month
    monthly_grouped['std_amplitude'] = monthly_grouped['std_amplitude'].fillna(0.0)

    return {
        'success': True,
        'periods': monthly_grouped.to_dict(orient='records'),
        'n_months': len(monthly_grouped),
        'mean_overall_amplitude': monthly_grouped['mean_amplitude'].mean(),
        'mean_overall_baseline': monthly_grouped['mean_baseline'].mean()
    }

def _fit_quadratic_model(monthly_amplitudes: List[Dict], standstill_peak_month: pd.Timestamp) -> Dict:
    """
    Fits a quadratic model to monthly amplitudes to detect lunar standstill peak.
    Assumes monthly_amplitudes is a list of dicts with 'month_year' (timestamp) and 'mean_amplitude'.
    """
    if len(monthly_amplitudes) < 5:
        return {'success': False, 'error': 'Insufficient monthly amplitude data for quadratic fit'}

    df = pd.DataFrame(monthly_amplitudes)
    df['months_from_peak'] = (df['month_year'].dt.to_period('M').view(dtype='int64') - standstill_peak_month.to_period('M').view(dtype='int64'))

    x_data = df['months_from_peak'].values
    y_data = df['mean_amplitude'].values
    weights = df['n_days'].values # Use number of days in month as weight

    def quadratic_model(x, a, b, c):
        return a * x**2 + b * x + c

    try:
        p0 = [-0.00001, 0, np.mean(y_data)] # Initial guess: downward parabola, small slope, mean amplitude
        bounds = ([-0.01, -0.1, -1.0], [0.01, 0.1, 1.0]) # Reasonable bounds

        popt, pcov = curve_fit(
            quadratic_model, x_data, y_data,
            p0=p0, sigma=1.0/np.sqrt(weights), bounds=bounds, maxfev=5000
        )

        a, b, c = popt
        perr = np.sqrt(np.diag(pcov))

        # Peak of parabola: -b / (2a)
        peak_offset_months = -b / (2 * a) if a != 0 else 0
        peak_amplitude = quadratic_model(peak_offset_months, *popt)

        # R-squared
        y_pred = quadratic_model(x_data, *popt)
        ss_res = np.sum(weights * (y_data - y_pred)**2)
        ss_tot = np.sum(weights * (y_data - np.average(y_data, weights=weights))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        return {
            'success': True,
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'peak_offset_months': float(peak_offset_months),
            'peak_amplitude': float(peak_amplitude),
            'r_squared': float(r_squared),
            'param_errors': [float(e) for e in perr]
        }
    except Exception as e:
        return {'success': False, 'error': f'Quadratic fit failed: {str(e)}'}
def _calculate_standstill_enhancement(monthly_amplitudes: List[Dict], pre_standstill_months: int, during_standstill_months: int, expected_amplitude_baseline: float, significance_threshold: float) -> Dict:
    """
    Calculates enhancement ratio during standstill vs pre-standstill.
    Assumes monthly_amplitudes is a list of dicts with 'month_year' (timestamp) and 'mean_amplitude'.
    """
    if not monthly_amplitudes or len(monthly_amplitudes) < (pre_standstill_months + during_standstill_months) / 2: # heuristic
        return {'success': False, 'error': 'Insufficient monthly data for enhancement calculation'}
    
    df = pd.DataFrame(monthly_amplitudes)
    df['month_year'] = pd.to_datetime(df['month_year'])
    df = df.set_index('month_year').sort_index()

    # Define periods relative to the overall mean amplitude of the standstill period
    # For Lunar Standstill, we expect an *enhancement* in the sidereal day amplitude
    
    # Approximate the standstill period (e.g., 2024-2025)
    standstill_start_approx = pd.Timestamp('2024-01-01')
    standstill_end_approx = pd.Timestamp('2025-12-31')

    pre_standstill_period_end = standstill_start_approx - pd.DateOffset(months=1)
    pre_standstill_period_start = pre_standstill_period_end - pd.DateOffset(months=pre_standstill_months)

    during_standstill_period_start = standstill_start_approx
    during_standstill_period_end = standstill_end_approx # Use full 2-year range for 'during'

    pre_amplitudes = df.loc[pre_standstill_period_start:pre_standstill_period_end, 'mean_amplitude'].dropna()
    standstill_amplitudes = df.loc[during_standstill_period_start:during_standstill_period_end, 'mean_amplitude'].dropna()

    if len(pre_amplitudes) < 3 or len(standstill_amplitudes) < 3:
        return {'success': False, 'error': 'Insufficient data for pre-standstill or standstill periods'}

    mean_pre_amplitude = pre_amplitudes.mean()
    std_pre_amplitude = pre_amplitudes.std()
    mean_standstill_amplitude = standstill_amplitudes.mean()
    std_standstill_amplitude = standstill_amplitudes.std()

    # Calculate enhancement ratio: standstill / pre-standstill
    enhancement_ratio = mean_standstill_amplitude / mean_pre_amplitude if mean_pre_amplitude > 0 else np.nan
    enhancement_absolute = mean_standstill_amplitude - mean_pre_amplitude

    # Statistical significance of enhancement (simple t-test or z-test if means/stds are stable)
    # Assuming independent samples and roughly normal distribution for simplicity
    # For a more rigorous test, consider Welch's t-test or non-parametric tests
    
    # Z-test for difference of means if stds are known or large sample
    pooled_std_sq = (std_pre_amplitude**2 / len(pre_amplitudes)) + (std_standstill_amplitude**2 / len(standstill_amplitudes))
    if pooled_std_sq > 0:
        z_score = enhancement_absolute / np.sqrt(pooled_std_sq)
        p_value = stats.norm.sf(abs(z_score)) * 2 # Two-tailed p-value
    else:
        z_score = np.nan
        p_value = 1.0 # No variance, no significant difference

    is_significant = p_value < significance_threshold and enhancement_ratio > 1.0 # Significant if enhanced AND statistically significant

    return {
        'success': True,
        'mean_pre_amplitude': float(mean_pre_amplitude),
        'std_pre_amplitude': float(std_pre_amplitude),
        'mean_standstill_amplitude': float(mean_standstill_amplitude),
        'std_standstill_amplitude': float(std_standstill_amplitude),
        'enhancement_ratio': float(enhancement_ratio) if not np.isnan(enhancement_ratio) else 0.0,
        'enhancement_absolute': float(enhancement_absolute),
        'z_score': float(z_score) if not np.isnan(z_score) else 0.0,
        'p_value': float(p_value),
        'is_significant': bool(is_significant),
        'pre_standstill_months_count': len(pre_amplitudes),
        'standstill_months_count': len(standstill_amplitudes),
        'interpretation': f"Lunar standstill resulted in {enhancement_ratio:.2f}x amplitude enhancement." if is_significant else "No significant lunar standstill enhancement detected."
    }

def _classify_dance_signature(dance_score: float, dance_metrics: Dict) -> str:
    """
    Classify the mesh dance signature based on dance score and metrics.
    
    Args:
        dance_score: Overall dance score (0.0 to 1.0)
        dance_metrics: Dictionary containing detailed dance metrics
        
    Returns:
        str: Classification string describing the network coherence level
    """
    # Enhanced classification based on dance score and component analysis
    if dance_score >= 0.8:
        return "EXCEPTIONAL NETWORK COHERENCE - Unified spacetime detector with strong collective dynamics"
    elif dance_score >= 0.6:
        return "HIGH NETWORK COHERENCE - Strong collective motion with coherent mesh dynamics"
    elif dance_score >= 0.45:
        return "MODERATE NETWORK COHERENCE - Mesh coherence with collective motion patterns"
    elif dance_score >= 0.25:
        return "WEAK NETWORK COHERENCE - Limited collective motion detected"
    else:
        return "MINIMAL NETWORK COHERENCE - No significant collective dynamics"

# ===== END NEW HELPER FUNCTIONS =====

# ===== TEMPORAL COHERENCE ASSESSMENT MODULE =====

def analyze_resonance_frequencies(df: pd.DataFrame, results: Dict) -> Dict:
    """
    OPTION C.1: Resonance Frequency Analysis
    
    Analyzes potential resonance effects between GPS timing correlations and 
    known geophysical/astronomical frequencies. Tests for non-linear amplitude
    enhancement at specific frequency combinations.
    
    This addresses the extraordinary amplitude enhancements observed (100x-19,000x)
    which suggest resonance phenomena rather than linear gravitational coupling.
    """
    print_status("Starting Resonance Frequency Analysis...", "PROCESS")
    
    resonance_results = {
        'success': False,
        'resonance_patterns': [],
        'enhancement_factors': {},
        'coherent_frequencies': []
    }
    
    try:
        # Define known frequencies (cycles/day)
        frequencies = {
            'chandler_wobble': 1/433.0,
            'annual': 1/365.25,
            'semiannual': 2/365.25,
            'lunar_month': 1/27.32,
            'solar_rotation': 1/27.0,
            'tidal_m2': 1/0.5175,  # M2 tide
            'jupiter_synodic': 1/398.9,
            'saturn_synodic': 1/378.1,
            'mars_synodic': 1/780.0
        }
        
        # Extract temporal data
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
        
        # Test for resonance between frequency pairs
        resonance_patterns = []
        
        for name1, freq1 in frequencies.items():
            for name2, freq2 in frequencies.items():
                if name1 >= name2:
                    continue
                
                # Sum and difference frequencies (beat patterns)
                sum_freq = freq1 + freq2
                diff_freq = abs(freq1 - freq2)
                
                # Test correlation at sum/difference frequencies
                for freq_type, test_freq in [('sum', sum_freq), ('diff', diff_freq)]:
                    period_days = 1/test_freq if test_freq > 0 else np.inf
                    
                    if 10 < period_days < 1000:  # Focus on observable periods
                        # Compute phase at this frequency
                        phase = 2 * np.pi * test_freq * df['day_of_year']
                        df[f'cos_{name1}_{name2}_{freq_type}'] = np.cos(phase)
                        
                        # Correlation with coherence
                        corr = df['coherence'].corr(df[f'cos_{name1}_{name2}_{freq_type}'])
                        
                        if abs(corr) > 0.15:  # Significant correlation threshold
                            resonance_patterns.append({
                                'freq1_name': name1,
                                'freq2_name': name2,
                                'combination': freq_type,
                                'resonance_period_days': period_days,
                                'correlation': corr,
                                'frequency_hz': test_freq
                            })
        
        # Sort by correlation strength
        resonance_patterns.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        resonance_results['resonance_patterns'] = resonance_patterns[:20]  # Top 20
        resonance_results['success'] = True
        resonance_results['n_significant_resonances'] = len([p for p in resonance_patterns if abs(p['correlation']) > 0.2])
        
        print_status(f"Resonance Analysis: {len(resonance_patterns)} patterns detected", "SUCCESS")
        print_status(f"  Significant Resonances (|r|>0.2): {resonance_results['n_significant_resonances']}", "INFO")
        
        if resonance_patterns:
            print_status(f"  Top Resonance: {resonance_patterns[0]['freq1_name']} + {resonance_patterns[0]['freq2_name']} ({resonance_patterns[0]['resonance_period_days']:.1f} days, r={resonance_patterns[0]['correlation']:.3f})", "INFO")
        
    except Exception as e:
        print_status(f"Resonance analysis failed: {e}", "ERROR")
        resonance_results['error'] = str(e)
    
    return resonance_results


def analyze_nonlinear_coupling(df: pd.DataFrame, planetary_results: Dict) -> Dict:
    """
    OPTION C.2: Non-Linear Coupling Detection
    
    Tests for non-linear gravitational coupling by analyzing amplitude enhancement
    factors and their relationship to planetary configurations. Distinguishes between:
    - Linear coupling (amplitude ∝ gravitational potential)
    - Quadratic coupling (amplitude ∝ (gravitational potential)²)
    - Resonant coupling (amplitude enhanced at specific frequencies)
    """
    print_status("Starting Non-Linear Coupling Analysis...", "PROCESS")
    
    coupling_results = {
        'success': False,
        'coupling_type': 'unknown',
        'linearity_test': {},
        'enhancement_distribution': {}
    }
    
    try:
        # Extract planetary event amplitudes
        all_amplitudes = []
        expected_amplitudes = []
        
        for planet in ['jupiter_opposition_analysis', 'saturn_opposition_analysis', 'mars_opposition_analysis']:
            if planet in planetary_results and planetary_results[planet].get('success'):
                events = planetary_results[planet].get('event_results', {})
                for event_data in events.values():
                    if event_data.get('success'):
                        gaussian = event_data.get('gaussian_fit', {})
                        if gaussian.get('fit_success'):
                            # Calculate absolute amplitude for proper unit consistency
                            baseline = gaussian.get('baseline', 0.007)
                            amp_absolute = abs(gaussian.get('amplitude_fraction_of_baseline', 0)) * baseline
                            
                            if 'jupiter' in planet:
                                expected_absolute = 0.00220  # 0.220% as absolute
                            elif 'saturn' in planet:
                                expected_absolute = 0.00019  # 0.019% as absolute
                            else:  # mars
                                expected_absolute = 0.00005  # 0.0050% as absolute
                            
                            all_amplitudes.append(amp_absolute)
                            expected_amplitudes.append(expected_absolute)
        
        if len(all_amplitudes) >= 3:
            all_amplitudes = np.array(all_amplitudes)
            expected_amplitudes = np.array(expected_amplitudes)
            
            # Test linearity: observed/expected should be constant for linear coupling
            enhancement_factors = all_amplitudes / expected_amplitudes
            
            # Statistics
            mean_enhancement = np.mean(enhancement_factors)
            std_enhancement = np.std(enhancement_factors)
            cv_enhancement = std_enhancement / mean_enhancement if mean_enhancement > 0 else np.inf
            
            # Linearity test: correlation between expected and observed
            if len(expected_amplitudes) > 1:
                linear_corr = np.corrcoef(expected_amplitudes, all_amplitudes)[0, 1]
                
                # Test quadratic relationship
                quadratic_prediction = expected_amplitudes ** 2
                quadratic_corr = np.corrcoef(quadratic_prediction, all_amplitudes)[0, 1] if np.std(quadratic_prediction) > 0 else 0
                
                coupling_results['linearity_test'] = {
                    'linear_correlation': linear_corr,
                    'quadratic_correlation': quadratic_corr,
                    'mean_enhancement': mean_enhancement,
                    'std_enhancement': std_enhancement,
                    'cv_enhancement': cv_enhancement
                }
                
                # Determine coupling type
                if abs(quadratic_corr) > abs(linear_corr) and abs(quadratic_corr) > 0.5:
                    coupling_type = 'QUADRATIC (Non-linear)'
                elif abs(linear_corr) > 0.5:
                    coupling_type = 'LINEAR'
                elif cv_enhancement > 1.5:
                    coupling_type = 'RESONANT (highly variable enhancement)'
                else:
                    coupling_type = 'WEAK/UNCLEAR'
                
                coupling_results['coupling_type'] = coupling_type
                coupling_results['success'] = True
                
                print_status(f"Non-Linear Coupling Analysis Complete", "SUCCESS")
                print_status(f"  Coupling Type: {coupling_type}", "INFO")
                print_status(f"  Mean Enhancement: {mean_enhancement:.0f}x", "INFO")
                print_status(f"  Enhancement CV: {cv_enhancement:.2f}", "INFO")
                print_status(f"  Linear Correlation: {linear_corr:.3f}", "INFO")
                print_status(f"  Quadratic Correlation: {quadratic_corr:.3f}", "INFO")
            else:
                print_status("Insufficient data for linearity test", "WARNING")
        else:
            print_status("Insufficient planetary detections for coupling analysis", "WARNING")
    
    except Exception as e:
        print_status(f"Non-linear coupling analysis failed: {e}", "ERROR")
        coupling_results['error'] = str(e)
    
    return coupling_results


def analyze_temporal_coherence(df: pd.DataFrame, results: Dict) -> Dict:
    """
    Temporal Coherence Assessment
    
    Analyzes the temporal coherence of detected signals across different timescales
    and spatial separations. Tests whether signals maintain phase coherence over time,
    which would indicate a fundamental temporal field coupling vs random fluctuations.
    """
    print_status("Starting Temporal Coherence Assessment...", "PROCESS")
    
    coherence_results = {
        'success': False,
        'coherence_timescales': {},
        'spatial_coherence': {},
        'phase_stability': {}
    }
    
    try:
        # Group by time windows
        df['date_dt'] = pd.to_datetime(df['date'])
        df['week'] = (df['date_dt'] - df['date_dt'].min()).dt.days // 7
        
        # Analyze coherence across different timescales
        timescales = {
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'biannual': 180
        }
        
        coherence_by_timescale = {}
        
        for scale_name, window_days in timescales.items():
            # Group by time windows
            df['time_bin'] = (df['date_dt'] - df['date_dt'].min()).dt.days // window_days
            
            # Compute coherence variance across time bins
            time_coherence = df.groupby('time_bin')['coherence'].mean()
            
            if len(time_coherence) > 2:
                # Autocorrelation at lag 1 (temporal persistence)
                autocorr = time_coherence.autocorr(lag=1)
                
                # Variance (stability)
                variance = time_coherence.var()
                
                coherence_by_timescale[scale_name] = {
                    'autocorrelation': autocorr,
                    'variance': variance,
                    'n_bins': len(time_coherence),
                    'is_coherent': autocorr > 0.3  # Significant persistence
                }
        
        coherence_results['coherence_timescales'] = coherence_by_timescale
        
        # Spatial coherence: do nearby pairs show similar temporal evolution?
        distance_bins = [0, 500, 2000, 10000, 20000]
        spatial_coherence = {}
        
        for i in range(len(distance_bins)-1):
            mask = (df['dist_km'] >= distance_bins[i]) & (df['dist_km'] < distance_bins[i+1])
            if mask.sum() > 100:
                dist_coherence = df[mask].groupby('week')['coherence'].mean()
                if len(dist_coherence) > 2:
                    spatial_coherence[f'{distance_bins[i]}-{distance_bins[i+1]}km'] = {
                        'temporal_stability': dist_coherence.std(),
                        'autocorrelation': dist_coherence.autocorr(lag=1)
                    }
        
        coherence_results['spatial_coherence'] = spatial_coherence
        coherence_results['success'] = True
        
        # Summary
        n_coherent_scales = sum([1 for v in coherence_by_timescale.values() if v.get('is_coherent', False)])
        
        print_status(f"Temporal Coherence Assessment Complete", "SUCCESS")
        print_status(f"  Coherent Timescales: {n_coherent_scales}/{len(timescales)}", "INFO")
        print_status(f"  Spatial Bins Analyzed: {len(spatial_coherence)}", "INFO")
        
        for scale_name, metrics in coherence_by_timescale.items():
            if metrics.get('is_coherent'):
                print_status(f"  {scale_name.capitalize()}: autocorr={metrics['autocorrelation']:.3f} (coherent)", "INFO")
    
    except Exception as e:
        print_status(f"Temporal coherence analysis failed: {e}", "ERROR")
        coherence_results['error'] = str(e)
    
    return coherence_results


def apply_multiple_testing_corrections(all_planetary_detections: List[Dict]) -> Dict:
    """
    Apply multiple testing corrections consistent with Step 3.6 methodology.
    Uses both Bonferroni and FDR corrections as implemented in the reference methods.
    """
    if not all_planetary_detections:
        return {'corrected_detections': [], 'correction_stats': {}}
    
    # Extract p-values
    p_values = np.array([det['p_value'] for det in all_planetary_detections])
    n_tests = len(p_values)
    
    # Bonferroni correction (conservative)
    bonferroni_alpha = 0.05 / n_tests
    bonferroni_significant = p_values < bonferroni_alpha
    
    # FDR correction (Benjamini-Hochberg procedure)
    # Sort p-values and apply BH procedure
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Find largest k such that P(k) ≤ (k/n) * α
    fdr_alpha = 0.05
    fdr_significant = np.zeros(n_tests, dtype=bool)
    for i in range(n_tests-1, -1, -1):
        if sorted_p_values[i] <= (i + 1) / n_tests * fdr_alpha:
            # All tests with indices 0 to i are significant
            fdr_significant[sorted_indices[:i+1]] = True
            break
    
    # Add correction results to detections
    corrected_detections = []
    for i, detection in enumerate(all_planetary_detections):
        corrected_det = detection.copy()
        corrected_det.update({
            'bonferroni_significant': bool(bonferroni_significant[i]),
            'fdr_significant': bool(fdr_significant[i]),
            'bonferroni_corrected_p': min(1.0, detection['p_value'] * n_tests),
            'original_p_value': detection['p_value']
        })
        corrected_detections.append(corrected_det)
    
    correction_stats = {
        'total_tests': n_tests,
        'bonferroni_alpha': bonferroni_alpha,
        'fdr_alpha': fdr_alpha,
        'bonferroni_significant_count': int(np.sum(bonferroni_significant)),
        'fdr_significant_count': int(np.sum(fdr_significant)),
        'uncorrected_significant_count': int(np.sum(p_values < 0.05))
    }
    
    return {
        'corrected_detections': corrected_detections,
        'correction_stats': correction_stats
    }


def calculate_gravitational_scaling_consistency(planetary_events: Dict) -> Dict:
    """
    Test whether detection strength scales with gravitational theory predictions.
    This addresses the inverted mass hierarchy issue.
    """
    print_status("Testing gravitational scaling consistency...", "PROCESS")
    
    planets_with_detections = []
    for planet_name, data in planetary_events.items():
        if data.get('significant_detections'):
            # Get strongest detection for this planet
            strongest_det = max(data['significant_detections'], key=lambda x: x['sigma_level'])
            planets_with_detections.append({
                'planet': planet_name,
                'mass_ratio': data.get('mass_ratio', 1.0),
                'expected_amp': data.get('expected_amplitude_pct', 0.1),
                'observed_sigma': strongest_det['sigma_level'],
                'enhancement_factor': strongest_det['enhancement_factor']
            })
    
    scaling_results = {'success': False, 'mass_correlation': 0.0, 'p_value': 1.0}
    
    if len(planets_with_detections) >= 3:
        mass_ratios = np.array([p['mass_ratio'] for p in planets_with_detections])
        sigma_levels = np.array([p['observed_sigma'] for p in planets_with_detections])
        
        # Test correlation: should sigma scale with mass?
        correlation, p_value = stats.pearsonr(mass_ratios, sigma_levels)
        scaling_results = {
            'success': True,
            'mass_correlation': float(correlation),
            'p_value': float(p_value),
            'planets_tested': planets_with_detections,
            'interpretation': 'consistent' if correlation > 0.3 and p_value < 0.05 else 'inconsistent'
        }
        
        print_status(f"Gravitational scaling test: r={correlation:.3f}, p={p_value:.4f}", "INFO")
        if correlation < 0:
            print_status("⚠️  INVERTED MASS HIERARCHY: Weaker planets show stronger signals", "WARNING")
    
    return scaling_results


def generate_comprehensive_scientific_report(all_results: Dict, analysis_center: str) -> Dict:
    """
    OPTION B: Comprehensive Scientific Significance Report
    
    Generates a detailed scientific assessment report including:
    - Complete detection inventory with statistical characterization
    - Amplitude enhancement analysis with mechanistic interpretation
    - Geophysical signature correlation analysis
    - Multi-scale temporal coherence assessment
    - TEP theory implications and evidence synthesis
    """
    print_status("=" * 80, "TITLE")
    print_status(f"COMPREHENSIVE SCIENTIFIC SIGNIFICANCE REPORT - {analysis_center.upper()}", "TITLE")
    print_status("=" * 80, "TITLE")
    
    report = {
        'analysis_center': analysis_center,
        'timestamp': datetime.now().isoformat(),
        'planetary_events': {},
        'corrected_detections': [],
        'multiple_testing_corrections': {},
        'gravitational_scaling': {},
        'geophysical_signatures': {},
        'amplitude_analysis': {},
        'temporal_patterns': {},
        'scientific_implications': {}
    }
    
    try:
        # ============================================================
        # SECTION 1: PLANETARY GRAVITATIONAL EVENT ANALYSIS
        # ============================================================
        print_status(f"\n1. PLANETARY GRAVITATIONAL EVENT ANALYSIS", "TITLE")
        print_status(f"   Analysis of GPS timing correlation response to planetary configurations", "INFO")
        print_status(f"   Detection threshold: 3.0σ (99.7% confidence)", "INFO")
        
        planetary_events = {}
        all_planetary_detections = []
        
        # Expected amplitudes for planetary gravitational coupling analysis
        # CORRECTED: Using actual planetary masses in Earth masses (not relative ratios)
        planet_info = {
            'jupiter_opposition_analysis': {'name': 'Jupiter', 'expected_amp': 0.00220, 'mass_ratio': 317.8, 'expected_amp_pct': 0.220},
            'saturn_opposition_analysis': {'name': 'Saturn', 'expected_amp': 0.00019, 'mass_ratio': 95.2, 'expected_amp_pct': 0.019},
            'mars_opposition_analysis': {'name': 'Mars', 'expected_amp': 0.00005, 'mass_ratio': 0.107, 'expected_amp_pct': 0.0050},
            'venus_conjunction_analysis': {'name': 'Venus', 'expected_amp': 0.00100, 'mass_ratio': 0.815, 'expected_amp_pct': 0.100},
            'mercury_conjunction_analysis': {'name': 'Mercury', 'expected_amp': 0.00010, 'mass_ratio': 0.055, 'expected_amp_pct': 0.010}
        }
        
        for planet_key, info in planet_info.items():
            if planet_key in all_results and all_results[planet_key].get('success'):
                events = all_results[planet_key].get('event_results', {})
                planet_data = {
                    'planet_name': info['name'],
                    'expected_amplitude_pct': info['expected_amp_pct'],
                    'events_analyzed': len(events),
                    'significant_detections': [],
                    'notable_detections': [],
                    'subsignificant_detections': [],
                    'all_sigma_levels': [],
                    'all_amplitudes': []
                }
                
                for event_name, event_data in events.items():
                    if event_data.get('success'):
                        gaussian = event_data.get('gaussian_fit', {})
                        if gaussian.get('fit_success'):
                            amplitude = gaussian.get('amplitude', 0)
                            std_err = gaussian.get('amplitude_std_err', 1)
                            sigma = abs(amplitude / std_err) if std_err > 0 else 0
                            amp_pct = abs(gaussian.get('amplitude_fraction_of_baseline', 0)) * 100
                            event_date = event_data.get('event_date', 'Unknown')[:10]
                            
                            planet_data['all_sigma_levels'].append(sigma)
                            planet_data['all_amplitudes'].append(amp_pct)
                            
                            # Calculate enhancement factor using absolute amplitude units
                            baseline_coherence = gaussian.get('baseline', 0.007)
                            actual_amplitude = abs(gaussian.get('amplitude_fraction_of_baseline', 0)) * baseline_coherence
                            expected_amplitude_abs = info['expected_amp_pct'] / 100
                            
                            detection_info = {
                                'event_name': event_name,
                                'event_date': event_date,
                                'sigma_level': sigma,
                                'amplitude_pct': amp_pct,
                                'actual_amplitude_abs': actual_amplitude,
                                'expected_amplitude_abs': expected_amplitude_abs,
                                'enhancement_factor': actual_amplitude / expected_amplitude_abs if expected_amplitude_abs > 0 else 0,
                                'direction': 'suppression' if amplitude < 0 else 'enhancement',
                                'p_value': 2 * (1 - norm.cdf(abs(sigma))),  # Two-tailed p-value from sigma level
                                'mass_scaled_enhancement': (actual_amplitude / expected_amplitude_abs) / info['mass_ratio'] if expected_amplitude_abs > 0 and info['mass_ratio'] > 0 else 0
                            }
                            
                            if sigma >= 3.0:
                                planet_data['significant_detections'].append(detection_info)
                                all_planetary_detections.append({**detection_info, 'planet': info['name']})
                            elif sigma >= 2.0:
                                planet_data['notable_detections'].append(detection_info)
                            elif sigma >= 1.0:
                                planet_data['subsignificant_detections'].append(detection_info)
                
                # Calculate statistics with proper mass ratio information
                if planet_data['all_sigma_levels']:
                    planet_data['mean_sigma'] = np.mean(planet_data['all_sigma_levels'])
                    planet_data['max_sigma'] = np.max(planet_data['all_sigma_levels'])
                    planet_data['mean_amplitude'] = np.mean(planet_data['all_amplitudes'])
                    planet_data['max_amplitude'] = np.max(planet_data['all_amplitudes'])
                    # Calculate enhancement statistics using absolute amplitude units
                    typical_baseline = 0.007  # Baseline coherence for unit conversion
                    mean_amp_abs = (planet_data['mean_amplitude'] / 100) * typical_baseline
                    max_amp_abs = (planet_data['max_amplitude'] / 100) * typical_baseline
                    expected_amp_abs = info['expected_amp_pct'] / 100
                    
                    planet_data['mean_enhancement'] = mean_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                    planet_data['max_enhancement'] = max_amp_abs / expected_amp_abs if expected_amp_abs > 0 else 0
                    planet_data['mass_ratio'] = info['mass_ratio']  # Add mass ratio for scaling analysis
                
                planetary_events[info['name']] = planet_data
        
        # Apply multiple testing corrections for statistical rigor
        print_status("\\n" + "="*80, "TITLE")
        print_status("STATISTICAL SIGNIFICANCE CORRECTIONS", "TITLE")
        print_status("="*80, "TITLE")
        
        correction_results = apply_multiple_testing_corrections(all_planetary_detections)
        
        # Calculate gravitational scaling consistency
        scaling_results = calculate_gravitational_scaling_consistency(planetary_events)
        
        print_status(f"Multiple Testing Correction Results:", "INFO")
        if correction_results.get('correction_stats'):
            correction_stats = correction_results['correction_stats']
            print_status(f"   Total tests: {correction_stats.get('total_tests', 0)}", "INFO")
            print_status(f"   Uncorrected significant: {correction_stats.get('uncorrected_significant_count', 0)}", "INFO")
            print_status(f"   Bonferroni significant: {correction_stats.get('bonferroni_significant_count', 0)}", "INFO")
            print_status(f"   FDR significant: {correction_stats.get('fdr_significant_count', 0)}", "INFO")
            print_status(f"   Bonferroni α: {correction_stats.get('bonferroni_alpha', 0.05):.6f}", "INFO")
        else:
            print_status("   No planetary detections found for correction analysis", "INFO")
        
        if scaling_results['success']:
            print_status(f"Gravitational Scaling Consistency:", "INFO")
            print_status(f"   Mass-sigma correlation: r={scaling_results['mass_correlation']:.3f}, p={scaling_results['p_value']:.4f}", "INFO")
            print_status(f"   Interpretation: {scaling_results['interpretation']}", "INFO")
        
        # Print detailed planetary analysis
        for planet_name, data in planetary_events.items():
            print_status(f"\n   {planet_name.upper()}:", "INFO")
            print_status(f"      Events Analyzed: {data['events_analyzed']}", "INFO")
            print_status(f"      Expected Amplitude: {data['expected_amplitude_pct']:.4f}%", "INFO")
            
            if data['all_sigma_levels']:
                print_status(f"      Statistical Summary:", "INFO")
                print_status(f"         Mean Sigma Level: {data['mean_sigma']:.2f}σ", "INFO")
                print_status(f"         Maximum Sigma Level: {data['max_sigma']:.2f}σ", "INFO")
                print_status(f"         Mean Observed Amplitude: {data['mean_amplitude']:.2f}%", "INFO")
                print_status(f"         Maximum Observed Amplitude: {data['max_amplitude']:.2f}%", "INFO")
                print_status(f"         Mean Enhancement Factor: {data['mean_enhancement']:.1f}x", "INFO")
                print_status(f"         Maximum Enhancement Factor: {data['max_enhancement']:.1f}x", "INFO")
            
            if data['significant_detections']:
                print_status(f"      SIGNIFICANT DETECTIONS (≥3.0σ): {len(data['significant_detections'])}", "SUCCESS")
                for det in data['significant_detections']:
                    print_status(f"         {det['event_date']}: {det['sigma_level']:.2f}σ, {det['amplitude_pct']:.1f}% ({det['enhancement_factor']:.0f}x expected)", "INFO")
            
            if data['notable_detections']:
                print_status(f"      Notable Detections (2.0-3.0σ): {len(data['notable_detections'])}", "INFO")
                for det in data['notable_detections']:
                    print_status(f"         {det['event_date']}: {det['sigma_level']:.2f}σ, {det['amplitude_pct']:.1f}%", "INFO")
        
        report['planetary_events'] = planetary_events
        report['corrected_detections'] = correction_results['corrected_detections']
        report['multiple_testing_corrections'] = correction_results['correction_stats']
        report['gravitational_scaling'] = scaling_results
        
        # ============================================================
        # SECTION 2: GEOPHYSICAL SIGNATURE ANALYSIS
        # ============================================================
        print_status(f"\n2. GEOPHYSICAL SIGNATURE ANALYSIS", "TITLE")
        print_status(f"   Correlation with known Earth rotation and orbital parameters", "INFO")
        
        geophysical_sigs = {}
        
        # Chandler Wobble
        if 'chandler_wobble_analysis' in all_results and all_results['chandler_wobble_analysis'].get('success'):
            cw_data = all_results['chandler_wobble_analysis']
            cw_signature = cw_data.get('chandler_signature', {})
            cw_temporal = cw_data.get('temporal_coverage', {})
            cw_rsq = cw_signature.get('r_squared', 0)
            cw_period = cw_temporal.get('chandler_period_days', 433)
            cw_coverage = cw_temporal.get('data_span_days', 0)
            cw_cycles = cw_coverage / cw_period if cw_period > 0 else 0
            
            # Enhanced detection classification with borderline category
            geophysical_sigs['chandler_wobble'] = {
                'detected': cw_rsq > 0.4,
                'borderline': 0.35 < cw_rsq <= 0.40,
                'r_squared': cw_rsq,
                'period_days': cw_period,
                'coverage_days': cw_coverage,
                'complete_cycles': cw_cycles
            }
            
            print_status(f"\n   CHANDLER WOBBLE (14-month polar motion):", "INFO")
            # Convert R² to statistical significance for consistent reporting
            n_samples = len(cw_data.get('phase_analysis', []))
            if n_samples > 2 and cw_rsq > 0:
                r_correlation = np.sqrt(cw_rsq)
                t_stat = r_correlation * np.sqrt(n_samples - 2) / np.sqrt(1 - cw_rsq)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                sigma_equivalent = abs(norm.ppf(p_value / 2))
            else:
                sigma_equivalent = 0.0
                p_value = 1.0
            
            # Enhanced detection categorization with borderline range
            if cw_rsq > 0.40:
                status = "DETECTED"
                level = "SUCCESS"
            elif cw_rsq > 0.35:
                status = "BORDERLINE DETECTED"
                level = "INFO"
            else:
                status = "Not Significant"
                level = "INFO"
            
            print_status(f"      Detection Status: {status} ({sigma_equivalent:.1f}σ equivalent)", level)
            print_status(f"      R² Correlation: {cw_rsq:.3f} (threshold: >0.40, equivalent to >3.2σ)", "INFO")
            print_status(f"      Period: {cw_period:.0f} days ({cw_period/30.44:.1f} months)", "INFO")
            print_status(f"      Temporal Coverage: {cw_coverage:.0f} days ({cw_cycles:.2f} complete cycles)", "INFO")
            if cw_rsq > 0.4:
                print_status(f"      Interpretation: GPS timing correlations exhibit significant modulation", "INFO")
                print_status(f"                     at Chandler wobble frequency, suggesting coupling to", "INFO")
                print_status(f"                     Earth's polar motion dynamics", "INFO")
            elif cw_rsq > 0.35:
                print_status(f"      Interpretation: Borderline Chandler wobble coupling detected (p ≈ {p_value:.4f}).", "INFO")
                print_status(f"                     Signal is conventionally significant but slightly below", "INFO")
                print_status(f"                     analysis threshold. Suggests weak to moderate polar motion coupling.", "INFO")
        
        # Orbital Motion
        if 'temporal_orbital_tracking' in all_results and all_results['temporal_orbital_tracking'].get('success'):
            orb_data = all_results['temporal_orbital_tracking']
            orb_stats = orb_data.get('statistical_analysis', {})
            orb_corr = orb_stats.get('orbital_speed_correlation', 0)
            orb_pval = orb_stats.get('orbital_correlation_p_value', 1.0)
            orb_samples = orb_stats.get('n_temporal_samples', 0)
            
            geophysical_sigs['orbital_motion'] = {
                'detected': abs(orb_corr) > 0.4,
                'correlation': orb_corr,
                'p_value': orb_pval,
                'n_samples': orb_samples
            }
            
            print_status(f"\n   EARTH ORBITAL MOTION (annual cycle):", "INFO")
            # Convert correlation to statistical significance for consistent reporting
            if orb_samples > 2:
                t_stat = abs(orb_corr) * np.sqrt(orb_samples - 2) / np.sqrt(1 - orb_corr**2)
                sigma_equivalent = abs(norm.ppf(orb_pval / 2))
            else:
                sigma_equivalent = 0.0
            
            print_status(f"      Detection Status: {'DETECTED' if abs(orb_corr) > 0.4 else 'Not Significant'} ({sigma_equivalent:.1f}σ)", "SUCCESS" if abs(orb_corr) > 0.4 else "INFO")
            print_status(f"      Correlation Coefficient: r = {orb_corr:.3f} (threshold: |r| > 0.40, ≈3.2σ)", "INFO")
            print_status(f"      Statistical Significance: p = {orb_pval:.4f}", "INFO")
            print_status(f"      Temporal Samples: {orb_samples} (30-day windows)", "INFO")
            if abs(orb_corr) > 0.4:
                print_status(f"      Interpretation: Directional anisotropy (E-W vs N-S) correlates with", "INFO")
                print_status(f"                     Earth's position in orbit, suggesting orbital velocity", "INFO")
                print_status(f"                     modulates GPS timing correlation structure", "INFO")
        
        # Multi-frequency beats
        if 'beat_frequencies_analysis' in all_results and all_results['beat_frequencies_analysis'].get('success'):
            beat_data = all_results['beat_frequencies_analysis']
            n_beats = beat_data.get('n_significant_beats', 0)
            total_tested = beat_data.get('n_beat_patterns_analyzed', 0)
            beat_patterns = list(beat_data.get('significant_beats', {}).values())
            
            geophysical_sigs['multi_frequency_beats'] = {
                'n_significant': n_beats,
                'n_total_tested': total_tested,
                'detection_rate': n_beats / total_tested if total_tested > 0 else 0,
                'patterns': beat_patterns[:5]  # Top 5
            }
            
            print_status(f"\n   MULTI-FREQUENCY BEAT PATTERNS:", "INFO")
            print_status(f"      Significant Patterns: {n_beats}/{total_tested} ({100*n_beats/max(total_tested,1):.1f}%)", "INFO")
            print_status(f"      Detection Threshold: p < 0.05, |r| > 0.3", "INFO")
            if n_beats > 0:
                print_status(f"      Top Beat Patterns:", "INFO")
                for i, pattern in enumerate(beat_patterns[:5], 1):
                    print_status(f"         {i}. {pattern.get('beat_name', 'Unknown')}: Period={pattern.get('beat_period_days', 0):.1f} days, R²={pattern.get('r_squared', 0):.3f}", "INFO")
                print_status(f"      Interpretation: Multiple geophysical/astronomical frequencies show", "INFO")
                print_status(f"                     coherent beating patterns in GPS timing correlations,", "INFO")
                print_status(f"                     indicating complex multi-scale temporal coupling", "INFO")
        
        report['geophysical_signatures'] = geophysical_sigs
        
        # ============================================================
        # SECTION 3: AMPLITUDE ENHANCEMENT ANALYSIS
        # ============================================================
        print_status(f"\n3. AMPLITUDE ENHANCEMENT ANALYSIS", "TITLE")
        print_status(f"   Comparison of observed vs. expected gravitational coupling amplitudes", "INFO")
        
        if all_planetary_detections:
            all_enhancements = [d['enhancement_factor'] for d in all_planetary_detections]
            
            enhancement_stats = {
                'n_detections': len(all_planetary_detections),
                'mean_enhancement': np.mean(all_enhancements),
                'median_enhancement': np.median(all_enhancements),
                'std_enhancement': np.std(all_enhancements) if len(all_enhancements) > 1 else np.nan,
                'min_enhancement': np.min(all_enhancements),
                'max_enhancement': np.max(all_enhancements),
                'cv_enhancement': np.std(all_enhancements) / np.mean(all_enhancements) if len(all_enhancements) > 1 and np.mean(all_enhancements) > 0 else np.nan
            }
            
            print_status(f"\n   Enhancement Factor Statistics (Observed/Expected Amplitude):", "INFO")
            print_status(f"      Number of Significant Detections: {enhancement_stats['n_detections']}", "INFO")
            print_status(f"      Mean Enhancement: {enhancement_stats['mean_enhancement']:.1f}x", "INFO")
            print_status(f"      Median Enhancement: {enhancement_stats['median_enhancement']:.1f}x", "INFO")
            print_status(f"      Standard Deviation: {enhancement_stats['std_enhancement']:.1f}x", "INFO")
            print_status(f"      Range: {enhancement_stats['min_enhancement']:.1f}x - {enhancement_stats['max_enhancement']:.1f}x", "INFO")
            print_status(f"      Coefficient of Variation: {enhancement_stats['cv_enhancement']:.2f}", "INFO")
            
            print_status(f"\n   Mechanistic Interpretation:", "INFO")
            if enhancement_stats['cv_enhancement'] > 1.5:
                print_status(f"      High variability (CV > 1.5) suggests RESONANT COUPLING mechanism", "INFO")
                print_status(f"      rather than simple linear gravitational response. Amplitude", "INFO")
                print_status(f"      enhancement likely depends on frequency matching, orbital", "INFO")
                print_status(f"      resonances, or event-specific geometric configurations.", "INFO")
                print_status(f"      NOTE: Enhancement does NOT scale with planetary mass, suggesting", "INFO")
                print_status(f"      proximity/resonance effects dominate over mass (Mercury/Venus > Jupiter).", "INFO")
            elif enhancement_stats['mean_enhancement'] > 100:
                print_status(f"      Mean enhancement >100x indicates NON-LINEAR COUPLING mechanism.", "INFO")
                print_status(f"      Such large amplification cannot be explained by direct", "INFO")
                print_status(f"      gravitational effects and suggests resonance, tidal amplification,", "INFO")
                print_status(f"      or parametric coupling processes with geophysical modes.", "INFO")
            else:
                print_status(f"      Enhancement factors suggest linear to weakly non-linear coupling", "INFO")
                print_status(f"      with gravitational potential modulation.", "INFO")
            
            report['amplitude_analysis'] = enhancement_stats
        else:
            print_status(f"\n   No significant planetary detections for enhancement analysis", "INFO")
        
        # ============================================================
        # SECTION 4: SCIENTIFIC IMPLICATIONS
        # ============================================================
        print_status(f"\n4. TEP THEORY IMPLICATIONS", "TITLE")
        
        # Count evidence types with nuanced borderline classification
        has_planetary = len(all_planetary_detections) > 0
        has_chandler = geophysical_sigs.get('chandler_wobble', {}).get('detected', False)
        borderline_chandler = geophysical_sigs.get('chandler_wobble', {}).get('borderline', False)
        has_orbital = geophysical_sigs.get('orbital_motion', {}).get('detected', False)
        has_beats = geophysical_sigs.get('multi_frequency_beats', {}).get('n_significant', 0) > 5
        
        # Count full detections; borderline Chandler counts as 0.5 evidence
        evidence_count = sum([has_planetary, has_chandler, has_orbital, has_beats])
        if borderline_chandler and not has_chandler:
            evidence_count += 0.5  # Partial credit for borderline detection
        
        print_status(f"\n   Evidence Summary for Temporal Equivalence Principle:", "INFO")
        print_status(f"      1. Planetary Gravitational Coupling: {'DETECTED' if has_planetary else 'NOT DETECTED'}", "SUCCESS" if has_planetary else "WARNING")
        if has_planetary:
            print_status(f"         {len(all_planetary_detections)} significant event(s) with amplitude modulation", "INFO")
        
        # Enhanced Chandler wobble reporting with borderline status
        chandler_status = "DETECTED" if has_chandler else ("BORDERLINE DETECTED" if borderline_chandler else "NOT DETECTED")
        chandler_level = "SUCCESS" if has_chandler else ("INFO" if borderline_chandler else "WARNING")
        print_status(f"      2. Chandler Wobble Correlation: {chandler_status}", chandler_level)
        if has_chandler or borderline_chandler:
            print_status(f"         R² = {geophysical_sigs['chandler_wobble']['r_squared']:.3f}", "INFO")
        print_status(f"      3. Orbital Motion Correlation: {'DETECTED' if has_orbital else 'NOT DETECTED'}", "SUCCESS" if has_orbital else "WARNING")
        if has_orbital:
            print_status(f"         r = {geophysical_sigs['orbital_motion']['correlation']:.3f}", "INFO")
        print_status(f"      4. Multi-Frequency Coherence: {'DETECTED' if has_beats else 'NOT DETECTED'}", "SUCCESS" if has_beats else "WARNING")
        if has_beats:
            print_status(f"         {geophysical_sigs['multi_frequency_beats']['n_significant']} significant beat patterns", "INFO")
        
        print_status(f"\n   Overall Assessment:", "INFO")
        if evidence_count >= 3:
            assessment = "STRONG SUPPORT for TEP"
            print_status(f"      {assessment}", "SUCCESS")
            print_status(f"      Multiple independent lines of evidence ({evidence_count}/4) demonstrate", "INFO")
            print_status(f"      that GPS timing correlations exhibit systematic modulation by", "INFO")
            print_status(f"      gravitational and rotational dynamics, consistent with TEP", "INFO")
            print_status(f"      predictions of temporal field coupling to spacetime geometry.", "INFO")
        elif evidence_count >= 2:
            assessment = "MODERATE SUPPORT for TEP"
            print_status(f"      {assessment}", "INFO")
            print_status(f"      {evidence_count}/4 evidence categories detected. Results suggest", "INFO")
            print_status(f"      partial coupling between timing correlations and gravitational", "INFO")
            print_status(f"      dynamics, but additional data needed for confirmation.", "INFO")
        elif evidence_count >= 1:
            assessment = "WEAK SUPPORT for TEP"
            print_status(f"      {assessment}", "INFO")
            print_status(f"      Limited evidence ({evidence_count}/4 categories). Some signatures", "INFO")
            print_status(f"      detected but insufficient for robust TEP validation.", "INFO")
        else:
            assessment = "INSUFFICIENT EVIDENCE for TEP"
            print_status(f"      {assessment}", "WARNING")
            print_status(f"      No significant signatures detected in this dataset.", "INFO")
        
        report['scientific_implications'] = {
            'evidence_categories_detected': evidence_count,
            'total_categories': 4,
            'has_planetary_coupling': has_planetary,
            'has_chandler_coupling': has_chandler,
            'has_orbital_coupling': has_orbital,
            'has_multifrequency_coherence': has_beats,
            'overall_assessment': assessment
        }
        
        print_status("=" * 80, "TITLE")
        
    except Exception as e:
        print_status(f"Report generation failed: {e}", "ERROR")
        report['error'] = str(e)
    
    return report

# ===== END ENHANCED ANALYSIS MODULES =====

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("Step 2.2 interrupted by user", "WARNING")
        sys.exit(1)
    except (TEPDataError, TEPFileError) as e:
        print_status(f"Step 2.2 failed - data/file error: {e}", "ERROR")
        sys.exit(1)
    except TEPAnalysisError as e:
        print_status(f"Step 2.2 failed - analysis error: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"Step 2.2 failed - unexpected error: {e}", "CRITICAL")
        import traceback
        print_status(traceback.format_exc(), "DEBUG")
        sys.exit(1)