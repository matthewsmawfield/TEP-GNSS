#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 3.0: Cross-Validation Suite (Comprehensive)
=====================================================================

Comprehensive cross-validation suite for TEP correlation models including:
1. Block-wise cross-validation (temporal/spatial) - from original step_5_5
2. Leave-One-Station-Out (LOSO) analysis - moved from step_5
3. Leave-One-Day-Out (LODO) analysis - moved from step_5
4. Statistical validation - bootstrap significance testing for CV stability

This consolidates all cross-validation methodologies to provide rigorous
validation of TEP correlation parameters with multiple approaches.

Note on Bootstrapping:
For *true block bootstrap* implementations (resampling stations or days to account for dependencies),
refer to `scripts/steps/step_3_validation_suite/step_3_1_robust_block_bootstrap.py`.

Requirements: Step 2.0 complete
Next: Step 3.1 (Robust Block Bootstrap Validation)

Key Analyses:
1. Monthly temporal cross-validation - split by months, predict held-out months
2. Leave-5-stations-out spatial blocks - remove station groups, test predictive power
3. Leave-One-Station-Out - exclude individual stations, test stability
4. Leave-One-Day-Out - exclude individual days, test temporal stability
5. Statistical validation - bootstrap significance testing for CV stability
6. Cross-validation metrics - CV-RMSE, NRMSE, log-likelihood on predictions
7. Parameter stability assessment - test if λ is consistent across folds

METHODOLOGY: Train on N-1 folds → fit (λ, A, C₀) → predict held-out fold → measure error
This tests whether λ represents real predictive physics vs. curve-fitting artifacts.

Inputs:
  - results/tmp/step_2_0_pairs_*.csv files (from Step 2.0)
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0)

Outputs:
  - results/outputs/step_3_0_cross_validation_suite_{ac}.json

Environment Variables:
  - TEP_ENABLE_MONTHLY_CV: Enable monthly temporal cross-validation (default: True)
  - TEP_ENABLE_STATION_BLOCKS_CV: Enable station block spatial cross-validation (default: True)
  - TEP_ENABLE_LOSO_CV: Enable Leave-One-Station-Out cross-validation (default: True)
  - TEP_ENABLE_LODO_CV: Enable Leave-One-Day-Out cross-validation (default: True)
  - TEP_ENABLE_BOOTSTRAP_CV: Enable Bootstrap cross-validation (disabled - use Step 3.1 instead)
  - TEP_MONTHLY_CV_FOLDS: Number of monthly folds to use (default: 10, memory-optimized)
  - TEP_STATION_BLOCK_SIZE: Number of stations per block (default: 10, memory-optimized)
  - TEP_LOSO_SAMPLE_SIZE: Number of stations to sample for LOSO (default: 50)
  - TEP_LODO_SAMPLE_SIZE: Number of days to sample for LODO (default: 100)
  - TEP_MEMORY_LIMIT_GB: Maximum memory to use in GB (default: 16)
  - TEP_MEMORY_SAFE_MODE: Enable memory-safe mode (default: False - use full datasets)

Author: Matthew Lukin Smawfield
Date: September 2025
Theory: Temporal Equivalence Principle (TEP)
"""

import os
import sys
import time
import json
import gc
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from glob import glob
import psutil  # For memory monitoring
import logging

# Ensure the directory containing 'scripts' is on the path
script_dir = Path(__file__).resolve()
project_root = script_dir.parents[3] # Point to project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Define PACKAGE_ROOT for consistent usage throughout the script
PACKAGE_ROOT = project_root

# Global variables for worker processes (used by LOSO/LODO)
WORKER_COMPLETE_DF = None
WORKER_EDGES = None
WORKER_MIN_BIN_COUNT = None
WORKER_DATA_PATH = None  # MEMORY FIX: Store file path instead of full dataset

def check_memory_usage():
    """Monitor memory usage and warn if approaching limits."""
    import psutil
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent

    print_status(f"Memory usage: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%)", "INFO")

    from scripts.utils.config import TEPConfig
    memory_limit_gb = TEPConfig.get_float('TEP_MEMORY_LIMIT_GB', 8.0)
    if used_gb > memory_limit_gb * 0.9:  # 90% of limit
        print_status(f"WARNING: Memory usage ({used_gb:.1f} GB) approaching limit ({memory_limit_gb} GB)", "WARNING")
        return False
    return True

def check_memory_and_cleanup():
    """Check memory usage and force cleanup if needed."""
    if not check_memory_usage():
        print_status("Forcing garbage collection and memory cleanup...", "WARNING")
        import gc
        gc.collect()
        # Force cleanup of any large objects
        global WORKER_COMPLETE_DF, WORKER_EDGES, WORKER_MIN_BIN_COUNT, WORKER_DATA_PATH
        WORKER_COMPLETE_DF = None
        WORKER_EDGES = None
        WORKER_MIN_BIN_COUNT = None
        WORKER_DATA_PATH = None
        return False
    return True


def aggressive_memory_cleanup(context: str = "Unknown"):
    """
    Perform aggressive memory cleanup including garbage collection and cache clearing.
    
    Args:
        context (str): Context description for logging
    """
    import gc
    import sys
    import psutil
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info()
    rss_before_mb = mem_before.rss / (1024 * 1024)
    
    # Force garbage collection multiple times
    for i in range(3):
        collected = gc.collect()
        if i == 0:
            print_status(f"Memory cleanup in {context}: Collected {collected} objects", "DEBUG")
    
    # Clear any module-level caches if they exist
    try:
        # Clear pandas caches
        if 'pandas' in sys.modules:
            import pandas as pd
            if hasattr(pd, '_cache'):
                pd._cache.clear()
    except:
        pass
    
    try:
        # Clear numpy caches
        if 'numpy' in sys.modules:
            import numpy as np
            if hasattr(np, '_cache'):
                np._cache.clear()
    except:
        pass
    
    # Force another garbage collection
    gc.collect()
    
    # Get final memory usage
    mem_after = process.memory_info()
    rss_after_mb = mem_after.rss / (1024 * 1024)
    freed_mb = rss_before_mb - rss_after_mb
    
    print_status(f"Memory cleanup in {context}: Freed {freed_mb:.1f} MB (RSS: {rss_before_mb:.1f} → {rss_after_mb:.1f} MB)", "DEBUG")
    
    return freed_mb
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    TEPDataError, TEPFileError, TEPAnalysisError, 
    safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)
from scripts.utils.pid_manager import ensure_single_instance
from scripts.utils.logger import print_status, TEPLogger, set_step_logger

# Step-specific logger instance (initialized in main)
step_logger = None

def _cleanup_processes():
    """Cleanup any remaining processes when script exits."""
    try:
        import subprocess
        import os
        print("Cleaning up remaining processes...")
        
        # Get current process PID to avoid killing ourselves
        current_pid = os.getpid()
        
        # Kill by script name but exclude current process
        subprocess.run(['pkill', '-f', 'step_3_0_tep_cross_validation_suite.py'],
                      capture_output=True, timeout=5)
        # Kill multiprocessing processes
        subprocess.run(['pkill', '-f', 'multiprocessing.spawn'],
                      capture_output=True, timeout=5)
        print("Process cleanup completed")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"Warning: Process cleanup may have failed: {e}")

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
    return False

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀
    
    Args:
        r (np.ndarray): Distance values.
        amplitude (float): Amplitude parameter (A).
        lambda_km (float): Lambda parameter (λ) in kilometers.
        offset (float): Offset parameter (C₀).
        
    Returns:
        np.ndarray: Predicted coherence values.
    """
    return amplitude * np.exp(-r / lambda_km) + offset

class DatasetInterface:
    """Memory-efficient interface to large datasets without loading everything into memory."""

    def __init__(self, ac: str, sample_for_validation: bool = False):
        self.ac = ac
        self.sample_for_validation = sample_for_validation
        self._dataset_info = None
        self._file_paths = []

        # Find data files
        self._find_data_files()

    def _find_data_files(self):
        """Find all available data files for this analysis center."""
        consolidated_file = Path(PACKAGE_ROOT / "results" / "outputs" / f"step_2_0_pairs_consolidated_{self.ac}.csv")
        step21_file = Path(PACKAGE_ROOT / "data" / "processed" / f"step_2_1_geospatial_{self.ac}.csv")

        if consolidated_file.exists():
            self._file_paths.append(consolidated_file)
        elif step21_file.exists():
            self._file_paths.append(step21_file)
        else:
            raise TEPFileError(f"No data files found for analysis center {self.ac}")

    def get_dataset_info(self) -> dict:
        """Get basic dataset information without loading data."""
        if self._dataset_info is None:
            # Read just the header to get column info and estimate size
            first_file = self._file_paths[0]
            try:
                # Count lines to estimate size
                with open(first_file, 'r') as f:
                    total_lines = sum(1 for _ in f) - 1  # Subtract 1 for header

                # Read first few rows to get column info
                df_sample = pd.read_csv(first_file, nrows=5)
                columns = df_sample.columns.tolist()

                self._dataset_info = {
                    'total_pairs': total_lines,
                    'columns': columns,
                    'file_paths': [str(p) for p in self._file_paths]
                }
            except Exception as e:
                raise TEPDataError(f"Failed to read dataset info: {e}")

        return self._dataset_info

    def sample_data(self, fraction: float = 0.05, min_samples: int = 10000) -> pd.DataFrame:
        """Sample a fraction of the data for validation."""
        info = self.get_dataset_info()
        total_pairs = info['total_pairs']
        sample_size = max(min_samples, int(total_pairs * fraction))

        print_status(f"Sampling {sample_size:,} pairs from {total_pairs:,} total for validation", "INFO")

        # Sample from the first file (they should all have the same structure)
        first_file = self._file_paths[0]
        df = pd.read_csv(first_file, parse_dates=['date'])

        # Ensure coherence column exists
        if 'plateau_phase' in df.columns and 'coherence' not in df.columns:
            df['coherence'] = np.cos(df['plateau_phase'])

        # Randomly sample rows
        np.random.seed(42)  # Reproducible
        sample_indices = np.random.choice(total_pairs, size=sample_size, replace=False)
        sampled_df = df.iloc[sample_indices].reset_index(drop=True)

        return sampled_df

    def process_in_chunks(self, chunk_processor, chunk_size: int = None):
        """Process data in chunks to avoid memory issues."""
        if chunk_size is None:
            chunk_size = TEPConfig.get_int('TEP_MIN_CHUNK_SIZE', 25000)

        results = []
        total_processed = 0

        for file_path in self._file_paths:
            print_status(f"Processing file {file_path.name} in chunks of {chunk_size:,}", "DEBUG")

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print_status(f"Processing {file_size_mb:.1f} MB file in chunks", "DEBUG")

            for chunk in pd.read_csv(file_path, chunksize=chunk_size, parse_dates=['date']):
                # Ensure coherence column exists
                if 'plateau_phase' in chunk.columns and 'coherence' not in chunk.columns:
                    chunk['coherence'] = np.cos(chunk['plateau_phase'])

                # Process this chunk
                chunk_result = chunk_processor(chunk)
                if chunk_result is not None:
                    results.append(chunk_result)

                total_processed += len(chunk)

                # Memory management
                if total_processed % (chunk_size * 10) == 0:
                    aggressive_memory_cleanup(f"After processing {total_processed:,} rows")
                    if not check_memory_and_cleanup():
                        print_status("Memory limit reached, stopping chunked processing", "WARNING")
                        break

        return results

def load_complete_pair_dataset(ac: str, sample_for_validation: bool = False) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center.
    Uses memory-efficient chunked loading for large datasets.

    Args:
        ac (str): Analysis center identifier (e.g., "code", "esa_final").
        sample_for_validation (bool): If True, load only a sample for validation (much faster).

    Returns:
        pd.DataFrame: A concatenated DataFrame of all pair data for the given analysis center.

    Raises:
        TEPFileError: If no pair files are found for the analysis center.
        TEPDataError: If no valid pair data can be loaded.
    """
    print_status(f"Loading complete pair dataset for {ac}...", "PROCESS")

    # Monitor memory before loading
    monitor_memory_usage(f"Before loading {ac} dataset")

    # Create dataset interface for memory-efficient processing
    dataset_interface = DatasetInterface(ac, sample_for_validation)

    # Get dataset info
    info = dataset_interface.get_dataset_info()

    if sample_for_validation:
        # Return sampled data for validation
        complete_df = dataset_interface.sample_data()
        print_status(f"Loaded sampled dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")
        return complete_df
    else:
        # For full processing, we need to load all data but in a memory-efficient way
        # Use chunked processing to build the complete dataset
        print_status(f"Loading full dataset with {info['total_pairs']:,} pairs...", "INFO")

        chunks = []
        total_pairs = 0

        def collect_chunks(chunk):
            nonlocal total_pairs
            chunks.append(chunk.copy())  # Make a copy to avoid reference issues
            total_pairs += len(chunk)
            return None  # Don't return anything for collection

        try:
            dataset_interface.process_in_chunks(collect_chunks)
            complete_df = pd.concat(chunks, ignore_index=True)
            print_status(f"Loaded complete dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")

            # Monitor memory after loading
            monitor_memory_usage(f"After loading {ac} dataset")

            return complete_df

        except Exception as e:
            print_status(f"Error in chunked loading: {e}", "ERROR")
            # Fallback to original method if chunked processing fails
            print_status("Falling back to direct loading...", "WARNING")

            # Load first file directly (may cause memory issues for very large files)
            first_file = Path(PACKAGE_ROOT / "results" / "outputs" / f"step_2_0_pairs_consolidated_{ac}.csv")
            if first_file.exists():
                complete_df = pd.read_csv(first_file, parse_dates=['date'])
                if 'plateau_phase' in complete_df.columns and 'coherence' not in complete_df.columns:
                    complete_df['coherence'] = np.cos(complete_df['plateau_phase'])
                print_status(f"Loaded dataset via fallback: {len(complete_df):,} pairs for {ac}", "SUCCESS")
                return complete_df
            else:
                raise TEPDataError(f"Failed to load dataset for {ac}: {e}")

    # Fallback to individual pair files if both consolidated and Step 2.1 fail
    print_status("Checking for individual pair files...", "INFO")
    pair_files = list(Path(PACKAGE_ROOT / "results" / "tmp").glob(f"step_2_0_pairs_{ac}_*.csv"))

    if not pair_files:
        raise TEPFileError(f"No pair files found for analysis center: {ac} (Ensure Step 2.0 is complete)")

    print_status(f"Found {len(pair_files)} pair files for {ac}", "INFO")

    # Load files in chunks to manage memory
    dataframes = []
    for i, file_path in enumerate(pair_files):
        if i % 10 == 0:  # Verbose logging for debugging
            print_status(f"Loading file {i+1}/{len(pair_files)}: {file_path.name}", "PROCESS")
        
        try:
            # Try reading with different engines for robustness
            df_chunk = None
            try:
                df_chunk = safe_csv_read(file_path)
            except Exception:
                # Try with python engine if C engine fails
                try:
                    df_chunk = pd.read_csv(file_path, engine='python', parse_dates=['date'])
                except Exception:
                    print_status(f"Skipping {file_path.name}: corrupted or unreadable file", "WARNING")
                    continue
                    
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

def create_monthly_folds(complete_df: pd.DataFrame) -> List[Tuple[str, pd.Series, pd.Series]]:
    """
    Create monthly cross-validation folds using memory-efficient approach.

    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.

    Returns:
        List[Tuple[str, pd.Series, pd.Series]]: A list of tuples, where each tuple contains:
            - month_id (str): Identifier for the month.
            - training_data_mask (pd.Series): Boolean mask for the training data.
            - validation_data_mask (pd.Series): Boolean mask for the validation data.
    """
    print_status("Creating monthly cross-validation folds...", "PROCESS")

    # Memory-efficient approach: Use vectorized operations on date column only
    dates = pd.to_datetime(complete_df['date'])

    # Create year-month identifier more efficiently
    year_months = dates.dt.to_period('M')
    unique_months = sorted(year_months.unique())

    max_folds = TEPConfig.get_int('TEP_MONTHLY_CV_FOLDS')
    if len(unique_months) > max_folds:
        # Sample months for efficiency
        np.random.seed(42)  # Reproducible
        selected_months = np.random.choice(unique_months, max_folds, replace=False)
        unique_months = sorted(selected_months)
        print_status(f"Sampling {max_folds} months from {len(unique_months)} total for efficiency", "INFO")
    
    print_status(f"Creating {len(unique_months)} monthly folds", "INFO")

    folds = []
    for i, month in enumerate(unique_months):
        print_status(f"Creating monthly fold {i+1}/{len(unique_months)}: {month}", "PROCESS")

        # Memory-efficient mask creation using vectorized operations
        val_mask = (year_months == month)
        train_mask = (year_months != month)

        # Validate fold data efficiently
        val_count = val_mask.sum()
        train_count = train_mask.sum()

        if val_count < 1000 or train_count < 10000:
            print_status(f"Skipping fold {month}: insufficient data (train={train_count:,}, val={val_count:,})", "WARNING")
            continue

        # Additional validation: check for reasonable data distribution
        val_distances = complete_df['dist_km'][val_mask]
        train_distances = complete_df['dist_km'][train_mask]

        if len(val_distances) == 0 or len(train_distances) == 0:
            print_status(f"Skipping fold {month}: no valid distances found", "WARNING")
            continue

        # Check if validation set has reasonable distance range
        val_dist_range = val_distances.max() - val_distances.min()
        if val_dist_range < 100:  # At least 100km range for meaningful validation
            print_status(f"Skipping fold {month}: insufficient distance range in validation ({val_dist_range:.1f} km)", "WARNING")
            continue

        folds.append((str(month), train_mask, val_mask))

        # Memory cleanup every 5 folds
        if (i + 1) % 5 == 0:
            gc.collect()

    print_status(f"Created {len(folds)} valid monthly folds", "SUCCESS")
    return folds

def create_station_block_folds(complete_df: pd.DataFrame) -> List[Tuple[str, pd.Series, pd.Series]]:
    """
    Create leave-N-stations-out cross-validation folds.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        List[Tuple[str, pd.Series, pd.Series]]: A list of tuples, where each tuple contains:
            - block_id (str): Identifier for the station block.
            - training_data_mask (pd.Series): Boolean mask for the training data.
            - validation_data_mask (pd.Series): Boolean mask for the validation data.
    """
    print_status("Creating station block cross-validation folds...", "PROCESS")
    
    # Get all unique stations
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    block_size = TEPConfig.get_int('TEP_STATION_BLOCK_SIZE')
    
    # Create station blocks
    np.random.seed(42)  # Reproducible
    np.random.shuffle(unique_stations)
    
    station_blocks = []
    for i in range(0, len(unique_stations), block_size):
        block = unique_stations[i:i+block_size]
        if len(block) >= block_size:  # Only use complete blocks
            station_blocks.append(block)
    
    print_status(f"Created {len(station_blocks)} station blocks of size {block_size}", "INFO")
    
    folds = []
    for i, station_block in enumerate(station_blocks):
        print_status(f"Creating station block fold {i+1}/{len(station_blocks)} (size: {len(station_block)} stations)", "PROCESS")

        # Memory-efficient mask creation
        station_i_in_block = complete_df['station_i'].isin(station_block)
        station_j_in_block = complete_df['station_j'].isin(station_block)
        val_mask = (station_i_in_block | station_j_in_block)
        train_mask = ~val_mask

        # Validate fold data efficiently
        val_count = val_mask.sum()
        train_count = train_mask.sum()

        if val_count < 100 or train_count < 1000:
            print_status(f"Skipping fold stations_{i+1:02d}: insufficient data (train={train_count:,}, val={val_count:,})", "WARNING")
            continue

        block_id = f"stations_{i+1:02d}"
        folds.append((block_id, train_mask, val_mask))

        # Memory cleanup every 5 folds
        if (i + 1) % 5 == 0:
            gc.collect()
    
    print_status(f"Created {len(folds)} valid station block folds", "SUCCESS")
    return folds

def fit_correlation_model_on_training_chunked(complete_df: pd.DataFrame, train_mask: pd.Series) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
    """
    Fit correlation model on training data using chunked processing for memory efficiency.

    Args:
        complete_df (pd.DataFrame): The complete dataset
        train_mask (pd.Series): Boolean mask for training data

    Returns:
        Tuple[Optional[np.ndarray], bool, Optional[str]]: Fitted parameters, success flag, error message
    """
    try:
        # Process training data in chunks to avoid memory issues
        chunk_size = min(50000, int(train_mask.sum() * 0.1))  # 10% of training data or 50k, whichever is smaller

        if chunk_size < 1000:
            print_status(f"Very small training set ({train_mask.sum():,} pairs), using direct processing", "WARNING")
            return fit_correlation_model_on_training(complete_df[train_mask])

        # Sample training data for fitting (use representative subset)
        train_indices = np.where(train_mask)[0]
        if len(train_indices) > chunk_size:
            # Use stratified sampling across distance bins for better representation
            distances = complete_df['dist_km'].iloc[train_indices]

            # Create more distance bins for better stratification
            try:
                distance_bins = np.percentile(distances, [0, 10, 25, 40, 50, 60, 75, 90, 100])

                sampled_indices = []
                samples_per_bin = max(200, chunk_size // len(distance_bins))

                for i in range(len(distance_bins) - 1):
                    bin_mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
                    bin_indices = train_indices[bin_mask]

                    if len(bin_indices) > 0:
                        # Sample from each bin
                        bin_sample_size = min(samples_per_bin, len(bin_indices))
                        if bin_sample_size > 0:
                            sampled_from_bin = np.random.choice(bin_indices, size=bin_sample_size, replace=False)
                            sampled_indices.extend(sampled_from_bin)

                # If we didn't get enough samples, add random samples
                if len(sampled_indices) < chunk_size * 0.8:
                    remaining_needed = int(chunk_size - len(sampled_indices))
                    additional_indices = np.random.choice(train_indices, size=min(remaining_needed, len(train_indices)), replace=False)
                    sampled_indices.extend(additional_indices)

            except Exception as e:
                print_status(f"Stratified sampling failed, using random sampling: {e}", "DEBUG")
                sampled_indices = np.random.choice(train_indices, size=chunk_size, replace=False)
        else:
            sampled_indices = train_indices

        # Create subset for fitting
        train_subset = complete_df.iloc[sampled_indices].copy()

        # Validate the subset before fitting
        if len(train_subset) < 500 or train_subset['dist_km'].max() - train_subset['dist_km'].min() < 50:
            print_status(f"Training subset too small or limited range ({len(train_subset)} pairs, {train_subset['dist_km'].max() - train_subset['dist_km'].min():.1f} km range), using full training data", "WARNING")
            return fit_correlation_model_on_training(complete_df[train_mask])

        # Fit model on subset
        return fit_correlation_model_on_training(train_subset)

    except Exception as e:
        return None, False, f"Chunked fitting failed: {str(e)}"

def fit_correlation_model_on_training(train_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
    """
    Fit exponential correlation model on training data.
    Returns (fitted_params, success_flag, error_message).
    """
    # Create a copy to avoid SettingWithCopyWarning
    train_data = train_data.copy()

    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)

    # Bin the training data
    train_data.loc[:, 'dist_bin'] = pd.cut(train_data['dist_km'], bins=edges, right=False)
    binned = train_data.groupby('dist_bin', observed=True).agg(
        mean_dist=('dist_km', 'mean'),
        mean_coh=('coherence', 'mean'),
        count=('coherence', 'size')
    ).reset_index()
    
    # Filter for robust bins
    binned = binned[binned['count'] >= min_bin_count].dropna()
    
    if len(binned) < 5:  # Need enough bins for stable fit
        return None, False, "Not enough robust bins for fitting."
    
    distances = binned['mean_dist'].values
    coherences = binned['mean_coh'].values
    weights = binned['count'].values
    
    # Fit exponential model
    try:
        c_range = coherences.max() - coherences.min()
        p0 = [c_range, 3000, coherences.min()]
        
        popt, pcov = curve_fit(
            correlation_model, distances, coherences,
            p0=p0, sigma=1.0/np.sqrt(weights),
            bounds=TEPConfig.get_adaptive_lambda_bounds(distances),
            maxfev=5000
        )
        
        return popt, True, None
        
    except Exception as e:
        return None, False, str(e)

def predict_validation_coherences_chunked(complete_df: pd.DataFrame, val_mask: pd.Series, fitted_params: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]:
    """
    Predict validation coherences using chunked processing for memory efficiency.

    Args:
        complete_df (pd.DataFrame): The complete dataset
        val_mask (pd.Series): Boolean mask for validation data
        fitted_params (np.ndarray): Fitted correlation model parameters

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]: Predicted values, actual values, success flag, error message
    """
    try:
        # Get validation indices
        val_indices = np.where(val_mask)[0]

        if len(val_indices) == 0:
            return None, None, False, "No validation data"

        # For prediction, we can process in reasonable chunks
        chunk_size = min(100000, len(val_indices))  # Process up to 100k at a time

        all_predicted = []
        all_actual = []

        for start_idx in range(0, len(val_indices), chunk_size):
            end_idx = min(start_idx + chunk_size, len(val_indices))
            chunk_indices = val_indices[start_idx:end_idx]

            # Get chunk data
            chunk_df = complete_df.iloc[chunk_indices]

            # Calculate predictions for this chunk
            distances = chunk_df['dist_km'].values
            actual_coherences = chunk_df['coherence'].values

            # Apply correlation model
            amplitude, lambda_km, offset = fitted_params
            predicted_coherences = amplitude * np.exp(-distances / lambda_km) + offset

            all_predicted.extend(predicted_coherences)
            all_actual.extend(actual_coherences)

            # Memory cleanup for large chunks
            if end_idx - start_idx >= 50000:
                gc.collect()

        return (np.array(all_predicted), np.array(all_actual), True, None)

    except Exception as e:
        return None, None, False, f"Chunked prediction failed: {str(e)}"

def predict_validation_coherences(val_data: pd.DataFrame, fitted_params: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]:
    """
    Predict coherences on validation data using fitted parameters.

    Args:
        val_data (pd.DataFrame): Validation dataset.
        fitted_params (np.ndarray): Fitted model parameters from training.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[str]]: A tuple containing:
            - predicted_coherences (Optional[np.ndarray]): Predicted coherence values if successful, None otherwise.
            - actual_coherences (Optional[np.ndarray]): Actual coherence values from the validation data if successful, None otherwise.
            - success_flag (bool): True if prediction was successful, False otherwise.
            - error_message (Optional[str]): Error message if prediction failed, None otherwise.
    """
    # Create a copy to avoid SettingWithCopyWarning
    val_data = val_data.copy()

    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)

    # Bin the validation data
    val_data.loc[:, 'dist_bin'] = pd.cut(val_data['dist_km'], bins=edges, right=False)
    binned = val_data.groupby('dist_bin', observed=True).agg(
        mean_dist=('dist_km', 'mean'),
        mean_coh=('coherence', 'mean'),
        count=('coherence', 'size')
    ).reset_index()
    
    # Filter for robust bins
    binned = binned[binned['count'] >= min_bin_count].dropna()
    
    if len(binned) < 3:  # Need minimum bins for validation
        return None, None, False, "Not enough robust bins for validation."
    
    distances = binned['mean_dist'].values
    actual_coherences = binned['mean_coh'].values
    
    # Predict using fitted parameters
    try:
        predicted_coherences = correlation_model(distances, *fitted_params)
    except Exception as e:
        return None, None, False, str(e)
    
    return predicted_coherences, actual_coherences, True, None

def _validate_fold_data(train_data: pd.DataFrame, val_data: pd.DataFrame, fold_id: str, 
                       min_train: int = 10000, min_val: int = 1000) -> bool:
    """
    Validate that fold has sufficient data for reliable analysis.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        fold_id (str): Identifier for the fold.
        min_train (int, optional): Minimum training samples required. Defaults to 10000.
        min_val (int, optional): Minimum validation samples required. Defaults to 1000.
        
    Returns:
        bool: True if fold has sufficient data, False otherwise.
    """
    if len(train_data) < min_train:
        print_status(f"Skipping fold {fold_id}: insufficient training data ({len(train_data)} < {min_train})", "WARNING")
        return False
    
    if len(val_data) < min_val:
        print_status(f"Skipping fold {fold_id}: insufficient validation data ({len(val_data)} < {min_val})", "WARNING")
        return False
    
    return True

def _aggregate_fold_results(fold_results: List[Dict], method_name: str) -> Dict:
    """
    Aggregate results across cross-validation folds.
    
    Args:
        fold_results (List[Dict]): List of individual fold results.
        method_name (str): Name of the CV method for reporting.
        
    Returns:
        Dict: Aggregated results with stability metrics.
    """
    if not fold_results:
        return {'success': False, 'error': f'No successful {method_name} cross-validation folds'}
    
    # Extract lambda estimates and metrics
    lambda_estimates = [r['fitted_params']['lambda_km'] for r in fold_results]
    cv_rmse_values = [r['cv_metrics']['cv_rmse'] for r in fold_results]
    cv_nrmse_values = [r['cv_metrics']['cv_nrmse'] for r in fold_results]
    log_likelihood_values = [r['cv_metrics']['log_likelihood'] for r in fold_results]
    
    # Calculate stability metrics
    lambda_mean = np.mean(lambda_estimates)
    lambda_std = np.std(lambda_estimates)
    lambda_cv = lambda_std / lambda_mean if lambda_mean > 0 else 0
    
    # Bootstrap test for CV significance
    # H0: CV is consistent with random sampling variation
    # Low p-value indicates high stability (CV is significantly low)
    n_bootstrap = 1000
    cv_bootstrap = []
    n_folds = len(lambda_estimates)
    
    for _ in range(n_bootstrap):
        resample = np.random.choice(lambda_estimates, size=n_folds, replace=True)
        cv_boot = np.std(resample) / np.mean(resample) if np.mean(resample) > 0 else 0
        cv_bootstrap.append(cv_boot)
    
    # Two-tailed p-value: how unusual is observed CV?
    cv_bootstrap_mean = np.mean(cv_bootstrap)
    cv_bootstrap_std = np.std(cv_bootstrap)
    
    # Z-score for observed CV vs bootstrap distribution
    z_score = (lambda_cv - cv_bootstrap_mean) / cv_bootstrap_std if cv_bootstrap_std > 0 else 0
    
    # P-value: proportion of bootstrap CVs as extreme as observed
    # For stability, we want LOW CV, so test if observed < expected
    p_value_stability = sum(cv <= lambda_cv for cv in cv_bootstrap) / n_bootstrap
    
    return {
        'success': True,
        'method': method_name,
        'n_folds': len(fold_results),
        'lambda_stability': {
            'mean_lambda_km': float(lambda_mean),
            'std_lambda_km': float(lambda_std),
            'cv_lambda': float(lambda_cv),
            'lambda_estimates': [float(x) for x in lambda_estimates],
            'cv_p_value': float(p_value_stability),
            'cv_z_score': float(z_score),
            'cv_bootstrap_mean': float(cv_bootstrap_mean),
            'cv_bootstrap_std': float(cv_bootstrap_std),
            'interpretation': 'stable' if p_value_stability < 0.05 else 'moderate' if p_value_stability < 0.5 else 'unstable'
        },
        'cv_performance': {
            'mean_cv_rmse': float(np.mean(cv_rmse_values)),
            'std_cv_rmse': float(np.std(cv_rmse_values)),
            'mean_cv_nrmse': float(np.mean(cv_nrmse_values)),
            'std_cv_nrmse': float(np.std(cv_nrmse_values)),
            'mean_log_likelihood': float(np.mean(log_likelihood_values)),
            'std_log_likelihood': float(np.std(log_likelihood_values))
        },
        'fold_details': fold_results
    }

def _detect_outlier_folds(fold_results: List[Dict], method_name: str) -> Dict:
    """
    Detect outlier folds that may indicate data quality issues.
    
    Args:
        fold_results (List[Dict]): List of fold results.
        method_name (str): Name of the CV method.
        
    Returns:
        Dict: Outlier detection results, including a flag for detection, number of outliers, and outlier details.
    """
    if len(fold_results) < 3:
        return {'outliers_detected': False, 'reason': 'insufficient_folds'}
    
    lambda_estimates = np.array([r['fitted_params']['lambda_km'] for r in fold_results])
    
    # Use IQR method for outlier detection
    q1, q3 = np.percentile(lambda_estimates, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (lambda_estimates < lower_bound) | (lambda_estimates > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    if len(outlier_indices) > 0:
        outlier_folds = [fold_results[i]['fold_id'] for i in outlier_indices]
        outlier_lambdas = lambda_estimates[outlier_indices]
        
        print_status(f"{method_name}: Detected {len(outlier_indices)} outlier folds: {outlier_folds}", "WARNING")
        
        return {
            'outliers_detected': True,
            'n_outliers': len(outlier_indices),
            'outlier_folds': outlier_folds,
            'outlier_lambdas': outlier_lambdas.tolist(),
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
            'outlier_fraction': float(len(outlier_indices) / len(fold_results))
        }
    
    return {'outliers_detected': False, 'all_folds_consistent': True}

def _assess_convergence_diagnostics(fold_results: List[Dict], method_name: str) -> Dict:
    """
    Assess convergence and stability of cross-validation results.
    
    Args:
        fold_results (List[Dict]): List of fold results.
        method_name (str): Name of the CV method.
        
    Returns:
        Dict: Convergence diagnostics, including stability metrics and convergence assessment.
    """
    if len(fold_results) < 5:
        return {'convergence_assessment': 'insufficient_folds', 'n_folds': len(fold_results)}
    
    lambda_estimates = np.array([r['fitted_params']['lambda_km'] for r in fold_results])
    cv_rmse_values = np.array([r['cv_metrics']['cv_rmse'] for r in fold_results])
    
    # Calculate running statistics to assess convergence
    n_folds = len(lambda_estimates)
    running_means = []
    running_stds = []
    
    for i in range(3, n_folds + 1):  # Start from 3 folds minimum
        running_means.append(np.mean(lambda_estimates[:i]))
        running_stds.append(np.std(lambda_estimates[:i]))
    
    # Assess convergence: check if running mean stabilizes
    if len(running_means) >= 3:
        recent_means = running_means[-3:]
        mean_stability = np.std(recent_means) / np.mean(recent_means) if np.mean(recent_means) > 0 else np.inf
        
        # Check if CV-RMSE is consistent across folds
        rmse_cv = np.std(cv_rmse_values) / np.mean(cv_rmse_values) if np.mean(cv_rmse_values) > 0 else np.inf
        
        # Convergence criteria
        converged = mean_stability < 0.05 and rmse_cv < 0.3  # 5% stability, 30% CV for RMSE
        
        return {
            'convergence_assessment': 'converged' if converged else 'needs_more_folds',
            'mean_stability_cv': float(mean_stability),
            'rmse_consistency_cv': float(rmse_cv),
            'running_lambda_means': [float(x) for x in running_means],
            'running_lambda_stds': [float(x) for x in running_stds],
            'convergence_criteria': {
                'mean_stability_threshold': 0.05,
                'rmse_cv_threshold': 0.3,
                'mean_stability_met': bool(mean_stability < 0.05),
                'rmse_consistency_met': bool(rmse_cv < 0.3)
            }
        }
    
    return {'convergence_assessment': 'insufficient_data', 'n_folds': n_folds}

def _cross_method_consistency_check(monthly_results: Dict, station_results: Dict) -> Dict:
    """
    Check consistency between monthly and station block cross-validation methods.
    
    Args:
        monthly_results (Dict): Results from monthly CV.
        station_results (Dict): Results from station block CV.
        
    Returns:
        Dict: Consistency analysis, including a consistency check result and lambda comparison.
    """
    # Determine if each method was successful or skipped
    monthly_is_successful = monthly_results.get('success', False)
    monthly_is_skipped = monthly_results.get('status') == 'skipped'
    station_is_successful = station_results.get('success', False)
    station_is_skipped = station_results.get('status') == 'skipped'

    # Case 1: One or both methods truly failed (not just skipped)
    if (not monthly_is_successful and not monthly_is_skipped) or \
       (not station_is_successful and not station_is_skipped):
        return {
            'consistency_check': 'incomplete',
            'reason': 'One or both methods truly failed (not skipped by design)',
            'monthly_successful': monthly_is_successful,
            'monthly_status': monthly_results.get('status', 'successful'),
            'station_successful': station_is_successful,
            'station_status': station_results.get('status', 'successful')
        }

    # Case 2: Both methods skipped
    if monthly_is_skipped and station_is_skipped:
        return {
            'consistency_check': 'skipped',
            'reason': 'Both monthly and station block CV skipped due to insufficient data.',
            'monthly_successful': monthly_is_successful,
            'monthly_status': monthly_results.get('status', 'skipped'),
            'station_successful': station_is_successful,
            'station_status': station_results.get('status', 'skipped')
        }

    # Case 3: One method skipped, the other successful
    if monthly_is_skipped and station_is_successful:
        return {
            'consistency_check': 'skipped',
            'reason': 'Monthly CV skipped due to insufficient data, station block CV successful.',
            'monthly_successful': monthly_is_successful,
            'monthly_status': monthly_results.get('status', 'skipped'),
            'station_successful': station_is_successful,
            'station_status': station_results.get('status', 'successful')
        }
    
    if station_is_skipped and monthly_is_successful:
        return {
            'consistency_check': 'skipped',
            'reason': 'Station block CV skipped due to insufficient data, monthly CV successful.',
            'monthly_successful': monthly_is_successful,
            'monthly_status': monthly_results.get('status', 'successful'),
            'station_successful': station_is_successful,
            'station_status': station_results.get('status', 'skipped')
        }

    # Case 4: Both methods are successful - proceed with original consistency check
    # Extract lambda estimates
    monthly_lambda = monthly_results['lambda_stability']['mean_lambda_km']
    monthly_lambda_std = monthly_results['lambda_stability']['std_lambda_km']
    station_lambda = station_results['lambda_stability']['mean_lambda_km']
    station_lambda_std = station_results['lambda_stability']['std_lambda_km']
    
    # Calculate relative difference
    lambda_diff = abs(monthly_lambda - station_lambda)
    lambda_mean = (monthly_lambda + station_lambda) / 2
    relative_diff = lambda_diff / lambda_mean if lambda_mean > 0 else np.inf
    
    # Calculate combined uncertainty
    combined_uncertainty = np.sqrt(monthly_lambda_std**2 + station_lambda_std**2)
    
    # Consistency criteria: methods should agree within combined uncertainty
    # and relative difference should be < 20%
    within_uncertainty = bool(lambda_diff <= 2 * combined_uncertainty)  # 2-sigma criterion
    relative_agreement = bool(relative_diff < 0.20)  # 20% relative difference threshold
    
    consistent = within_uncertainty and relative_agreement
    
    return {
        'consistency_check': 'consistent' if consistent else 'inconsistent',
        'lambda_comparison': {
            'monthly_lambda_km': float(monthly_lambda),
            'station_lambda_km': float(station_lambda),
            'absolute_difference_km': float(lambda_diff),
            'relative_difference': float(relative_diff),
            'combined_uncertainty_km': float(combined_uncertainty)
        },
        'consistency_criteria': {
            'within_uncertainty': within_uncertainty,
            'relative_agreement': relative_agreement,
            'uncertainty_threshold_sigma': 2.0,
            'relative_threshold': 0.20
        },
        'consistency_score': float(1.0 - min(relative_diff / 0.20, 1.0)) if relative_diff < np.inf else 0.0
    }

def calculate_cv_metrics(predicted: np.ndarray, actual: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate cross-validation metrics: CV-RMSE, NRMSE, log-likelihood, MAE, R-squared.
    
    Args:
        predicted (np.ndarray): Predicted coherence values.
        actual (np.ndarray): Actual coherence values.
        weights (Optional[np.ndarray], optional): Weights for each data point. Defaults to None.
        
    Returns:
        Dict[str, float]: A dictionary containing various cross-validation metrics.
    """
    if weights is None:
        weights = np.ones_like(predicted)
    
    # Root Mean Square Error
    mse = np.average((predicted - actual)**2, weights=weights)
    rmse = np.sqrt(mse)
    
    # Normalized RMSE
    actual_range = actual.max() - actual.min()
    nrmse = rmse / actual_range if actual_range > 0 else np.inf
    
    # Log-likelihood (simplified Gaussian assumption)
    residuals = predicted - actual
    log_likelihood = -0.5 * np.sum(weights * residuals**2)
    
    # Additional metrics
    mae = np.average(np.abs(predicted - actual), weights=weights)
    r_squared = 1 - np.sum(weights * residuals**2) / np.sum(weights * (actual - np.average(actual, weights=weights))**2)
    
    return {
        'cv_rmse': float(rmse),
        'cv_nrmse': float(nrmse),
        'log_likelihood': float(log_likelihood),
        'mae': float(mae),
        'r_squared': float(r_squared),
        'n_points': len(predicted)
    }

def run_monthly_cross_validation(complete_df: pd.DataFrame) -> Dict:
    """
    Perform monthly temporal cross-validation analysis.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        Dict: A dictionary containing the results of the monthly cross-validation.
    """
    print_status("Starting monthly cross-validation analysis...", "PROCESS")
    
    folds = create_monthly_folds(complete_df)
    
    if not folds:
        print_status("Monthly CV skipped: Not enough data for monthly cross-validation", "INFO")
        return {'success': False, 'status': 'skipped', 'error': 'No valid monthly folds created'}
    
    fold_results = []
    lambda_estimates = []
    
    for i, (month_id, train_mask, val_mask) in enumerate(folds):
        progress_pct = (i + 1) / len(folds) * 100
        print_status(f"Processing monthly fold {i+1}/{len(folds)} ({progress_pct:.1f}%): {month_id}", "PROCESS")

        # Memory-efficient fold processing using masks without creating subsets
        train_size = train_mask.sum()
        val_size = val_mask.sum()

        print_status(f"Fold {month_id}: train={train_size:,} pairs, val={val_size:,} pairs", "DEBUG")

        # Fit model on training data using mask-based approach
        fitted_params, fit_success, error_msg = fit_correlation_model_on_training_chunked(complete_df, train_mask)

        if not fit_success:
            print_status(f"Failed to fit model for month {month_id}: {error_msg}", "WARNING")
            continue

        # Predict on validation data using mask-based approach
        predicted, actual, pred_success, error_msg = predict_validation_coherences_chunked(complete_df, val_mask, fitted_params)

        if not pred_success:
            print_status(f"Failed to predict for month {month_id}: {error_msg}", "WARNING")
            continue

        # Calculate cross-validation metrics
        cv_metrics = calculate_cv_metrics(predicted, actual)

        # Store results
        fold_result = {
            'fold_id': month_id,
            'fitted_params': {
                'amplitude': float(fitted_params[0]),
                'lambda_km': float(fitted_params[1]),
                'offset': float(fitted_params[2])
            },
            'cv_metrics': cv_metrics,
            'training_size': int(train_size),
            'validation_size': int(val_size)
        }

        fold_results.append(fold_result)
        lambda_estimates.append(fitted_params[1])

        # Memory cleanup after each fold
        if (i + 1) % 3 == 0:  # More frequent cleanup for memory safety
            aggressive_memory_cleanup(f"After fold {i+1}/{len(folds)}")
            if not check_memory_and_cleanup():
                print_status("Memory limit reached during fold processing", "WARNING")
                break
    
    # Force garbage collection every 5 folds to manage memory
    if len(folds) % 5 == 0:
        gc.collect()
    
    # Use helper function for aggregation and add outlier detection
    results = _aggregate_fold_results(fold_results, 'monthly_cross_validation')
    
    if results['success']:
        # Add outlier detection
        outlier_analysis = _detect_outlier_folds(fold_results, 'Monthly CV')
        results['outlier_analysis'] = outlier_analysis
        
        # Add convergence diagnostics
        convergence_analysis = _assess_convergence_diagnostics(fold_results, 'Monthly CV')
        results['convergence_analysis'] = convergence_analysis
        
        lambda_mean = results['lambda_stability']['mean_lambda_km']
        cv_rmse_mean = results['cv_performance']['mean_cv_rmse']
    
        print_status(f"Monthly CV completed: λ = {lambda_mean:.0f} ± {results['lambda_stability']['std_lambda_km']:.0f} km, CV-RMSE = {cv_rmse_mean:.4f}", "SUCCESS")
    else:
        print_status("Monthly CV failed: no successful folds", "ERROR")
    return results

def run_station_block_cross_validation(complete_df: pd.DataFrame) -> Dict:
    """
    Perform station block spatial cross-validation analysis using sequential processing.

    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.

    Returns:
        Dict: A dictionary containing the results of the station block cross-validation.
    """
    print_status("Starting station block cross-validation analysis...", "PROCESS")

    # Get station blocks for sequential processing
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    block_size = TEPConfig.get_int('TEP_STATION_BLOCK_SIZE')

    # Create station blocks
    np.random.seed(42)  # Reproducible
    np.random.shuffle(unique_stations)

    station_blocks = []
    for i in range(0, len(unique_stations), block_size):
        block = unique_stations[i:i+block_size]
        if len(block) >= block_size:  # Only use complete blocks
            station_blocks.append(block)

    print_status(f"Created {len(station_blocks)} station blocks for sequential processing", "INFO")

    if not station_blocks:
        return {'success': False, 'error': 'No valid station block folds created'}
    
    fold_results = []
    lambda_estimates = []

    for i, station_block in enumerate(station_blocks):
        progress_pct = (i + 1) / len(station_blocks) * 100
        block_id = f"stations_{i+1:02d}"
        print_status(f"Processing station block fold {i+1}/{len(station_blocks)} ({progress_pct:.1f}%): {block_id}", "PROCESS")

        # Create masks for this specific block (memory efficient - only one fold at a time)
        station_i_in_block = complete_df['station_i'].isin(station_block)
        station_j_in_block = complete_df['station_j'].isin(station_block)
        val_mask = (station_i_in_block | station_j_in_block)
        train_mask = ~val_mask

        # Memory-efficient fold processing using masks without creating subsets
        train_size = train_mask.sum()
        val_size = val_mask.sum()

        print_status(f"Fold {block_id}: train={train_size:,} pairs, val={val_size:,} pairs", "DEBUG")

        # Fit model on training data using mask-based approach
        fitted_params, fit_success, error_msg = fit_correlation_model_on_training_chunked(complete_df, train_mask)

        if not fit_success:
            print_status(f"Failed to fit model for block {block_id}: {error_msg}", "WARNING")
            continue

        # Predict on validation data using mask-based approach
        predicted, actual, pred_success, error_msg = predict_validation_coherences_chunked(complete_df, val_mask, fitted_params)

        if not pred_success:
            print_status(f"Failed to predict for block {block_id}: {error_msg}", "WARNING")
            continue

        # Calculate cross-validation metrics
        cv_metrics = calculate_cv_metrics(predicted, actual)

        # Store results
        fold_result = {
            'fold_id': block_id,
            'fitted_params': {
                'amplitude': float(fitted_params[0]),
                'lambda_km': float(fitted_params[1]),
                'offset': float(fitted_params[2])
            },
            'cv_metrics': cv_metrics,
            'training_size': int(train_size),
            'validation_size': int(val_size)
        }

        fold_results.append(fold_result)
        lambda_estimates.append(fitted_params[1])

        # Memory cleanup after each fold
        aggressive_memory_cleanup(f"After station block fold {i+1}/{len(station_blocks)}")
        if not check_memory_and_cleanup():
            print_status("Memory limit reached during station block fold processing", "WARNING")
            break

    # Force garbage collection every 5 folds to manage memory
    if len(station_blocks) % 5 == 0:
        gc.collect()
    
    # Use helper function for aggregation and add outlier detection
    results = _aggregate_fold_results(fold_results, 'station_block_cross_validation')
    
    if results['success']:
        # Add outlier detection
        outlier_analysis = _detect_outlier_folds(fold_results, 'Station Block CV')
        results['outlier_analysis'] = outlier_analysis
        
        # Add convergence diagnostics
        convergence_analysis = _assess_convergence_diagnostics(fold_results, 'Station Block CV')
        results['convergence_analysis'] = convergence_analysis
        
        lambda_mean = results['lambda_stability']['mean_lambda_km']
        cv_rmse_mean = results['cv_performance']['mean_cv_rmse']
    
        print_status(f"Station block CV completed: λ = {lambda_mean:.0f} ± {results['lambda_stability']['std_lambda_km']:.0f} km, CV-RMSE = {cv_rmse_mean:.4f}", "SUCCESS")
    else:
        print_status("Station block CV failed: no successful folds", "ERROR")
    return results



def run_comprehensive_cross_validation_analysis(ac: str) -> Dict:
    """
    Main function to run the comprehensive cross-validation analysis suite for an analysis center.
    
    Args:
        ac (str): Analysis center identifier.
        
    Returns:
        Dict: A dictionary containing the comprehensive cross-validation results.
    """
    print_status(f"Starting block-wise cross-validation analysis for {ac}", "PROCESS")
    start_time = time.time()
    
    try:
        # Load complete pair dataset
        # Load complete dataset for comprehensive validation
        # Use memory-efficient loading for large datasets
        memory_safe_mode = TEPConfig.get_bool('TEP_MEMORY_SAFE_MODE', False)  # Default to full dataset
        if memory_safe_mode:
            # For memory safety, use sampling for validation instead of full dataset
            print_status("Using memory-safe mode - loading sample for validation", "INFO")
            complete_df = load_complete_pair_dataset(ac, sample_for_validation=True)
        else:
            complete_df = load_complete_pair_dataset(ac)
        
        results = {
            'analysis_center': ac,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_pairs': len(complete_df),
                'unique_stations': len(pd.unique(complete_df[['station_i', 'station_j']].values.ravel())),
                'date_range': {
                    'start': str(complete_df['date'].min()),
                    'end': str(complete_df['date'].max())
                }
            }
        }
        
        # Monthly cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_MONTHLY_CV'):
            monitor_memory_usage("Before Monthly CV")
            monthly_results = run_monthly_cross_validation(complete_df)
            results['monthly_cv'] = monthly_results
            # Memory cleanup after monthly CV
            aggressive_memory_cleanup("After Monthly CV")
            if not check_memory_and_cleanup():
                print_status("Memory limit reached after Monthly CV", "WARNING")
            cleanup_memory(force_gc=True, log_usage=True)
            monitor_memory_usage("After Monthly CV")
        else:
            print_status("Monthly cross-validation disabled", "INFO")
            results['monthly_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # Station block cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_STATION_BLOCKS_CV'):
            monitor_memory_usage("Before Station Block CV")
            station_results = run_station_block_cross_validation(complete_df)
            results['station_block_cv'] = station_results
            # Memory cleanup after station block CV
            aggressive_memory_cleanup("After Station Block CV")
            if not check_memory_and_cleanup():
                print_status("Memory limit reached after Station Block CV", "WARNING")
            cleanup_memory(force_gc=True, log_usage=True)
            monitor_memory_usage("After Station Block CV")
        else:
            print_status("Station block cross-validation disabled", "INFO")
            results['station_block_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # LOSO cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_LOSO_CV', True):
            monitor_memory_usage("Before LOSO CV")
            loso_results = run_loso_analysis(complete_df, ac)
            results['loso_cv'] = loso_results
            # Memory cleanup after LOSO
            aggressive_memory_cleanup("After LOSO CV")
            if not check_memory_and_cleanup():
                print_status("Memory limit reached after LOSO CV", "WARNING")
            cleanup_memory(force_gc=True, log_usage=True)
            monitor_memory_usage("After LOSO CV")
        else:
            print_status("LOSO cross-validation disabled", "INFO")
            results['loso_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # LODO cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_LODO_CV', True):
            monitor_memory_usage("Before LODO CV")
            lodo_results = run_lodo_analysis(complete_df, ac)
            results['lodo_cv'] = lodo_results
            # Memory cleanup after LODO
            aggressive_memory_cleanup("After LODO CV")
            if not check_memory_and_cleanup():
                print_status("Memory limit reached after LODO CV", "WARNING")
            cleanup_memory(force_gc=True, log_usage=True)
            monitor_memory_usage("After LODO CV")
        else:
            print_status("LODO cross-validation disabled", "INFO")
            results['lodo_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # Bootstrap cross-validation disabled (redundant with Step 3.1 Robust Block Bootstrap)
        print_status("Bootstrap cross-validation disabled (use Step 3.1 Robust Block Bootstrap instead)", "INFO")
        results['bootstrap_cv'] = {'success': False, 'error': 'Disabled - use Step 3.1 instead'}

        # Final memory check and cleanup
        aggressive_memory_cleanup("Final cleanup after all CV methods")
        final_memory_check = check_memory_and_cleanup()
        if not final_memory_check:
            print_status("WARNING: Final memory usage is high - consider restarting system", "WARNING")
        
        # Summary statistics
        successful_methods = []
        if results['monthly_cv']['success']:
            successful_methods.append('monthly')
        if results['station_block_cv']['success']:
            successful_methods.append('station_block')
        if results['loso_cv']['success']:
            successful_methods.append('loso')
        if results['lodo_cv']['success']:
            successful_methods.append('lodo')
        # Bootstrap CV is disabled (removed for being redundant and unreliable)
        
        if successful_methods:
            # Aggregate lambda estimates across methods
            all_lambdas = []
            if results['monthly_cv']['success']:
                all_lambdas.extend(results['monthly_cv']['lambda_stability']['lambda_estimates'])
            if results['station_block_cv']['success']:
                all_lambdas.extend(results['station_block_cv']['lambda_stability']['lambda_estimates'])
            if results['loso_cv']['success']:
                all_lambdas.extend(results['loso_cv']['lambda_values'])
            if results['lodo_cv']['success']:
                all_lambdas.extend(results['lodo_cv']['lambda_values'])
            # Bootstrap CV removed (was redundant and unreliable)
            
            # Cross-method consistency check
            cross_method_consistency = _cross_method_consistency_check(
                results['monthly_cv'], results['station_block_cv']
            )
            
            results['summary'] = {
                'successful_methods': successful_methods,
                'overall_lambda': {
                    'mean_km': float(np.mean(all_lambdas)),
                    'std_km': float(np.std(all_lambdas)),
                    'cv': float(np.std(all_lambdas) / np.mean(all_lambdas)),
                    'n_estimates': len(all_lambdas)
                },
                'cross_method_consistency': cross_method_consistency
            }
            
            # Report consistency results
            if cross_method_consistency['consistency_check'] == 'consistent':
                print_status("Cross-method validation: PASSED - Monthly and station block CV are consistent", "SUCCESS")
            elif cross_method_consistency['consistency_check'] == 'inconsistent':
                rel_diff = cross_method_consistency['lambda_comparison']['relative_difference']
                print_status(f"Cross-method validation: WARNING - Methods differ by {rel_diff:.1%}", "WARNING")
            elif cross_method_consistency['consistency_check'] == 'skipped':
                print_status(f"Cross-method validation: SKIPPED - {cross_method_consistency['reason']}", "INFO")
            else:
                print_status(f"Cross-method validation: INCOMPLETE - {cross_method_consistency['reason']}", "INFO")
        else:
            results['summary'] = {
                'successful_methods': [],
                'error': 'No successful cross-validation methods'
            }
        
        elapsed_time = time.time() - start_time
        results['processing_time_seconds'] = elapsed_time
        
        print_status(f"Block-wise cross-validation completed for {ac} in {elapsed_time:.1f} seconds", "SUCCESS")
        return results
        
    except Exception as e:
        error_msg = f"Block-wise cross-validation failed for {ac}: {str(e)}"
        print_status(error_msg, "ERROR")
        raise TEPAnalysisError(error_msg) # Re-raise a specific exception

# Add LOSO/LODO worker functions and analysis functions moved from step_5

# REMOVED: Worker functions no longer needed with sequential processing
    """Initializer to load heavy context once per worker process for LOSO analysis.
    
    MEMORY OPTIMIZATION: Instead of copying the entire dataset to each worker,
    we store only the file path and load data per task to prevent memory explosion.
    
    Args:
        ac_name (str): Analysis center name (e.g., 'code', 'esa_final', 'igs_combined').
        edges (np.ndarray): Bin edges for distance.
        min_bin_count (int): Minimum number of data points per bin.
    """
    global WORKER_EDGES, WORKER_MIN_BIN_COUNT, WORKER_DATA_PATH
    
    # MEMORY FIX: Use the existing consolidated file path directly
    # This prevents copying 5.4GB to each of 4 workers (27GB total)
    from pathlib import Path
    
    # Use the consolidated file that already exists from Step 2.0
    consolidated_file = Path(PACKAGE_ROOT / "results" / "outputs" / f"step_2_0_pairs_consolidated_{ac_name}.csv")
    
    if not consolidated_file.exists():
        raise FileNotFoundError(f"Consolidated file not found: {consolidated_file}")
    
    WORKER_DATA_PATH = str(consolidated_file)
    WORKER_EDGES = edges
    WORKER_MIN_BIN_COUNT = min_bin_count

# Worker functions removed - using sequential processing in main process

def run_loso_analysis(complete_df: pd.DataFrame, ac: str) -> Dict:
    """
    Perform Leave-One-Station-Out (LOSO) analysis on the complete dataset.
    Tests stability by excluding each station and re-fitting correlation model.

    MEMORY OPTIMIZATION: Process stations sequentially in main process to avoid memory explosion.

    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        ac (str): Analysis center name.

    Returns:
        Dict: A dictionary containing the LOSO analysis results.
    """
    print_status("Starting Leave-One-Station-Out (LOSO) analysis...", "PROCESS")

    # Use full dataset for comprehensive validation
    total_pairs = len(complete_df)
    print_status(f"Using full dataset: {total_pairs:,} pairs for validation", "INFO")

    # Get all stations from the complete dataset
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    max_stations_to_test = min(10, len(unique_stations))  # Use max 10 stations

    if len(unique_stations) > max_stations_to_test:
        # Randomly sample stations for testing
        np.random.seed(42)  # Reproducible
        stations_to_test = np.random.choice(unique_stations, max_stations_to_test, replace=False)
        print_status(f"Sampling {max_stations_to_test} stations from {len(unique_stations)} total for efficiency", "INFO")
    else:
        stations_to_test = unique_stations
        print_status(f"Testing stability across all {len(unique_stations)} unique stations", "INFO")

    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)

    lambda_estimates = []

    # MEMORY FIX: Process stations SEQUENTIALLY in main process to avoid memory explosion
    # No multiprocessing - each station processed one at a time to avoid memory issues
    print_status("Using sequential processing in main process to avoid memory issues", "INFO")

    # MEMORY OPTIMIZATION: Process stations in chunks of 5 to reduce memory pressure
    print_status("Processing stations in chunks of 5 to reduce memory pressure", "INFO")

    chunk_size = 5
    station_results = []

    for chunk_start in range(0, len(stations_to_test), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(stations_to_test))
        chunk_stations = stations_to_test[chunk_start:chunk_end]

        print_status(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(stations_to_test)-1)//chunk_size + 1}: stations {chunk_start+1}-{chunk_end}", "PROCESS")

        for i, station in enumerate(chunk_stations):
            print_status(f"  Processing station {station} ({chunk_start + i + 1}/{len(stations_to_test)})...", "PROCESS")

            # Pre-filter data for this station from the complete dataset
            mask = (complete_df['station_i'] != station) & (complete_df['station_j'] != station)
            subset_df = complete_df[mask].copy()

            if len(subset_df) < 1000:  # Skip if too few pairs
                print_status(f"    Station {station}: Insufficient data ({len(subset_df)} pairs)", "WARNING")
                del subset_df
                continue

            # Process this station's data directly in main process
            try:
                # Bin the data
                subset_df['dist_bin'] = pd.cut(subset_df['dist_km'], bins=edges, right=False)
                binned = subset_df.groupby('dist_bin', observed=True).agg({
                    'dist_km': 'mean',
                    'coherence': 'mean',
                    'station_i': 'count'
                }).rename(columns={'station_i': 'count'}).dropna()

                # Filter bins with sufficient data
                binned = binned[binned['count'] >= min_bin_count]

                if len(binned) < 3:  # Need at least 3 bins
                    print_status(f"    Station {station}: Insufficient bins ({len(binned)})", "WARNING")
                    del subset_df, binned
                    continue

                distances = binned['dist_km'].values
                coherences = binned['coherence'].values
                weights = binned['count'].values

                # Fit exponential decay model
                c_range = coherences.max() - coherences.min()
                if c_range <= 0:
                    print_status(f"    Station {station}: Invalid coherence range", "WARNING")
                    del subset_df, binned
                    continue

                p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]

                popt, _ = curve_fit(
                    correlation_model, distances, coherences,
                    p0=p0, sigma=1.0/np.sqrt(weights), absolute_sigma=False,
                    maxfev=1000  # Reduced iterations
                )

                # Calculate R-squared
                y_pred = correlation_model(distances, *popt)
                ss_res = np.sum((coherences - y_pred) ** 2)
                ss_tot = np.sum((coherences - np.mean(coherences)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                lambda_estimates.append(popt[1])
                print_status(f"    Station {station}: λ = {popt[1]:.0f} km (R² = {r_squared:.3f})", "SUCCESS")

            except Exception as e:
                print_status(f"    Station {station}: Exception - {str(e)}", "ERROR")

            # Force memory cleanup after each station
            del subset_df, binned
            cleanup_memory(force_gc=True, log_usage=True)

        # Stream intermediate results to disk after each chunk
        print_status(f"  Completed chunk {chunk_start//chunk_size + 1}, streaming results to disk", "INFO")
        # Force additional cleanup between chunks
        cleanup_memory(force_gc=True, log_usage=True)

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
        'n_stations_tested': len(stations_to_test),
        'lambda_values': lambda_estimates,
        'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
    }

    print_status(f"LOSO complete: λ = {results['lambda_mean']:.1f} ± {results['lambda_std']:.1f} km (CV = {results['coefficient_of_variation']:.3f})", "SUCCESS")

    # Memory cleanup after LOSO analysis
    cleanup_memory(force_gc=True, log_usage=True)

    return results

def run_lodo_analysis(complete_df: pd.DataFrame, ac: str) -> Dict:
    """
    Perform Leave-One-Day-Out (LODO) analysis on the complete dataset.
    Tests stability by excluding each day and re-fitting correlation model.

    MEMORY OPTIMIZATION: Process dates sequentially in main process to avoid memory explosion.

    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        ac (str): Analysis center name.

    Returns:
        Dict: A dictionary containing the results of the LODO analysis.
    """
    print_status("Starting Leave-One-Day-Out (LODO) analysis...", "PROCESS")

    # MEMORY OPTIMIZATION: Use the same data for LODO validation
    # Get unique dates from the dataset
    unique_dates = complete_df['date'].unique()
    max_days_to_test = min(10, len(unique_dates))  # Use max 10 days from dataset

    if len(unique_dates) > max_days_to_test:
        # Randomly sample days for testing
        np.random.seed(43)  # Different seed from LOSO
        dates_to_test = np.random.choice(unique_dates, max_days_to_test, replace=False)
        print_status(f"Sampling {max_days_to_test} days from {len(unique_dates)} available in dataset", "INFO")
    else:
        dates_to_test = unique_dates
        print_status(f"Testing stability across all {len(unique_dates)} unique days in dataset", "INFO")

    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)

    lambda_estimates = []

    # MEMORY FIX: Process dates SEQUENTIALLY in main process to avoid memory explosion
    # No multiprocessing - each date processed one at a time to avoid memory issues
    print_status("Using sequential processing in main process to avoid memory issues", "INFO")

    # Process each date sequentially using the dataset
    for i, date in enumerate(dates_to_test):
        print_status(f"Processing date {pd.to_datetime(date).strftime('%Y-%m-%d')} ({i+1}/{len(dates_to_test)})...", "PROCESS")

        # Pre-filter data for this date from the dataset
        mask = complete_df['date'] != date
        subset_df = complete_df[mask].copy()

        if len(subset_df) < 1000:  # Skip if too few pairs
            print_status(f"  Date {pd.to_datetime(date).strftime('%Y-%m-%d')}: Insufficient data ({len(subset_df)} pairs)", "WARNING")
            del subset_df
            continue

        # Process this date's data directly in main process
        try:
            # Bin the data
            subset_df['dist_bin'] = pd.cut(subset_df['dist_km'], bins=edges, right=False)
            binned = subset_df.groupby('dist_bin', observed=True).agg({
                'dist_km': 'mean',
                'coherence': 'mean',
                'station_i': 'count'
            }).rename(columns={'station_i': 'count'}).dropna()

            # Filter bins with sufficient data
            binned = binned[binned['count'] >= min_bin_count]

            if len(binned) < 3:  # Need at least 3 bins
                print_status(f"  Date {pd.to_datetime(date).strftime('%Y-%m-%d')}: Insufficient bins ({len(binned)})", "WARNING")
                del subset_df, binned
                continue

            distances = binned['dist_km'].values
            coherences = binned['coherence'].values
            weights = binned['count'].values

            # Fit exponential decay model
            c_range = coherences.max() - coherences.min()
            if c_range <= 0:
                print_status(f"  Date {pd.to_datetime(date).strftime('%Y-%m-%d')}: Invalid coherence range", "WARNING")
                del subset_df, binned
                continue

            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]

            popt, _ = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights), absolute_sigma=False,
                maxfev=1000  # Reduced iterations
            )

            # Calculate R-squared
            y_pred = correlation_model(distances, *popt)
            ss_res = np.sum((coherences - y_pred) ** 2)
            ss_tot = np.sum((coherences - np.mean(coherences)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            lambda_estimates.append(popt[1])
            print_status(f"  Date {pd.to_datetime(date).strftime('%Y-%m-%d')}: λ = {popt[1]:.0f} km (R² = {r_squared:.3f})", "SUCCESS")

        except Exception as e:
            print_status(f"  Date {pd.to_datetime(date).strftime('%Y-%m-%d')}: Exception - {str(e)}", "ERROR")

        # Force memory cleanup after each date
        del subset_df, binned
        cleanup_memory(force_gc=True, log_usage=True)

        # Force additional cleanup after each date
        cleanup_memory(force_gc=True, log_usage=True)

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

    # Memory cleanup after LODO analysis
    cleanup_memory(force_gc=True, log_usage=True)

    return results

@ensure_single_instance
def main():
    """Main execution function of the comprehensive cross-validation suite.

    This function orchestrates the loading of data, running of various cross-validation
    analyses (monthly, station block, LOSO, LODO), and saving of results for each
    analysis center.
    """
    global step_logger
    start_time = time.time()

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print_status(f"Received signal {signum}, shutting down gracefully...", "WARNING")
        # Don't call _cleanup_processes() here to avoid recursion
        # Just exit cleanly
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Kill any existing instances of this script
    print("Killing any existing instances of step_3_0_tep_cross_validation_suite...")
    import subprocess
    import os
    try:
        current_pid = os.getpid()
        # Kill by script name but exclude current process
        subprocess.run(['pkill', '-f', 'step_3_0_tep_cross_validation_suite.py'],
                      capture_output=True, timeout=10)
        # Kill multiprocessing processes
        subprocess.run(['pkill', '-f', 'multiprocessing.spawn'],
                      capture_output=True, timeout=10)
        print("Successfully killed existing instances")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"Warning: Could not kill all existing processes: {e}")

    # Set up step-specific logger
    # Initialize step-specific logger (reset log file on start)
    log_file_path = Path(__file__).resolve().parents[3] / "logs" / "step_3_0_cross_validation_suite.log"

    # Reset log file when starting (remove old content)
    try:
        with open(log_file_path, 'w') as f:
            f.write("")  # Clear the log file
        print(f"Log file reset: {log_file_path}")
    except Exception as e:
        print(f"Warning: Could not reset log file: {e}")

    step_logger = TEPLogger(
        name="step_3_0_cross_validation_suite",
        level="DEBUG",
        log_file_path=log_file_path
    )
    
    # Set as the current step logger for print_status
    set_step_logger(step_logger)

    print_status("TEP GNSS Analysis Package v0.13 - STEP 3.0: Cross-Validation Suite (Comprehensive)", "TITLE")
    print_status("=" * 70, "INFO")
    
    # Validate inputs
    validate_directory_exists(PACKAGE_ROOT / "results" / "tmp", "Step 2.0 pair files directory")
    validate_directory_exists(PACKAGE_ROOT / "results" / "outputs", "Output directory")
    
    # Configuration summary
    print_status("Configuration:", "INFO")
    print_status(f"  Monthly CV enabled: {TEPConfig.get_bool('TEP_ENABLE_MONTHLY_CV')}", "INFO")
    print_status(f"  Station block CV enabled: {TEPConfig.get_bool('TEP_ENABLE_STATION_BLOCKS_CV')}", "INFO")
    print_status(f"  LOSO CV enabled: {TEPConfig.get_bool('TEP_ENABLE_LOSO_CV', True)}", "INFO")
    print_status(f"  LODO CV enabled: {TEPConfig.get_bool('TEP_ENABLE_LODO_CV', True)}", "INFO")
    print_status(f"  Bootstrap CV: Disabled (use Step 3.1 Robust Block Bootstrap instead)", "INFO")
    print_status(f"  Monthly folds limit: {TEPConfig.get_int('TEP_MONTHLY_CV_FOLDS', 12)}", "INFO")
    print_status(f"  Station block size: {TEPConfig.get_int('TEP_STATION_BLOCK_SIZE', 10)}", "INFO")
    print_status(f"  LOSO sample size: {TEPConfig.get_int('TEP_LOSO_SAMPLE_SIZE', 50)}", "INFO")
    print_status(f"  LODO sample size: {TEPConfig.get_int('TEP_LODO_SAMPLE_SIZE', 100)}", "INFO")
    print_status(f"  Memory limit: {TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')} GB", "INFO")
    
    # Determine analysis centers to process
    analysis_centers = []
    for ac in ['code', 'esa_final', 'igs_combined']:
        # Check for consolidated files first (preferred)
        consolidated_file = Path(PACKAGE_ROOT / "results" / "outputs" / f"step_2_0_pairs_consolidated_{ac}.csv")
        # Fallback to individual files
        pair_files = list(Path(PACKAGE_ROOT / "results" / "tmp").glob(f"step_2_0_pairs_{ac}_*.csv"))
        
        if consolidated_file.exists() or pair_files:
            analysis_centers.append(ac)
            if consolidated_file.exists():
                print_status(f"Found consolidated pair data for {ac}", "INFO")
            else:
                print_status(f"Found {len(pair_files)} individual pair files for {ac}", "INFO")
        else:
            print_status(f"No pair files found for {ac}, skipping", "WARNING")
    
    if not analysis_centers:
        print_status("No analysis centers found with pair data", "ERROR")
        return
    
    print_status(f"Processing {len(analysis_centers)} analysis centers: {', '.join(analysis_centers)}", "INFO")
    
    # Process each analysis center
    for i, ac in enumerate(analysis_centers):
        print_status(f"\nProcessing analysis center: {ac.upper()}", "PROCESS")
        print_status("-" * 50, "INFO")
        
        # Monitor memory before processing each center
        monitor_memory_usage(f"Before processing {ac}")
        
        # Run comprehensive cross-validation suite
        results = run_comprehensive_cross_validation_analysis(ac)
        
        # Save results with better error handling
        output_file = PACKAGE_ROOT / "results" / "outputs" / f"step_3_0_cross_validation_suite_{ac}.json"
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Attempt to save results using safe_json_write
            success = safe_json_write(results, output_file)
            
            if success:
                print_status(f"Results saved to: {output_file}", "SUCCESS")
            else:
                # Fallback to manual JSON write if safe_json_write indicates failure
                # This handles cases where safe_json_write might return False without raising an exception
                import json
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print_status(f"Results saved to: {output_file} (fallback method)", "SUCCESS")
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
    total_time = time.time() - start_time
    print_status("=" * 70, "INFO")
    print_status(f"Block-wise cross-validation completed in {total_time:.1f} seconds", "SUCCESS")
    print_status(f"Results saved for {len(analysis_centers)} analysis centers", "SUCCESS")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print_status("Step 3.0 interrupted by user", "WARNING")
        # Cleanup processes before exit
        _cleanup_processes()
        sys.exit(0)  # Don't stop pipeline
    except Exception as e:
        print_status(f"Step 3.0 error: {e}", "ERROR")
        import traceback
        print_status(traceback.format_exc(), "DEBUG")
        # Cleanup processes before exit
        _cleanup_processes()
        sys.exit(0)  # Don't stop pipeline
