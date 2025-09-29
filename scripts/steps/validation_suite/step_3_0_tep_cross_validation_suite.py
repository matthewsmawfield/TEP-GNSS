#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 3.0: Cross-Validation Suite (Comprehensive)
=====================================================================

Comprehensive cross-validation suite for TEP correlation models including:
1. Block-wise cross-validation (temporal/spatial) - from original step_5_5
2. Leave-One-Station-Out (LOSO) analysis - moved from step_5
3. Leave-One-Day-Out (LODO) analysis - moved from step_5
4. Pairwise bootstrap analysis - refined from original block bootstrap

This consolidates all cross-validation methodologies to provide rigorous
validation of TEP correlation parameters with multiple approaches.

Note on Bootstrapping:
For *true block bootstrap* implementations (resampling stations or days to account for dependencies),
refer to `scripts/steps/core_analysis/step_2_0_tep_correlation_analysis.py` and `scripts/steps/advanced_analysis_and_visualization/step_4_3_high_resolution_astronomical_events.py`.
For *pairwise bootstrap* implementations, refer to `scripts/steps/validation_suite/step_3_3_methodology_validation.py`.

Requirements: Step 2.0 complete
Next: Step 3.1 (Robust Block Bootstrap Validation)

Key Analyses:
1. Monthly temporal cross-validation - split by months, predict held-out months
2. Leave-5-stations-out spatial blocks - remove station groups, test predictive power
3. Leave-One-Station-Out - exclude individual stations, test stability
4. Leave-One-Day-Out - exclude individual days, test temporal stability
5. Pairwise bootstrap - resample pairs for uncertainty assessment
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
  - TEP_ENABLE_BOOTSTRAP_CV: Enable Pairwise Bootstrap cross-validation (default: True)
  - TEP_MONTHLY_CV_FOLDS: Number of monthly folds to use (default: 10, memory-optimized)
  - TEP_STATION_BLOCK_SIZE: Number of stations per block (default: 10, memory-optimized)
  - TEP_LOSO_SAMPLE_SIZE: Number of stations to sample for LOSO (default: 50)
  - TEP_LODO_SAMPLE_SIZE: Number of days to sample for LODO (default: 100)
  - TEP_BOOTSTRAP_SAMPLES: Number of bootstrap samples (default: 200)
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
from datetime import datetime, timedelta
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

# Global variables for worker processes (used by LOSO/LODO)
WORKER_COMPLETE_DF = None
WORKER_EDGES = None
WORKER_MIN_BIN_COUNT = None

# Import TEP utilities for better configuration and error handling
import sys
sys.path.insert(0, str(ROOT))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    TEPAnalysisError, safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)

def print_status(message, level="INFO"):
    """Enhanced status printing with timestamp and color coding.
    
    Args:
        message (str): The message to print.
        level (str, optional): The log level (e.g., "INFO", "WARNING", "ERROR"). Defaults to "INFO".
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

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
    """Monitor memory usage and warn if approaching limits.
    
    Returns:
        bool: True if memory usage is within limits, False otherwise.
    """
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

def load_complete_pair_dataset(ac: str) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center.
    Reuses the chunked loading approach from step_5 for memory efficiency.
    
    Args:
        ac (str): Analysis center identifier (e.g., "code", "esa_final").
        
    Returns:
        pd.DataFrame: A concatenated DataFrame of all pair data for the given analysis center.
        
    Raises:
        TEPFileError: If no pair files are found for the analysis center.
        TEPDataError: If no valid pair data can be loaded.
    """
    print_status(f"Loading complete pair dataset for {ac}...", "PROCESS")
    
    # Find all pair files for this analysis center
    pair_files = list(Path(ROOT / "results" / "tmp").glob(f"step_2_0_pairs_{ac}_*.csv"))
    
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
                    df_chunk = pd.read_csv(file_path, engine='python')
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
    Create monthly cross-validation folds.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        List[Tuple[str, pd.Series, pd.Series]]: A list of tuples, where each tuple contains:
            - month_id (str): Identifier for the month.
            - training_data_mask (pd.Series): Boolean mask for the training data.
            - validation_data_mask (pd.Series): Boolean mask for the validation data.
    """
    print_status("Creating monthly cross-validation folds...", "PROCESS")
    
    # Convert date column to datetime if needed
    if complete_df['date'].dtype == 'object':
        complete_df['date'] = pd.to_datetime(complete_df['date'])
    
    # Create year-month identifier
    complete_df['year_month'] = complete_df['date'].dt.to_period('M')
    unique_months = sorted(complete_df['year_month'].unique())
    
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
        
        train_mask = complete_df['year_month'] != month
        val_mask = complete_df['year_month'] == month
        
        if not _validate_fold_data(complete_df[train_mask], complete_df[val_mask], str(month)):
            continue
        
        folds.append((str(month), train_mask, val_mask))
        
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
        # Validation set: pairs involving any station in the block
        val_mask = (complete_df['station_i'].isin(station_block) | 
                   complete_df['station_j'].isin(station_block))
        
        # Training set: pairs not involving any station in the block
        train_mask = ~val_mask
        
        # Use helper function for validation with appropriate thresholds for station blocks
        if not _validate_fold_data(complete_df[train_mask], complete_df[val_mask], f"stations_{i+1:02d}", min_train=1000, min_val=100):
            continue
        
        block_id = f"stations_{i+1:02d}"
        folds.append((block_id, train_mask, val_mask))
        
        if (i + 1) % 5 == 0:
            gc.collect()
    
    print_status(f"Created {len(folds)} valid station block folds", "SUCCESS")
    return folds

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
            bounds=([1e-10, 100, -1], [2, 20000, 1]),
            maxfev=5000
        )
        
        return popt, True, None
        
    except Exception as e:
        return None, False, str(e)

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
    
    return {
        'success': True,
        'method': method_name,
        'n_folds': len(fold_results),
        'lambda_stability': {
            'mean_lambda_km': float(lambda_mean),
            'std_lambda_km': float(lambda_std),
            'cv_lambda': float(lambda_cv),
            'lambda_estimates': [float(x) for x in lambda_estimates]
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
                'mean_stability_met': mean_stability < 0.05,
                'rmse_consistency_met': rmse_cv < 0.3
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
    if not (monthly_results.get('success', False) and station_results.get('success', False)):
        return {
            'consistency_check': 'incomplete',
            'reason': 'One or both methods failed',
            'monthly_success': monthly_results.get('success', False),
            'station_success': station_results.get('success', False)
        }
    
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
    within_uncertainty = lambda_diff <= 2 * combined_uncertainty  # 2-sigma criterion
    relative_agreement = relative_diff < 0.20  # 20% relative difference threshold
    
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
        return {'success': False, 'error': 'No valid monthly folds created'}
    
    fold_results = []
    lambda_estimates = []
    
    for i, (month_id, train_mask, val_mask) in enumerate(folds):
        progress_pct = (i + 1) / len(folds) * 100
        print_status(f"Processing monthly fold {i+1}/{len(folds)} ({progress_pct:.1f}%): {month_id}", "PROCESS")
        
        # Fit model on training data
        fitted_params, fit_success, error_msg = fit_correlation_model_on_training(complete_df[train_mask])
        
        if not fit_success:
            print_status(f"Failed to fit model for month {month_id}: {error_msg}", "WARNING")
            continue
        
        # Predict on validation data
        predicted, actual, pred_success, error_msg = predict_validation_coherences(complete_df[val_mask], fitted_params)
        
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
            'training_size': len(complete_df[train_mask]),
            'validation_size': len(complete_df[val_mask])
        }
        
        fold_results.append(fold_result)
        lambda_estimates.append(fitted_params[1])
    
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
    Perform station block spatial cross-validation analysis.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        Dict: A dictionary containing the results of the station block cross-validation.
    """
    print_status("Starting station block cross-validation analysis...", "PROCESS")
    
    folds = create_station_block_folds(complete_df)
    
    if not folds:
        return {'success': False, 'error': 'No valid station block folds created'}
    
    fold_results = []
    lambda_estimates = []
    
    for i, (block_id, train_mask, val_mask) in enumerate(folds):
        progress_pct = (i + 1) / len(folds) * 100
        print_status(f"Processing station block fold {i+1}/{len(folds)} ({progress_pct:.1f}%): {block_id}", "PROCESS")
        
        # Fit model on training data
        fitted_params, fit_success, error_msg = fit_correlation_model_on_training(complete_df[train_mask])
        
        if not fit_success:
            print_status(f"Failed to fit model for block {block_id}: {error_msg}", "WARNING")
            continue
        
        # Predict on validation data
        predicted, actual, pred_success, error_msg = predict_validation_coherences(complete_df[val_mask], fitted_params)
        
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
            'training_size': len(complete_df[train_mask]),
            'validation_size': len(complete_df[val_mask])
        }
        
        fold_results.append(fold_result)
        lambda_estimates.append(fitted_params[1])
    
    # Force garbage collection every 5 folds to manage memory
    if len(folds) % 5 == 0:
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
            monthly_results = run_monthly_cross_validation(complete_df)
            results['monthly_cv'] = monthly_results
        else:
            print_status("Monthly cross-validation disabled", "INFO")
            results['monthly_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # Station block cross-validation  
        if TEPConfig.get_bool('TEP_ENABLE_STATION_BLOCKS_CV'):
            station_results = run_station_block_cross_validation(complete_df)
            results['station_block_cv'] = station_results
        else:
            print_status("Station block cross-validation disabled", "INFO")
            results['station_block_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # LOSO cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_LOSO_CV', True):
            loso_results = run_loso_analysis(complete_df)
            results['loso_cv'] = loso_results
        else:
            print_status("LOSO cross-validation disabled", "INFO")
            results['loso_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # LODO cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_LODO_CV', True):
            lodo_results = run_lodo_analysis(complete_df)
            results['lodo_cv'] = lodo_results
        else:
            print_status("LODO cross-validation disabled", "INFO")
            results['lodo_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
        # Bootstrap cross-validation
        if TEPConfig.get_bool('TEP_ENABLE_BOOTSTRAP_CV', True):
            bootstrap_results = run_pairwise_bootstrap_analysis(complete_df)
            results['bootstrap_cv'] = bootstrap_results
        else:
            print_status("Bootstrap cross-validation disabled", "INFO")
            results['bootstrap_cv'] = {'success': False, 'error': 'Disabled by configuration'}
        
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
        if results['bootstrap_cv']['success']:
            successful_methods.append('bootstrap')
        
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
            if results['bootstrap_cv']['success']:
                all_lambdas.extend(results['bootstrap_cv']['lambda_values'])
            
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
            else:
                print_status("Cross-method validation: INCOMPLETE - One method failed", "INFO")
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
        return {
            'analysis_center': ac,
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

# Add LOSO/LODO worker functions and analysis functions moved from step_5

def _init_loso_worker_context(complete_df, edges, min_bin_count):
    """Initializer to load heavy context once per worker process for LOSO analysis.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        edges (np.ndarray): Bin edges for distance.
        min_bin_count (int): Minimum number of data points per bin.
    """
    global WORKER_COMPLETE_DF, WORKER_EDGES, WORKER_MIN_BIN_COUNT
    WORKER_COMPLETE_DF = complete_df
    WORKER_EDGES = edges
    WORKER_MIN_BIN_COUNT = min_bin_count

def _init_lodo_worker_context(complete_df, edges, min_bin_count):
    """Initializer to load heavy context once per worker process for LODO analysis.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        edges (np.ndarray): Bin edges for distance.
        min_bin_count (int): Minimum number of data points per bin.
    """
    global WORKER_COMPLETE_DF, WORKER_EDGES, WORKER_MIN_BIN_COUNT
    WORKER_COMPLETE_DF = complete_df
    WORKER_EDGES = edges
    WORKER_MIN_BIN_COUNT = min_bin_count

def _process_single_station_loso(station_to_exclude):
    """
    Process a single station exclusion for LOSO analysis using worker context.
    
    Args:
        station_to_exclude (str): The ID of the station to exclude.
        
    Returns:
        Optional[Dict]: A dictionary containing the fitted lambda and other parameters if successful, or error details.
    """
    try:
        global WORKER_COMPLETE_DF, WORKER_EDGES, WORKER_MIN_BIN_COUNT
        complete_df = WORKER_COMPLETE_DF
        edges = WORKER_EDGES
        min_bin_count = WORKER_MIN_BIN_COUNT
        
        if complete_df is None or edges is None or min_bin_count is None:
            return {
                'station': station_to_exclude,
                'error': 'Worker context not properly initialized.',
                'debug': 'context_init_failure'
            }
        
        # Create subset excluding this station
        subset_df = complete_df[
            (complete_df['station_i'] != station_to_exclude) & 
            (complete_df['station_j'] != station_to_exclude)
        ].copy()
        
        if len(subset_df) < 1000:  # Skip if too few pairs
            return {
                'station': station_to_exclude,
                'error': f'Insufficient data: {len(subset_df)} pairs',
                'debug': 'too_few_pairs'
            }
        
        # Bin the data
        subset_df['dist_bin'] = pd.cut(subset_df['dist_km'], bins=edges, right=False)
        binned = subset_df.groupby('dist_bin', observed=True).agg({
            'dist_km': 'mean',
            'coherence': 'mean',
            'station_i': 'count'
        }).rename(columns={'station_i': 'count'})
        binned.columns = ['mean_dist', 'mean_coh', 'count']
        
        # Filter bins with sufficient data
        binned = binned[binned['count'] >= min_bin_count]
        
        if len(binned) < 3:  # Need at least 3 bins for fitting
            return {
                'station': station_to_exclude,
                'error': f'Insufficient bins: {len(binned)} bins',
                'debug': 'too_few_bins'
            }
        
        # Fit exponential model
        try:
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            # Check if we have valid data
            if len(distances) == 0 or len(coherences) == 0:
                return None
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(distances)) or np.any(~np.isfinite(coherences)):
                return {
                    'station': station_to_exclude,
                    'error': 'Invalid data: NaN or infinite values',
                    'debug': 'invalid_data'
                }
            
            # Initial parameter estimates
            c_range = coherences.max() - coherences.min()
            if c_range <= 0 or not np.isfinite(c_range):
                return {
                    'station': station_to_exclude,
                    'error': f'Invalid coherence range: {c_range}',
                    'debug': 'invalid_range'
                }
                
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            bounds = ([1e-10, 100, -1], [5, 20000, 1])
            
            # Ensure we have clean numpy arrays
            distances = np.asarray(distances, dtype=float)
            coherences = np.asarray(coherences, dtype=float)
            weights = np.asarray(weights, dtype=float)
            p0 = np.asarray(p0, dtype=float)
            
            popt, _ = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights), bounds=bounds, maxfev=5000
            )
            
            return {
                'station': station_to_exclude,
                'lambda_km': float(popt[1]),
                'amplitude': float(popt[0]),
                'offset': float(popt[2]),
                'n_pairs': len(subset_df),
                'n_bins': len(binned)
            }
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
            # Return debug info instead of None to see what's failing
            return {
                'station': station_to_exclude,
                'error': str(e),
                'n_pairs': len(subset_df) if 'subset_df' in locals() else 0,
                'n_bins': len(binned) if 'binned' in locals() else 0,
                'debug': 'fit_failed'
            }
            
    except Exception as e:
        return None

def _process_single_date_lodo(date_to_exclude):
    """
    Process a single date for LODO analysis.
    Excludes the specified date and fits correlation model.
    
    Args:
        date_to_exclude (str): The date to exclude from the analysis.
        
    Returns:
        Optional[Dict]: A dictionary containing the fitted lambda and other parameters if successful, or error details.
    """
    try:
        # Access worker-global context
        complete_df = WORKER_COMPLETE_DF
        edges = WORKER_EDGES
        min_bin_count = WORKER_MIN_BIN_COUNT
        
        if complete_df is None or edges is None or min_bin_count is None:
            return {
                'date': date_to_exclude,
                'error': 'Worker context not properly initialized.',
                'debug': 'context_init_failure'
            }
        
        # Filter out pairs from this date
        subset_df = complete_df[complete_df['date'] != date_to_exclude].copy()
        
        if len(subset_df) < 1000:  # Skip if too little data remains
            return {
                'date': date_to_exclude,
                'error': f'Insufficient data: {len(subset_df)} pairs',
                'debug': 'too_few_pairs',
                'total_pairs': len(complete_df),
                'remaining_pairs': len(subset_df)
            }
        
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
            return {
                'date': date_to_exclude,
                'error': f'Insufficient bins: {len(binned)} bins',
                'debug': 'too_few_bins',
                'total_bins': len(subset_df.groupby('dist_bin', observed=True)),
                'robust_bins': len(binned),
                'min_bin_count': min_bin_count,
                'bin_counts': binned['count'].tolist() if len(binned) > 0 else []
            }
        
        # Fit exponential model
        try:
            distances = binned['mean_dist'].values
            coherences = binned['mean_coh'].values
            weights = binned['count'].values
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(distances)) or np.any(~np.isfinite(coherences)):
                return {
                    'date': date_to_exclude,
                    'error': 'Invalid data: NaN or infinite values',
                    'debug': 'invalid_data'
                }
            
            # Initial parameter estimates
            c_range = coherences.max() - coherences.min()
            if c_range <= 0 or not np.isfinite(c_range):
                return {
                    'date': date_to_exclude,
                    'error': f'Invalid coherence range: {c_range}',
                    'debug': 'invalid_range'
                }
                
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            # Ensure we have clean numpy arrays
            distances = np.asarray(distances, dtype=float)
            coherences = np.asarray(coherences, dtype=float)
            weights = np.asarray(weights, dtype=float)
            p0 = np.asarray(p0, dtype=float)
            
            popt, _ = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=([1e-10, 100, -1], [5, 20000, 1]),
                maxfev=5000
            )
            
            return {
                'date': date_to_exclude,
                'lambda_km': float(popt[1]),
                'amplitude': float(popt[0]),
                'offset': float(popt[2]),
                'n_pairs': len(subset_df),
                'n_bins': len(binned)
            }
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
            return {
                'date': date_to_exclude,
                'error': str(e),
                'n_pairs': len(subset_df) if 'subset_df' in locals() else 0,
                'n_bins': len(binned) if 'binned' in locals() else 0,
                'debug': 'fit_failed'
            }
            
    except Exception as e:
        return None

def run_loso_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform Leave-One-Station-Out (LOSO) analysis on the complete dataset.
    Tests stability by excluding each station and re-fitting correlation model.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        Dict: A dictionary containing the results of the LOSO analysis.
    """
    print_status("Starting Leave-One-Station-Out (LOSO) analysis...", "PROCESS")
    
    # Get all unique stations
    unique_stations = pd.unique(complete_df[['station_i', 'station_j']].values.ravel())
    
    # OPTIMIZATION: Sample stations for computational efficiency
    max_stations_to_test = TEPConfig.get_int('TEP_LOSO_SAMPLE_SIZE', 50)  # Default: 50 stations
    
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
    
    # OPTIMIZATION: Use parallel processing
    max_workers = min(8, mp.cpu_count())  # Use 8 workers
    print_status(f"Using parallel processing with {max_workers} workers for LOSO analysis", "INFO")
    
    # Process stations in batches
    batch_size = max(5, max_workers)
    
    for batch_start in range(0, len(stations_to_test), batch_size):
        batch_end = min(batch_start + batch_size, len(stations_to_test))
        batch_stations = stations_to_test[batch_start:batch_end]
        
        print_status(f"Processing LOSO batch {batch_start//batch_size + 1}: stations {batch_start+1}-{batch_end}/{len(stations_to_test)}", "PROCESS")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers,
                                     initializer=_init_loso_worker_context,
                                     initargs=(complete_df, edges, min_bin_count)) as executor:
                
                # Submit batch of tasks
                future_to_station = {}
                global_station_idx = batch_start # Track overall progress
                for i, station in enumerate(batch_stations):
                    global_station_idx += 1
                    print_status(f"  Submitting station {station} for processing ({i+1}/{len(batch_stations)}, total {global_station_idx}/{len(stations_to_test)})", "DEBUG")
                    future = executor.submit(_process_single_station_loso, station)
                    future_to_station[future] = station
                
                print_status(f"  Waiting for {len(batch_stations)} stations to complete...", "PROCESS")
                completed_count = 0
                
                # Collect results as they complete
                for future in as_completed(future_to_station, timeout=600):  # 10 minute timeout for batch
                    station = future_to_station[future]
                    completed_count += 1
                    
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per result
                        if result is not None and 'lambda_km' in result:
                            lambda_estimates.append(result['lambda_km'])
                            print_status(f"  [{completed_count}/{len(batch_stations)}] Station {station}: λ = {result['lambda_km']:.1f} km", "SUCCESS")
                        elif result is not None and 'error' in result:
                            print_status(f"  [{completed_count}/{len(batch_stations)}] Station {station}: {result['error']}", "WARNING")
                        else:
                            print_status(f"  [{completed_count}/{len(batch_stations)}] Station {station}: No result returned", "WARNING")
                    except Exception as e:
                        print_status(f"  [{completed_count}/{len(batch_stations)}] Station {station} failed: {e}", "WARNING")
                        continue
                        
        except Exception as e:
            print_status(f"Batch processing failed: {e}", "ERROR")
            # Sequential fallback for this batch
            for station in batch_stations:
                mask = (complete_df['station_i'] != station) & (complete_df['station_j'] != station)
                subset_df = complete_df[mask].copy()
                
                if len(subset_df) < 1000:
                    continue
                
                # Quick sequential processing
                subset_df['dist_bin'] = pd.cut(subset_df['dist_km'], bins=edges, right=False)
                binned = subset_df.groupby('dist_bin', observed=True).agg({
                    'dist_km': 'mean', 'coherence': 'mean', 'station_i': 'count'
                }).rename(columns={'station_i': 'count'})
                binned.columns = ['mean_dist', 'mean_coh', 'count']
                binned = binned[binned['count'] >= min_bin_count]
                
                if len(binned) >= 3:
                    try:
                        distances, coherences = binned['mean_dist'].values, binned['mean_coh'].values
                        c_range = coherences.max() - coherences.min()
                        p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
                        weights = binned['count'].values
                        popt, _ = curve_fit(correlation_model, distances, coherences, p0=p0, 
                                          sigma=1.0/np.sqrt(weights), bounds=([1e-10, 100, -1], [5, 20000, 1]), maxfev=5000)
                        lambda_estimates.append(popt[1])
                        print_status(f"Sequential {station}: λ = {popt[1]:.1f} km", "SUCCESS")
                    except:
                        continue
    
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
    return results

def run_lodo_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform Leave-One-Day-Out (LODO) analysis on the complete dataset.
    Tests stability by excluding each day and re-fitting correlation model.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        
    Returns:
        Dict: A dictionary containing the results of the LODO analysis.
    """
    print_status("Starting Leave-One-Day-Out (LODO) analysis...", "PROCESS")
    
    # Get all unique dates
    unique_dates = complete_df['date'].unique()
    
    # OPTIMIZATION: Sample days for computational efficiency
    max_days_to_test = TEPConfig.get_int('TEP_LODO_SAMPLE_SIZE', 100)  # Default: 100 days
    
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
    
    # Try parallel processing first
    try:
        max_workers = min(8, mp.cpu_count())  # Use 8 workers
        batch_size = max(10, max_workers)  # Process in batches
        
        print_status(f"Using parallel processing with {max_workers} workers for LODO analysis", "INFO")
        
        # Process dates in batches
        for batch_start in range(0, len(dates_to_test), batch_size):
            batch_end = min(batch_start + batch_size, len(dates_to_test))
            batch_dates = dates_to_test[batch_start:batch_end]
            
            print_status(f"Processing LODO batch {batch_start//batch_size + 1}: dates {batch_start+1}-{batch_end}/{len(dates_to_test)}", "PROCESS")
            
            with ProcessPoolExecutor(max_workers=max_workers,
                                   initializer=_init_lodo_worker_context,
                                   initargs=(complete_df, edges, min_bin_count)) as executor:
                
                # Submit tasks
                future_to_date = {}
                for i, date in enumerate(batch_dates):
                    print_status(f"  Submitting date {date} for processing ({i+1}/{len(batch_dates)})", "DEBUG")
                    future = executor.submit(_process_single_date_lodo, date)
                    future_to_date[future] = date
                
                print_status(f"  Waiting for {len(batch_dates)} dates to complete...", "PROCESS")
                completed_count = 0
                
                # Collect results as they complete
                for future in as_completed(future_to_date, timeout=600):  # 10 minute timeout for batch
                    date = future_to_date[future]
                    completed_count += 1
                    
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per result
                        if result is not None and 'lambda_km' in result:
                            lambda_estimates.append(result['lambda_km'])
                            print_status(f"  [{completed_count}/{len(batch_dates)}] Date {date}: λ = {result['lambda_km']:.1f} km", "SUCCESS")
                        elif result is not None and 'error' in result:
                            print_status(f"  [{completed_count}/{len(batch_dates)}] Date {date}: {result['error']}", "WARNING")
                        else:
                            print_status(f"  [{completed_count}/{len(batch_dates)}] Date {date}: No result returned", "WARNING")
                    except Exception as e:
                        print_status(f"  [{completed_count}/{len(batch_dates)}] Date {date} failed: {e}", "WARNING")
                        continue
                        
    except Exception as e:
        print_status(f"Parallel processing failed: {e}", "ERROR")
        print_status("Falling back to sequential processing...", "WARNING")
        
        # Sequential fallback
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
                    bounds=([1e-10, 100, -1], [5, 20000, 1]),
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

def _perform_bootstrap_fit(complete_df: pd.DataFrame, edges: np.ndarray, min_bin_count: int) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
    """
    Fits the correlation model on a single bootstrap sample.
    
    Args:
        complete_df (pd.DataFrame): The complete pair-level dataset.
        edges (np.ndarray): Bin edges for distance.
        min_bin_count (int): Minimum number of data points per bin.
        
    Returns:
        Tuple[Optional[np.ndarray], bool, Optional[str]]: A tuple containing:
            - fitted_params (Optional[np.ndarray]): Fitted model parameters if successful, None otherwise.
            - success_flag (bool): True if fitting was successful, False otherwise.
            - error_message (Optional[str]): Error message if fitting failed, None otherwise.
    """
    try:
        # Randomly sample pairs for the bootstrap sample with replacement.
        # This is a simplified row-wise bootstrap of individual pairs.
        # A more complex 'block bootstrap' would typically resample larger units
        # (e.g., stations or days) to account for inherent spatial or temporal dependencies.
        
        # Create a copy of complete_df to avoid modifying the original
        bootstrap_df = complete_df.sample(frac=1, replace=True).reset_index(drop=True)
        
        # Bin the data
        bootstrap_df['dist_bin'] = pd.cut(bootstrap_df['dist_km'], bins=edges, right=False)
        binned = bootstrap_df.groupby('dist_bin', observed=True).agg(
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
            p0 = [c_range, TEPConfig.get_float('TEP_INITIAL_LAMBDA_GUESS'), coherences.min()]
            
            popt, pcov = curve_fit(
                correlation_model, distances, coherences,
                p0=p0, sigma=1.0/np.sqrt(weights),
                bounds=([1e-10, 100, -1], [2, 20000, 1]),
                maxfev=5000
            )
            
            return popt, True, None
            
        except Exception as e:
            return None, False, str(e)
            
    except Exception as e:
        return None, False, str(e)

def run_pairwise_bootstrap_analysis(complete_df: pd.DataFrame) -> Dict:
    """
    Perform pairwise bootstrap cross-validation analysis.
    
    This method resamples individual pairs with replacement to assess model stability
    and parameter uncertainty, distinguishing it from a block bootstrap that would
    resample larger, dependent blocks of data (e.g., stations or days).
    """
    print_status("Starting pairwise bootstrap analysis...", "PROCESS")

    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    bootstrap_samples = TEPConfig.get_int('TEP_BOOTSTRAP_SAMPLES')
    lambda_estimates = []

    for i in range(bootstrap_samples):
        if (i + 1) % 10 == 0:
            print_status(f"Processing bootstrap sample {i+1}/{bootstrap_samples}...", "PROCESS")
        
        fitted_params, success, error_msg = _perform_bootstrap_fit(complete_df, edges, min_bin_count)
        
        if success:
            lambda_estimates.append(fitted_params[1])
        else:
            print_status(f"Bootstrap sample {i+1} failed to fit: {error_msg}", "WARNING")

    if not lambda_estimates:
        return {'success': False, 'error': 'No successful fits in bootstrap analysis'}
    
    results = {
        'success': True,
        'lambda_mean': float(np.mean(lambda_estimates)),
        'lambda_std': float(np.std(lambda_estimates)),
        'lambda_min': float(np.min(lambda_estimates)),
        'lambda_max': float(np.max(lambda_estimates)),
        'n_successful_fits': len(lambda_estimates),
        'n_bootstrap_samples': bootstrap_samples,
        'lambda_values': [float(x) for x in lambda_estimates],
        'coefficient_of_variation': float(np.std(lambda_estimates) / np.mean(lambda_estimates))
    }
    
    print_status(f"Pairwise Bootstrap CV complete: λ = {results['lambda_mean']:.1f} ± {results['lambda_std']:.1f} km (CV = {results['coefficient_of_variation']:.3f})", "SUCCESS")
    return results

def main():
    """Main execution function of the comprehensive cross-validation suite.
    
    This function orchestrates the loading of data, running of various cross-validation
    analyses (monthly, station block, LOSO, LODO), and saving of results for each
    analysis center.
    """
    start_time = time.time()
    
    print_status("TEP GNSS Analysis Package v0.13 - STEP 3.0: Cross-Validation Suite (Comprehensive)", "TITLE")
    print_status("=" * 70, "INFO")
    
    # Validate inputs
    validate_directory_exists(ROOT / "results" / "tmp", "Step 2.0 pair files directory")
    validate_directory_exists(ROOT / "results" / "outputs", "Output directory")
    
    # Configuration summary
    print_status("Configuration:", "INFO")
    print_status(f"  Monthly CV enabled: {TEPConfig.get_bool('TEP_ENABLE_MONTHLY_CV')}", "INFO")
    print_status(f"  Station block CV enabled: {TEPConfig.get_bool('TEP_ENABLE_STATION_BLOCKS_CV')}", "INFO")
    print_status(f"  LOSO CV enabled: {TEPConfig.get_bool('TEP_ENABLE_LOSO_CV', True)}", "INFO")
    print_status(f"  LODO CV enabled: {TEPConfig.get_bool('TEP_ENABLE_LODO_CV', True)}", "INFO")
    print_status(f"  Pairwise Bootstrap CV enabled: {TEPConfig.get_bool('TEP_ENABLE_BOOTSTRAP_CV', True)}", "INFO")
    print_status(f"  Monthly folds limit: {TEPConfig.get_int('TEP_MONTHLY_CV_FOLDS', 12)}", "INFO")
    print_status(f"  Station block size: {TEPConfig.get_int('TEP_STATION_BLOCK_SIZE', 10)}", "INFO")
    print_status(f"  LOSO sample size: {TEPConfig.get_int('TEP_LOSO_SAMPLE_SIZE', 50)}", "INFO")
    print_status(f"  LODO sample size: {TEPConfig.get_int('TEP_LODO_SAMPLE_SIZE', 100)}", "INFO")
    print_status(f"  Bootstrap samples: {TEPConfig.get_int('TEP_BOOTSTRAP_SAMPLES', 200)}", "INFO")
    print_status(f"  Memory limit: {TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')} GB", "INFO")
    
    # Determine analysis centers to process
    analysis_centers = []
    for ac in ['code', 'esa_final', 'igs_combined']:
        pair_files = list(Path(ROOT / "results" / "tmp").glob(f"step_2_0_pairs_{ac}_*.csv"))
        if pair_files:
            analysis_centers.append(ac)
        else:
            print_status(f"No pair files found for {ac}, skipping", "WARNING")
    
    if not analysis_centers:
        print_status("No analysis centers found with pair data", "ERROR")
        return
    
    print_status(f"Processing {len(analysis_centers)} analysis centers: {', '.join(analysis_centers)}", "INFO")
    
    # Process each analysis center
    for ac in analysis_centers:
        print_status(f"\nProcessing analysis center: {ac.upper()}", "PROCESS")
        print_status("-" * 50, "INFO")
        
        # Run comprehensive cross-validation suite
        results = run_comprehensive_cross_validation_analysis(ac)
        
        # Save results with better error handling
        output_file = ROOT / "results" / "outputs" / f"step_3_0_cross_validation_suite_{ac}.json"
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
        
        # Memory cleanup
        gc.collect()
        check_memory_usage()
    
    # Final summary
    total_time = time.time() - start_time
    print_status("=" * 70, "INFO")
    print_status(f"Block-wise cross-validation completed in {total_time:.1f} seconds", "SUCCESS")
    print_status(f"Results saved for {len(analysis_centers)} analysis centers", "SUCCESS")

if __name__ == "__main__":
    main()
