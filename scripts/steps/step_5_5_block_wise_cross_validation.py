#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 5.5: Block-wise Cross-Validation (Gold Standard)
=========================================================================

Performs gold standard block-wise cross-validation of TEP correlation models
using temporal (monthly) and spatial (station block) validation approaches.

This implements the rigorous validation methodology where λ, A, C₀ parameters
are fitted on training blocks and used to predict held-out binned means,
with CV-RMSE/NRMSE and log-likelihood validation metrics.

Requirements: Step 3 complete
Next: Step 6 (Null Tests)

Key Analyses:
1. Monthly temporal cross-validation - split by months, predict held-out months
2. Leave-5-stations-out spatial blocks - remove station groups, test predictive power
3. Cross-validation metrics - CV-RMSE, NRMSE, log-likelihood on predictions
4. Parameter stability assessment - test if λ is consistent across folds

METHODOLOGY: Train on N-1 folds → fit (λ, A, C₀) → predict held-out fold → measure error
This tests whether λ represents real predictive physics vs. curve-fitting artifacts.

Inputs:
  - results/tmp/step_3_pairs_*.csv files (from Step 3)
  - results/outputs/step_3_correlation_{ac}.json (from Step 3)

Outputs:
  - results/outputs/step_5_5_block_wise_cv_{ac}.json

Environment Variables:
  - TEP_ENABLE_MONTHLY_CV: Enable monthly temporal cross-validation (default: True)
  - TEP_ENABLE_STATION_BLOCKS_CV: Enable station block spatial cross-validation (default: True)
  - TEP_MONTHLY_CV_FOLDS: Number of monthly folds to use (default: 10, memory-optimized)
  - TEP_STATION_BLOCK_SIZE: Number of stations per block (default: 10, memory-optimized)
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

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def load_complete_pair_dataset(ac: str) -> pd.DataFrame:
    """
    Load the complete pair-level dataset for an analysis center.
    Reuses the chunked loading approach from step_5 for memory efficiency.
    """
    print_status(f"Loading complete pair dataset for {ac}...", "PROCESS")
    
    # Find all pair files for this analysis center
    pair_files = list(Path(ROOT / "results" / "tmp").glob(f"step_3_pairs_{ac}_*.csv"))
    
    if not pair_files:
        raise TEPFileError(f"No pair files found for analysis center: {ac}")
    
    print_status(f"Found {len(pair_files)} pair files for {ac}", "INFO")
    
    # Load files in chunks to manage memory
    dataframes = []
    for i, file_path in enumerate(pair_files):
        if i % 10 == 0:
            print_status(f"Loading file {i+1}/{len(pair_files)}: {file_path.name}", "PROCESS")
            check_memory_usage()
        
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
        raise TEPDataError(f"No valid pair data loaded for {ac}")
    
    # Concatenate all chunks
    print_status(f"Concatenating {len(dataframes)} data chunks...", "PROCESS")
    complete_df = pd.concat(dataframes, ignore_index=True)
    
    # Clean up memory
    del dataframes
    gc.collect()
    
    print_status(f"Loaded complete dataset: {len(complete_df):,} pairs for {ac}", "SUCCESS")
    check_memory_usage()
    
    return complete_df

def create_monthly_folds(complete_df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Create monthly cross-validation folds.
    Returns list of (month_id, training_data, validation_data) tuples.
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
    for month in unique_months:
        # Validation set: current month
        val_data = complete_df[complete_df['year_month'] == month].copy()
        
        # Training set: all other months
        train_data = complete_df[complete_df['year_month'] != month].copy()
        
        # Use helper function for validation
        if not _validate_fold_data(train_data, val_data, str(month)):
            continue
        
        month_id = str(month)
        folds.append((month_id, train_data, val_data))
    
    print_status(f"Created {len(folds)} valid monthly folds", "SUCCESS")
    return folds

def create_station_block_folds(complete_df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Create leave-N-stations-out cross-validation folds.
    Returns list of (block_id, training_data, validation_data) tuples.
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
        # Validation set: pairs involving any station in the block
        val_mask = (complete_df['station_i'].isin(station_block) | 
                   complete_df['station_j'].isin(station_block))
        val_data = complete_df[val_mask].copy()
        
        # Training set: pairs not involving any station in the block
        train_data = complete_df[~val_mask].copy()
        
        # Use helper function for validation with appropriate thresholds for station blocks
        if not _validate_fold_data(train_data, val_data, f"stations_{i+1:02d}", min_train=1000, min_val=100):
            continue
        
        block_id = f"stations_{i+1:02d}"
        folds.append((block_id, train_data, val_data))
    
    print_status(f"Created {len(folds)} valid station block folds", "SUCCESS")
    return folds

def fit_correlation_model_on_training(train_data: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    Fit exponential correlation model on training data.
    Returns (fitted_params, success_flag).
    """
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    # Bin the training data
    train_data['dist_bin'] = pd.cut(train_data['dist_km'], bins=edges, right=False)
    binned = train_data.groupby('dist_bin', observed=True).agg(
        mean_dist=('dist_km', 'mean'),
        mean_coh=('coherence', 'mean'),
        count=('coherence', 'size')
    ).reset_index()
    
    # Filter for robust bins
    binned = binned[binned['count'] >= min_bin_count].dropna()
    
    if len(binned) < 5:  # Need enough bins for stable fit
        return None, False
    
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
        
        return popt, True
        
    except Exception:
        return None, False

def predict_validation_coherences(val_data: pd.DataFrame, fitted_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Predict coherences on validation data using fitted parameters.
    Returns (predicted_coherences, actual_coherences, success_flag).
    """
    # Analysis parameters
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
    
    # Bin the validation data
    val_data['dist_bin'] = pd.cut(val_data['dist_km'], bins=edges, right=False)
    binned = val_data.groupby('dist_bin', observed=True).agg(
        mean_dist=('dist_km', 'mean'),
        mean_coh=('coherence', 'mean'),
        count=('coherence', 'size')
    ).reset_index()
    
    # Filter for robust bins
    binned = binned[binned['count'] >= min_bin_count].dropna()
    
    if len(binned) < 3:  # Need minimum bins for validation
        return None, None, False
    
    distances = binned['mean_dist'].values
    actual_coherences = binned['mean_coh'].values
    
    # Predict using fitted parameters
    predicted_coherences = correlation_model(distances, *fitted_params)
    
    return predicted_coherences, actual_coherences, True

def _validate_fold_data(train_data: pd.DataFrame, val_data: pd.DataFrame, fold_id: str, 
                       min_train: int = 10000, min_val: int = 1000) -> bool:
    """
    Validate that fold has sufficient data for reliable analysis.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset  
        fold_id: Identifier for the fold
        min_train: Minimum training samples required
        min_val: Minimum validation samples required
        
    Returns:
        bool: True if fold has sufficient data
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
        fold_results: List of individual fold results
        method_name: Name of the CV method for reporting
        
    Returns:
        Dict: Aggregated results with stability metrics
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
        fold_results: List of fold results
        method_name: Name of the CV method
        
    Returns:
        Dict: Outlier detection results
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
        fold_results: List of fold results
        method_name: Name of the CV method
        
    Returns:
        Dict: Convergence diagnostics
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
        monthly_results: Results from monthly CV
        station_results: Results from station block CV
        
    Returns:
        Dict: Consistency analysis
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
    Calculate cross-validation metrics: CV-RMSE, NRMSE, log-likelihood.
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
    """
    print_status("Starting monthly cross-validation analysis...", "PROCESS")
    
    folds = create_monthly_folds(complete_df)
    
    if not folds:
        return {'success': False, 'error': 'No valid monthly folds created'}
    
    fold_results = []
    lambda_estimates = []
    
    for i, (month_id, train_data, val_data) in enumerate(folds):
        progress_pct = (i + 1) / len(folds) * 100
        print_status(f"Processing monthly fold {i+1}/{len(folds)} ({progress_pct:.1f}%): {month_id}", "PROCESS")
        
        # Fit model on training data
        fitted_params, fit_success = fit_correlation_model_on_training(train_data)
        
        if not fit_success:
            print_status(f"Failed to fit model for month {month_id}", "WARNING")
            continue
        
        # Predict on validation data
        predicted, actual, pred_success = predict_validation_coherences(val_data, fitted_params)
        
        if not pred_success:
            print_status(f"Failed to predict for month {month_id}", "WARNING")
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
            'training_size': len(train_data),
            'validation_size': len(val_data)
        }
        
        fold_results.append(fold_result)
        lambda_estimates.append(fitted_params[1])
    
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
    """
    print_status("Starting station block cross-validation analysis...", "PROCESS")
    
    folds = create_station_block_folds(complete_df)
    
    if not folds:
        return {'success': False, 'error': 'No valid station block folds created'}
    
    fold_results = []
    lambda_estimates = []
    
    for i, (block_id, train_data, val_data) in enumerate(folds):
        progress_pct = (i + 1) / len(folds) * 100
        print_status(f"Processing station block fold {i+1}/{len(folds)} ({progress_pct:.1f}%): {block_id}", "PROCESS")
        
        # Fit model on training data
        fitted_params, fit_success = fit_correlation_model_on_training(train_data)
        
        if not fit_success:
            print_status(f"Failed to fit model for block {block_id}", "WARNING")
            continue
        
        # Predict on validation data
        predicted, actual, pred_success = predict_validation_coherences(val_data, fitted_params)
        
        if not pred_success:
            print_status(f"Failed to predict for block {block_id}", "WARNING")
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
            'training_size': len(train_data),
            'validation_size': len(val_data)
        }
        
        fold_results.append(fold_result)
        lambda_estimates.append(fitted_params[1])
    
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

def run_block_wise_cross_validation_analysis(ac: str) -> Dict:
    """
    Main function to run block-wise cross-validation analysis for an analysis center.
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
        
        # Summary statistics
        successful_methods = []
        if results['monthly_cv']['success']:
            successful_methods.append('monthly')
        if results['station_block_cv']['success']:
            successful_methods.append('station_block')
        
        if successful_methods:
            # Aggregate lambda estimates across methods
            all_lambdas = []
            if results['monthly_cv']['success']:
                all_lambdas.extend(results['monthly_cv']['lambda_stability']['lambda_estimates'])
            if results['station_block_cv']['success']:
                all_lambdas.extend(results['station_block_cv']['lambda_stability']['lambda_estimates'])
            
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

def main():
    """Main execution function"""
    start_time = time.time()
    
    print_status("TEP GNSS Analysis Package v0.13 - STEP 5.5: Block-wise Cross-Validation (Enhanced)", "TITLE")
    print_status("=" * 70, "INFO")
    
    # Validate inputs
    validate_directory_exists(ROOT / "results" / "tmp", "Step 3 pair files directory")
    validate_directory_exists(ROOT / "results" / "outputs", "Output directory")
    
    # Configuration summary
    print_status("Configuration:", "INFO")
    print_status(f"  Monthly CV enabled: {TEPConfig.get_bool('TEP_ENABLE_MONTHLY_CV')}", "INFO")
    print_status(f"  Station block CV enabled: {TEPConfig.get_bool('TEP_ENABLE_STATION_BLOCKS_CV')}", "INFO")
    print_status(f"  Monthly folds limit: {TEPConfig.get_int('TEP_MONTHLY_CV_FOLDS')}", "INFO")
    print_status(f"  Station block size: {TEPConfig.get_int('TEP_STATION_BLOCK_SIZE')}", "INFO")
    print_status(f"  Memory limit: {TEPConfig.get_float('TEP_MEMORY_LIMIT_GB')} GB", "INFO")
    
    # Determine analysis centers to process
    analysis_centers = []
    for ac in ['code', 'esa_final', 'igs_combined']:
        pair_files = list(Path(ROOT / "results" / "tmp").glob(f"step_3_pairs_{ac}_*.csv"))
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
        
        # Run block-wise cross-validation
        results = run_block_wise_cross_validation_analysis(ac)
        
        # Save results with better error handling
        output_file = ROOT / "results" / "outputs" / f"step_5_5_block_wise_cv_{ac}.json"
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            success = safe_json_write(results, output_file)
            
            if success:
                print_status(f"Results saved to: {output_file}", "SUCCESS")
            else:
                # Try manual JSON write as fallback
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
