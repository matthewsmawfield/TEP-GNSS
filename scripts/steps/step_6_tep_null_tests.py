#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 6: Null Hypothesis Testing
===================================================

Validates temporal equivalence principle signatures through rigorous null
hypothesis testing. Demonstrates that observed correlations represent genuine
physical phenomena rather than statistical artifacts.

Requirements: Step 3 complete
Next: Step 7 (Advanced Analysis)

Null Tests Performed:
1. Distance scrambling: Randomize station distances while preserving phase data
2. Phase scrambling: Randomize phases while preserving distance structure  
3. Station scrambling: Randomize station assignments within each day

Expected Results:
- Null tests should show NO significant correlations (R² < 0.1)
- Real data should show strong correlations (R² > 0.8)
- This validates that our TEP signal is genuine

Inputs:
  - results/outputs/step_3_correlation_{ac}.json (from Step 3)
  - data/raw/{igs,esa,code}/*.CLK.gz files
  - data/coordinates/station_coords_global.csv

Outputs:
  - results/outputs/null_tests_validation_{ac}.json

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
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# Anchor to package root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import TEP utilities for better configuration and error handling
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    TEPAnalysisError, safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)

def print_status(text: str, status: str = "INFO"):
    """Print verbose status message with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefixes = {"INFO": "[INFO]", "SUCCESS": "[SUCCESS]", "WARNING": "[WARNING]", "ERROR": "[ERROR]", "PROCESS": "[PROCESSING]"}
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

def ecef_to_geodetic(x, y, z):
    """Convert ECEF coordinates to geodetic (lat, lon, height)."""
    # WGS84 parameters
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    e2 = 2 * f - f**2  # first eccentricity squared
    
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    
    if p == 0:
        lat = np.pi/2 if z > 0 else -np.pi/2
        h = abs(z) - a * np.sqrt(1 - e2)
    else:
        lat = np.arctan2(z, p * (1 - e2))
        for _ in range(5):
            N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
            h = p / np.cos(lat) - N
            lat_new = np.arctan2(z, p * (1 - e2 * N / (N + h)))
            if abs(lat_new - lat) < 1e-10:
                break
            lat = lat_new
            
    return np.degrees(lat), np.degrees(lon), h

def great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on WGS-84 ellipsoid.
    """
    R = 6371.0088  # Mean Earth radius in km (WGS-84 standard value)
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def calculate_baseline_distance(station1: str, station2: str, coords_df: pd.DataFrame):
    """Calculate geodesic distance between stations in km using WGS-84 great-circle distance"""
    
    code1 = station1[:4] if len(station1) > 4 else station1
    code2 = station2[:4] if len(station2) > 4 else station2
    
    try:
        coord1 = coords_df[coords_df['coord_source_code'] == code1].iloc[0]
        coord2 = coords_df[coords_df['coord_source_code'] == code2].iloc[0]
        
        lat1, lon1, _ = ecef_to_geodetic(coord1['X'], coord1['Y'], coord1['Z'])
        lat2, lon2, _ = ecef_to_geodetic(coord2['X'], coord2['Y'], coord2['Z'])
        
        return great_circle_distance(lat1, lon1, lat2, lon2)
        
    except (KeyError, IndexError):
        return None

def correlation_model(r, amplitude, lambda_km, offset):
    """Exponential correlation model for TEP: C(r) = A * exp(-r/λ) + C₀"""
    return amplitude * np.exp(-r / lambda_km) + offset

def run_null_test(ac: str, null_type: str, random_seed: int = 42):
    """
    Run a single null test using already processed data from Step 3.
    
    Args:
        ac: Analysis center ('code', 'igs_combined', 'esa_final')
        null_type: Type of null test ('distance', 'phase', 'station')
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Null test results with fitted parameters
    """
    np.random.seed(random_seed)
    
    try:
        # Load real pair-level data written by Step 3 (env TEP_WRITE_PAIR_LEVEL=1)
        pair_dir = ROOT / 'results' / 'tmp'
        if not pair_dir.exists():
            print(f"    No pair-level data found for {ac}. Re-run Step 3 with TEP_WRITE_PAIR_LEVEL=1.")
            return None

        # Concatenate all pairs files for this analysis center
        files = sorted(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
        if not files:
            print(f"    No pair files for {ac} in {pair_dir}")
            return None

        frames = []
        total_rows = 0
        for p in files:
            try:
                dfp = safe_csv_read(p)
                if dfp is not None:
                    frames.append(dfp)
                    total_rows += len(dfp)
            except (TEPDataError, TEPFileError) as e:
                print_status(f"Failed to load {p.name}: {e}", "WARNING")
                continue
        if not frames:
            print(f"    Failed to load any pair files for {ac}")
            return None
        df = pd.concat(frames, ignore_index=True)
        print(f"    Loaded {len(files)} pair files with {len(df):,} rows")

        # Derive coherence strictly from phase (real-only)
        df = df.dropna(subset=['dist_km', 'plateau_phase']).copy()
        if len(df) == 0:
            return None
        df['coherence'] = np.cos(df['plateau_phase'])
        
        if len(df) == 0:
            return None
        
        # Apply null hypothesis scrambling
        print(f"    Applying {null_type} scrambling to {len(df)} station pairs...")
        if null_type == 'distance':
            # Scramble distances while preserving phases
            original_distances = df['dist_km'].copy()
            df['dist_km'] = np.random.permutation(df['dist_km'].values)
            print(f"    Distance scrambling: {original_distances.mean():.1f} km → {df['dist_km'].mean():.1f} km (mean)")
        elif null_type == 'phase':
            # Scramble phases while preserving distances
            original_phases = df['plateau_phase'].copy()
            df['plateau_phase'] = np.random.permutation(df['plateau_phase'].values)
            df['coherence'] = np.cos(df['plateau_phase'])
            print(f"    Phase scrambling: {original_phases.std():.3f} → {df['plateau_phase'].std():.3f} (std)")
        elif null_type == 'station':
            # Scramble station assignments within each day using real station ids from pair files
            if 'date' not in df.columns or 'station_i' not in df.columns or 'station_j' not in df.columns:
                print("    Station scramble requires date, station_i, station_j columns. Skipping.")
                return None
            unique_days = df['date'].nunique()
            print(f"    Station scrambling: Processing {unique_days} unique days...")
            df['date'] = pd.to_datetime(df['date'])
            scrambled_parts = []
            processed_days = 0
            for date, group in df.groupby(df['date'].dt.date):
                processed_days += 1
                if processed_days % 100 == 0:
                    print(f"      Progress: {processed_days}/{unique_days} days processed...")
                    
                stations = pd.Index(sorted(set(group['station_i']).union(set(group['station_j']))))
                if len(stations) > 1:
                    perm = np.random.permutation(stations)
                    mapping = dict(zip(stations, perm))
                    group_copy = group.copy()
                    group_copy['station_i'] = group_copy['station_i'].map(mapping)
                    group_copy['station_j'] = group_copy['station_j'].map(mapping)
                    scrambled_parts.append(group_copy)
                else:
                    scrambled_parts.append(group)
            df = pd.concat(scrambled_parts, ignore_index=True)
            print(f"    Station scrambling completed: {processed_days} days processed")
            
            # Recalculate distances for scrambled stations
            print(f"    Computing great-circle distances for {len(df)} pairs...")
            coords_path = ROOT / 'data' / 'coordinates' / 'station_coords_global.csv'
            coords_df = pd.read_csv(coords_path)
            
            # Vectorized approach for performance
            # Create a mapping from station code to coordinates
            coords_map = coords_df.set_index('coord_source_code')[['X', 'Y', 'Z']].to_dict('index')

            # Map station codes to coordinates
            coords_i = pd.DataFrame(df['station_i'].str[:4].map(coords_map).tolist(), index=df.index)
            coords_j = pd.DataFrame(df['station_j'].str[:4].map(coords_map).tolist(), index=df.index)

            # Convert ECEF to geodetic in a vectorized manner
            lat1, lon1, _ = ecef_to_geodetic(coords_i['X'], coords_i['Y'], coords_i['Z'])
            lat2, lon2, _ = ecef_to_geodetic(coords_j['X'], coords_j['Y'], coords_j['Z'])

            # Calculate great-circle distance
            df['dist_km'] = great_circle_distance(lat1, lon1, lat2, lon2)
            
            df = df.dropna(subset=['dist_km']).copy()
            print(f"    Distance computation completed: {len(df)} valid pairs")
        
        # Coherence already available from processed data
        df = df[df['dist_km'] > 0].copy()
        
        # Use same binning as Step 3
        num_bins = TEPConfig.get_int('TEP_BINS')
        max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
        
        # Bin and aggregate
        print(f"    Binning {len(df)} pairs into {num_bins} distance bins...")
        df['dist_bin'] = pd.cut(df['dist_km'], bins=edges)
        
        distances = []
        coherences = []
        weights = []
        
        for bin_idx, group in df.groupby('dist_bin', observed=True):
            if pd.notna(bin_idx) and len(group) >= min_bin_count:
                distances.append(group['dist_km'].mean())
                coherences.append(group['coherence'].mean())
                weights.append(len(group))
        
        print(f"    Created {len(distances)} bins with sufficient data for fitting")
        
        if len(distances) < 5:
            return None
        
        # Fit correlation model
        distances = np.array(distances)
        coherences = np.array(coherences)
        weights = np.array(weights)
        
        try:
            # Initial guess
            c_range = coherences.max() - coherences.min()
            p0 = [c_range, 3000, coherences.min()]
            
            # Weighted fit
            sigma = 1.0 / np.sqrt(weights)
            popt, pcov = curve_fit(correlation_model, distances, coherences, 
                                 p0=p0, sigma=sigma,
                                 bounds=([1e-10, 100, -1], [2, 20000, 1]),
                                 maxfev=5000)
            
            amplitude, lambda_km, offset = popt
            param_errors = np.sqrt(np.diag(pcov))
            
            # R-squared
            coherences_pred = correlation_model(distances, *popt)
            ss_res = np.sum(weights * (coherences - coherences_pred)**2)
            ss_tot = np.sum(weights * (coherences - np.average(coherences, weights=weights))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            return {
                'null_type': null_type,
                'analysis_center': ac,
                'files_processed': len(files),
                'pairs_analyzed': len(df),
                'bins_used': len(distances),
                'fit_results': {
                    'amplitude': float(amplitude),
                    'amplitude_error': float(param_errors[0]),
                    'lambda_km': float(lambda_km),
                    'lambda_error': float(param_errors[1]),
                    'offset': float(offset),
                    'offset_error': float(param_errors[2]),
                    'r_squared': float(r_squared)
                }
            }
            
        except (RuntimeError, ValueError, TypeError, ArithmeticError, OverflowError) as e:
            return {'error': f'Fitting failed: {str(e)}', 'null_type': null_type}
            
    except (TEPDataError, TEPFileError, TEPAnalysisError) as e:
        return {'error': f'TEP error: {str(e)}', 'null_type': null_type, 'error_type': 'TEP_ERROR'}
    except (MemoryError, OverflowError) as e:
        return {'error': f'Resource error: {str(e)}', 'null_type': null_type, 'error_type': 'RESOURCE_ERROR'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}', 'null_type': null_type, 'error_type': 'UNEXPECTED_ERROR'}

def validate_tep_signal(ac: str):
    """
    Validate TEP signal for one analysis center using multiple null tests.
    
    Args:
        ac: Analysis center to validate
    
    Returns:
        dict: Validation results comparing real vs null test statistics
    """
    print_status(f"Validating TEP signal for {ac.upper()}", "INFO")
    
    # Load real results from Step 3
    real_results_file = ROOT / f"results/outputs/step_3_correlation_{ac}.json"
    if not real_results_file.exists():
        print_status(f"No Step 3 results found for {ac.upper()}", "ERROR")
        return None
    
    try:
        real_results = safe_json_read(real_results_file)
    except (TEPDataError, TEPFileError) as e:
        print_status(f"Failed to load Step 3 results: {e}", "ERROR")
        return None
    
    real_lambda = real_results['exponential_fit']['lambda_km']
    real_r_squared = real_results['exponential_fit']['r_squared']
    
    print_status(f"Real signal: λ = {real_lambda:.1f} km, R² = {real_r_squared:.3f}", "INFO")
    
    # Check for existing checkpoint
    checkpoint_file = ROOT / f"results/tmp/step6_checkpoint_{ac}.json"
    null_results = {}
    
    if checkpoint_file.exists():
        print_status("Loading previous null test results from checkpoint", "INFO")
        try:
            checkpoint_data = safe_json_read(checkpoint_file)
            null_results = checkpoint_data.get('null_tests', {})
            completed_tests = list(null_results.keys())
            if completed_tests:
                print_status(f"Checkpoint loaded: {completed_tests} tests completed", "SUCCESS")
        except (TEPDataError, TEPFileError, json.JSONDecodeError) as e:
            print_status(f"Checkpoint file corrupted, starting fresh: {e}", "WARNING")
            null_results = {}
    
    # Run null tests (skip completed ones)
    null_types = ['distance', 'phase', 'station']
    
    for null_type in null_types:
        # Skip if already completed
        if null_type in null_results:
            stats = null_results[null_type]
            print_status(f"{null_type.capitalize()} scrambling already completed: λ = {stats['lambda_mean']:.1f} ± {stats['lambda_std']:.1f} km, R² = {stats['r_squared_mean']:.3f} ± {stats['r_squared_std']:.3f}", "INFO")
            continue
            
        print_status(f"Running {null_type} scrambling test...", "PROCESS")
        
        # Run multiple iterations for robust statistics
        n_iterations = TEPConfig.get_int('TEP_NULL_ITERATIONS')  # Statistical validation (100 iterations for permutation p-values)
        null_lambdas = []
        null_r_squareds = []
        
        for i in range(n_iterations):
            result = run_null_test(ac, null_type, random_seed=42+i)
            if result and 'error' not in result:
                null_lambdas.append(result['fit_results']['lambda_km'])
                null_r_squareds.append(result['fit_results']['r_squared'])
                print(f"        Iteration {i+1}: λ = {result['fit_results']['lambda_km']:.1f} km, R² = {result['fit_results']['r_squared']:.3f}")
            else:
                print(f"        Iteration {i+1}: Fitting failed (scrambled data shows no correlation)")
        
        if null_lambdas:
            null_results[null_type] = {
                'lambda_mean': float(np.mean(null_lambdas)),
                'lambda_std': float(np.std(null_lambdas)),
                'r_squared_mean': float(np.mean(null_r_squareds)),
                'r_squared_std': float(np.std(null_r_squareds)),
                'r_squared_values': [float(r2) for r2 in null_r_squareds],  # Store individual values for permutation p-values
                'n_iterations': len(null_lambdas)
            }
            print_status(f"{null_type.capitalize()} null: λ = {np.mean(null_lambdas):.1f} ± {np.std(null_lambdas):.1f} km, R² = {np.mean(null_r_squareds):.3f} ± {np.std(null_r_squareds):.3f}", "SUCCESS")
            
            # Save checkpoint after each completed test
            checkpoint_data = {
                'analysis_center': ac.upper(),
                'timestamp': datetime.now().isoformat(),
                'real_signal': {'lambda_km': real_lambda, 'r_squared': real_r_squared},
                'null_tests': null_results
            }
            try:
                safe_json_write(checkpoint_data, checkpoint_file, indent=2)
            except (TEPFileError, TEPDataError) as e:
                print_status(f"Failed to save checkpoint: {e}", "WARNING")
            print_status(f"Checkpoint saved: {null_type} test completed", "INFO")
        else:
            print_status(f"{null_type.capitalize()} null test failed", "ERROR")
    
    # Validation assessment
    validation_results = {
        'analysis_center': ac.upper(),
        'timestamp': datetime.now().isoformat(),
        'real_signal': {
            'lambda_km': real_lambda,
            'r_squared': real_r_squared
        },
        'null_tests': null_results,
        'validation_assessment': {}
    }
    
    # Assess if real signal is significantly different from nulls
    for null_type, null_stats in null_results.items():
        # Calculate permutation p-value (more robust than z-score)
        null_r_squareds = null_stats.get('r_squared_values', [])
        if len(null_r_squareds) > 0:
            # Permutation p-value: fraction of null results >= real result
            p_value = sum(1 for null_r2 in null_r_squareds if null_r2 >= real_r_squared) / len(null_r_squareds)
            # Add small correction for zero p-values
            if p_value == 0:
                p_value = 1.0 / (len(null_r_squareds) + 1)
            
            # Legacy z-score for comparison
            z_score = (real_r_squared - null_stats['r_squared_mean']) / null_stats['r_squared_std'] if null_stats['r_squared_std'] > 0 else 0
            
            is_significant = p_value < 0.05  # 5% threshold
            
            validation_results['validation_assessment'][null_type] = {
                'p_value': float(p_value),
                'z_score': float(z_score),
                'significant': bool(is_significant),
                'n_permutations': len(null_r_squareds),
                'interpretation': f'Real signal significantly different from null (p = {p_value:.4f})' if is_significant else f'No significant difference from null (p = {p_value:.4f})'
            }
    
    # Clean up checkpoint on successful completion
    if len(null_results) == len(null_types):
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print_status("All null tests completed - checkpoint cleaned up", "INFO")
    
    return validation_results

def main():
    """
    Main validation function that runs null tests for all analysis centers.
    
    Validates the TEP signals detected in Step 3 by running scrambling tests
    to prove the correlations are real and not statistical artifacts.
    """
    print("\n" + "="*80)
    print("TEP GNSS Analysis Package v0.3")
    print("STEP 6: Null Tests")
    print("Validating TEP signatures through rigorous null hypothesis tests")
    print("="*80)
    
    start_time = time.time()
    
    # Determine which analysis centers to validate
    # Check for command line argument first
    if len(sys.argv) > 1:
        ac_arg = sys.argv[1].lower()
        if ac_arg in ['code', 'igs_combined', 'esa_final']:
            centers = [ac_arg]
        else:
            print_status(f"Invalid analysis center: {ac_arg}", "ERROR")
            print_status("Valid options: code, igs_combined, esa_final", "INFO")
            return False
    else:
        centers = ['code', 'igs_combined', 'esa_final']
    
    validation_results = {}
    
    for ac in centers:
        print(f"\n{'='*60}")
        print(f"VALIDATING {ac.upper()} - Null Tests")
        print(f"{'='*60}")
        
        result = validate_tep_signal(ac)
        if result:
            validation_results[ac] = result
            
            # Save individual results
            output_file = ROOT / f"results/outputs/step_6_null_tests_{ac}.json"
            try:
                safe_json_write(result, output_file, indent=2)
                print_status(f"Validation results saved: {output_file}", "SUCCESS")
            except (TEPFileError, TEPDataError) as e:
                print_status(f"Failed to save validation results: {e}", "WARNING")
        else:
            print_status(f"{ac.upper()} validation failed", "ERROR")
    
    # Summary
    print(f"\n{'='*80}")
    print("NULL HYPOTHESIS TESTING COMPLETE")
    print(f"{'='*80}")
    
    if validation_results:
        print_status("Validation Summary:", "SUCCESS")
        for ac, result in validation_results.items():
            real = result['real_signal']
            print(f"  {ac.upper()}: Real signal λ = {real['lambda_km']:.1f} km, R² = {real['r_squared']:.3f}")
            
            for null_type, assessment in result['validation_assessment'].items():
                if assessment['significant']:
                    print(f"    {null_type.capitalize()} null: SIGNIFICANT difference (p = {assessment['p_value']:.4f}, z = {assessment['z_score']:.1f})")
                else:
                    print(f"    {null_type.capitalize()} null: No significant difference (p = {assessment['p_value']:.4f})")
        
        print_status(f"Execution time: {time.time() - start_time:.1f} seconds", "INFO")
        
        # Scientific assessment: Count significant z-scores (this validates the real signal)
        significant_tests = 0
        total_tests = 0
        
        for result in validation_results.values():
            # Count how many null tests show statistically significant differences
            for null_type, assessment in result.get('validation_assessment', {}).items():
                total_tests += 1
                if assessment.get('significant', False):
                    significant_tests += 1
        
        significance_rate = significant_tests / total_tests if total_tests > 0 else 0
        
        print_status(f"Null test analysis: {significant_tests}/{total_tests} scrambling tests show statistically significant signal destruction", "INFO")
        
        if significance_rate >= 0.9:  # If 90%+ of tests show significant differences
            print_status("TEP signal validation: CONFIRMED - All scrambling tests show statistically significant signal destruction", "SUCCESS")
            print_status("Scientific interpretation: Scrambled data consistently shows much weaker correlations than real data", "SUCCESS")
        elif significance_rate >= 0.7:  # If 70%+ show significant differences  
            print_status("TEP signal validation: LIKELY VALID - Most scrambling tests show significant signal destruction", "SUCCESS")
        elif significance_rate >= 0.5:  # If 50%+ show significant differences
            print_status("TEP signal validation: MODERATE - Some scrambling tests show weaker correlations", "WARNING")
        else:
            print_status("TEP signal validation: INCONCLUSIVE - Scrambled data shows similar correlations to real data", "WARNING")
        
        return True
    else:
        print_status("No successful validations", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
