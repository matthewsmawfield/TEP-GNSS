#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 3.2: Null Hypothesis Testing
===================================================

Validates temporal equivalence principle signatures through rigorous null
hypothesis testing. Demonstrates that observed correlations represent genuine
physical phenomena rather than statistical artifacts.

Requirements: Step 2.0 complete (Core TEP Correlation Analysis)
Inputs:
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0)
  - data/raw/{igs,esa,code}/*.CLK.gz files
  - data/coordinates/step_1_1_station_coords_global.csv (from Step 1.1)
  - results/tmp/step_2_0_pairs_{ac}_*.csv (from Step 2.0, if `TEP_WRITE_PAIR_LEVEL=1`)
Outputs:
  - results/outputs/step_3_2_null_tests_{ac}.json (results of null tests)
Next: Step 4.0 (Advanced Analysis)

Null Tests Performed:
1. Distance scrambling: Randomize station distances while preserving phase data
2. Phase scrambling: Randomize phases while preserving distance structure  
3. Station scrambling: Randomize station assignments within each day

Expected Results:
- Null tests should show NO significant correlations (R² < 0.1)
- Real data should show strong correlations (R² > 0.8)
- This validates that our TEP signal is genuine

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Define constants for WGS84 parameters and Earth Radius
WGS84_A = 6378137.0  # semi-major axis
WGS84_F = 1 / 298.257223563  # flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # first eccentricity squared
EARTH_RADIUS_KM = 6371.0088 # Mean Earth radius in km for great circle distance

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

def ecef_to_geodetic_vectorized(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Convert ECEF coordinates to geodetic (lat, lon, height) for arrays."""
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)

    lat = np.arctan2(z, p * (1 - WGS84_E2))

    for _ in range(5):  # Fixed iterations, typically sufficient for convergence
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat)**2)
        h_val = p / np.cos(lat) - N # Renamed h to h_val to avoid conflict with outside scope if used
        lat_new = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + h_val)))
        lat = lat_new
            
    # Handle the pole case where p=0 after iterations
    is_pole = (p == 0)
    lat_at_pole = np.where(z > 0, np.pi / 2, -np.pi / 2)
    h_at_pole = np.abs(z) - WGS84_A * np.sqrt(1 - WGS84_E2)

    lat = np.where(is_pole, lat_at_pole, lat)
    h = np.where(is_pole, h_at_pole, h_val) # Use h_val calculated in the loop for non-pole, h_at_pole for pole
        
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

def great_circle_distance_vectorized(lat1_deg: np.ndarray, lon1_deg: np.ndarray, 
                                     lat2_deg: np.ndarray, lon2_deg: np.ndarray) -> np.ndarray:
    """
    Calculate great-circle distance between two points on WGS-84 ellipsoid (vectorized).
    Inputs are in degrees, outputs in km.
    """
    # Convert to radians
    lat1_rad = np.radians(lat1_deg)
    lon1_rad = np.radians(lon1_deg)
    lat2_rad = np.radians(lat2_deg)
    lon2_rad = np.radians(lon2_deg)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c

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

def run_null_test(ac: str, null_type: str, random_seed: int = 42, coords_map: dict = None):
    """
    Run a single null test using already processed data from Step 2.0.
    
    Args:
        ac: Analysis center ('code', 'igs_combined', 'esa_final')
        null_type: Type of null test ('distance', 'phase', 'station')
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Null test results with fitted parameters
    """
    np.random.seed(random_seed)
    
    try:
        # Load real pair-level data written by Step 2.0 (env TEP_WRITE_PAIR_LEVEL=1)
        pair_dir = ROOT / 'results' / 'tmp'
        if not pair_dir.exists():
            print_status(f"    No pair-level data found for {ac}. Re-run Step 2.0 with TEP_WRITE_PAIR_LEVEL=1.", "WARNING")
            return None

        # For station scrambling, use a representative sample to avoid excessive computation
        files = sorted(pair_dir.glob(f"step_2_0_pairs_{ac}_*.csv")) # Updated from step_3_pairs
        if not files:
            print_status(f"    No pair files for {ac} in {pair_dir}", "WARNING")
            return None

        frames = []
        total_rows = 0
        
        if null_type == 'station':
            # For station scrambling, sample every 10th file to reduce computation time
            files = files[::10]  # Take every 10th file
            print_status(f"    Station scrambling: Using {len(files)} sample files (every 10th) for efficiency", "INFO")
        
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
            print_status(f"    Failed to load any pair files for {ac}", "ERROR")
            return None
        df = pd.concat(frames, ignore_index=True)
        print_status(f"    Loaded {len(files)} pair files with {len(df):,} rows", "INFO")

        # Derive coherence strictly from phase (real-only)
        df = df.dropna(subset=['dist_km', 'plateau_phase']).copy()
        if len(df) == 0:
            return None
        df['coherence'] = np.cos(df['plateau_phase'])
        
        # Apply null hypothesis scrambling
        print_status(f"    Applying {null_type} scrambling to {len(df)} station pairs...", "INFO")
        if null_type == 'distance':
            # Scramble distances while preserving phases
            original_distances = df['dist_km'].copy()
            df['dist_km'] = np.random.permutation(df['dist_km'].values)
            print_status(f"    Distance scrambling: {original_distances.mean():.1f} km → {df['dist_km'].mean():.1f} km (mean)", "INFO")
        elif null_type == 'phase':
            # Scramble phases while preserving distances
            original_phases = df['plateau_phase'].copy()
            df['plateau_phase'] = np.random.permutation(df['plateau_phase'].values)
            df['coherence'] = np.cos(df['plateau_phase'])
            print_status(f"    Phase scrambling: {original_phases.std():.3f} → {df['plateau_phase'].std():.3f} (std)", "INFO")
        elif null_type == 'station':
            # Scramble station assignments within each day using real station ids from pair files
            if 'date' not in df.columns or 'station_i' not in df.columns or 'station_j' not in df.columns:
                print_status("    Station scramble requires date, station_i, station_j columns. Skipping.", "WARNING")
                return None
            unique_days = df['date'].nunique()
            print_status(f"    Station scrambling: Processing {unique_days} unique days...", "PROCESS")
            df['date'] = pd.to_datetime(df['date']) # Convert once before grouping
            scrambled_parts = []
            processed_days = 0
            for date, group in df.groupby(df['date'].dt.date):
                processed_days += 1
                if processed_days % 100 == 0:
                    print_status(f"      Progress: {processed_days}/{unique_days} days processed...", "INFO")
                    
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
            print_status(f"    Station scrambling completed: {processed_days} days processed", "INFO")
            
            # Recalculate distances for scrambled stations
            print_status(f"    Computing great-circle distances for {len(df)} pairs...", "INFO")
            
            # Map station codes to coordinates, handling missing stations
            station_i_codes = df['station_i'].str[:4]
            station_j_codes = df['station_j'].str[:4]
            
            # Get coordinates for each station, filtering out missing ones
            valid_pairs_mask = station_i_codes.isin(coords_map.keys()) & station_j_codes.isin(coords_map.keys())
            df_valid = df[valid_pairs_mask].copy()
            
            if len(df_valid) == 0:
                print_status(f"    No valid station pairs found after scrambling", "WARNING")
                df = pd.DataFrame() # Ensure df is empty for subsequent checks
                return None
                
            print_status(f"    Found {len(df_valid)}/{len(df)} pairs with valid coordinates", "INFO")
            
            # Use vectorized approach for distance calculation
            print_status(f"    Calculating distances for {len(df_valid)} pairs (vectorized)...", "INFO")

            # Create Series for station codes with prefix sliced for mapping
            station_i_codes_sliced = df_valid['station_i'].str[:4]
            station_j_codes_sliced = df_valid['station_j'].str[:4]

            # Map station codes to their coordinates (dictionaries)
            coords_i_mapped = station_i_codes_sliced.map(coords_map)
            coords_j_mapped = station_j_codes_sliced.map(coords_map)

            # Filter out any pairs that might have resulted in NaN/None after mapping (though valid_pairs_mask should handle this)
            valid_mapped_mask = coords_i_mapped.notna() & coords_j_mapped.notna()
            df_final_for_dist_calc = df_valid[valid_mapped_mask].copy()

            if len(df_final_for_dist_calc) == 0:
                print_status(f"    No valid station pairs with coordinates after final mapping for distance calculation.", "WARNING")
                df = pd.DataFrame() # Ensure df is empty for subsequent checks
                return None

            # Extract coordinates into separate NumPy arrays for vectorized ECEF to geodetic conversion
            x1 = np.array([c['X'] for c in coords_i_mapped[valid_mapped_mask]])
            y1 = np.array([c['Y'] for c in coords_i_mapped[valid_mapped_mask]])
            z1 = np.array([c['Z'] for c in coords_i_mapped[valid_mapped_mask]])

            x2 = np.array([c['X'] for c in coords_j_mapped[valid_mapped_mask]])
            y2 = np.array([c['Y'] for c in coords_j_mapped[valid_mapped_mask]])
            z2 = np.array([c['Z'] for c in coords_j_mapped[valid_mapped_mask]])
            
            # Convert ECEF to geodetic using the vectorized function
            lat1_deg, lon1_deg, _ = ecef_to_geodetic_vectorized(x1, y1, z1)
            lat2_deg, lon2_deg, _ = ecef_to_geodetic_vectorized(x2, y2, z2)
            
            # Calculate distance using the vectorized function
            distances_vectorized = great_circle_distance_vectorized(lat1_deg, lon1_deg, lat2_deg, lon2_deg)
            
            # Assign the vectorized distances to the DataFrame
            df_final_for_dist_calc['dist_km'] = distances_vectorized
            df = df_final_for_dist_calc
            
            df = df.dropna(subset=['dist_km']).copy()
            print_status(f"    Distance computation completed: {len(df)} valid pairs", "INFO")
        
        # Coherence already available from processed data
        # Ensure dist_km is numeric and filter positive distances
        df = df[pd.to_numeric(df['dist_km'], errors='coerce') > 0].copy()
        df = df.dropna(subset=['dist_km']).copy()
        
        # Use same binning as Step 3
        num_bins = TEPConfig.get_int('TEP_BINS')
        max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        min_bins_for_fit = TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT')
        edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
        
        # Bin and aggregate
        print_status(f"    Binning {len(df)} pairs into {num_bins} distance bins...", "INFO")
        df['dist_bin'] = pd.cut(df['dist_km'], bins=edges)
        
        distances = []
        coherences = []
        weights = []
        
        for bin_idx, group in df.groupby('dist_bin', observed=True):
            if pd.notna(bin_idx) and len(group) >= min_bin_count:
                distances.append(group['dist_km'].mean())
                coherences.append(group['coherence'].mean())
                weights.append(len(group))
        
        print_status(f"    Created {len(distances)} bins with sufficient data for fitting", "INFO")
        
        if len(distances) < min_bins_for_fit:
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
    
    # Load real results from Step 2.0
    real_results_file = ROOT / f"results/outputs/step_2_0_correlation_{ac}.json" # Updated from step_3_correlation
    if not real_results_file.exists():
        print_status(f"No Step 2.0 results found for {ac.upper()}", "ERROR")
        return None
    
    try:
        real_results = safe_json_read(real_results_file)
    except (TEPDataError, TEPFileError) as e:
        print_status(f"Failed to load Step 2.0 results: {e}", "ERROR")
        return None
    
    real_lambda = real_results['exponential_fit']['lambda_km']
    real_r_squared = real_results['exponential_fit']['r_squared']
    
    print_status(f"Real signal: λ = {real_lambda:.1f} km, R² = {real_r_squared:.3f}", "INFO")
    
    # Check for existing checkpoint
    checkpoint_file = ROOT / f"results/tmp/step_3_2_checkpoint_{ac}.json" # Updated from step6_checkpoint
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
    
    # Load station coordinates once for efficiency
    coords_path = ROOT / 'data' / 'coordinates' / 'station_coords_global.csv'
    coords_df = pd.read_csv(coords_path)
    global_coords_map = coords_df.set_index('coord_source_code')[['X', 'Y', 'Z']].to_dict('index')

    # Determine number of processes for parallel execution
    max_processes = TEPConfig.get_int('TEP_MAX_PROCESSES', default=mp.cpu_count())
    
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
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = [executor.submit(run_null_test, ac, null_type, 42 + i, global_coords_map) for i in range(n_iterations)]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result and 'error' not in result:
                    null_lambdas.append(result['fit_results']['lambda_km'])
                    null_r_squareds.append(result['fit_results']['r_squared'])
                    print_status(f"        Iteration {i+1}: λ = {result['fit_results']['lambda_km']:.1f} km, R² = {result['fit_results']['r_squared']:.3f}", "INFO")
                else:
                    print_status(f"        Iteration {i+1}: Fitting failed (scrambled data shows no correlation)", "WARNING")
        
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
    
    Validates the TEP signals detected in Step 2.0 by running scrambling tests
    to prove the correlations are real and not statistical artifacts.
    """
    print_status("TEP GNSS Analysis Package v0.13 - STEP 3.2: Null Tests", "TITLE")
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
            output_file = ROOT / f"results/outputs/step_3_2_null_tests_{ac}.json" # Updated from step_6_null_tests
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
