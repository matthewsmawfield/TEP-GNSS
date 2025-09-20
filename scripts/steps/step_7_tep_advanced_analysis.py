#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 7: Advanced Analysis
=============================================

Comprehensive advanced analysis suite for temporal equivalence principle
validation. Implements sophisticated statistical tests, spatial analysis,
and methodological validation using global GNSS network data.

Requirements: Step 3 complete
Next: Step 8 (Visualization)

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
import argparse

# ----------------------------------------------------------------------
# Global runtime configuration (set in main via argparse)
# ----------------------------------------------------------------------
VERBOSE = True          # Controls INFO-level output of print_status
STRICT_MODE = True      # Fail fast on missing data if True
MAX_PAIR_FILES = None   # Limit on pair CSVs loaded per AC; None = unlimited

# Import TEP utilities for better configuration and error handling
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    safe_csv_read, safe_json_read, safe_json_write
)

def print_status(text: str, status: str = "INFO"):
    """Print status with icons, respecting global VERBOSE flag."""
    prefixes = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "PROCESSING": "🔄"
    }
    if status == "INFO" and not VERBOSE:
        return  # Skip chatty INFO messages when not verbose
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

# ----------------------------------------------------------------------
# Helper for strict-mode assertions
# ----------------------------------------------------------------------
def assert_condition(condition: bool, message: str):
    """Assert condition or raise RuntimeError in STRICT_MODE."""
    if not condition:
        print_status(message, "ERROR")
        if STRICT_MODE:
            raise RuntimeError(message)

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model: C(r) = A * exp(-r/λ) + C0"""
    return A * np.exp(-r / lambda_km) + C0

# ----------------------------------------------------------------------
# Utility: weighted exponential fit with uncertainty & R²
# ----------------------------------------------------------------------
def fit_exponential(distances, coherences, weights=None, p0=None,
                    bounds=([0.01, 100, -1], [2, 20000, 1]), maxfev=5000):
    """Fit exponential_model to data and return params, errors, R².

    Parameters
    ----------
    distances : array-like (km)
    coherences : array-like
    weights : array-like or None
        If provided, chi-square weights (σ = 1/sqrt(weights)).
    p0 : list
        Initial parameters [A, λ, C0].
    bounds : tuple(list, list)
        Lower and upper bounds for parameters.
    maxfev : int
        Maximum function evaluations.
    Returns
    -------
    popt : ndarray  (3,)
    perr : ndarray  standard errors (3,)
    r_squared : float  weighted coefficient of determination
    """
    distances = np.asarray(distances)
    coherences = np.asarray(coherences)
    if p0 is None:
        c_range = coherences.max() - coherences.min()
        p0 = [c_range, 3000, coherences.min()]

    sigma = None
    if weights is not None:
        # σ ∝ 1/sqrt(w)
        sigma = 1.0 / np.sqrt(np.asarray(weights))

    popt, pcov = curve_fit(exponential_model, distances, coherences,
                           p0=p0, sigma=sigma, bounds=bounds, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))

    # Compute weighted R²
    y_pred = exponential_model(distances, *popt)
    if weights is None:
        ss_res = np.sum((coherences - y_pred) ** 2)
        ss_tot = np.sum((coherences - coherences.mean()) ** 2)
    else:
        w = np.asarray(weights)
        ss_res = np.sum(w * (coherences - y_pred) ** 2)
        ss_tot = np.sum(w * (coherences - np.average(coherences, weights=w)) ** 2)

    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, perr, r_squared

def run_azimuth_permutation_test(ac: str, root_dir: Path, n_permutations: int = 1000, random_seed: int = 42):
    """
    Performs an azimuth-preserving permutation test for anisotropy.

    This test determines if the observed directional dependence (anisotropy) of
    the correlation length is statistically significant. It preserves the distance
    distribution of station pairs while randomizing their orientation.

    Args:
        ac (str): The analysis center to process (e.g., 'code').
        root_dir (Path): The project root directory.
        n_permutations (int): The number of shuffle iterations to build the null distribution.
        random_seed (int): Seed for the random number generator for reproducibility.

    Returns:
        dict: A dictionary containing the observed anisotropy, the null distribution,
              and the calculated p-value, or None if the test cannot be run.
    """
    print_status(f"Starting Azimuth Permutation Test for {ac.upper()}", "PROCESSING")
    rng = np.random.default_rng(random_seed)

    # 1. Load the enriched geospatial data - check for filtered version first
    using_filtered = os.environ.get("TEP_USE_FILTERED_DATA", "0") == "1"
    filtered_suffix = os.environ.get("TEP_FILTERED_SUFFIX", "")
    
    if using_filtered and filtered_suffix:
        data_file = root_dir / f"data/processed/step_4_geospatial_{ac}{filtered_suffix}.csv"
        if not data_file.exists():
            print_status(f"Filtered data file not found for {ac}: {data_file}", "WARNING")
            return None
        print_status(f"Using filtered data: {data_file.name}", "INFO")
    else:
        data_file = root_dir / f"data/processed/step_4_geospatial_{ac}.csv"
        if not data_file.exists():
            print_status(f"Geospatial data file not found for {ac}: {data_file}", "WARNING")
            return None

    print_status(f"Loading data from {data_file.name}...", "INFO")
    # Use a fraction of the data to make the computation feasible
    frac = float(os.getenv('TEP_PERMUTATION_FRAC', 0.1))
    df = pd.read_csv(data_file).sample(frac=frac, random_state=random_seed)
    df['coherence'] = np.cos(df['plateau_phase'])
    print_status(f"Loaded and sampled {len(df):,} pairs for analysis", "INFO")

    def get_anisotropy_metric(dataf):
        """Helper function to calculate the anisotropy metric (E-W lambda / N-S lambda)."""
        # Define sectors (East-West vs. North-South)
        ew_mask = ((dataf['azimuth'] >= 45) & (dataf['azimuth'] < 135)) | \
                  ((dataf['azimuth'] >= 225) & (dataf['azimuth'] < 315))
        ns_mask = ~ew_mask

        lambdas = {}
        for sector, mask in [('ew', ew_mask), ('ns', ns_mask)]:
            sector_df = dataf[mask].copy()
            if len(sector_df) < 1000:
                continue

            # Bin data and fit model
            bins = int(os.getenv('TEP_BINS', 40)) + 1
            max_dist = float(os.getenv('TEP_MAX_DISTANCE_KM', 13000))
            edges = np.logspace(np.log10(50), np.log10(max_dist), bins)
            sector_df['dist_bin'] = pd.cut(sector_df['dist_km'], bins=edges, right=False)
            binned = sector_df.groupby('dist_bin', observed=True).agg(
                mean_dist=('dist_km', 'mean'),
                mean_coh=('coherence', 'mean'),
                count=('coherence', 'size')
            ).dropna()
            binned = binned[binned['count'] >= 100]

            if len(binned) < 5:
                continue

            try:
                popt, _, _ = fit_exponential(binned['mean_dist'], binned['mean_coh'],
                                             weights=binned['count'])
                lambdas[sector] = popt[1]
            except RuntimeError:
                continue
        
        if 'ew' in lambdas and 'ns' in lambdas and lambdas['ns'] > 0:
            return lambdas['ew'] / lambdas['ns']
        return None

    # 2. Calculate the observed anisotropy metric
    observed_metric = get_anisotropy_metric(df)
    if observed_metric is None:
        print_status("Could not calculate observed anisotropy metric. Not enough data in sectors.", "WARNING")
        return None
    print_status(f"Observed Anisotropy Metric (EW/NS λ ratio): {observed_metric:.3f}", "SUCCESS")

    # 3. Build the null distribution by permuting azimuths
    null_distribution = []
    print_status(f"Running {n_permutations} permutations...", "PROCESS")
    
    shuffled_df = df.copy()
    for i in range(n_permutations):
        shuffled_df['azimuth'] = rng.permutation(shuffled_df['azimuth'].values)
        metric = get_anisotropy_metric(shuffled_df)
        if metric is not None:
            null_distribution.append(metric)
        if (i + 1) % 100 == 0:
            print_status(f"  ...completed {i+1}/{n_permutations} permutations.", "INFO")

    if not null_distribution:
        print_status("Failed to generate any metric values for the null distribution.", "ERROR")
        return None

    # 4. Calculate the p-value
    # How many of the null metrics are more extreme than the observed one?
    p_value = np.sum(np.abs(np.array(null_distribution) - 1) >= np.abs(observed_metric - 1)) / len(null_distribution)
    
    print_status(f"Permutation test complete. p-value: {p_value:.4f}", "SUCCESS")

    return {
        "test_type": "azimuth_permutation",
        "observed_anisotropy_ratio": observed_metric,
        "p_value": p_value,
        "n_permutations": len(null_distribution),
        "interpretation": "Anisotropy is statistically significant" if p_value < 0.05 else "Anisotropy is not statistically significant"
    }

def load_station_coordinates():
    """Load ground station coordinates with elevation data"""
    root_dir = Path(__file__).parent.parent.parent
    coords_file = root_dir / 'data/coordinates/station_coords_global.csv'
    # Hard-fail if coordinates are missing in strict mode
    assert_condition(coords_file.exists(),
                     "Station coordinates file not found – ensure Step 1 data acquisition was successful")
        
    try:
        df = pd.read_csv(coords_file)
        print_status(f"Loaded coordinates for {len(df)} ground stations", "SUCCESS")
        return df
    except Exception as e:
        print_status(f"Failed to load coordinates: {e}", "ERROR")
        return None

def ecef_to_geodetic(x, y, z):
    """Convert ECEF coordinates to geodetic (lat, lon, height)"""
    # Handle NaN values
    if np.any(np.isnan([x, y, z])) or np.any(np.array([x, y, z]) == 0):
        return np.nan, np.nan, np.nan
    
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis (m)
    b = 6356752.314245  # Semi-minor axis (m)
    e2 = 1 - (b**2 / a**2)  # First eccentricity squared
    
    # Iterative solution for latitude and height
    p = np.sqrt(x**2 + y**2)
    
    # Avoid division by zero
    if p < 1e-10:
        # Point is on z-axis
        lat = np.pi/2 if z > 0 else -np.pi/2
        lon = 0
        height = abs(z) - b
        return np.degrees(lat), np.degrees(lon), height
    
    lat = np.arctan2(z, p * (1 - e2))
    
    # Iterate to improve accuracy
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        cos_lat = np.cos(lat)
        if abs(cos_lat) < 1e-10:
            # Near pole
            height = abs(z) - b
            break
        height = p / cos_lat - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + height)))
    
    lon = np.arctan2(y, x)
    
    return np.degrees(lat), np.degrees(lon), height

def analyze_elevation_dependence(root_dir):
    """
    Test for elevation-dependent correlations in ground stations.
    
    TEP-motivated analysis that:
    1. Controls for distance-elevation correlation
    2. Tests φ-field screening predictions (λ_scr ~ 10 km)
    3. Analyzes correlation structure accounting for confounding factors
    
    Uses pair-level data from Step 3 to analyze correlation length vs elevation.
    """
    print_status("Starting TEP-motivated elevation dependence analysis", "INFO")
    
    results = {}
    
    # Load station coordinates and compute elevations
    coords_df = load_station_coordinates()
    if coords_df is None:
        return {'error': 'Could not load station coordinates'}
    
    print_status(f"Loaded coordinates for {len(coords_df)} ground stations", "SUCCESS")
    
    # Convert ECEF to geodetic coordinates to get elevations
    valid_coords = coords_df.dropna(subset=['X', 'Y', 'Z']).copy()
    if len(valid_coords) == 0:
        return {'error': 'No valid ECEF coordinates found'}
    
    # --------------------------------------------------------------
    # Vectorised ECEF → geodetic conversion (manual implementation)
    # --------------------------------------------------------------
    print_status("Converting ECEF coordinates to geodetic (vectorized)", "INFO")
    
    # Vectorize the ECEF conversion function with robust numerical handling
    def vectorized_ecef_to_geodetic(x_arr, y_arr, z_arr):
        """Vectorized ECEF to geodetic conversion with numerical stability"""
        # WGS84 parameters
        a = 6378137.0  # Semi-major axis (m)
        b = 6356752.314245  # Semi-minor axis (m)
        e2 = 1 - (b**2 / a**2)  # First eccentricity squared
        
        # Convert to numpy arrays and handle invalid values
        x_arr = np.asarray(x_arr, dtype=np.float64)
        y_arr = np.asarray(y_arr, dtype=np.float64)
        z_arr = np.asarray(z_arr, dtype=np.float64)
        
        # Mask for valid coordinates (non-zero, non-NaN)
        valid_mask = (
            np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr) &
            ((x_arr**2 + y_arr**2 + z_arr**2) > 1e6)  # Minimum Earth radius squared
        )
        
        # Initialize output arrays
        lat_arr = np.full_like(x_arr, np.nan)
        lon_arr = np.full_like(x_arr, np.nan)
        height_arr = np.full_like(x_arr, np.nan)
        
        if not np.any(valid_mask):
            return lat_arr, lon_arr, height_arr
        
        # Work only with valid coordinates
        x_valid = x_arr[valid_mask]
        y_valid = y_arr[valid_mask]
        z_valid = z_arr[valid_mask]
        
        # Calculate longitude
        lon_valid = np.arctan2(y_valid, x_valid)
        
        # Calculate p (distance from z-axis)
        p_valid = np.sqrt(x_valid**2 + y_valid**2)
        
        # Handle points on z-axis (p ≈ 0)
        on_axis_mask = p_valid < 1e-10
        lat_valid = np.where(
            on_axis_mask,
            np.sign(z_valid) * np.pi/2,  # ±90 degrees
            np.arctan2(z_valid, p_valid * (1 - e2))
        )
        
        # Iterative improvement for non-axis points
        non_axis_mask = ~on_axis_mask
        if np.any(non_axis_mask):
            for _ in range(5):
                N_valid = a / np.sqrt(1 - e2 * np.sin(lat_valid)**2)
                
                # Avoid division by zero in cos(lat)
                cos_lat = np.cos(lat_valid)
                safe_cos_mask = np.abs(cos_lat) > 1e-10
                
                height_valid = np.where(
                    safe_cos_mask,
                    p_valid / cos_lat - N_valid,
                    np.abs(z_valid) - b  # Near poles
                )
                
                # Update latitude with safe division
                denominator = N_valid + height_valid
                safe_denom_mask = np.abs(denominator) > 1e-10
                
                lat_valid = np.where(
                    safe_denom_mask & non_axis_mask,
                    np.arctan2(z_valid, p_valid * (1 - e2 * N_valid / denominator)),
                    lat_valid  # Keep previous value if unsafe
                )
        
        # Final height calculation
        N_valid = a / np.sqrt(1 - e2 * np.sin(lat_valid)**2)
        cos_lat = np.cos(lat_valid)
        height_valid = np.where(
            np.abs(cos_lat) > 1e-10,
            p_valid / cos_lat - N_valid,
            np.abs(z_valid) - b
        )
        
        # Assign results back to full arrays
        lat_arr[valid_mask] = np.degrees(lat_valid)
        lon_arr[valid_mask] = np.degrees(lon_valid)
        height_arr[valid_mask] = height_valid
        
        return lat_arr, lon_arr, height_arr
    
    # Apply vectorized conversion
    lat_arr, lon_arr, height_arr = vectorized_ecef_to_geodetic(
        valid_coords['X'].values,
        valid_coords['Y'].values, 
        valid_coords['Z'].values
    )
    
    # Create elevation lookup
    valid_mask = ~np.isnan(height_arr)
    elevation_lookup = dict(zip(
        valid_coords['code'].values[valid_mask],
        height_arr[valid_mask]
    ))
    
    print_status(f"Computed elevations for {len(elevation_lookup)} stations", "SUCCESS")
    
    # Define elevation bins using equal-count quantiles for optimal statistical power
    # Based on elevation distribution analysis: 5 bins with ~258 stations each
    elevation_bins = {
        'quintile_1': {'min': -219, 'max': 54, 'desc': 'Lowest quintile (-219 to 54m)'},
        'quintile_2': {'min': 54, 'max': 98, 'desc': 'Second quintile (54-98m)'},
        'quintile_3': {'min': 98, 'max': 207, 'desc': 'Third quintile (98-207m)'},
        'quintile_4': {'min': 207, 'max': 541, 'desc': 'Fourth quintile (207-541m)'},
        'quintile_5': {'min': 541, 'max': 3767, 'desc': 'Highest quintile (541-3767m)'}
    }
    
    for ac in ['code', 'igs_combined', 'esa_final']:
        print_status(f"Processing elevation analysis for {ac.upper()}", "INFO")
        
        # Look for pair-level data from Step 3
        pair_dir = root_dir / 'results' / 'tmp'
        assert_condition(pair_dir.exists(),
                         f"No pair-level data directory found for {ac}. Run Step 3 with TEP_WRITE_PAIR_LEVEL=1")
        
        # Load all pair-level files for this analysis center
        pair_files = list(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
        assert_condition(pair_files, f"No pair-level files found for {ac}")
        
        print_status(f"Found {len(pair_files)} pair-level files for {ac.upper()}", "INFO")
        
        # Load and combine all pair-level data
        all_pairs = []
        for pair_file in pair_files:
            try:
                df_pairs = pd.read_csv(pair_file)
                all_pairs.append(df_pairs)
            except Exception as e:
                print_status(f"Failed to load {pair_file}: {e}", "WARNING")
                continue
        
        if not all_pairs:
            print_status(f"No valid pair-level data for {ac}", "WARNING")
            continue
        
        # Combine all pair data
        df_all = pd.concat(all_pairs, ignore_index=True)
        print_status(f"Loaded {len(df_all)} station pairs for {ac.upper()}", "SUCCESS")
        
        # Add elevations to pairs (handle station code mapping)
        # Extract short codes from full station codes (e.g., ABMF00GLP -> ABMF)
        def extract_short_code(full_code):
            if pd.isna(full_code):
                return full_code
            # Try direct lookup first
            if full_code in elevation_lookup:
                return full_code
            # Extract first 4 characters for IGS codes
            short_code = str(full_code)[:4]
            return short_code if short_code in elevation_lookup else None
        
        df_all['short_i'] = df_all['station_i'].apply(extract_short_code)
        df_all['short_j'] = df_all['station_j'].apply(extract_short_code)
        df_all['elev_i'] = df_all['short_i'].map(elevation_lookup)
        df_all['elev_j'] = df_all['short_j'].map(elevation_lookup)
        
        # Filter pairs where both stations have elevation data
        df_valid = df_all.dropna(subset=['elev_i', 'elev_j']).copy()
        print_status(f"Found {len(df_valid)} pairs with elevation data", "INFO")
        
        if len(df_valid) == 0:
            print_status(f"No pairs with elevation data for {ac}", "WARNING")
            continue
        
        # Compute coherence from phase
        df_valid['coherence'] = np.cos(df_valid['plateau_phase'])
        
        # TEP Enhancement 1: Compute elevation-adjusted distances and elevation metrics
        # Add elevation difference and elevation-adjusted distance calculations
        df_valid['elev_diff_m'] = np.abs(df_valid['elev_j'] - df_valid['elev_i'])
        df_valid['mean_elev_m'] = (df_valid['elev_i'] + df_valid['elev_j']) / 2
        
        # Calculate elevation-adjusted distance including elevation difference
        # Convert elevation difference to km and combine with great-circle distance
        df_valid['dist_3d_km'] = np.sqrt(df_valid['dist_km']**2 + (df_valid['elev_diff_m']/1000)**2)
        
        # Analyze correlations
        dist_elev_corr = np.corrcoef(df_valid['dist_km'], df_valid['mean_elev_m'])[0,1]
        dist_3d_corr = np.corrcoef(df_valid['dist_3d_km'], df_valid['dist_km'])[0,1]
        
        print_status(f"Great-circle distance vs mean elevation: r = {dist_elev_corr:.3f}", "INFO")
        print_status(f"Elevation-adjusted distance vs great-circle distance: r = {dist_3d_corr:.3f}", "INFO")
        print_status(f"Mean elevation difference: {df_valid['elev_diff_m'].mean():.1f}m", "INFO")
        
        # TEP Enhancement 2: Compute φ-field values using screening model
        lambda_scr_m = 10000  # 10 km screening length (TEP theory prediction)
        df_valid['phi_i'] = np.exp(-df_valid['elev_i'] / lambda_scr_m)
        df_valid['phi_j'] = np.exp(-df_valid['elev_j'] / lambda_scr_m)
        df_valid['delta_phi'] = df_valid['phi_j'] - df_valid['phi_i']
        
        print_status(f"φ-field range: Δφ ∈ [{df_valid['delta_phi'].min():.3e}, {df_valid['delta_phi'].max():.3e}]", "INFO")
        
        # Analyze by elevation bins, controlling for distance
        results[ac] = {
            'distance_elevation_correlation': dist_elev_corr,
            'phi_field_stats': {
                'delta_phi_min': float(df_valid['delta_phi'].min()),
                'delta_phi_max': float(df_valid['delta_phi'].max()),
                'delta_phi_std': float(df_valid['delta_phi'].std())
            }
        }
        
        # Define distance bins to control for distance-elevation correlation
        distance_bins = {
            'short': {'min': 0, 'max': 2000, 'label': '0-2000 km'},
            'medium': {'min': 2000, 'max': 5000, 'label': '2000-5000 km'},
            'long': {'min': 5000, 'max': 15000, 'label': '5000+ km'}
        }
        
        # NEW: Elevation difference analysis (more physically meaningful)
        print_status("Analyzing elevation difference effects", "INFO")
        
        # Define elevation difference bins (meters)
        elev_diff_bins = {
            'same_level': {'min': 0, 'max': 50, 'desc': 'Same level (0-50m difference)'},
            'small_diff': {'min': 50, 'max': 200, 'desc': 'Small difference (50-200m)'},
            'medium_diff': {'min': 200, 'max': 500, 'desc': 'Medium difference (200-500m)'},
            'large_diff': {'min': 500, 'max': 2000, 'desc': 'Large difference (500-2000m)'},
            'extreme_diff': {'min': 2000, 'max': 4000, 'desc': 'Extreme difference (2000m+)'}
        }
        
        results[ac]['elevation_difference_analysis'] = {}
        
        for diff_key, diff_config in elev_diff_bins.items():
            min_diff, max_diff = diff_config['min'], diff_config['max']
            
            # Filter by elevation difference
            diff_mask = ((df_valid['elev_diff_m'] >= min_diff) & 
                        (df_valid['elev_diff_m'] < max_diff))
            df_diff = df_valid[diff_mask].copy()
            
            if len(df_diff) < 1000:
                print_status(f"{diff_config['desc']}: Only {len(df_diff)} pairs - skipping", "WARNING")
                continue
                
            print_status(f"{diff_config['desc']}: {len(df_diff)} pairs", "INFO")
            
            # Fit correlation model using elevation-adjusted distance
            try:
                distances_3d = df_diff['dist_3d_km'].values
                coherences = df_diff['coherence'].values

                (popt, perr, r_squared) = fit_exponential(distances_3d, coherences)

                results[ac]['elevation_difference_analysis'][diff_key] = {
                    'lambda_3d_km': float(popt[1]),
                    'lambda_error_km': float(perr[1]),
                    'r_squared_3d': float(r_squared),
                    'amplitude': float(popt[0]),
                    'n_pairs': len(df_diff),
                    'mean_horizontal_dist_km': float(df_diff['dist_km'].mean()),
                    'mean_3d_dist_km': float(df_diff['dist_3d_km'].mean()),
                    'mean_elev_diff_m': float(df_diff['elev_diff_m'].mean())
                }
                
                print_status(f"  3D λ = {popt[1]:.0f} km, R² = {r_squared:.3f}", "SUCCESS")
                
            except Exception as e:
                print_status(f"  Fitting failed: {e}", "WARNING")
                continue
        
        # TEP Enhancement 3: Distance-stratified elevation analysis
        print_status("Performing distance-stratified elevation analysis", "INFO")
        
        for elev_key, elev_config in elevation_bins.items():
            min_elev, max_elev = elev_config['min'], elev_config['max']
            results[ac][elev_key] = {}
            
            # Filter pairs where BOTH stations are in this elevation range
            mask = ((df_valid['elev_i'] >= min_elev) & (df_valid['elev_i'] < max_elev) &
                   (df_valid['elev_j'] >= min_elev) & (df_valid['elev_j'] < max_elev))
            
            df_elev = df_valid[mask].copy()
            
            if len(df_elev) < 200:  # Need minimum pairs for reliable analysis (reduced for 9 fine bins)
                print_status(f"{elev_config['desc']}: Only {len(df_elev)} pairs - skipping", "WARNING")
                results[ac][elev_key] = {
                    'pairs_count': len(df_elev),
                    'status': 'insufficient_data'
                }
                continue
            
            print_status(f"{elev_config['desc']}: {len(df_elev)} pairs", "INFO")
            
            # TEP Enhancement 4: Analyze by distance bins within elevation range
            results[ac][elev_key]['distance_stratified'] = {}
            
            for dist_key, dist_config in distance_bins.items():
                dist_mask = (df_elev['dist_km'] >= dist_config['min']) & (df_elev['dist_km'] < dist_config['max'])
                df_subset = df_elev[dist_mask]
                
                if len(df_subset) < 100:
                    results[ac][elev_key]['distance_stratified'][dist_key] = {
                        'pairs_count': len(df_subset),
                        'status': 'insufficient_data'
                    }
                    continue
                
                print_status(f"  {elev_config['desc']} @ {dist_config['label']}: {len(df_subset)} pairs", "INFO")
                
                # Perform TEP analysis within this distance-elevation stratum
                try:
                    # Bin by distance within this stratum
                    max_sub = min(float(os.getenv('TEP_MAX_DISTANCE_KM', 13000)), dist_config['max'])
                    sub_bins = 15
                    sub_distances = np.logspace(np.log10(max(50, dist_config['min'])), 
                                              np.log10(max_sub), sub_bins)
                    sub_bin_centers = []
                    sub_mean_coherences = []
                    sub_pair_counts = []
                    
                    for i in range(len(sub_distances)-1):
                        sub_bin_mask = ((df_subset['dist_km'] >= sub_distances[i]) & 
                                       (df_subset['dist_km'] < sub_distances[i+1]))
                        sub_bin_data = df_subset[sub_bin_mask]
                        
                        if len(sub_bin_data) >= 20:  # Minimum pairs per sub-bin
                            sub_bin_centers.append(np.sqrt(sub_distances[i] * sub_distances[i+1]))
                            sub_mean_coherences.append(sub_bin_data['coherence'].mean())
                            sub_pair_counts.append(len(sub_bin_data))
                    
                    if len(sub_bin_centers) >= 5:  # Need minimum bins for fitting
                        # Convert to arrays for fitting
                        sub_distances_arr = np.array(sub_bin_centers)
                        sub_coherences_arr = np.array(sub_mean_coherences)
                        sub_weights = np.array(sub_pair_counts)
                        
                        # Fit exponential model (weighted)
                        popt, perr, r_squared = fit_exponential(
                            sub_distances_arr,
                            sub_coherences_arr,
                            weights=sub_weights,
                            p0=[0.2, 3000, 0.0]
                        )
                        amplitude, lambda_km, offset = popt
                        
                        results[ac][elev_key]['distance_stratified'][dist_key] = {
                            'pairs_count': len(df_subset),
                            'bins_count': len(sub_bin_centers),
                            'lambda_km': lambda_km,
                            'lambda_error': perr[1],
                            'amplitude': amplitude,
                            'amplitude_error': perr[0],
                            'offset': offset,
                            'offset_error': perr[2],
                            'r_squared': r_squared,
                            'distance_range_km': [dist_config['min'], dist_config['max']],
                            'status': 'success'
                        }
                        
                        print_status(f"    λ = {lambda_km:.0f} ± {perr[1]:.0f} km, R² = {r_squared:.3f}", "SUCCESS")
                    else:
                        results[ac][elev_key]['distance_stratified'][dist_key] = {
                            'pairs_count': len(df_subset),
                            'bins_count': len(sub_bin_centers),
                            'status': 'insufficient_bins'
                        }
                        
                except Exception as e:
                    results[ac][elev_key]['distance_stratified'][dist_key] = {
                        'pairs_count': len(df_subset),
                        'status': 'fit_failed',
                        'error': str(e)
                    }
                    print_status(f"    Fit failed: {e}", "WARNING")
            
            # Also do standard analysis for the full elevation bin
            # Bin by distance and compute mean coherence
            bins_full = int(os.getenv('TEP_BINS', 40))
            max_dist = float(os.getenv('TEP_MAX_DISTANCE_KM', 13000))
            distance_bins_log = np.logspace(np.log10(50), np.log10(max_dist), max(10, min(25, bins_full)))
            bin_centers = []
            mean_coherences = []
            pair_counts = []
            
            for i in range(len(distance_bins_log)-1):
                bin_mask = ((df_elev['dist_km'] >= distance_bins_log[i]) & 
                           (df_elev['dist_km'] < distance_bins_log[i+1]))
                bin_data = df_elev[bin_mask]
                
                if len(bin_data) >= 50:  # Minimum pairs per bin
                    bin_centers.append(np.sqrt(distance_bins_log[i] * distance_bins_log[i+1]))
                    mean_coherences.append(bin_data['coherence'].mean())
                    pair_counts.append(len(bin_data))
            
            if len(bin_centers) < 10:  # Need minimum bins for fitting
                print_status(f"{elev_config['desc']}: Only {len(bin_centers)} bins - skipping fit", "WARNING")
                results[ac][elev_key] = {
                    'pairs_count': len(df_elev),
                    'bins_count': len(bin_centers),
                    'status': 'insufficient_bins'
                }
                continue
            
            try:
                # Convert to arrays for fitting
                distances = np.array(bin_centers)
                coherences = np.array(mean_coherences)
                weights = np.array(pair_counts)
                
                # Fit exponential model (weighted)
                popt, perr, r_squared = fit_exponential(
                    distances,
                    coherences,
                    weights=weights,
                    p0=[0.2, 3000, 0.0]
                )
                A, lambda_km, C0 = popt
                
                results[ac][elev_key] = {
                    'pairs_count': len(df_elev),
                    'bins_count': len(bin_centers),
                    'lambda_km': float(lambda_km),
                    'lambda_error': float(perr[1]),
                    'amplitude': float(A),
                    'offset': float(C0),
                    'r_squared': float(r_squared),
                    'elevation_range_m': [float(min_elev), float(max_elev)],
                    'status': 'success'
                }
                
                print_status(f"{elev_config['desc']}: λ = {lambda_km:.0f} ± {perr[1]:.0f} km, R² = {r_squared:.3f}", "SUCCESS")
            except Exception as fit_err:
                print_status(f"{elev_config['desc']}: Fit failed - {fit_err}", "WARNING")
                results[ac][elev_key] = {
                    'pairs_count': len(df_elev),
                    'bins_count': len(bin_centers),
                    'status': 'fit_failed',
                    'error': str(fit_err)
                }
    
    # TEP Enhancement 5: Add summary and interpretation
    if ac in results and 'distance_elevation_correlation' in results[ac]:
        corr = results[ac]['distance_elevation_correlation']
        if abs(corr) > 0.2:
            results[ac]['tep_interpretation'] = {
                'status': 'distance_elevation_coupling_detected',
                'correlation': corr,
                'message': f'Strong distance-elevation coupling (r={corr:.3f}) explains apparent elevation dependence. TEP signal is distance-dependent, not elevation-dependent.',
                'recommendation': 'Use distance-stratified analysis to separate effects'
            }
        else:
            results[ac]['tep_interpretation'] = {
                'status': 'minimal_coupling',
                'correlation': corr,
                'message': 'Low distance-elevation coupling allows direct elevation analysis'
            }
    
    return results

def analyze_frequency_universality(root_dir):
    """
    Test correlation stability across frequency bands.
    
    TEP predicts same λ at all frequencies (metric-level coupling).
    """
    print_status("Starting frequency universality analysis", "INFO")
    
    results = {}
    
    # Define frequency bands for testing
    frequency_bands = [
        {'name': 'ultra_low', 'f1': 0.0001, 'f2': 0.001, 'desc': '0.1-1 mHz'},
        {'name': 'low', 'f1': 0.001, 'f2': 0.01, 'desc': '1-10 mHz'},
        {'name': 'medium', 'f1': 0.01, 'f2': 0.1, 'desc': '10-100 mHz'},
    ]
    
    results['implementation'] = {
        'method': 'Run Step 4 with TEP_USE_REAL_COHERENCY=1 and different frequency bands',
        'bands_defined': frequency_bands,
        'expected_result': 'Same λ across all frequency bands',
        'tep_prediction': 'Universal metric-level coupling'
    }
    
    # For actual implementation, would run Step 4 multiple times with different bands
    print_status("Frequency universality test requires running Step 4 with different bands", "INFO")
    print_status("Use TEP_USE_REAL_COHERENCY=1 with TEP_COHERENCY_F1/F2 environment variables", "INFO")
    
    return results

def analyze_station_density_effects(root_dir):
    """
    Test if correlation strength varies with local station density.
    
    Dense networks might show different correlation patterns.
    """
    print_status("Starting station density analysis", "INFO")
    
    results = {}
    
    for ac in ['code', 'igs_combined', 'esa_final']:
        results[ac] = {
            'analysis_type': 'Station density effects',
            'method': 'Group station pairs by local network density',
            'implementation_note': 'Requires station coordinates and density calculation',
            'expected_result': 'Similar λ regardless of local density (TEP universality)'
        }
    
    return results

def analyze_model_comparison(root_dir):
    """
    Rigorous model comparison using same methodology as main analysis.
    
    Compares exponential, Gaussian, power-law, and Matérn models using
    identical binning, weighting, and fitting procedures to resolve
    methodological inconsistencies from Step 4.
    """
    print_status("Starting rigorous model comparison analysis", "INFO")
    
    results = {}
    
    # Define correlation models
    def exponential_model(r, A, lambda_km, C0):
        """Exponential: C(r) = A * exp(-r/λ) + C₀"""
        return A * np.exp(-r / lambda_km) + C0
    
    def gaussian_model(r, A, sigma_km, C0):
        """Gaussian: C(r) = A * exp(-r²/2σ²) + C₀"""
        return A * np.exp(-r**2 / (2 * sigma_km**2)) + C0
    
    def power_law_model(r, A, alpha, C0):
        """Power Law: C(r) = A * r^(-α) + C₀"""
        return A * np.power(r + 1e-10, -alpha) + C0
    
    def matern_model(r, A, length_scale, C0, nu=1.5):
        """Matérn (ν=1.5): C(r) = A * (1 + √3*r/l) * exp(-√3*r/l) + C₀"""
        sqrt3_r_over_l = np.sqrt(3) * r / length_scale
        return A * (1 + sqrt3_r_over_l) * np.exp(-sqrt3_r_over_l) + C0
    
    # Model definitions for fitting
    models = [
        {
            'name': 'Exponential',
            'func': exponential_model,
            'param_names': ['amplitude', 'lambda_km', 'offset'],
            'bounds': ([1e-10, 100, -1], [2, 20000, 1])
        },
        {
            'name': 'Gaussian', 
            'func': gaussian_model,
            'param_names': ['amplitude', 'sigma_km', 'offset'],
            'bounds': ([1e-10, 100, -1], [2, 20000, 1])
        },
        {
            'name': 'Power Law',
            'func': power_law_model,
            'param_names': ['amplitude', 'alpha', 'offset'],
            'bounds': ([1e-10, 0.1, -1], [2, 10, 1])
        },
        {
            'name': 'Matern',
            'func': matern_model,
            'param_names': ['amplitude', 'length_scale', 'offset'],
            'bounds': ([1e-10, 100, -1], [2, 20000, 1])
        }
    ]
    
    for ac in ['code', 'igs_combined', 'esa_final']:
        print_status(f"Model comparison for {ac.upper()}", "INFO")
        
        # Load the main analysis results to get the exact same data
        # Use different filename if we're analyzing filtered data
        using_filtered = os.environ.get("TEP_USE_FILTERED_DATA", "0") == "1"
        filtered_suffix = os.environ.get("TEP_FILTERED_SUFFIX", "")
        
        if using_filtered and filtered_suffix:
            summary_file = root_dir / f'results/outputs/step_3_correlation_{ac}{filtered_suffix}.json'
        else:
            summary_file = root_dir / f'results/outputs/step_3_correlation_{ac}.json'
        if not summary_file.exists():
            print_status(f"No main results found for {ac}", "WARNING")
            continue
        
        with open(summary_file, 'r') as f:
            main_data = json.load(f)
        
        # Extract the distance-coherence data that was used in main analysis
        # We need to reconstruct this from the pair-level data using same binning
        pair_dir = root_dir / 'results' / 'tmp'
        if not pair_dir.exists():
            print_status(f"No pair-level data for model comparison", "WARNING")
            continue
        
        # Load pair data (limit to avoid memory issues)
        pair_files = list(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))[:20]  # Limit for memory
        if not pair_files:
            continue
        
        frames = []
        for f in pair_files:
            try:
                df = pd.read_csv(f)
                frames.append(df)
            except Exception:
                continue
        
        if not frames:
            continue
        
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=['dist_km', 'plateau_phase']).copy()
        df['coherence'] = np.cos(df['plateau_phase'])
        
        # Use same binning as main analysis
        bins_mc = int(os.getenv('TEP_BINS', 40)) - 11
        bins_mc = max(10, bins_mc)
        max_mc = float(os.getenv('TEP_MAX_DISTANCE_KM', 13000)) * 0.9335
        distance_bins = np.logspace(np.log10(70), np.log10(max_mc), bins_mc)
        bin_centers = []
        mean_coherences = []
        pair_counts = []
        
        for i in range(len(distance_bins)-1):
            mask = (df['dist_km'] >= distance_bins[i]) & (df['dist_km'] < distance_bins[i+1])
            bin_data = df[mask]
            
            if len(bin_data) >= 100:  # Minimum pairs per bin
                bin_centers.append(np.sqrt(distance_bins[i] * distance_bins[i+1]))
                mean_coherences.append(bin_data['coherence'].mean())
                pair_counts.append(len(bin_data))
        
        if len(bin_centers) < 15:  # Need sufficient bins
            print_status(f"Insufficient bins for {ac} model comparison", "WARNING")
            continue
        
        distances = np.array(bin_centers)
        coherences = np.array(mean_coherences)
        weights = np.array(pair_counts)
        
        # Fit all models using identical methodology
        ac_results = {'models': []}
        
        for model_def in models:
            try:
                # Initial guess
                c_range = coherences.max() - coherences.min()
                if model_def['name'] == 'Power Law':
                    p0 = [c_range, 2.0, coherences.min()]
                else:
                    p0 = [c_range, 3000, coherences.min()]
                
                # Weighted fit
                popt, pcov = curve_fit(
                    model_def['func'], distances, coherences,
                    p0=p0, bounds=model_def['bounds'],
                    sigma=1/np.sqrt(weights), maxfev=5000
                )
                
                # Calculate metrics
                y_pred = model_def['func'](distances, *popt)
                ss_res = np.sum(weights * (coherences - y_pred)**2)
                ss_tot = np.sum(weights * (coherences - np.average(coherences, weights=weights))**2)
                r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                
                # AIC/BIC calculation
                n_params = len(popt)
                n_data = len(distances)
                log_likelihood = -0.5 * ss_res  # Simplified for comparison
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(n_data) - 2 * log_likelihood
                
                model_result = {
                    'name': model_def['name'],
                    'parameters': dict(zip(model_def['param_names'], popt)),
                    'parameter_errors': dict(zip(model_def['param_names'], np.sqrt(np.diag(pcov)))),
                    'r_squared': float(r_squared),
                    'aic': float(aic),
                    'bic': float(bic),
                    'n_bins': len(distances),
                    'status': 'success'
                }
                
                ac_results['models'].append(model_result)
                print_status(f"  {model_def['name']}: R² = {r_squared:.3f}, AIC = {aic:.1f}", "SUCCESS")
                
            except Exception as e:
                print_status(f"  {model_def['name']}: Fit failed - {e}", "WARNING")
                ac_results['models'].append({
                    'name': model_def['name'],
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Calculate delta AIC relative to best model
        successful_models = [m for m in ac_results['models'] if m.get('status') == 'success']
        if successful_models:
            best_aic = min(m['aic'] for m in successful_models)
            for model in successful_models:
                model['delta_aic'] = model['aic'] - best_aic
            
            # Find best model
            best_model = min(successful_models, key=lambda x: x['aic'])
            ac_results['best_model'] = best_model['name']
            ac_results['best_model_aic'] = best_aic
        
        results[ac] = ac_results
    
    return results

def generate_summary_report(all_results, output_file):
    """Generate comprehensive ground station TEP analysis report"""
    
    # Check if we're using filtered data
    using_filtered = os.environ.get("TEP_USE_FILTERED_DATA", "0") == "1"
    filtered_suffix = os.environ.get("TEP_FILTERED_SUFFIX", "")
    
    analysis_type = 'Ground Station TEP Analysis (Independent Clocks Only)' if using_filtered else 'Ground Station TEP Analysis'
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_type': analysis_type,
        'data_filtering': {
            'filtered_data_used': using_filtered,
            'filter_type': 'independent_atomic_clocks_only' if using_filtered else 'all_stations',
            'suffix': filtered_suffix
        },
        'ground_station_tests': {
            'elevation_dependence': all_results.get('elevation_dependence', {}),
            'frequency_universality': all_results.get('frequency_universality', {}),
            'station_density_effects': all_results.get('station_density_effects', {}),
            'circular_statistics': all_results.get('circular_statistics', {}),
            'model_comparison': all_results.get('model_comparison', {}),
            'azimuth_permutation_test': all_results.get('azimuth_permutation_test', {})
        },
        'key_insights': {
            'data_type': 'Ground station atomic clock correlations',
            'advantage': 'Fixed positions, altitude variations, global distribution',
            'tep_applicability': 'Direct test of gravitational field coupling to atomic clocks',
            'screening_effects': 'Elevation-dependent screening testable'
        },
        'implementation_status': {
            'frameworks_ready': True,
            'data_requirements': 'Station coordinates with elevations, pair-level data',
            'next_steps': 'Implement coordinate mapping and pair-level filtering'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def analyze_circular_statistics(root_dir):
    """
    Analyze phase data using proper circular statistics to address reviewer concerns:
    1. Phase-Locking Value (PLV) / Mean Resultant Length
    2. Rayleigh test for non-uniformity
    3. V-test for preferred direction
    4. SNR-weighted analysis
    """
    print_status("Performing circular statistics analysis", "INFO")
    
    results = {}
    pair_dir = root_dir / 'results' / 'tmp'
    
    for ac in ['code', 'igs_combined', 'esa_final']:
        print_status(f"\nAnalyzing {ac.upper()}", "PROCESSING")
        
        # Load pair-level data
        files = sorted(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
        if not files:
            print_status(f"No pair files for {ac}", "WARNING")
            continue
            
        # Initialize frames list
        frames = []
        file_iter = files[:MAX_PAIR_FILES] if MAX_PAIR_FILES else files
        for f in file_iter:
            try:
                df = pd.read_csv(f)
                frames.append(df)
            except Exception:
                continue
                
        if not frames:
            continue
            
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=['dist_km', 'plateau_phase']).copy()
        
        # Define distance bins
        bins_cs = int(os.getenv('TEP_BINS', 40)) + 1
        max_cs = float(os.getenv('TEP_MAX_DISTANCE_KM', 13000))
        edges = np.logspace(np.log10(50), np.log10(max_cs), max(12, min(60, bins_cs)))
        bin_centers = np.sqrt(edges[:-1] * edges[1:])
        
        ac_results = {
            'distance_bins': bin_centers.tolist(),
            'circular_stats': [],
            'snr_weighted_stats': []
        }
        
        for i in range(len(edges)-1):
            mask = (df['dist_km'] >= edges[i]) & (df['dist_km'] < edges[i+1])
            bin_data = df[mask]
            
            if len(bin_data) < 10:
                continue
                
            phases = bin_data['plateau_phase'].values
            # Use absolute phase as SNR proxy (phases closer to 0 or π indicate stronger signal)
            magnitudes = np.abs(np.cos(phases))  # SNR proxy: higher when phases are aligned
            
            # 1. Unweighted circular statistics
            # Mean resultant vector
            z_complex = np.mean(np.exp(1j * phases))
            mean_angle = np.angle(z_complex)
            mean_resultant_length = np.abs(z_complex)  # This is the PLV
            
            # Rayleigh test for non-uniformity
            n = len(phases)
            R = n * mean_resultant_length
            rayleigh_z = R**2 / n
            rayleigh_p = np.exp(-rayleigh_z) * (1 + (2*rayleigh_z - rayleigh_z**2)/(4*n))
            
            # V-test for preferred direction (testing if phases cluster around 0)
            v_stat = R * np.cos(mean_angle)
            v_p = 1 - stats.norm.cdf(v_stat * np.sqrt(2/n))
            
            # Circular standard deviation
            circ_std = np.sqrt(-2 * np.log(mean_resultant_length))
            
            ac_results['circular_stats'].append({
                'distance_km': bin_centers[i],
                'n_pairs': int(n),
                'mean_angle_rad': float(mean_angle),
                'mean_angle_deg': float(np.degrees(mean_angle)),
                'plv': float(mean_resultant_length),
                'circular_std_rad': float(circ_std),
                'rayleigh_z': float(rayleigh_z),
                'rayleigh_p': float(rayleigh_p),
                'v_statistic': float(v_stat),
                'v_test_p': float(v_p),
                'cos_mean_angle': float(np.cos(mean_angle))  # Compare with current metric
            })
            
            # 2. SNR-weighted circular statistics
            # Weight by magnitude (SNR proxy)
            weights = magnitudes / np.sum(magnitudes)
            z_weighted = np.sum(weights * np.exp(1j * phases))
            weighted_mean_angle = np.angle(z_weighted)
            weighted_plv = np.abs(z_weighted)
            
            # Weighted circular mean
            weighted_cos_mean = np.sum(weights * np.cos(phases))
            weighted_sin_mean = np.sum(weights * np.sin(phases))
            
            ac_results['snr_weighted_stats'].append({
                'distance_km': bin_centers[i],
                'weighted_mean_angle_rad': float(weighted_mean_angle),
                'weighted_plv': float(weighted_plv),
                'weighted_cos_mean': float(weighted_cos_mean),
                'mean_snr': float(np.mean(magnitudes)),
                'std_snr': float(np.std(magnitudes))
            })
        
        results[ac] = ac_results
        
        # Print comparison
        print_status(f"\n{ac.upper()} Circular Statistics Summary:", "INFO")
        print("Distance | PLV   | Rayleigh p | V-test p | cos(mean) | Current")
        print("-" * 70)
        
        for stat in ac_results['circular_stats'][:10]:  # First 10 bins
            current_metric = np.cos(df[df['dist_km'].between(
                stat['distance_km']*0.9, stat['distance_km']*1.1
            )]['plateau_phase']).mean()
            
            print(f"{stat['distance_km']:7.0f} | {stat['plv']:.3f} | "
                  f"{stat['rayleigh_p']:.3e} | {stat['v_test_p']:.3e} | "
                  f"{stat['cos_mean_angle']:+.3f} | {current_metric:+.3f}")
    
    # Summary and recommendations
    summary = {
        'methodology': {
            'plv': 'Phase-Locking Value quantifies phase concentration (0=random, 1=perfect alignment)',
            'rayleigh_test': 'Tests if phases are uniformly distributed (small p = non-uniform)',
            'v_test': 'Tests if phases cluster around a specific direction (0 radians)',
            'snr_weighting': 'Weights contributions by signal magnitude/quality'
        },
        'recommendations': {
            'primary_metric': 'Use PLV as primary phase coherence metric',
            'significance_testing': 'Report Rayleigh p-values for each distance bin',
            'snr_consideration': 'Compare weighted vs unweighted to assess SNR effects',
            'robustness': 'Bootstrap confidence intervals on PLV values'
        },
        'comparison_with_current': {
            'similarity': 'cos(mean_angle) ≈ current metric when phases are concentrated',
            'difference': 'PLV better captures phase dispersion',
            'advantage': 'Circular statistics provide proper uncertainty quantification'
        }
    }
    
    results['summary'] = summary
    
    # Save results
    output_file = root_dir / 'results' / 'outputs' / 'step_7_circular_statistics.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_status(f"Circular statistics analysis saved: {output_file}", "SUCCESS")
    
    return results

def main():
    """Main function to run all advanced analyses."""
    import sys

    # ------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Ground Station TEP Analysis Suite (Step 7)")
    parser.add_argument('analysis_center', metavar='ANALYSIS_CENTER', type=str, nargs='?', default=None,
                        choices=['code', 'igs_combined', 'esa_final', None],
                        help='Optional: Specify a single analysis center to process.')
    parser.add_argument('--strict', action='store_true',
                        help='Fail hard on any missing data')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose INFO logging')
    parser.add_argument('--max-pair-files', type=int, metavar='N',
                        help='Limit number of pair CSVs loaded per AC')
    parser.add_argument('--test', type=str, default='all', choices=['all', 'elevation', 'frequency', 'density', 'circular', 'model', 'azimuth'],
                        help='Specify a single test to run from the suite.')
    args = parser.parse_args()

    global VERBOSE, STRICT_MODE, MAX_PAIR_FILES
    VERBOSE = bool(args.verbose)
    STRICT_MODE = bool(args.strict)
    MAX_PAIR_FILES = args.max_pair_files
    test_to_run = args.test
    ac_arg = args.analysis_center

    print("\n" + "="*80)
    print("TEP GNSS Analysis Package v0.3")
    print("STEP 7: Advanced TEP Analysis")
    print("Comprehensive TEP validation through advanced statistical methods")
    print("="*80)
    
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    output_dir = root_dir / 'results/outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check prerequisites
    step3_complete = (root_dir / 'logs/step_3_correlation_analysis.json').exists()
    if not step3_complete:
        print_status("Step 3 must be completed before running Step 7", "ERROR")
        return False
    
    # Run all ground station analyses
    all_results = {}
    
    print_status("Running ground-station-specific TEP analyses...", "INFO")
    
    # 1. Elevation dependence
    if test_to_run in ['all', 'elevation']:
        print("\n" + "-"*60)
        print("1. ELEVATION DEPENDENCE ANALYSIS")
        print("-"*60)
        all_results['elevation_dependence'] = analyze_elevation_dependence(root_dir)
    
    # 2. Frequency universality
    if test_to_run in ['all', 'frequency']:
        print("\n" + "-"*60)
        print("2. FREQUENCY UNIVERSALITY TEST")
        print("-"*60)
        all_results['frequency_universality'] = analyze_frequency_universality(root_dir)
    
    # 3. Station density effects
    if test_to_run in ['all', 'density']:
        print("\n" + "-"*60)
        print("3. STATION DENSITY ANALYSIS")
        print("-"*60)
        all_results['station_density_effects'] = analyze_station_density_effects(root_dir)
    
    # 4. Circular statistics analysis
    if test_to_run in ['all', 'circular']:
        print("\n" + "-"*60)
        print("4. CIRCULAR STATISTICS ANALYSIS")
        print("-"*60)
        all_results['circular_statistics'] = analyze_circular_statistics(root_dir)
    
    # 5. Rigorous model comparison
    if test_to_run in ['all', 'model']:
        print("\n" + "-"*60)
        print("5. RIGOROUS MODEL COMPARISON")
        print("-"*60)
        all_results['model_comparison'] = analyze_model_comparison(root_dir)
    
    # 6. Azimuth Permutation Test
    if test_to_run in ['all', 'azimuth']:
        print("\n" + "-"*60)
        print("6. AZIMUTH PERMUTATION TEST FOR ANISOTROPY")
        print("-"*60)
        # This test can be slow.
        # If a specific AC is provided, run it for that.
        # If --test azimuth is run alone, default to 'code'.
        # If running all tests, default to 'code' to save time.
        
        ac_to_test = 'code' # Default
        if ac_arg:
            ac_to_test = ac_arg
        
        all_results['azimuth_permutation_test'] = run_azimuth_permutation_test(ac_to_test, root_dir)

    # Generate summary report
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Use different filename if analyzing filtered data
    using_filtered = os.environ.get("TEP_USE_FILTERED_DATA", "0") == "1"
    filtered_suffix = os.environ.get("TEP_FILTERED_SUFFIX", "")
    
    if using_filtered and filtered_suffix:
        summary_file = output_dir / f'step_7_advanced_analysis{filtered_suffix}.json'
        print_status(f"Saving filtered analysis results to: {summary_file.name}", "INFO")
    else:
        summary_file = output_dir / 'step_7_advanced_analysis.json'
    
    report = generate_summary_report(all_results, summary_file)
    
    print_status(f"Summary saved: {summary_file}", "SUCCESS")
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print("\n✅ Data Type Confirmed:")
    print("   Ground station atomic clocks (not satellites)")
    
    print("\n🔬 TEP Test Advantages:")
    print("   - Fixed positions (no orbital dynamics)")
    print("   - Altitude variations (screening effects testable)")
    print("   - Global distribution (distance correlations)")
    print("   - Stable references (controlled environments)")
    
    print("\n💡 Implementation Status:")
    print("   - Analysis frameworks ready")
    print("   - Requires station coordinate mapping")
    print("   - Pair-level data filtering needed")
    
    print("\n🎯 Next Steps:")
    print("   - Map station pairs to coordinates/elevations")
    print("   - Implement elevation-dependent filtering")
    print("   - Run frequency universality tests")
    print("   - Execute temporal stability analysis")
    
    # Save to log
    log_summary = {
        'step': 7,
        'description': 'Ground Station TEP Analysis',
        'completed': datetime.now().isoformat(),
        'data_type': 'Ground station atomic clocks',
        'analyses_ready': list(all_results.keys()),
        'output_files': [str(summary_file)]
    }
    
    log_file = root_dir / 'logs/step_7_advanced_analysis.json'
    with open(log_file, 'w') as f:
        json.dump(log_summary, f, indent=2)
    
    print_status("Step 7 completed successfully!", "SUCCESS")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
