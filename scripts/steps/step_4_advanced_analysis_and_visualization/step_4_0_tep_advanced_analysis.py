#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 4.0: Advanced Analysis
=============================================

This streamlined script focuses on the most valuable, non-redundant advanced analyses.
Analyses already performed in other steps (e.g., anisotropy in Step 5) have been removed.

Valuable Analyses Included:
- Elevation dependence analysis (with corrected coordinate mapping)
- Circular statistics analysis (a unique statistical approach)
- Rigorous model comparison (for statistical validation)

Requirements: Step 2.1 complete (Geospatial Data Processing)
Inputs:
  - data/coordinates/step_1_1_station_coords_global.csv (from Step 1.1)
  - data/processed/step_2_1_geospatial_{ac}.csv (from Step 2.1)
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0)
Outputs:
  - results/outputs/step_4_0_advanced_analysis.json (consolidated results)
  - results/outputs/step_4_0_circular_statistics_streamlined.json
Next: Step 4.1 (Visualization)

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
# Removed matplotlib.pyplot as plt and argparse as they are not used and are often problematic in non-interactive scripts.

# Anchor to package root
PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# Import TEP utilities for better configuration and error handling
sys.path.insert(0, str(PACKAGE_ROOT))
from scripts.utils.config import TEPConfig
from scripts.utils.pid_manager import ensure_single_instance
from scripts.utils.logger import print_status, TEPLogger, set_step_logger

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_4_0_tep_advanced_analysis",
    level="DEBUG",
    log_file_path=Path(__file__).resolve().parents[3] / "logs" / "step_4_0_tep_advanced_analysis.log"
)

# Register step logger so print_status uses it
set_step_logger(step_logger)
from scripts.utils.exceptions import (
    TEPAnalysisError, TEPDataError, TEPFileError, safe_csv_read, safe_json_write
)

# Removed redundant initialization

def assert_condition(condition: bool, message: str):
    """Assert condition or raise TEPDataError."""
    if not condition:
        raise TEPDataError(message)

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model: C(r) = A * exp(-r/λ) + C0"""
    return A * np.exp(-r / lambda_km) + C0

def fit_exponential(distances, coherences, weights=None, p0=None,
                    bounds=([0.01, 100, -1], [2, 20000, 1]), maxfev=5000):
    """Fit exponential_model to data and return params, errors, R²."""
    distances = np.asarray(distances)
    coherences = np.asarray(coherences)
    if len(distances) < TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT', 3):
        raise TEPDataError(f"Insufficient data for exponential fit: {len(distances)} bins (min: {TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT', 3)})")

    if p0 is None:
        c_range = coherences.max() - coherences.min()
        p0 = [c_range, TEPConfig.get_int('TEP_CORRELATION_LENGTH_INITIAL_GUESS', 1000), coherences.min()]

    sigma = None
    if weights is not None:
        sigma = 1.0 / np.sqrt(np.asarray(weights))

    try:
        popt, pcov = curve_fit(exponential_model, distances, coherences,
                               p0=p0, sigma=sigma, bounds=bounds, maxfev=maxfev)
    except RuntimeError as e:
        raise TEPAnalysisError(f"Curve fitting failed: {e}")

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

def load_station_coordinates():
    """Load ground station coordinates with elevation data"""
    coords_file = PACKAGE_ROOT / 'data/coordinates/step_1_1_station_coords_global.csv'
    
    assert_condition(coords_file.exists(),
                     "Station coordinates file not found – ensure Step 1.1 data acquisition and Step 1.2 coordinate validation were successful")
        
    try:
        df = safe_csv_read(coords_file)
        print_status(f"Loaded coordinates for {len(df)} ground stations", "SUCCESS")
        return df
    except Exception as e:
        print_status(f"Failed to load station coordinates: {e}", "ERROR")
        assert_condition(False, f"Failed to load station coordinates: {e}")

def xyz_to_lla(x, y, z):
    """Convert ECEF XYZ to Latitude/Longitude/Altitude using WGS84"""
    # WGS84 constants
    a = 6378137.0          # Semi-major axis (m)
    f = 1 / 298.257223563  # Flattening
    e2 = 2 * f - f**2      # First eccentricity squared
    
    # Convert to numpy arrays
    x, y, z = map(np.asarray, [x, y, z])
    
    # Longitude
    lon = np.arctan2(y, x)
    
    # Latitude (iterative)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    # Iterate for better precision
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    
    # Height
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    height = p / np.cos(lat) - N
    
    return np.degrees(lat), np.degrees(lon), height

def analyze_elevation_dependence_fixed(root_dir):
    """
    FIXED elevation dependence analysis with improved coordinate mapping.
    Addresses the coordinate mapping issues for IGS_COMBINED and ESA_FINAL.
    """
    print_status("Starting FIXED elevation dependence analysis", "INFO")
    
    # Load station coordinates
    coords_df = load_station_coordinates()
    
    # Use existing height_m column or convert XYZ to elevation
    if 'height_m' in coords_df.columns:
        print_status("Using existing height_m column as elevation", "INFO")
        coords_df['elevation_m'] = coords_df['height_m']
    elif all(col in coords_df.columns for col in ['X', 'Y', 'Z']):
        print_status("Converting XYZ coordinates to elevation", "INFO")
        lats, lons, elevs = xyz_to_lla(coords_df['X'], coords_df['Y'], coords_df['Z'])
        coords_df['elevation_m'] = elevs
        print_status("XYZ to elevation conversion complete", "SUCCESS")
    else:
        print_status("No elevation data found in coordinate file", "ERROR")
        return {}
    
    # Create elevation and geomagnetic lookup with multiple station code formats
    station_lookup = {}
    
    for _, row in coords_df.iterrows():
        if pd.isna(row.get('elevation_m')):
            continue
            
        station_code = str(row['code']).strip().upper()
        elev = float(row['elevation_m'])
        geomag_lat = row.get('geomag_lat', None)
        
        station_data = {
            'elevation_m': elev,
            'geomag_lat': geomag_lat,
            'lat_deg': row.get('lat_deg', None),
            'lon_deg': row.get('lon_deg', None)
        }
        
        # Add multiple formats to lookup
        station_lookup[station_code] = station_data
        
        # Add short codes (first 4 chars)
        if len(station_code) >= 4:
            station_lookup[station_code[:4]] = station_data
        
        # Add without numbers/suffixes
        import re
        clean_code = re.sub(r'[0-9]+.*$', '', station_code)
        if clean_code and clean_code != station_code:
            station_lookup[clean_code] = station_data
    
    # Count stations with geomagnetic data
    stations_with_geomag = sum(1 for data in station_lookup.values() if data['geomag_lat'] is not None)
    print_status(f"Created station lookup with {len(station_lookup)} entries", "SUCCESS")
    print_status(f"Stations with geomagnetic coordinates: {stations_with_geomag}", "INFO")
    
    results = {}
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in analysis_centers:
        print_status(f"Processing elevation analysis for {ac.upper()}", "INFO")
        
        # Use processed data from Step 2.1 (includes azimuth, quality filtering, and geospatial enhancements)
        geospatial_file = root_dir / 'data' / 'processed' / f'step_2_1_geospatial_{ac}.csv'
        
        if geospatial_file.exists():
            print_status(f"Using Step 2.1 processed data: {geospatial_file.name}", "INFO")
            try:
                df_all = pd.read_csv(geospatial_file, parse_dates=['date'])
                # Ensure coherence column exists
                if 'plateau_phase' in df_all.columns and 'coherence' not in df_all.columns:
                    df_all['coherence'] = np.cos(df_all['plateau_phase'])
                print_status(f"Loaded Step 2.1 processed dataset: {len(df_all):,} pairs for {ac}", "SUCCESS")
            except Exception as e:
                print_status(f"Failed to load Step 2.1 processed data: {e}", "WARNING")
                continue
        else:
            print_status(f"Step 2.1 processed file not found for {ac}: {geospatial_file}", "ERROR")
            continue
        print_status(f"Loaded {len(df_all)} station pairs for {ac.upper()}", "SUCCESS")
        
        # FIXED: Better station code mapping with geomagnetic data
        def extract_short_code(full_code):
            if pd.isna(full_code):
                return None
            
            full_str = str(full_code).strip().upper()
            
            # Try direct lookup first
            if full_str in station_lookup:
                return full_str
            
            # Try various patterns
            patterns_to_try = [
                full_str[:4],  # First 4 characters
                re.sub(r'[0-9]+.*$', '', full_str),  # Remove numbers/suffixes
                full_str[:-3] if len(full_str) > 3 else None,  # Remove last 3 chars
                full_str[:3] if len(full_str) >= 3 else None,  # First 3 characters
            ]
            
            for pattern in patterns_to_try:
                if pattern and pattern in station_lookup:
                    return pattern
            
            return None
        
        df_all['short_i'] = df_all['station_i'].apply(extract_short_code)
        df_all['short_j'] = df_all['station_j'].apply(extract_short_code)
        df_all['elev_i'] = df_all['short_i'].map(lambda x: station_lookup.get(x, {}).get('elevation_m') if x else None)
        df_all['elev_j'] = df_all['short_j'].map(lambda x: station_lookup.get(x, {}).get('elevation_m') if x else None)
        df_all['geomag_lat_i'] = df_all['short_i'].map(lambda x: station_lookup.get(x, {}).get('geomag_lat') if x else None)
        df_all['geomag_lat_j'] = df_all['short_j'].map(lambda x: station_lookup.get(x, {}).get('geomag_lat') if x else None)
        
        # Filter pairs where both stations have elevation data
        df_valid = df_all.dropna(subset=['elev_i', 'elev_j']).copy()
        if len(df_valid) > 0:
            print_status(f"Found {len(df_valid)} pairs with elevation data for {ac.upper()}", "SUCCESS")
        else:
            print_status(f"Found {len(df_valid)} pairs with elevation data for {ac.upper()}", "WARNING")
   
        if len(df_valid) == 0:
            results[ac] = {'error': 'No pairs with elevation data after coordinate mapping fix'}
            continue
        
        # Compute coherence and elevation metrics - handle different column names
        if 'phase' in df_valid.columns:
            df_valid['coherence'] = np.cos(df_valid['phase'])
        elif 'plateau_phase' in df_valid.columns:
            df_valid['coherence'] = np.cos(df_valid['plateau_phase'])
        else:
            print_status(f"No phase column found for {ac}", "ERROR")
            results[ac] = {'error': 'No phase column found'}
            continue
        df_valid['elev_diff_m'] = np.abs(df_valid['elev_j'] - df_valid['elev_i'])
        df_valid['mean_elev_m'] = (df_valid['elev_i'] + df_valid['elev_j']) / 2
        
        # Compute geomagnetic metrics where available
        df_valid['geomag_diff'] = np.abs(df_valid['geomag_lat_j'] - df_valid['geomag_lat_i'])
        df_valid['mean_geomag_lat'] = (df_valid['geomag_lat_i'] + df_valid['geomag_lat_j']) / 2
        
        # Elevation quintile analysis
        quintile_results = {}
        quintiles = np.percentile(df_valid['mean_elev_m'], [0, 20, 40, 60, 80, 100])
        
        for i in range(5):
            mask = (df_valid['mean_elev_m'] >= quintiles[i]) & (df_valid['mean_elev_m'] < quintiles[i+1])
            if i == 4:  # Include the maximum in the last quintile
                mask = (df_valid['mean_elev_m'] >= quintiles[i])
            
            subset = df_valid[mask].copy()
            if len(subset) < TEPConfig.get_int('TEP_MIN_BIN_COUNT', 10):  # Need minimum data
                continue
            
            # Bin analysis
            edges = np.logspace(np.log10(TEPConfig.get_float('TEP_MIN_DISTANCE_FOR_FIT', 100)), np.log10(TEPConfig.get_float('TEP_MAX_DISTANCE_FOR_FIT', 10000)), TEPConfig.get_int('TEP_NUM_BINS_FOR_FIT', 20))
            subset = subset.copy()  # Avoid SettingWithCopyWarning
            subset['dist_bin'] = pd.cut(subset['dist_km'], bins=edges, right=False)
            binned = subset.groupby('dist_bin', observed=True).agg(
                mean_dist=('dist_km', 'mean'),
                mean_coh=('coherence', 'mean'),
                count=('coherence', 'size')
            ).reset_index()
            
            binned = binned[binned['count'] >= TEPConfig.get_int('TEP_MIN_PAIRS_PER_BIN', 5)].dropna()
            if len(binned) < TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT', 3):
                continue
            
            # Fit exponential model
            try:
                popt, perr, r_squared = fit_exponential(
                    binned['mean_dist'].values,
                    binned['mean_coh'].values,
                    binned['count'].values
                )
                
                quintile_results[f"quintile_{i+1}"] = {
                    'elevation_range_m': [float(quintiles[i]), float(quintiles[i+1])],
                    'lambda_km': float(popt[1]),
                    'lambda_error_km': float(perr[1]),
                    'amplitude': float(popt[0]),
                    'offset': float(popt[2]),
                    'r_squared': float(r_squared),
                    'n_pairs': len(subset),
                    'n_bins': len(binned)
                }
                
                print_status(f"  Quintile {i+1} ({quintiles[i]:.0f}-{quintiles[i+1]:.0f}m): λ = {popt[1]:.0f} ± {perr[1]:.0f} km, R² = {r_squared:.3f}", "SUCCESS")
                
            except Exception as e:
                print_status(f"  Quintile {i+1} fit failed: {e}", "WARNING")
                continue
        
        # Geomagnetic-Elevation Stratified Analysis
        geomag_stratified_results = {}
        df_with_geomag = df_valid.dropna(subset=['geomag_lat_i', 'geomag_lat_j']).copy()
        
        if len(df_with_geomag) > 0:
            print_status(f"Performing geomagnetic-elevation stratified analysis with {len(df_with_geomag)} pairs", "INFO")
            
            # Create 2D stratification: elevation bins × geomagnetic latitude bins
            num_elev_bins = TEPConfig.get_int('TEP_NUM_ELEV_BINS_FOR_STRATIFIED', 3) # e.g., 3
            num_geomag_bins = TEPConfig.get_int('TEP_NUM_GEOMAG_BINS_FOR_STRATIFIED', 3) # e.g., 3
            elev_bins = np.percentile(df_with_geomag['mean_elev_m'], np.linspace(0, 100, num_elev_bins + 1))
            geomag_bins = np.percentile(df_with_geomag['mean_geomag_lat'], np.linspace(0, 100, num_geomag_bins + 1))
            
            for i in range(num_elev_bins):  # Elevation bins
                for j in range(num_geomag_bins):  # Geomagnetic bins
                    elev_mask = (df_with_geomag['mean_elev_m'] >= elev_bins[i]) & (df_with_geomag['mean_elev_m'] < elev_bins[i+1])
                    geomag_mask = (df_with_geomag['mean_geomag_lat'] >= geomag_bins[j]) & (df_with_geomag['mean_geomag_lat'] < geomag_bins[j+1])
                    
                    if i == num_elev_bins - 1:  # Include maximum in last elevation bin
                        elev_mask = (df_with_geomag['mean_elev_m'] >= elev_bins[i])
                    if j == num_geomag_bins - 1:  # Include maximum in last geomagnetic bin  
                        geomag_mask = (df_with_geomag['mean_geomag_lat'] >= geomag_bins[j])
                    
                    subset = df_with_geomag[elev_mask & geomag_mask].copy()
                    
                    if len(subset) < TEPConfig.get_int('TEP_MIN_BIN_COUNT', 10):  # Need minimum data
                        continue
                    
                    # Bin analysis for this stratum
                    edges = np.logspace(np.log10(TEPConfig.get_float('TEP_MIN_DISTANCE_FOR_FIT', 100)), np.log10(TEPConfig.get_float('TEP_MAX_DISTANCE_FOR_FIT', 10000)), TEPConfig.get_int('TEP_NUM_BINS_FOR_FIT', 20)) # Fewer bins for smaller samples
                    subset = subset.copy()  # Avoid SettingWithCopyWarning
                    subset['dist_bin'] = pd.cut(subset['dist_km'], bins=edges, right=False)
                    binned = subset.groupby('dist_bin', observed=True).agg(
                        mean_dist=('dist_km', 'mean'),
                        mean_coh=('coherence', 'mean'),
                        count=('coherence', 'size')
                    ).reset_index()
                    
                    binned = binned[binned['count'] >= TEPConfig.get_int('TEP_MIN_PAIRS_PER_BIN', 5)].dropna()
                    if len(binned) < TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT', 3):
                        continue
                    
                    # Fit exponential model
                    try:
                        popt, perr, r_squared = fit_exponential(
                            binned['mean_dist'].values,
                            binned['mean_coh'].values,
                            binned['count'].values
                        )
                        
                        stratum_key = f"elev_bin_{i+1}_geomag_bin_{j+1}"
                        geomag_stratified_results[stratum_key] = {
                            'elevation_range_m': [float(elev_bins[i]), float(elev_bins[i+1])],
                            'geomag_lat_range': [float(geomag_bins[j]), float(geomag_bins[j+1])],
                            'lambda_km': float(popt[1]),
                            'lambda_error_km': float(perr[1]),
                            'amplitude': float(popt[0]),
                            'offset': float(popt[2]),
                            'r_squared': float(r_squared),
                            'n_pairs': len(subset),
                            'n_bins': len(binned)
                        }
                        
                        print_status(f"  Stratum E{i+1}G{j+1} ({elev_bins[i]:.0f}-{elev_bins[i+1]:.0f}m, {geomag_bins[j]:.1f}-{geomag_bins[j+1]:.1f}°): λ = {popt[1]:.0f} ± {perr[1]:.0f} km", "SUCCESS")
                        
                    except Exception as e:
                        print_status(f"  Stratum E{i+1}G{j+1} fit failed: {e}", "WARNING")
                        continue
        
        results[ac] = {
            'total_pairs': len(df_all),
            'pairs_with_elevation': len(df_valid),
            'pairs_with_geomagnetic': len(df_with_geomag) if len(df_valid) > 0 else 0,
            'elevation_coverage_percent': 100 * len(df_valid) / len(df_all),
            'geomagnetic_coverage_percent': 100 * len(df_with_geomag) / len(df_all) if len(df_valid) > 0 else 0,
            'quintile_analysis': quintile_results,
            'geomagnetic_stratified_analysis': geomag_stratified_results,
            'elevation_range_m': [float(df_valid['mean_elev_m'].min()), float(df_valid['mean_elev_m'].max())],
            'geomagnetic_range': [float(df_with_geomag['mean_geomag_lat'].min()), float(df_with_geomag['mean_geomag_lat'].max())] if len(df_with_geomag) > 0 else None,
            'coordinate_mapping_fixed': True,
            'geomagnetic_enhancement': True
        }
    
    return results

def analyze_regional_jackknife_simplified(root_dir):
    """
    Simplified regional jackknife analysis with reduced memory usage.
    Only tests exclusion of one major region (Andes) to avoid memory issues.
    """
    print_status("Starting Simplified Regional Jackknife Analysis", "INFO")
    
    # Test only one major region to reduce memory usage
    region = {'lat_range': (-60, 15), 'lon_range': (-85, -65), 'description': 'Andes Mountains'}
    
    # Load station coordinates
    coords_df = load_station_coordinates()
    
    results = {}
    # Only process CODE to reduce memory usage
    analysis_centers = ['code']
    
    for ac in analysis_centers:
        print_status(f"Simplified regional jackknife analysis for {ac.upper()}", "INFO")
        
        # Use processed data from Step 2.1
        geospatial_file = root_dir / 'data' / 'processed' / f'step_2_1_geospatial_{ac}.csv'
        
        if not geospatial_file.exists():
            print_status(f"Step 2.1 processed file not found for {ac}: {geospatial_file}", "ERROR")
            results[ac] = {'error': 'Step 2.1 processed file not found'}
            continue
        
        try:
            # Load only a sample of the data to reduce memory usage
            df_all = pd.read_csv(geospatial_file, parse_dates=['date'], nrows=1000000)  # Limit to 1M rows
            if 'plateau_phase' in df_all.columns and 'coherence' not in df_all.columns:
                df_all['coherence'] = np.cos(df_all['plateau_phase'])
            print_status(f"Loaded simplified dataset: {len(df_all):,} pairs for {ac}", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to load Step 2.1 processed data: {e}", "WARNING")
            results[ac] = {'error': f'Failed to load data: {e}'}
            continue
        
        # Simple coordinate mapping
        station_coords = {}
        for _, row in coords_df.iterrows():
            code = str(row['code']).strip().upper()
            station_coords[code] = {
                'lat': row.get('lat_deg'),
                'lon': row.get('lon_deg'),
                'elevation': row.get('height_m')
            }
        
        def get_station_coords(station_code):
            code = str(station_code).strip().upper()
            return station_coords.get(code[:4], station_coords.get(code))
        
        # Add coordinates (simplified)
        df_all['coords_i'] = df_all['station_i'].apply(get_station_coords)
        df_all['coords_j'] = df_all['station_j'].apply(get_station_coords)
        
        # Filter and extract coordinates
        df_with_coords = df_all[df_all['coords_i'].notna() & df_all['coords_j'].notna()].copy()
        
        if len(df_with_coords) == 0:
            results[ac] = {'error': 'No pairs with valid coordinates'}
            continue
        
        df_with_coords['lat_i'] = df_with_coords['coords_i'].apply(lambda x: x['lat'] if x else None)
        df_with_coords['lon_i'] = df_with_coords['coords_i'].apply(lambda x: x['lon'] if x else None)
        df_with_coords['lat_j'] = df_with_coords['coords_j'].apply(lambda x: x['lat'] if x else None)
        df_with_coords['lon_j'] = df_with_coords['coords_j'].apply(lambda x: x['lon'] if x else None)
        df_with_coords['elev_i'] = df_with_coords['coords_i'].apply(lambda x: x['elevation'] if x else None)
        df_with_coords['elev_j'] = df_with_coords['coords_j'].apply(lambda x: x['elevation'] if x else None)
        
        # Filter valid elevation data
        df_valid = df_with_coords.dropna(subset=['lat_i', 'lon_i', 'lat_j', 'lon_j', 'elev_i', 'elev_j']).copy()
        
        if len(df_valid) == 0:
            results[ac] = {'error': 'No pairs with valid elevation data'}
            continue
        
        # Determine coherence column
        if 'phase' in df_valid.columns:
            df_valid['coherence'] = np.cos(df_valid['phase'])
        elif 'plateau_phase' in df_valid.columns:
            df_valid['coherence'] = np.cos(df_valid['plateau_phase'])
        else:
            results[ac] = {'error': 'No phase column found'}
            continue
        
        df_valid['mean_elev_m'] = (df_valid['elev_i'] + df_valid['elev_j']) / 2
        
        print_status(f"Base dataset: {len(df_valid)} pairs with coordinates and elevation", "INFO")
        
        # Test exclusion of Andes region
        print_status(f"  Testing exclusion of {region['description']}", "INFO")
        
        lat_range = region['lat_range']
        lon_range = region['lon_range']
        
        # Check if either station is in the region
        in_region_i = (
            (df_valid['lat_i'] >= lat_range[0]) & (df_valid['lat_i'] <= lat_range[1]) &
            (df_valid['lon_i'] >= lon_range[0]) & (df_valid['lon_i'] <= lon_range[1])
        )
        in_region_j = (
            (df_valid['lat_j'] >= lat_range[0]) & (df_valid['lat_j'] <= lat_range[1]) &
            (df_valid['lon_j'] >= lon_range[0]) & (df_valid['lon_j'] <= lon_range[1])
        )
        
        # Exclude pairs where either station is in the region
        df_excluded = df_valid[~(in_region_i | in_region_j)].copy()
        excluded_pairs = len(df_valid) - len(df_excluded)
        
        print_status(f"    Excluded {excluded_pairs} pairs ({100*excluded_pairs/len(df_valid):.1f}%)", "INFO")
        
        if len(df_excluded) < 100:  # Need sufficient data
            results[ac] = {'error': 'Insufficient data after exclusion', 'excluded_pairs': excluded_pairs}
            continue
        
        # Simple correlation analysis on excluded dataset
        try:
            # Just compute basic statistics
            mean_coherence_excluded = df_excluded['coherence'].mean()
            mean_coherence_full = df_valid['coherence'].mean()
            coherence_change = mean_coherence_excluded - mean_coherence_full
            
            results[ac] = {
                'total_pairs': len(df_all),
                'pairs_with_elevation': len(df_valid),
                'excluded_pairs': excluded_pairs,
                'exclusion_percent': 100 * excluded_pairs / len(df_valid),
                'remaining_pairs': len(df_excluded),
                'mean_coherence_full': float(mean_coherence_full),
                'mean_coherence_excluded': float(mean_coherence_excluded),
                'coherence_change': float(coherence_change),
                'region_tested': region['description'],
                'simplified': True
            }
            
            print_status(f"    Mean coherence change: {coherence_change:+.4f}", "SUCCESS")
            
        except Exception as e:
            results[ac] = {'error': f'Analysis failed: {str(e)}'}
            print_status(f"    Analysis failed: {e}", "WARNING")
    
    return results

def analyze_circular_statistics(root_dir):
    """
    Analyze phase data using proper circular statistics.
    """
    print_status("Performing circular statistics analysis", "INFO")
    
    all_results = {}
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in analysis_centers:
        print_status(f"\n--- Analyzing {ac.upper()} ---", "INFO")
        
        # Load pair-level data
        pair_dir = root_dir / 'results/tmp'
        pair_files = sorted(pair_dir.glob(f"step_2_0_pairs_{ac}_*.csv")) # Updated from step_3_pairs
        
        if not pair_files:
            continue
        
        # Load sample of data for circular statistics
        df_chunks = []
        file_limit = TEPConfig.get_file_limits()[ac]
        for pfile in pair_files[:file_limit]:  # Note: file_limit is configured via TEPConfig, not necessarily fixed at 5.
            try:
                chunk = safe_csv_read(pfile)
                df_chunks.append(chunk)
            except Exception as e:
                print_status(f"Failed to load {pfile}: {e}", "WARNING")
                continue
        
        if not df_chunks:
            continue
        
        df = pd.concat(df_chunks, ignore_index=True)
        
        # Distance binning
        edges = np.logspace(np.log10(TEPConfig.get_float('TEP_MIN_DISTANCE_FOR_CIRCULAR_STATS', 100)), np.log10(TEPConfig.get_float('TEP_MAX_DISTANCE_FOR_CIRCULAR_STATS', 10000)), TEPConfig.get_int('TEP_NUM_BINS_FOR_CIRCULAR_STATS', 10))  # 10 bins
        df['dist_bin'] = pd.cut(df['dist_km'], bins=edges, right=False)
        
        results = {}
        print_status("Distance | PLV   | Rayleigh p | V-test p | cos(mean) | Current", "INFO")
        print_status("-"*70, "INFO")
        
        for bin_label, group in df.groupby('dist_bin', observed=True):
            if len(group) < TEPConfig.DEFAULTS['TEP_MIN_BIN_COUNT']:
                continue
            
            # Handle different phase column names across centers
            if 'phase' in group.columns:
                phases = group['phase'].values
            elif 'plateau_phase' in group.columns:
                phases = group['plateau_phase'].values
            else:
                continue  # Skip if no phase column
            coherences = np.cos(phases)
            
            # Phase Locking Value (PLV)
            plv = np.abs(np.mean(np.exp(1j * phases)))
            
            # Rayleigh test for uniformity (corrected implementation)
            try:
                # Corrected Rayleigh test implementation
                n = len(phases)
                R = n * plv  # R statistic
                rayleigh_stat = 2 * R**2 / n  # Correct formula: 2*R²/n
                # Correct p-value approximation for large n
                if rayleigh_stat < 50:
                    rayleigh_p = np.exp(-rayleigh_stat * (1 - (2 + rayleigh_stat) / (4 * n)))
                else:
                    rayleigh_p = 0.0
            except:
                rayleigh_p = np.nan
            
            # V-test for preferred direction (0 radians)
            try:
                # Calculate V-statistic for a hypothesized mean direction of 0 radians
                # V = sum(cos(thet-i - mu_0)) where mu_0 = 0
                v_statistic = np.sum(np.cos(phases))
                n = len(phases)

                # For large n, the V-test statistic Z = V_statistic / sqrt(n/2) is approximately standard normal
                # We are testing if the mean direction is significantly different from 0 in the direction of 0.
                # This is typically a one-tailed test.
                if n > 1 and v_statistic >= 0: # V-test is directional, only meaningful if V_statistic is positive for target direction
                    Z = v_statistic / np.sqrt(n / 2)
                    v_p = stats.norm.sf(Z) * 2 # Using two-tailed p-value; one-tailed (stats.norm.sf(Z)) could be used if direction is strictly positive.
                else:
                    v_p = np.nan # Not enough data or V_statistic is negative for the direction of interest
            except Exception as e:
                print_status(f"    V-test calculation failed: {e}", "WARNING")
                v_p = np.nan
            
            # Circular statistics
            mean_direction = np.angle(np.mean(np.exp(1j * phases)))
            cos_mean = np.cos(mean_direction)
            current_coherence = np.mean(coherences)
            
            # Store results
            dist_center = group['dist_km'].mean()
            results[f"{dist_center:.0f}km"] = {
                'distance_km': dist_center,
                'plv': plv,
                'rayleigh_p': rayleigh_p,
                'v_test_p': v_p,
                'cos_mean_direction': cos_mean,
                'mean_coherence': current_coherence,
                'n_samples': len(group)
            }
            
            # Print formatted results
            print_status(f"{dist_center:8.0f} | {plv:.3f} | {rayleigh_p:10.3e} | {v_p:8.3e} | {cos_mean:+.3f} | {current_coherence:+.3f}", "INFO")
        
        all_results[ac] = results
    
    # Save results
    output_file = root_dir / 'results/outputs/step_4_0_circular_statistics_streamlined.json'
    safe_json_write(all_results, output_file, indent=2)
    
    print_status(f"Circular statistics analysis saved: {output_file}", "SUCCESS")
    return all_results

def analyze_model_comparison(root_dir):
    """Rigorous model comparison using multiple correlation models."""
    print_status("Starting rigorous model comparison analysis", "INFO")
    
    # Define models to compare
    def gaussian_model(r, A, sigma, C0):
        return A * np.exp(-0.5 * (r/sigma)**2) + C0
    
    def power_law_model(r, A, alpha, C0):
        return A * (r + 1)**(-alpha) + C0
    
    def matern_model(r, A, length_scale, C0):
        # Simplified Matérn with ν=1.5
        sqrt3_r = np.sqrt(3) * r / length_scale
        return A * (1 + sqrt3_r) * np.exp(-sqrt3_r) + C0
    
    models = {
        'Exponential': (exponential_model, ([0.01, 100, -1], [2, 20000, 1])),
        'Gaussian': (gaussian_model, ([0.01, 100, -1], [2, 20000, 1])),
        'Power Law': (power_law_model, ([0.01, 0.1, -1], [2, 5, 1])),
        'Matern': (matern_model, ([0.01, 100, -1], [2, 20000, 1]))
    }
    
    results = {}
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in analysis_centers:
        # Load correlation results from Step 2.0
        step_2_0_file = root_dir / f'results/outputs/step_2_0_correlation_{ac}.json' # Updated from step_3_correlation
        if not step_2_0_file.exists():
            continue
        
        with open(step_2_0_file, 'r') as f:
            step_2_0_data = json.load(f)
        
        if 'binned_correlations' not in step_2_0_data:
            continue
        
        binned = step_2_0_data['binned_correlations']
        distances = np.array([b['mean_distance_km'] for b in binned])
        coherences = np.array([b['mean_coherence'] for b in binned])
        weights = np.array([b['pair_count'] for b in binned])
        
        ac_results = {}
        
        for model_name, (model_func, bounds) in models.items():
            try:
                # Fit model
                sigma = 1.0 / np.sqrt(weights)
                popt, pcov = curve_fit(model_func, distances, coherences,
                                     sigma=sigma, bounds=bounds, maxfev=5000)
                
                # Calculate metrics
                y_pred = model_func(distances, *popt)
                ss_res = np.sum(weights * (coherences - y_pred)**2)
                ss_tot = np.sum(weights * (coherences - np.average(coherences, weights=weights))**2)
                r_squared = 1 - ss_res/ss_tot
                
                # AIC calculation (already correct - standardized formulation)
                n = len(distances)
                k = len(popt)
                aic = 2*k + n*np.log(ss_res/n)  # Standard AIC formulation
                
                # Log-likelihood for likelihood ratio tests
                # For weighted least squares with normal errors
                log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
                
                ac_results[model_name] = {
                    'r_squared': float(r_squared),
                    'aic': float(aic),
                    'parameters': popt.tolist(),
                    'parameter_errors': np.sqrt(np.diag(pcov)).tolist(),
                    'n_params': int(k),
                    'n_samples': int(n),
                    'log_likelihood': float(log_likelihood),
                    'rss': float(ss_res)
                }
                
                print_status(f"  {model_name}: R² = {r_squared:.3f}, AIC = {aic:.1f}", "SUCCESS")
                
            except Exception as e:
                print_status(f"  {model_name}: Fit failed ({e})", "WARNING")
                continue
        
        # Calculate pairwise likelihood ratio test p-values
        # LR test is valid for nested models, use AIC weights for non-nested
        if 'Exponential' in ac_results:
            exponential_ll = ac_results['Exponential']['log_likelihood']
            exponential_k = ac_results['Exponential']['n_params']
            
            for model_name in ac_results:
                if model_name != 'Exponential':
                    model_ll = ac_results[model_name]['log_likelihood']
                    model_k = ac_results[model_name]['n_params']
                    
                    # Likelihood ratio statistic
                    lr_stat = -2 * (exponential_ll - model_ll)
                    df_diff = abs(model_k - exponential_k)
                    
                    # For non-nested models, LR test is approximate
                    # Use Vuong test or AIC weights instead for rigor
                    # Here we use Wilks' theorem approximation (chi-square)
                    from scipy import stats as scipy_stats
                    
                    if df_diff > 0:
                        p_value = 1 - scipy_stats.chi2.cdf(abs(lr_stat), df_diff)
                        ac_results[model_name]['lr_test_vs_exponential'] = {
                            'lr_statistic': float(lr_stat),
                            'df': int(df_diff),
                            'p_value': float(p_value),
                            'note': 'Approximate test; assumes models are nested or nearly so'
                        }
        
        results[ac] = ac_results
    
    return results

@ensure_single_instance
def main():
    """Main entry point for Step 4.0 Advanced Analysis."""

    print_status("TEP GNSS Analysis Package v0.13 - STEP 4.0: Advanced Analysis", "INFO")
    print_status("Focused validation: Elevation, Circular Statistics, Model Comparison", "INFO")
    
    root_dir = PACKAGE_ROOT  # Use PACKAGE_ROOT consistently
    output_dir = root_dir / 'results/outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check prerequisites
        step_2_1_complete_log = root_dir / 'results/outputs/step_2_1_geospatial_processing.json'
        if not step_2_1_complete_log.exists():
            raise TEPFileError(f"Step 2.1 must be completed before running Step 4.0. Log file not found: {step_2_1_complete_log}")
        
        all_results = {}
        
        # 1. Elevation dependence analysis (enhanced with geomagnetic stratification)
        print_status("\n" + "-"*60, "INFO")
        print_status("1. ELEVATION DEPENDENCE ANALYSIS (Enhanced with Geomagnetic Stratification)", "INFO")
        print_status("-"*60, "INFO")
        all_results['elevation_dependence'] = analyze_elevation_dependence_fixed(root_dir)
        
        # 2. Regional jackknife analysis (SIMPLIFIED for memory efficiency)
        print_status("\n" + "-"*60, "INFO")
        print_status("2. REGIONAL JACKKNIFE ANALYSIS (SIMPLIFIED)", "INFO")
        print_status("-"*60, "INFO")
        print_status("Running simplified regional jackknife analysis with reduced memory usage", "INFO")
        all_results['regional_jackknife'] = analyze_regional_jackknife_simplified(root_dir)
        
        # 3. Circular statistics (unique to Step 4.0)
        print_status("\n" + "-"*60, "INFO")
        print_status("3. CIRCULAR STATISTICS ANALYSIS", "INFO")
        print_status("-"*60, "INFO")
        all_results['circular_statistics'] = analyze_circular_statistics(root_dir)
        
        # 4. Model comparison (unique to Step 4.0)
        print_status("\n" + "-"*60, "INFO")
        print_status("4. RIGOROUS MODEL COMPARISON", "INFO")
        print_status("-"*60, "INFO")
        all_results['model_comparison'] = analyze_model_comparison(root_dir)
        
        # Save consolidated results
        output_file = output_dir / 'step_4_0_advanced_analysis.json'
        try:
            safe_json_write({
                'step': "4.0",
                'version': '1.0-streamlined',
                'timestamp': datetime.now().isoformat(),
                'analyses_performed': [
                    'elevation_dependence',
                    'circular_statistics',
                    'model_comparison'
                ],
                'results': all_results
            }, output_file, indent=2)
        except IOError as e:
            raise TEPFileError(f"Failed to write advanced analysis results to {output_file}: {e}")
        
        print_status("\n" + "="*80, "INFO")
        print_status("STEP 4.0 ADVANCED ANALYSIS COMPLETE", "INFO")
        print_status("="*80, "INFO")
        print_status(f"Results saved: {output_file}", "SUCCESS")
        print_status("Note: Anisotropy/azimuth analyses are performed in Step 2.2.", "INFO")
        
        return True

    except TEPFileError as e:
        print_status(f"Advanced analysis failed due to file error: {e}", "ERROR")
        sys.exit(1)
    except TEPDataError as e:
        print_status(f"Advanced analysis failed due to data error: {e}", "ERROR")
        sys.exit(1)
    except TEPAnalysisError as e:
        print_status(f"Advanced analysis failed due to analysis error: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred during advanced analysis: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
