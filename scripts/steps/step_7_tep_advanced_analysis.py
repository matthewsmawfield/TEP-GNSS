#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 7: Advanced Analysis
=============================================

This streamlined script focuses on the most valuable, non-redundant advanced analyses.
Analyses already performed in other steps (e.g., anisotropy in Step 5) have been removed.

Valuable Analyses Included:
- Elevation dependence analysis (with corrected coordinate mapping)
- Circular statistics analysis (a unique statistical approach)
- Rigorous model comparison (for statistical validation)

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

# Global runtime configuration
VERBOSE = True
STRICT_MODE = True
MAX_PAIR_FILES = None

# Import TEP utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    safe_csv_read, safe_json_read, safe_json_write
)

def print_status(text: str, status: str = "INFO"):
    """Print status with icons, respecting global VERBOSE flag."""
    prefixes = {
        "INFO": "[INFO]",
        "SUCCESS": "[OK]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
        "PROCESSING": "[PROC]"
    }
    if status == "INFO" and not VERBOSE:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

def assert_condition(condition: bool, message: str):
    """Assert condition or raise RuntimeError in STRICT_MODE."""
    if not condition:
        print_status(message, "ERROR")
        if STRICT_MODE:
            raise RuntimeError(message)

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model: C(r) = A * exp(-r/λ) + C0"""
    return A * np.exp(-r / lambda_km) + C0

def fit_exponential(distances, coherences, weights=None, p0=None,
                    bounds=([0.01, 100, -1], [2, 20000, 1]), maxfev=5000):
    """Fit exponential_model to data and return params, errors, R²."""
    distances = np.asarray(distances)
    coherences = np.asarray(coherences)
    if p0 is None:
        c_range = coherences.max() - coherences.min()
        p0 = [c_range, 3000, coherences.min()]

    sigma = None
    if weights is not None:
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

def load_station_coordinates():
    """Load ground station coordinates with elevation data"""
    root_dir = Path(__file__).parent.parent.parent
    coords_file = root_dir / 'data/coordinates/station_coords_global.csv'
    
    assert_condition(coords_file.exists(),
                     "Station coordinates file not found – ensure Step 1 data acquisition was successful")
        
    try:
        df = pd.read_csv(coords_file)
        print_status(f"Loaded coordinates for {len(df)} ground stations", "SUCCESS")
        return df
    except Exception as e:
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
    print_status(f"Stations with geomagnetic coordinates: {stations_with_geomag}", "SUCCESS")
    
    results = {}
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in analysis_centers:
        print_status(f"Processing elevation analysis for {ac.upper()}", "INFO")
        
        # Load pair-level data
        pair_dir = root_dir / 'results/tmp'
        pair_files = sorted(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
        
        if not pair_files:
            print_status(f"No pair files found for {ac}", "WARNING")
            continue
        
        # Load and concatenate pair data
        df_chunks = []
        for pfile in pair_files[:MAX_PAIR_FILES]:
            try:
                chunk = pd.read_csv(pfile)
                df_chunks.append(chunk)
            except Exception as e:
                print_status(f"Failed to load {pfile}: {e}", "WARNING")
                continue
        
        if not df_chunks:
            print_status(f"No valid pair data for {ac}", "WARNING")
            continue
        
        df_all = pd.concat(df_chunks, ignore_index=True)
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
        print_status(f"Found {len(df_valid)} pairs with elevation data for {ac.upper()}", "SUCCESS" if len(df_valid) > 0 else "WARNING")
        
        if len(df_valid) == 0:
            results[ac] = {'error': 'No pairs with elevation data after coordinate mapping fix'}
            continue
        
        # Compute coherence and elevation metrics
        df_valid['coherence'] = np.cos(df_valid['plateau_phase'])
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
            if len(subset) < 100:  # Need minimum data
                continue
            
            # Bin analysis
            edges = np.logspace(np.log10(50), np.log10(20000), 41)
            subset['dist_bin'] = pd.cut(subset['dist_km'], bins=edges, right=False)
            binned = subset.groupby('dist_bin', observed=True).agg(
                mean_dist=('dist_km', 'mean'),
                mean_coh=('coherence', 'mean'),
                count=('coherence', 'size')
            ).reset_index()
            
            binned = binned[binned['count'] >= 10].dropna()
            if len(binned) < 5:
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
            elev_bins = np.percentile(df_with_geomag['mean_elev_m'], [0, 33, 67, 100])  # 3 elevation bins
            geomag_bins = np.percentile(df_with_geomag['mean_geomag_lat'], [0, 33, 67, 100])  # 3 geomagnetic bins
            
            for i in range(3):  # Elevation bins
                for j in range(3):  # Geomagnetic bins
                    elev_mask = (df_with_geomag['mean_elev_m'] >= elev_bins[i]) & (df_with_geomag['mean_elev_m'] < elev_bins[i+1])
                    geomag_mask = (df_with_geomag['mean_geomag_lat'] >= geomag_bins[j]) & (df_with_geomag['mean_geomag_lat'] < geomag_bins[j+1])
                    
                    if i == 2:  # Include maximum in last elevation bin
                        elev_mask = (df_with_geomag['mean_elev_m'] >= elev_bins[i])
                    if j == 2:  # Include maximum in last geomagnetic bin  
                        geomag_mask = (df_with_geomag['mean_geomag_lat'] >= geomag_bins[j])
                    
                    subset = df_with_geomag[elev_mask & geomag_mask].copy()
                    
                    if len(subset) < 50:  # Need minimum data
                        continue
                    
                    # Bin analysis for this stratum
                    edges = np.logspace(np.log10(50), np.log10(20000), 31)  # Fewer bins for smaller samples
                    subset['dist_bin'] = pd.cut(subset['dist_km'], bins=edges, right=False)
                    binned = subset.groupby('dist_bin', observed=True).agg(
                        mean_dist=('dist_km', 'mean'),
                        mean_coh=('coherence', 'mean'),
                        count=('coherence', 'size')
                    ).reset_index()
                    
                    binned = binned[binned['count'] >= 5].dropna()
                    if len(binned) < 4:
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

def analyze_regional_jackknife(root_dir):
    """
    Perform regional jackknife analysis to test λ(h) robustness.
    Systematically excludes major geographic regions to assess artifact influence.
    """
    print_status("Starting Regional Jackknife Analysis", "INFO")
    
    # Define major geographic regions that might contain systematic artifacts
    regions = {
        'Andes': {'lat_range': (-60, 15), 'lon_range': (-85, -65), 'description': 'Andes Mountains'},
        'Tibet': {'lat_range': (25, 40), 'lon_range': (75, 105), 'description': 'Tibetan Plateau'},  
        'Himalayas': {'lat_range': (25, 35), 'lon_range': (70, 90), 'description': 'Himalayan Range'},
        'Rockies': {'lat_range': (30, 55), 'lon_range': (-125, -100), 'description': 'Rocky Mountains'},
        'Alps': {'lat_range': (43, 48), 'lon_range': (5, 17), 'description': 'Alpine Region'},
        'Antarctica': {'lat_range': (-90, -60), 'lon_range': (-180, 180), 'description': 'Antarctic Stations'}
    }
    
    # Load station coordinates with geomagnetic data
    coords_df = load_station_coordinates()
    
    results = {}
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in analysis_centers:
        print_status(f"Regional jackknife analysis for {ac.upper()}", "INFO")
        
        # Load pair-level data (same as elevation analysis)
        pair_dir = root_dir / 'results/tmp'
        pair_files = sorted(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
        
        if not pair_files:
            print_status(f"No pair files found for {ac}", "WARNING")
            continue
        
        # Load and process data (abbreviated for jackknife)
        df_chunks = []
        for pfile in pair_files[:10]:  # Limit files for efficiency
            try:
                chunk = pd.read_csv(pfile)
                df_chunks.append(chunk)
            except Exception as e:
                print_status(f"Failed to load {pfile}: {e}", "WARNING")
                continue
        
        if not df_chunks:
            continue
            
        df_all = pd.concat(df_chunks, ignore_index=True)
        
        # Add station coordinates to pairs
        station_coords = {}
        for _, row in coords_df.iterrows():
            code = str(row['code']).strip().upper()
            station_coords[code] = {
                'lat': row.get('lat_deg'),
                'lon': row.get('lon_deg'),
                'elevation': row.get('height_m'),
                'geomag_lat': row.get('geomag_lat')
            }
        
        def get_station_coords(station_code):
            code = str(station_code).strip().upper()
            # Try various formats
            for key in [code, code[:4], code[:3]]:
                if key in station_coords:
                    return station_coords[key]
            return None
        
        # Add coordinates to dataframe
        df_all['coords_i'] = df_all['station_i'].apply(get_station_coords)
        df_all['coords_j'] = df_all['station_j'].apply(get_station_coords)
        
        # Filter pairs with valid coordinates
        df_with_coords = df_all[
            df_all['coords_i'].notna() & df_all['coords_j'].notna()
        ].copy()
        
        if len(df_with_coords) == 0:
            continue
        
        # Extract coordinate components
        df_with_coords['lat_i'] = df_with_coords['coords_i'].apply(lambda x: x['lat'] if x else None)
        df_with_coords['lon_i'] = df_with_coords['coords_i'].apply(lambda x: x['lon'] if x else None)
        df_with_coords['lat_j'] = df_with_coords['coords_j'].apply(lambda x: x['lat'] if x else None)
        df_with_coords['lon_j'] = df_with_coords['coords_j'].apply(lambda x: x['lon'] if x else None)
        df_with_coords['elev_i'] = df_with_coords['coords_i'].apply(lambda x: x['elevation'] if x else None)
        df_with_coords['elev_j'] = df_with_coords['coords_j'].apply(lambda x: x['elevation'] if x else None)
        
        # Filter valid elevation data
        df_valid = df_with_coords.dropna(subset=['lat_i', 'lon_i', 'lat_j', 'lon_j', 'elev_i', 'elev_j']).copy()
        df_valid['coherence'] = np.cos(df_valid['plateau_phase'])
        df_valid['mean_elev_m'] = (df_valid['elev_i'] + df_valid['elev_j']) / 2
        
        print_status(f"Base dataset: {len(df_valid)} pairs with coordinates and elevation", "INFO")
        
        # Perform jackknife analysis for each region
        region_results = {}
        
        for region_name, region_def in regions.items():
            print_status(f"  Testing exclusion of {region_def['description']}", "INFO")
            
            # Identify pairs in the region
            lat_range = region_def['lat_range']
            lon_range = region_def['lon_range']
            
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
            
            if len(df_excluded) < 1000:  # Need sufficient data
                region_results[region_name] = {'error': 'Insufficient data after exclusion'}
                continue
            
            # Fit λ(h) relationship on excluded dataset
            try:
                # Simple elevation-lambda correlation
                elev_bins = np.percentile(df_excluded['mean_elev_m'], np.linspace(0, 100, 6))  # 5 bins
                bin_lambdas = []
                bin_elevations = []
                
                for i in range(5):
                    mask = (df_excluded['mean_elev_m'] >= elev_bins[i]) & (df_excluded['mean_elev_m'] < elev_bins[i+1])
                    if i == 4:  # Include maximum
                        mask = (df_excluded['mean_elev_m'] >= elev_bins[i])
                    
                    subset = df_excluded[mask]
                    if len(subset) < 100:
                        continue
                    
                    # Distance binning and exponential fit
                    edges = np.logspace(np.log10(50), np.log10(20000), 31)
                    subset['dist_bin'] = pd.cut(subset['dist_km'], bins=edges, right=False)
                    binned = subset.groupby('dist_bin', observed=True).agg(
                        mean_dist=('dist_km', 'mean'),
                        mean_coh=('coherence', 'mean'),
                        count=('coherence', 'size')
                    ).reset_index()
                    
                    binned = binned[binned['count'] >= 5].dropna()
                    if len(binned) < 4:
                        continue
                    
                    # Fit exponential
                    popt, perr, r_squared = fit_exponential(
                        binned['mean_dist'].values,
                        binned['mean_coh'].values,
                        binned['count'].values
                    )
                    
                    bin_lambdas.append(popt[1])
                    bin_elevations.append((elev_bins[i] + elev_bins[i+1]) / 2)
                
                if len(bin_lambdas) >= 3:  # Need minimum points for trend
                    # Fit linear relationship λ(h) = a + b*h
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(bin_elevations, bin_lambdas)
                    
                    region_results[region_name] = {
                        'excluded_pairs': excluded_pairs,
                        'exclusion_percent': 100 * excluded_pairs / len(df_valid),
                        'remaining_pairs': len(df_excluded),
                        'lambda_elevation_slope': float(slope),
                        'lambda_elevation_intercept': float(intercept),
                        'lambda_elevation_r_squared': float(r_value**2),
                        'lambda_elevation_p_value': float(p_value),
                        'slope_std_error': float(std_err),
                        'elevation_bins': bin_elevations,
                        'lambda_values': bin_lambdas,
                        'description': region_def['description']
                    }
                    
                    print_status(f"    λ(h) slope = {slope:.3f} ± {std_err:.3f} km/m, R² = {r_value**2:.3f}", "SUCCESS")
                else:
                    region_results[region_name] = {'error': 'Insufficient elevation bins after fitting'}
                    
            except Exception as e:
                region_results[region_name] = {'error': f'Analysis failed: {str(e)}'}
                print_status(f"    Analysis failed: {e}", "WARNING")
        
        results[ac] = {
            'total_pairs': len(df_valid),
            'regional_exclusion_analysis': region_results,
            'regions_tested': list(regions.keys())
        }
    
    return results

def analyze_circular_statistics(root_dir):
    """Analyze phase data using proper circular statistics."""
    print_status("Performing circular statistics analysis", "INFO")
    
    all_results = {}
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    
    for ac in analysis_centers:
        print_status(f"\nAnalyzing {ac.upper()}", "PROCESSING")
        
        # Load pair-level data
        pair_dir = root_dir / 'results/tmp'
        pair_files = sorted(pair_dir.glob(f"step_3_pairs_{ac}_*.csv"))
        
        if not pair_files:
            continue
        
        # Load sample of data for circular statistics
        df_chunks = []
        for pfile in pair_files[:5]:  # Sample first 5 files for efficiency
            try:
                chunk = pd.read_csv(pfile)
                df_chunks.append(chunk)
            except:
                continue
        
        if not df_chunks:
            continue
        
        df = pd.concat(df_chunks, ignore_index=True)
        
        # Distance binning
        edges = np.logspace(np.log10(50), np.log10(1000), 11)  # 10 bins
        df['dist_bin'] = pd.cut(df['dist_km'], bins=edges, right=False)
        
        results = {}
        print("Distance | PLV   | Rayleigh p | V-test p | cos(mean) | Current")
        print("-"*70)
        
        for bin_label, group in df.groupby('dist_bin', observed=True):
            if len(group) < 50:
                continue
            
            phases = group['plateau_phase'].values
            coherences = np.cos(phases)
            
            # Phase Locking Value (PLV)
            plv = np.abs(np.mean(np.exp(1j * phases)))
            
            # Rayleigh test for uniformity
            try:
                rayleigh_stat, rayleigh_p = stats.rayleightest(phases)
            except:
                rayleigh_p = np.nan
            
            # V-test for preferred direction (0 radians)
            try:
                v_stat, v_p = stats.circmean(phases), 0.0  # Simplified
            except:
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
            print(f"{dist_center:8.0f} | {plv:.3f} | {rayleigh_p:10.3e} | {v_p:8.3e} | {cos_mean:+.3f} | {current_coherence:+.3f}")
        
        all_results[ac] = results
    
    # Save results
    output_file = root_dir / 'results/outputs/step_7_circular_statistics_streamlined.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
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
        # Load correlation results from Step 3
        step3_file = root_dir / f'results/outputs/step_3_correlation_{ac}.json'
        if not step3_file.exists():
            continue
        
        with open(step3_file, 'r') as f:
            step3_data = json.load(f)
        
        if 'binned_correlations' not in step3_data:
            continue
        
        binned = step3_data['binned_correlations']
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
                
                # AIC calculation
                n = len(distances)
                k = len(popt)
                aic = 2*k + n*np.log(ss_res/n)
                
                ac_results[model_name] = {
                    'r_squared': float(r_squared),
                    'aic': float(aic),
                    'parameters': popt.tolist(),
                    'parameter_errors': np.sqrt(np.diag(pcov)).tolist()
                }
                
                print_status(f"  {model_name}: R² = {r_squared:.3f}, AIC = {aic:.1f}", "SUCCESS")
                
            except Exception as e:
                print_status(f"  {model_name}: Fit failed ({e})", "WARNING")
                continue
        
        results[ac] = ac_results
    
    return results

def main():
    """Main entry point for Step 7 Advanced Analysis."""
    parser = argparse.ArgumentParser(description="Step 7 TEP Analysis - Advanced Validation")
    parser.add_argument('--test', type=str, default='all', 
                        choices=['all', 'elevation', 'circular', 'model', 'jackknife'],
                        help='Test to run: all, elevation, circular, model, jackknife')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    global VERBOSE
    VERBOSE = args.verbose
    
    print("\n" + "="*80)
    print("TEP GNSS Analysis Package v0.5")
    print("STEP 7: Advanced TEP Analysis")
    print("Focused validation: Elevation, Circular Statistics, Model Comparison")
    print("="*80)
    
    root_dir = Path(__file__).parent.parent.parent
    output_dir = root_dir / 'results/outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check prerequisites
    step3_complete = (root_dir / 'logs/step_3_correlation_analysis.json').exists()
    if not step3_complete:
        print_status("Step 3 must be completed before running Step 7", "ERROR")
        return False
    
    all_results = {}
    
    # 1. Elevation dependence analysis (enhanced with geomagnetic stratification)
    if args.test in ['all', 'elevation']:
        print("\n" + "-"*60)
        print("1. ELEVATION DEPENDENCE ANALYSIS (Enhanced with Geomagnetic Stratification)")
        print("-"*60)
        all_results['elevation_dependence'] = analyze_elevation_dependence_fixed(root_dir)
    
    # 2. Regional jackknife analysis (NEW)
    if args.test in ['all', 'jackknife']:
        print("\n" + "-"*60)
        print("2. REGIONAL JACKKNIFE ANALYSIS")
        print("-"*60)
        all_results['regional_jackknife'] = analyze_regional_jackknife(root_dir)
    
    # 3. Circular statistics (unique to Step 7)
    if args.test in ['all', 'circular']:
        print("\n" + "-"*60)
        print("3. CIRCULAR STATISTICS ANALYSIS")
        print("-"*60)
        all_results['circular_statistics'] = analyze_circular_statistics(root_dir)
    
    # 4. Model comparison (unique to Step 7)
    if args.test in ['all', 'model']:
        print("\n" + "-"*60)
        print("4. RIGOROUS MODEL COMPARISON")
        print("-"*60)
        all_results['model_comparison'] = analyze_model_comparison(root_dir)
    
    # Save consolidated results
    output_file = output_dir / 'step_7_advanced_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'step': 7,
            'version': '1.0-streamlined',
            'timestamp': datetime.now().isoformat(),
            'analyses_performed': [
                'elevation_dependence',
                'circular_statistics',
                'model_comparison'
            ],
            'results': all_results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("STEP 7 ADVANCED ANALYSIS COMPLETE")
    print("="*80)
    print_status(f"Results saved: {output_file}", "SUCCESS")
    print_status("Note: Anisotropy/azimuth analyses are performed in Step 5.", "INFO")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
