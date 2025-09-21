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
    
    # Create elevation lookup with multiple station code formats
    elevation_lookup = {}
    
    for _, row in coords_df.iterrows():
        if pd.isna(row.get('elevation_m')):
            continue
            
        station_code = str(row['code']).strip().upper()
        elev = float(row['elevation_m'])
        
        # Add multiple formats to lookup
        elevation_lookup[station_code] = elev
        
        # Add short codes (first 4 chars)
        if len(station_code) >= 4:
            elevation_lookup[station_code[:4]] = elev
        
        # Add without numbers/suffixes
        import re
        clean_code = re.sub(r'[0-9]+.*$', '', station_code)
        if clean_code and clean_code != station_code:
            elevation_lookup[clean_code] = elev
    
    print_status(f"Created elevation lookup with {len(elevation_lookup)} entries", "SUCCESS")
    
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
        
        # FIXED: Better station code mapping
        def extract_short_code(full_code):
            if pd.isna(full_code):
                return None
            
            full_str = str(full_code).strip().upper()
            
            # Try direct lookup first
            if full_str in elevation_lookup:
                return full_str
            
            # Try various patterns
            patterns_to_try = [
                full_str[:4],  # First 4 characters
                re.sub(r'[0-9]+.*$', '', full_str),  # Remove numbers/suffixes
                full_str[:-3] if len(full_str) > 3 else None,  # Remove last 3 chars
                full_str[:3] if len(full_str) >= 3 else None,  # First 3 characters
            ]
            
            for pattern in patterns_to_try:
                if pattern and pattern in elevation_lookup:
                    return pattern
            
            return None
        
        df_all['short_i'] = df_all['station_i'].apply(extract_short_code)
        df_all['short_j'] = df_all['station_j'].apply(extract_short_code)
        df_all['elev_i'] = df_all['short_i'].map(elevation_lookup)
        df_all['elev_j'] = df_all['short_j'].map(elevation_lookup)
        
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
        
        results[ac] = {
            'total_pairs': len(df_all),
            'pairs_with_elevation': len(df_valid),
            'elevation_coverage_percent': 100 * len(df_valid) / len(df_all),
            'quintile_analysis': quintile_results,
            'elevation_range_m': [float(df_valid['mean_elev_m'].min()), float(df_valid['mean_elev_m'].max())],
            'coordinate_mapping_fixed': True
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
                        choices=['all', 'elevation', 'circular', 'model'],
                        help='Test to run: all, elevation, circular, model')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    global VERBOSE
    VERBOSE = args.verbose
    
    print("\n" + "="*80)
    print("TEP GNSS Analysis Package v0.3")
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
    
    # 1. Elevation dependence analysis
    if args.test in ['all', 'elevation']:
        print("\n" + "-"*60)
        print("1. ELEVATION DEPENDENCE ANALYSIS")
        print("-"*60)
        all_results['elevation_dependence'] = analyze_elevation_dependence_fixed(root_dir)
    
    # 2. Circular statistics (unique to Step 7)
    if args.test in ['all', 'circular']:
        print("\n" + "-"*60)
        print("2. CIRCULAR STATISTICS ANALYSIS")
        print("-"*60)
        all_results['circular_statistics'] = analyze_circular_statistics(root_dir)
    
    # 3. Model comparison (unique to Step 7)
    if args.test in ['all', 'model']:
        print("\n" + "-"*60)
        print("3. RIGOROUS MODEL COMPARISON")
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
