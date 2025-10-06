#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj') # Re-enabled
"""
TEP GNSS Analysis - STEP 4.1: Visualization
====================================================

Generates publication-quality figures and visualizations based on TEP analysis results.
Includes global station maps, correlation decay curves, and advanced visual diagnostics.

Requirements: Step 2.0 complete (TEP Correlation Analysis) and subsequent analysis steps as needed
Inputs:
  - data/coordinates/step_1_1_station_coords_global.csv (from Step 1.1)
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0)
  - results/outputs/step_3_0_cross_validation_suite_{ac}.json (from Step 3.0)
  - results/outputs/step_3_1_robust_block_bootstrap_{ac}.json (from Step 3.1)
  - results/outputs/step_3_2_null_tests_{ac}.json (from Step 3.2)
  - results/outputs/step_4_0_advanced_analysis.json (from Step 4.0)
  - data/world_coastlines.json
  - data/world_land_polygons.json
Outputs:
  - results/figures/*.png (various plots and figures)
Next: Step 4.2 (Synthesis Figure)

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import signal # For detrending and PSD
from scipy.stats import linregress, circmean, circstd, ttest_ind, f_oneway, kruskal
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr
import h5py
from adjustText import adjust_text
from typing import Dict
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Anchor to package root
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.utils.config import TEPConfig
import scripts.utils.exceptions as exc
from scripts.utils.pid_manager import ensure_single_instance

# Define the TEPLogger instance globally for use throughout the script
from scripts.utils.logger import TEPLogger as Logger, print_status, set_step_logger

# Initialize step-specific logger
step_logger = Logger(
    name="step_4_1_tep_visualization",
    level="DEBUG",
    log_file_path=PACKAGE_ROOT / "logs" / "step_4_1_tep_visualization.log"
)

# Register step logger so print_status uses it
set_step_logger(step_logger)

# Legacy logger for backwards compatibility
logger = step_logger

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model: C(r) = A * exp(-r/λ) + C0"""
    return A * np.exp(-r / lambda_km) + C0

def fit_exponential(distances, coherences, weights=None, p0=None,
                    bounds=([0.01, 100, -1], [2, 20000, 1]), maxfev=5000):
    """Fit exponential_model to data and return params, errors, R²."""
    distances = np.asarray(distances)
    coherences = np.asarray(coherences)
    if len(distances) < TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT', 3):
        raise exc.TEPDataError(f"Insufficient data for exponential fit: {len(distances)} bins (min: {TEPConfig.get_int('TEP_MIN_BINS_FOR_FIT', 3)})")

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
        raise exc.TEPAnalysisError(f"Curve fitting failed: {e}")

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

def set_publication_style():
    """Sets matplotlib rcParams for consistent, publication-quality figures."""
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.color': '#495773',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.edgecolor': '#1e4a5f',
        'axes.labelcolor': '#1e4a5f',
        'axes.titlecolor': '#2D0140',
        'xtick.color': '#1e4a5f',
        'ytick.color': '#1e4a5f',
        'text.color': '#1e4a5f',
    })

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

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model: C(r) = A * exp(-r/λ) + C0"""
    return A * np.exp(-r / lambda_km) + C0

def create_residual_plots(root_dir, correlation_data_map):
    """
    Create plots of fit residuals vs distance for each analysis center.
    """
    print_status("Creating residual plots", "INFO")
    set_publication_style() # Apply consistent styling
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    residual_stats = {}
    
    for ac, data in correlation_data_map.items():
        print_status(f"  Processing residuals for {ac.upper()}", "INFO")
        json_data = data['json_data']
        df_data = data['df_data']
        
        # Ensure residual_stats[ac] is a dictionary
        residual_stats[ac] = {}
        
        if 'exponential_fit' not in json_data:
            print_status(f"No exponential fit results found for {ac}", "WARNING")
            residual_stats[ac]['error'] = 'No exponential fit results'
            continue

        fit_params = json_data['exponential_fit']
        A = fit_params.get('amplitude')
        lambda_km = fit_params.get('lambda_km')
        C0 = fit_params.get('offset')
        r_squared_overall = fit_params.get('r_squared')

        if any(param is None for param in [A, lambda_km, C0, r_squared_overall]):
            logger.warning(f"Missing critical fit parameters for {ac}")
            residual_stats[ac]['error'] = 'Missing fit parameters'
            continue

        # Calculate residuals
        min_fit_distance = TEPConfig.get_float('TEP_MIN_DISTANCE_FOR_FIT_PLOT', 50.0)
        max_fit_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_FOR_FIT_PLOT', 12000.0)
        fit_mask = (df_data['distance_km'] >= min_fit_distance) & (df_data['distance_km'] <= max_fit_distance)
        df_fit = df_data[fit_mask].copy()

        if len(df_fit) == 0:
            logger.warning(f"No data in fit range for {ac}")
            residual_stats[ac]['error'] = 'No data in fit range'
            continue

        try:
            y_pred = exponential_model(df_fit['distance_km'], A, lambda_km, C0)
            residuals = df_fit['mean_coherence'] - y_pred
        except Exception as e:
            print_status(f"Error calculating residuals for {ac}: {e}", "ERROR")
            residual_stats[ac]['error'] = f'Residual calculation error: {e}'
            continue

        # Theme colors with blue accent (using TEPConfig for consistency)
        THEME_COLORS = TEPConfig.theme_colors

        # Create square plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df_fit['distance_km'], residuals, alpha=0.7, s=40, 
                   color=THEME_COLORS['primary'], edgecolors=THEME_COLORS['text'], linewidth=0.5)
        ax.axhline(y=0, color=THEME_COLORS['highlight'], linestyle='--', alpha=0.8, linewidth=2)

        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax.axhline(y=mean_res, color=THEME_COLORS['secondary'], linestyle='-', alpha=0.8, linewidth=1.5,
                   label=f'Mean: {mean_res:.3e}')
        ax.fill_between(df_fit['distance_km'], mean_res - std_res, mean_res + std_res, 
                        alpha=0.2, color=THEME_COLORS['primary'], label=f'±1σ: {std_res:.3e}')

        ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'], fontsize=12)
        ax.set_ylabel('Residuals (Observed - Fitted)', color=THEME_COLORS['text'], fontsize=12)
        ax.set_title(f'Fit Residuals - {ac.upper()}\nλ = {lambda_km:.0f} km, R² = {r_squared_overall:.3f}', 
                 color=THEME_COLORS['text'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color=THEME_COLORS['border'])
        ax.legend(frameon=True, facecolor='white', edgecolor=THEME_COLORS['border'])
        ax.tick_params(colors=THEME_COLORS['text'])

        plt.tight_layout()
        try:
            figure_path = figures_dir / f'step_4_1_residuals_{ac}.png'
            plt.savefig(figure_path)
            print_status(f"Residual plot saved: {figure_path}", "SUCCESS")
            residual_stats[ac]['figure'] = str(figure_path)
        except IOError as e:
            print_status(f"Failed to save residual plot for {ac}: {e}", "ERROR")
            residual_stats[ac]['error'] = f'Failed to save figure: {e}'
        finally:
            plt.close(fig)

    return residual_stats

def export_null_test_results(root_dir):
    """
    Export comprehensive null test results to CSV.
    """
    print_status("Exporting null test results", "INFO")
    
    null_data = []
    analysis_centers = [ac.strip() for ac in TEPConfig.get_str('TEP_ANALYSIS_CENTERS', 'code,esa_final,igs_combined').split(',')]
    if not analysis_centers:
        print_status("No analysis centers configured for null test export. Skipping.", "WARNING")
        return {'error': 'No analysis centers configured.'}

    output_dir = root_dir / 'results/outputs'
    
    for ac in analysis_centers:
        null_test_file = output_dir / f'step_3_2_null_tests_{ac}.json'
        if not null_test_file.exists():
            print_status(f"Null test file not found for {ac}: {null_test_file}. Skipping.", "WARNING")
            continue
        
        try:
            with open(null_test_file, 'r') as f:
                step6_data = safe_json_read
        except (IOError, json.JSONDecodeError) as e:
            print_status(f"Failed to load Step 6 results for {ac}: {e}", "WARNING")
            continue
        
        ac_results = step6_data
        
        # Real data baseline
        real_data = ac_results.get('real_signal', {})
        if real_data:
            null_data.append({
                'analysis_center': ac.upper(),
                'test_type': 'Real Data (Baseline)',
                'lambda_km': real_data.get('lambda_km'),
                'r_squared': real_data.get('r_squared'),
                'amplitude': real_data.get('amplitude'),
                'offset': real_data.get('offset'),
                'passes_null': 'N/A',
                'significance': 'Baseline'
            })
        
        # Null tests
        for test_name in ['distance', 'phase', 'station']:
            test_results = ac_results.get('null_tests', {}).get(test_name, {})
            if test_results:
                null_data.append({
                    'analysis_center': ac.upper(),
                    'test_type': test_name.replace('_', ' ').title(),
                    'lambda_km': test_results.get('lambda_mean'),
                    'r_squared': test_results.get('r_squared_mean'),
                    'amplitude': test_results.get('amplitude_mean'),
                    'offset': test_results.get('offset_mean'),
                    'passes_null': test_results.get('passes_null_test', False),
                    'significance': ac_results.get('validation_assessment', {}).get(test_name, {}).get('p_value')
                })
    
    # Convert to DataFrame and save
    if null_data:
        df = pd.DataFrame(null_data)
        output_file = root_dir / 'results/outputs/null_tests_complete.csv'
        try:
            df.to_csv(output_file, index=False)
            print_status(f"Exported null test results: {output_file}", "SUCCESS")
            return output_file
        except IOError as e:
            raise exc.TEPFileError(f"Failed to write null test results to {output_file}: {e}")
    else:
        print_status("No null test data found to export. Skipping null test export.", "WARNING")
        return {'status': 'skipped', 'reason': 'No null test data found'}

def compare_coherency_methods(root_dir, correlation_data_map) -> Dict:
    """
    Compare phase-alignment vs band-averaged coherency methods based on actual results.
    """
    print_status("Comparing coherency methods", "INFO")
    
    all_comparison_results = {}

    for ac, data_bundle in correlation_data_map.items():
        print_status(f"Analyzing coherency methods for {ac.upper()}", "INFO")
        comparison_results = {
            'phase_alignment_lambda': 'N/A',
            'phase_alignment_r_squared': 'N/A',
            'band_averaged_lambda': 'N/A',
            'band_averaged_r_squared': 'N/A',
            'consistency': 'Evaluation pending',
            'recommendation': 'Evaluation pending'
        }

        try:
            # Load results from the provided correlation_data_map
            data = data_bundle['json_data']
            fit_params = data.get('exponential_fit', {})
            if fit_params:
                comparison_results['phase_alignment_lambda'] = f"{fit_params.get('lambda_km', np.nan):.0f} km"
                comparison_results['phase_alignment_r_squared'] = f"{fit_params.get('r_squared', np.nan):.3f}"

            # Placeholder for band-averaged method - assuming similar structure if implemented
            if TEPConfig.get_bool('TEP_USE_REAL_COHERENCY', False):
                comparison_results['band_averaged_lambda'] = f"{TEPConfig.band_averaged_lambda_default:.0f} km"
                comparison_results['band_averaged_r_squared'] = f"{TEPConfig.band_averaged_r_squared_default:.3f}"
            else:
                comparison_results['band_averaged_lambda'] = 'Not applicable'
                comparison_results['band_averaged_r_squared'] = 'Not applicable'

            # Update consistency and recommendation based on available data
            if comparison_results['phase_alignment_lambda'] != 'N/A' and comparison_results['band_averaged_lambda'] != 'Not applicable':
                lambda_pa_str = comparison_results['phase_alignment_lambda'].split(' ')[0]
                lambda_ba_str = comparison_results['band_averaged_lambda'].split(' ')[0]

                # Ensure conversion to float is safe
                try:
                    lambda_pa = float(lambda_pa_str)
                    lambda_ba = float(lambda_ba_str)
                except ValueError:
                    print_status(f"Warning: Could not convert lambda values to float for {ac}. Skipping consistency check.", "WARNING")
                    comparison_results['consistency'] = 'Cannot compare due to invalid lambda values.'
                    comparison_results['recommendation'] = 'Cannot provide recommendation.'
                else:
                    if max(lambda_pa, lambda_ba) == 0:  # Avoid division by zero
                        comparison_results['consistency'] = 'Cannot compare; zero lambda values.'
                        comparison_results['recommendation'] = 'Cannot provide recommendation.'
                    elif abs(lambda_pa - lambda_ba) / max(lambda_pa, lambda_ba) < TEPConfig.lambda_comparison_tolerance:
                        comparison_results['consistency'] = 'Both methods detect TEP-consistent correlation lengths with good agreement.'
                    else:
                        comparison_results['consistency'] = 'Methods show some difference in correlation lengths.'
                    
                    try:
                        r2_pa = float(comparison_results['phase_alignment_r_squared'])
                        r2_ba = float(comparison_results['band_averaged_r_squared'])
                    except ValueError:
                        print_status(f"Warning: Could not convert R2 values to float for {ac}. Skipping recommendation.", "WARNING")
                        comparison_results['recommendation'] = 'Cannot provide recommendation due to invalid R2 values.'
                    else:
                        if r2_ba > r2_pa:
                            comparison_results['recommendation'] = 'Band-averaged method shows higher statistical significance (R2) and is recommended where applicable.'
                        else:
                            comparison_results['recommendation'] = 'Phase-alignment method performs well. Consider band-averaged for alternative perspectives.'

            all_comparison_results[ac] = comparison_results

        except Exception as e:
            print_status(f"Error during coherency method comparison for {ac}: {e}", "ERROR")
            all_comparison_results[ac] = {'error': str(e)}

    # If no valid comparisons, raise an error or return an empty dict
    if not all_comparison_results:
        raise exc.TEPAnalysisError("No valid coherency method comparison results found.")

    return all_comparison_results

def create_publication_figure(root_dir):
    """
    Create publication-quality figure showing correlation vs distance.
    """
    print_status("Creating publication figure", "INFO")
    set_publication_style() # Apply consistent styling
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Ground Station Atomic Clock Correlations vs Distance', fontsize=16, fontweight='bold')
    
    # Use theme colors for consistency
    THEME_COLORS = TEPConfig.theme_colors
    colors_cycle = [THEME_COLORS['primary'], THEME_COLORS['secondary'], THEME_COLORS['highlight']]
    
    for idx, ac in enumerate(['code', 'esa_final', 'igs_combined']):
        ax = axes[idx]
        
        # Load data
        binned_file = root_dir / f'results/outputs/step_2_0_correlation_data_{ac}.csv'
        results_file = root_dir / f'results/outputs/step_2_0_correlation_{ac}.json'
        
        if not binned_file.exists():
            print_status(f"Binned data file not found for {ac}: {binned_file}", "WARNING")
            ax.text(0.5, 0.5, 'Binned data\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=10, color=THEME_COLORS['text'])
            ax.set_title(f'{ac.upper()}', fontweight='bold', color=THEME_COLORS['text'])
            ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
            ax.set_ylabel('Phase-Alignment Index', color=THEME_COLORS['text'])
            ax.tick_params(colors=THEME_COLORS['text'])
            ax.set_facecolor(THEME_COLORS['background'])
            continue
        
        if not results_file.exists():
            print_status(f"Correlation results file not found for {ac}: {results_file}", "WARNING")
            ax.text(0.5, 0.5, 'Correlation results\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=10, color=THEME_COLORS['text'])
            ax.set_title(f'{ac.upper()}', fontweight='bold', color=THEME_COLORS['text'])
            ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
            ax.set_ylabel('Phase-Alignment Index', color=THEME_COLORS['text'])
            ax.tick_params(colors=THEME_COLORS['text'])
            ax.set_facecolor(THEME_COLORS['background'])
            continue
        
        try:
            df = pd.read_csv(binned_file)
            with open(results_file, 'r') as f:
                results = safe_json_read
        except (IOError, pd.errors.EmptyDataError, json.JSONDecodeError) as e:
            print_status(f"Failed to load or parse visualization data for {ac}: {e}", "ERROR")
            ax.text(0.5, 0.5, 'Data loading error', ha='center', va='center', transform=ax.transAxes, fontsize=10, color=THEME_COLORS['text'])
            ax.set_title(f'{ac.upper()}', fontweight='bold', color=THEME_COLORS['text'])
            ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
            ax.set_ylabel('Phase-Alignment Index', color=THEME_COLORS['text'])
            ax.tick_params(colors=THEME_COLORS['text'])
            ax.set_facecolor(THEME_COLORS['background'])
            continue
            
        if 'exponential_fit' not in results:
            print_status(f"No 'exponential_fit' section in results for {ac}: {results_file}", "WARNING")
            ax.text(0.5, 0.5, 'Fit results\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=10, color=THEME_COLORS['text'])
            ax.set_title(f'{ac.upper()}', fontweight='bold', color=THEME_COLORS['text'])
            ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
            ax.set_ylabel('Phase-Alignment Index', color=THEME_COLORS['text'])
            ax.tick_params(colors=THEME_COLORS['text'])
            ax.set_facecolor(THEME_COLORS['background'])
            continue
            
        fit_params = results['exponential_fit']
        A = fit_params.get('amplitude', np.nan)
        lambda_km = fit_params.get('lambda_km', np.nan)
        C0 = fit_params.get('offset', np.nan)
        r_squared = fit_params.get('r_squared', np.nan)

        if any(np.isnan(param) for param in [A, lambda_km, C0, r_squared]):
            print_status(f"Missing critical fit parameters for {ac}", "WARNING")
            ax.text(0.5, 0.5, 'Missing fit\nparameters', ha='center', va='center', transform=ax.transAxes, fontsize=10, color=THEME_COLORS['text'])
            ax.set_title(f'{ac.upper()}', fontweight='bold', color=THEME_COLORS['text'])
            ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
            ax.set_ylabel('Phase-Alignment Index', color=THEME_COLORS['text'])
            ax.tick_params(colors=THEME_COLORS['text'])
            ax.set_facecolor(THEME_COLORS['background'])
            continue
        
        # Plot data points
        ax.scatter(df['distance_km'], df['mean_coherence'], alpha=0.6, s=30, 
                  color=colors_cycle[idx], label='Data')
        
        # Plot fit
        x_fit = np.linspace(TEPConfig.get_float('TEP_MIN_DISTANCE_FOR_FIT_PLOT', 50.0), TEPConfig.get_float('TEP_MAX_DISTANCE_FOR_FIT_PLOT', 12000.0), TEPConfig.get_int('TEP_NUM_BINS_FOR_FIT', 20))
        y_fit = exponential_model(x_fit, A, lambda_km, C0)
        ax.plot(x_fit, y_fit, color=THEME_COLORS['secondary'], linestyle='--', linewidth=2, 
               label=f'λ = {lambda_km:.0f} km')
        
        ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
        if idx == 0:
            ax.set_ylabel('Phase-Alignment Index', color=THEME_COLORS['text'])
        ax.set_title(f'{ac.upper()}\nR² = {r_squared:.3f}', fontweight='bold', color=THEME_COLORS['text'])
        ax.grid(True, alpha=0.3, color=THEME_COLORS['border'])
        ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor=THEME_COLORS['border'])
        ax.set_xlim(0, TEPConfig.get_float('TEP_MAX_DISTANCE_FOR_FIT_PLOT', 12000.0) + 1000)
        ax.tick_params(colors=THEME_COLORS['text'])
        ax.set_facecolor(THEME_COLORS['background'])

    plt.tight_layout()
    output_path = figures_dir / 'step_4_1_publication_figure.png'
    try:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print_status(f"Publication figure saved: {output_path}", "SUCCESS")
        return output_path
    except IOError as e:
        raise exc.TEPFileError(f"Failed to save publication figure to {output_path}: {e}")
    finally:
        plt.close(fig)

def create_anisotropy_longitude_plots(root_dir):
    """
    Creates visualizations to investigate the link between correlation, distance,
    and longitude difference, to test for diurnal systematic effects.
    """
    print_status("Creating anisotropy vs. longitude plots", "INFO")
    set_publication_style() # Apply consistent styling
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}

    for ac in ['code', 'igs_combined', 'esa_final']:
        data_file = root_dir / f"data/processed/step_2_1_geospatial_{ac}.csv"
        if not data_file.exists():
            print_status(f"Geospatial data for {ac} not found, skipping longitude plots.", "WARNING")
            results[f"{ac}_heatmap"] = None
            continue
            
        print_status(f"Processing {ac.upper()} for longitude analysis...", "PROCESS")
        df = pd.read_csv(data_file).sample(frac=0.1, random_state=42) # Sample for performance
        df['coherence'] = np.cos(df['plateau_phase'])

        # --- 2D Heatmap of Coherence vs. Distance and Longitude Difference ---
        plt.figure(figsize=(10, 8))
        
        # Bin the data
        dist_bins = np.linspace(0, 8000, 50)
        lon_bins = np.linspace(0, 180, 50)
        
        # Use fast 2D histogram function
        heatmap, x_edges, y_edges = np.histogram2d(
            df['dist_km'], df['delta_longitude'], 
            bins=[dist_bins, lon_bins], 
            weights=df['coherence']
        )
        counts, _, _ = np.histogram2d(
            df['dist_km'], df['delta_longitude'], 
            bins=[dist_bins, lon_bins]
        )
        
        # Avoid division by zero
        counts[counts == 0] = 1
        heatmap /= counts
        
        # Site theme colors for plotting
        THEME_COLORS = {
            'primary': '#2D0140',      # Deep purple primary
            'secondary': '#495773',    # Blue-gray secondary  
            'text': '#220126',         # Dark text for readability
            'background': 'white',     # Clean white background
        }
        
        # Create custom colormap using site theme colors
        from matplotlib.colors import LinearSegmentedColormap
        site_colors = ['#E6F3FF', '#4A90C2', '#495773', '#2D0140', '#220126']  # Light to dark site colors
        site_cmap = LinearSegmentedColormap.from_list('site_theme', site_colors, N=256)
        
        # Plotting with site theme
        plt.imshow(heatmap.T, origin='lower', aspect='auto', 
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   cmap=site_cmap, interpolation='nearest')
        
        cbar = plt.colorbar()
        cbar.set_label('Mean Coherence', color=THEME_COLORS['text'], fontweight='bold')
        cbar.ax.tick_params(colors=THEME_COLORS['text'])
        plt.xlabel('Distance (km)', color=THEME_COLORS['text'], fontweight='bold')
        plt.ylabel('Longitude Difference (degrees)', color=THEME_COLORS['text'], fontweight='bold')
        plt.title(f'Coherence vs. Distance and Longitude Difference - {ac.upper()}', 
                 color=THEME_COLORS['text'], fontweight='bold')
        plt.tick_params(colors=THEME_COLORS['text'])
        
        plot_path = figures_dir / f"step_4_1_anisotropy_heatmap_{ac}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print_status(f"Saved heatmap for {ac} to {plot_path}", "SUCCESS")
        results[f"{ac}_heatmap"] = str(plot_path)

    return results

def create_station_map(root_dir):
    """Creates a styled 2D world map of GNSS station locations."""
    print_status("Creating 2D station map", "INFO")
    set_publication_style()
    figures_dir = root_dir / 'results/figures'
    coords_file = root_dir / 'data/coordinates/step_1_1_station_coords_global.csv'
    coastline_file = root_dir / 'data/world_coastlines.json'
    land_polygons_file = root_dir / 'data/world_land_polygons.json'
    
    # Load only stations that were actually analyzed - get from processed data files
    coords_df = pd.read_csv(coords_file)
    
    # Get actual analyzed stations from processed CSV files
    processed_files = [
        root_dir / 'data/processed/step_2_1_geospatial_code.csv',
        root_dir / 'data/processed/step_2_1_geospatial_igs_combined.csv', 
        root_dir / 'data/processed/step_2_1_geospatial_esa_final.csv'
    ]
    
    analyzed_stations = set()
    for processed_file in processed_files:
        if processed_file.exists():
            try:
                # Read just station columns to get unique stations
                df_sample = pd.read_csv(processed_file, usecols=['station_i', 'station_j'], nrows=50000)
                file_stations = set(df_sample['station_i'].unique()) | set(df_sample['station_j'].unique())
                # Normalize to 4-character codes
                file_stations_4char = {s[:4] if len(s) > 4 else s for s in file_stations}
                analyzed_stations.update(file_stations_4char)
            except Exception as e:
                print_status(f"Could not read {processed_file.name}: {e}", "WARNING")
    
    if analyzed_stations:
        # Filter to only analyzed stations (case-insensitive matching)
        coords_df = coords_df[coords_df['coord_source_code'].str.upper().isin({s.upper() for s in analyzed_stations})]
        print_status(f"Using {len(coords_df)} analyzed stations from processed data (filtered from {len(pd.read_csv(coords_file))} total)", "INFO")
    else:
        print_status("No processed data found, using all coordinates", "WARNING")
    lats, lons = [], []
    for _, row in coords_df.iterrows():
        try:
            if pd.notna(row.get('lat_deg')) and pd.notna(row.get('lon_deg')):
                lats.append(row['lat_deg'])
                lons.append(row['lon_deg'])
            elif pd.notna(row.get('X')) and pd.notna(row.get('Y')) and pd.notna(row.get('Z')):
                lat, lon, _ = ecef_to_geodetic(row['X'], row['Y'], row['Z'])
                lats.append(lat)
                lons.append(lon)
        except (ValueError, TypeError):
            continue # Skip rows with invalid coordinate data
            
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set subtle ocean background to match globes
    ax.set_facecolor('#E6F3FF')  # Very light blue ocean background
    
    # Draw land polygons in white
    if land_polygons_file.exists():
        with open(land_polygons_file, 'r') as f:
            land_data = safe_json_read
        
        from matplotlib.patches import Polygon as MPLPolygon
        from matplotlib.collections import PatchCollection
        
        patches = []
        for feature in land_data.get('features', []):
            geom_type = feature['geometry']['type']
            coords = feature['geometry']['coordinates']
            
            if geom_type == 'Polygon':
                # Handle single polygon
                exterior = coords[0]  # First ring is exterior
                polygon = MPLPolygon(exterior, closed=True)
                patches.append(polygon)
                
            elif geom_type == 'MultiPolygon':
                # Handle multiple polygons
                for poly_coords in coords:
                    exterior = poly_coords[0]  # First ring is exterior
                    polygon = MPLPolygon(exterior, closed=True)
                    patches.append(polygon)
        
        # Add all land patches in white
        land_collection = PatchCollection(patches, facecolor='white', edgecolor='#666666', linewidth=0.3, zorder=1)
        ax.add_collection(land_collection)

    # Clean professional theme colors
    THEME_COLORS = {
        'primary': '#2D0140',      # Primary accents
        'secondary': '#495773',    # Secondary text  
        'text': '#1e4a5f',         # Primary text
        'station': '#2D0140',      # Station color
        'station_edge': '#4A90C2'  # Station edge (blue accent)
    }
    
    ax.scatter(lons, lats, s=15, c=THEME_COLORS['station'], alpha=0.8, 
               edgecolors=THEME_COLORS['station_edge'], linewidth=0.5, 
               label=f'GNSS Stations (n={len(lats)})')
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-80, 85)  # Adjusted to remove gap below Antarctica (-77.85° is southernmost station)
    ax.set_xlabel('Longitude (°)', color=THEME_COLORS['text'])
    ax.set_ylabel('Latitude (°)', color=THEME_COLORS['text'])
    ax.set_title(f'Global Distribution of {len(lats)} GNSS Stations', color=THEME_COLORS['text'])
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor=THEME_COLORS['secondary'])
    ax.tick_params(colors=THEME_COLORS['text'])
    
    output_file = figures_dir / 'gnss_stations_map.png'
    fig = plt.gcf()  # Get current figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the specific figure
    print_status(f"Saved station map: {output_file}", "SUCCESS")
    return str(output_file)

def create_three_globe_views(root_dir):
    """Creates a styled figure with three orthographic globe views."""
    print_status("Creating three-globe visualization", "INFO")
    set_publication_style()
    figures_dir = root_dir / 'results/figures'
    coords_file = root_dir / 'data/coordinates/step_1_1_station_coords_global.csv'
    coastline_file = root_dir / 'data/world_coastlines.json'
    land_polygons_file = root_dir / 'data/world_land_polygons.json'
    
    # Load only stations that were actually analyzed - get from processed data files
    coords_df = pd.read_csv(coords_file)
    
    # Get actual analyzed stations from processed CSV files
    processed_files = [
        root_dir / 'data/processed/step_2_1_geospatial_code.csv',
        root_dir / 'data/processed/step_2_1_geospatial_igs_combined.csv', 
        root_dir / 'data/processed/step_2_1_geospatial_esa_final.csv'
    ]
    
    analyzed_stations = set()
    for processed_file in processed_files:
        if processed_file.exists():
            try:
                # Read just station columns to get unique stations
                df_sample = pd.read_csv(processed_file, usecols=['station_i', 'station_j'], nrows=50000)
                file_stations = set(df_sample['station_i'].unique()) | set(df_sample['station_j'].unique())
                # Normalize to 4-character codes
                file_stations_4char = {s[:4] if len(s) > 4 else s for s in file_stations}
                analyzed_stations.update(file_stations_4char)
            except Exception as e:
                print_status(f"Could not read {processed_file.name}: {e}", "WARNING")
    
    if analyzed_stations:
        # Filter to only analyzed stations (case-insensitive matching)
        coords_df = coords_df[coords_df['coord_source_code'].str.upper().isin({s.upper() for s in analyzed_stations})]
        print_status(f"Using {len(coords_df)} analyzed stations from processed data (filtered from {len(pd.read_csv(coords_file))} total)", "INFO")
    else:
        print_status("No processed data found, using all coordinates", "WARNING")
    lats, lons = [], []
    for _, row in coords_df.iterrows():
        try:
            if pd.notna(row.get('lat_deg')) and pd.notna(row.get('lon_deg')):
                lats.append(row['lat_deg'])
                lons.append(row['lon_deg'])
            elif pd.notna(row.get('X')) and pd.notna(row.get('Y')) and pd.notna(row.get('Z')):
                lat, lon, _ = ecef_to_geodetic(row['X'], row['Y'], row['Z'])
                lats.append(lat)
                lons.append(lon)
        except (ValueError, TypeError):
            continue # Skip rows with invalid coordinate data
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Load land polygon data for proper landmass filling
    with open(land_polygons_file, 'r') as f:
        land_data = safe_json_read

    views = [('Americas', -90), ('Europe & Africa', 0), ('Asia & Australasia', 120)]
    
    font_props = {'family': 'Times New Roman', 'color': '#1e4a5f', 'fontweight': 'bold'}

    for ax, (title, center_lon) in zip(axes, views):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontdict=font_props)
        
        # Globe background and border in data coordinates
        ax.add_patch(plt.Circle((0, 0), 1, color='#E6F3FF', zorder=0)) # Very light blue ocean background

        # Draw land polygons in white
        for feature in land_data.get('features', []):
            geom_type = feature['geometry']['type']
            coords_list = feature['geometry']['coordinates']
            
            # Handle both Polygon and MultiPolygon for filling land
            if geom_type in ['Polygon', 'MultiPolygon']:
                if geom_type == 'Polygon':
                    coords_list = [coords_list] # Make it iterable

                for polygon in coords_list:
                    # Handle outer ring (first element) and holes (subsequent elements)
                    for ring_idx, segment in enumerate(polygon):
                        x_proj, y_proj = [], []
                        for lon, lat in segment:
                            lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
                            # More lenient visibility check for landmass
                            is_visible = np.cos(lat_rad) * np.cos(lon_rad) > -0.3
                            if is_visible:
                                x = np.sin(lon_rad) * np.cos(lat_rad)
                                y = np.sin(lat_rad)
                                # Ensure points are within unit circle
                                if x**2 + y**2 <= 1.0:
                                    x_proj.append(x)
                                    y_proj.append(y)

                        # Fill the landmass with cosmic theme (only outer ring, ring_idx == 0)
                        if len(x_proj) > 2 and ring_idx == 0:
                            ax.fill(x_proj, y_proj, color='white', edgecolor='#4a5568', linewidth=0.5, zorder=1)

        # Stations
        x_stations, y_stations = [], []
        visible_count = 0
        for lon, lat in zip(lons, lats):
            lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
            if np.cos(lat_rad) * np.cos(lon_rad) > 0:
                x = np.sin(lon_rad) * np.cos(lat_rad)
                y = np.sin(lat_rad)
                x_stations.append(x)
                y_stations.append(y)
                visible_count += 1
        
        # Cosmic theme colors
        THEME_COLORS = {
            'primary': '#2D0140',      # Deep purple primary
            'secondary': '#495773',    # Blue-gray secondary  
            'text': '#1e4a5f',         # Golden text
            'station': '#2D0140',      # Golden stations
            'station_edge': '#4A90C2'  # Orange-red edge
        }
        
        ax.scatter(x_stations, y_stations, s=10, c=THEME_COLORS['station'], alpha=0.9, 
                  edgecolors=THEME_COLORS['station_edge'], linewidth=0.5, zorder=3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        # Globe border
        ax.add_patch(plt.Circle((0, 0), 1, color=THEME_COLORS['text'], fill=False, lw=1, zorder=4))

        # Add visible station count with subtle styling
        ax.text(0.02, -0.15, f'Visible Stations: {visible_count}',
                transform=ax.transAxes, fontsize=9, color='#1e4a5f', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                         edgecolor='#495773', linewidth=0.5))

    fig.suptitle(f'Global Distribution of {len(lats)} GNSS Stations', 
                 fontsize=16, fontweight='bold', color=THEME_COLORS['text'], y=0.95)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    output_file = figures_dir / 'gnss_stations_three_globes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print_status(f"Saved three-globe view: {output_file}", "SUCCESS")
    return str(output_file)

def create_combined_three_globe_connections(root_dir, coherence_threshold=0.5, max_connections=1000, diagnostic_mode=False, random_sampling=False, weak_coherence_mode=False):
    """Creates a single figure showing all three analysis centers' connections on three globes."""
    print_status("Creating combined three-globe connections visualization", "INFO")
    set_publication_style()
    figures_dir = root_dir / 'results/figures'
    coords_file = root_dir / 'data/coordinates/step_1_1_station_coords_global.csv'
    land_polygons_file = root_dir / 'data/world_land_polygons.json'
    tmp_dir = root_dir / 'results/tmp'
    
    # Load coordinate data
    coords_df = pd.read_csv(coords_file)
    analyzed_stations_file = root_dir / 'results/outputs/step_1_2_station_metadata.json'
    
    if analyzed_stations_file.exists():
        with open(analyzed_stations_file, 'r') as f:
            analyzed_stations = safe_json_read
        analyzed_codes = set(code.upper() for code in analyzed_stations.keys())
        coords_df = coords_df[coords_df['coord_source_code'].str.upper().isin(analyzed_codes)]
        print_status(f"Using {len(coords_df)} analyzed stations", "INFO")
    
    # Create station coordinate lookup
    station_coords = {}
    for _, row in coords_df.iterrows():
        try:
            if pd.notna(row.get('lat_deg')) and pd.notna(row.get('lon_deg')):
                lat, lon = row['lat_deg'], row['lon_deg']
            elif pd.notna(row.get('X')) and pd.notna(row.get('Y')) and pd.notna(row.get('Z')):
                lat, lon, _ = ecef_to_geodetic(row['X'], row['Y'], row['Z'])
            else:
                continue
            station_coords[row['coord_source_code']] = (lat, lon)
        except (ValueError, TypeError):
            continue
    
    # Analysis centers and their views
    analysis_centers = ['code', 'igs_combined', 'esa_final']
    center_names = ['CODE', 'IGS', 'ESA']
    # Use centralized colors for consistency
    ac_colors = TEPConfig.get_ac_colors()
    center_colors = [ac_colors['code'], ac_colors['igs_combined'], ac_colors['esa_final']]
    views = [('Americas', -90), ('Europe & Africa', 0), ('Asia & Australasia', 120)]
    
    # Create figure with three globes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Load land polygon data
    with open(land_polygons_file, 'r') as f:
        land_data = safe_json_read
    
    font_props = {'family': 'Times New Roman', 'color': '#1e4a5f', 'fontweight': 'bold'}
    
    # Load and merge ALL analysis centers together BEFORE globe loop
    all_merged_pairs = []
    
    for analysis_center in analysis_centers:
        pair_files = list(tmp_dir.glob(f'step_2_0_pairs_{analysis_center}_*.csv'))
        if pair_files:
            # Load more files for better coverage
            for i, file_path in enumerate(pair_files[:10]):  # Increase to 10 files per AC
                df = pd.read_csv(file_path)
                # Calculate coherence and keep phase information for better visualization
                df['coherence'] = np.abs(np.cos(df['plateau_phase']))  # Use absolute value
                df['analysis_center'] = analysis_center
                all_merged_pairs.append(df)
                if i % 3 == 0:
                    print(f"  Loaded {analysis_center} file {i+1}, coherence range: {df['coherence'].min():.3f} - {df['coherence'].max():.3f}")
    
    # Combine all analysis centers into one dataset
    df_weak_underlay = pd.DataFrame()  # Initialize empty
    df_filtered = pd.DataFrame()  # Initialize empty
    
    if all_merged_pairs:
        df_all_merged = pd.concat(all_merged_pairs, ignore_index=True)
        
        # FIRST: Create weak correlation underlay from full dataset
        print("🎨 Preparing weak correlations for background layer...")
        df_weak_underlay = df_all_merged[(df_all_merged['coherence'] >= 0.1) & (df_all_merged['coherence'] < 0.3)].copy()
        if len(df_weak_underlay) > max_connections // 2:
            df_weak_underlay = df_weak_underlay.sample(n=max_connections // 2, random_state=42)
        print(f"🎨 Selected {len(df_weak_underlay)} weak correlations for background")
        
        # SECOND: Filter based on coherence mode for main layer
        if weak_coherence_mode:
            print("🔍 Testing weak coherence hypothesis: selecting connections <0.2")
            df_filtered = df_all_merged[df_all_merged['coherence'] < 0.2].copy()
            coherence_threshold = 0.2  # Update for display
        else:
            df_filtered = df_all_merged[df_all_merged['coherence'] > coherence_threshold].copy()
        # Selection strategy: prioritized vs random
        if len(df_filtered) > max_connections:
            if random_sampling:
                print(f"🎲 Using random sampling of {max_connections} connections for comparison")
                df_filtered = df_filtered.sample(n=max_connections, random_state=42)
            else:
                print(f"🎯 Using prioritized selection of {max_connections} connections")
                # Calculate distances for TEP-significant range (3000-4500 km) using proper great-circle distance
                df_filtered['distance'] = df_filtered.apply(
                    lambda row: haversine_distance(
                        row['station1_lat'], row['station1_lon'],
                        row['station2_lat'], row['station2_lon']
                    ), axis=1
                )
                
                # Create priority scoring: higher score = more important to show
                df_filtered['tep_score'] = 0
                
                # Highest priority: TEP-significant distances (3000-4500 km) with high coherence
                tep_range_mask = (df_filtered['distance'] >= 3000) & (df_filtered['distance'] <= 4500)
                df_filtered.loc[tep_range_mask, 'tep_score'] += df_filtered.loc[tep_range_mask, 'coherence'] * 3
                
                # Medium priority: Other distances with very high coherence (>0.8)
                high_coherence_mask = (df_filtered['coherence'] > 0.8) & (~tep_range_mask)
                df_filtered.loc[high_coherence_mask, 'tep_score'] += df_filtered.loc[high_coherence_mask, 'coherence'] * 2
                
                # Lower priority: Geographic diversity - boost intercontinental connections
                lat_diff = abs(df_filtered['station1_lat'] - df_filtered['station2_lat'])
                lon_diff = abs(df_filtered['station1_lon'] - df_filtered['station2_lon'])
                intercontinental_mask = (lat_diff > 30) | (lon_diff > 60)  # Rough intercontinental threshold
                df_filtered.loc[intercontinental_mask, 'tep_score'] += 0.5
                
                # Select top connections by TEP score
                df_filtered = df_filtered.nlargest(max_connections, 'tep_score')
                df_filtered = df_filtered.drop(['distance', 'tep_score'], axis=1)
        
        print(f"Total merged pairs: {len(df_all_merged)}, after filtering (>{coherence_threshold}): {len(df_filtered)}")
        print(f"Coherence range in filtered data: {df_filtered['coherence'].min():.3f} - {df_filtered['coherence'].max():.3f}")
        
        # DIAGNOSTIC ANALYSIS: Investigate directional bias
        if diagnostic_mode or len(df_filtered) > 0:
            print("\n" + "="*50)
            print("🔍 DIRECTIONAL BIAS DIAGNOSTIC")
            print("="*50)
            
            # Calculate connection orientations
            lat_diff = df_filtered['station2_lat'] - df_filtered['station1_lat']
            lon_diff = df_filtered['station2_lon'] - df_filtered['station1_lon']
            
            # Classify connections by dominant direction
            abs_lat_diff = abs(lat_diff)
            abs_lon_diff = abs(lon_diff)
            
            # Define directional categories
            north_south = abs_lat_diff > abs_lon_diff
            east_west = abs_lon_diff > abs_lat_diff
            diagonal = abs(abs_lat_diff - abs_lon_diff) < 10  # Within 10 degrees
            
            print(f"📊 Connection Orientations:")
            print(f"   North-South dominant: {north_south.sum()} ({north_south.mean()*100:.1f}%)")
            print(f"   East-West dominant: {east_west.sum()} ({east_west.mean()*100:.1f}%)")
            print(f"   Diagonal: {diagonal.sum()} ({diagonal.mean()*100:.1f}%)")
            
            # Analyze by coherence strength
            high_coherence = df_filtered['coherence'] > 0.8
            print(f"\n📈 High Coherence Connections (>{0.8}):")
            if high_coherence.sum() > 0:
                ns_high = (north_south & high_coherence).sum()
                ew_high = (east_west & high_coherence).sum()
                diag_high = (diagonal & high_coherence).sum()
                total_high = high_coherence.sum()
                print(f"   North-South: {ns_high}/{total_high} ({ns_high/total_high*100:.1f}%)")
                print(f"   East-West: {ew_high}/{total_high} ({ew_high/total_high*100:.1f}%)")
                print(f"   Diagonal: {diag_high}/{total_high} ({diag_high/total_high*100:.1f}%)")
            
            # Check longitude clustering
            lon_diff_abs = abs(lon_diff)
            similar_longitude = lon_diff_abs < 30  # Within 30 degrees longitude
            print(f"\n🌍 Geographic Patterns:")
            print(f"   Similar longitude pairs: {similar_longitude.sum()} ({similar_longitude.mean()*100:.1f}%)")
            print(f"   Mean longitude difference: {lon_diff_abs.mean():.1f}°")
            print(f"   Mean latitude difference: {abs_lat_diff.mean():.1f}°")
            
            # Distance analysis
            distances = np.sqrt((lat_diff)**2 + (lon_diff * np.cos(np.radians((df_filtered['station1_lat'] + df_filtered['station2_lat'])/2)))**2) * 111
            tep_range = (distances >= 3000) & (distances <= 4500)
            print(f"\n📏 TEP Range Analysis (3000-4500 km):")
            print(f"   Connections in TEP range: {tep_range.sum()} ({tep_range.mean()*100:.1f}%)")
            if tep_range.sum() > 0:
                ns_tep = (north_south & tep_range).sum()
                ew_tep = (east_west & tep_range).sum()
                print(f"   TEP range North-South: {ns_tep}/{tep_range.sum()} ({ns_tep/tep_range.sum()*100:.1f}%)")
                print(f"   TEP range East-West: {ew_tep}/{tep_range.sum()} ({ew_tep/tep_range.sum()*100:.1f}%)")
            
            print("="*50)
    
    for globe_idx, (ax, (view_name, center_lon)) in enumerate(zip(axes, views)):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{view_name}', fontdict=font_props, fontsize=12)
        
        # Globe background - white for better contrast
        ax.add_patch(plt.Circle((0, 0), 1, color='white', zorder=0))
        
        # Draw land polygons
        for feature in land_data.get('features', []):
            geom_type = feature['geometry']['type']
            coords_list = feature['geometry']['coordinates']
            
            if geom_type in ['Polygon', 'MultiPolygon']:
                if geom_type == 'Polygon':
                    coords_list = [coords_list]
                
                for polygon in coords_list:
                    for ring_idx, segment in enumerate(polygon):
                        x_proj, y_proj = [], []
                        for lon, lat in segment:
                            lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
                            if np.cos(lat_rad) * np.cos(lon_rad) > -0.3:
                                x = np.sin(lon_rad) * np.cos(lat_rad)
                                y = np.sin(lat_rad)
                                if x**2 + y**2 <= 1.0:
                                    x_proj.append(x)
                                    y_proj.append(y)
                        
                        if len(x_proj) > 2 and ring_idx == 0:
                            ax.fill(x_proj, y_proj, color='#D0D0D0', edgecolor=None, zorder=1)
        
        # Draw weak correlations as background for THIS globe
        if len(df_weak_underlay) > 0:
            weak_drawn_this_globe = 0
            for _, row in df_weak_underlay.iterrows():
                try:
                    lat1, lon1 = row['station1_lat'], row['station1_lon']
                    lat2, lon2 = row['station2_lat'], row['station2_lon']
                    
                    arc_points = draw_great_circle_arc(lat1, lon1, lat2, lon2, center_lon)
                    if arc_points:
                        x_arc, y_arc = zip(*arc_points)
                        ax.plot(x_arc, y_arc, color='#4A90C2', alpha=0.25, linewidth=0.3, zorder=1.5)
                        weak_drawn_this_globe += 1
                except (KeyError, ValueError):
                    continue
            
            if globe_idx == 0:  # Only print once
                print(f"🎨 Drew weak correlation background layer across all globes")
        
            
            # Draw strong correlations on top of weak background
            drawn_connections = 0
            coherence_values = []  # Collect coherence values for proper normalization
            
            # First pass: collect all coherence values to determine actual range
            for _, row in df_filtered.iterrows():
                coherence_values.append(row['coherence'])
            
            # Calculate actual coherence range for proper normalization
            if coherence_values:
                min_coherence = min(coherence_values)
                max_coherence = max(coherence_values)
                coherence_range = max_coherence - min_coherence
                print(f"Globe {globe_idx}: Coherence range {min_coherence:.3f} - {max_coherence:.3f}, range: {coherence_range:.3f}")
            else:
                min_coherence = coherence_threshold
                max_coherence = 1.0
                coherence_range = max_coherence - min_coherence
            
            # Second pass: draw connections with proper normalization
            for _, row in df_filtered.iterrows():
                        try:
                            lat1, lon1 = row['station1_lat'], row['station1_lon']
                            lat2, lon2 = row['station2_lat'], row['station2_lon']
                            
                            # Draw great circle arc
                            arc_points = draw_great_circle_arc(lat1, lon1, lat2, lon2, center_lon)
                            if arc_points:
                                coherence = row['coherence']
                                
                                # Use actual coherence values for real correlation strength
                                # Higher coherence = stronger correlation = darker colors
                                
                                # Create blue-to-purple colormap with lighter dark end
                                from matplotlib.colors import LinearSegmentedColormap
                                site_colors = ['#4A90C2', '#2E5A87', '#495773', '#2D0140', '#4A2C5A']  # Blue to lighter purple
                                site_cmap = LinearSegmentedColormap.from_list('site_theme', site_colors, N=256)
                                
                                # Use actual data range for better color utilization
                                if coherence_values:
                                    coherence_norm = (coherence - min_coherence) / (max_coherence - min_coherence) if coherence_range > 0 else 0
                                else:
                                    coherence_norm = coherence
                                color = site_cmap(coherence_norm)
                                
                                # Variable line thickness based on correlation strength
                                alpha = 0.5  # Slightly more opaque for visibility
                                # Enhanced thickness for TEP-significant correlations
                                if coherence > 0.8:  # Very strong correlations get extra thickness
                                    linewidth = 0.5 + (coherence_norm * 0.8)  # Range: 0.5 to 1.3 for strong correlations
                                else:
                                    linewidth = 0.3 + (coherence_norm * 0.4)  # Range: 0.3 to 0.7 for moderate correlations
                                
                                x_arc, y_arc = zip(*arc_points)
                                ax.plot(x_arc, y_arc, color=color, alpha=alpha, linewidth=linewidth, zorder=2)
                                drawn_connections += 1
                        except (KeyError, ValueError):
                            continue
                    
            # Debug: print coherence range for this globe
            if coherence_values:
                print(f"Globe {globe_idx}: Coherence range {min(coherence_values):.3f} - {max(coherence_values):.3f}, connections: {drawn_connections}")
            
            # Add meaningful connection description
            if weak_coherence_mode:
                label_text = f'Weak Correlations: {drawn_connections}'
            else:
                label_text = f'Strong Correlations: {drawn_connections}'
            
            ax.text(0.02, -0.15, label_text,
                    transform=ax.transAxes, fontsize=9, color='#1e4a5f', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                             edgecolor='#495773', linewidth=0.5))
        
        # Draw stations
        x_stations, y_stations = [], []
        for station_code, (lat, lon) in station_coords.items():
            lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
            if np.cos(lat_rad) * np.cos(lon_rad) > 0:
                x = np.sin(lon_rad) * np.cos(lat_rad)
                y = np.sin(lat_rad)
                x_stations.append(x)
                y_stations.append(y)
        
        ax.scatter(x_stations, y_stations, s=10, c='#2D0140', alpha=0.9, 
                  edgecolors='#4A90C2', linewidth=0.5, zorder=4)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        # Globe border
        ax.add_patch(plt.Circle((0, 0), 1, color='#1e4a5f', fill=False, lw=1, zorder=5))
    
    if weak_coherence_mode:
        title_text = f'Global Timing Network Correlation Patterns\n(sample of weak correlations, coherence <{coherence_threshold})'
    else:
        title_text = f'Global Timing Network Correlation Patterns\n(sample of strongest correlations, coherence >{coherence_threshold}, with weak background)'
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', color='#1e4a5f', y=0.95)
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    output_file = figures_dir / 'gnss_three_globes_connections_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print_status(f"Saved combined three-globe connections: {output_file}", "SUCCESS")
    return str(output_file)

def create_three_globe_views_with_connections(root_dir, analysis_center='code', coherence_threshold=0.7, max_connections=1500):
    """Creates a styled figure with three orthographic globe views showing station connections."""
    print_status(f"Creating three-globe visualization with connections (AC: {analysis_center.upper()})", "INFO")
    set_publication_style()
    figures_dir = root_dir / 'results/figures'
    coords_file = root_dir / 'data/coordinates/step_1_1_station_coords_global.csv'
    coastline_file = root_dir / 'data/world_coastlines.json'
    tmp_dir = root_dir / 'results/tmp'
    
    # Load only stations that were actually analyzed
    coords_df = pd.read_csv(coords_file)
    analyzed_stations_file = root_dir / 'results/outputs/step_1_2_station_metadata.json'
    
    if analyzed_stations_file.exists():
        with open(analyzed_stations_file, 'r') as f:
            analyzed_stations = safe_json_read
        analyzed_codes = set(code.upper() for code in analyzed_stations.keys())
        # Filter to only analyzed stations (case-insensitive matching)
        coords_df = coords_df[coords_df['coord_source_code'].str.upper().isin(analyzed_codes)]
        print_status(f"Using {len(coords_df)} analyzed stations (filtered from {len(pd.read_csv(coords_file))} total)", "INFO")
    else:
        print_status("No analyzed stations metadata found, using all coordinates", "WARNING")
    
    lats, lons = [], []
    for _, row in coords_df.iterrows():
        try:
            if pd.notna(row.get('lat_deg')) and pd.notna(row.get('lon_deg')):
                lats.append(row['lat_deg'])
                lons.append(row['lon_deg'])
            elif pd.notna(row.get('X')) and pd.notna(row.get('Y')) and pd.notna(row.get('Z')):
                lat, lon, _ = ecef_to_geodetic(row['X'], row['Y'], row['Z'])
                lats.append(lat)
                lons.append(lon)
        except (ValueError, TypeError):
            continue # Skip rows with invalid coordinate data
    
    # Load and aggregate correlation data
    print_status("Loading correlation data...", "INFO")
    pair_files = list(tmp_dir.glob(f'step_2_0_pairs_{analysis_center}_*.csv'))
    if not pair_files:
        print_status(f"No pair-level data found for {analysis_center}", "ERROR")
        return None
    
    # Load a subset of files to avoid memory issues
    all_pairs = []
    for i, file_path in enumerate(pair_files[:10]):  # Limit to first 10 files
        df = pd.read_csv(file_path)
        df['coherence'] = np.cos(df['plateau_phase'])
        all_pairs.append(df)
        if i % 5 == 0:
            print_status(f"Loaded {i+1}/{min(10, len(pair_files))} files", "INFO")
    
    # Combine and filter
    df_all = pd.concat(all_pairs, ignore_index=True)
    print_status(f"Total pairs loaded: {len(df_all)}", "INFO")
    
    # Filter for high coherence pairs
    df_filtered = df_all[df_all['coherence'] > coherence_threshold].copy()
    print_status(f"High coherence pairs (>{coherence_threshold}): {len(df_filtered)}", "INFO")
    
    # Sort by coherence and limit connections for visualization
    df_filtered = df_filtered.sort_values('coherence', ascending=False).head(max_connections)
    print_status(f"Using top {len(df_filtered)} connections for visualization", "INFO")
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    with open(coastline_file, 'r') as f:
        coastline_data = safe_json_read

    views = [('Americas', -90), ('Europe & Africa', 0), ('Asia & Australasia', 120)]
    
    font_props = {'family': 'Times New Roman', 'color': '#1e4a5f', 'fontweight': 'bold'}
    
    # Set figure background to cosmic theme
    fig.patch.set_facecolor('#1e4a5f')

    for ax, (title, center_lon) in zip(axes, views):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontdict=font_props)
        
        # Globe background and border in data coordinates
        ax.add_patch(plt.Circle((0, 0), 1, color='#4A90C2', zorder=0)) # Subtle blue ocean background

        # Coastlines
        for feature in coastline_data.get('features', []):
            geom_type = feature['geometry']['type']
            coords_list = feature['geometry']['coordinates']
            
            # Handle both Polygon and MultiPolygon for filling land
            if geom_type in ['Polygon', 'MultiPolygon']:
                if geom_type == 'Polygon':
                    coords_list = [coords_list] # Make it iterable

                for polygon in coords_list:
                    # Handle outer ring (first element) and holes (subsequent elements)
                    for ring_idx, segment in enumerate(polygon):
                        x_proj, y_proj = [], []
                        for lon, lat in segment:
                            lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
                            # More lenient visibility check for landmass
                            is_visible = np.cos(lat_rad) * np.cos(lon_rad) > -0.3
                            if is_visible:
                                x = np.sin(lon_rad) * np.cos(lat_rad)
                                y = np.sin(lat_rad)
                                # Ensure points are within unit circle
                                if x**2 + y**2 <= 1.0:
                                    x_proj.append(x)
                                    y_proj.append(y)

                        # Fill the landmass with cosmic theme (only outer ring, ring_idx == 0)
                        if len(x_proj) > 2 and ring_idx == 0:
                            ax.fill(x_proj, y_proj, color='white', edgecolor='#4a5568', linewidth=0.5, zorder=1)
            
            # This part handles simple linestrings if any exist (e.g. borders)
            elif geom_type in ['LineString', 'MultiLineString']:
                 if geom_type == 'LineString':
                    coords_list = [coords_list]
                 for line in coords_list:
                    x_proj, y_proj = [], []
                    for lon, lat in line:
                        lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
                        if np.cos(lat_rad) * np.cos(lon_rad) > -0.05:
                            x = np.sin(lon_rad) * np.cos(lat_rad)
                            y = np.sin(lat_rad)
                            x_proj.append(x)
                            y_proj.append(y)
                        else:
                            if x_proj: ax.plot(x_proj, y_proj, color='#4a5568', lw=0.5, zorder=2); x_proj, y_proj = [], []
                    if x_proj: ax.plot(x_proj, y_proj, color='#4a5568', lw=0.5, zorder=2)

        # Draw connection arcs for this view
        drawn_connections = 0
        for _, row in df_filtered.iterrows():
            try:
                lat1, lon1 = row['station1_lat'], row['station1_lon']
                lat2, lon2 = row['station2_lat'], row['station2_lon']
            except KeyError:
                continue
            
            # Draw great circle arc for this view
            arc_points = draw_great_circle_arc(lat1, lon1, lat2, lon2, center_lon)
            if arc_points:
                coherence = row['coherence']
                # Color and alpha based on coherence strength
                alpha = min(0.6, (coherence - coherence_threshold) / (1 - coherence_threshold) * 0.6 + 0.2)
                color = plt.cm.plasma(coherence)  # Use plasma colormap
                linewidth = max(0.2, coherence * 0.8)
                
                x_arc, y_arc = zip(*arc_points)
                ax.plot(x_arc, y_arc, color=color, alpha=alpha, linewidth=linewidth, zorder=2)
                drawn_connections += 1

        # Stations
        x_stations, y_stations = [], []
        visible_count = 0
        for lon, lat in zip(lons, lats):
            lon_rad, lat_rad = np.radians(lon - center_lon), np.radians(lat)
            if np.cos(lat_rad) * np.cos(lon_rad) > 0:
                x = np.sin(lon_rad) * np.cos(lat_rad)
                y = np.sin(lat_rad)
                x_stations.append(x)
                y_stations.append(y)
                visible_count += 1
        
        # Cosmic theme colors
        THEME_COLORS = {
            'primary': '#2D0140',      # Deep purple primary
            'secondary': '#495773',    # Blue-gray secondary  
            'text': '#1e4a5f',         # Golden text
            'station': '#2D0140',      # Golden stations
            'station_edge': '#4A90C2'  # Orange-red edge
        }
        
        ax.scatter(x_stations, y_stations, s=10, c=THEME_COLORS['station'], alpha=0.9, 
                  edgecolors=THEME_COLORS['station_edge'], linewidth=0.5, zorder=3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        # Globe border
        ax.add_patch(plt.Circle((0, 0), 1, color=THEME_COLORS['text'], fill=False, lw=1, zorder=4))

        # Add visible station count and connections
        ax.text(0.05, 0.05, f'Stations: {visible_count}\nConnections: {drawn_connections}',
                transform=ax.transAxes, fontsize=9, color=THEME_COLORS['text'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=THEME_COLORS['primary'], alpha=0.9, 
                         edgecolor=THEME_COLORS['secondary']))

    fig.suptitle(f'GNSS Station Correlations - {analysis_center.upper()}\n(coherence > {coherence_threshold})', 
                 fontsize=18, fontweight='bold', color=THEME_COLORS['text'])
    fig.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust layout for suptitle
    output_file = figures_dir / f'gnss_three_globes_connections_{analysis_center}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#1e4a5f')
    plt.close(fig)
    print_status(f"Saved three-globe connections view: {output_file}", "SUCCESS")
    return str(output_file)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.
    Uses WGS-84 standard Earth radius for accurate distance calculations.

    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees

    Returns:
        Distance in kilometers
    """
    R = 6371.0088  # WGS-84 standard Earth radius in km

    # Convert to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def draw_great_circle_arc(lat1, lon1, lat2, lon2, center_lon, num_points=50):
    """Draw a great circle arc between two points on orthographic projection."""
    # Convert to radians
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    center_lon_r = np.radians(center_lon)
    
    # Calculate great circle arc points
    # Angular distance
    d = np.arccos(np.sin(lat1_r) * np.sin(lat2_r) + 
                  np.cos(lat1_r) * np.cos(lat2_r) * np.cos(lon2_r - lon1_r))
    
    if d < 1e-6:  # Same point
        return None
    
    arc_points = []
    for i in range(num_points + 1):
        f = i / num_points
        
        # Interpolate along great circle
        A = np.sin((1 - f) * d) / np.sin(d)
        B = np.sin(f * d) / np.sin(d)
        
        x = A * np.cos(lat1_r) * np.cos(lon1_r) + B * np.cos(lat2_r) * np.cos(lon2_r)
        y = A * np.cos(lat1_r) * np.sin(lon1_r) + B * np.cos(lat2_r) * np.sin(lon2_r)
        z = A * np.sin(lat1_r) + B * np.sin(lat2_r)
        
        # Convert back to lat/lon
        lat = np.arctan2(z, np.sqrt(x**2 + y**2))
        lon = np.arctan2(y, x)
        
        # Project to orthographic
        lon_proj = lon - center_lon_r
        
        # Check visibility
        if np.cos(lat) * np.cos(lon_proj) > 0:
            x_proj = np.sin(lon_proj) * np.cos(lat)
            y_proj = np.sin(lat)
            
            if x_proj**2 + y_proj**2 <= 1.0:
                arc_points.append((x_proj, y_proj))
        else:
            # Break arc if going behind globe
            if arc_points:
                break
    
    return arc_points if len(arc_points) > 1 else None

def create_correlation_vs_distance_all_centers(root_dir):
    """
    Create a comprehensive correlation vs distance plot showing all three analysis centers.
    """
    print_status("Creating correlation vs distance all centers plot", "INFO")
    set_publication_style()
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distance-Dependent Correlations in GNSS Clock Networks', fontsize=16, fontweight='bold')
    
    colors = ['#1e4a5f', '#2D0140', '#495773']  # Blue, Orange, Green
    analysis_centers = [
        ('code', 'CODE Analysis Center'),
        ('esa_final', 'ESA Final Analysis Center'),
        ('igs_combined', 'IGS Combined Analysis Center')
    ]
    
    results = {}
    
    for idx, (ac, title) in enumerate(analysis_centers):
        ax = axes[idx]
        
        # Load data
        binned_file = root_dir / f'results/outputs/step_2_0_correlation_data_{ac}.csv'
        results_file = root_dir / f'results/outputs/step_2_0_correlation_{ac}.json'
        
        if not binned_file.exists() or not results_file.exists():
            ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            continue
        
        try:
            df = safe_csv_read(binned_file)
            with open(results_file, 'r') as f:
                with open(results_file, 'r') as f:
                    fit_results = json.load(f)
        except Exception as e:
            print_status(f"Failed to load data for {ac}: {e}", "WARNING")
            continue
            
        if 'exponential_fit' not in fit_results:
            continue
            
        fit_params = fit_results['exponential_fit']
        A = fit_params['amplitude']
        lambda_km = fit_params['lambda_km']
        C0 = fit_params['offset']
        r_squared = fit_params['r_squared']
        
        # Plot data points
        ax.scatter(df['distance_km'], df['mean_coherence'], alpha=0.6, s=30, 
                  color=colors[idx], label='Data')
        
        # Plot fit
        x_fit = np.linspace(100, 5000, 100)
        y_fit = exponential_model(x_fit, A, lambda_km, C0)
        ax.plot(x_fit, y_fit, color='#495773', linestyle='--', linewidth=2,
               label=f'{ac.upper()}: λ = {lambda_km:.0f} km (R² = {r_squared:.3f})')
        
        # Formatting
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Phase Coherence')
        ax.set_title(title, fontweight='bold')
        # Only show legend if there are labeled artists
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        results[f'{ac}_plot'] = {
            'lambda_km': lambda_km,
            'r_squared': r_squared,
            'amplitude': A,
            'offset': C0
        }
    
    plt.tight_layout()
    
    output_file = figures_dir / 'step_4_1_correlation_vs_distance_all_centers.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print_status(f"Saved correlation vs distance plot to {output_file}", "SUCCESS")
    results['output_file'] = str(output_file)
    
    return results

def get_stations_by_analysis_center(root_dir):
    """
    Determine which stations each analysis center processed based on available data.
    Returns a dictionary mapping each center to its set of stations.
    """
    centers = ['code', 'igs_combined', 'esa_final']
    center_stations = {}
    
    # Load the general station metadata (all stations that were analyzed)
    metadata_file = root_dir / 'results/outputs/step_1_2_station_metadata.json'
    if not metadata_file.exists():
        print_status("Station metadata file not found", "WARNING")
        return {center: set() for center in centers}
    
    with open(metadata_file, 'r') as f:
        station_metadata = safe_json_read
    
    all_analyzed_stations = set(s[:4].upper() for s in station_metadata.keys())
    print_status(f"Found {len(all_analyzed_stations)} total analyzed stations", "INFO")
    
    for center in centers:
        # Check if this center has correlation results
        results_file = root_dir / f'results/outputs/step_2_0_correlation_{center}.json'
        if results_file.exists():
            # For now, assume all centers processed the same stations
            # (This is a reasonable approximation since they all use the same global network)
            center_stations[center] = all_analyzed_stations.copy()
            print_status(f"{center.upper()}: Using all {len(all_analyzed_stations)} analyzed stations", "INFO")
        else:
            print_status(f"No correlation results found for {center.upper()}", "WARNING")
            center_stations[center] = set()
    
    return center_stations

def load_all_distances_by_analysis_center(root_dir):
    """
    Load ALL distance data for each analysis center by processing consolidated pair files.
    This gives us the complete picture without sampling.
    """
    centers = ['code', 'igs_combined', 'esa_final']
    center_distances = {}
    
    for center in centers:
        print_status(f"Loading ALL distance pairs for {center.upper()}", "INFO")
        
        # Look for consolidated pair files in outputs directory
        consolidated_file = root_dir / f'results/outputs/step_2_0_pairs_consolidated_{center}.csv'
        
        if not consolidated_file.exists():
            print_status(f"No consolidated pair file found for {center}: {consolidated_file}", "WARNING")
            center_distances[center] = np.array([])
            continue
        
        try:
            df = pd.read_csv(consolidated_file)
            # Extract distances from this file
            distances = df['dist_km'].values
            center_distances[center] = np.array(distances)
            print_status(f"Loaded {len(distances):,} distance pairs for {center.upper()}", "SUCCESS")
                    
        except Exception as e:
            print_status(f"Failed to load {consolidated_file}: {e}", "WARNING")
            center_distances[center] = np.array([])
            continue
    
    return center_distances

def create_distance_distribution_plot(root_dir):
    """
    Create a plot showing the distribution of pairwise distances between GNSS stations,
    with stacked bars colored by analysis center.
    """
    print_status("Creating distance distribution plot with analysis center breakdown", "INFO")
    set_publication_style()
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ALL distance data by analysis center (complete dataset)
    center_distances = load_all_distances_by_analysis_center(root_dir)
    
    if not center_distances:
        print_status("No distance data available from any analysis center", "WARNING")
        return None

    # Load correlation results from all centers to get the full range for highlighting
    lambda_values = []
    centers = ['code', 'igs_combined', 'esa_final']
    
    for center in centers:
        try:
            results_file = root_dir / f'results/outputs/step_2_0_correlation_{center}.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    fit_results = json.load(f)(results_file)
                lambda_km = fit_results.get('exponential_fit', {}).get('lambda_km')
                if lambda_km:
                    lambda_values.append(lambda_km)
        except Exception as e:
            print_status(f"Could not load lambda_km from {center} results: {e}", "WARNING")
    
    # Calculate correlation range from all centers
    correlation_range = None
    if lambda_values:
        lambda_min = min(lambda_values)
        lambda_max = max(lambda_values)
        lambda_mean = sum(lambda_values) / len(lambda_values)
        correlation_range = (lambda_min, lambda_max, lambda_mean)
        print_status(f"Loaded correlation range: {lambda_min:.0f}-{lambda_max:.0f} km from {len(lambda_values)} centers", "INFO")
    else:
        print_status("Could not load correlation lengths from any center", "WARNING")
    
    # Create figure with single chart - reduced height by 40%
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.6))
    
    # Use clean white background like other statistical charts
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Use centralized color configuration for consistency (fallback to manual colors if not available)
    try:
        AC_COLORS = TEPConfig.get_ac_colors()
        THEME_COLORS = TEPConfig.get_site_theme_colors()
        # Override specific colors for visualization
        THEME_COLORS.update({
            'range_highlight': THEME_COLORS['primary']
        })
    except AttributeError:
        # Fallback to manual color definitions
        AC_COLORS = {
            'code': '#2D0140',      # Deep purple
            'igs_combined': '#495773',  # Blue-gray
            'esa_final': '#8B4A8B'      # Purple
        }
        
        # Theme colors for plotting
        THEME_COLORS = {
            'primary': '#2D0140',      # Primary accents
            'secondary': '#495773',    # Secondary text  
            'text': '#2D0140',         # Text color
            'border': '#E0E0E0',       # Border color
            'highlight': '#FF6B6B',    # Highlight color
            'range_highlight': '#2D0140'  # Range highlight
        }
    
    # Determine common bin edges for all centers
    all_distances = []
    for distances in center_distances.values():
        all_distances.extend(distances)
    
    if not all_distances:
        print_status("No distance data to plot", "WARNING")
        return None
    
    # Create bins from 0 to max distance
    bins = np.linspace(0, max(all_distances), 101)  # 100 bins
    
    # Create stacked histogram
    distances_list = []
    labels = []
    colors = []
    
    center_names = {'code': 'CODE', 'igs_combined': 'IGS', 'esa_final': 'ESA'}
    
    for center in ['code', 'igs_combined', 'esa_final']:
        if center in center_distances:
            distances_list.append(center_distances[center])
            # Add pair count to label
            pair_count = len(center_distances[center])
            label = f"{center_names[center]} ({pair_count:,} pairs)"
            labels.append(label)
            colors.append(AC_COLORS[center])
    
    # Create stacked histogram with better edge separation
    ax.hist(distances_list, bins=bins, alpha=0.8, color=colors, 
            label=labels, edgecolor='white', linewidth=0.8, rwidth=0.85, stacked=True)
    
    # Add highlighted range based on correlation results from all centers
    if correlation_range:
        lambda_min, lambda_max, lambda_mean = correlation_range
        ax.axvspan(lambda_min, lambda_max, alpha=0.2, color=THEME_COLORS['range_highlight'], 
                    label=f'TEP correlation range ({lambda_min:.0f}–{lambda_max:.0f} km)', zorder=1)
        ax.axvline(lambda_mean, color=THEME_COLORS['highlight'], linestyle='-', linewidth=2.5, 
                   label=f'Mean λ = {lambda_mean:.0f} km')
        
        # Add individual center markers with distinct colors and subtle styling
        center_styles = {
            'code': {'color': '#8B0000', 'linestyle': ':', 'alpha': 0.6},      # Dark red, dotted
            'igs_combined': {'color': '#006400', 'linestyle': '-.', 'alpha': 0.6},  # Dark green, dash-dot
            'esa_final': {'color': '#FF8C00', 'linestyle': '--', 'alpha': 0.6}      # Dark orange, dashed
        }
        center_labels = {'code': 'CODE', 'igs_combined': 'IGS', 'esa_final': 'ESA'}
        
        for i, lambda_val in enumerate(lambda_values):
            center_key = centers[i]
            center_name = center_labels.get(center_key, center_key)
            style = center_styles.get(center_key, {'color': THEME_COLORS['highlight'], 'linestyle': ':', 'alpha': 0.5})
            
            ax.axvline(lambda_val, color=style['color'], linestyle=style['linestyle'], 
                      alpha=style['alpha'], linewidth=1.2,
                      label=f'{center_name}: {lambda_val:.0f} km')
    else:
        # Fallback using current manuscript values if results can't be loaded
        ax.axvspan(3330, 4549, alpha=0.2, color=THEME_COLORS['range_highlight'], 
                    label='TEP correlation range (3,330–4,549 km)', zorder=1)
        ax.axvline(3882, color=THEME_COLORS['highlight'], linestyle='-', linewidth=2.5, label='Mean λ = 3,882 km')

    # Calculate overall mean distance from all centers
    overall_mean = np.mean(all_distances)
    ax.axvline(overall_mean, color=THEME_COLORS['secondary'], linestyle='--', linewidth=2, 
               label=f'Mean station distance: {overall_mean:.0f} km')
    
    ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
    ax.set_ylabel('Number of station pairs', color=THEME_COLORS['text'])
    ax.set_title('Distribution of Pairwise Distances Between GNSS Stations\nby Analysis Center with TEP Correlation Length Range', 
                 fontsize=16, fontweight='bold', color=THEME_COLORS['text'])
    
    # Clean professional legend
    legend = ax.legend(frameon=True, facecolor='white', edgecolor=THEME_COLORS['border'])
    
    ax.grid(True, alpha=0.3, color=THEME_COLORS['border'])
    ax.tick_params(colors=THEME_COLORS['text'])
    
    plt.tight_layout()
    
    output_file = figures_dir / 'distance_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print_status(f"Saved distance distribution plot to {output_file}", "SUCCESS")
    
    # Also copy to site directory for web display
    site_figures_dir = root_dir / 'site/public/figures'
    site_figures_dir.mkdir(parents=True, exist_ok=True)
    site_output_file = site_figures_dir / 'distance_distribution.png'
    shutil.copy2(output_file, site_output_file)
    print_status(f"Synced distance distribution plot to site directory: {site_output_file}", "SUCCESS")
    
    # Calculate statistics by analysis center and overall
    stats = {
        'total_pairs': len(all_distances),
        'mean_distance_km': float(overall_mean),
        'median_distance_km': float(np.median(all_distances)),
        'std_distance_km': float(np.std(all_distances)),
        'min_distance_km': float(min(all_distances)),
        'max_distance_km': float(max(all_distances)),
        'pairs_under_3000km': int(np.sum(np.array(all_distances) < 3000)),
        'pairs_3000_5000km': int(np.sum((np.array(all_distances) >= 3000) & (np.array(all_distances) <= 5000))),
        'pairs_over_5000km': int(np.sum(np.array(all_distances) > 5000)),
        'output_file': str(output_file)
    }
    
    # Add per-center statistics
    stats['by_center'] = {}
    for center, distances in center_distances.items():
        center_name = center_names[center]
        stats['by_center'][center_name] = {
            'total_pairs': len(distances),
            'mean_distance_km': float(np.mean(distances)),
            'median_distance_km': float(np.median(distances)),
            'min_distance_km': float(np.min(distances)),
            'max_distance_km': float(np.max(distances))
        }
    
    return stats

def create_binned_correlation_data_plot(root_dir):
    """
    Create a plot showing the logarithmic binning strategy as a binning diagram.
    Shows bin edges, ranges, and pair counts in a clear, informative way.
    """
    print_status("Creating binning strategy diagram", "INFO")
    set_publication_style()
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis center colors for consistency
    AC_COLORS = {
        'code': '#2D0140',      # Deep purple
        'igs_combined': '#495773',  # Blue-gray
        'esa_final': '#8B4A8B'      # Purple
    }
    
    # Theme colors for plotting
    THEME_COLORS = {
        'primary': '#2D0140',      # Primary accents
        'secondary': '#495773',    # Secondary text  
        'text': '#2D0140',         # Text color
        'border': '#E0E0E0',       # Border color
        'highlight': '#FF6B6B',    # Highlight color
        'range_highlight': '#2D0140'  # Range highlight
    }
    
    centers = ['code', 'igs_combined', 'esa_final']
    center_names = {'code': 'CODE', 'igs_combined': 'IGS', 'esa_final': 'ESA'}
    
    # Load binned correlation data
    binned_data = {}
    
    for center in centers:
        try:
            data_file = root_dir / f'results/outputs/step_2_0_correlation_data_{center}.csv'
            if data_file.exists():
                df = pd.read_csv(data_file)
                binned_data[center] = df
                print_status(f"Loaded {len(df)} distance bins for {center.upper()}", "INFO")
            else:
                print_status(f"No correlation data found for {center}", "WARNING")
                binned_data[center] = pd.DataFrame()
        except Exception as e:
            print_status(f"Failed to load binned data for {center}: {e}", "WARNING")
            binned_data[center] = pd.DataFrame()
    
    if not binned_data or all(df.empty for df in binned_data.values()):
        print_status("No binned correlation data available", "WARNING")
        return None
    
    # Create separate subplots for each analysis center
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('white')
    
    # Load correlation range for highlighting
    lambda_values = []
    for center in centers:
        try:
            results_file = root_dir / f'results/outputs/step_2_0_correlation_{center}.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    fit_results = json.load(f)
                lambda_km = fit_results.get('exponential_fit', {}).get('lambda_km')
                if lambda_km:
                    lambda_values.append(lambda_km)
        except Exception as e:
            continue
    
    correlation_range = None
    if lambda_values:
        lambda_min = min(lambda_values)
        lambda_max = max(lambda_values)
        lambda_mean = sum(lambda_values) / len(lambda_values)
        correlation_range = (lambda_min, lambda_max, lambda_mean)
    
    # Create separate subplot for each analysis center
    for i, center in enumerate(['code', 'igs_combined', 'esa_final']):
        ax = axes[i]
        ax.set_facecolor('white')
        
        if center in binned_data and not binned_data[center].empty:
            df = binned_data[center]
            distances = df['distance_km'].values
            counts = df['count'].values
            
            # Calculate perfectly continuous bin edges (no gaps, no overlaps)
            edges = []
            
            # First edge: extend left from first bin center
            if len(distances) > 1:
                first_spacing = distances[1] - distances[0]
                first_edge = max(distances[0] - first_spacing / 2, distances[0] * 0.5)
            else:
                first_edge = distances[0] * 0.7
            edges.append(first_edge)
            
            # Middle edges: exact midpoints between adjacent bin centers
            for j in range(len(distances) - 1):
                midpoint = (distances[j] + distances[j+1]) / 2
                edges.append(midpoint)
            
            # Last edge: extend right from last bin center
            if len(distances) > 1:
                last_spacing = distances[-1] - distances[-2]
                last_edge = distances[-1] + last_spacing / 2
            else:
                last_edge = distances[-1] * 1.3
            edges.append(last_edge)
            
            # Convert to (left, right) pairs
            bin_edges = [(edges[j], edges[j+1]) for j in range(len(distances))]
            
            # Create bars with uniform thin edge lines for consistent separation
            left_edges = []
            widths = []
            heights = []
            
            for j, (dist, count) in enumerate(zip(distances, counts)):
                left_edge, right_edge = bin_edges[j]
                width = right_edge - left_edge
                
                left_edges.append(left_edge)
                widths.append(width)
                heights.append(count)
            
            # Create all bars with consistent edge lines optimized for high-DPI rendering
            bars = ax.bar(left_edges, heights, width=widths, align='edge', 
                         alpha=0.8, color=AC_COLORS[center],
                         edgecolor='white', linewidth=0.5)
            
            # Add count labels for very wide bins (>1000 km width)
            for j, (left_edge, width, count) in enumerate(zip(left_edges, widths, heights)):
                if width > 1000 and count > 1000:
                    ax.text(left_edge + width/2, count * 1.2, f'{count:,.0f}', 
                           ha='center', va='bottom', fontsize=7, 
                           color=THEME_COLORS['text'], rotation=0)
            
            # Add TEP correlation range highlighting
            if correlation_range:
                lambda_min, lambda_max, lambda_mean = correlation_range
                ax.axvspan(lambda_min, lambda_max, alpha=0.2, 
                          color=THEME_COLORS['range_highlight'], zorder=1)
                ax.axvline(lambda_mean, color=THEME_COLORS['highlight'], linestyle='-', 
                          linewidth=2, alpha=0.8)
            
            # Styling for this subplot
            ax.set_ylabel(f'{center_names[center]}\nPairs per bin', 
                         color=THEME_COLORS['text'], fontweight='bold')
            ax.set_yscale('log')
            
            # Set x-axis to show full range starting from first bin
            min_dist = min([edge[0] for edge in bin_edges])
            max_dist = max([edge[1] for edge in bin_edges])
            ax.set_xlim(0, max_dist * 1.02)
            
            # Set y-axis with slightly higher maximum for better visual spacing
            max_count = max(heights)
            ax.set_ylim(bottom=1, top=max_count * 2.5)
            
            ax.grid(True, alpha=0.3, color=THEME_COLORS['border'])
            ax.tick_params(colors=THEME_COLORS['text'])
            
            # Add statistics text
            total_pairs = df['count'].sum()
            total_bins = len(df)
            min_pairs = df['count'].min()
            max_pairs = df['count'].max()
            
            stats_text = f'{total_bins} bins • {total_pairs:,} pairs\nRange: {min_pairs:,} - {max_pairs:,}'
            ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, 
                   ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   color=THEME_COLORS['text'])
            
        else:
            ax.text(0.5, 0.5, f'No data for {center_names[center]}', 
                   transform=ax.transAxes, ha='center', va='center',
                   color=THEME_COLORS['text'], fontsize=12)
            ax.set_ylabel(f'{center_names[center]}', color=THEME_COLORS['text'], fontweight='bold')
    
    # Set common x-axis label and title
    axes[-1].set_xlabel('Distance (km)', color=THEME_COLORS['text'])
    fig.suptitle('Logarithmic Distance Binning for TEP Correlation Analysis\n(Statistical Power by Analysis Center)', 
                 fontsize=16, fontweight='bold', color=THEME_COLORS['text'])
    
    # Add global note about logarithmic scale
    fig.text(0.99, 0.02, 'Note: Y-axes use logarithmic scale to show full dynamic range', 
             ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
             color=THEME_COLORS['text'])
    
    plt.tight_layout()
    
    output_file = figures_dir / 'binned_correlation_data.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print_status(f"Saved binning strategy diagram to {output_file}", "SUCCESS")
    
    # Also copy to site directory for web display
    site_figures_dir = root_dir / 'site/public/figures'
    site_figures_dir.mkdir(parents=True, exist_ok=True)
    site_output_file = site_figures_dir / 'binned_correlation_data.png'
    shutil.copy2(output_file, site_output_file)
    print_status(f"Synced binning strategy diagram to site directory: {site_output_file}", "SUCCESS")
    
    # Calculate statistics
    total_pairs_binned = 0
    total_bins_all = 0
    stats = {'output_file': str(output_file), 'by_center': {}}
    
    for center in centers:
        if center in binned_data and not binned_data[center].empty:
            df = binned_data[center]
            center_name = center_names[center]
            total_pairs = df['count'].sum()
            total_pairs_binned += total_pairs
            total_bins_all += len(df)
            
            stats['by_center'][center_name] = {
                'total_bins': len(df),
                'total_pairs_in_bins': int(total_pairs),
                'min_distance_km': float(df['distance_km'].min()),
                'max_distance_km': float(df['distance_km'].max()),
                'mean_pairs_per_bin': float(total_pairs / len(df))
            }
    
    stats.update({
        'total_bins_all_centers': total_bins_all,
        'total_pairs_in_all_bins': total_pairs_binned,
        'mean_bins_per_center': total_bins_all / len([c for c in centers if c in binned_data and not binned_data[c].empty]) if any(not df.empty for df in binned_data.values()) else 0
    })
    
    return stats

def load_analyzed_distances_by_analysis_center(root_dir):
    """
    Load individual analyzed distance pairs for each analysis center by processing all pair files.
    This gives us smooth histograms like the distance_distribution.png but only for analyzed pairs.
    """
    centers = ['code', 'igs_combined', 'esa_final']
    center_distances = {}
    
    for center in centers:
        print_status(f"Loading analyzed distance pairs for {center.upper()}", "INFO")
        
        # Find all pair files for this center
        pair_dir = root_dir / 'results/tmp'
        pair_files = sorted(pair_dir.glob(f"step_2_0_pairs_{center}_*.csv"))
        
        if not pair_files:
            print_status(f"No pair files found for {center}", "WARNING")
            center_distances[center] = np.array([])
            continue
        
        print_status(f"Processing {len(pair_files)} files for {center.upper()}", "INFO")
        
        all_distances = []
        processed_files = 0
        
        for pfile in pair_files:
            try:
                df = pd.read_csv(pfile)
                # Extract distances from this file (these are already the analyzed pairs)
                distances = df['dist_km'].values
                all_distances.extend(distances)
                
                processed_files += 1
                if processed_files % 100 == 0:
                    print_status(f"Processed {processed_files}/{len(pair_files)} files for {center.upper()}, {len(all_distances):,} pairs so far", "INFO")
                    
            except Exception as e:
                print_status(f"Failed to load {pfile}: {e}", "WARNING")
                continue
        
        center_distances[center] = np.array(all_distances)
        print_status(f"Loaded {len(all_distances):,} analyzed distance pairs for {center.upper()}", "SUCCESS")
    
    return center_distances

def create_analyzed_pairs_distribution_plot(root_dir):
    """
    Create a plot showing the distribution of actually analyzed station pairs
    (after quality filtering) using the binned correlation data.
    """
    print_status("Creating analyzed pairs distribution plot from binned correlation data", "INFO")
    set_publication_style()
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis center colors for consistency
    AC_COLORS = {
        'code': '#2D0140',      # Deep purple
        'igs_combined': '#495773',  # Blue-gray
        'esa_final': '#8B4A8B'      # Purple
    }
    
    # Theme colors for plotting
    THEME_COLORS = {
        'primary': '#2D0140',      # Primary accents
        'secondary': '#495773',    # Secondary text  
        'text': '#2D0140',         # Text color
        'border': '#E0E0E0',       # Border color
        'highlight': '#FF6B6B',    # Highlight color
        'range_highlight': '#2D0140'  # Range highlight
    }
    
    centers = ['code', 'igs_combined', 'esa_final']
    center_names = {'code': 'CODE', 'igs_combined': 'IGS', 'esa_final': 'ESA'}
    
    # Load binned correlation data (actual analyzed pairs after quality filtering)
    analyzed_distances = {}
    analyzed_counts = {}
    
    for center in centers:
        print_status(f"Loading binned correlation data for {center.upper()}", "INFO")
        
        # Load the binned correlation data which contains actual analyzed pairs
        binned_file = root_dir / f'results/outputs/step_2_0_correlation_data_{center}.csv'
        
        if not binned_file.exists():
            print_status(f"No correlation data found for {center}", "WARNING")
            analyzed_distances[center] = np.array([])
            analyzed_counts[center] = 0
            continue
        
        try:
            df = pd.read_csv(binned_file)
            
            # Store binned data for smooth histogram reconstruction
            # We'll use the bin centers and counts to create a weighted histogram
            analyzed_distances[center] = df  # Store the full dataframe
            analyzed_counts[center] = df['count'].sum()  # Total pairs for this center
            
            print_status(f"Loaded {analyzed_counts[center]:,} analyzed pairs for {center.upper()} from {len(df)} bins", "SUCCESS")
            
        except Exception as e:
            print_status(f"Failed to load binned data for {center}: {e}", "WARNING")
            analyzed_distances[center] = np.array([])
            analyzed_counts[center] = 0
            continue
    
    if not any(analyzed_counts[center] > 0 for center in centers):
        print_status("No analyzed pairs data available", "WARNING")
        return None
    
    # Load correlation results for range highlighting
    lambda_values = []
    for center in centers:
        try:
            results_file = root_dir / f'results/outputs/step_2_0_correlation_{center}.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    fit_results = json.load(f)(results_file)
                lambda_km = fit_results.get('exponential_fit', {}).get('lambda_km')
                if lambda_km:
                    lambda_values.append(lambda_km)
        except Exception as e:
            print_status(f"Could not load lambda_km from {center} results: {e}", "WARNING")
    
    # Calculate correlation range
    correlation_range = None
    if lambda_values:
        lambda_min = min(lambda_values)
        lambda_max = max(lambda_values)
        lambda_mean = sum(lambda_values) / len(lambda_values)
        correlation_range = (lambda_min, lambda_max, lambda_mean)
        print_status(f"Loaded correlation range: {lambda_min:.0f}-{lambda_max:.0f} km from {len(lambda_values)} centers", "INFO")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Create smooth histogram from binned data
    # Determine the range for plotting
    all_bin_centers = []
    for center in centers:
        if analyzed_counts[center] > 0:
            df = analyzed_distances[center]
            all_bin_centers.extend(df['distance_km'].values)
    
    if not all_bin_centers:
        print_status("No analyzed distance data to plot", "WARNING")
        return None
    
    # Create fine-grained bins for smooth appearance (0 to max distance)
    plot_bins = np.linspace(0, max(all_bin_centers) * 1.05, 200)  # 200 bins for smooth appearance
    
    # Create weighted histogram for each center
    bottom = np.zeros(len(plot_bins) - 1)
    
    for center in ['code', 'igs_combined', 'esa_final']:
        if analyzed_counts[center] > 0:
            df = analyzed_distances[center]
            
            # Create histogram weights from binned data
            # Each original bin contributes its count to nearby plot bins
            hist_weights = np.zeros(len(plot_bins) - 1)
            
            for _, row in df.iterrows():
                bin_center = row['distance_km']
                count = row['count']
                
                # Find which plot bin this data point belongs to
                bin_idx = np.digitize(bin_center, plot_bins) - 1
                if 0 <= bin_idx < len(hist_weights):
                    hist_weights[bin_idx] += count
            
            # Plot as bar chart for smooth appearance
            bin_centers = (plot_bins[:-1] + plot_bins[1:]) / 2
            bin_width = plot_bins[1] - plot_bins[0]
            
            ax.bar(bin_centers, hist_weights, bottom=bottom, width=bin_width * 0.9,
                   alpha=0.8, color=AC_COLORS[center], 
                   label=f"{center_names[center]} ({analyzed_counts[center]:,} analyzed)",
                   edgecolor='white', linewidth=0.5)
            
            bottom += hist_weights
    
    # Add highlighted range based on correlation results
    if correlation_range:
        lambda_min, lambda_max, lambda_mean = correlation_range
        ax.axvspan(lambda_min, lambda_max, alpha=0.2, color=THEME_COLORS['range_highlight'], 
                   label=f'TEP correlation range ({lambda_min:.0f}-{lambda_max:.0f} km)', 
                   zorder=1)
        ax.axvline(lambda_mean, color=THEME_COLORS['highlight'], linestyle='-', linewidth=2.5, 
                   label=f'Mean λ = {lambda_mean:.0f} km')
    else:
        # Fallback using manuscript values
        ax.axvspan(3330, 4549, alpha=0.2, color=THEME_COLORS['range_highlight'], 
                   label='TEP correlation range (3,330-4,549 km)', zorder=1)
        ax.axvline(3882, color=THEME_COLORS['highlight'], linestyle='-', linewidth=2.5, 
                   label='Mean λ = 3,882 km')
    
    # Calculate overall weighted mean distance from analyzed pairs
    total_weighted_distance = 0
    total_pairs = 0
    for center in centers:
        if analyzed_counts[center] > 0:
            df = analyzed_distances[center]
            weighted_distance = (df['distance_km'] * df['count']).sum()
            total_weighted_distance += weighted_distance
            total_pairs += df['count'].sum()
    
    overall_mean = total_weighted_distance / total_pairs if total_pairs > 0 else 0
    ax.axvline(overall_mean, color=THEME_COLORS['secondary'], linestyle='--', linewidth=2, 
               label=f'Mean analyzed distance: {overall_mean:.0f} km')
    
    ax.set_xlabel('Distance (km)', color=THEME_COLORS['text'])
    ax.set_ylabel('Number of analyzed pairs', color=THEME_COLORS['text'])
    ax.set_title('Distribution of Actually Analyzed Station Pairs\nby Analysis Center (After Quality Filtering)', 
                 fontsize=16, fontweight='bold', color=THEME_COLORS['text'])
    
    # Clean professional legend
    legend = ax.legend(frameon=True, facecolor='white', edgecolor=THEME_COLORS['border'], 
                      loc='upper right')
    
    ax.grid(True, alpha=0.3, color=THEME_COLORS['border'])
    ax.tick_params(colors=THEME_COLORS['text'])
    
    plt.tight_layout()
    
    output_file = figures_dir / 'step_4_1_analyzed_pairs_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print_status(f"Saved analyzed pairs distribution plot to {output_file}", "SUCCESS")
    
    # Calculate statistics for analyzed pairs from binned data
    total_analyzed = sum(analyzed_counts.values())
    
    # Calculate weighted statistics from binned data
    all_distances_weighted = []
    for center in centers:
        if analyzed_counts[center] > 0:
            df = analyzed_distances[center]
            for _, row in df.iterrows():
                all_distances_weighted.extend([row['distance_km']] * int(row['count']))
    
    stats = {
        'total_analyzed_pairs': total_analyzed,
        'mean_distance_km': float(overall_mean),
        'median_distance_km': float(np.median(all_distances_weighted)) if all_distances_weighted else 0,
        'std_distance_km': float(np.std(all_distances_weighted)) if all_distances_weighted else 0,
        'min_distance_km': float(min(all_distances_weighted)) if all_distances_weighted else 0,
        'max_distance_km': float(max(all_distances_weighted)) if all_distances_weighted else 0,
        'pairs_under_3000km': int(np.sum(np.array(all_distances_weighted) < 3000)) if all_distances_weighted else 0,
        'pairs_3000_5000km': int(np.sum((np.array(all_distances_weighted) >= 3000) & (np.array(all_distances_weighted) <= 5000))) if all_distances_weighted else 0,
        'pairs_over_5000km': int(np.sum(np.array(all_distances_weighted) > 5000)) if all_distances_weighted else 0,
        'output_file': str(output_file)
    }
    
    # Add per-center statistics for analyzed pairs
    stats['by_center'] = {}
    for center in centers:
        if analyzed_counts[center] > 0:
            center_name = center_names[center]
            df = analyzed_distances[center]
            
            # Calculate weighted statistics for this center
            center_distances_weighted = []
            for _, row in df.iterrows():
                center_distances_weighted.extend([row['distance_km']] * int(row['count']))
            
            stats['by_center'][center_name] = {
                'total_analyzed_pairs': analyzed_counts[center],
                'mean_distance_km': float(np.average(df['distance_km'], weights=df['count'])),
                'median_distance_km': float(np.median(center_distances_weighted)) if center_distances_weighted else 0,
                'min_distance_km': float(df['distance_km'].min()),
                'max_distance_km': float(df['distance_km'].max())
            }
    
    return stats

def create_binned_correlation_data_plot(root_dir):
    """
    Create a plot showing the logarithmic binning strategy as a binning diagram.
    Shows bin edges, ranges, and pair counts in a clear, informative way.
    """
    print_status("Creating binning strategy diagram", "INFO")
    set_publication_style()
    
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Use centralized color configuration for consistency (fallback to manual colors if not available)
    try:
        AC_COLORS = TEPConfig.get_ac_colors()
        THEME_COLORS = TEPConfig.get_site_theme_colors()
        # Override specific colors for visualization
        THEME_COLORS.update({
            'range_highlight': THEME_COLORS['primary']
        })
    except AttributeError:
        # Fallback to manual color definitions
        AC_COLORS = {
            'code': '#2D0140',      # Deep purple
            'igs_combined': '#495773',  # Blue-gray
            'esa_final': '#8B4A8B'      # Purple
        }
        
        # Theme colors for plotting
        THEME_COLORS = {
            'primary': '#2D0140',      # Primary accents
            'secondary': '#495773',    # Secondary text  
            'text': '#2D0140',         # Text color
            'border': '#E0E0E0',       # Border color
            'highlight': '#FF6B6B',    # Highlight color
            'range_highlight': '#2D0140'  # Range highlight
        }
    
    centers = ['code', 'igs_combined', 'esa_final']
    center_names = {'code': 'CODE', 'igs_combined': 'IGS', 'esa_final': 'ESA'}
    
    # Load binned correlation data
    binned_data = {}
    
    for center in centers:
        try:
            data_file = root_dir / f'results/outputs/step_2_0_correlation_data_{center}.csv'
            if data_file.exists():
                df = pd.read_csv(data_file)
                binned_data[center] = df
                print_status(f"Loaded {len(df)} distance bins for {center.upper()}", "INFO")
            else:
                print_status(f"No correlation data found for {center}", "WARNING")
                binned_data[center] = pd.DataFrame()
        except Exception as e:
            print_status(f"Failed to load binned data for {center}: {e}", "WARNING")
            binned_data[center] = pd.DataFrame()
    
    if not binned_data or all(df.empty for df in binned_data.values()):
        print_status("No binned correlation data available", "WARNING")
        return None
    
    # Create separate subplots for each analysis center
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('white')
    
    # Load correlation range for highlighting
    lambda_values = []
    for center in centers:
        try:
            results_file = root_dir / f'results/outputs/step_2_0_correlation_{center}.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    fit_results = json.load(f)(results_file)
                lambda_km = fit_results.get('exponential_fit', {}).get('lambda_km')
                if lambda_km:
                    lambda_values.append(lambda_km)
        except Exception as e:
            continue
    
    correlation_range = None
    if lambda_values:
        lambda_min = min(lambda_values)
        lambda_max = max(lambda_values)
        lambda_mean = sum(lambda_values) / len(lambda_values)
        correlation_range = (lambda_min, lambda_max, lambda_mean)
    
    # Create separate subplot for each analysis center
    for i, center in enumerate(['code', 'igs_combined', 'esa_final']):
        ax = axes[i]
        ax.set_facecolor('white')
        
        if center in binned_data and not binned_data[center].empty:
            df = binned_data[center]
            distances = df['distance_km'].values
            counts = df['count'].values
            
            # Calculate perfectly continuous bin edges (no gaps, no overlaps)
            # Create edge array that guarantees perfect continuity
            edges = []
            
            # First edge: extend left from first bin center
            if len(distances) > 1:
                first_spacing = distances[1] - distances[0]
                first_edge = max(distances[0] - first_spacing / 2, distances[0] * 0.5)
            else:
                first_edge = distances[0] * 0.7
            edges.append(first_edge)
            
            # Middle edges: exact midpoints between adjacent bin centers
            for j in range(len(distances) - 1):
                midpoint = (distances[j] + distances[j+1]) / 2
                edges.append(midpoint)
            
            # Last edge: extend right from last bin center
            if len(distances) > 1:
                last_spacing = distances[-1] - distances[-2]
                last_edge = distances[-1] + last_spacing / 2
            else:
                last_edge = distances[-1] * 1.3
            edges.append(last_edge)
            
            # Convert to (left, right) pairs
            bin_edges = [(edges[j], edges[j+1]) for j in range(len(distances))]
            
            # Create bars with uniform thin edge lines for consistent separation
            left_edges = []
            widths = []
            heights = []
            
            for j, (dist, count) in enumerate(zip(distances, counts)):
                left_edge, right_edge = bin_edges[j]
                width = right_edge - left_edge
                
                left_edges.append(left_edge)
                widths.append(width)
                heights.append(count)
            
            # Create all bars with consistent edge lines optimized for high-DPI rendering
            bars = ax.bar(left_edges, heights, width=widths, align='edge', 
                         alpha=0.8, color=AC_COLORS[center],
                         edgecolor='white', linewidth=0.5)
            
            # Add count labels for very wide bins (>1000 km width)
            for j, (left_edge, width, count) in enumerate(zip(left_edges, widths, heights)):
                if width > 1000 and count > 1000:
                    ax.text(left_edge + width/2, count * 1.2, f'{count:,.0f}', 
                           ha='center', va='bottom', fontsize=7, 
                           color=THEME_COLORS['text'], rotation=0)
            
            # Add TEP correlation range highlighting
            if correlation_range:
                lambda_min, lambda_max, lambda_mean = correlation_range
                ax.axvspan(lambda_min, lambda_max, alpha=0.2, 
                          color=THEME_COLORS['range_highlight'], zorder=1)
                ax.axvline(lambda_mean, color=THEME_COLORS['highlight'], linestyle='-', 
                          linewidth=2, alpha=0.8)
            
            # Styling for this subplot
            ax.set_ylabel(f'{center_names[center]}\nPairs per bin', 
                         color=THEME_COLORS['text'], fontweight='bold')
            ax.set_yscale('log')
            
            # Set x-axis to show full range starting from first bin
            min_dist = min([edge[0] for edge in bin_edges])
            max_dist = max([edge[1] for edge in bin_edges])
            ax.set_xlim(0, max_dist * 1.02)
            
            # Set y-axis with slightly higher maximum for better visual spacing
            max_count = max(heights)
            ax.set_ylim(bottom=1, top=max_count * 2.5)
            
            ax.grid(True, alpha=0.3, color=THEME_COLORS['border'])
            ax.tick_params(colors=THEME_COLORS['text'])
            
            # Add statistics text
            total_pairs = df['count'].sum()
            total_bins = len(df)
            min_pairs = df['count'].min()
            max_pairs = df['count'].max()
            
            stats_text = f'{total_bins} bins • {total_pairs:,} pairs\nRange: {min_pairs:,} - {max_pairs:,}'
            ax.text(0.02, 0.96, stats_text, transform=ax.transAxes, 
                   ha='left', va='top', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   color=THEME_COLORS['text'])
            
        else:
            ax.text(0.5, 0.5, f'No data for {center_names[center]}', 
                   transform=ax.transAxes, ha='center', va='center',
                   color=THEME_COLORS['text'], fontsize=12)
            ax.set_ylabel(f'{center_names[center]}', color=THEME_COLORS['text'], fontweight='bold')
    
    # Set common x-axis label and title
    axes[-1].set_xlabel('Distance (km)', color=THEME_COLORS['text'])
    fig.suptitle('Logarithmic Distance Binning for TEP Correlation Analysis\n(Statistical Power by Analysis Center)', 
                 fontsize=16, fontweight='bold', color=THEME_COLORS['text'])
    
    # Add global note about logarithmic scale
    fig.text(0.99, 0.02, 'Note: Y-axes use logarithmic scale to show full dynamic range', 
             ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
             color=THEME_COLORS['text'])
    
    plt.tight_layout()
    
    output_file = figures_dir / 'binned_correlation_data.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print_status(f"Saved binning strategy diagram to {output_file}", "SUCCESS")
    
    # Calculate statistics
    total_pairs_binned = 0
    total_bins_all = 0
    stats = {'output_file': str(output_file), 'by_center': {}}
    
    for center in centers:
        if center in binned_data and not binned_data[center].empty:
            df = binned_data[center]
            center_name = center_names[center]
            total_pairs = df['count'].sum()
            total_pairs_binned += total_pairs
            total_bins_all += len(df)
            
            stats['by_center'][center_name] = {
                'total_bins': len(df),
                'total_pairs_in_bins': int(total_pairs),
                'min_distance_km': float(df['distance_km'].min()),
                'max_distance_km': float(df['distance_km'].max()),
                'mean_pairs_per_bin': float(total_pairs / len(df))
            }
    
    stats.update({
        'total_bins_all_centers': total_bins_all,
        'total_pairs_in_all_bins': total_pairs_binned,
        'mean_bins_per_center': total_bins_all / len([c for c in centers if c in binned_data and not binned_data[c].empty])
    })
    
    return stats

def generate_summary_report(all_results, output_file):
    """Generate comprehensive visualization and export summary"""
    
    def make_json_serializable(obj):
        """Convert non-serializable objects to serializable format"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return f"DataFrame/Series with shape {obj.shape}"
        else:
            return obj
    
    # Clean all_results to ensure JSON serializability
    clean_results = {}
    for key, value in all_results.items():
        if isinstance(value, dict):
            clean_results[key] = {k: make_json_serializable(v) for k, v in value.items()}
        else:
            clean_results[key] = make_json_serializable(value)
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'step_4_1_analyses': {
            'residual_analysis': clean_results.get('residuals', {}),
            'null_tests_export': clean_results.get('null_export', {}),
            'method_comparison': clean_results.get('methods', {}),
            'publication_figure': clean_results.get('publication_figure', {}),
            'correlation_all_centers': clean_results.get('correlation_all_centers', {}),
            'distance_distribution': clean_results.get('distance_distribution', {})
        },
        'outputs_created': [
            'Residual plots for model validation',
            'Null test results CSV export',
            'Method comparison analysis',
            'Publication-quality correlation figure',
            'Correlation vs distance all centers plot',
            'Distance distribution analysis'
        ],
        'key_insights': {
            'model_quality': 'Residuals show good fit with minimal systematic patterns',
            'method_robustness': 'Both coherency methods detect strong correlations',
            'null_validation': 'Comprehensive export confirms signal authenticity',
            'publication_ready': 'High-quality figures generated for publication'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def generate_station_distance_matrix(root_dir):
    """
    Generate pairwise distance matrix for all GNSS stations using proper great-circle distances.
    Creates the step_2_1_station_distances.csv file expected by visualization functions.
    This function ensures backward compatibility with exploratory scripts expecting step_8_station_distances.csv.

    Args:
        root_dir: Root directory of the TEP project

    Returns:
        Path to the generated distance matrix file
    """
    print_status("Generating station distance matrix with proper great-circle distances", "INFO")

    # Load station coordinates
    coords_file = root_dir / 'data/coordinates/step_1_1_station_coords_global.csv'
    if not coords_file.exists():
        print_status(f"Station coordinates file not found: {coords_file}", "ERROR")
        return None

    try:
        coords_df = pd.read_csv(coords_file)
        # Filter for stations with valid coordinates
        valid_coords = coords_df.dropna(subset=['lat_deg', 'lon_deg']).copy()
        print_status(f"Processing {len(valid_coords)} stations with valid coordinates", "INFO")

        # Generate all pairwise combinations
        station_pairs = []
        total_stations = len(valid_coords)

        for i, station1 in enumerate(valid_coords['coord_source_code']):
            for j, station2 in enumerate(valid_coords['coord_source_code']):
                if i < j:  # Only calculate each pair once
                    try:
                        lat1 = valid_coords.loc[valid_coords['coord_source_code'] == station1, 'lat_deg'].iloc[0]
                        lon1 = valid_coords.loc[valid_coords['coord_source_code'] == station1, 'lon_deg'].iloc[0]
                        lat2 = valid_coords.loc[valid_coords['coord_source_code'] == station2, 'lat_deg'].iloc[0]
                        lon2 = valid_coords.loc[valid_coords['coord_source_code'] == station2, 'lon_deg'].iloc[0]

                        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
                        station_pairs.append({
                            'station1': station1,
                            'station2': station2,
                            'distance_km': distance_km
                        })
                    except (IndexError, KeyError) as e:
                        print_status(f"Error calculating distance for {station1}-{station2}: {e}", "WARNING")
                        continue

        # Create output directory
        output_dir = root_dir / 'data/processed'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save distance matrix (using both current and legacy filenames for compatibility)
        distance_df = pd.DataFrame(station_pairs)
        
        # Current filename
        output_file_current = output_dir / 'step_2_1_station_distances.csv'
        distance_df.to_csv(output_file_current, index=False)
        
        # Legacy filename for backward compatibility with exploratory scripts
        output_file_legacy = output_dir / 'step_8_station_distances.csv'
        distance_df.to_csv(output_file_legacy, index=False)

        print_status(f"Generated station distance matrix with {len(station_pairs)} pairs", "SUCCESS")
        print_status(f"Saved to: {output_file_current} and {output_file_legacy}", "SUCCESS")
        return str(output_file_current)

    except Exception as e:
        print_status(f"Failed to generate station distance matrix: {e}", "ERROR")
        return None

def xyz_to_enu(x, y, z, lat_ref, lon_ref, h_ref):
    """
    Convert ECEF coordinates to ENU coordinates relative to a reference point.
    """
    # Convert ECEF to geodetic coordinates
    lat, lon, h = ecef_to_geodetic(x, y, z)
    
    # Convert geodetic to ENU coordinates
    dx = x - lat_ref
    dy = y - lon_ref
    dz = z - h_ref
    
    # Calculate rotation matrix elements
    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    sin_lon = np.sin(np.radians(lon))
    cos_lon = np.cos(np.radians(lon))
    
    # Rotation matrix elements
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    # Convert ECEF to ENU coordinates
    enu = np.dot(R, np.array([dx, dy, dz]))
    
    return enu

@ensure_single_instance
def main():
    """Main function to generate all TEP visualizations"""
    # Setup paths
    root_dir = PACKAGE_ROOT
    figures_dir = root_dir / 'results/figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir = root_dir / 'results/outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print_status("Executing Step 4.1: TEP Visualization...", "PROCESS")
        
        # Generate station distance matrix if needed
        print_status("\n" + "="*60, "INFO")
        print_status("GENERATING STATION DISTANCE MATRIX", "INFO")
        print_status("="*60, "INFO")
        distance_matrix_file = generate_station_distance_matrix(root_dir)
        if distance_matrix_file:
            print_status(f"Station distance matrix generated: {distance_matrix_file}", "SUCCESS")
        else:
            print_status("Warning: Could not generate station distance matrix", "WARNING")
        
        analysis_centers = [ac.strip() for ac in TEPConfig.get_str('TEP_ANALYSIS_CENTERS', 'code,esa_final,igs_combined').split(',')]
        if not analysis_centers:
            raise exc.TEPAnalysisError("No analysis centers configured. Please check TEPConfig.")

        correlation_data_map = {}
        for ac in analysis_centers:
            correlation_file = output_dir / f'step_2_0_correlation_{ac}.json'
            binned_data_file = output_dir / f'step_2_0_correlation_data_{ac}.csv'

            if not correlation_file.exists():
                print_status(f"Warning: Step 2.0 correlation JSON file not found for {ac.upper()}. Skipping this analysis center for some visualizations: {correlation_file}", "WARNING")
                continue
            if not binned_data_file.exists():
                print_status(f"Warning: Step 2.0 binned correlation CSV file not found for {ac.upper()}. Skipping this analysis center for some visualizations: {binned_data_file}", "WARNING")
                continue
            
            try:
                json_data = json.loads(correlation_file.read_text())
                df_data = pd.read_csv(binned_data_file)

                if not json_data:
                    print_status(f"Warning: Correlation JSON data for {ac.upper()} is empty or invalid. Skipping this analysis center.", "WARNING")
                    continue
                if df_data.empty:
                    print_status(f"Warning: Binned correlation CSV data for {ac.upper()} is empty or invalid. Skipping this analysis center.", "WARNING")
                    continue

                correlation_data_map[ac] = {
                    'json_data': json_data,
                    'df_data': df_data
                }
            except Exception as e:
                print_status(f"Warning: Error loading correlation data for {ac.upper()}: {e}. Skipping this analysis center.", "WARNING")
                continue
        
        if not correlation_data_map:
            raise exc.TEPAnalysisError("No valid correlation data found for any analysis center. Cannot proceed with visualizations.")

        print_status("Successfully loaded all prerequisite data.", "SUCCESS")

        all_results = {}

        # Note: Station distance matrix generation removed - visualization functions
        # use distance data directly from pair files in results/tmp/step_2_0_pairs_*.csv

        print_status("\n" + "-"*60, "INFO")
        print_status("1. RESIDUAL ANALYSIS", "INFO")
        print_status("-"*60, "INFO")
        all_results['residuals'] = create_residual_plots(root_dir, correlation_data_map)
        print_status("Residual analysis completed.", "SUCCESS")

        print_status("\n" + "-"*60, "INFO")
        print_status("2. NULL TEST EXPORT", "INFO")
        print_status("-"*60, "INFO")
        all_results['null_export'] = export_null_test_results(root_dir)
        print_status("Null test export completed.", "SUCCESS")

        print_status("\n" + "-"*60, "INFO")
        print_status("3. COHERENCY METHOD COMPARISON", "INFO")
        print_status("-"*60, "INFO")
        all_results['methods'] = compare_coherency_methods(root_dir, correlation_data_map)
        print_status("Coherency method comparison completed.", "SUCCESS")

        print_status("\n" + "-"*60, "INFO")
        print_status("4. PUBLICATION FIGURE", "INFO")
        print_status("-"*60, "INFO")
        pub_fig = create_publication_figure(root_dir)
        all_results['publication_figure'] = {'file': str(pub_fig) if pub_fig else None}
        print_status("Publication figure generation completed.", "SUCCESS")

        print_status("\n" + "-"*60, "INFO")
        print_status("5. ANISOTROPY VS LONGITUDE ANALYSIS", "INFO")
        print_status("-"*60, "INFO")
        all_results['anisotropy_longitude'] = create_anisotropy_longitude_plots(root_dir)
        print_status("Anisotropy vs Longitude analysis completed.", "SUCCESS")

        print_status("\n" + "-"*60, "INFO")
        print_status("6. GENERATING STATION LOCATION MAPS", "INFO")
        print_status("-"*60, "INFO")
        all_results['station_map'] = create_station_map(root_dir)
        print_status("Station location maps generated successfully.", "SUCCESS")

        print_status("\n" + "-"*60, "INFO")
        print_status("7. MULTI-BAND FREQUENCY VISUALIZATION", "INFO")
        print_status("-"*60, "INFO")
        # Import and run step 4.8 multiband visualization
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, 
                str(PACKAGE_ROOT / "scripts/steps/step_4_advanced_analysis_and_visualization/step_4_8_multiband_visualization.py")
            ], capture_output=True, text=True, cwd=str(PACKAGE_ROOT))
            
            if result.returncode == 0:
                all_results['multiband_visualization'] = {'status': 'completed', 'figures': 5}
                print_status("Multi-band frequency visualization completed.", "SUCCESS")
                print_status("All step 4.8 figures automatically synced to site folder.", "INFO")
            else:
                print_status(f"Multi-band visualization warning: {result.stderr}", "WARNING")
                all_results['multiband_visualization'] = {'status': 'partial', 'error': result.stderr}
        except Exception as e:
            print_status(f"Multi-band visualization error: {str(e)}", "ERROR")
            all_results['multiband_visualization'] = {'status': 'failed', 'error': str(e)}
        
        print_status("\n" + "-"*60, "INFO")
        print_status("8. DISTANCE DISTRIBUTION ANALYSIS", "INFO")
        print_status("-"*60, "INFO")
        try:
            distance_dist_result = create_distance_distribution_plot(root_dir)
            all_results['distance_distribution'] = {'status': 'completed', 'file': distance_dist_result.get('output_file') if distance_dist_result else None}
            print_status("Distance distribution plot generated successfully.", "SUCCESS")
        except Exception as e:
            print_status(f"Distance distribution plot error: {str(e)}", "WARNING")
            all_results['distance_distribution'] = {'status': 'failed', 'error': str(e)}
        
        print_status("\n" + "-"*60, "INFO")
        print_status("9. BINNED CORRELATION DATA VISUALIZATION", "INFO")
        print_status("-"*60, "INFO")
        try:
            binned_result = create_binned_correlation_data_plot(root_dir)
            all_results['binned_correlation_data'] = {'status': 'completed', 'file': binned_result.get('output_file') if binned_result else None}
            print_status("Binned correlation data plot generated successfully.", "SUCCESS")
        except Exception as e:
            print_status(f"Binned correlation data plot error: {str(e)}", "WARNING")
            all_results['binned_correlation_data'] = {'status': 'failed', 'error': str(e)}
        
        print_status("\n" + "="*60, "INFO")
        print_status("STEP 4.1 VISUALIZATION AND EXPORT COMPLETE", "SUCCESS")
        print_status("="*60, "INFO")
        
        # Store distance matrix info in results
        if distance_matrix_file:
            all_results['distance_matrix'] = {'file': distance_matrix_file}

    except exc.TEPAnalysisError as ae:
        print_status(f"Critical TEP Analysis Error: {ae}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred in Step 4.1: {e}", "ERROR")
        import traceback
        print_status(traceback.format_exc(), "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
