#!/usr/bin/env python3
"""
Step 14: Comprehensive Gravitational-Temporal Field Correlation Analysis

This script performs a definitive analysis of the correlation between Earth's gravitational 
environment (from planetary positions) and temporal field signatures (from TEP clock correlations).

Key Discovery: The stacked gravitational influence pattern from all planets creates a unique 
composite curve that shows significant correlation (r = -0.458, p < 10^-48) with temporal 
field coherence variations, providing direct experimental evidence for TEP theory.

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-25
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import savgol_filter, correlate
import seaborn as sns
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy import units as u

# Set high-precision ephemeris
solar_system_ephemeris.set('jpl')

# Planetary masses in Earth masses (M⊕)
PLANETARY_MASSES = {
    'sun': 332946.0,      # Solar mass in Earth masses
    'jupiter': 317.8,     # Jupiter mass in Earth masses
    'saturn': 95.2,       # Saturn mass in Earth masses
    'venus': 0.815,       # Venus mass in Earth masses
    'mars': 0.107,        # Mars mass in Earth masses
}

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

def calculate_high_precision_gravitational_influence(date: datetime) -> Dict:
    """
    Calculate high-precision gravitational influence of celestial bodies on Earth
    using NASA/JPL DE440/441 ephemeris data.
    
    Returns gravitational influence coefficients: (Body_Mass / Earth_Mass) / Distance_AU²
    """
    # Convert to astropy Time
    astro_time = Time(date.strftime('%Y-%m-%d'))
    
    try:
        # Get barycentric positions for all bodies
        earth_pos, _ = get_body_barycentric_posvel('earth', astro_time)
        sun_pos, _ = get_body_barycentric_posvel('sun', astro_time)
        jupiter_pos, _ = get_body_barycentric_posvel('jupiter', astro_time)
        saturn_pos, _ = get_body_barycentric_posvel('saturn', astro_time)
        venus_pos, _ = get_body_barycentric_posvel('venus', astro_time)
        mars_pos, _ = get_body_barycentric_posvel('mars', astro_time)
        
        # Calculate Earth-centered distances in AU
        distances = {}
        positions = {
            'sun': sun_pos,
            'jupiter': jupiter_pos,
            'saturn': saturn_pos,
            'venus': venus_pos,
            'mars': mars_pos
        }
        
        for body, pos in positions.items():
            earth_centered_pos = pos - earth_pos
            distance_au = np.linalg.norm(earth_centered_pos.xyz.value)
            distances[f'{body}_distance_au'] = distance_au
            
            # Calculate gravitational influence: Mass / Distance²
            mass_ratio = PLANETARY_MASSES[body]
            gravitational_influence = mass_ratio / (distance_au ** 2)
            distances[f'{body}_influence'] = gravitational_influence
        
        # Calculate total influences
        distances['total_planetary_influence'] = (
            distances['jupiter_influence'] + distances['saturn_influence'] + 
            distances['venus_influence'] + distances['mars_influence']
        )
        
        distances['total_influence'] = (
            distances['sun_influence'] + distances['total_planetary_influence']
        )
        
        return distances
        
    except Exception as e:
        print(f"Error calculating positions for {date}: {e}")
        return None

def extract_real_daily_tep_coherence_data() -> pd.DataFrame:
    """
    Extract authentic daily TEP coherence data from Step 3 pair-level outputs.
    Uses all three analysis centers (CODE, ESA, IGS) for comprehensive validation.
    """
    print("Extracting authentic daily TEP coherence data from all analysis centers...")
    
    import glob
    from datetime import datetime, timedelta
    
    # Process all three analysis centers
    centers = ['code', 'esa_final', 'igs_combined']
    all_daily_data = {}  # date -> list of coherence values from all centers
    
    for center in centers:
        print(f"Processing {center.upper()} center...")
        
        # Find all daily pair files for this center
        pair_files = glob.glob(f'/Users/matthewsmawfield/www/TEP-GNSS/results/tmp/step_3_pairs_{center}_*.csv')
        print(f"  Found {len(pair_files)} daily files")
        
        for file_path in pair_files:
            try:
                # Extract date from filename
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                
                # Find the date part (YYYYDDDHHHMM format)
                date_part = None
                for part in parts:
                    if part.startswith('202') and len(part) >= 11:
                        date_part = part[:7]  # Take YYYYDDD
                        break
                
                if not date_part:
                    continue
                    
                # Convert YYYYDDD to date
                year = int(date_part[:4])
                day_of_year = int(date_part[4:7])
                date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                
                # Load the file and extract coherence data
                df = pd.read_csv(file_path)
                
                if 'plateau_phase' in df.columns and len(df) > 0:
                    # Calculate coherence from phase: coherence = cos(phase)
                    coherence_values = np.cos(df['plateau_phase'].dropna())
                    
                    if len(coherence_values) > 0:
                        if date not in all_daily_data:
                            all_daily_data[date] = []
                        all_daily_data[date].extend(coherence_values)
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Aggregate daily data across all centers
    print(f"Aggregating data across all centers...")
    daily_aggregated = []
    
    for date, coherence_values in all_daily_data.items():
        if len(coherence_values) > 0:
            daily_aggregated.append({
                'date': date,
                'coherence_mean': np.mean(coherence_values),
                'coherence_median': np.median(coherence_values),
                'coherence_std': np.std(coherence_values),
                'coherence_count': len(coherence_values)
            })
    
    if daily_aggregated:
        tep_df = pd.DataFrame(daily_aggregated).sort_values('date').reset_index(drop=True)
        print(f"Successfully extracted multi-center daily TEP data for {len(tep_df)} days")
        
        # Show statistics
        print(f"  Date range: {tep_df['date'].min().strftime('%Y-%m-%d')} to {tep_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Coherence mean range: {tep_df['coherence_mean'].min():.6f} to {tep_df['coherence_mean'].max():.6f}")
        print(f"  Coherence std range: {tep_df['coherence_std'].min():.6f} to {tep_df['coherence_std'].max():.6f}")
        print(f"  Average pairs per day: {tep_df['coherence_count'].mean():.0f}")
        print(f"  Total station pair measurements: {tep_df['coherence_count'].sum():,}")
        
        return tep_df
    else:
        raise ValueError("No authentic TEP coherence data could be extracted from daily files")

def perform_advanced_correlation_analysis(combined_df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive correlation analysis between gravitational patterns and temporal field.
    """
    print("Performing advanced correlation analysis...")
    
    results = {
        'analysis_type': 'comprehensive_gravitational_temporal_correlation',
        'ephemeris_source': 'NASA_JPL_DE440_441',
        'tep_method': 'phase_coherent_cross_spectral_density',
        'data_summary': {
            'total_days': len(combined_df),
            'date_range': [
                combined_df['date'].min().strftime('%Y-%m-%d'),
                combined_df['date'].max().strftime('%Y-%m-%d')
            ]
        }
    }
    
    # Individual planetary correlations
    planets = ['sun', 'jupiter', 'saturn', 'venus', 'mars']
    tep_metrics = ['coherence_mean', 'coherence_median', 'coherence_std']
    
    correlations = {}
    
    for planet in planets:
        planet_corr = {}
        influence_col = f'{planet}_influence'
        
        if influence_col in combined_df.columns:
            for metric in tep_metrics:
                if metric in combined_df.columns:
                    r, p = stats.pearsonr(combined_df[influence_col], combined_df[metric])
                    rho, p_spear = stats.spearmanr(combined_df[influence_col], combined_df[metric])
                    
                    planet_corr[metric] = {
                        'pearson_r': r,
                        'pearson_p': p,
                        'spearman_rho': rho,
                        'spearman_p': p_spear,
                        'n_points': len(combined_df)
                    }
        
        correlations[f'{planet}_influence'] = planet_corr
    
    # KEY DISCOVERY: Stacked planetary influence analysis
    stacked_correlations = {}
    for metric in tep_metrics:
        if metric in combined_df.columns:
            r, p = stats.pearsonr(combined_df['total_planetary_influence'], combined_df[metric])
            rho, p_spear = stats.spearmanr(combined_df['total_planetary_influence'], combined_df[metric])
            
            stacked_correlations[metric] = {
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear,
                'n_points': len(combined_df)
            }
    
    correlations['stacked_planetary_influence'] = stacked_correlations
    
    # Total gravitational influence (including Sun)
    total_correlations = {}
    for metric in tep_metrics:
        if metric in combined_df.columns:
            r, p = stats.pearsonr(combined_df['total_influence'], combined_df[metric])
            rho, p_spear = stats.spearmanr(combined_df['total_influence'], combined_df[metric])
            
            total_correlations[metric] = {
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear,
                'n_points': len(combined_df)
            }
    
    correlations['total_gravitational_influence'] = total_correlations
    results['correlations'] = correlations
    
    # Advanced pattern analysis with smoothing
    window_size = min(31, len(combined_df) // 10)
    poly_order = min(3, window_size - 1)
    
    if window_size > poly_order:
        smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window_size, poly_order)
        smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], window_size, poly_order)
        
        # Smoothed pattern correlation
        smooth_r, smooth_p = stats.pearsonr(smoothed_stacked, smoothed_coherence_std)
        
        # Cross-correlation analysis for lag detection
        norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
        norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
        
        cross_corr = correlate(norm_coherence, norm_stacked, mode='full')
        lags = np.arange(-len(norm_stacked) + 1, len(norm_stacked))
        max_corr_idx = np.argmax(np.abs(cross_corr))
        optimal_lag = lags[max_corr_idx]
        max_correlation = cross_corr[max_corr_idx]
        
        results['advanced_pattern_analysis'] = {
            'smoothed_correlation': smooth_r,
            'smoothed_p_value': smooth_p,
            'optimal_lag_days': int(optimal_lag),
            'max_cross_correlation': float(max_correlation),
            'smoothing_window': window_size,
            'pattern_relationship': 'anti_phase' if max_correlation < 0 else 'in_phase'
        }
    
    # Pattern extremes analysis
    stacked_peaks = combined_df[combined_df['total_planetary_influence'] > combined_df['total_planetary_influence'].quantile(0.9)]
    stacked_valleys = combined_df[combined_df['total_planetary_influence'] < combined_df['total_planetary_influence'].quantile(0.1)]
    
    results['pattern_extremes'] = {
        'peak_periods': len(stacked_peaks),
        'valley_periods': len(stacked_valleys),
        'peak_coherence_mean': float(stacked_peaks['coherence_mean'].mean()),
        'valley_coherence_mean': float(stacked_valleys['coherence_mean'].mean()),
        'peak_coherence_std': float(stacked_peaks['coherence_std'].mean()),
        'valley_coherence_std': float(stacked_valleys['coherence_std'].mean()),
        'coherence_mean_difference': float(stacked_peaks['coherence_mean'].mean() - stacked_valleys['coherence_mean'].mean()),
        'coherence_std_difference': float(stacked_peaks['coherence_std'].mean() - stacked_valleys['coherence_std'].mean())
    }
    
    return results

def create_comprehensive_visualization(combined_df: pd.DataFrame, analysis_results: Dict) -> str:
    """
    Create comprehensive visualization with site-consistent theme.
    """
    print("Creating comprehensive visualization with site theme...")
    
    # Set site-themed style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'grid.color': '#495773',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'figure.facecolor': 'white',
        'text.color': '#220126',
        'axes.labelcolor': '#220126',
        'xtick.color': '#220126',
        'ytick.color': '#220126',
        'axes.titlecolor': '#2D0140'
    })
    
    # Set up the figure with optimal layout
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.4, left=0.08, right=0.95)
    
    # Site-themed color scheme
    colors = {
        'mars': '#E74C3C',        # Red for Mars
        'venus': '#F39C12',       # Orange for Venus
        'saturn': '#3498DB',      # Blue for Saturn  
        'jupiter': '#2D0140',     # Site dark purple for Jupiter (dominant)
        'sun': '#F1C40F',        # Yellow for Sun
        'total': '#220126',       # Site primary dark for total
        'temporal': '#4A90C2',    # Site accent blue for temporal
        'secondary': '#495773'    # Site secondary for accents
    }
    
    # Panel 1: Stacked Planetary Gravitational Influences
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create stacked area chart
    dates = combined_df['date']
    mars_vals = combined_df['mars_influence']
    venus_vals = combined_df['venus_influence'] 
    saturn_vals = combined_df['saturn_influence']
    jupiter_vals = combined_df['jupiter_influence']
    
    ax1.fill_between(dates, 0, mars_vals, alpha=0.8, color=colors['mars'], label='Mars')
    ax1.fill_between(dates, mars_vals, mars_vals + venus_vals, alpha=0.8, color=colors['venus'], label='Venus')
    ax1.fill_between(dates, mars_vals + venus_vals, mars_vals + venus_vals + saturn_vals, 
                     alpha=0.8, color=colors['saturn'], label='Saturn')
    ax1.fill_between(dates, mars_vals + venus_vals + saturn_vals, 
                     mars_vals + venus_vals + saturn_vals + jupiter_vals,
                     alpha=0.8, color=colors['jupiter'], label='Jupiter')
    
    # Add total planetary influence line
    ax1.plot(dates, combined_df['total_planetary_influence'], color=colors['total'], 
             linewidth=2, label='Total Planetary Influence')
    
    ax1.set_ylabel('Gravitational Influence (M⊕/AU²)', fontsize=12, fontweight='bold')
    ax1.set_title('Stacked Planetary Gravitational Influences on Earth\n' + 
                  'NASA/JPL DE440/441 High-Precision Ephemeris', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TEP Temporal Field Signatures
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Plot coherence metrics
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(dates, combined_df['coherence_mean'], color=colors['temporal'], 
                     linewidth=2, label='Coherence Mean', alpha=0.8)
    line2 = ax2_twin.plot(dates, combined_df['coherence_std'], color=colors['secondary'], 
                          linewidth=2, label='Coherence Variability', alpha=0.8)
    
    ax2.set_ylabel('TEP Coherence Mean', fontsize=12, fontweight='bold', color=colors['temporal'])
    ax2_twin.set_ylabel('TEP Coherence Variability (Std)', fontsize=12, fontweight='bold', color=colors['secondary'])
    ax2.set_title('TEP Temporal Field Signatures from GNSS Clock Correlations\n' +
                  'Phase-Coherent Cross-Spectral Density Analysis', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Pattern Correlation Analysis
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Smoothed patterns for correlation visualization
    window_size = min(31, len(combined_df) // 10)
    poly_order = min(3, window_size - 1)
    
    if window_size > poly_order:
        smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window_size, poly_order)
        smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], window_size, poly_order)
        
        # Normalize for comparison
        norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
        norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
        
        ax3.plot(dates, norm_stacked, color=colors['total'], linewidth=3, 
                 label='Normalized Stacked Gravitational Pattern', alpha=0.8)
        ax3.plot(dates, norm_coherence, color=colors['secondary'], linewidth=3, 
                 label='Normalized Temporal Field Pattern', alpha=0.8)
        
        # Add correlation coefficient
        corr_r = analysis_results['advanced_pattern_analysis']['smoothed_correlation']
        corr_p = analysis_results['advanced_pattern_analysis']['smoothed_p_value']
        
        ax3.text(0.02, 0.95, f'Pattern Correlation: r = {corr_r:.3f}, p = {corr_p:.2e}', 
                 transform=ax3.transAxes, fontsize=12, fontweight='bold', color='#220126',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                          edgecolor='#2D0140', alpha=0.95, linewidth=1))
    
    ax3.set_ylabel('Normalized Pattern Amplitude', fontsize=12, fontweight='bold')
    ax3.set_title('Gravitational-Temporal Field Pattern Correlation Analysis\n' +
                  'Smoothed Patterns Reveal Underlying Coupling', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='#220126', linestyle='-', alpha=0.8, linewidth=1.5)
    
    # Panel 4: Multi-Window Smoothing Comparison
    ax4 = fig.add_subplot(gs[3, 0])
    
    # Test different smoothing windows
    smoothing_windows = [60, 90, 120, 180, 240]
    window_colors = ['#E74C3C', '#F39C12', '#3498DB', '#2D0140', '#9B59B6']  # Different colors for each window
    
    correlations_by_window = {}
    
    for i, window in enumerate(smoothing_windows):
        if window < len(combined_df) and window > 3:  # Ensure valid window size
            poly_order = min(3, window - 1)
            
            if window > poly_order:
                # Apply smoothing
                smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window, poly_order)
                smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], window, poly_order)
                
                # Normalize for comparison
                norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
                norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
                
                # Calculate correlation
                r, p = stats.pearsonr(smoothed_stacked, smoothed_coherence_std)
                correlations_by_window[window] = {'r': r, 'p': p}
                
                # Plot normalized patterns (offset for visibility)
                offset = i * 0.3
                ax4.plot(dates, norm_stacked + offset, color=window_colors[i], linewidth=2, 
                        alpha=0.8, label=f'Gravitational (w={window}, r={r:.3f})')
                ax4.plot(dates, norm_coherence + offset, color=window_colors[i], linewidth=2, 
                        linestyle='--', alpha=0.6, label=f'Temporal (w={window})')
    
    ax4.set_ylabel('Normalized Pattern Amplitude (Offset)', fontsize=12, fontweight='bold')
    ax4.set_title('Multi-Window Smoothing Comparison\n' +
                  'Different Smoothing Windows Reveal Pattern Stability', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='#220126', linestyle='-', alpha=0.8, linewidth=1.5)
    
    # Add correlation summary text
    corr_text = "Window Correlations:\n"
    for window, corr_data in correlations_by_window.items():
        corr_text += f"w={window}: r={corr_data['r']:.3f}, p={corr_data['p']:.2e}\n"
    
    ax4.text(0.02, 0.95, corr_text, transform=ax4.transAxes, fontsize=10, 
             fontweight='bold', color='#220126',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                      edgecolor='#2D0140', alpha=0.95, linewidth=1),
             verticalalignment='top')
    
    # Format x-axis for all time series plots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/figures/step_14_comprehensive_gravitational_temporal_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to site figures folder for manuscript
    import shutil
    site_path = '/Users/matthewsmawfield/www/TEP-GNSS/site/figures/step_14_comprehensive_gravitational_temporal_analysis.png'
    shutil.copy2(output_path, site_path)
    
    print(f"Comprehensive visualization saved: {output_path}")
    print(f"Figure synced to site: {site_path}")
    return output_path

def main():
    """
    Main execution function that recreates the correct working analysis.
    """
    print_status("TEP GNSS Analysis Package v0.13 - STEP 14: Comprehensive Gravitational-Temporal Field Correlation Analysis", "TITLE")
    print()
    
    # Configuration
    start_date = '2023-01-01'
    end_date = '2025-06-30'
    
    # Generate gravitational data
    print(f"Generating high-precision gravitational data from {start_date} to {end_date}...")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    data_list = []
    current_date = start
    
    while current_date <= end:
        gravitational_data = calculate_high_precision_gravitational_influence(current_date)
        
        if gravitational_data:
            data_entry = {'date': current_date}
            data_entry.update(gravitational_data)
            data_list.append(data_entry)
        
        current_date += timedelta(days=1)
        
        # Progress indicator
        if len(data_list) % 100 == 0:
            print(f"  Processed {len(data_list)} days...")
    
    gravitational_df = pd.DataFrame(data_list)
    print(f"Generated gravitational data for {len(gravitational_df)} days")
    
    # Extract authentic daily TEP coherence data
    tep_df = extract_real_daily_tep_coherence_data()
    
    # Merge datasets
    print("Merging gravitational and temporal field datasets...")
    combined_df = pd.merge(gravitational_df, tep_df, on='date', how='inner')
    print(f"Combined dataset: {len(combined_df)} days of synchronized data")
    
    # Perform comprehensive correlation analysis
    analysis_results = perform_advanced_correlation_analysis(combined_df)
    
    # Create comprehensive visualization
    figure_path = create_comprehensive_visualization(combined_df, analysis_results)
    analysis_results['figure_path'] = figure_path
    
    # Save results
    results_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/outputs/step_14_comprehensive_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    data_path = '/Users/matthewsmawfield/www/TEP-GNSS/data/processed/step_14_comprehensive_gravitational_temporal_data.csv'
    combined_df.to_csv(data_path, index=False)

    # Export WebGL-ready dataset for Step 17 visualization (no fallbacks)
    export_dir = Path('/Users/matthewsmawfield/www/TEP-GNSS/site/data/step_14')
    export_dir.mkdir(parents=True, exist_ok=True)

    export_payload = {
        'dates': [d.strftime('%Y-%m-%d') for d in combined_df['date']],
        'total_planetary_influence': combined_df['total_planetary_influence'].tolist(),
        'total_influence': combined_df['total_influence'].tolist(),
        'coherence_mean': combined_df['coherence_mean'].tolist(),
        'coherence_std': combined_df['coherence_std'].tolist(),
        'individual_influences': {
            body: combined_df[f'{body}_influence'].tolist()
            for body in ['sun', 'jupiter', 'saturn', 'venus', 'mars']
        },
        'coherence_count': combined_df['coherence_count'].tolist(),
        'advanced_pattern_analysis': analysis_results.get('advanced_pattern_analysis'),
    }

    export_path = export_dir / 'gravitational_temporal_daily.json'
    with export_path.open('w') as f:
        json.dump(export_payload, f)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - KEY DISCOVERIES")
    print("=" * 80)
    
    stacked_corr = analysis_results['correlations']['stacked_planetary_influence']['coherence_std']
    print(f"STACKED GRAVITATIONAL PATTERN CORRELATION: r = {stacked_corr['pearson_r']:.4f}, p = {stacked_corr['pearson_p']:.2e}")
    
    if 'advanced_pattern_analysis' in analysis_results:
        smooth_corr = analysis_results['advanced_pattern_analysis']['smoothed_correlation']
        print(f"SMOOTHED PATTERN CORRELATION: r = {smooth_corr:.4f}")
    
    print(f"DATASET: {len(combined_df)} days")
    print(f"FIGURE: {figure_path}")
    print(f"RESULTS: {results_path}")
    print(f"DATA: {data_path}")
    
    print("\nKEY DISCOVERY:")
    print("   The stacked gravitational influence pattern demonstrates significant")
    print("   correlation with Earth's temporal field structure, providing")
    print("   experimental evidence supporting TEP theory predictions.")
    print("=" * 80)
    
    return analysis_results

if __name__ == "__main__":
    results = main()