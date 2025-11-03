#!/usr/bin/env python3
"""
Create Step 4.4-style time series visualization from Step 2.2 Longspan Results
Shows 25 years of CODE data with stacked charts.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter
from scipy import stats

# Paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.exceptions import safe_json_read, safe_csv_read
from scripts.utils.logger import print_status

def load_step_2_2_and_geospatial_data():
    """Load Step 2.2 results and extract daily time series from Step 2.1 geospatial data."""
    print_status("Loading Step 2.2 results and geospatial data...", "INFO")
    
    # Load Step 2.2 JSON
    json_path = ROOT / 'results/outputs/code_longspan/step_2_2_geospatial_temporal_analysis_code.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Step 2.2 results not found at {json_path}")
    
    results = safe_json_read(json_path)
    
    # Load Step 2.1 geospatial data to extract daily coherence time series
    geospatial_file = ROOT / 'data/processed/code_longspan/step_2_1_geospatial_code.csv'
    
    if not geospatial_file.exists():
        raise FileNotFoundError(f"Step 2.1 geospatial data not found at {geospatial_file}")
    
    print_status(f"Loading geospatial data from {geospatial_file.name} (this may take a while for 25 years)...", "INFO")
    
    # Read in chunks to handle large file
    chunk_size = 1000000
    chunks = []
    for chunk in pd.read_csv(geospatial_file, chunksize=chunk_size):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    print_status(f"Loaded {len(df):,} pairs from geospatial data", "INFO")
    
    # Convert date and calculate coherence
    df['date'] = pd.to_datetime(df['date'])
    df['coherence'] = np.cos(df['plateau_phase'])
    
    # Group by date
    print_status("Aggregating daily statistics...", "INFO")
    daily_data = df.groupby(df['date'].dt.date).agg({
        'coherence': ['mean', 'std', 'median', 'count']
    }).reset_index()
    daily_data.columns = ['date', 'coherence_mean', 'coherence_std', 'coherence_median', 'coherence_count']
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data = daily_data.sort_values('date').reset_index(drop=True)
    
    print_status(f"Created daily time series: {len(daily_data)} days from {daily_data['date'].min().date()} to {daily_data['date'].max().date()}", "SUCCESS")
    
    return results, daily_data

def calculate_planetary_influences(daily_data):
    """Calculate planetary gravitational influences using JPL ephemeris."""
    print_status("Calculating high-precision planetary influences...", "INFO")
    
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
    from astropy import units as u
    
    solar_system_ephemeris.set('jpl')
    
    PLANETARY_MASSES = {
        'sun': 332946.0,
        'jupiter': 317.8,
        'saturn': 95.2,
        'venus': 0.815,
        'mars': 0.107,
        'mercury': 0.055
    }
    
    influences = []
    
    for idx, row in daily_data.iterrows():
        if idx % 100 == 0:
            print_status(f"  Processing day {idx+1}/{len(daily_data)}", "INFO")
        
        date = row['date']
        astro_time = Time(date.strftime('%Y-%m-%d'))
        
        try:
            earth_pos, _ = get_body_barycentric_posvel('earth', astro_time)
            
            day_influences = {
                'date': date,
                'coherence_mean': row['coherence_mean'],
                'coherence_std': row['coherence_std'],
                'coherence_median': row['coherence_median']
            }
            
            for body in ['sun', 'jupiter', 'saturn', 'venus', 'mars']:
                body_pos, _ = get_body_barycentric_posvel(body, astro_time)
                earth_centered_pos = body_pos - earth_pos
                distance_au = np.linalg.norm(earth_centered_pos.xyz.value)
                
                mass_ratio = PLANETARY_MASSES[body]
                influence = mass_ratio / (distance_au ** 2)
                
                day_influences[f'{body}_influence'] = influence
            
            day_influences['total_planetary_influence'] = (
                day_influences['jupiter_influence'] + 
                day_influences['saturn_influence'] + 
                day_influences['venus_influence'] + 
                day_influences['mars_influence']
            )
            
            day_influences['total_influence'] = (
                day_influences['sun_influence'] + 
                day_influences['total_planetary_influence']
            )
            
            influences.append(day_influences)
            
        except Exception as e:
            print_status(f"Error calculating influences for {date}: {e}", "WARNING")
            continue
    
    combined_df = pd.DataFrame(influences)
    print_status(f"Calculated planetary influences for {len(combined_df)} days", "SUCCESS")
    
    return combined_df

def create_timeseries_visualization(combined_df, results):
    """Create 4-panel time series visualization."""
    print_status("Creating comprehensive time series visualization...", "INFO")
    
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
    
    # Create figure
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.4, left=0.08, right=0.95)
    
    # Color scheme
    colors = {
        'mars': '#E74C3C',
        'venus': '#F39C12',
        'saturn': '#3498DB',
        'jupiter': '#2D0140',
        'sun': '#F1C40F',
        'total': '#220126',
        'temporal': '#4A90C2',
        'secondary': '#495773'
    }
    
    dates = combined_df['date']
    
    # Panel 1: Stacked Planetary Gravitational Influences
    ax1 = fig.add_subplot(gs[0, 0])
    
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
    
    ax1.plot(dates, combined_df['total_planetary_influence'], color=colors['total'],
             linewidth=2, label='Total Planetary Influence')
    
    ax1.set_ylabel('Gravitational Influence (M_Earth/AU²)', fontsize=12, fontweight='bold')
    ax1.set_title('Stacked Planetary Gravitational Influences on Earth (CODE 25-Year Longspan)\n' +
                  'NASA/JPL High-Precision Ephemeris', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TEP Temporal Field Signatures
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(dates, combined_df['coherence_mean'], color=colors['temporal'],
                     linewidth=2, label='Coherence Mean', alpha=0.8)
    line2 = ax2_twin.plot(dates, combined_df['coherence_std'], color=colors['secondary'],
                          linewidth=2, label='Coherence Variability', alpha=0.8)
    
    ax2.set_ylabel('TEP Coherence Mean', fontsize=12, fontweight='bold', color=colors['temporal'])
    ax2_twin.set_ylabel('TEP Coherence Variability (Std)', fontsize=12, fontweight='bold', color=colors['secondary'])
    ax2.set_title('TEP Temporal Field Signatures from GNSS Clock Correlations (CODE)\n' +
                  'Phase-Coherent Cross-Spectral Density Analysis', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Pattern Correlation Analysis
    ax3 = fig.add_subplot(gs[2, 0])
    
    if len(combined_df) >= 100:
        window_size = min(227, len(combined_df) // 4)
        if window_size % 2 == 0:
            window_size -= 1
        poly_order = 3
        
        smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window_size, poly_order)
        smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], window_size, poly_order)
        
        norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
        norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
        
        ax3.plot(dates, norm_stacked, color=colors['total'], linewidth=3,
                 label='Normalized Stacked Gravitational Pattern', alpha=0.8)
        ax3.plot(dates, norm_coherence, color=colors['secondary'], linewidth=3,
                 label='Normalized Temporal Field Pattern', alpha=0.8)
        
        r, p = stats.pearsonr(smoothed_stacked, smoothed_coherence_std)
        ax3.text(0.02, 0.95, f'Pattern Correlation: r = {r:.3f}, p = {p:.2e}',
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
    
    if len(combined_df) >= 100:
        smoothing_windows = [60, 90, 120, 180, 240]
        window_colors = ['#E74C3C', '#F39C12', '#3498DB', '#2D0140', '#9B59B6']
        
        correlations_by_window = {}
        
        for i, window in enumerate(smoothing_windows):
            adjusted_window = min(window, len(combined_df) // 4)
            if adjusted_window % 2 == 0:
                adjusted_window -= 1
            if adjusted_window < 31:
                adjusted_window = 31
            
            poly_order = 3
            
            if adjusted_window > poly_order:
                smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], adjusted_window, poly_order)
                smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], adjusted_window, poly_order)
                
                norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
                norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
                
                r, p = stats.pearsonr(smoothed_stacked, smoothed_coherence_std)
                correlations_by_window[window] = {'r': r, 'p': p}
                
                offset = i * 0.3
                ax4.plot(dates, norm_stacked + offset, color=window_colors[i], linewidth=2,
                        alpha=0.8, label=f'Gravitational (w={window}, r={r:.3f})')
                ax4.plot(dates, norm_coherence + offset, color=window_colors[i], linewidth=2,
                        linestyle='--', alpha=0.6, label=f'Temporal (w={window})')
        
        ax4.set_ylabel('Normalized Pattern Amplitude (Offset)', fontsize=12, fontweight='bold')
        ax4.set_title('Multi-Window Smoothing Comparison (CODE 25-Year Longspan)\n' +
                      'Different Smoothing Windows Reveal Pattern Stability', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9, ncol=2)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='#220126', linestyle='-', alpha=0.8, linewidth=1.5)
        
        corr_text = "Window Correlations:\n"
        for window, corr_data in correlations_by_window.items():
            corr_text += f"w={window}: r={corr_data['r']:.3f}, p={corr_data['p']:.2e}\n"
        
        ax4.text(0.02, 0.95, corr_text, transform=ax4.transAxes, fontsize=10,
                 fontweight='bold', color='#220126',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF',
                          edgecolor='#2D0140', alpha=0.95, linewidth=1),
                 verticalalignment='top')
    
    # Format x-axis
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))  # Every 2 years for 25-year span
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.subplots_adjust(hspace=0.3)
    
    # Save to different filename
    output_path = ROOT / 'results/figures/step_2_2_longspan_code_25year_timeseries.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Time series visualization saved: {output_path}", "SUCCESS")
    return str(output_path)

def main():
    """Main execution."""
    try:
        # Load data
        results, daily_data = load_step_2_2_and_geospatial_data()
        
        # Calculate planetary influences
        combined_df = calculate_planetary_influences(daily_data)
        
        # Create visualization
        output_path = create_timeseries_visualization(combined_df, results)
        
        print_status("✓ Step 2.2 longspan time series visualization complete", "SUCCESS")
        print_status(f"  Output: {output_path}", "INFO")
        return True
        
    except Exception as e:
        print_status(f"✗ Visualization failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
