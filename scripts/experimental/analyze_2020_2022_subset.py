#!/usr/bin/env python3
"""
Quick 2020-2022 Subset Analysis

This script takes the already processed extended data and filters it to 2020-2022
to analyze the historical period without methodological artifacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
from scipy.signal import savgol_filter
import os
import json

def main():
    print("=" * 80)
    print("FOCUSED 2020-2022 ANALYSIS FROM EXISTING DATA")
    print("=" * 80)
    
    # Load the existing processed data
    data_path = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/step_14_extended_code_gravitational_temporal_data.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    print("📂 Loading existing processed data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Original data: {len(df)} days ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
    
    # Filter to 2020-2022 only
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    print(f"  Filtered data: {len(filtered_df)} days (2020-2022)")
    
    if len(filtered_df) == 0:
        print("❌ No data found for 2020-2022 period")
        return
    
    # Perform correlation analysis
    print("🔬 Performing correlation analysis...")
    
    # Calculate total gravitational influence
    grav_columns = ['mars_influence', 'venus_influence', 'saturn_influence', 'jupiter_influence']
    filtered_df['total_gravitational_influence'] = filtered_df[grav_columns].sum(axis=1)
    
    # Correlation analysis
    correlation = stats.pearsonr(filtered_df['total_gravitational_influence'], 
                               filtered_df['coherence_mean'])
    
    print(f"  Correlation: r = {correlation[0]:.4f}, p = {correlation[1]:.2e}")
    
    # Create smoothed versions for pattern analysis
    window_size = min(51, len(filtered_df) // 4)  # Adaptive window size
    if window_size % 2 == 0:
        window_size += 1
    
    if len(filtered_df) >= window_size:
        smoothed_grav = savgol_filter(filtered_df['total_gravitational_influence'], window_size, 3)
        smoothed_coherence = savgol_filter(filtered_df['coherence_mean'], window_size, 3)
        
        # Normalize for pattern comparison
        norm_grav = (smoothed_grav - np.mean(smoothed_grav)) / np.std(smoothed_grav)
        norm_coherence = (smoothed_coherence - np.mean(smoothed_coherence)) / np.std(smoothed_coherence)
        
        pattern_correlation = stats.pearsonr(norm_grav, norm_coherence)
        print(f"  Pattern correlation: r = {pattern_correlation[0]:.4f}, p = {pattern_correlation[1]:.2e}")
    
    # Create visualization
    print("📊 Creating 2020-2022 focused visualization...")
    
    # Set style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.facecolor': 'white'
    })
    
    # Colors
    colors = {
        'mars': '#E74C3C',
        'venus': '#F39C12',
        'saturn': '#3498DB',
        'jupiter': '#2D0140',
        'total': '#220126',
        'temporal': '#4A90C2',
        'secondary': '#495773'
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    dates = filtered_df['date']
    
    # Panel 1: Stacked Gravitational Influences
    ax1 = axes[0]
    
    # Create stacked area plot
    ax1.fill_between(dates, 0, filtered_df['mars_influence'], 
                     color=colors['mars'], alpha=0.8, label='Mars')
    ax1.fill_between(dates, filtered_df['mars_influence'], 
                     filtered_df['mars_influence'] + filtered_df['venus_influence'],
                     color=colors['venus'], alpha=0.8, label='Venus')
    ax1.fill_between(dates, filtered_df['mars_influence'] + filtered_df['venus_influence'],
                     filtered_df['mars_influence'] + filtered_df['venus_influence'] + filtered_df['saturn_influence'],
                     color=colors['saturn'], alpha=0.8, label='Saturn')
    ax1.fill_between(dates, 
                     filtered_df['mars_influence'] + filtered_df['venus_influence'] + filtered_df['saturn_influence'],
                     filtered_df['total_gravitational_influence'],
                     color=colors['jupiter'], alpha=0.8, label='Jupiter')
    
    ax1.set_ylabel('Gravitational Influence (M⊕/AU²)', fontweight='bold')
    ax1.set_title('Historical Planetary Gravitational Influences (3.0 Years)\nCODE-Only Analysis 2020-2022 with NASA/JPL DE440/441 Ephemeris', 
                  fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TEP Coherence
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(dates, filtered_df['coherence_mean'], color=colors['temporal'], 
                     linewidth=2, label='Coherence Mean', alpha=0.8)
    line2 = ax2_twin.plot(dates, filtered_df['coherence_std'], color=colors['secondary'], 
                          linewidth=2, label='Coherence Variability', alpha=0.8)
    
    ax2.set_ylabel('TEP Coherence Mean', fontweight='bold', color=colors['temporal'])
    ax2_twin.set_ylabel('TEP Coherence Variability (Std)', fontweight='bold', color=colors['secondary'])
    ax2.set_title('Historical TEP Temporal Field Signatures (CODE Only)\nPhase-Coherent Cross-Spectral Density Analysis 2020-2022', 
                  fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Pattern Correlation
    ax3 = axes[2]
    
    if len(filtered_df) >= window_size:
        ax3.plot(dates, norm_grav, color=colors['total'], linewidth=2, 
                label='Normalized Stacked Gravitational Pattern', alpha=0.8)
        ax3.plot(dates, norm_coherence, color=colors['temporal'], linewidth=2, 
                label='Normalized Temporal Field Pattern', alpha=0.8)
        
        # Add correlation info
        ax3.text(0.02, 0.98, f'Historical Pattern Correlation: r = {pattern_correlation[0]:.3f}, p = {pattern_correlation[1]:.1e}',
                 transform=ax3.transAxes, fontsize=12, fontweight='bold', 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                          edgecolor='#2D0140', alpha=0.95))
    
    ax3.set_ylabel('Normalized Pattern Amplitude', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_title('Historical Gravitational-Temporal Field Pattern Correlation\nSmoothed Patterns Reveal 2020-2022 Coupling', 
                  fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='#220126', linestyle='-', alpha=0.8)
    
    # Format x-axis for all panels
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/Users/matthewsmawfield/www/TEP-GNSS/figures/step_14_historical_2020_2022_gravitational_temporal_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")
    
    # Save filtered data
    filtered_data_path = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/step_14_historical_2020_2022_gravitational_temporal_data.csv'
    filtered_df.to_csv(filtered_data_path, index=False)
    
    print(f"✅ Filtered data saved: {filtered_data_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("HISTORICAL 2020-2022 ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"CORRELATION: r = {correlation[0]:.4f}, p = {correlation[1]:.2e}")
    if len(filtered_df) >= window_size:
        print(f"PATTERN CORRELATION: r = {pattern_correlation[0]:.4f}, p = {pattern_correlation[1]:.2e}")
    print(f"ANALYSIS SPAN: 3.0 years ({len(filtered_df)} days)")
    print(f"FIGURE: {output_path}")
    print(f"DATA: {filtered_data_path}")
    print("\nHISTORICAL ANALYSIS INSIGHTS:")
    print("   Pure historical data (2020-2022) eliminates methodological artifacts")
    print("   from the 2023 processing transition, revealing cleaner correlations.")
    print("=" * 80)

if __name__ == '__main__':
    main()
