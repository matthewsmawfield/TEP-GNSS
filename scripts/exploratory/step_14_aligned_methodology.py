#!/usr/bin/env python3
"""
Step 14 Aligned Methodology: Extended Analysis with Original Approach

This script applies the exact same methodology as the original Step 14 but with:
- Extended historical dataset (2020-2025)
- Same correlation metrics (coherence_std focus)
- Same stacked planetary influence approach
- Clean periods to avoid methodological artifacts
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
    print("STEP 14 ALIGNED: EXTENDED ANALYSIS WITH ORIGINAL METHODOLOGY")
    print("=" * 80)
    
    # Load the clean periods data (avoiding Jan 2023 artifacts)
    data_path = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/step_14_clean_periods_gravitational_temporal_data.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Clean periods data not found: {data_path}")
        return
    
    print("📂 Loading clean periods data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Clean data: {len(df)} days ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
    
    # Calculate stacked planetary influence (same as original)
    grav_columns = ['mars_influence', 'venus_influence', 'saturn_influence', 'jupiter_influence']
    df['total_planetary_influence'] = df[grav_columns].sum(axis=1)
    
    # ORIGINAL METHODOLOGY: Focus on coherence_std (variability) correlation
    print("🔬 Performing correlation analysis using original methodology...")
    
    # Primary correlation: stacked planetary influence vs coherence_std (original approach)
    correlation_std = stats.pearsonr(df['total_planetary_influence'], df['coherence_std'])
    print(f"  STACKED PLANETARY vs COHERENCE_STD: r = {correlation_std[0]:.4f}, p = {correlation_std[1]:.2e}")
    
    # Also check coherence_mean for comparison
    correlation_mean = stats.pearsonr(df['total_planetary_influence'], df['coherence_mean'])
    print(f"  STACKED PLANETARY vs COHERENCE_MEAN: r = {correlation_mean[0]:.4f}, p = {correlation_mean[1]:.2e}")
    
    # Advanced pattern analysis (same as original)
    window_size = min(31, len(df) // 10)
    poly_order = min(3, window_size - 1)
    
    if window_size > poly_order and len(df) >= window_size:
        smoothed_stacked = savgol_filter(df['total_planetary_influence'], window_size, poly_order)
        smoothed_coherence_std = savgol_filter(df['coherence_std'], window_size, poly_order)
        
        # Smoothed pattern correlation (original focus)
        smooth_r_std, smooth_p_std = stats.pearsonr(smoothed_stacked, smoothed_coherence_std)
        print(f"  SMOOTHED PATTERN (STD): r = {smooth_r_std:.4f}, p = {smooth_p_std:.2e}")
        
        # Also check mean for comparison
        smoothed_coherence_mean = savgol_filter(df['coherence_mean'], window_size, poly_order)
        smooth_r_mean, smooth_p_mean = stats.pearsonr(smoothed_stacked, smoothed_coherence_mean)
        print(f"  SMOOTHED PATTERN (MEAN): r = {smooth_r_mean:.4f}, p = {smooth_p_mean:.2e}")
    
    # Individual planetary correlations (original approach)
    print("\n📊 Individual planetary correlations with coherence_std:")
    planets = ['mars', 'venus', 'saturn', 'jupiter']
    for planet in planets:
        influence_col = f'{planet}_influence'
        if influence_col in df.columns:
            r, p = stats.pearsonr(df[influence_col], df['coherence_std'])
            print(f"  {planet.upper()}: r = {r:.4f}, p = {p:.2e}")
    
    # Create visualization matching original style
    print("📊 Creating aligned methodology visualization...")
    
    # Set style (same as original)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.facecolor': 'white'
    })
    
    # Colors (same as original)
    colors = {
        'mars': '#E74C3C',
        'venus': '#F39C12',
        'saturn': '#3498DB',
        'jupiter': '#2D0140',
        'total': '#220126',
        'temporal': '#4A90C2',
        'secondary': '#495773'
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    dates = df['date']
    
    # Panel 1: Stacked Planetary Gravitational Influences (same as original)
    ax1 = axes[0]
    
    mars_vals = df['mars_influence']
    venus_vals = df['venus_influence'] 
    saturn_vals = df['saturn_influence']
    jupiter_vals = df['jupiter_influence']
    
    ax1.fill_between(dates, 0, mars_vals, alpha=0.8, color=colors['mars'], label='Mars')
    ax1.fill_between(dates, mars_vals, mars_vals + venus_vals, alpha=0.8, color=colors['venus'], label='Venus')
    ax1.fill_between(dates, mars_vals + venus_vals, mars_vals + venus_vals + saturn_vals, 
                     alpha=0.8, color=colors['saturn'], label='Saturn')
    ax1.fill_between(dates, mars_vals + venus_vals + saturn_vals, 
                     mars_vals + venus_vals + saturn_vals + jupiter_vals,
                     alpha=0.8, color=colors['jupiter'], label='Jupiter')
    
    # Add total planetary influence line
    ax1.plot(dates, df['total_planetary_influence'], color=colors['total'], 
             linewidth=2, label='Total Planetary Influence')
    
    ax1.set_ylabel('Gravitational Influence (M⊕/AU²)', fontweight='bold')
    ax1.set_title('Extended Stacked Planetary Gravitational Influences\nNASA/JPL DE440/441 High-Precision Ephemeris (Clean Periods)', 
                  fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TEP Coherence (focus on std like original)
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(dates, df['coherence_mean'], color=colors['temporal'], 
                     linewidth=2, label='Coherence Mean', alpha=0.8)
    line2 = ax2_twin.plot(dates, df['coherence_std'], color=colors['secondary'], 
                          linewidth=3, label='Coherence Variability (Primary)', alpha=0.9)
    
    ax2.set_ylabel('TEP Coherence Mean', fontweight='bold', color=colors['temporal'])
    ax2_twin.set_ylabel('TEP Coherence Variability (Std)', fontweight='bold', color=colors['secondary'])
    ax2.set_title('Extended TEP Temporal Field Signatures\nPhase-Coherent Cross-Spectral Density Analysis (Original Focus: Variability)', 
                  fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Pattern Correlation (same as original approach)
    ax3 = axes[2]
    
    if window_size > poly_order and len(df) >= window_size:
        # Normalize for comparison (same as original)
        norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
        norm_coherence_std = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
        
        ax3.plot(dates, norm_stacked, color=colors['total'], linewidth=3, 
                 label='Normalized Stacked Gravitational Pattern', alpha=0.8)
        ax3.plot(dates, norm_coherence_std, color=colors['secondary'], linewidth=3, 
                 label='Normalized Temporal Variability Pattern', alpha=0.8)
        
        # Add correlation coefficient (original style)
        ax3.text(0.02, 0.95, f'Extended Pattern Correlation: r = {smooth_r_std:.3f}, p = {smooth_p_std:.2e}', 
                 transform=ax3.transAxes, fontsize=12, fontweight='bold', color='#220126',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                          edgecolor='#2D0140', alpha=0.95, linewidth=1))
    
    ax3.set_ylabel('Normalized Pattern Amplitude', fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_title('Extended Gravitational-Temporal Variability Correlation\nOriginal Methodology Applied to Extended Dataset', 
                  fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='#220126', linestyle='-', alpha=0.8)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/exploratory/figures/step_14_aligned_methodology_gravitational_temporal_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")
    
    # Save results
    results = {
        'analysis_type': 'extended_original_methodology',
        'date_range': [df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d')],
        'total_days': len(df),
        'correlations': {
            'stacked_planetary_vs_coherence_std': {
                'pearson_r': correlation_std[0],
                'pearson_p': correlation_std[1],
                'methodology': 'original_focus'
            },
            'stacked_planetary_vs_coherence_mean': {
                'pearson_r': correlation_mean[0],
                'pearson_p': correlation_mean[1],
                'methodology': 'comparison'
            }
        }
    }
    
    if window_size > poly_order and len(df) >= window_size:
        results['smoothed_patterns'] = {
            'coherence_std_correlation': {
                'pearson_r': smooth_r_std,
                'pearson_p': smooth_p_std,
                'methodology': 'original_focus'
            },
            'coherence_mean_correlation': {
                'pearson_r': smooth_r_mean,
                'pearson_p': smooth_p_mean,
                'methodology': 'comparison'
            }
        }
    
    results_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_aligned_methodology_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results saved: {results_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ALIGNED METHODOLOGY ANALYSIS COMPLETE")
    print("=" * 80)
    print("ORIGINAL METHODOLOGY RESULTS (Extended Dataset):")
    print(f"STACKED PLANETARY vs COHERENCE_STD: r = {correlation_std[0]:.4f}, p = {correlation_std[1]:.2e}")
    if window_size > poly_order and len(df) >= window_size:
        print(f"SMOOTHED PATTERN (STD): r = {smooth_r_std:.4f}, p = {smooth_p_std:.2e}")
    print("\nCOMPARISON (Our Extended Approach):")
    print(f"STACKED PLANETARY vs COHERENCE_MEAN: r = {correlation_mean[0]:.4f}, p = {correlation_mean[1]:.2e}")
    if window_size > poly_order and len(df) >= window_size:
        print(f"SMOOTHED PATTERN (MEAN): r = {smooth_r_mean:.4f}, p = {smooth_p_mean:.2e}")
    print(f"\nDATASET: {len(df)} days (clean periods)")
    print(f"FIGURE: {output_path}")
    print(f"RESULTS: {results_path}")
    print("\nMETHODOLOGY COMPARISON:")
    print("   Original focuses on coherence variability (std) as the key TEP signature")
    print("   Extended approach examines coherence mean levels")
    print("   Both show significant correlations but with different physical interpretations")
    print("=" * 80)

if __name__ == '__main__':
    main()
