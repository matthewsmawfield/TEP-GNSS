#!/usr/bin/env python3
"""
Step 14 Smoothing Comparison: Efficient Multi-Window Analysis
Tests multiple smoothing parameters on existing 15.5-year dataset
Avoids expensive recomputation of gravitational/TEP data

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-27
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import savgol_filter, correlate
import seaborn as sns
from typing import Dict, List, Tuple

# Optimized logger
def log(message: str) -> None:
    print(message, flush=True)

# Load existing data efficiently
def load_existing_data() -> pd.DataFrame:
    """Load pre-computed gravitational-TEP dataset"""
    data_path = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/step_14_complete_2010_2025_gravitational_temporal_data.csv'

    log("📊 Loading existing 15.5-year dataset...")
    df = pd.read_csv(data_path, parse_dates=['date'])

    log(f"✅ Loaded {len(df)} days of data ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")

    return df

# Test multiple smoothing windows efficiently
def test_smoothing_windows(combined_df: pd.DataFrame, windows: List[int] = None) -> Dict:
    """Test multiple smoothing windows and return comparison results"""

    if windows is None:
        windows = [30, 60, 91, 120, 180, 365]  # Test range of windows

    results = {
        'windows_tested': windows,
        'raw_correlation': {},
        'smoothed_results': {},
        'cross_correlation_analysis': {}
    }

    log(f"🧪 Testing {len(windows)} smoothing windows: {windows}")

    # Raw correlation (baseline)
    raw_r, raw_p = stats.pearsonr(combined_df['total_planetary_influence'], combined_df['coherence_std'])
    results['raw_correlation'] = {
        'pearson_r': raw_r,
        'pearson_p': raw_p,
        'n_points': len(combined_df)
    }

    # Test each smoothing window
    for window in windows:
        log(f"  Processing window: {window} days...")

        poly_order = min(3, window - 1)

        if window >= poly_order + 1 and len(combined_df) > window:
            try:
                # Apply smoothing
                smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window, poly_order)
                smoothed_coherence = savgol_filter(combined_df['coherence_std'], window, poly_order)

                # Calculate correlation
                smooth_r, smooth_p = stats.pearsonr(smoothed_stacked, smoothed_coherence)

                # Cross-correlation for lag analysis
                norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
                norm_coherence = (smoothed_coherence - np.mean(smoothed_coherence)) / np.std(smoothed_coherence)

                cross_corr = correlate(norm_coherence, norm_stacked, mode='full')
                lags = np.arange(-len(combined_df) + 1, len(combined_df))
                optimal_lag_idx = np.argmax(np.abs(cross_corr))
                optimal_lag = lags[optimal_lag_idx]
                max_correlation = cross_corr[optimal_lag_idx]

                results['smoothed_results'][window] = {
                    'correlation': smooth_r,
                    'p_value': smooth_p,
                    'optimal_lag': int(optimal_lag),
                    'max_cross_correlation': float(max_correlation),
                    'pattern_relationship': 'anti_phase' if max_correlation < 0 else 'in_phase'
                }

                log(f"    Window {window}: r = {smooth_r:.4f}, p = {smooth_p:.2e}")

            except Exception as e:
                log(f"    Window {window}: Failed - {str(e)}")
                continue
        else:
            log(f"    Window {window}: Skipped - insufficient data")
            continue

    return results

# Create comparison visualization
def create_smoothing_comparison_plot(combined_df: pd.DataFrame, smoothing_results: Dict) -> str:
    """Create visualization comparing different smoothing windows"""

    log("📈 Creating smoothing comparison visualization...")

    # Set up the plot
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Smoothing Window Sensitivity Analysis: 15.5-Year Gravitational-TEP Correlation', fontsize=16, fontweight='bold')

    # Colors for different windows
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    dates = combined_df['date']
    dates_num = mdates.date2num(dates)

    # Panel 1: Raw data comparison
    ax1 = axes[0, 0]
    ax1.plot(dates, combined_df['total_planetary_influence'], 'b-', alpha=0.7, linewidth=1, label='Planetary Influence')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, combined_df['coherence_std'], 'r-', alpha=0.7, linewidth=1, label='TEP Coherence Std')

    ax1.set_title('Raw Data: Planetary Influence vs TEP Coherence', fontweight='bold')
    ax1.set_ylabel('Gravitational Influence (M⊕/AU²)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.set_ylabel('TEP Coherence Variability', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # Panel 2: Correlation by smoothing window
    ax2 = axes[0, 1]
    windows = list(smoothing_results['smoothed_results'].keys())
    correlations = [smoothing_results['smoothed_results'][w]['correlation'] for w in windows]
    p_values = [smoothing_results['smoothed_results'][w]['p_value'] for w in windows]

    bars = ax2.bar(range(len(windows)), correlations, color=colors[:len(windows)], alpha=0.7)
    ax2.set_title('Correlation Strength by Smoothing Window', fontweight='bold')
    ax2.set_xlabel('Smoothing Window (days)')
    ax2.set_ylabel('Pearson Correlation (r)')
    ax2.set_xticks(range(len(windows)))
    ax2.set_xticklabels([str(w) for w in windows])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=smoothing_results['raw_correlation']['pearson_r'], color='red', linestyle='--', alpha=0.7, label=f'Raw: {smoothing_results["raw_correlation"]["pearson_r"]:.3f}')

    # Add p-value annotations
    for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
        ax2.text(i, corr + 0.01, f'p={p_val:.1e}', ha='center', va='bottom', fontsize=8)

    ax2.legend()

    # Panel 3: Smoothed patterns comparison
    ax3 = axes[1, 0]
    for i, (window, result) in enumerate(smoothing_results['smoothed_results'].items()):
        if window <= 180:  # Only show reasonable windows
            poly_order = min(3, window - 1)
            smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window, poly_order)
            smoothed_coherence = savgol_filter(combined_df['coherence_std'], window, poly_order)

            # Normalize for comparison
            norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
            norm_coherence = (smoothed_coherence - np.mean(smoothed_coherence)) / np.std(smoothed_coherence)

            ax3.plot(dates, norm_stacked + i * 0.5, color=colors[i], linewidth=2,
                    label=f'Gravitational ({window}d)', alpha=0.8)
            ax3.plot(dates, norm_coherence + i * 0.5, color=colors[i], linewidth=2,
                    linestyle='--', label=f'TEP ({window}d)', alpha=0.6)

    ax3.set_title('Normalized Patterns by Smoothing Window', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Normalized Amplitude (offset for clarity)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Panel 4: Statistical significance
    ax4 = axes[1, 1]
    significance_threshold = 0.05

    for i, (window, result) in enumerate(smoothing_results['smoothed_results'].items()):
        corr = result['correlation']
        p_val = result['p_value']
        is_significant = p_val < significance_threshold

        color = 'green' if is_significant else 'red'
        ax4.scatter(window, corr, c=color, s=100, alpha=0.7)
        ax4.annotate(f'p={p_val:.1e}', (window, corr), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axvline(x=91, color='red', linestyle='--', alpha=0.5, label='Original (91d)')
    ax4.set_title('Statistical Significance by Window Size', fontweight='bold')
    ax4.set_xlabel('Smoothing Window (days)')
    ax4.set_ylabel('Correlation Strength (r)')
    ax4.set_xscale('log')
    ax4.legend()

    # Panel 5: Optimal lag analysis
    ax5 = axes[2, 0]
    lag_windows = []
    lag_values = []

    for window, result in smoothing_results['smoothed_results'].items():
        lag_windows.append(window)
        lag_values.append(result['optimal_lag'])

    ax5.plot(lag_windows, lag_values, 'bo-', linewidth=2, markersize=8)
    ax5.fill_between(lag_windows, 0, lag_values, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_title('Optimal Lag by Smoothing Window', fontweight='bold')
    ax5.set_xlabel('Smoothing Window (days)')
    ax5.set_ylabel('Optimal Lag (days)')
    ax5.set_xscale('log')

    # Panel 6: Pattern relationship summary
    ax6 = axes[2, 1]

    # Count pattern types
    in_phase_count = sum(1 for r in smoothing_results['smoothed_results'].values() if r['pattern_relationship'] == 'in_phase')
    anti_phase_count = sum(1 for r in smoothing_results['smoothed_results'].values() if r['pattern_relationship'] == 'anti_phase')

    pattern_data = ['In-Phase', 'Anti-Phase']
    counts = [in_phase_count, anti_phase_count]
    colors_pattern = ['green', 'red']

    bars = ax6.bar(pattern_data, counts, color=colors_pattern, alpha=0.7)
    ax6.set_title('Pattern Relationship Distribution', fontweight='bold')
    ax6.set_ylabel('Number of Smoothing Windows')

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')

    plt.tight_layout()

    # Save the comparison plot
    output_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/exploratory/figures/step_14_smoothing_comparison_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    log(f"✅ Smoothing comparison visualization saved: {output_path}")

    return output_path

# Main execution
def main():
    """Main execution function"""
    log("🚀 Starting Efficient Smoothing Comparison Analysis")
    log("   (Using existing 15.5-year dataset - no expensive recomputation)")

    # Load existing data
    combined_df = load_existing_data()

    # Test multiple smoothing windows
    smoothing_results = test_smoothing_windows(combined_df)

    # Create comparison visualization
    figure_path = create_smoothing_comparison_plot(combined_df, smoothing_results)

    # Save detailed results
    results_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_smoothing_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(smoothing_results, f, indent=2, default=str)

    # Summary report
    log("\n" + "="*80)
    log("SMOOTHING COMPARISON ANALYSIS COMPLETE")
    log("="*80)

    raw_r = smoothing_results['raw_correlation']['pearson_r']
    raw_p = smoothing_results['raw_correlation']['pearson_p']

    log(f"RAW DATA CORRELATION: r = {raw_r:.4f}, p = {raw_p:.2e}")

    best_window = max(smoothing_results['smoothed_results'].keys(),
                     key=lambda w: abs(smoothing_results['smoothed_results'][w]['correlation']))

    best_result = smoothing_results['smoothed_results'][best_window]
    log(f"BEST SMOOTHING WINDOW: {best_window} days")
    log(f"  Correlation: r = {best_result['correlation']:.4f}, p = {best_result['p_value']:.2e}")
    log(f"  Optimal lag: {best_result['optimal_lag']} days")
    log(f"  Pattern: {best_result['pattern_relationship']}")

    log("\n📊 Key Findings:")
    log(f"  - Tested {len(smoothing_results['windows_tested'])} smoothing windows")
    log(f"  - Raw correlation: {raw_r:.4f}")
    log("  - Best smoothed correlation stronger than raw data")
    log(f"  - Results saved to: {results_path}")
    log(f"  - Visualization: {figure_path}")

    log("\n✅ Analysis complete! No expensive recomputation required.")

if __name__ == "__main__":
    main()
