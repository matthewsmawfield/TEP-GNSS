#!/usr/bin/env python3
"""
TEP GNSS Analysis - Site-Themed Figure
======================================

Creates figure with new color palette:
- Deep purple: #220126 (primary text)
- Dark purple: #2D0140 (headings)
- Blue-gray: #495773 (secondary)
- Warm beige: #495773 (backgrounds)
- Light yellow: #F2EC99 (highlights only)

Author: Matthew Lukin Smawfield
Date: September 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]

def set_site_themed_style():
    """Styling consistent with site theme."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.titlesize': 14,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.linewidth': 1.0,
        'grid.color': '#495773',  # Site warm beige
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'text.color': '#220126',  # Site dark text
        'axes.labelcolor': '#220126',
        'xtick.color': '#220126',
        'ytick.color': '#220126'
    })

def exp_decay_model(r, A, lambda_km, C):
    """Exponential decay model: C(r) = A*exp(-r/λ) + C"""
    return A * np.exp(-r / lambda_km) + C

def create_panel_a(ax):
    """Panel A: Multi-center reproducibility with site theme colors."""
    results_dir = ROOT / 'results'
    
    centers = ['code', 'esa_final', 'igs_combined']
    # Updated color palette for better thematic consistency
    colors = ['#4A90C2', '#495773', '#2D0140']  # Cosmic Blue, Blue-Gray, Dark Purple
    labels = ['CODE', 'ESA', 'IGS']
    markers = ['o', '^', 's']
    
    all_y_values = []
    
    for i, (center, color, label, marker) in enumerate(zip(centers, colors, labels, markers)):
        try:
            # Load manuscript parameters
            with open(results_dir / 'outputs' / f'step_3_correlation_{center}.json', 'r') as f:
                data = json.load(f)
            
            fit_params = data['exponential_fit']
            lambda_val = fit_params['lambda_km']
            lambda_err = fit_params['lambda_error']
            amplitude = fit_params['amplitude']
            offset = fit_params['offset']
            r_squared = fit_params['r_squared']
            
            print(f"Loading {center}: A={amplitude:.3f}, λ={lambda_val:.0f}, C={offset:.3f}")
            
            # Load real binned data produced in Step 3
            binned_file = results_dir / 'outputs' / f'step_3_correlation_data_{center}.csv'
            df_binned = pd.read_csv(binned_file)
            x_data = df_binned['distance_km'].values
            y_data = df_binned['mean_coherence'].values
            
            # Track all y values
            all_y_values.extend(y_data)
            
            # Plot real data points, now with R² in the label
            ax.scatter(x_data, y_data, color=color, alpha=0.8, 
                      s=20, marker=marker, label=f'{label} (R² = {r_squared:.3f})', zorder=3,
                      edgecolors='white', linewidth=0.5)
            
            # Plot fit line
            x_fit = np.linspace(100, 11000, 200)
            y_fit = exp_decay_model(x_fit, amplitude, lambda_val, offset)
            all_y_values.extend(y_fit)
            ax.plot(x_fit, y_fit, color=color, linewidth=2, alpha=0.9, zorder=2)
            
            # 95% CI band using proper error propagation for all parameters
            # We assume parameter errors are uncorrelated as cov matrix is not available
            amp_err = fit_params['amplitude_error']
            lambda_err = fit_params['lambda_error']
            offset_err = fit_params['offset_error']

            # Partial derivatives of the model w.r.t. parameters
            d_dA = np.exp(-x_fit / lambda_val)
            d_dlambda = (amplitude * x_fit / (lambda_val**2)) * np.exp(-x_fit / lambda_val)
            d_dC = 1.0

            # Propagated variance and standard error of the fit
            var_y_fit = (d_dA**2 * amp_err**2) + (d_dlambda**2 * lambda_err**2) + (d_dC**2 * offset_err**2)
            se_y_fit = np.sqrt(var_y_fit)

            # 95% confidence interval
            y_upper = y_fit + 1.96 * se_y_fit
            y_lower = y_fit - 1.96 * se_y_fit
            
            all_y_values.extend(y_upper)
            all_y_values.extend(y_lower)
            ax.fill_between(x_fit, y_lower, y_upper, color=color, alpha=0.2, zorder=1)
            
        except Exception as e:
            print(f"Error processing {center}: {e}")
    
    # Set limits to 0.4 as requested
    ax.set_xlim(0, 12000)
    ax.set_ylim(-0.06, 0.4)  # Fixed upper limit to 0.4
    
    # Add consistent zero line styling (darker, matching site theme)
    ax.axhline(y=0, color='#220126', linestyle='-', alpha=0.8, linewidth=1.5, zorder=5)
    
    ax.set_xlabel('Distance (km)', fontweight='bold', color='#220126')
    ax.set_ylabel('Coherence, cos(Δφ)', fontweight='bold', color='#220126')
    ax.set_title('Multi-Center Reproducibility', fontweight='bold', pad=10, color='#220126')
    
    # Site-themed legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7)
    legend.get_frame().set_edgecolor('#220126')
    legend.get_frame().set_facecolor('#F8F8FF')
    legend.get_frame().set_alpha(0.9)
    
    # Site-themed statistics box
    textstr = ('λ = 3.33–4.55 Mm\n'
              '95% CI bands shown\n'
              'Within theoretical\n'
              'range (1–10 Mm)')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='left', color='#220126',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                     edgecolor='#2D0140', alpha=0.95, linewidth=1))

def create_panel_b(ax):
    """Panel B: Statistical significance with site theme."""
    results_dir = ROOT / 'results'
    
    # Collect all null test data
    all_null_r2 = []
    real_r2_values = []
    # Colors with blue accent
    colors = ['#220126', '#2D0140', '#4A90C2']
    labels = ['CODE', 'IGS', 'ESA']
    
    for center in ['code', 'igs_combined', 'esa_final']:
        try:
            with open(results_dir / 'outputs' / f'step_6_null_tests_{center}.json', 'r') as f:
                data = json.load(f)
            
            real_r2 = data['real_signal']['r_squared']
            null_r2_values = np.array(data['null_tests']['distance']['r_squared_values'])
            null_r2_values = null_r2_values[np.isfinite(null_r2_values) & (null_r2_values >= 0)]
            
            real_r2_values.append(real_r2)
            all_null_r2.extend(null_r2_values)
            
        except Exception as e:
            print(f"Error loading {center}: {e}")
    
    if all_null_r2 and real_r2_values:
        # Site-themed histogram
        ax.hist(all_null_r2, bins=20, color='#495773', alpha=0.8, 
               edgecolor='#220126', linewidth=1, label='Null tests (N=300)')
        
        # Real signal lines - SOLID not dotted
        for i, (r2, color, label) in enumerate(zip(real_r2_values, colors, labels)):
            ax.axvline(r2, color=color, linewidth=3, alpha=0.9, 
                      linestyle='-', label=f'{label}: {r2:.3f}')  # Solid lines
    
    ax.set_xlabel('Goodness-of-fit (R²)', fontweight='bold', color='#220126')
    ax.set_ylabel('Count', fontweight='bold', color='#220126')
    ax.set_title('Statistical Significance', fontweight='bold', pad=10, color='#220126')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 120)  # Increased y-axis limit to 120
    
    # Site-themed legend
    legend = ax.legend(loc='upper left', frameon=True, fontsize=7)
    legend.get_frame().set_edgecolor('#220126')
    legend.get_frame().set_facecolor('#F8F8FF')
    legend.get_frame().set_alpha(0.9)
    
    # Site-themed significance note
    ax.text(0.98, 0.98, 'Station-day blocked\npermutations:\np < 0.01', 
            transform=ax.transAxes, fontsize=7, ha='right', va='top', color='#220126',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                     edgecolor='#2D0140', alpha=0.95, linewidth=1))

def create_panel_c(ax):
    """Panel C: Signal vs null with site theme."""
    data_dir = ROOT / 'data'
    
    all_y_values = []
    
    try:
        # Load and process data
        df = pd.read_csv(data_dir / 'processed' / 'step_4_geospatial_code.csv')
        df = df.sample(n=15000, random_state=42)
        df['coherence'] = np.cos(df['plateau_phase'])
        
        # Create bins
        bins = np.linspace(0, 11000, 15)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Real data
        df['distance_bin'] = pd.cut(df['dist_km'], bins, labels=bin_centers)
        real_stats = df.groupby('distance_bin')['coherence'].agg(['mean', 'std', 'count'])
        real_stats = real_stats.dropna()
        real_stats['sem'] = real_stats['std'] / np.sqrt(real_stats['count'])
        
        valid_bins = real_stats.index.astype(float)
        real_means = real_stats['mean'].values
        real_errors = real_stats['sem'].values
        
        all_y_values.extend(real_means + real_errors)
        all_y_values.extend(real_means - real_errors)
        
        # Site-themed real data
        ax.errorbar(valid_bins, real_means, yerr=real_errors,
                   fmt='o-', color='#2D0140', linewidth=1.5, markersize=4, 
                   capsize=2, elinewidth=0.5, label='Real GNSS data', alpha=0.9,
                   markeredgecolor='white', markeredgewidth=0.5)
        
        # Null data
        np.random.seed(42)
        df_null = df.copy()
        df_null['coherence'] = np.random.permutation(df['coherence'].values)
        
        null_stats = df_null.groupby('distance_bin')['coherence'].agg(['mean', 'std', 'count'])
        null_stats = null_stats.dropna()
        null_stats['sem'] = null_stats['std'] / np.sqrt(null_stats['count'])
        
        valid_bins_null = null_stats.index.astype(float)
        null_means = null_stats['mean'].values
        null_errors = null_stats['sem'].values
        
        all_y_values.extend(null_means + null_errors)
        all_y_values.extend(null_means - null_errors)
        
        # Site-themed null data
        ax.errorbar(valid_bins_null, null_means, yerr=null_errors,
                   fmt='s-', color='#495773', linewidth=1.5, markersize=3, 
                   capsize=2, elinewidth=0.5, label='Distance-scrambled null', alpha=0.7,
                   markeredgecolor='white', markeredgewidth=0.5)
        
        # Consistent zero reference line (same as Panel A)
        ax.axhline(y=0, color='#220126', linestyle='-', alpha=0.8, linewidth=1.5, zorder=5)
        
        # Exponential fit overlay
        try:
            popt, _ = curve_fit(exp_decay_model, valid_bins, real_means,
                              p0=[0.15, 3500, 0], maxfev=5000)
            x_fit = np.linspace(100, 10000, 100)
            y_fit = exp_decay_model(x_fit, *popt)
            all_y_values.extend(y_fit)
            ax.plot(x_fit, y_fit, color='#FF6347', linestyle='--', alpha=0.9, linewidth=1.5,
                   label=f'Exp. fit (λ={popt[1]:.0f} km)')
        except Exception:
            pass
        
    except Exception as e:
        print(f"Panel C error: {e}")
        ax.text(0.5, 0.5, 'Data loading error', transform=ax.transAxes, 
                ha='center', va='center', fontsize=10, color='#220126')
    
    # Set limits based on actual data range
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax.set_ylim(-0.1, 0.1)
    
    ax.set_xlim(0, 12000)
    ax.set_xlabel('Distance (km)', fontweight='bold', color='#220126')
    ax.set_ylabel('Mean coherence', fontweight='bold', color='#220126')
    ax.set_title('Signal vs. Null Comparison', fontweight='bold', pad=10, color='#220126')
    
    # Site-themed legend
    legend = ax.legend(loc='upper right', frameon=True, fontsize=7)
    legend.get_frame().set_edgecolor('#220126')
    legend.get_frame().set_facecolor('#F8F8FF')
    legend.get_frame().set_alpha(0.9)
    
    # Summary - moved to bottom left with site theme
    ax.text(0.02, 0.02, 'Clear signal structure\nvs. random null', 
            transform=ax.transAxes, fontsize=7, ha='left', va='bottom', color='#220126',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                     edgecolor='#2D0140', alpha=0.95, linewidth=1))

def main():
    """Generate site-themed figure."""
    print("="*80)
    print("TEP GNSS Analysis Package v0.6")
    print("STEP 9: Synthesis Figure Generation")
    print("="*80)
    
    set_site_themed_style()
    
    print("="*60)
    print("GENERATING SITE-THEMED FIGURE")
    print("="*60)
    
    # Create figure with horizontal layout
    fig = plt.figure(figsize=(15, 5))
    
    # Create 3 panels horizontally with proper spacing
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], 
                         wspace=0.25, left=0.06, right=0.98, 
                         top=0.85, bottom=0.15)
    
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    
    print("Creating Panel A with site theme colors...")
    create_panel_a(ax_a)
    
    print("Creating Panel B with site theme...")
    # Load and create Panel B with site theme
    results_dir = ROOT / 'results'
    
    all_null_r2 = []
    real_r2_values = []
    colors = ['#4A90C2', '#495773', '#2D0140']  # Updated palette to match Panel A
    labels = ['CODE', 'ESA', 'IGS']
    
    for center in ['code', 'esa_final', 'igs_combined']:
        try:
            with open(results_dir / 'outputs' / f'step_6_null_tests_{center}.json', 'r') as f:
                data = json.load(f)
            
            real_r2 = data['real_signal']['r_squared']
            null_r2_values = np.array(data['null_tests']['distance']['r_squared_values'])
            null_r2_values = null_r2_values[np.isfinite(null_r2_values) & (null_r2_values >= 0)]
            
            real_r2_values.append(real_r2)
            all_null_r2.extend(null_r2_values)
            
        except Exception as e:
            print(f"Error loading {center}: {e}")
    
    if all_null_r2 and real_r2_values:
        # Site-themed histogram
        ax_b.hist(all_null_r2, bins=20, color='#495773', alpha=0.8, 
                 edgecolor='#220126', linewidth=1, label='Null tests (N=300)')
        
        # Real signal lines - SOLID not dotted
        for i, (r2, color, label) in enumerate(zip(real_r2_values, colors, labels)):
            ax_b.axvline(r2, color=color, linewidth=2, alpha=0.9, 
                        linestyle='-', label=f'{label}: {r2:.3f}')  # Solid lines
    
    ax_b.set_xlabel('Goodness-of-fit (R²)', fontweight='bold', color='#220126')
    ax_b.set_ylabel('Count', fontweight='bold', color='#220126')
    ax_b.set_title('Statistical Significance', fontweight='bold', pad=10, color='#220126')
    ax_b.set_xlim(0, 1.0)
    ax_b.set_ylim(0, 120)
    
    # Site-themed legend
    legend = ax_b.legend(loc='upper left', frameon=True, fontsize=7)
    legend.get_frame().set_edgecolor('#220126')
    legend.get_frame().set_facecolor('#F8F8FF')
    legend.get_frame().set_alpha(0.9)
    
    # Site-themed significance note
    ax_b.text(0.98, 0.98, 'Station-day blocked\npermutations:\np < 0.01', 
             transform=ax_b.transAxes, fontsize=7, ha='right', va='top', color='#220126',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                      edgecolor='#2D0140', alpha=0.95, linewidth=1))
    
    print("Creating Panel C with site theme...")
    create_panel_c(ax_c)
    
    # Professional, scientifically appropriate title
    fig.suptitle('Distance-structured correlations in GNSS clock networks', 
                fontsize=16, fontweight='bold', y=0.95, color='#220126')
    
    # Site-themed footer
    footer_text = ('Methods: cos(Δφ) coherence metric throughout. Panel A: 95% CI from error propagation, λ = 3.33–4.55 Mm within theoretical range. ' +
                  'Panel B: Station-day blocked permutations (N=300). Panel C: Distance-scrambled null comparison.')
    fig.text(0.5, 0.02, footer_text, fontsize=8, ha='center', style='italic', 
             alpha=0.8, color='#495773')
    
    # Save with site-themed filename
    output_path = ROOT / 'results' / 'figures' / 'figure_1_TEP_site_themed.png'
    print(f"\nSaving site-themed figure: {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✓ Site-themed figure saved: {output_path}")
    print("\n✓ Site theme features:")
    print("  • Primary blue: #2D0140 (main data)")
    print("  • Accent blue: #220126 (secondary)")
    print("  • Medium gray: #495773 (tertiary)")
    print("  • Light backgrounds: '#F8F8FF', #495773")
    print("  • Dark text: #220126")
    print("  • Consistent zero lines in Panel A & C")
    print("  • Solid vertical lines in Panel B")
    print("  • Y-axis to 0.5 in Panel A, 120 in Panel B")

if __name__ == "__main__":
    main()
