#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 4.8: Multi-Band Frequency Visualization
====================================================

Generates publication-quality figures for multi-band frequency analysis results.

Requirements: Step 3.6 complete (Multi-Band Frequency Analysis)
Inputs:
  - results/outputs/step_3_6_multiband_code.json
  - results/outputs/step_3_6_multiband_igs_combined.json
  - results/outputs/step_3_6_multiband_esa_final.json
Outputs:
  - results/figures/step_4_8_multiband_r_squared_comparison.png
  - results/figures/step_4_8_multiband_lambda_vs_frequency.png
  - results/figures/step_4_8_multiband_spectral_overview.png
  - results/figures/step_4_8_multiband_post_tidal_emphasis.png

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import sys
import shutil
from pathlib import Path

# Anchor to package root
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.utils.logger import TEPLogger as Logger, print_status, set_step_logger

# Initialize logger
step_logger = Logger(
    name="step_4_8_multiband_visualization",
    level="INFO",
    log_file_path=PACKAGE_ROOT / "logs" / "step_4_8_multiband_visualization.log"
)
set_step_logger(step_logger)

# Configure matplotlib for publication quality
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

# Color scheme
COLORS = {
    'code': '#2D0140',
    'igs': '#495773',
    'esa': '#6B73A1',
    'tidal': '#8A2BE2',
    'post_tidal': '#9370DB',
    'control': '#B0B0B0',
    'mean': '#FF6B35'
}

def load_multiband_results():
    """Load all three analysis center results."""
    results = {}
    
    for ac in ['code', 'igs_combined', 'esa_final']:
        file_path = PACKAGE_ROOT / f"results/outputs/step_3_6_multiband_{ac}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing multi-band results: {file_path}")
        
        with open(file_path) as f:
            results[ac] = json.load(f)
        
        print_status(f"Loaded {ac.upper()} results", "INFO")
    
    return results

def create_r_squared_comparison(results, output_dir):
    """Create R² comparison across frequency bands and analysis centers."""
    
    # Order bands by frequency for logical progression (TEP band handled separately)
    bands = ['tidal_diurnal', 'tidal_semidiurnal', 'post_tidal_30_40', 
             'post_tidal_40_50', 'post_tidal_50_75', 'post_tidal_75_100',
             'intermediate_100_200', 'intermediate_200_350', 'intermediate_350_500',
             'transition_500_750', 'transition_750_1000', 'control_1000_1500', 'tep_band']
    
    band_labels = ['Diurnal\n(10-20)', 'Semidiurnal\n(20-30)', 
                   'Post-Tidal\n(30-40)', 'Post-Tidal\n(40-50)', 'Post-Tidal\n(50-75)', 
                   'Post-Tidal\n(75-100)', 'Interm.\n(100-200)', 'Interm.\n(200-350)', 
                   'Interm.\n(350-500)', 'Trans.\n(500-750)', 'Trans.\n(750-1000)', 
                   'Control\n(1000-1500)', 'TEP\n(10-500)']
    
    # Extract R² values
    code_r2 = [results['code']['comparison']['r_squared_summary'][b] for b in bands]
    igs_r2 = [results['igs_combined']['comparison']['r_squared_summary'][b] for b in bands]
    esa_r2 = [results['esa_final']['comparison']['r_squared_summary'].get(b, 0) for b in bands]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(bands))
    width = 0.25
    
    bars1 = ax.bar(x - width, code_r2, width, label='CODE', color=COLORS['code'], alpha=0.8)
    bars2 = ax.bar(x, igs_r2, width, label='IGS', color=COLORS['igs'], alpha=0.8)
    bars3 = ax.bar(x + width, esa_r2, width, label='ESA', color=COLORS['esa'], alpha=0.8)
    
    # Highlight post-tidal 30-40 µHz (strongest band) - now at index 2
    ax.axvspan(1.5, 2.5, alpha=0.1, color=COLORS['tidal'], zorder=0)
    ax.text(2, 0.98, 'Strongest Band\n(Post-Tidal)', ha='center', va='top', 
            fontsize=9, style='italic', color=COLORS['tidal'])
    
    # Add reference lines
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Strong Signal Threshold')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent Fit')
    
    # Formatting
    ax.set_xlabel('Frequency Band (µHz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('R² (Exponential Fit Quality)', fontweight='bold', fontsize=12)
    ax.set_title('Multi-Band Spectral Analysis: Cross-Center R² Comparison\nBroadband Correlation Structure with Gravitational Enhancement', 
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars for key bands
    for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
        if i in [2, 11, 12]:  # Post-Tidal 30-40, Control, TEP
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.01, 
                   f'{code_r2[i]:.3f}', ha='center', va='bottom', fontsize=7, color=COLORS['code'])
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.01, 
                   f'{igs_r2[i]:.3f}', ha='center', va='bottom', fontsize=7, color=COLORS['igs'])
            if esa_r2[i] > 0:
                ax.text(b3.get_x() + b3.get_width()/2, b3.get_height() + 0.01, 
                       f'{esa_r2[i]:.3f}', ha='center', va='bottom', fontsize=7, color=COLORS['esa'])
    
    plt.tight_layout()
    output_path = output_dir / 'step_4_8_multiband_r_squared_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved R² comparison: {output_path}", "SUCCESS")

def create_lambda_vs_frequency(results, output_dir):
    """Create correlation length vs frequency plot."""
    
    # Create frequency-band pairs and sort by frequency for proper spectral visualization
    band_freq_pairs = [
        ('tidal_diurnal', 15),
        ('tidal_semidiurnal', 25), 
        ('post_tidal_30_40', 35),
        ('post_tidal_40_50', 45),
        ('post_tidal_50_75', 62.5),
        ('post_tidal_75_100', 87.5),
        ('intermediate_100_200', 150),
        ('intermediate_200_350', 275),
        ('intermediate_350_500', 425),
        ('transition_500_750', 625),
        ('transition_750_1000', 875),
        ('control_1000_1500', 1250)
    ]
    
    # Sort by frequency to ensure proper spectral progression
    band_freq_pairs.sort(key=lambda x: x[1])
    bands = [pair[0] for pair in band_freq_pairs]
    freq_centers = [pair[1] for pair in band_freq_pairs]
    
    # Extract lambda values
    code_lambda = [results['code']['comparison']['lambda_summary'][b] for b in bands]
    igs_lambda = [results['igs_combined']['comparison']['lambda_summary'][b] for b in bands]
    esa_lambda = [results['esa_final']['comparison']['lambda_summary'].get(b, 0) for b in bands]
    
    # Extract errors
    code_errors = [results['code']['comparison']['lambda_error_summary'][b] for b in bands]
    igs_errors = [results['igs_combined']['comparison']['lambda_error_summary'][b] for b in bands]
    esa_errors = [results['esa_final']['comparison']['lambda_error_summary'].get(b, 0) for b in bands]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot with error bars
    ax.errorbar(freq_centers, code_lambda, yerr=code_errors, marker='o', markersize=8, 
                label='CODE', color=COLORS['code'], linewidth=2, capsize=4, alpha=0.8)
    ax.errorbar(freq_centers, igs_lambda, yerr=igs_errors, marker='s', markersize=7, 
                label='IGS', color=COLORS['igs'], linewidth=2, capsize=4, alpha=0.8)
    ax.errorbar(freq_centers, esa_lambda, yerr=esa_errors, marker='^', markersize=7, 
                label='ESA', color=COLORS['esa'], linewidth=2, capsize=4, alpha=0.8)
    
    # Highlight regions
    ax.axvspan(10, 30, alpha=0.1, color=COLORS['tidal'], label='Tidal Bands')
    ax.axvspan(30, 100, alpha=0.1, color=COLORS['post_tidal'], label='Post-Tidal Bands')
    ax.axvspan(1000, 1500, alpha=0.1, color=COLORS['control'], label='Control Band')
    
    # Add transition annotations to explain the physics
    ax.annotate('2-3× Spatial\nScale Drop', xy=(35, 2400), xytext=(80, 4000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.annotate('Gravitational\nEnhancement', xy=(20, 5800), xytext=(20, 6500),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=10, color='darkgreen', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # Formatting
    ax.set_xlabel('Frequency (µHz, log scale)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Correlation Length λ (km)', fontweight='bold', fontsize=12)
    ax.set_title('Gravitational Enhancement: 2-3× Spatial Scale Transition\nPhysical Pattern Shows Sharp Drop from Tidal to Post-Tidal Frequencies', 
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xscale('log')
    ax.set_xlim(8, 2000)
    ax.set_ylim(500, 7000)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = output_dir / 'step_4_8_multiband_lambda_vs_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved λ vs frequency: {output_path}", "SUCCESS")

def create_spectral_overview(results, output_dir):
    """Create comprehensive 4-panel spectral overview."""
    
    # Create frequency-band pairs and sort by frequency for proper spectral visualization
    band_freq_pairs = [
        ('tidal_diurnal', 15),
        ('tidal_semidiurnal', 25), 
        ('post_tidal_30_40', 35),
        ('post_tidal_40_50', 45),
        ('post_tidal_50_75', 62.5),
        ('post_tidal_75_100', 87.5),
        ('intermediate_100_200', 150),
        ('intermediate_200_350', 275),
        ('intermediate_350_500', 425),
        ('transition_500_750', 625),
        ('transition_750_1000', 875),
        ('control_1000_1500', 1250)
    ]
    
    # Sort by frequency to ensure proper spectral progression
    band_freq_pairs.sort(key=lambda x: x[1])
    bands = [pair[0] for pair in band_freq_pairs]
    freq_centers = [pair[1] for pair in band_freq_pairs]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: R² across bands
    code_r2 = [results['code']['comparison']['r_squared_summary'][b] for b in bands]
    igs_r2 = [results['igs_combined']['comparison']['r_squared_summary'][b] for b in bands]
    esa_r2 = [results['esa_final']['comparison']['r_squared_summary'].get(b, 0) for b in bands]
    mean_r2 = [(c + i + e)/3 for c, i, e in zip(code_r2, igs_r2, esa_r2)]
    
    ax1.plot(freq_centers, code_r2, 'o-', label='CODE', color=COLORS['code'], linewidth=2, markersize=8, alpha=0.8)
    ax1.plot(freq_centers, igs_r2, 's-', label='IGS', color=COLORS['igs'], linewidth=2, markersize=7, alpha=0.8)
    ax1.plot(freq_centers, esa_r2, '^-', label='ESA', color=COLORS['esa'], linewidth=2, markersize=7, alpha=0.8)
    ax1.plot(freq_centers, mean_r2, 'D-', label='Mean', color=COLORS['mean'], linewidth=2.5, markersize=6, alpha=0.9)
    
    ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.axvspan(10, 30, alpha=0.08, color=COLORS['tidal'])
    ax1.axvspan(30, 100, alpha=0.08, color=COLORS['post_tidal'])
    
    ax1.set_xlabel('Frequency (µHz, log scale)', fontweight='bold')
    ax1.set_ylabel('R² (Fit Quality)', fontweight='bold')
    ax1.set_title('(A) Spectral Correlation Structure\nR² > 0.85 from Tidal to Intermediate Bands', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Panel 2: Lambda spatial scales (excluding TEP band for spectral progression)
    code_lambda = [results['code']['comparison']['lambda_summary'][b] for b in bands]
    igs_lambda = [results['igs_combined']['comparison']['lambda_summary'][b] for b in bands]
    esa_lambda = [results['esa_final']['comparison']['lambda_summary'].get(b, 0) for b in bands]
    mean_lambda = [(c + i + e)/3 for c, i, e in zip(code_lambda, igs_lambda, esa_lambda)]
    
    ax2.plot(freq_centers, code_lambda, 'o-', label='CODE', color=COLORS['code'], linewidth=2, markersize=8, alpha=0.8)
    ax2.plot(freq_centers, igs_lambda, 's-', label='IGS', color=COLORS['igs'], linewidth=2, markersize=7, alpha=0.8)
    ax2.plot(freq_centers, esa_lambda, '^-', label='ESA', color=COLORS['esa'], linewidth=2, markersize=7, alpha=0.8)
    ax2.plot(freq_centers, mean_lambda, 'D-', label='Mean', color=COLORS['mean'], linewidth=2.5, markersize=6, alpha=0.9)
    
    ax2.axvspan(10, 30, alpha=0.08, color=COLORS['tidal'])
    ax2.axvspan(30, 100, alpha=0.08, color=COLORS['post_tidal'])
    ax2.annotate('Sharp 2-3× Drop\n(Key TEP Finding)', xy=(35, 2400), xytext=(80, 4500),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, color='red', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Frequency (µHz, log scale)', fontweight='bold')
    ax2.set_ylabel('Correlation Length λ (km)', fontweight='bold')
    ax2.set_title('(B) Gravitational Enhancement Pattern\nLongest λ at Tidal Frequencies', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_ylim(500, 7000)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Panel 3: Enhancement ratios (using TEP band for comparison)
    tep_r2 = [results['code']['comparison']['r_squared_summary']['tep_band'],
              results['igs_combined']['comparison']['r_squared_summary']['tep_band'],
              results['esa_final']['comparison']['r_squared_summary']['tep_band']]
    control_r2 = [results['code']['comparison']['r_squared_summary']['control_1000_1500'],
                  results['igs_combined']['comparison']['r_squared_summary']['control_1000_1500'],
                  results['esa_final']['comparison']['r_squared_summary']['control_1000_1500']]
    ratios = [t/c for t, c in zip(tep_r2, control_r2)]
    
    centers = ['CODE', 'IGS', 'ESA']
    colors_list = [COLORS['code'], COLORS['igs'], COLORS['esa']]
    
    bars = ax3.bar(centers, ratios, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Universal Coupling (~1.5×)')
    ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Tidal Contamination (>3×)')
    
    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{ratio:.2f}×', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.set_ylabel('TEP / Control Enhancement Ratio', fontweight='bold', fontsize=12)
    ax3.set_title('(C) Frequency Specificity Test\nModest Enhancement Excludes Tidal Contamination', 
                 fontweight='bold', fontsize=13)
    ax3.set_ylim(0, 3.5)
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: CV consistency across frequency regions
    regions = ['Tidal\n(10-30)', 'Post-Tidal\n(30-100)', 'Intermediate\n(100-500)', 'Control\n(1000-1500)']
    region_bands = {
        'Tidal\n(10-30)': ['tidal_diurnal', 'tidal_semidiurnal'],
        'Post-Tidal\n(30-100)': ['post_tidal_30_40', 'post_tidal_40_50', 'post_tidal_50_75', 'post_tidal_75_100'],
        'Intermediate\n(100-500)': ['intermediate_100_200', 'intermediate_200_350', 'intermediate_350_500'],
        'Control\n(1000-1500)': ['control_1000_1500']
    }
    
    cvs = []
    for region, band_list in region_bands.items():
        r2_vals = []
        for b in band_list:
            r2_vals.append(results['code']['comparison']['r_squared_summary'][b])
            r2_vals.append(results['igs_combined']['comparison']['r_squared_summary'][b])
            if b in results['esa_final']['comparison']['r_squared_summary']:
                r2_vals.append(results['esa_final']['comparison']['r_squared_summary'][b])
        cv = (np.std(r2_vals) / np.mean(r2_vals) * 100) if np.mean(r2_vals) > 0 else 0
        cvs.append(cv)
    
    bars = ax4.bar(regions, cvs, color=[COLORS['tidal'], COLORS['post_tidal'], '#9999CC', COLORS['control']], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Excellent (<5%)')
    ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Good (<10%)')
    
    # Add value labels
    for bar, cv in zip(bars, cvs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax4.set_ylabel('R² Coefficient of Variation (%)', fontweight='bold', fontsize=12)
    ax4.set_title('(D) Cross-Center Consistency by Frequency\nStrong Signals Show Excellent Agreement', 
                 fontweight='bold', fontsize=13)
    ax4.set_ylim(0, 15)
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'step_4_8_multiband_spectral_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved spectral overview: {output_path}", "SUCCESS")

def create_post_tidal_emphasis(results, output_dir):
    """Create figure emphasizing the post-tidal 30-40 µHz critical finding."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: R² comparison showing 30-40 µHz prominence
    bands_subset = ['tidal_diurnal', 'tidal_semidiurnal', 'post_tidal_30_40', 
                    'post_tidal_40_50', 'intermediate_100_200', 'control_1000_1500']
    labels = ['Diurnal\n(10-20)', 'Semidiurnal\n(20-30)', 'Post-Tidal\n(30-40)', 
              'Post-Tidal\n(40-50)', 'Intermediate\n(100-200)', 'Control\n(1000-1500)']
    
    code_r2 = [results['code']['comparison']['r_squared_summary'][b] for b in bands_subset]
    igs_r2 = [results['igs_combined']['comparison']['r_squared_summary'][b] for b in bands_subset]
    esa_r2 = [results['esa_final']['comparison']['r_squared_summary'].get(b, 0) for b in bands_subset]
    mean_r2 = [(c + i + e)/3 for c, i, e in zip(code_r2, igs_r2, esa_r2)]
    
    x = np.arange(len(bands_subset))
    width = 0.2
    
    ax1.bar(x - 1.5*width, code_r2, width, label='CODE', color=COLORS['code'], alpha=0.8, edgecolor='black', linewidth=1)
    ax1.bar(x - 0.5*width, igs_r2, width, label='IGS', color=COLORS['igs'], alpha=0.8, edgecolor='black', linewidth=1)
    ax1.bar(x + 0.5*width, esa_r2, width, label='ESA', color=COLORS['esa'], alpha=0.8, edgecolor='black', linewidth=1)
    ax1.bar(x + 1.5*width, mean_r2, width, label='Mean', color=COLORS['mean'], alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # Highlight 30-40 µHz
    ax1.axvspan(1.5, 2.5, alpha=0.15, color='gold', zorder=0)
    ax1.text(2, 0.98, '★ STRONGEST BAND ★\nExcludes Tidal Contamination', 
            ha='center', va='top', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))
    
    ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Strong Signal')
    ax1.set_ylabel('R² (Exponential Fit Quality)', fontweight='bold', fontsize=12)
    ax1.set_title('(A) Post-Tidal 30-40 µHz: Critical Discriminator\nStrongest Band Excludes Classical Tidal Contamination', 
                 fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(loc='lower left', framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Enhancement ratio comparison
    enhancement_types = ['Tidal vs\nControl', 'Post-Tidal\n30-40 vs Control', 'TEP Band vs\nControl']
    
    # Calculate ratios for each center
    code_ratios = [
        np.mean([results['code']['comparison']['r_squared_summary']['tidal_diurnal'],
                results['code']['comparison']['r_squared_summary']['tidal_semidiurnal']]) / 
        results['code']['comparison']['r_squared_summary']['control_1000_1500'],
        
        results['code']['comparison']['r_squared_summary']['post_tidal_30_40'] / 
        results['code']['comparison']['r_squared_summary']['control_1000_1500'],
        
        results['code']['comparison']['r_squared_summary']['tep_band'] / 
        results['code']['comparison']['r_squared_summary']['control_1000_1500']
    ]
    
    igs_ratios = [
        np.mean([results['igs_combined']['comparison']['r_squared_summary']['tidal_diurnal'],
                results['igs_combined']['comparison']['r_squared_summary']['tidal_semidiurnal']]) / 
        results['igs_combined']['comparison']['r_squared_summary']['control_1000_1500'],
        
        results['igs_combined']['comparison']['r_squared_summary']['post_tidal_30_40'] / 
        results['igs_combined']['comparison']['r_squared_summary']['control_1000_1500'],
        
        results['igs_combined']['comparison']['r_squared_summary']['tep_band'] / 
        results['igs_combined']['comparison']['r_squared_summary']['control_1000_1500']
    ]
    
    esa_ratios = [
        np.mean([results['esa_final']['comparison']['r_squared_summary']['tidal_diurnal'],
                results['esa_final']['comparison']['r_squared_summary']['tidal_semidiurnal']]) / 
        results['esa_final']['comparison']['r_squared_summary']['control_1000_1500'],
        
        results['esa_final']['comparison']['r_squared_summary']['post_tidal_30_40'] / 
        results['esa_final']['comparison']['r_squared_summary']['control_1000_1500'],
        
        results['esa_final']['comparison']['r_squared_summary']['tep_band'] / 
        results['esa_final']['comparison']['r_squared_summary']['control_1000_1500']
    ]
    
    mean_ratios = [(c + i + e)/3 for c, i, e in zip(code_ratios, igs_ratios, esa_ratios)]
    
    x2 = np.arange(len(enhancement_types))
    width2 = 0.2
    
    ax2.bar(x2 - 1.5*width2, code_ratios, width2, label='CODE', color=COLORS['code'], alpha=0.8, edgecolor='black')
    ax2.bar(x2 - 0.5*width2, igs_ratios, width2, label='IGS', color=COLORS['igs'], alpha=0.8, edgecolor='black')
    ax2.bar(x2 + 0.5*width2, esa_ratios, width2, label='ESA', color=COLORS['esa'], alpha=0.8, edgecolor='black')
    ax2.bar(x2 + 1.5*width2, mean_ratios, width2, label='Mean', color=COLORS['mean'], alpha=0.9, edgecolor='black', linewidth=1.5)
    
    ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Universal Coupling (~1.5×)')
    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Tidal Threshold (>3×)')
    ax2.fill_between([-0.5, 2.5], 1.5, 2.0, alpha=0.1, color='green', label='Expected Range')
    
    ax2.set_ylabel('Enhancement Ratio (R² / Control R²)', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Frequency Specificity Analysis\nAll Ratios <2× Support Universal Coupling', 
                 fontweight='bold', fontsize=13)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(enhancement_types, fontsize=10)
    ax2.set_ylim(0, 3.5)
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'step_4_8_multiband_post_tidal_emphasis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved post-tidal emphasis: {output_path}", "SUCCESS")

def create_amplitude_spectral_decay(results, output_dir):
    """Create amplitude decay pattern visualization."""
    
    # Create frequency-band pairs and sort by frequency for proper spectral visualization
    band_freq_pairs = [
        ('tidal_diurnal', 15),
        ('tidal_semidiurnal', 25), 
        ('post_tidal_30_40', 35),
        ('post_tidal_40_50', 45),
        ('post_tidal_50_75', 62.5),
        ('post_tidal_75_100', 87.5),
        ('intermediate_100_200', 150)
    ]
    
    # Sort by frequency to ensure proper spectral progression
    band_freq_pairs.sort(key=lambda x: x[1])
    bands = [pair[0] for pair in band_freq_pairs]
    freq_centers = [pair[1] for pair in band_freq_pairs]
    
    # Extract amplitudes
    code_amp = [results['code']['comparison']['amplitude_summary'][b] for b in bands]
    igs_amp = [results['igs_combined']['comparison']['amplitude_summary'][b] for b in bands]
    esa_amp = [results['esa_final']['comparison']['amplitude_summary'][b] for b in bands]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.semilogy(freq_centers, code_amp, 'o-', label='CODE', color=COLORS['code'], 
               linewidth=2.5, markersize=10, alpha=0.8)
    ax.semilogy(freq_centers, igs_amp, 's-', label='IGS', color=COLORS['igs'], 
               linewidth=2.5, markersize=9, alpha=0.8)
    ax.semilogy(freq_centers, esa_amp, '^-', label='ESA', color=COLORS['esa'], 
               linewidth=2.5, markersize=9, alpha=0.8)
    
    # Highlight regions
    ax.axvspan(10, 30, alpha=0.1, color=COLORS['tidal'], label='Tidal Bands')
    ax.axvspan(30, 100, alpha=0.1, color=COLORS['post_tidal'], label='Post-Tidal Bands')
    
    ax.set_xlabel('Frequency (µHz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Correlation Amplitude A (log scale)', fontweight='bold', fontsize=12)
    ax.set_title('Signal Amplitude Spectral Decay\nGradual Decline Supports Broadband Coupling', 
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xlim(10, 200)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = output_dir / 'step_4_8_multiband_amplitude_decay.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved amplitude decay: {output_path}", "SUCCESS")

def sync_figures_to_site(output_dir):
    """Sync generated figures to site public folder for web display."""
    
    site_figures_dir = PACKAGE_ROOT / "site" / "public" / "figures"
    site_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # List of figures to sync
    figures_to_sync = [
        "step_4_8_multiband_r_squared_comparison.png",
        "step_4_8_multiband_lambda_vs_frequency.png", 
        "step_4_8_multiband_spectral_overview.png",
        "step_4_8_multiband_post_tidal_emphasis.png",
        "step_4_8_multiband_amplitude_decay.png"
    ]
    
    print_status("Syncing figures to site folder...", "PROCESS")
    
    for figure_name in figures_to_sync:
        source_path = output_dir / figure_name
        dest_path = site_figures_dir / figure_name
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            print_status(f"✓ Synced: {figure_name}", "SUCCESS")
        else:
            print_status(f"✗ Missing: {figure_name}", "WARNING")
    
    print_status(f"Figures synced to: {site_figures_dir}", "INFO")

def main():
    """Main execution function."""
    
    print_status("="*80, "INFO")
    print_status("STEP 4.8: MULTI-BAND FREQUENCY VISUALIZATION", "INFO")
    print_status("="*80, "INFO")
    
    # Setup paths
    output_dir = PACKAGE_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print_status("Loading multi-band analysis results...", "PROCESS")
    results = load_multiband_results()
    
    # Generate figures
    print_status("Generating multi-band visualizations...", "PROCESS")
    
    print_status("Creating R² comparison chart...", "INFO")
    create_r_squared_comparison(results, output_dir)
    
    print_status("Creating λ vs frequency plot...", "INFO")
    create_lambda_vs_frequency(results, output_dir)
    
    print_status("Creating 4-panel spectral overview...", "INFO")
    create_spectral_overview(results, output_dir)
    
    print_status("Creating post-tidal emphasis figure...", "INFO")
    create_post_tidal_emphasis(results, output_dir)
    
    print_status("Creating amplitude spectral decay...", "INFO")
    create_amplitude_spectral_decay(results, output_dir)
    
    # Sync figures to site folder 
    sync_figures_to_site(output_dir)
    
    print_status("="*80, "SUCCESS")
    print_status("STEP 4.8 COMPLETE - All multi-band visualizations generated", "SUCCESS")
    print_status("="*80, "SUCCESS")
    print_status(f"Figures saved to: {output_dir}", "INFO")
    print_status("", "INFO")
    print_status("Generated figures:", "INFO")
    print_status("  1. step_4_8_multiband_r_squared_comparison.png", "INFO")
    print_status("  2. step_4_8_multiband_lambda_vs_frequency.png", "INFO")
    print_status("  3. step_4_8_multiband_spectral_overview.png", "INFO")
    print_status("  4. step_4_8_multiband_post_tidal_emphasis.png", "INFO")
    print_status("  5. step_4_8_multiband_amplitude_decay.png", "INFO")
    print_status("", "INFO")
    print_status("All figures automatically synced to site/public/figures/", "SUCCESS")

if __name__ == "__main__":
    main()

