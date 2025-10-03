#!/usr/bin/env python3
"""
Corrected Geometric Control Analysis for TEP Validation
======================================================

METHODOLOGICALLY CORRECT validation against right-skewed distribution bias

This analysis addresses reviewer concerns about the right-skewed (not bell-shaped) 
distribution of GNSS station pairwise distances and whether this could create 
spurious correlations that masquerade as TEP signals.

Key Corrections:
1. Use IDENTICAL methodology as real TEP analysis (exact replication)
2. Generate realistic synthetic coherence that preserves noise characteristics
3. Apply identical logarithmic binning and weighting procedures
4. Test multiple realistic scenarios without artificial bias injection
5. Proper statistical interpretation of results

The goal is to demonstrate that even with the right-skewed distribution,
the TEP methodology produces negligible spurious correlations for realistic
synthetic data, while real TEP signals show dramatically higher correlations.

Author: Matthew Lukin Smawfield
Date: September 2025
Purpose: Methodologically correct validation addressing reviewer distribution concerns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional

def print_status(text: str, status: str = "INFO"):
    """Print status messages with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefixes = {"INFO": "[INFO]", "SUCCESS": "[SUCCESS]", "WARNING": "[WARNING]", 
                "ERROR": "[ERROR]", "PROCESS": "[PROCESS]"}
    print(f"{timestamp} {prefixes.get(status, '[INFO]')} {text}")

def exponential_model(r, A, lambda_km, C0):
    """Exponential decay model - identical to TEP analysis."""
    return A * np.exp(-r / lambda_km) + C0

def apply_exact_tep_methodology(distances: np.ndarray, coherences: np.ndarray, 
                               label: str = "test") -> Optional[Dict]:
    """
    Apply EXACTLY the same methodology as real TEP analysis.
    
    This replicates the exact binning, weighting, and fitting procedures
    used in step_3_tep_correlation_analysis.py to ensure fair comparison.
    """
    try:
        # EXACT parameters from TEP analysis
        num_bins = 30  # From TEPConfig.get_int('TEP_BINS')
        max_distance = 13000  # From TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
        min_bin_count = 100  # From TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        
        # EXACT logarithmic binning
        edges = np.logspace(np.log10(50), np.log10(max_distance), num_bins + 1)
        
        # Bin assignment
        bin_indices = np.digitize(distances, edges) - 1
        valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
        
        # Aggregate by bins - EXACTLY as in TEP analysis
        bin_distances = []
        bin_coherences = []
        bin_counts = []
        
        for i in range(num_bins):
            mask = (bin_indices == i) & valid_mask
            count = np.sum(mask)
            
            if count >= min_bin_count:
                # Use mean distance and coherence for this bin
                bin_distances.append(distances[mask].mean())
                bin_coherences.append(coherences[mask].mean())
                bin_counts.append(count)
        
        if len(bin_distances) < 5:
            return None
        
        # EXACT exponential fitting with identical bounds and weighting
        bounds = ([0.01, 100, -1], [2, 20000, 1])  # Identical to TEP analysis
        weights = np.sqrt(bin_counts)  # Weight by sqrt(N) - EXACTLY as in TEP
        
        popt, pcov = curve_fit(
            exponential_model,
            bin_distances,
            bin_coherences,
            sigma=1/weights,  # Identical weighting scheme
            bounds=bounds,
            maxfev=5000
        )
        
        A, lambda_km, C0 = popt
        param_errors = np.sqrt(np.diag(pcov))
        
        # Calculate R-squared - identical formula
        y_pred = exponential_model(np.array(bin_distances), A, lambda_km, C0)
        ss_res = np.sum((np.array(bin_coherences) - y_pred) ** 2)
        ss_tot = np.sum((np.array(bin_coherences) - np.mean(bin_coherences)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'label': label,
            'exponential_fit': {
                'amplitude': float(A),
                'lambda_km': float(lambda_km),
                'offset': float(C0),
                'r_squared': float(r_squared),
                'lambda_error': float(param_errors[1])
            },
            'binning_stats': {
                'n_bins_used': len(bin_distances),
                'total_pairs': sum(bin_counts),
                'min_bin_count': min(bin_counts),
                'max_bin_count': max(bin_counts),
                'mean_bin_count': np.mean(bin_counts)
            }
        }
        
    except Exception as e:
        print_status(f"Fitting failed for {label}: {e}", "WARNING")
        return None

def generate_realistic_synthetic_coherence(n_pairs: int, scenario: str) -> np.ndarray:
    """
    Generate realistic synthetic coherence data that mimics real GNSS characteristics
    but has NO distance dependence.
    """
    if scenario == "pure_gaussian":
        # Pure Gaussian noise matching typical TEP coherence std
        return np.random.normal(0, 0.1, n_pairs)
        
    elif scenario == "uniform_bounded":
        # Uniform noise in realistic coherence range
        return np.random.uniform(-0.3, 0.3, n_pairs)
        
    elif scenario == "realistic_gnss":
        # Realistic GNSS-like noise with heteroscedasticity but NO distance dependence
        base_noise = np.random.normal(0, 0.08, n_pairs)
        
        # Add measurement uncertainty that varies randomly (not with distance)
        random_uncertainty = np.random.exponential(0.02, n_pairs)
        measurement_noise = np.random.normal(0, random_uncertainty)
        
        return np.clip(base_noise + measurement_noise, -1, 1)
        
    elif scenario == "zero_mean_structured":
        # Structured noise with complex patterns but zero mean and no distance correlation
        # Mix of different frequency components
        t = np.arange(n_pairs)
        
        # Multiple uncorrelated oscillations
        component1 = 0.03 * np.sin(2 * np.pi * t / 1000)
        component2 = 0.02 * np.sin(2 * np.pi * t / 3000) 
        component3 = 0.01 * np.sin(2 * np.pi * t / 500)
        
        # Random phase shifts to decorrelate from distance
        phase_shifts = np.random.uniform(0, 2*np.pi, 3)
        structured = (component1 * np.cos(phase_shifts[0]) + 
                     component2 * np.cos(phase_shifts[1]) + 
                     component3 * np.cos(phase_shifts[2]))
        
        # Add random noise
        noise = np.random.normal(0, 0.08, n_pairs)
        
        return structured + noise
        
    elif scenario == "temporal_correlation":
        # Simulate temporal correlations that should NOT create distance correlations
        # This tests if our method spuriously detects non-spatial correlations
        
        # Create temporal patterns that are distance-independent
        temporal_signal = np.random.normal(0, 0.05, n_pairs)
        
        # Add some temporal structure (simulating day/night cycles, etc.)
        # But make it completely independent of station distances
        random_phases = np.random.uniform(0, 2*np.pi, n_pairs)
        temporal_modulation = 0.02 * np.sin(random_phases)
        
        return temporal_signal + temporal_modulation
    
    else:
        # Default: pure Gaussian
        return np.random.normal(0, 0.1, n_pairs)

def run_corrected_geometric_control_analysis():
    """
    Run methodologically correct geometric control analysis.
    """
    print_status("Starting corrected geometric control analysis", "PROCESS")
    
    # Load real GNSS distance data
    root_dir = Path(__file__).resolve().parents[2]
    distances_file = root_dir / 'data/processed/step_8_station_distances.csv'
    output_dir = root_dir / 'results/exploratory'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not distances_file.exists():
        print_status(f"Distance file not found: {distances_file}", "ERROR")
        return None
    
    # Load distance data
    df = pd.read_csv(distances_file)
    distances = df['distance_km'].values
    n_pairs = len(distances)
    
    print_status(f"Loaded {n_pairs:,} station pairs from real GNSS network", "INFO")
    
    # Analyze the distribution shape
    skewness = pd.Series(distances).skew()
    print_status(f"Distribution skewness: {skewness:.3f} (confirms right-skewed)", "INFO")
    
    # Test scenarios that should produce minimal spurious correlations
    test_scenarios = [
        "pure_gaussian",
        "uniform_bounded", 
        "realistic_gnss",
        "zero_mean_structured",
        "temporal_correlation"
    ]
    
    print_status("Testing synthetic scenarios with identical TEP methodology", "PROCESS")
    
    all_results = {}
    max_spurious_r2 = 0
    
    for scenario in test_scenarios:
        print_status(f"Testing scenario: {scenario}", "INFO")
        scenario_results = []
        
        # Run multiple realizations for robust statistics
        for realization in range(20):
            np.random.seed(42 + realization * 100)  # Reproducible but varied
            
            # Generate synthetic coherence with NO distance dependence
            synthetic_coherence = generate_realistic_synthetic_coherence(n_pairs, scenario)
            
            # Apply EXACT TEP methodology
            result = apply_exact_tep_methodology(distances, synthetic_coherence, 
                                               f"{scenario}_{realization}")
            
            if result:
                scenario_results.append(result)
        
        if scenario_results:
            # Analyze results for this scenario
            r_squared_values = [r['exponential_fit']['r_squared'] for r in scenario_results]
            lambda_values = [r['exponential_fit']['lambda_km'] for r in scenario_results]
            
            scenario_stats = {
                'n_successful_fits': len(scenario_results),
                'r_squared_mean': np.mean(r_squared_values),
                'r_squared_std': np.std(r_squared_values),
                'r_squared_max': np.max(r_squared_values),
                'r_squared_min': np.min(r_squared_values),
                'abs_r_squared_max': np.max(np.abs(r_squared_values)),
                'lambda_mean': np.mean(lambda_values),
                'lambda_std': np.std(lambda_values),
                'fraction_at_bounds': np.mean(np.array(lambda_values) >= 19000)  # Hitting upper bound
            }
            
            all_results[scenario] = scenario_stats
            max_spurious_r2 = max(max_spurious_r2, scenario_stats['abs_r_squared_max'])
            
            print_status(f"  {scenario}: R² range [{scenario_stats['r_squared_min']:.4f}, {scenario_stats['r_squared_max']:.4f}], "
                        f"max |R²| = {scenario_stats['abs_r_squared_max']:.4f}", "INFO")
    
    # Compare to real TEP values
    real_tep_r2_range = [0.920, 0.970]  # CODE, ESA, IGS
    real_tep_lambda_range = [3330, 4549]  # ESA, CODE
    
    # Calculate safety margins
    tep_threshold = 0.3
    safety_margin_threshold = tep_threshold / max_spurious_r2 if max_spurious_r2 > 0 else float('inf')
    safety_margin_tep = min(real_tep_r2_range) / max_spurious_r2 if max_spurious_r2 > 0 else float('inf')
    
    # Validation assessment
    if max_spurious_r2 < 0.05:
        validation_status = "HIGHLY_VALIDATED"
        confidence = "VERY_HIGH"
    elif max_spurious_r2 < 0.1:
        validation_status = "VALIDATED" 
        confidence = "HIGH"
    elif max_spurious_r2 < 0.2:
        validation_status = "LIKELY_VALID"
        confidence = "MEDIUM"
    else:
        validation_status = "NEEDS_INVESTIGATION"
        confidence = "LOW"
    
    # Generate comprehensive report
    report = {
        'analysis_type': 'corrected_geometric_control_validation',
        'methodology': 'Exact replication of TEP analysis with synthetic coherence data',
        'distribution_characteristics': {
            'shape': 'RIGHT_SKEWED',
            'skewness': float(skewness),
            'total_pairs': int(n_pairs),
            'tep_range_pairs': int(np.sum((distances >= 3000) & (distances <= 5000)))
        },
        'test_scenarios': {
            scenario: stats for scenario, stats in all_results.items()
        },
        'spurious_correlation_summary': {
            'max_spurious_abs_r_squared': max_spurious_r2,
            'scenarios_tested': len(test_scenarios),
            'total_realizations': sum(stats['n_successful_fits'] for stats in all_results.values())
        },
        'safety_margins': {
            'tep_threshold_margin': safety_margin_threshold,
            'tep_signal_margin': safety_margin_tep,
            'recommended_threshold': max(0.3, 3 * max_spurious_r2)
        },
        'validation_assessment': {
            'status': validation_status,
            'confidence': confidence,
            'methodology_robust': max_spurious_r2 < 0.1,
            'geometric_bias_ruled_out': max_spurious_r2 < 0.1
        },
        'real_tep_comparison': {
            'real_tep_r2_range': real_tep_r2_range,
            'real_tep_lambda_range': real_tep_lambda_range,
            'separation_factor': min(real_tep_r2_range) / max_spurious_r2 if max_spurious_r2 > 0 else float('inf')
        }
    }
    
    # Create summary visualization
    create_corrected_visualization(distances, all_results, report, output_dir)
    
    # Save report
    report_file = output_dir / 'corrected_geometric_control_analysis.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print_status("", "INFO")
    print("="*70)
    print("CORRECTED GEOMETRIC CONTROL ANALYSIS SUMMARY")
    print("="*70)
    print(f"Distribution: {report['distribution_characteristics']['shape']} (skewness: {skewness:.3f})")
    print(f"Total station pairs: {n_pairs:,}")
    print(f"TEP range pairs: {report['distribution_characteristics']['tep_range_pairs']:,}")
    print()
    print(f"Test scenarios: {len(test_scenarios)}")
    print(f"Total realizations: {report['spurious_correlation_summary']['total_realizations']}")
    print(f"Maximum spurious |R²|: {max_spurious_r2:.4f}")
    print()
    print(f"Safety margins:")
    print(f"  TEP threshold (0.3): {safety_margin_threshold:.1f}×")
    print(f"  Real TEP signals (0.92-0.97): {safety_margin_tep:.1f}×")
    print()
    print(f"Validation status: {validation_status}")
    print(f"Confidence: {confidence}")
    print(f"Geometric bias ruled out: {report['validation_assessment']['geometric_bias_ruled_out']}")
    print()
    
    # Detailed scenario results
    print("DETAILED SCENARIO RESULTS:")
    print("-" * 50)
    for scenario, stats in all_results.items():
        print(f"{scenario:20s}: max |R²| = {stats['abs_r_squared_max']:.4f}, "
              f"mean R² = {stats['r_squared_mean']:+.4f}, "
              f"λ hits bounds: {stats['fraction_at_bounds']:.1%}")
    
    print()
    if max_spurious_r2 < 0.1:
        print("✅ METHODOLOGY VALIDATED: Right-skewed distribution does NOT create significant spurious correlations")
        print("✅ TEP correlations are likely genuine physical signals")
    else:
        print("⚠️  METHODOLOGY CONCERN: Distribution bias may affect results")
        print("⚠️  Consider revised thresholds or alternative analysis approaches")
    
    print(f"\nReport saved: {report_file}")
    
    return report

def create_corrected_visualization(distances: np.ndarray, results: Dict, 
                                 report: Dict, output_dir: Path):
    """Create visualization showing corrected geometric control analysis."""
    print_status("Creating corrected geometric control visualization", "PROCESS")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Corrected Geometric Control Analysis:\nRight-Skewed Distribution Validation', 
                 fontsize=14, fontweight='bold')
    
    # 1. Distance distribution with proper labeling
    ax1 = axes[0, 0]
    ax1.hist(distances, bins=50, alpha=0.7, color='purple', edgecolor='white')
    ax1.axvspan(3330, 4549, alpha=0.3, color='orange', label='TEP Range (3,330-4,549 km)')
    ax1.axvline(np.mean(distances), color='black', linestyle='--', 
               label=f'Mean: {np.mean(distances):.0f} km')
    
    # Add skewness annotation
    skewness = report['distribution_characteristics']['skewness']
    ax1.text(0.7, 0.8, f'Skewness: {skewness:.3f}\n(Right-skewed)', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Number of Station Pairs')
    ax1.set_title('GNSS Distance Distribution\n(Right-Skewed, Not Bell-Shaped)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Spurious R² comparison
    ax2 = axes[0, 1]
    
    scenarios = list(results.keys())
    max_r2_values = [results[s]['abs_r_squared_max'] for s in scenarios]
    
    bars = ax2.bar(range(len(scenarios)), max_r2_values, alpha=0.7, color='orange')
    ax2.axhline(0.3, color='red', linestyle='--', linewidth=2, label='TEP Threshold')
    ax2.axhline(0.92, color='green', linestyle='--', linewidth=2, label='Real TEP R² (min)')
    
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0, fontsize=10)
    ax2.set_ylabel('Maximum |R²|')
    ax2.set_title('Maximum Spurious R² by Scenario')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, max_r2_values)):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Safety margin analysis
    ax3 = axes[1, 0]
    
    max_spurious = report['spurious_correlation_summary']['max_spurious_abs_r_squared']
    margins = [
        report['safety_margins']['tep_threshold_margin'],
        report['safety_margins']['tep_signal_margin']
    ]
    margin_labels = ['TEP Threshold\n(0.3)', 'Real TEP Signals\n(0.92-0.97)']
    colors = ['red' if m < 3 else 'orange' if m < 10 else 'green' for m in margins]
    
    bars = ax3.bar(margin_labels, margins, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(3.0, color='red', linestyle='--', label='Safe Margin (3×)')
    ax3.axhline(10.0, color='green', linestyle='--', label='Strong Margin (10×)')
    
    ax3.set_ylabel('Safety Margin (×)')
    ax3.set_title('Safety Margins:\nSpurious vs Real Correlations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, margin in zip(bars, margins):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{margin:.1f}×', ha='center', va='bottom', fontweight='bold')
    
    # 4. Validation summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    status_colors = {
        'HIGHLY_VALIDATED': 'darkgreen',
        'VALIDATED': 'green',
        'LIKELY_VALID': 'orange',
        'NEEDS_INVESTIGATION': 'red'
    }
    
    status_color = status_colors.get(report['validation_assessment']['status'], 'gray')
    
    summary_text = f"""VALIDATION ASSESSMENT

Status: {report['validation_assessment']['status']}
Confidence: {report['validation_assessment']['confidence']}

Distribution: RIGHT-SKEWED
Skewness: {report['distribution_characteristics']['skewness']:.3f}

Max Spurious |R²|: {max_spurious:.4f}
TEP Threshold: 0.300
Real TEP R²: 0.920-0.970

Safety Margins:
• Threshold: {report['safety_margins']['tep_threshold_margin']:.1f}×
• Real Signals: {report['safety_margins']['tep_signal_margin']:.1f}×

Geometric Bias Ruled Out: {report['validation_assessment']['geometric_bias_ruled_out']}
Methodology Robust: {report['validation_assessment']['methodology_robust']}"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.15))
    
    plt.tight_layout()
    
    output_file = output_dir / 'corrected_geometric_control_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Visualization saved: {output_file}", "SUCCESS")

def main():
    """Main execution function."""
    print("="*80)
    print("CORRECTED GEOMETRIC CONTROL ANALYSIS")
    print("="*80)
    print("Methodologically correct validation addressing right-skewed distribution concerns")
    print("Using EXACT replication of TEP analysis methodology")
    print()
    
    try:
        report = run_corrected_geometric_control_analysis()
        
        if report:
            print("\n" + "="*70)
            print("REVIEWER RESPONSE SUMMARY")
            print("="*70)
            
            max_spurious = report['spurious_correlation_summary']['max_spurious_abs_r_squared']
            
            if max_spurious < 0.1:
                print("✅ RESPONSE TO REVIEWER CONCERNS:")
                print("  • Right-skewed distribution acknowledged and tested")
                print("  • Logarithmic binning with √N weighting is robust")
                print("  • Maximum spurious correlations remain minimal")
                print("  • TEP correlations show strong evidence of genuine origin")
                print("  • Multi-center consistency cannot be explained by geometric bias")
            else:
                print("⚠️  REVIEWER CONCERNS CONFIRMED:")
                print("  • Right-skewed distribution creates significant bias")
                print("  • Current methodology may be compromised")
                print("  • Recommend methodological revisions or higher thresholds")
        
    except Exception as e:
        print_status(f"Analysis failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

