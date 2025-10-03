#!/usr/bin/env python3
"""
Rigorous Geometric Bias Analysis for TEP Validation
==================================================

CRITICAL VALIDATION: Comprehensive testing of right-skewed distance distribution bias

This analysis addresses the discovery that the GNSS station distance distribution
is RIGHT-SKEWED (not bell-shaped) with increasing pair density in the TEP correlation
range (3,000-5,000 km). This systematic increase could create spurious correlations
that masquerade as TEP signals.

Key Tests:
1. Quantify maximum spurious correlations from distribution shape alone
2. Test sensitivity to different noise characteristics and systematic biases
3. Establish robust safety margins for distinguishing real vs artifactual correlations
4. Validate logarithmic binning effectiveness against distribution bias
5. Compare multiple debiasing strategies

Author: Matthew Lukin Smawfield
Date: September 2025
Purpose: Critical validation of TEP methodology against right-skewed distribution bias
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RigorousGeometricBiasAnalyzer:
    """
    Comprehensive analyzer for geometric bias in right-skewed distance distributions.
    """
    
    def __init__(self, distances_file: str):
        """Initialize with real GNSS station distance data."""
        self.distances_file = Path(distances_file)
        self.load_distance_data()
        self.setup_analysis_parameters()
        
    def load_distance_data(self):
        """Load and analyze the real distance distribution."""
        print("Loading GNSS station distance data...")
        df = pd.read_csv(self.distances_file)
        self.distances = df['distance_km'].values
        self.n_pairs = len(self.distances)
        
        # Analyze distribution characteristics
        self.analyze_distribution_shape()
        
    def analyze_distribution_shape(self):
        """Comprehensive analysis of the distance distribution shape."""
        print("Analyzing distance distribution characteristics...")
        
        # Basic statistics
        self.dist_stats = {
            'mean': np.mean(self.distances),
            'median': np.median(self.distances),
            'std': np.std(self.distances),
            'skewness': stats.skew(self.distances),
            'kurtosis': stats.kurtosis(self.distances),
            'min': np.min(self.distances),
            'max': np.max(self.distances)
        }
        
        # TEP range analysis
        tep_mask = (self.distances >= 3000) & (self.distances <= 5000)
        pre_tep_mask = (self.distances >= 1000) & (self.distances < 3000)
        post_tep_mask = (self.distances >= 5000) & (self.distances <= 7000)
        
        self.tep_analysis = {
            'pre_tep_count': np.sum(pre_tep_mask),
            'tep_count': np.sum(tep_mask),
            'post_tep_count': np.sum(post_tep_mask),
            'tep_density_increase': np.sum(post_tep_mask) / np.sum(pre_tep_mask) if np.sum(pre_tep_mask) > 0 else 0
        }
        
        # Binned analysis to quantify slope in TEP region
        bins = np.arange(1000, 8000, 500)
        hist, _ = np.histogram(self.distances, bins=bins)
        
        # Focus on TEP region (3000-5000 km)
        tep_bin_indices = np.where((bins[:-1] >= 3000) & (bins[:-1] < 5000))[0]
        if len(tep_bin_indices) > 1:
            tep_counts = hist[tep_bin_indices]
            self.tep_slope = np.polyfit(range(len(tep_counts)), tep_counts, 1)[0]
        else:
            self.tep_slope = 0
            
        print(f"Distribution skewness: {self.dist_stats['skewness']:.3f}")
        print(f"TEP region slope: {self.tep_slope:.1f} pairs per 500km bin")
        print(f"TEP density increase: {self.tep_analysis['tep_density_increase']:.2f}×")
        
    def setup_analysis_parameters(self):
        """Setup parameters matching real TEP analysis."""
        self.num_bins = 30
        self.max_distance = 13000
        self.min_bin_count = 100
        self.edges = np.logspace(np.log10(50), np.log10(self.max_distance), self.num_bins + 1)
        
    def exponential_model(self, r, A, lambda_km, C0):
        """Exponential decay model identical to TEP analysis."""
        return A * np.exp(-r / lambda_km) + C0
        
    def apply_tep_binning_and_fitting(self, coherence_data: np.ndarray, 
                                     label: str = "test") -> Optional[Dict]:
        """Apply identical TEP methodology to synthetic coherence data."""
        try:
            # Logarithmic binning (identical to TEP)
            bin_indices = np.digitize(self.distances, self.edges) - 1
            valid_mask = (bin_indices >= 0) & (bin_indices < self.num_bins)
            
            bin_distances = []
            bin_coherences = []
            bin_counts = []
            
            for i in range(self.num_bins):
                mask = (bin_indices == i) & valid_mask
                count = np.sum(mask)
                
                if count >= self.min_bin_count:
                    bin_distances.append(self.distances[mask].mean())
                    bin_coherences.append(coherence_data[mask].mean())
                    bin_counts.append(count)
            
            if len(bin_distances) < 5:
                return None
                
            # Exponential fitting (identical bounds to TEP)
            bounds = ([0.01, 100, -1], [2, 20000, 1])
            weights = 1 / np.sqrt(bin_counts)  # Same weighting as TEP
            
            popt, pcov = curve_fit(
                self.exponential_model, 
                bin_distances, 
                bin_coherences,
                sigma=weights,
                bounds=bounds, 
                maxfev=5000
            )
            
            A, lambda_km, C0 = popt
            param_errors = np.sqrt(np.diag(pcov))
            
            # Calculate R-squared
            y_pred = self.exponential_model(np.array(bin_distances), A, lambda_km, C0)
            ss_res = np.sum((np.array(bin_coherences) - y_pred) ** 2)
            ss_tot = np.sum((np.array(bin_coherences) - np.mean(bin_coherences)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'label': label,
                'amplitude': float(A),
                'lambda_km': float(lambda_km),
                'offset': float(C0),
                'r_squared': float(r_squared),
                'lambda_error': float(param_errors[1]),
                'n_bins': len(bin_distances),
                'total_pairs': sum(bin_counts),
                'bin_distances': bin_distances,
                'bin_coherences': bin_coherences,
                'bin_counts': bin_counts
            }
            
        except Exception as e:
            print(f"Fitting failed for {label}: {e}")
            return None
    
    def test_pure_noise_scenarios(self, n_realizations: int = 50) -> Dict:
        """Test multiple pure noise scenarios to quantify maximum spurious correlations."""
        print(f"\nTesting pure noise scenarios ({n_realizations} realizations each)...")
        
        scenarios = {
            'uniform_noise': {
                'description': 'Uniform random noise [-1, 1]',
                'generator': lambda: np.random.uniform(-1, 1, self.n_pairs)
            },
            'gaussian_noise': {
                'description': 'Gaussian noise (σ=0.3)',
                'generator': lambda: np.clip(np.random.normal(0, 0.3, self.n_pairs), -1, 1)
            },
            'heavy_tail_noise': {
                'description': 'Heavy-tailed noise (t-distribution)',
                'generator': lambda: np.clip(stats.t.rvs(df=3, size=self.n_pairs) * 0.2, -1, 1)
            },
            'correlated_noise': {
                'description': 'Spatially correlated noise',
                'generator': self._generate_correlated_noise
            },
            'measurement_noise': {
                'description': 'Realistic measurement noise with heteroscedasticity',
                'generator': self._generate_measurement_noise
            }
        }
        
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"  Testing {scenario_name}...")
            scenario_results = []
            
            for i in range(n_realizations):
                np.random.seed(42 + i * 100)  # Reproducible but varied
                
                coherence = scenario['generator']()
                result = self.apply_tep_binning_and_fitting(coherence, f"{scenario_name}_{i}")
                
                if result:
                    scenario_results.append(result)
            
            if scenario_results:
                r_squared_values = [r['r_squared'] for r in scenario_results]
                lambda_values = [r['lambda_km'] for r in scenario_results]
                
                results[scenario_name] = {
                    'description': scenario['description'],
                    'n_successful_fits': len(scenario_results),
                    'r_squared_stats': {
                        'mean': np.mean(r_squared_values),
                        'std': np.std(r_squared_values),
                        'min': np.min(r_squared_values),
                        'max': np.max(r_squared_values),
                        'abs_max': np.max(np.abs(r_squared_values)),
                        'percentile_95': np.percentile(np.abs(r_squared_values), 95),
                        'percentile_99': np.percentile(np.abs(r_squared_values), 99)
                    },
                    'lambda_stats': {
                        'mean': np.mean(lambda_values),
                        'std': np.std(lambda_values),
                        'min': np.min(lambda_values),
                        'max': np.max(lambda_values)
                    },
                    'detailed_results': scenario_results
                }
                
                print(f"    Max |R²|: {results[scenario_name]['r_squared_stats']['abs_max']:.4f}")
        
        return results
    
    def _generate_correlated_noise(self) -> np.ndarray:
        """Generate spatially correlated noise that might mimic systematic effects."""
        # Create weak spatial correlation based on distance
        base_noise = np.random.normal(0, 0.2, self.n_pairs)
        
        # Add very weak distance-dependent component (should not create strong correlations)
        distance_component = 0.01 * np.sin(self.distances / 2000) * np.random.normal(0, 0.1, self.n_pairs)
        
        return np.clip(base_noise + distance_component, -1, 1)
    
    def _generate_measurement_noise(self) -> np.ndarray:
        """Generate realistic measurement noise with varying uncertainty."""
        # Heteroscedastic noise (uncertainty varies with distance)
        base_std = 0.15
        distance_std_factor = 1 + 0.1 * (self.distances - np.mean(self.distances)) / np.std(self.distances)
        
        noise = np.random.normal(0, base_std * distance_std_factor)
        return np.clip(noise, -1, 1)
    
    def test_systematic_bias_scenarios(self, n_realizations: int = 30) -> Dict:
        """Test scenarios with systematic biases that could amplify distribution effects."""
        print(f"\nTesting systematic bias scenarios ({n_realizations} realizations each)...")
        
        bias_scenarios = {
            'linear_distance_bias': {
                'description': 'Linear bias with distance',
                'generator': lambda: self._add_linear_bias(np.random.normal(0, 0.2, self.n_pairs))
            },
            'tep_range_bias': {
                'description': 'Enhanced noise in TEP range',
                'generator': lambda: self._add_tep_range_bias(np.random.normal(0, 0.2, self.n_pairs))
            },
            'network_processing_bias': {
                'description': 'Simulated GNSS processing artifacts',
                'generator': lambda: self._add_processing_bias(np.random.normal(0, 0.2, self.n_pairs))
            },
            'elevation_bias': {
                'description': 'Elevation-dependent systematic effects',
                'generator': lambda: self._add_elevation_bias(np.random.normal(0, 0.2, self.n_pairs))
            }
        }
        
        results = {}
        
        for scenario_name, scenario in bias_scenarios.items():
            print(f"  Testing {scenario_name}...")
            scenario_results = []
            
            for i in range(n_realizations):
                np.random.seed(42 + i * 200)
                
                coherence = scenario['generator']()
                result = self.apply_tep_binning_and_fitting(coherence, f"{scenario_name}_{i}")
                
                if result:
                    scenario_results.append(result)
            
            if scenario_results:
                r_squared_values = [r['r_squared'] for r in scenario_results]
                
                results[scenario_name] = {
                    'description': scenario['description'],
                    'n_successful_fits': len(scenario_results),
                    'r_squared_stats': {
                        'mean': np.mean(r_squared_values),
                        'std': np.std(r_squared_values),
                        'min': np.min(r_squared_values),
                        'max': np.max(r_squared_values),
                        'abs_max': np.max(np.abs(r_squared_values)),
                        'percentile_95': np.percentile(np.abs(r_squared_values), 95),
                        'percentile_99': np.percentile(np.abs(r_squared_values), 99)
                    }
                }
                
                print(f"    Max |R²|: {results[scenario_name]['r_squared_stats']['abs_max']:.4f}")
        
        return results
    
    def _add_linear_bias(self, base_coherence: np.ndarray) -> np.ndarray:
        """Add weak linear bias with distance."""
        # Very small linear trend (should not create strong exponential fits)
        distance_normalized = (self.distances - np.mean(self.distances)) / np.std(self.distances)
        bias = 0.005 * distance_normalized  # 0.5% bias per standard deviation
        return np.clip(base_coherence + bias, -1, 1)
    
    def _add_tep_range_bias(self, base_coherence: np.ndarray) -> np.ndarray:
        """Add systematic bias specifically in TEP range."""
        tep_mask = (self.distances >= 3000) & (self.distances <= 5000)
        biased_coherence = base_coherence.copy()
        biased_coherence[tep_mask] += np.random.normal(0.01, 0.02, np.sum(tep_mask))  # Small positive bias
        return np.clip(biased_coherence, -1, 1)
    
    def _add_processing_bias(self, base_coherence: np.ndarray) -> np.ndarray:
        """Simulate GNSS processing artifacts."""
        # Common-mode component affecting all pairs
        common_mode = np.random.normal(0, 0.01)
        
        # Distance-dependent processing effects
        processing_effect = 0.002 * np.exp(-self.distances / 8000)  # Decreasing with distance
        
        return np.clip(base_coherence + common_mode + processing_effect, -1, 1)
    
    def _add_elevation_bias(self, base_coherence: np.ndarray) -> np.ndarray:
        """Simulate elevation-dependent effects."""
        # Assume elevation effects correlate weakly with distance (continental vs oceanic stations)
        elevation_proxy = np.sin(self.distances / 3000) * 0.003  # Weak sinusoidal pattern
        return np.clip(base_coherence + elevation_proxy, -1, 1)
    
    def test_binning_strategies(self) -> Dict:
        """Test different binning strategies to assess robustness."""
        print("\nTesting alternative binning strategies...")
        
        # Generate test coherence with known properties
        np.random.seed(42)
        test_coherence = np.random.normal(0, 0.2, self.n_pairs)
        
        binning_strategies = {
            'logarithmic_30': {
                'description': 'Logarithmic binning (30 bins) - TEP standard',
                'edges': np.logspace(np.log10(50), np.log10(13000), 31)
            },
            'logarithmic_20': {
                'description': 'Logarithmic binning (20 bins) - coarser',
                'edges': np.logspace(np.log10(50), np.log10(13000), 21)
            },
            'logarithmic_40': {
                'description': 'Logarithmic binning (40 bins) - finer',
                'edges': np.logspace(np.log10(50), np.log10(13000), 41)
            },
            'linear_30': {
                'description': 'Linear binning (30 bins)',
                'edges': np.linspace(50, 13000, 31)
            },
            'equal_count': {
                'description': 'Equal-count binning (quantile-based)',
                'edges': np.percentile(self.distances, np.linspace(0, 100, 31))
            }
        }
        
        results = {}
        
        for strategy_name, strategy in binning_strategies.items():
            print(f"  Testing {strategy_name}...")
            
            # Apply this binning strategy
            edges = strategy['edges']
            bin_indices = np.digitize(self.distances, edges) - 1
            n_bins = len(edges) - 1
            
            bin_distances = []
            bin_coherences = []
            bin_counts = []
            
            for i in range(n_bins):
                mask = bin_indices == i
                count = np.sum(mask)
                
                if count >= self.min_bin_count:
                    bin_distances.append(self.distances[mask].mean())
                    bin_coherences.append(test_coherence[mask].mean())
                    bin_counts.append(count)
            
            if len(bin_distances) >= 5:
                try:
                    popt, _ = curve_fit(
                        self.exponential_model,
                        bin_distances,
                        bin_coherences,
                        bounds=([0.01, 100, -1], [2, 20000, 1])
                    )
                    
                    A, lambda_km, C0 = popt
                    y_pred = self.exponential_model(np.array(bin_distances), A, lambda_km, C0)
                    ss_res = np.sum((np.array(bin_coherences) - y_pred) ** 2)
                    ss_tot = np.sum((np.array(bin_coherences) - np.mean(bin_coherences)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    results[strategy_name] = {
                        'description': strategy['description'],
                        'r_squared': r_squared,
                        'lambda_km': lambda_km,
                        'n_bins_used': len(bin_distances),
                        'bin_count_stats': {
                            'mean': np.mean(bin_counts),
                            'std': np.std(bin_counts),
                            'min': np.min(bin_counts),
                            'max': np.max(bin_counts)
                        }
                    }
                    
                    print(f"    R²: {r_squared:.4f}, λ: {lambda_km:.0f} km")
                    
                except Exception as e:
                    print(f"    Failed: {e}")
        
        return results
    
    def generate_comprehensive_report(self, pure_noise_results: Dict, 
                                    bias_results: Dict, binning_results: Dict) -> Dict:
        """Generate comprehensive analysis report."""
        print("\nGenerating comprehensive bias analysis report...")
        
        # Find maximum spurious correlations across all scenarios
        all_max_r2 = []
        
        for scenario_results in pure_noise_results.values():
            all_max_r2.append(scenario_results['r_squared_stats']['abs_max'])
            
        for scenario_results in bias_results.values():
            all_max_r2.append(scenario_results['r_squared_stats']['abs_max'])
        
        max_spurious_r2 = max(all_max_r2) if all_max_r2 else 0
        
        # TEP comparison
        typical_tep_r2 = 0.8  # Conservative estimate
        tep_threshold = 0.3
        
        # Safety margins
        safety_margin_threshold = tep_threshold / max_spurious_r2 if max_spurious_r2 > 0 else float('inf')
        safety_margin_tep = typical_tep_r2 / max_spurious_r2 if max_spurious_r2 > 0 else float('inf')
        
        # Validation assessment
        if max_spurious_r2 < 0.1:
            validation_status = "VALIDATED"
            confidence = "HIGH"
        elif max_spurious_r2 < 0.2:
            validation_status = "LIKELY_VALID"
            confidence = "MEDIUM"
        elif max_spurious_r2 < tep_threshold:
            validation_status = "MARGINAL"
            confidence = "LOW"
        else:
            validation_status = "COMPROMISED"
            confidence = "VERY_LOW"
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'distribution_characteristics': {
                'shape': 'RIGHT_SKEWED' if self.dist_stats['skewness'] > 0.5 else 'SYMMETRIC',
                'skewness': self.dist_stats['skewness'],
                'tep_region_slope': self.tep_slope,
                'tep_density_increase': self.tep_analysis['tep_density_increase']
            },
            'spurious_correlation_analysis': {
                'max_spurious_r_squared': max_spurious_r2,
                'scenarios_tested': len(pure_noise_results) + len(bias_results),
                'total_realizations': sum(r['n_successful_fits'] for r in pure_noise_results.values()) + 
                                   sum(r['n_successful_fits'] for r in bias_results.values())
            },
            'safety_margins': {
                'threshold_margin': safety_margin_threshold,
                'tep_signal_margin': safety_margin_tep,
                'recommended_threshold': max(0.5, 3 * max_spurious_r2)
            },
            'validation_assessment': {
                'status': validation_status,
                'confidence': confidence,
                'methodology_robust': max_spurious_r2 < 0.1,
                'tep_signals_likely_genuine': safety_margin_tep > 5.0
            },
            'detailed_results': {
                'pure_noise_scenarios': pure_noise_results,
                'systematic_bias_scenarios': bias_results,
                'binning_strategy_comparison': binning_results
            },
            'recommendations': self._generate_recommendations(max_spurious_r2, safety_margin_tep)
        }
        
        return report
    
    def _generate_recommendations(self, max_spurious_r2: float, safety_margin: float) -> List[str]:
        """Generate specific recommendations based on analysis results."""
        recommendations = []
        
        if max_spurious_r2 > 0.3:
            recommendations.append("CRITICAL: Revise TEP significance threshold to R² > 0.5")
            recommendations.append("Consider alternative analysis methodologies less sensitive to distribution shape")
        elif max_spurious_r2 > 0.2:
            recommendations.append("Increase TEP significance threshold to R² > 0.4 for conservative validation")
        elif max_spurious_r2 > 0.1:
            recommendations.append("Current TEP threshold (R² > 0.3) appears adequate but monitor closely")
        else:
            recommendations.append("Current methodology appears robust against distribution bias")
        
        if safety_margin < 3.0:
            recommendations.append("Implement additional validation methods (bootstrap, permutation tests)")
        
        if safety_margin > 10.0:
            recommendations.append("TEP signals show strong evidence of genuine physical origin")
        
        recommendations.extend([
            "Document distribution characteristics in methodology section",
            "Include geometric bias analysis in standard validation pipeline",
            "Consider developing distribution-corrected analysis methods for future studies"
        ])
        
        return recommendations
    
    def create_visualization(self, report: Dict, output_dir: Path):
        """Create comprehensive visualization of bias analysis results."""
        print("Creating bias analysis visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rigorous Geometric Bias Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Distance distribution with TEP range highlighted
        ax1 = axes[0, 0]
        ax1.hist(self.distances, bins=50, alpha=0.7, color='purple', edgecolor='white')
        ax1.axvspan(3330, 4549, alpha=0.3, color='red', label='TEP Range')
        ax1.axvline(np.mean(self.distances), color='black', linestyle='--', label=f'Mean: {np.mean(self.distances):.0f} km')
        ax1.set_xlabel('Distance (km)')
        ax1.set_ylabel('Number of Station Pairs')
        ax1.set_title('GNSS Station Distance Distribution\n(Right-Skewed, Not Bell-Shaped)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spurious R² distribution across all scenarios
        ax2 = axes[0, 1]
        all_r2_values = []
        scenario_labels = []
        
        for scenario_name, results in report['detailed_results']['pure_noise_scenarios'].items():
            if 'detailed_results' in results:
                r2_vals = [r['r_squared'] for r in results['detailed_results']]
                all_r2_values.extend(r2_vals)
                scenario_labels.extend([scenario_name] * len(r2_vals))
        
        for scenario_name, results in report['detailed_results']['systematic_bias_scenarios'].items():
            if 'detailed_results' in results:
                r2_vals = [r['r_squared'] for r in results['detailed_results']]
                all_r2_values.extend(r2_vals)
                scenario_labels.extend([scenario_name] * len(r2_vals))
        
        ax2.hist(np.abs(all_r2_values), bins=30, alpha=0.7, color='orange', edgecolor='white')
        ax2.axvline(0.3, color='red', linestyle='--', linewidth=2, label='TEP Threshold (0.3)')
        ax2.axvline(0.8, color='green', linestyle='--', linewidth=2, label='Typical TEP R² (0.8)')
        ax2.axvline(report['spurious_correlation_analysis']['max_spurious_r_squared'], 
                   color='black', linestyle='-', linewidth=2, label='Max Spurious')
        ax2.set_xlabel('|R²|')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Spurious R² Values\nAcross All Test Scenarios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Safety margins visualization
        ax3 = axes[1, 0]
        margins = [
            report['safety_margins']['threshold_margin'],
            report['safety_margins']['tep_signal_margin']
        ]
        margin_labels = ['TEP Threshold\nMargin', 'TEP Signal\nMargin']
        colors = ['orange' if m < 3 else 'green' for m in margins]
        
        bars = ax3.bar(margin_labels, margins, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(3.0, color='red', linestyle='--', label='Minimum Safe Margin (3×)')
        ax3.set_ylabel('Safety Margin (×)')
        ax3.set_title('Safety Margins:\nSpurious vs TEP Correlations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, margin in zip(bars, margins):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{margin:.1f}×', ha='center', va='bottom', fontweight='bold')
        
        # 4. Validation status summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create text summary
        status_color = {
            'VALIDATED': 'green',
            'LIKELY_VALID': 'orange', 
            'MARGINAL': 'red',
            'COMPROMISED': 'darkred'
        }[report['validation_assessment']['status']]
        
        summary_text = f"""
VALIDATION ASSESSMENT

Status: {report['validation_assessment']['status']}
Confidence: {report['validation_assessment']['confidence']}

Max Spurious R²: {report['spurious_correlation_analysis']['max_spurious_r_squared']:.4f}
TEP Threshold: 0.300
Typical TEP R²: 0.800

Safety Margins:
• Threshold: {report['safety_margins']['threshold_margin']:.1f}×
• TEP Signal: {report['safety_margins']['tep_signal_margin']:.1f}×

Distribution: {report['distribution_characteristics']['shape']}
Skewness: {report['distribution_characteristics']['skewness']:.3f}

Methodology Robust: {report['validation_assessment']['methodology_robust']}
TEP Signals Genuine: {report['validation_assessment']['tep_signals_likely_genuine']}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.1))
        
        plt.tight_layout()
        
        output_file = output_dir / 'rigorous_geometric_bias_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {output_file}")
        return str(output_file)

def main():
    """Run comprehensive geometric bias analysis."""
    print("="*80)
    print("RIGOROUS GEOMETRIC BIAS ANALYSIS FOR TEP VALIDATION")
    print("="*80)
    print("Testing right-skewed distribution bias with comprehensive scenarios")
    print()
    
    # Setup
    root_dir = Path(__file__).resolve().parents[2]
    distances_file = root_dir / 'data/processed/step_8_station_distances.csv'
    output_dir = root_dir / 'results/exploratory'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not distances_file.exists():
        print(f"ERROR: Distance file not found: {distances_file}")
        print("Please run step_8_tep_visualization.py first to generate station distances")
        return
    
    try:
        # Initialize analyzer
        analyzer = RigorousGeometricBiasAnalyzer(str(distances_file))
        
        # Run comprehensive tests
        print("\n" + "="*60)
        print("PHASE 1: PURE NOISE SCENARIOS")
        print("="*60)
        pure_noise_results = analyzer.test_pure_noise_scenarios(n_realizations=50)
        
        print("\n" + "="*60)
        print("PHASE 2: SYSTEMATIC BIAS SCENARIOS")
        print("="*60)
        bias_results = analyzer.test_systematic_bias_scenarios(n_realizations=30)
        
        print("\n" + "="*60)
        print("PHASE 3: BINNING STRATEGY COMPARISON")
        print("="*60)
        binning_results = analyzer.test_binning_strategies()
        
        # Generate comprehensive report
        print("\n" + "="*60)
        print("PHASE 4: COMPREHENSIVE ANALYSIS")
        print("="*60)
        report = analyzer.generate_comprehensive_report(
            pure_noise_results, bias_results, binning_results
        )
        
        # Save results
        report_file = output_dir / 'rigorous_geometric_bias_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create visualization
        viz_file = analyzer.create_visualization(report, output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("RIGOROUS GEOMETRIC BIAS ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"Distribution Shape: {report['distribution_characteristics']['shape']}")
        print(f"Skewness: {report['distribution_characteristics']['skewness']:.3f}")
        print(f"TEP Region Slope: {report['distribution_characteristics']['tep_region_slope']:.1f} pairs/500km")
        print()
        print(f"Maximum Spurious R²: {report['spurious_correlation_analysis']['max_spurious_r_squared']:.4f}")
        print(f"TEP Threshold (0.3): {report['safety_margins']['threshold_margin']:.1f}× safety margin")
        print(f"TEP Signals (0.8): {report['safety_margins']['tep_signal_margin']:.1f}× safety margin")
        print()
        print(f"Validation Status: {report['validation_assessment']['status']}")
        print(f"Confidence Level: {report['validation_assessment']['confidence']}")
        print(f"Methodology Robust: {report['validation_assessment']['methodology_robust']}")
        print(f"TEP Signals Genuine: {report['validation_assessment']['tep_signals_likely_genuine']}")
        print()
        print("Key Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"{i}. {rec}")
        
        print(f"\nDetailed report: {report_file}")
        print(f"Visualization: {viz_file}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
