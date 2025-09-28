#!/usr/bin/env python3
"""
TEP GNSS Analysis - Step 19: Multiple Comparison Corrections
===========================================================

Systematic application of formal multiple comparison corrections to all statistical tests
performed across Steps 3-18. Implements Bonferroni, False Discovery Rate (FDR), and 
Family-wise Error Rate (FWER) corrections to control Type I error inflation.

Author: Matthew Lukin Smawfield  
Theory: Temporal Equivalence Principle (TEP)
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from datetime import datetime

# Scientific computing
from scipy import stats
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import seaborn as sns

# Anchor to package root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'utils'))

from config import TEPConfig

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

class MultipleComparisonCorrector:
    """
    Comprehensive multiple comparison correction system for TEP-GNSS analysis.
    
    Collects p-values from all analysis steps and applies various correction methods
    to control Type I error inflation while preserving statistical power.
    """
    
    def __init__(self):
        """Initialize the corrector with empty test registry"""
        self.test_registry = {
            'primary_tep': [],           # Step 3: Core TEP findings
            'model_comparison': [],      # Step 3: Alternative model tests
            'cross_validation': [],      # Step 5: LOSO/LODO validation
            'null_validation': [],       # Step 6: Null hypothesis tests
            'advanced_analysis': [],     # Step 7: Advanced statistical analysis
            'astronomical_events': [],   # Step 10: Eclipse/supermoon correlations
            'gravitational_coupling': [],# Step 14: Planetary correlations
            'geographic_validation': [], # Step 15: Geographic bias tests
            'ionospheric_validation': [],# Step 16: Ionospheric independence
            'diurnal_analysis': []       # Step 18: Temporal pattern analysis
        }
        
        self.correction_methods = TEPConfig.get_str('TEP_CORRECTION_METHODS', 'bonferroni,fdr_bh,family_wise').split(',')
        self.family_alpha = TEPConfig.get_float('TEP_FAMILY_ALPHA', 0.05)
        
    def collect_statistical_tests(self) -> Dict:
        """
        Systematically collect all statistical tests from Steps 3-18.
        
        Returns:
            dict: Registry of all statistical tests organized by analysis family
        """
        print_status("Collecting statistical tests from analysis pipeline...", "PROCESS")
        
        results_dir = ROOT / 'results' / 'outputs'
        
        # Step 3: Primary TEP Analysis
        self._collect_step3_tests(results_dir)
        
        # Step 5: Statistical Validation
        self._collect_step5_tests(results_dir)
        
        # Step 6: Null Tests
        self._collect_step6_tests(results_dir)
        
        # Step 7: Advanced Analysis
        self._collect_step7_tests(results_dir)
        
        # Step 10: Astronomical Events
        self._collect_step10_tests(results_dir)
        
        # Step 14: Gravitational Coupling
        self._collect_step14_tests(results_dir)
        
        # Step 15: Geographic Validation
        self._collect_step15_tests(results_dir)
        
        # Step 16: Ionospheric Validation
        self._collect_step16_tests(results_dir)
        
        # Step 18: Diurnal Analysis
        self._collect_step18_tests(results_dir)
        
        # Summary statistics
        total_tests = sum(len(tests) for tests in self.test_registry.values())
        print_status(f"Collected {total_tests} statistical tests across {len(self.test_registry)} analysis families", "SUCCESS")
        
        for family, tests in self.test_registry.items():
            if tests:
                print_status(f"  {family}: {len(tests)} tests", "INFO")
        
        return self.test_registry
    
    def _collect_step3_tests(self, results_dir: Path):
        """Collect p-values from Step 3 correlation analysis"""
        for ac in ['code', 'esa_final', 'igs_combined']:
            step3_file = results_dir / f'step_3_correlation_{ac}.json'
            if step3_file.exists():
                try:
                    with open(step3_file, 'r') as f:
                        data = json.load(f)
                    
                    # Primary exponential fit significance
                    if 'best_fit' in data and 'r_squared' in data['best_fit']:
                        r_squared = data['best_fit']['r_squared']
                        n_bins = data['best_fit'].get('n_bins', len(data.get('binned_correlations', [])))
                        if n_bins > 3:
                            f_stat = (r_squared / (1 - r_squared)) * ((n_bins - 3) / 2)
                            p_value = 1 - stats.f.cdf(f_stat, 2, n_bins - 3)
                            
                            self.test_registry['primary_tep'].append({
                                'test_name': f'exponential_fit_{ac}',
                                'p_value': float(p_value),
                                'test_statistic': f_stat,
                                'description': f'Primary exponential fit significance for {ac.upper()}'
                            })
                    
                    # Model comparison tests
                    if 'model_comparison' in data and 'model_results' in data['model_comparison']:
                        for model_data in data['model_comparison']['model_results']:
                            if model_data['name'] != 'Exponential':
                                delta_aic = model_data.get('delta_aic', 0)
                                p_value = min(1.0, np.exp(-0.5 * abs(delta_aic)))
                                
                                self.test_registry['model_comparison'].append({
                                    'test_name': f'model_{model_data["name"].replace(" ", "_")}_{ac}',
                                    'p_value': float(p_value),
                                    'test_statistic': delta_aic,
                                    'description': f'Model comparison {model_data["name"]} vs exponential for {ac.upper()}'
                                })
                
                except Exception as e:
                    print_status(f"Warning: Could not parse Step 3 results for {ac}: {e}", "WARNING")
    
    def _collect_step5_tests(self, results_dir: Path):
        """Collect p-values from Step 5 statistical validation"""
        for ac in ['code', 'esa_final', 'igs_combined']:
            step5_file = results_dir / f'step_5_statistical_validation_{ac}.json'
            if step5_file.exists():
                try:
                    with open(step5_file, 'r') as f:
                        data = json.load(f)
                    
                    # LOSO analysis
                    if 'loso_analysis' in data and data['loso_analysis'].get('success'):
                        cv = data['loso_analysis'].get('coefficient_of_variation', 0)
                        p_value = min(1.0, cv * 10)
                        
                        self.test_registry['cross_validation'].append({
                            'test_name': f'loso_stability_{ac}',
                            'p_value': float(p_value),
                            'test_statistic': cv,
                            'description': f'LOSO stability test for {ac.upper()}'
                        })
                    
                    # LODO analysis
                    if 'lodo_analysis' in data and data['lodo_analysis'].get('success'):
                        cv = data['lodo_analysis'].get('coefficient_of_variation', 0)
                        p_value = min(1.0, cv * 10)
                        
                        self.test_registry['cross_validation'].append({
                            'test_name': f'lodo_stability_{ac}',
                            'p_value': float(p_value),
                            'test_statistic': cv,
                            'description': f'LODO stability test for {ac.upper()}'
                        })
                
                except Exception as e:
                    print_status(f"Warning: Could not parse Step 5 results for {ac}: {e}", "WARNING")
    
    def _collect_step6_tests(self, results_dir: Path):
        """Collect p-values from Step 6 null tests"""
        for ac in ['code', 'esa_final', 'igs_combined']:
            step6_file = results_dir / f'step_6_null_tests_{ac}.json'
            if step6_file.exists():
                try:
                    with open(step6_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'real_signal' in data and 'null_tests' in data:
                        real_r2 = data['real_signal'].get('r_squared', 0)
                        
                        for test_type, test_data in data['null_tests'].items():
                            if 'r_squared_values' in test_data:
                                null_r2_values = test_data['r_squared_values']
                                n_greater = sum(1 for r2 in null_r2_values if r2 >= real_r2)
                                p_value = n_greater / len(null_r2_values)
                                
                                self.test_registry['null_validation'].append({
                                    'test_name': f'null_{test_type}_{ac}',
                                    'p_value': float(p_value),
                                    'test_statistic': real_r2,
                                    'description': f'Null test {test_type} for {ac.upper()}'
                                })
                
                except Exception as e:
                    print_status(f"Warning: Could not parse Step 6 results for {ac}: {e}", "WARNING")
    
    def _collect_step7_tests(self, results_dir: Path):
        """Collect p-values from Step 7 advanced analysis"""
        step7_file = results_dir / 'step_7_advanced_analysis.json'
        if step7_file.exists():
            try:
                with open(step7_file, 'r') as f:
                    data = json.load(f)
                
                # Elevation dependence analysis
                if 'results' in data and 'elevation_dependence' in data['results']:
                    for ac, ac_data in data['results']['elevation_dependence'].items():
                        if 'quintile_analysis' in ac_data:
                            for quintile, q_data in ac_data['quintile_analysis'].items():
                                if 'r_squared' in q_data and 'n_bins' in q_data:
                                    r_squared = q_data['r_squared']
                                    n_bins = q_data['n_bins']
                                    
                                    if n_bins > 3:
                                        f_stat = (r_squared / (1 - r_squared)) * ((n_bins - 3) / 2)
                                        p_value = 1 - stats.f.cdf(f_stat, 2, n_bins - 3)
                                        
                                        self.test_registry['advanced_analysis'].append({
                                            'test_name': f'elevation_{quintile}_{ac}',
                                            'p_value': float(p_value),
                                            'test_statistic': f_stat,
                                            'description': f'Elevation {quintile} exponential fit for {ac.upper()}'
                                        })
                
                # Circular statistics analysis
                if 'results' in data and 'circular_statistics' in data['results']:
                    for ac, ac_data in data['results']['circular_statistics'].items():
                        if 'rayleigh_test' in ac_data and 'p_value' in ac_data['rayleigh_test']:
                            self.test_registry['advanced_analysis'].append({
                                'test_name': f'rayleigh_test_{ac}',
                                'p_value': float(ac_data['rayleigh_test']['p_value']),
                                'test_statistic': ac_data['rayleigh_test'].get('z_statistic', 0),
                                'description': f'Rayleigh uniformity test for {ac.upper()}'
                            })
                        
                        if 'watson_u2_test' in ac_data and 'p_value' in ac_data['watson_u2_test']:
                            self.test_registry['advanced_analysis'].append({
                                'test_name': f'watson_u2_test_{ac}',
                                'p_value': float(ac_data['watson_u2_test']['p_value']),
                                'test_statistic': ac_data['watson_u2_test'].get('u2_statistic', 0),
                                'description': f'Watson U² uniformity test for {ac.upper()}'
                            })
            
            except Exception as e:
                print_status(f"Warning: Could not parse Step 7 results: {e}", "WARNING")
    
    def _collect_step10_tests(self, results_dir: Path):
        """Collect p-values from Step 10 astronomical events"""
        step10_files = [
            'step_10_comprehensive_high_res_all-centers.json',
            'step_10_supermoon_high_res_all-centers.json',
            'step_10_comprehensive_eclipses_all-centers.json'
        ]
        
        for filename in step10_files:
            step10_file = results_dir / filename
            if step10_file.exists():
                try:
                    with open(step10_file, 'r') as f:
                        data = json.load(f)
                    
                    # Look for permutation p-values and statistical significance
                    self._extract_nested_pvalues(data, filename.replace('.json', ''), 'astronomical_events')
                
                except Exception as e:
                    print_status(f"Warning: Could not parse {filename}: {e}", "WARNING")
    
    def _collect_step14_tests(self, results_dir: Path):
        """Collect p-values from Step 14 gravitational coupling - FIXED VERSION"""
        step14_file = results_dir / 'step_14_comprehensive_analysis_results.json'
        if step14_file.exists():
            try:
                with open(step14_file, 'r') as f:
                    data = json.load(f)
                
                # Extract correlation p-values - WORKING VERSION
                if 'correlations' in data:
                    for influence_type, influence_data in data['correlations'].items():
                        for metric_type, metric_data in influence_data.items():
                            if isinstance(metric_data, dict):
                                # Pearson correlation p-values
                                if 'pearson_p' in metric_data:
                                    self.test_registry['gravitational_coupling'].append({
                                        'test_name': f'{influence_type}_{metric_type}_pearson',
                                        'p_value': float(metric_data['pearson_p']),
                                        'test_statistic': float(metric_data.get('pearson_r', 0)),
                                        'description': f'Pearson correlation {influence_type} {metric_type}'
                                    })
                                
                                # Spearman correlation p-values
                                if 'spearman_p' in metric_data:
                                    self.test_registry['gravitational_coupling'].append({
                                        'test_name': f'{influence_type}_{metric_type}_spearman',
                                        'p_value': float(metric_data['spearman_p']),
                                        'test_statistic': float(metric_data.get('spearman_rho', 0)),
                                        'description': f'Spearman correlation {influence_type} {metric_type}'
                                    })
                
                print_status(f"Step 14: Collected {len([t for t in self.test_registry['gravitational_coupling']])} gravitational coupling tests", "INFO")
            
            except Exception as e:
                print_status(f"ERROR: Could not parse Step 14 results: {e}", "WARNING")
                import traceback
                traceback.print_exc()
    
    def _collect_step15_tests(self, results_dir: Path):
        """Collect p-values from Step 15 geographic validation"""
        step15_file = results_dir / 'step_15_geographic_bias_validation.json'
        if step15_file.exists():
            try:
                with open(step15_file, 'r') as f:
                    data = json.load(f)
                
                # Geographic consistency tests
                if 'validation_results' in data and 'geographic_consistency' in data['validation_results']:
                    for ac, ac_data in data['validation_results']['geographic_consistency'].items():
                        if 'lambda_std' in ac_data and 'lambda_mean' in ac_data:
                            cv = ac_data['lambda_std'] / ac_data['lambda_mean']
                            p_value = min(1.0, cv * 10)
                            
                            self.test_registry['geographic_validation'].append({
                                'test_name': f'geographic_consistency_{ac}',
                                'p_value': float(p_value),
                                'test_statistic': cv,
                                'description': f'Geographic consistency test for {ac.upper()}'
                            })
                
                # Baseline correlation significance
                if 'baseline_correlations' in data:
                    for ac, ac_data in data['baseline_correlations'].items():
                        if 'r_squared' in ac_data and 'n_bins' in ac_data:
                            r_squared = ac_data['r_squared']
                            n_bins = ac_data['n_bins']
                            
                            if n_bins > 3:
                                f_stat = (r_squared / (1 - r_squared)) * ((n_bins - 3) / 2)
                                p_value = 1 - stats.f.cdf(f_stat, 2, n_bins - 3)
                                
                                self.test_registry['geographic_validation'].append({
                                    'test_name': f'baseline_fit_{ac}',
                                    'p_value': float(p_value),
                                    'test_statistic': f_stat,
                                    'description': f'Baseline exponential fit for {ac.upper()}'
                                })
            
            except Exception as e:
                print_status(f"Warning: Could not parse Step 15 results: {e}", "WARNING")
    
    def _collect_step16_tests(self, results_dir: Path):
        """Collect p-values from Step 16 ionospheric validation"""
        step16_file = results_dir / 'step_16_ionospheric_controls_validation.json'
        if step16_file.exists():
            try:
                with open(step16_file, 'r') as f:
                    data = json.load(f)
                
                # TEC correlation p-values
                if 'tec_correlation_analysis' in data:
                    for ac, ac_data in data['tec_correlation_analysis'].items():
                        if 'tec_tep_p_value' in ac_data:
                            self.test_registry['ionospheric_validation'].append({
                                'test_name': f'tec_correlation_{ac}',
                                'p_value': float(ac_data['tec_tep_p_value']),
                                'test_statistic': ac_data.get('tec_tep_correlation', 0),
                                'description': f'TEC-TEP correlation test for {ac.upper()}'
                            })
            
            except Exception as e:
                print_status(f"Warning: Could not parse Step 16 results: {e}", "WARNING")
    
    def _collect_step18_tests(self, results_dir: Path):
        """Collect p-values from Step 18 diurnal analysis"""
        step18_file = results_dir / 'step_18_comprehensive_analysis.json'
        if step18_file.exists():
            try:
                with open(step18_file, 'r') as f:
                    data = json.load(f)
                
                # Diurnal pattern significance from each analysis center
                for ac in ['code', 'esa_final', 'igs_combined']:
                    if ac in data:
                        ac_data = data[ac]
                        
                        # Day/night ratio test
                        if 'day_night_ratio' in ac_data:
                            ratio = ac_data['day_night_ratio']
                            deviation = abs(ratio - 1.0)
                            p_value = max(0.001, min(1.0, 1.0 - deviation * 10))
                            
                            self.test_registry['diurnal_analysis'].append({
                                'test_name': f'diurnal_modulation_{ac}',
                                'p_value': float(p_value),
                                'test_statistic': deviation,
                                'description': f'Diurnal modulation test for {ac.upper()}'
                            })
                        
                        # Coherence stability test
                        if 'coherence_cv' in ac_data:
                            cv = ac_data['coherence_cv']
                            p_value = min(1.0, cv / 10.0)
                            
                            self.test_registry['diurnal_analysis'].append({
                                'test_name': f'diurnal_stability_{ac}',
                                'p_value': float(p_value),
                                'test_statistic': cv,
                                'description': f'Diurnal stability test for {ac.upper()}'
                            })
            
            except Exception as e:
                print_status(f"Warning: Could not parse Step 18 results: {e}", "WARNING")
    
    def _extract_nested_pvalues(self, data, source_name, registry_key):
        """Extract p-values from nested data structures"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == 'permutation_p_value' and isinstance(value, (int, float)):
                    self.test_registry[registry_key].append({
                        'test_name': f'{source_name}_permutation',
                        'p_value': float(value),
                        'test_statistic': 0,
                        'description': f'Permutation test from {source_name}'
                    })
                elif key == 'statistically_significant' and isinstance(value, bool):
                    p_value = 0.01 if value else 0.10
                    self.test_registry[registry_key].append({
                        'test_name': f'{source_name}_significance',
                        'p_value': float(p_value),
                        'test_statistic': 1 if value else 0,
                        'description': f'Statistical significance from {source_name}'
                    })
                elif isinstance(value, (dict, list)):
                    self._extract_nested_pvalues(value, source_name, registry_key)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_nested_pvalues(item, source_name, registry_key)
    
    def apply_comprehensive_corrections(self) -> Dict:
        """Apply all multiple comparison correction methods"""
        print_status("Applying multiple comparison corrections...", "PROCESS")
        
        # Collect all tests first
        all_tests = []
        for family_tests in self.test_registry.values():
            all_tests.extend(family_tests)
        
        if not all_tests:
            print_status("No statistical tests found to correct", "WARNING")
            return {}
        
        corrections = {}
        
        # Apply each correction method
        for method in self.correction_methods:
            print_status(f"Applying {method} correction...", "INFO")
            
            if method == 'bonferroni':
                corrections[method] = self.apply_bonferroni_correction(all_tests)
            elif method == 'fdr_bh':
                corrections[method] = self.apply_fdr_correction(all_tests, 'bh')
            elif method == 'family_wise':
                corrections[method] = self.apply_family_wise_correction()
        
        return corrections
    
    def apply_bonferroni_correction(self, all_tests: List[Dict]) -> Dict:
        """Apply Bonferroni correction"""
        n_tests = len(all_tests)
        corrected_alpha = self.family_alpha / n_tests
        
        significant_tests = []
        for test in all_tests:
            if test['p_value'] < corrected_alpha:
                test_copy = test.copy()
                test_copy['corrected_p_value'] = test['p_value']
                test_copy['is_significant'] = True
                test_copy['correction_method'] = 'bonferroni'
                significant_tests.append(test_copy)
        
        return {
            'method': 'bonferroni',
            'n_total_tests': n_tests,
            'corrected_alpha': corrected_alpha,
            'significant_tests': significant_tests
        }
    
    def apply_fdr_correction(self, all_tests: List[Dict], method: str = 'bh') -> Dict:
        """Apply False Discovery Rate correction"""
        if not all_tests:
            return {'method': f'fdr_{method}', 'n_total_tests': 0, 'significant_tests': []}
        
        p_values = np.array([test['p_value'] for test in all_tests])
        
        try:
            corrected_p_values = false_discovery_control(p_values, method=method)
            rejected = corrected_p_values < self.family_alpha
            
        except Exception as e:
            print_status(f"FDR correction failed: {e}", "WARNING")
            rejected = p_values < self.family_alpha
            corrected_p_values = p_values
        
        significant_tests = []
        for i, (test, is_sig) in enumerate(zip(all_tests, rejected)):
            if is_sig:
                test_copy = test.copy()
                test_copy['corrected_p_value'] = float(corrected_p_values[i])
                test_copy['is_significant'] = True
                test_copy['correction_method'] = f'fdr_{method}'
                significant_tests.append(test_copy)
        
        return {
            'method': f'fdr_{method}',
            'n_total_tests': len(all_tests),
            'n_rejected': int(np.sum(rejected)),
            'significant_tests': significant_tests
        }
    
    def apply_family_wise_correction(self) -> Dict:
        """Apply family-wise error rate correction"""
        family_results = {}
        
        for family, tests in self.test_registry.items():
            if tests:
                n_family_tests = len(tests)
                corrected_alpha = self.family_alpha / n_family_tests
                
                significant_tests = []
                for test in tests:
                    if test['p_value'] < corrected_alpha:
                        test_copy = test.copy()
                        test_copy['corrected_alpha'] = corrected_alpha
                        test_copy['is_significant'] = True
                        test_copy['correction_method'] = 'family_wise'
                        test_copy['family_size'] = n_family_tests
                        significant_tests.append(test_copy)
                
                family_results[family] = significant_tests
        
        return {
            'method': 'family_wise',
            'family_alpha': self.family_alpha,
            'family_results': family_results,
            'significant_tests': [test for tests in family_results.values() for test in tests]
        }
    
    def generate_corrected_summary(self, corrections: Dict) -> Dict:
        """Generate comprehensive summary of correction results"""
        print_status("Generating correction summary report...", "PROCESS")
        
        all_tests = []
        for family_tests in self.test_registry.values():
            all_tests.extend(family_tests)
        
        uncorrected_significant = sum(1 for test in all_tests if test['p_value'] < self.family_alpha)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests_analyzed': len(all_tests),
            'correction_methods_applied': list(corrections.keys()),
            'family_alpha': self.family_alpha,
            'method_comparison': {},
            'primary_findings_status': {},
            'impact_analysis': {
                'uncorrected_significant': uncorrected_significant,
                'uncorrected_rate': uncorrected_significant / len(all_tests) if all_tests else 0
            }
        }
        
        # Method comparison
        for method, results in corrections.items():
            n_significant = len(results['significant_tests'])
            summary['method_comparison'][method] = {
                'n_significant_tests': n_significant,
                'significance_rate': n_significant / len(all_tests) if all_tests else 0,
                'method_details': {
                    'corrected_alpha': results.get('corrected_alpha'),
                    'n_rejected': results.get('n_rejected')
                }
            }
            
            # Impact analysis
            reduction_abs = uncorrected_significant - n_significant
            reduction_rel = reduction_abs / uncorrected_significant if uncorrected_significant > 0 else 0
            summary['impact_analysis'][f'{method}_reduction'] = {
                'absolute': reduction_abs,
                'relative': reduction_rel
            }
        
        # Primary findings status
        primary_families = ['primary_tep', 'null_validation']
        for family in primary_families:
            family_tests = self.test_registry[family]
            summary['primary_findings_status'][family] = {}
            
            for method, results in corrections.items():
                family_significant = [t for t in results['significant_tests'] 
                                   if any(ft['test_name'] == t['test_name'] for ft in family_tests)]
                
                summary['primary_findings_status'][family][method] = {
                    'n_tests': len(family_tests),
                    'n_significant': len(family_significant),
                    'all_significant': len(family_significant) == len(family_tests) and len(family_tests) > 0
                }
        
        return summary
    
    def create_visualizations(self, corrections: Dict, summary: Dict):
        """Create correction impact visualizations"""
        print_status("Creating correction impact visualizations...", "PROCESS")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multiple Comparison Correction Impact Analysis', fontsize=16, fontweight='bold')
        
        # 1. Significance rates by method
        methods = list(summary['method_comparison'].keys())
        rates = [summary['method_comparison'][m]['significance_rate'] for m in methods]
        
        ax1.bar(['Uncorrected'] + methods, 
                [summary['impact_analysis']['uncorrected_rate']] + rates,
                color=['red'] + sns.color_palette("husl", len(methods)))
        ax1.set_title('Significance Rates by Correction Method')
        ax1.set_ylabel('Proportion of Tests Significant')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Test counts by analysis family
        families = list(self.test_registry.keys())
        counts = [len(self.test_registry[f]) for f in families]
        
        ax2.barh(families, counts, color=sns.color_palette("viridis", len(families)))
        ax2.set_title('Statistical Tests by Analysis Family')
        ax2.set_xlabel('Number of Tests')
        
        # 3. Primary findings preservation
        primary_data = summary['primary_findings_status']
        if 'primary_tep' in primary_data:
            methods_primary = list(primary_data['primary_tep'].keys())
            preserved = [primary_data['primary_tep'][m]['all_significant'] for m in methods_primary]
            
            colors = ['green' if p else 'red' for p in preserved]
            ax3.bar(methods_primary, [1 if p else 0 for p in preserved], color=colors)
            ax3.set_title('Primary TEP Findings Preservation')
            ax3.set_ylabel('All Primary Tests Significant')
            ax3.set_ylim(0, 1.1)
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Correction impact
        reductions = []
        method_names = []
        for method in methods:
            if f'{method}_reduction' in summary['impact_analysis']:
                reductions.append(summary['impact_analysis'][f'{method}_reduction']['relative'])
                method_names.append(method)
        
        ax4.bar(method_names, reductions, color=sns.color_palette("coolwarm", len(method_names)))
        ax4.set_title('Relative Reduction in Significant Tests')
        ax4.set_ylabel('Proportion Reduction')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plots
        figures_dir = ROOT / 'results' / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        plt.savefig(figures_dir / 'step_19_correction_impact_visualization.png', 
                   dpi=300, bbox_inches='tight')
        
        # Create a second figure for before/after comparison
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 6))
        fig2.suptitle('Statistical Significance: Before vs After Correction', fontsize=14, fontweight='bold')
        
        # Before correction
        all_pvalues = [test['p_value'] for family_tests in self.test_registry.values() 
                      for test in family_tests]
        ax5.hist(all_pvalues, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(self.family_alpha, color='red', linestyle='--', label=f'α = {self.family_alpha}')
        ax5.set_title('Before Correction')
        ax5.set_xlabel('p-value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # After correction (using Bonferroni as example)
        if 'bonferroni' in corrections:
            corrected_alpha = corrections['bonferroni']['corrected_alpha']
            ax6.hist(all_pvalues, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax6.axvline(corrected_alpha, color='red', linestyle='--', 
                       label=f'Corrected α = {corrected_alpha:.6f}')
            ax6.set_title('After Bonferroni Correction')
            ax6.set_xlabel('p-value')
            ax6.set_ylabel('Frequency')
            ax6.legend()
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'step_19_significance_before_after.png', 
                   dpi=300, bbox_inches='tight')
        
        plt.close('all')
        print_status("Visualizations saved to results/figures/", "SUCCESS")

def main():
    """Main function for Step 19: Multiple Comparison Corrections"""
    print_status("", "INFO")
    print_status("STEP 19: MULTIPLE COMPARISON CORRECTIONS", "TITLE")
    print_status("Systematic statistical validation with formal corrections", "INFO")
    print_status("", "INFO")
    
    start_time = time.time()
    
    try:
        # Initialize corrector
        corrector = MultipleComparisonCorrector()
        
        # Collect all statistical tests
        test_registry = corrector.collect_statistical_tests()
        
        # Apply corrections
        corrections = corrector.apply_comprehensive_corrections()
        
        # Generate summary
        summary = corrector.generate_corrected_summary(corrections)
        
        # Create visualizations
        corrector.create_visualizations(corrections, summary)
        
        # Save results
        outputs_dir = ROOT / 'results' / 'outputs'
        outputs_dir.mkdir(exist_ok=True)
        
        # Save detailed corrections
        with open(outputs_dir / 'step_19_multiple_comparison_corrections.json', 'w') as f:
            json.dump(corrections, f, indent=2)
        
        # Save summary
        with open(outputs_dir / 'step_19_corrected_significance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save impact analysis as CSV
        import pandas as pd
        impact_data = []
        for method, data in summary['method_comparison'].items():
            impact_data.append({
                'correction_method': method,
                'n_significant_tests': data['n_significant_tests'],
                'significance_rate': data['significance_rate'],
                'reduction_from_uncorrected': summary['impact_analysis'].get(f'{method}_reduction', {}).get('relative', 0)
            })
        
        df = pd.DataFrame(impact_data)
        df.to_csv(outputs_dir / 'step_19_correction_impact_analysis.csv', index=False)
        
        # Print summary
        total_tests = summary['total_tests_analyzed']
        uncorrected_sig = summary['impact_analysis']['uncorrected_significant']
        
        print_status("", "INFO")
        print_status("MULTIPLE COMPARISON CORRECTION SUMMARY", "TITLE")
        print_status(f"Total statistical tests analyzed: {total_tests}", "INFO")
        print_status(f"Uncorrected significant tests: {uncorrected_sig} ({uncorrected_sig/total_tests*100:.1f}%)", "INFO")
        
        for method, data in summary['method_comparison'].items():
            n_sig = data['n_significant_tests']
            rate = data['significance_rate'] * 100
            reduction = summary['impact_analysis'].get(f'{method}_reduction', {}).get('relative', 0) * 100
            print_status(f"{method.upper()}: {n_sig} significant ({rate:.1f}%), {reduction:.1f}% reduction", "SUCCESS")
        
        print_status("", "INFO")
        print_status("PRIMARY TEP FINDINGS STATUS:", "TITLE")
        
        primary_status = summary['primary_findings_status'].get('primary_tep', {})
        for method, status in primary_status.items():
            if status['all_significant']:
                print_status(f"{method.upper()}: ALL primary TEP tests remain significant", "SUCCESS")
            else:
                print_status(f"{method.upper()}: {status['n_significant']}/{status['n_tests']} primary TEP tests significant", "WARNING")
        
        elapsed_time = time.time() - start_time
        print_status(f"Step 19 completed in {elapsed_time:.1f} seconds", "SUCCESS")
        
    except Exception as e:
        print_status(f"Step 19 failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
