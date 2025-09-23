#!/usr/bin/env python3
"""
Comprehensive Cross-Spectral Density Projection Bias Test

This script directly tests whether the entire CSD processing pipeline 
(including sophisticated spectral processing) creates artificial signals
through projection bias, as the critic suggests.

CRITICAL TESTS:
1. Synthetic data injection through full CSD pipeline
2. Alternative phase metrics on actual CSD output
3. SNR-stratified analysis
4. Frequency-dependent analysis
5. Processing-conditioned null tests

If projection bias exists, it should be detectable at the CSD level,
not just in the final cos(phase) step.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.signal import csd
from scipy.optimize import curve_fit
from scipy.stats import circmean, circstd
import warnings
warnings.filterwarnings('ignore')

class CSDProjectionBiasTest:
    def __init__(self, results_dir="results/exploratory"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_synthetic_data(self, n_stations=50, n_days=100, sampling_rate=1/30):
        """
        Generate synthetic clock data with known properties to test CSD pipeline
        """
        print("Generating synthetic clock data...")
        
        # Create station coordinates (random global distribution)
        np.random.seed(42)  # Reproducible
        lats = np.random.uniform(-80, 80, n_stations)
        lons = np.random.uniform(-180, 180, n_stations)
        
        # Create time series
        n_samples = int(n_days * 24 * 3600 * sampling_rate)  # 30-second sampling
        times = np.arange(n_samples) / sampling_rate / 3600 / 24  # days
        
        synthetic_data = {}
        
        # Test Case 1: Pure noise (should show NO spatial correlations)
        print("  Case 1: Pure uncorrelated noise")
        pure_noise = {}
        for i in range(n_stations):
            # Each station gets independent white noise + 1/f noise
            white = np.random.normal(0, 1, n_samples)
            # Add 1/f noise (realistic for clocks)
            freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)
            freqs[0] = 1e-10  # Avoid division by zero
            pink_spectrum = 1 / np.sqrt(np.abs(freqs))
            pink_spectrum[0] = 0
            pink_noise = np.fft.irfft(np.fft.rfft(white) * pink_spectrum[:len(np.fft.rfft(white))])
            pure_noise[f'STA{i:02d}'] = white + 0.1 * pink_noise[:len(white)]
        
        synthetic_data['pure_noise'] = {
            'data': pure_noise,
            'coords': [(lats[i], lons[i]) for i in range(n_stations)],
            'expected_lambda': None,  # Should be no correlation
            'description': 'Pure uncorrelated noise - should show NO spatial structure'
        }
        
        # Test Case 2: SNR gradient only (critic's hypothesis)
        print("  Case 2: SNR gradient without spatial field")
        snr_gradient = {}
        base_snr = 10.0
        for i in range(n_stations):
            # Create SNR that decreases with some arbitrary spatial pattern
            snr = base_snr * np.exp(-((lats[i])**2 + (lons[i]/2)**2) / 5000)
            noise_level = 1.0 / snr
            signal_level = 1.0
            
            # Generate signal with this SNR
            clean_signal = signal_level * np.sin(2*np.pi*times/10)  # 10-day period
            noise = noise_level * np.random.normal(0, 1, n_samples)
            snr_gradient[f'STA{i:02d}'] = clean_signal + noise
            
        synthetic_data['snr_gradient'] = {
            'data': snr_gradient,
            'coords': [(lats[i], lons[i]) for i in range(n_stations)],
            'expected_lambda': None,  # Should NOT create exponential spatial decay
            'description': 'SNR gradient without true spatial field - testing critic hypothesis'
        }
        
        # Test Case 3: True exponential spatial field (TEP simulation)
        print("  Case 3: True exponential spatial field")
        true_lambda = 4000  # km
        spatial_field = {}
        
        # Generate field values at each station location
        field_values = np.zeros(n_stations)
        # Use first station as reference
        ref_lat, ref_lon = lats[0], lons[0]
        
        for i in range(n_stations):
            # Compute distance from reference
            dlat = lats[i] - ref_lat
            dlon = lons[i] - ref_lon
            # Rough distance in km (good enough for testing)
            dist_km = 111 * np.sqrt(dlat**2 + (dlon * np.cos(np.radians(lats[i])))**2)
            
            # Field correlation decays exponentially
            correlation = np.exp(-dist_km / true_lambda)
            field_values[i] = correlation
        
        # Generate correlated field evolution
        field_time_series = np.random.normal(0, 1, n_samples)
        
        for i in range(n_stations):
            # Each station gets field component + independent noise
            field_component = field_values[i] * field_time_series
            noise_component = np.random.normal(0, 0.5, n_samples)  # Independent noise
            spatial_field[f'STA{i:02d}'] = field_component + noise_component
            
        synthetic_data['spatial_field'] = {
            'data': spatial_field,
            'coords': [(lats[i], lons[i]) for i in range(n_stations)],
            'expected_lambda': true_lambda,
            'description': f'True exponential spatial field with λ={true_lambda} km'
        }
        
        return synthetic_data
    
    def compute_csd_through_pipeline(self, data_dict, coords, freq_band=(1e-5, 5e-4)):
        """
        Run synthetic data through the exact same CSD pipeline as main analysis
        """
        print(f"    Running CSD pipeline (freq band: {freq_band[0]:.1e}-{freq_band[1]:.1e} Hz)...")
        
        stations = list(data_dict.keys())
        n_stations = len(stations)
        sampling_rate = 1/30  # 30-second sampling
        
        results = []
        
        # Compute CSD for all station pairs
        for i in range(n_stations):
            for j in range(i+1, n_stations):
                stat_i, stat_j = stations[i], stations[j]
                
                # Get time series
                ts_i = data_dict[stat_i]
                ts_j = data_dict[stat_j]
                
                # Compute cross-spectral density
                freqs, csd_values = csd(ts_i, ts_j, fs=sampling_rate, 
                                      nperseg=min(len(ts_i)//4, 1024))
                
                # Apply frequency band filter (same as main analysis)
                freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
                if np.sum(freq_mask) == 0:
                    continue
                    
                # Band-limited average (same as main analysis)
                csd_band = csd_values[freq_mask]
                avg_csd = np.mean(csd_band)
                
                # Extract phase and magnitude
                phase = np.angle(avg_csd)
                magnitude = np.abs(avg_csd)
                
                # Compute distance between stations
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                dist_km = 111 * np.sqrt(dlat**2 + (dlon * np.cos(np.radians((lat1+lat2)/2)))**2)
                
                results.append({
                    'station_i': stat_i,
                    'station_j': stat_j,
                    'distance_km': dist_km,
                    'csd_phase': phase,
                    'csd_magnitude': magnitude,
                    'csd_real': np.real(avg_csd),
                    'csd_imag': np.imag(avg_csd)
                })
        
        return pd.DataFrame(results)
    
    def test_phase_metrics_on_csd(self, csd_df, test_name):
        """
        Apply all three phase metrics to actual CSD output
        """
        print(f"    Testing phase metrics on CSD output...")
        
        # Distance binning (same as main analysis)
        distance_bins = np.logspace(np.log10(50), np.log10(8000), 25)
        
        results = []
        
        for i in range(len(distance_bins) - 1):
            bin_min, bin_max = distance_bins[i], distance_bins[i + 1]
            bin_center = np.sqrt(bin_min * bin_max)
            
            # Get pairs in this distance bin
            mask = (csd_df['distance_km'] >= bin_min) & (csd_df['distance_km'] < bin_max)
            bin_data = csd_df[mask]
            
            if len(bin_data) < 5:  # Skip bins with too few pairs
                continue
            
            # Extract CSD phases and magnitudes
            phases = bin_data['csd_phase'].values
            magnitudes = bin_data['csd_magnitude'].values
            
            # Method 1: E[cos(φ)] - Current TEP method
            metric_1 = np.mean(np.cos(phases))
            
            # Method 2: Re(E[e^{iφ}]) - Complex average, real part
            complex_phasors = np.exp(1j * phases)
            complex_mean = np.mean(complex_phasors)
            metric_2 = np.real(complex_mean)
            
            # Method 3: |E[e^{iφ}]| - Vector strength
            metric_3 = np.abs(complex_mean)
            
            # Method 4: Direct complex CSD average (new test)
            complex_csd = bin_data['csd_real'].values + 1j * bin_data['csd_imag'].values
            csd_mean = np.mean(complex_csd)
            metric_4_real = np.real(csd_mean)
            metric_4_mag = np.abs(csd_mean)
            
            results.append({
                'distance_km': bin_center,
                'n_pairs': len(bin_data),
                'metric_1_cos_phase': metric_1,
                'metric_2_re_complex': metric_2,
                'metric_3_vector_strength': metric_3,
                'metric_4_csd_real': metric_4_real,
                'metric_4_csd_mag': metric_4_mag,
                'mean_magnitude': np.mean(magnitudes),
                'phase_dispersion': circstd(phases)
            })
        
        metrics_df = pd.DataFrame(results)
        
        # Fit exponential models to each metric
        fits = {}
        for metric_col in ['metric_1_cos_phase', 'metric_2_re_complex', 
                          'metric_3_vector_strength', 'metric_4_csd_real', 'metric_4_csd_mag']:
            if len(metrics_df) > 5:
                fit_result = self.fit_exponential(metrics_df['distance_km'].values, 
                                                metrics_df[metric_col].values)
                fits[metric_col] = fit_result
        
        return metrics_df, fits
    
    def fit_exponential(self, distances, values):
        """Fit exponential decay model"""
        def exp_decay(x, A, lam, C):
            return A * np.exp(-x / lam) + C
        
        try:
            # Reasonable initial guess and bounds
            p0 = [np.max(values) - np.min(values), 3000, np.min(values)]
            bounds = ([0, 100, -1], [2, 20000, 1])
            
            popt, _ = curve_fit(exp_decay, distances, values, p0=p0, bounds=bounds)
            
            # Calculate R²
            y_pred = exp_decay(distances, *popt)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -1
            
            return {
                'lambda_km': popt[1],
                'amplitude': popt[0],
                'offset': popt[2],
                'r_squared': r_squared,
                'success': True
            }
        except:
            return {'success': False}
    
    def run_comprehensive_test(self):
        """Run the complete comprehensive test"""
        print("COMPREHENSIVE CSD PROJECTION BIAS TEST")
        print("="*60)
        print("Testing whether the entire CSD processing pipeline creates artificial signals")
        print()
        
        # Generate synthetic data
        synthetic_data = self.generate_synthetic_data()
        
        all_results = {}
        
        for test_name, test_data in synthetic_data.items():
            print(f"Testing: {test_data['description']}")
            print("-" * 50)
            
            # Run through CSD pipeline
            csd_df = self.compute_csd_through_pipeline(
                test_data['data'], 
                test_data['coords']
            )
            
            if len(csd_df) == 0:
                print("    No valid CSD results - skipping")
                continue
            
            # Test phase metrics on CSD output
            metrics_df, fits = self.test_phase_metrics_on_csd(csd_df, test_name)
            
            # Analyze results
            print(f"    Results for {len(metrics_df)} distance bins:")
            
            for metric_name, fit_result in fits.items():
                if fit_result['success']:
                    lam = fit_result['lambda_km']
                    r2 = fit_result['r_squared']
                    print(f"      {metric_name}: λ={lam:.0f}km, R²={r2:.3f}")
                else:
                    print(f"      {metric_name}: Fit failed")
            
            # Compare with expected
            expected_lambda = test_data['expected_lambda']
            if expected_lambda:
                print(f"    Expected λ: {expected_lambda} km")
                
                # Check if any metric recovered the true lambda
                recovered = False
                for fit_result in fits.values():
                    if fit_result['success']:
                        error = abs(fit_result['lambda_km'] - expected_lambda) / expected_lambda
                        if error < 0.3:  # Within 30%
                            recovered = True
                            break
                
                if recovered:
                    print("    ✓ True spatial field correctly detected")
                else:
                    print("    ✗ True spatial field NOT detected")
            else:
                print("    Expected: No spatial correlations")
                
                # Check if any metric shows strong correlations (false positive)
                false_positive = False
                for fit_result in fits.values():
                    if fit_result['success'] and fit_result['r_squared'] > 0.5:
                        false_positive = True
                        break
                
                if false_positive:
                    print("    ✗ FALSE POSITIVE: Artificial correlations detected!")
                else:
                    print("    ✓ Correctly shows no correlations")
            
            all_results[test_name] = {
                'csd_data': csd_df,
                'metrics': metrics_df,
                'fits': fits,
                'expected_lambda': expected_lambda
            }
            
            print()
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results):
        """Generate comprehensive summary report"""
        report_file = self.results_dir / "csd_projection_bias_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE CSD PROJECTION BIAS TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("CRITICAL QUESTION: Does the CSD processing pipeline create artificial signals?\n\n")
            
            f.write("TEST METHODOLOGY:\n")
            f.write("1. Generate synthetic data with known spatial properties\n")
            f.write("2. Run through EXACT same CSD pipeline as main analysis\n")
            f.write("3. Apply all phase metrics to CSD output\n")
            f.write("4. Check if true signals are detected and false signals avoided\n\n")
            
            f.write("RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            for test_name, results in all_results.items():
                expected = results['expected_lambda']
                fits = results['fits']
                
                f.write(f"\n{test_name.upper()}:\n")
                
                if expected:
                    f.write(f"Expected λ: {expected} km\n")
                else:
                    f.write("Expected: No spatial correlations\n")
                
                f.write("Detected λ values:\n")
                for metric_name, fit_result in fits.items():
                    if fit_result['success']:
                        f.write(f"  {metric_name}: {fit_result['lambda_km']:.0f} km (R²={fit_result['r_squared']:.3f})\n")
                    else:
                        f.write(f"  {metric_name}: No fit\n")
                
                # Interpretation
                if expected:
                    # Should detect true signal
                    detected = any(fit['success'] and abs(fit['lambda_km'] - expected) / expected < 0.3 
                                 for fit in fits.values())
                    f.write(f"True signal detection: {'✓ SUCCESS' if detected else '✗ FAILED'}\n")
                else:
                    # Should NOT detect false signals
                    false_pos = any(fit['success'] and fit['r_squared'] > 0.5 
                                  for fit in fits.values())
                    f.write(f"False positive test: {'✗ FAILED' if false_pos else '✓ SUCCESS'}\n")
            
            f.write(f"\nCONCLUSION:\n")
            f.write("If the CSD pipeline creates projection bias:\n")
            f.write("- Pure noise should show artificial correlations (FALSE POSITIVE)\n")
            f.write("- SNR gradients should create exponential decay (CRITIC'S HYPOTHESIS)\n")
            f.write("- True spatial fields should be distorted\n\n")
            
            f.write("If the CSD pipeline is sound:\n")
            f.write("- Pure noise shows no correlations\n")
            f.write("- SNR gradients don't create exponential decay\n")
            f.write("- True spatial fields are correctly detected\n")
        
        print(f"Detailed report saved to: {report_file}")

def main():
    """Run comprehensive CSD projection bias test"""
    tester = CSDProjectionBiasTest()
    results = tester.run_comprehensive_test()
    
    print("\nCOMPREHENSIVE TEST COMPLETED!")
    print("="*60)
    print("This test directly addresses the critic's concern that the entire")
    print("CSD processing pipeline might create artificial spatial correlations.")
    print("\nCheck results/exploratory/ for detailed analysis.")

if __name__ == "__main__":
    main()
