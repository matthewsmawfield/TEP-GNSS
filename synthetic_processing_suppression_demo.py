#!/usr/bin/env python3
"""
Synthetic Demonstration: GNSS Processing Suppression Paradox
==========================================================

This script demonstrates how standard GNSS processing suppresses globally 
coherent signals while phase-only metrics survive the suppression.

Based on claims from TEP-GNSS analysis:
- Standard GNSS processing applies network constraints that suppress 
  spatially correlated signals
- Phase-coherent analysis survives this suppression where amplitude-based 
  methods fail

References cited in TEP-GNSS:
- IERS Conventions 2010, Section 5.4.1 (IGS network processing)
- Dach et al., 2007, GPS Solutions (CODE network solution)

Author: Matthew Lukin Smawfield
Date: October 2025
Purpose: Demonstrate processing suppression paradox claims
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class GNSSProcessingSimulator:
    """
    Simulates GNSS processing effects on synthetic clock signals.
    
    Demonstrates the "processing suppression paradox":
    1. Generate synthetic globally coherent clock signals
    2. Apply standard GNSS processing constraints (amplitude suppression)
    3. Show phase-only metrics survive while amplitude metrics fail
    """
    
    def __init__(self, n_stations: int = 20, n_days: int = 100, 
                 sampling_rate_hz: float = 1/1800):  # 30-minute sampling
        self.n_stations = n_stations
        self.n_days = n_days
        self.dt = 1 / sampling_rate_hz  # seconds
        self.n_samples = int(n_days * 24 * 3600 / self.dt)
        self.time = np.arange(self.n_samples) * self.dt / 3600  # hours
        
        # Generate station positions (random global distribution)
        np.random.seed(42)
        self.station_lats = np.random.uniform(-60, 60, n_stations)
        self.station_lons = np.random.uniform(-180, 180, n_stations)
        
        # Calculate inter-station distances (simplified great circle)
        self.distances = self._calculate_distances()
        
    def _calculate_distances(self) -> np.ndarray:
        """Calculate great circle distances between all station pairs."""
        R = 6371.0  # Earth radius in km
        distances = np.zeros((self.n_stations, self.n_stations))
        
        for i in range(self.n_stations):
            for j in range(self.n_stations):
                if i != j:
                    lat1, lon1 = np.radians(self.station_lats[i]), np.radians(self.station_lons[i])
                    lat2, lon2 = np.radians(self.station_lats[j]), np.radians(self.station_lons[j])
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    distances[i, j] = R * c
                    
        return distances
    
    def generate_synthetic_tep_field(self) -> np.ndarray:
        """
        Generate synthetic TEP field with exponential spatial correlations.
        
        This represents the underlying "dynamical time field" that affects
        all clocks globally with distance-dependent correlations.
        """
        # TEP correlation parameters (from actual data)
        correlation_length_km = 3800  # λ from actual analysis
        correlation_amplitude = 1e-14  # Fractional frequency effect
        
        # Create globally coherent base signal
        # Multiple frequency components in TEP band (10-500 μHz)
        freq_tep_low = 1e-5   # 10 μHz
        freq_tep_high = 5e-4  # 500 μHz
        
        # Generate base temporal pattern
        base_signal = np.zeros(self.n_samples)
        
        # Add multiple frequency components
        freqs = np.logspace(np.log10(freq_tep_low), np.log10(freq_tep_high), 5)
        for freq in freqs:
            omega = 2 * np.pi * freq
            amplitude = correlation_amplitude / np.sqrt(len(freqs))
            phase = np.random.uniform(0, 2*np.pi)
            base_signal += amplitude * np.sin(omega * self.time * 3600 + phase)
        
        # Apply spatial correlation structure
        field = np.zeros((self.n_stations, self.n_samples))
        
        # Reference station (station 0) gets base signal
        field[0, :] = base_signal
        
        # Other stations get correlated signals based on distance
        for i in range(1, self.n_stations):
            distance = self.distances[0, i]  # Distance from reference station
            
            # Exponential correlation decay: C(r) = A * exp(-r/λ) + C₀
            correlation_strength = np.exp(-distance / correlation_length_km)
            
            # Add correlated component plus independent noise
            noise_level = correlation_amplitude * 0.5
            independent_noise = np.random.normal(0, noise_level, self.n_samples)
            
            field[i, :] = (correlation_strength * base_signal + 
                          np.sqrt(1 - correlation_strength**2) * independent_noise)
        
        return field
    
    def apply_standard_gnss_processing(self, raw_signals: np.ndarray) -> np.ndarray:
        """
        Apply standard GNSS processing constraints that suppress global coherence.
        
        Implements key suppression mechanisms identified in TEP analysis:
        1. Network datum constraints (sum of corrections = 0)
        2. Common-mode removal
        3. Reference clock stabilization
        """
        processed_signals = raw_signals.copy()
        
        # 1. Network Datum Constraints (IERS Conventions 2010, Section 5.4.1)
        # Force sum of clock corrections to zero at each epoch
        for t in range(self.n_samples):
            epoch_mean = np.mean(processed_signals[:, t])
            processed_signals[:, t] -= epoch_mean
        
        # 2. Common-mode filtering
        # Remove signals common across multiple stations
        window_size = min(48, self.n_samples // 10)  # ~24 hours
        for i in range(self.n_stations):
            # High-pass filter to remove long-term common trends
            processed_signals[i, :] = signal.detrend(processed_signals[i, :])
            
            # Additional common-mode suppression
            if window_size > 1:
                smoothed = signal.savgol_filter(processed_signals[i, :], 
                                              min(window_size, len(processed_signals[i, :])), 
                                              1)
                processed_signals[i, :] -= 0.7 * smoothed  # Partial suppression
        
        # 3. Reference clock stabilization
        # Stabilize against ensemble average (further suppresses correlations)
        for t in range(self.n_samples):
            ensemble_mean = np.mean(processed_signals[:, t])
            processed_signals[:, t] -= 0.5 * ensemble_mean
        
        # 4. Add realistic processing noise
        processing_noise_level = 1e-15  # Typical GNSS processing noise
        processing_noise = np.random.normal(0, processing_noise_level, 
                                          (self.n_stations, self.n_samples))
        processed_signals += processing_noise
        
        return processed_signals
    
    def compute_cross_spectral_density(self, signal1: np.ndarray, 
                                     signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute cross-spectral density between two signals."""
        # Use Welch's method for robust spectral estimation
        nperseg = min(1024, len(signal1) // 4)
        
        freqs, csd = signal.csd(signal1, signal2, 
                               fs=1/self.dt, 
                               nperseg=nperseg,
                               detrend='linear')
        
        # Convert to μHz for consistency with TEP analysis
        freqs_microhz = freqs * 1e6
        
        return freqs_microhz, csd, nperseg
    
    def extract_tep_band_correlation(self, csd: np.ndarray, 
                                   freqs: np.ndarray) -> Tuple[float, float]:
        """
        Extract correlation metrics from TEP frequency band (10-500 μHz).
        
        Returns both amplitude-based and phase-only metrics to demonstrate
        the suppression paradox.
        """
        # TEP frequency band
        tep_band_mask = (freqs >= 10) & (freqs <= 500)
        
        if not np.any(tep_band_mask):
            return 0.0, 0.0
        
        csd_band = csd[tep_band_mask]
        
        # Standard amplitude-based coherency (fails after processing)
        magnitudes = np.abs(csd_band)
        mean_magnitude = np.mean(magnitudes)
        amplitude_correlation = mean_magnitude
        
        # Phase-only metric (survives processing)
        phases = np.angle(csd_band)
        
        # Magnitude-weighted circular averaging (as in TEP method)
        weights = magnitudes / np.sum(magnitudes) if np.sum(magnitudes) > 0 else np.ones_like(magnitudes)
        weighted_phase = np.angle(np.sum(weights * np.exp(1j * phases)))
        phase_correlation = np.cos(weighted_phase)
        
        return amplitude_correlation, phase_correlation
    
    def analyze_correlation_structure(self, signals: np.ndarray, 
                                    label: str) -> Dict:
        """Analyze distance-dependent correlation structure."""
        n_pairs = 0
        distances_list = []
        amplitude_correlations = []
        phase_correlations = []
        
        # Analyze all station pairs
        for i in range(self.n_stations):
            for j in range(i + 1, self.n_stations):
                distance = self.distances[i, j]
                
                # Skip very close pairs (< 100 km) and very distant (> 15000 km)
                if distance < 100 or distance > 15000:
                    continue
                
                # Compute cross-spectral density
                freqs, csd, _ = self.compute_cross_spectral_density(
                    signals[i, :], signals[j, :])
                
                # Extract correlations in TEP band
                amp_corr, phase_corr = self.extract_tep_band_correlation(csd, freqs)
                
                distances_list.append(distance)
                amplitude_correlations.append(amp_corr)
                phase_correlations.append(phase_corr)
                n_pairs += 1
        
        return {
            'label': label,
            'n_pairs': n_pairs,
            'distances': np.array(distances_list),
            'amplitude_correlations': np.array(amplitude_correlations),
            'phase_correlations': np.array(phase_correlations)
        }
    
    def fit_exponential_model(self, distances: np.ndarray, 
                            correlations: np.ndarray) -> Dict:
        """Fit exponential decay model: C(r) = A * exp(-r/λ) + C₀"""
        from scipy.optimize import curve_fit
        
        def exponential_model(r, A, lambda_km, C0):
            return A * np.exp(-r / lambda_km) + C0
        
        # Remove invalid correlations
        valid_mask = np.isfinite(correlations) & (correlations != 0)
        if np.sum(valid_mask) < 5:
            return {'success': False, 'reason': 'insufficient_valid_data'}
        
        distances_valid = distances[valid_mask]
        correlations_valid = correlations[valid_mask]
        
        try:
            # Initial parameter guesses
            p0 = [np.max(correlations_valid) - np.min(correlations_valid), 
                  3000,  # lambda ~ 3000 km
                  np.min(correlations_valid)]
            
            bounds = ([0, 500, -1], 
                     [1, 15000, 1])
            
            popt, pcov = curve_fit(exponential_model, distances_valid, 
                                 correlations_valid, p0=p0, bounds=bounds)
            
            # Calculate R²
            y_pred = exponential_model(distances_valid, *popt)
            ss_res = np.sum((correlations_valid - y_pred)**2)
            ss_tot = np.sum((correlations_valid - np.mean(correlations_valid))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'success': True,
                'A': popt[0],
                'lambda_km': popt[1],
                'C0': popt[2],
                'r_squared': r_squared,
                'param_errors': np.sqrt(np.diag(pcov))
            }
        
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    def run_demonstration(self) -> Dict:
        """Run complete demonstration of processing suppression paradox."""
        print("=" * 60)
        print("GNSS Processing Suppression Paradox Demonstration")
        print("=" * 60)
        print()
        print("Based on TEP-GNSS claims:")
        print("- Standard GNSS processing suppresses globally coherent signals")
        print("- Phase-only metrics survive suppression where amplitude fails")
        print()
        
        # 1. Generate synthetic TEP field
        print("1. Generating synthetic TEP field with exponential correlations...")
        raw_field = self.generate_synthetic_tep_field()
        print(f"   Generated field for {self.n_stations} stations over {self.n_days} days")
        print()
        
        # 2. Apply standard GNSS processing
        print("2. Applying standard GNSS processing constraints...")
        processed_field = self.apply_standard_gnss_processing(raw_field)
        print("   Applied network datum constraints, common-mode removal, and stabilization")
        print()
        
        # 3. Analyze raw signals
        print("3. Analyzing correlation structure in raw signals...")
        raw_analysis = self.analyze_correlation_structure(raw_field, "Raw Signals")
        print(f"   Analyzed {raw_analysis['n_pairs']} station pairs")
        
        # 4. Analyze processed signals  
        print("4. Analyzing correlation structure after processing...")
        processed_analysis = self.analyze_correlation_structure(processed_field, "Processed Signals")
        print()
        
        # 5. Fit exponential models
        results = {}
        
        for analysis, signal_type in [(raw_analysis, 'raw'), (processed_analysis, 'processed')]:
            print(f"5. Fitting exponential models for {signal_type} signals:")
            
            # Amplitude-based correlation
            amp_fit = self.fit_exponential_model(analysis['distances'], 
                                               analysis['amplitude_correlations'])
            
            # Phase-only correlation  
            phase_fit = self.fit_exponential_model(analysis['distances'],
                                                 analysis['phase_correlations'])
            
            results[signal_type] = {
                'amplitude_fit': amp_fit,
                'phase_fit': phase_fit,
                'analysis': analysis
            }
            
            if amp_fit['success']:
                print(f"   Amplitude-based: λ = {amp_fit['lambda_km']:.0f} km, "
                      f"R² = {amp_fit['r_squared']:.3f}")
            else:
                print(f"   Amplitude-based: FAILED ({amp_fit['reason']})")
            
            if phase_fit['success']:
                print(f"   Phase-only: λ = {phase_fit['lambda_km']:.0f} km, "
                      f"R² = {phase_fit['r_squared']:.3f}")
            else:
                print(f"   Phase-only: FAILED ({phase_fit['reason']})")
            print()
        
        return results
    
    def create_visualization(self, results: Dict):
        """Create visualization of the processing suppression paradox."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GNSS Processing Suppression Paradox Demonstration', fontsize=16)
        
        colors = {'raw': 'blue', 'processed': 'red'}
        
        for idx, (signal_type, data) in enumerate(results.items()):
            analysis = data['analysis']
            
            # Amplitude correlations
            ax_amp = axes[idx, 0]
            ax_amp.scatter(analysis['distances'], analysis['amplitude_correlations'], 
                          alpha=0.6, s=20, color=colors[signal_type])
            ax_amp.set_xlabel('Distance (km)')
            ax_amp.set_ylabel('Amplitude Correlation')
            ax_amp.set_title(f'{signal_type.title()} - Amplitude-Based')
            ax_amp.grid(True, alpha=0.3)
            
            if data['amplitude_fit']['success']:
                distances_model = np.linspace(100, 15000, 100)
                A, lam, C0 = data['amplitude_fit']['A'], data['amplitude_fit']['lambda_km'], data['amplitude_fit']['C0']
                correlations_model = A * np.exp(-distances_model / lam) + C0
                ax_amp.plot(distances_model, correlations_model, '--', 
                           color=colors[signal_type], alpha=0.8,
                           label=f'λ = {lam:.0f} km, R² = {data["amplitude_fit"]["r_squared"]:.3f}')
                ax_amp.legend()
            
            # Phase correlations
            ax_phase = axes[idx, 1]
            ax_phase.scatter(analysis['distances'], analysis['phase_correlations'], 
                           alpha=0.6, s=20, color=colors[signal_type])
            ax_phase.set_xlabel('Distance (km)')
            ax_phase.set_ylabel('Phase Correlation (cos φ)')
            ax_phase.set_title(f'{signal_type.title()} - Phase-Only')
            ax_phase.grid(True, alpha=0.3)
            
            if data['phase_fit']['success']:
                distances_model = np.linspace(100, 15000, 100)
                A, lam, C0 = data['phase_fit']['A'], data['phase_fit']['lambda_km'], data['phase_fit']['C0']
                correlations_model = A * np.exp(-distances_model / lam) + C0
                ax_phase.plot(distances_model, correlations_model, '--', 
                            color=colors[signal_type], alpha=0.8,
                            label=f'λ = {lam:.0f} km, R² = {data["phase_fit"]["r_squared"]:.3f}')
                ax_phase.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/matthewsmawfield/www/TEP-GNSS/processing_suppression_paradox_demo.png', 
                    dpi=300, bbox_inches='tight')
        print("Visualization saved as: processing_suppression_paradox_demo.png")
        return fig

def main():
    """Run the complete processing suppression paradox demonstration."""
    # Create simulator
    simulator = GNSSProcessingSimulator(n_stations=25, n_days=90)
    
    # Run demonstration
    results = simulator.run_demonstration()
    
    # Create visualization
    simulator.create_visualization(results)
    
    # Summary conclusions
    print("=" * 60)
    print("DEMONSTRATION CONCLUSIONS")
    print("=" * 60)
    print()
    
    raw_amp_success = results['raw']['amplitude_fit']['success']
    raw_phase_success = results['raw']['phase_fit']['success']
    proc_amp_success = results['processed']['amplitude_fit']['success']
    proc_phase_success = results['processed']['phase_fit']['success']
    
    print("Raw Signals (before processing):")
    if raw_amp_success:
        print(f"  ✓ Amplitude-based detection: λ = {results['raw']['amplitude_fit']['lambda_km']:.0f} km")
    else:
        print("  ✗ Amplitude-based detection: FAILED")
    
    if raw_phase_success:
        print(f"  ✓ Phase-only detection: λ = {results['raw']['phase_fit']['lambda_km']:.0f} km")
    else:
        print("  ✗ Phase-only detection: FAILED")
    print()
    
    print("Processed Signals (after GNSS processing constraints):")
    if proc_amp_success:
        print(f"  ✓ Amplitude-based detection: λ = {results['processed']['amplitude_fit']['lambda_km']:.0f} km")
    else:
        print("  ✗ Amplitude-based detection: SUPPRESSED")
    
    if proc_phase_success:
        print(f"  ✓ Phase-only detection: λ = {results['processed']['phase_fit']['lambda_km']:.0f} km")
    else:
        print("  ✗ Phase-only detection: FAILED")
    print()
    
    print("Processing Suppression Paradox Summary:")
    if proc_amp_success and proc_phase_success:
        amp_degradation = (results['raw']['amplitude_fit']['r_squared'] - 
                          results['processed']['amplitude_fit']['r_squared'])
        phase_degradation = (results['raw']['phase_fit']['r_squared'] - 
                           results['processed']['phase_fit']['r_squared'])
        
        print(f"  • Amplitude method degradation: ΔR² = {amp_degradation:.3f}")
        print(f"  • Phase method degradation: ΔR² = {phase_degradation:.3f}")
        
        if amp_degradation > phase_degradation:
            print("  ✓ PARADOX DEMONSTRATED: Phase-only metrics more robust to processing")
        else:
            print("  ? Paradox not clearly demonstrated in this synthetic case")
    
    elif proc_phase_success and not proc_amp_success:
        print("  ✓ STRONG PARADOX: Amplitude suppressed, phase-only survives")
        print("  This matches TEP-GNSS claims about processing suppression")
    
    else:
        print("  ? Results inconclusive - may need parameter adjustment")
    
    print()
    print("Note: This synthetic demonstration illustrates the claimed mechanisms.")
    print("Real GNSS processing involves additional complexities not modeled here.")

if __name__ == "__main__":
    main()
