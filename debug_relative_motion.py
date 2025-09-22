#!/usr/bin/env python3
"""
DEBUG SCRIPT: Isolate and test relative motion analysis
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import sys
import os

# Add the scripts directory to path
sys.path.append('/Users/matthewsmawfield/www/TEP-GNSS/scripts')

from steps.step_5_tep_statistical_validation import load_complete_geospatial_dataset, compute_azimuth
from scripts.utils.config import TEPConfig

def debug_relative_motion_analysis():
    """Debug version of relative motion analysis"""
    print("🔧 DEBUGGING RELATIVE MOTION ANALYSIS")
    print("=" * 50)

    # Load config
    config = TEPConfig()
    print("✅ Config loaded")

    # Load data
    print("📊 Loading geospatial dataset...")
    try:
        complete_df = load_complete_geospatial_dataset('esa_final')
        print(f"✅ Data loaded: {len(complete_df):,} pairs")
        print(f"📅 Date range: {complete_df['date'].min()} to {complete_df['date'].max()}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return

    # Check data quality
    print("\n📊 DATA QUALITY CHECKS:")
    print(f"   Missing azimuth: {complete_df['azimuth'].isna().sum()}")
    print(f"   Missing coherence: {complete_df['coherence'].isna().sum()}")
    print(f"   Coherence range: {complete_df['coherence'].min():.3f} to {complete_df['coherence'].max():.3f}")
    print(f"   Coherence std: {complete_df['coherence'].std():.6f}")

    # Calculate days_since_epoch
    complete_df['date'] = pd.to_datetime(complete_df['date'])
    epoch_date = pd.to_datetime('2023-01-01')
    complete_df['days_since_epoch'] = (complete_df['date'] - epoch_date).dt.days
    print(f"   Days since epoch range: {complete_df['days_since_epoch'].min()} to {complete_df['days_since_epoch'].max()}")

    # Test relative motion analysis
    print("\n🧪 TESTING RELATIVE MOTION ANALYSIS:")

    # Define beat frequencies
    beat_frequencies = {
        'tidal_m2_tidal_s2_diff': 1/14.765,  # ~14.8 days
        'chandler_annual_sum': 1/196.9,      # ~197 days
        'chandler_semiannual_sum': 1/127.9,  # ~128 days
        'annual_semiannual_sum': 1/121.8     # ~122 days
    }

    # Filter for reasonable data
    df = complete_df.copy()
    df = df[df['coherence'] >= 0.1]  # Only high quality correlations
    df = df[df['dist_km'] > 100]     # Avoid very close stations
    df = df[df['dist_km'] < 5000]    # Focus on medium distances

    print(f"   Filtered to {len(df):,} pairs for analysis")

    for freq_name, freq_value in beat_frequencies.items():
        print(f"\n   🎵 Testing {freq_name} (freq: {freq_value:.4f} cpd)")

        # Calculate phase
        df['beat_phase'] = (2 * np.pi * df['days_since_epoch'] * freq_value) % (2 * np.pi)

        # Create phase bins
        n_bins = 12
        df['phase_bin'] = (df['beat_phase'] // (2 * np.pi / n_bins)).astype(int)

        phase_tracking = []
        for bin_idx in range(n_bins):
            bin_data = df[df['phase_bin'] == bin_idx]
            if len(bin_data) >= 50:  # Lower threshold for debugging
                phase_tracking.append({
                    'phase_bin': bin_idx,
                    'phase_radians': bin_idx * 2 * np.pi / n_bins,
                    'mean_coherence': bin_data['coherence'].mean(),
                    'n_pairs': len(bin_data)
                })

        print(f"      Phase bins collected: {len(phase_tracking)}")

        if len(phase_tracking) < 4:
            print(f"      ❌ Insufficient phase bins: {len(phase_tracking)} (need ≥4)")
            continue

        # Extract data for correlation
        phases = [t['phase_radians'] for t in phase_tracking]
        coherences = [t['mean_coherence'] for t in phase_tracking]

        print(f"      Phase range: {min(phases):.3f} to {max(phases):.3f}")
        print(f"      Coherence range: {min(coherences):.3f} to {max(coherences):.3f}")
        print(f"      Coherence std: {np.std(coherences):.6f}")

        # Check for sufficient variation
        coherence_std = np.std(coherences)
        if coherence_std < 1e-6:
            print(f"      ❌ Insufficient coherence variation: {coherence_std:.2e}")
            continue

        # Try correlation analysis
        try:
            phase_sin = np.sin(phases)
            phase_cos = np.cos(phases)

            corr_sin, p_sin = pearsonr(coherences, phase_sin)
            corr_cos, p_cos = pearsonr(coherences, phase_cos)

            print(f"      Sin correlation: r={corr_sin:.3f}, p={p_sin:.3f}")
            print(f"      Cos correlation: r={corr_cos:.3f}, p={p_cos:.3f}")

            if abs(corr_sin) > abs(corr_cos):
                best_corr = corr_sin
                best_p = p_sin
            else:
                best_corr = corr_cos
                best_p = p_cos

            print(f"      Best correlation: r={best_corr:.3f}, p={best_p:.3f}")

            if best_p < 0.3 and abs(best_corr) > 0.2:
                print(f"      ✅ SIGNIFICANT BEAT DETECTED!")
            else:
                print(f"      ❌ Below significance threshold")

        except Exception as e:
            print(f"      ❌ Correlation failed: {e}")

    print("\n🔧 DEBUGGING COMPLETE")
    return df

if __name__ == "__main__":
    debug_relative_motion_analysis()
