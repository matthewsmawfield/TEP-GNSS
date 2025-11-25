#!/usr/bin/env python3
"""
Step 2.6: Validation Tests (CODE Long-Span Analysis)

Two critical validation tests that can run independently without re-running Step 2.2:

1. NULL EVENT CONTROL TEST
   Tests detection rate for RANDOM dates vs planetary alignments.
   If random dates show similar detection rates, planetary detection
   would not be specific to alignments.

2. TIDAL SCALING TEST (M/r³ vs M/r²)
   Tests whether planetary event amplitudes correlate with tidal potential
   (M/r³) or gravitational potential (M/r²). Absence of scaling is expected
   due to GNSS processing filter suppression.

Usage:
    # Run both tests
    python scripts/code_longspan/step_2_6_null_control.py --center code
    
    # Run specific tests only
    python scripts/code_longspan/step_2_6_null_control.py --center code --skip-null
    python scripts/code_longspan/step_2_6_null_control.py --center code --skip-tidal
    
    # Update Step 2.2 results with validation test outcomes
    python scripts/code_longspan/step_2_6_null_control.py --center code --update-step22

Runtime: ~5-15 minutes depending on which tests are run
"""

import os
import sys
import json
import time
import random
import argparse
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

# Project paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Namespace for file paths
NAMESPACE = os.environ.get('TEP_OUTPUT_NAMESPACE', 'code_longspan')


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_status(msg: str, level: str = "INFO"):
    """Print status message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "TITLE": "📊",
        "PROCESS": "🔄"
    }.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def safe_json_write(data: dict, filepath: Path, indent: int = 2):
    """Safely write JSON with NaN/Inf handling."""
    def convert_value(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_value(v) for v in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    clean_data = convert_value(data)
    with open(filepath, 'w') as f:
        json.dump(clean_data, f, indent=indent, default=str)


# ============================================================
# MODEL-FREE EVENT ANALYSIS WITH BLOCK PERMUTATION
# ============================================================

def block_permutation_test(coherences: np.ndarray, event_mask: np.ndarray,
                           n_permutations: int = 1000, block_size: int = 7,
                           rng: np.random.Generator = None) -> float:
    """
    Block permutation test that preserves temporal autocorrelation.
    
    Instead of permuting individual days (which breaks autocorrelation),
    we permute blocks of consecutive days. This gives valid p-values
    even when data has temporal structure.
    
    Args:
        coherences: Array of daily mean coherences
        event_mask: Boolean mask for event window days
        n_permutations: Number of permutations (default: 1000)
        block_size: Size of blocks to preserve autocorrelation (default: 7 days)
        rng: Random number generator
        
    Returns:
        Two-sided p-value from block permutation test
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(coherences)
    n_event = np.sum(event_mask)
    
    # Observed test statistic: difference in means
    observed_diff = np.abs(np.mean(coherences[event_mask]) - np.mean(coherences[~event_mask]))
    
    # Create blocks
    n_blocks = (n + block_size - 1) // block_size
    block_indices = [list(range(i * block_size, min((i + 1) * block_size, n))) 
                     for i in range(n_blocks)]
    
    # Count how many permutations have |diff| >= observed
    count_extreme = 0
    
    for _ in range(n_permutations):
        # Shuffle blocks (not individual elements)
        shuffled_blocks = rng.permutation(n_blocks)
        
        # Reconstruct shuffled indices
        shuffled_indices = []
        for block_idx in shuffled_blocks:
            shuffled_indices.extend(block_indices[block_idx])
        shuffled_indices = shuffled_indices[:n]  # Trim to original length
        
        # Apply shuffle to coherences
        perm_coherences = coherences[shuffled_indices]
        
        # Compute test statistic on permuted data
        # Use same event_mask positions (center of window)
        perm_diff = np.abs(np.mean(perm_coherences[event_mask]) - np.mean(perm_coherences[~event_mask]))
        
        if perm_diff >= observed_diff:
            count_extreme += 1
    
    # Two-sided p-value
    p_value = (count_extreme + 1) / (n_permutations + 1)  # +1 for observed
    return p_value


def analyze_event_model_free(event_data: pd.DataFrame, event_date: pd.Timestamp,
                              window_days: int, event_window: int = 15,
                              min_daily_pairs: int = 100,
                              n_permutations: int = 500,
                              block_size: int = 7,
                              rng: np.random.Generator = None) -> Dict:
    """
    Model-free analysis with BLOCK PERMUTATION test.
    
    Uses block permutation to account for temporal autocorrelation in daily
    coherence values. This gives properly calibrated p-values.
    
    Args:
        event_data: DataFrame with 'days_from_event' and 'coherence' columns
        event_date: The event date being analyzed
        window_days: Total half-window size (e.g., 120 days)
        event_window: Half-window around event to test (e.g., ±15 days)
        min_daily_pairs: Minimum pairs per day for inclusion
        n_permutations: Number of block permutations (default: 500)
        block_size: Block size in days to preserve autocorrelation (default: 7)
        rng: Random number generator for reproducibility
        
    Returns:
        Dict with analysis results including block-permutation p-value
    """
    # Bin by day and compute daily means
    daily_data = []
    for day in range(-window_days, window_days + 1):
        day_data = event_data[event_data['days_from_event'] == day]
        if len(day_data) >= min_daily_pairs:
            daily_data.append({
                'days_from_event': day,
                'mean_coherence': day_data['coherence'].mean(),
                'n_pairs': len(day_data)
            })
    
    if len(daily_data) < 20:
        return {'success': False, 'error': f'Insufficient daily data ({len(daily_data)} bins)'}
    
    days = np.array([d['days_from_event'] for d in daily_data])
    coherences = np.array([d['mean_coherence'] for d in daily_data])
    
    # Split into event window (±event_window days) and baseline (rest)
    event_mask = np.abs(days) <= event_window
    baseline_mask = np.abs(days) > event_window
    
    event_coherences = coherences[event_mask]
    baseline_coherences = coherences[baseline_mask]
    
    if len(event_coherences) < 5 or len(baseline_coherences) < 10:
        return {'success': False, 'error': 'Insufficient data in event or baseline window'}
    
    # Compute statistics
    event_mean = np.mean(event_coherences)
    baseline_mean = np.mean(baseline_coherences)
    baseline_std = np.std(baseline_coherences, ddof=1)
    
    # Effect size: difference in means normalized by baseline std
    effect_size = (event_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
    
    # Block permutation test (accounts for autocorrelation)
    p_value_block = block_permutation_test(
        coherences, event_mask, 
        n_permutations=n_permutations, 
        block_size=block_size,
        rng=rng
    )
    
    # Significance thresholds
    is_significant_p05 = p_value_block < 0.05
    is_significant_p01 = p_value_block < 0.01
    
    return {
        'success': True,
        'event_date': event_date.isoformat(),
        'n_event_days': int(np.sum(event_mask)),
        'n_baseline_days': int(np.sum(baseline_mask)),
        'model_free': {
            'event_mean': float(event_mean),
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'effect_size': float(effect_size),
            'p_value_block': float(p_value_block),
            'n_permutations': n_permutations,
            'block_size_days': block_size,
            'is_significant_p05': bool(is_significant_p05),
            'is_significant_p01': bool(is_significant_p01)
        }
    }


# ============================================================
# NULL EVENT CONTROL TEST (EFFECT SIZE COMPARISON)
# ============================================================

def run_null_event_control_test(complete_df: pd.DataFrame, 
                                 n_null_events: int = 156,
                                 window_days: int = 120,
                                 event_window: int = 15,
                                 min_pairs_per_day: int = 100,
                                 random_seed: int = 42) -> Dict:
    """
    NULL CONTROL TEST: Compare effect size DISTRIBUTIONS between planetary and null events.
    
    CORRECT METHODOLOGY (matching Step 3.2):
    - Don't count p-value detections (inflated by autocorrelation)
    - Instead, compute effect sizes for both planetary and null events
    - Test if planetary event effect sizes are SIGNIFICANTLY LARGER than null
    
    This is a critical falsification test. If planetary events have significantly
    larger effect sizes than random dates, the signal is specific to alignments.
    
    Args:
        complete_df: GPS pair dataset with 'date' and 'coherence' columns
        n_null_events: Number of random dates to test (default: 156)
        window_days: Total half-window size in days (default: 120)
        event_window: Half-window around event for comparison (default: 15)
        min_pairs_per_day: Minimum pairs required per day bin
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with effect size comparison between planetary and null events
    """
    print_status("="*70, "TITLE")
    print_status("NULL EVENT CONTROL TEST (EFFECT SIZE COMPARISON)", "TITLE")
    print_status("="*70, "TITLE")
    print_status(f"Testing {n_null_events} random dates vs planetary events", "INFO")
    print_status(f"Total window: ±{window_days} days", "INFO")
    print_status(f"Event window: ±{event_window} days", "INFO")
    print_status(f"Method: Compare effect size distributions (Mann-Whitney U)", "INFO")
    
    # Set seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    start_time = time.time()
    
    # ============================================================
    # STEP 1: Load planetary event effect sizes from Step 2.2
    # ============================================================
    print_status("Loading planetary event results from Step 2.2...", "PROCESS")
    
    step22_file = ROOT / "results/outputs" / NAMESPACE / "step_2_2_geospatial_temporal_analysis_code.json"
    planetary_effect_sizes = []
    planetary_events_info = []
    
    if step22_file.exists():
        try:
            with open(step22_file, 'r') as f:
                step22_results = json.load(f)
            
            # Extract effect sizes from Jupiter and Saturn opposition analyses
            # Data is in 'best_window_event_results' as dictionaries
            for analysis_key in ['jupiter_opposition_analysis', 'saturn_opposition_analysis']:
                analysis = step22_results.get(analysis_key, {})
                event_results = analysis.get('best_window_event_results', {})
                
                # event_results is a dict like {'Jupiter_Opposition_2000': {...}, ...}
                if isinstance(event_results, dict):
                    for event_name, event_data in event_results.items():
                        if not isinstance(event_data, dict):
                            continue
                        gauss = event_data.get('gaussian_fit', {})
                        if gauss and isinstance(gauss, dict):
                            # Use amplitude_snr as effect size (amplitude / baseline_std)
                            snr = gauss.get('amplitude_snr', 0)
                            if snr and snr > 0:
                                planetary_effect_sizes.append(abs(snr))
                                planetary_events_info.append({
                                    'event': event_name,
                                    'effect_size': abs(snr),
                                    'sigma': gauss.get('sigma_level', 0)
                                })
            
            print_status(f"Loaded {len(planetary_effect_sizes)} planetary event effect sizes", "SUCCESS")
            
        except Exception as e:
            print_status(f"Could not load Step 2.2 results: {e}", "WARNING")
    else:
        print_status(f"Step 2.2 results not found: {step22_file}", "WARNING")
    
    # ============================================================
    # STEP 2: Compute effect sizes for NULL events (random dates)
    # ============================================================
    print_status("Computing effect sizes for null (random) events...", "PROCESS")
    
    # Get date range from data
    data_start = complete_df['date'].min()
    data_end = complete_df['date'].max()
    
    # Exclude edge regions
    valid_start = data_start + pd.Timedelta(days=window_days + 30)
    valid_end = data_end - pd.Timedelta(days=window_days + 30)
    valid_range_days = (valid_end - valid_start).days
    
    print_status(f"Data range: {data_start.date()} to {data_end.date()}", "INFO")
    print_status(f"Valid test range: {valid_start.date()} to {valid_end.date()} ({valid_range_days} days)", "INFO")
    
    if valid_range_days < 100:
        return {'success': False, 'error': 'Insufficient date range for testing'}
    
    # Generate random dates
    random_dates = []
    for _ in range(n_null_events):
        random_offset = random.randint(0, valid_range_days)
        random_date = valid_start + pd.Timedelta(days=random_offset)
        random_dates.append(random_date)
    
    print_status(f"Generated {len(random_dates)} random test dates", "INFO")
    
    # Analyze each null event - compute effect sizes only (no p-values)
    null_effect_sizes = []
    null_results = []
    
    for i, event_date in enumerate(random_dates):
        window_start = event_date - pd.Timedelta(days=window_days)
        window_end = event_date + pd.Timedelta(days=window_days)
        
        window_data = complete_df[
            (complete_df['date'] >= window_start) & 
            (complete_df['date'] <= window_end)
        ].copy()
        
        if len(window_data) < min_pairs_per_day * 10:
            continue
        
        window_data['days_from_event'] = (window_data['date'] - event_date).dt.days
        
        # Compute effect size directly (no permutation test needed)
        result = analyze_event_model_free(
            window_data, event_date, window_days, event_window, min_pairs_per_day,
            n_permutations=10, block_size=7, rng=np.random.default_rng(random_seed + i)
        )
        
        if result.get('success'):
            mf = result.get('model_free', {})
            effect_size = abs(mf.get('effect_size', 0))
            null_effect_sizes.append(effect_size)
            null_results.append(result)
        
        # Progress update
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_null_events - i - 1) / rate if rate > 0 else 0
            print_status(f"   Processed {i+1}/{n_null_events} events ({remaining:.0f}s remaining)...", "PROCESS")
    
    elapsed_total = time.time() - start_time
    n_analyzed = len(null_effect_sizes)
    
    # ============================================================
    # STEP 3: Compare effect size DISTRIBUTIONS
    # ============================================================
    print_status("\n" + "-"*60, "INFO")
    print_status("EFFECT SIZE DISTRIBUTION COMPARISON", "TITLE")
    print_status("-"*60, "INFO")
    
    # Statistics
    null_mean = np.mean(null_effect_sizes) if null_effect_sizes else 0
    null_std = np.std(null_effect_sizes) if null_effect_sizes else 0
    null_median = np.median(null_effect_sizes) if null_effect_sizes else 0
    
    print_status(f"Null events analyzed: {n_analyzed}/{n_null_events}", "INFO")
    print_status(f"Analysis time: {elapsed_total:.1f} seconds", "INFO")
    print_status(f"", "INFO")
    print_status(f"NULL effect sizes:      mean={null_mean:.4f}, std={null_std:.4f}, median={null_median:.4f}", "INFO")
    
    # Compare with planetary if available
    test_passes = False
    mw_pvalue = 1.0
    mw_statistic = 0
    planetary_mean = 0
    planetary_std = 0
    planetary_median = 0
    effect_size_diff = 0
    
    if planetary_effect_sizes and null_effect_sizes:
        planetary_mean = np.mean(planetary_effect_sizes)
        planetary_std = np.std(planetary_effect_sizes)
        planetary_median = np.median(planetary_effect_sizes)
        
        print_status(f"PLANETARY effect sizes: mean={planetary_mean:.4f}, std={planetary_std:.4f}, median={planetary_median:.4f}", "INFO")
        print_status(f"", "INFO")
        
        # Mann-Whitney U test: Are planetary effect sizes > null effect sizes?
        try:
            mw_statistic, mw_pvalue = stats.mannwhitneyu(
                planetary_effect_sizes, null_effect_sizes,
                alternative='greater'  # Test if planetary > null
            )
            
            # Effect size (Cohen's d equivalent for Mann-Whitney)
            pooled_std = np.sqrt((planetary_std**2 + null_std**2) / 2)
            effect_size_diff = (planetary_mean - null_mean) / pooled_std if pooled_std > 0 else 0
            
            print_status(f"Mann-Whitney U test (planetary > null):", "INFO")
            print_status(f"  U statistic: {mw_statistic:.1f}", "INFO")
            print_status(f"  p-value: {mw_pvalue:.4f}", "INFO")
            print_status(f"  Effect size (Cohen's d): {effect_size_diff:.3f}", "INFO")
            
            # Test passes if planetary effect sizes are significantly larger
            test_passes = mw_pvalue < 0.05
            
        except Exception as e:
            print_status(f"Mann-Whitney test failed: {e}", "WARNING")
    else:
        print_status("Cannot compare: missing planetary event data", "WARNING")
    
    # ============================================================
    # STEP 4: Assessment
    # ============================================================
    print_status(f"", "INFO")
    
    if test_passes:
        print_status("RESULT: PASS ✓", "SUCCESS")
        print_status(f"Planetary events have significantly larger effect sizes than random dates", "SUCCESS")
        print_status(f"(Mann-Whitney p={mw_pvalue:.4f}, effect size={effect_size_diff:.3f})", "SUCCESS")
        print_status("This validates that planetary alignments produce specific coherence changes.", "SUCCESS")
    elif not planetary_effect_sizes:
        print_status("RESULT: INCOMPLETE", "WARNING")
        print_status("No planetary event data available for comparison.", "WARNING")
        print_status("Run Step 2.2 with planetary event analysis first.", "WARNING")
    else:
        print_status("RESULT: CONCERN ⚠️", "WARNING")
        print_status(f"Planetary effect sizes not significantly larger than null (p={mw_pvalue:.4f})", "WARNING")
        print_status("Planetary event detections may not be specific to alignments.", "WARNING")
    
    return {
        'success': True,
        'test_type': 'null_event_control_effect_size_comparison',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_null_events': n_null_events,
            'n_analyzed': n_analyzed,
            'n_planetary_events': len(planetary_effect_sizes),
            'window_days': window_days,
            'event_window': event_window,
            'min_pairs_per_day': min_pairs_per_day,
            'random_seed': random_seed,
            'method': 'Effect size distribution comparison (Mann-Whitney U)'
        },
        'null_distribution': {
            'mean_effect_size': float(null_mean),
            'std_effect_size': float(null_std),
            'median_effect_size': float(null_median),
            'n_samples': n_analyzed
        },
        'planetary_distribution': {
            'mean_effect_size': float(planetary_mean),
            'std_effect_size': float(planetary_std),
            'median_effect_size': float(planetary_median),
            'n_samples': len(planetary_effect_sizes)
        },
        'comparison': {
            'mann_whitney_u': float(mw_statistic),
            'p_value': float(mw_pvalue),
            'cohens_d': float(effect_size_diff),
            'planetary_significantly_larger': bool(test_passes)
        },
        'assessment': {
            'test_passes': test_passes,
            'interpretation': 'PASS' if test_passes else 'CONCERN',
            'conclusion': (
                f"Planetary events show significantly larger effect sizes (p={mw_pvalue:.4f})"
                if test_passes else
                f"No significant difference in effect sizes (p={mw_pvalue:.4f})"
            )
        },
        'execution_time_seconds': elapsed_total
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_geospatial_data(center: str, sample_rate: int = 1) -> pd.DataFrame:
    """
    Load Step 2.1 geospatial data with incremental concat to avoid memory spike.
    
    Args:
        center: Analysis center name
        sample_rate: Keep every Nth row (default: 1 = keep all data)
    """
    geospatial_file = ROOT / "data/processed" / NAMESPACE / f"step_2_1_geospatial_{center}.csv"
    
    if not geospatial_file.exists():
        raise FileNotFoundError(f"Step 2.1 data not found: {geospatial_file}")
    
    file_size_gb = geospatial_file.stat().st_size / (1024**3)
    print_status(f"Loading Step 2.1 data: {geospatial_file.name} ({file_size_gb:.2f} GB)", "PROCESS")
    
    # For large files, use incremental concat to avoid memory spike
    if file_size_gb > 5.0:
        if sample_rate > 1:
            print_status(f"Sampling 1/{sample_rate} of rows...", "INFO")
        else:
            print_status("Loading with incremental concat to manage memory...", "INFO")
        
        chunk_size = 2_000_000  # Smaller chunks
        total_rows = 0
        kept_rows = 0
        chunk_num = 0
        chunks = []  # Collect chunks, concat once at end (O(n) vs O(n²))
        
        for chunk in pd.read_csv(geospatial_file, chunksize=chunk_size, parse_dates=['date']):
            chunk_num += 1
            total_rows += len(chunk)
            
            # Optional sampling
            if sample_rate > 1:
                chunk = chunk.iloc[::sample_rate].copy()
            
            if 'plateau_phase' in chunk.columns and 'coherence' not in chunk.columns:
                chunk['coherence'] = np.cos(chunk['plateau_phase'])
            
            # Filter this chunk
            chunk = chunk.dropna(subset=['date', 'coherence'])
            chunk = chunk[~np.isnan(chunk['coherence']) & ~np.isinf(chunk['coherence'])]
            
            # Append to list (O(1)) instead of concat (O(n))
            chunks.append(chunk)
            kept_rows += len(chunk)
            
            # Progress every 10 chunks
            if chunk_num % 10 == 0:
                print_status(f"   Loaded {total_rows:,} rows, keeping {kept_rows:,}...", "PROCESS")
        
        # Single concat at end (O(n) total vs O(n²) with incremental concat)
        print_status(f"Concatenating {len(chunks)} chunks...", "PROCESS")
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        print_status(f"Loaded {len(df):,} rows from {total_rows:,} total", "SUCCESS")
        # df is now ready for common filtering
    else:
        df = pd.read_csv(geospatial_file, parse_dates=['date'])
        if 'plateau_phase' in df.columns and 'coherence' not in df.columns:
            df['coherence'] = np.cos(df['plateau_phase'])
    
    # Basic filtering (Common to both paths)
    df = df.dropna(subset=['date', 'coherence'])
    df = df[(df['coherence'] >= -1.0) & (df['coherence'] <= 1.0)]
    
    # Filter 3: Min distance 500km (CRITICAL: Matches Step 2.2 methodology)
    # Short baselines (<500km) are dominated by local noise/troposphere
    n_before_dist = len(df)
    df = df[df['dist_km'] >= 500.0]
    print_status(f"Applied 500km min distance filter: {n_before_dist:,} -> {len(df):,} pairs", "INFO")
    
    # Skip environmental regression for now to avoid OOM and because files are missing
    # if 'coherence_resid' in df.columns: ...
    
    print_status(f"Loaded {len(df):,} valid pairs", "SUCCESS")
    return df


# ============================================================
# TIDAL SCALING TEST
# ============================================================

def run_tidal_scaling_test(center: str) -> Dict:
    """
    Test correlation of planetary event amplitudes with TIDAL potential (M/r³)
    in addition to gravitational potential (M/r²).
    
    Tidal forces are physically more relevant for differential timing effects
    across a distributed network than simple gravitational potential.
    
    Args:
        center: Analysis center name
        
    Returns:
        Dict with scaling test results for both M/r² and M/r³
    """
    from scipy.stats import pearsonr, spearmanr
    
    print_status("="*70, "TITLE")
    print_status("TIDAL SCALING TEST (M/r³ vs M/r²)", "TITLE")
    print_status("="*70, "TITLE")
    
    # Load Step 2.2 results to get planetary event data
    results_file = ROOT / "results/outputs" / NAMESPACE / f"step_2_2_geospatial_temporal_analysis_{center}.json"
    
    if not results_file.exists():
        return {'success': False, 'error': f'Step 2.2 results not found: {results_file}'}
    
    with open(results_file, 'r') as f:
        step22_results = json.load(f)
    
    # Planet physical parameters (mass in kg, semi-major axis in AU for distance estimates)
    planet_params = {
        'Mercury': {'mass_kg': 3.301e23, 'typical_dist_au': 1.0},
        'Venus': {'mass_kg': 4.867e24, 'typical_dist_au': 0.72},
        'Mars': {'mass_kg': 6.417e23, 'typical_dist_au': 0.52},
        'Jupiter': {'mass_kg': 1.898e27, 'typical_dist_au': 4.2},
        'Saturn': {'mass_kg': 5.683e26, 'typical_dist_au': 8.5}
    }
    AU_TO_M = 1.496e11
    
    # Extract planetary event results
    comprehensive = step22_results.get('comprehensive_report', {})
    planetary_events = comprehensive.get('planetary_events', {})
    
    if not planetary_events:
        return {'success': False, 'error': 'No planetary event data found in Step 2.2 results'}
    
    # Collect significant events with amplitudes
    events_data = []
    
    for planet_name, planet_data in planetary_events.items():
        if planet_name not in planet_params:
            continue
            
        params = planet_params[planet_name]
        mass = params['mass_kg']
        dist_m = params['typical_dist_au'] * AU_TO_M
        
        sig_detections = planet_data.get('significant_detections', [])
        for det in sig_detections:
            amp = det.get('amplitude_pct', det.get('modulation_depth_percent', 0))
            sigma = det.get('sigma_level', 0)
            
            if amp > 0 and sigma >= 2.0:
                events_data.append({
                    'planet': planet_name,
                    'amplitude': amp,
                    'sigma_level': sigma,
                    'mass_kg': mass,
                    'dist_m': dist_m,
                    'grav_potential': mass / dist_m**2,  # M/r²
                    'tidal_potential': mass / dist_m**3   # M/r³
                })
    
    if len(events_data) < 5:
        return {
            'success': False, 
            'error': f'Insufficient events for scaling analysis ({len(events_data)} found, need ≥5)'
        }
    
    print_status(f"Analyzing {len(events_data)} significant planetary events", "INFO")
    
    # Extract arrays
    amplitudes = np.array([e['amplitude'] for e in events_data])
    grav_params = np.array([e['grav_potential'] for e in events_data])
    tidal_params = np.array([e['tidal_potential'] for e in events_data])
    
    # Normalize for correlation (log scale often better for power laws)
    log_amp = np.log10(amplitudes + 1e-10)
    log_grav = np.log10(grav_params + 1e-10)
    log_tidal = np.log10(tidal_params + 1e-10)
    
    # Pearson correlations (linear)
    r_grav_lin, p_grav_lin = pearsonr(amplitudes, grav_params)
    r_tidal_lin, p_tidal_lin = pearsonr(amplitudes, tidal_params)
    
    # Pearson correlations (log-log for power law)
    r_grav_log, p_grav_log = pearsonr(log_amp, log_grav)
    r_tidal_log, p_tidal_log = pearsonr(log_amp, log_tidal)
    
    # Spearman rank correlations (non-parametric)
    rho_grav, p_grav_spear = spearmanr(amplitudes, grav_params)
    rho_tidal, p_tidal_spear = spearmanr(amplitudes, tidal_params)
    
    # Determine best model
    best_model = 'neither'
    if p_tidal_lin < 0.05 or p_tidal_log < 0.05:
        best_model = 'tidal (M/r³)'
    elif p_grav_lin < 0.05 or p_grav_log < 0.05:
        best_model = 'gravitational (M/r²)'
    
    neither_significant = (p_grav_lin > 0.05 and p_grav_log > 0.05 and 
                          p_tidal_lin > 0.05 and p_tidal_log > 0.05)
    
    # Print results
    print_status(f"", "INFO")
    print_status("GRAVITATIONAL SCALING (M/r²):", "INFO")
    print_status(f"   Linear:   r = {r_grav_lin:.3f}, p = {p_grav_lin:.4f}", "INFO")
    print_status(f"   Log-log:  r = {r_grav_log:.3f}, p = {p_grav_log:.4f}", "INFO")
    print_status(f"   Spearman: ρ = {rho_grav:.3f}, p = {p_grav_spear:.4f}", "INFO")
    
    print_status(f"", "INFO")
    print_status("TIDAL SCALING (M/r³):", "INFO")
    print_status(f"   Linear:   r = {r_tidal_lin:.3f}, p = {p_tidal_lin:.4f}", "INFO")
    print_status(f"   Log-log:  r = {r_tidal_log:.3f}, p = {p_tidal_log:.4f}", "INFO")
    print_status(f"   Spearman: ρ = {rho_tidal:.3f}, p = {p_tidal_spear:.4f}", "INFO")
    
    print_status(f"", "INFO")
    if neither_significant:
        print_status("RESULT: Neither M/r² nor M/r³ shows significant scaling", "INFO")
        print_status("This is consistent with processing filter suppression of", "INFO")
        print_status("classical gravitational effects (see §2.1.3, §4.1.3)", "INFO")
    else:
        print_status(f"RESULT: Best model = {best_model}", "INFO")
    
    return {
        'success': True,
        'n_events': len(events_data),
        'gravitational_scaling': {
            'linear': {'r': float(r_grav_lin), 'p': float(p_grav_lin)},
            'log_log': {'r': float(r_grav_log), 'p': float(p_grav_log)},
            'spearman': {'rho': float(rho_grav), 'p': float(p_grav_spear)}
        },
        'tidal_scaling': {
            'linear': {'r': float(r_tidal_lin), 'p': float(p_tidal_lin)},
            'log_log': {'r': float(r_tidal_log), 'p': float(p_tidal_log)},
            'spearman': {'rho': float(rho_tidal), 'p': float(p_tidal_spear)}
        },
        'best_model': best_model,
        'neither_significant': neither_significant,
        'interpretation': (
            'No significant scaling detected - consistent with processing filter suppression'
            if neither_significant else
            f'{best_model} scaling detected'
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Step 2.6: Validation Tests (Null Control, Tidal Scaling)")
    parser.add_argument("--center", default="code", help="Analysis center (default: code)")
    parser.add_argument("--n-events", type=int, default=156, help="Number of null events (default: 156)")
    parser.add_argument("--window", type=int, default=120, help="Window size in days (default: 120)")
    parser.add_argument("--event-window", type=int, default=15, help="Event window half-size in days for comparison (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sample 1/N rows from large files (default: 1 = all data)")
    parser.add_argument("--skip-null", action="store_true", help="Skip null event control test")
    parser.add_argument("--skip-tidal", action="store_true", help="Skip tidal scaling test")
    parser.add_argument("--update-step22", action="store_true", help="Update Step 2.2 results file")
    args = parser.parse_args()
    
    print_status("="*70, "TITLE")
    print_status("STEP 2.6: VALIDATION TESTS", "TITLE")
    print_status("="*70, "TITLE")
    print_status(f"Analysis center: {args.center.upper()}", "INFO")
    print_status(f"Null events: {args.n_events}", "INFO")
    print_status(f"Window: ±{args.window} days", "INFO")
    print_status(f"Random seed: {args.seed}", "INFO")
    
    start_time = time.time()
    
    # Initialize combined results
    all_results = {
        'analysis_center': args.center.upper(),
        'timestamp': datetime.now().isoformat(),
        'tests_run': []
    }
    
    complete_df = None
    
    # ============================================================
    # TEST 1: Tidal Scaling (doesn't need geospatial data)
    # ============================================================
    if not args.skip_tidal:
        print_status(f"", "INFO")
        try:
            tidal_results = run_tidal_scaling_test(args.center)
            all_results['tidal_scaling'] = tidal_results
            all_results['tests_run'].append('tidal_scaling')
        except Exception as e:
            print_status(f"Tidal scaling test failed: {e}", "WARNING")
            all_results['tidal_scaling'] = {'success': False, 'error': str(e)}
    
    # ============================================================
    # TEST 2: Null Event Control (needs geospatial data)
    # ============================================================
    if not args.skip_null:
        print_status(f"", "INFO")
        # Load data if not already loaded
        if complete_df is None:
            try:
                complete_df = load_geospatial_data(args.center, sample_rate=args.sample_rate)
            except FileNotFoundError as e:
                print_status(str(e), "ERROR")
                return 1
        
        try:
            null_results = run_null_event_control_test(
                complete_df,
                n_null_events=args.n_events,
                window_days=args.window,
                event_window=args.event_window,
                random_seed=args.seed
            )
            all_results['null_event_control'] = null_results
            all_results['tests_run'].append('null_event_control')
        except Exception as e:
            print_status(f"Null event control test failed: {e}", "WARNING")
            all_results['null_event_control'] = {'success': False, 'error': str(e)}
    
    # Clean up
    if complete_df is not None:
        del complete_df
        gc.collect()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print_status(f"", "INFO")
    print_status("="*70, "TITLE")
    print_status("VALIDATION TESTS SUMMARY", "TITLE")
    print_status("="*70, "TITLE")
    
    n_pass = 0
    n_total = 0
    
    if 'tidal_scaling' in all_results and all_results['tidal_scaling'].get('success'):
        n_total += 1
        # "Neither significant" is actually expected/passing for this test
        if all_results['tidal_scaling'].get('neither_significant', False):
            n_pass += 1
            print_status("✓ Tidal Scaling: PASS (no M/r² or M/r³ scaling - expected)", "SUCCESS")
        else:
            print_status(f"⚠ Tidal Scaling: {all_results['tidal_scaling'].get('best_model', 'unknown')}", "INFO")
    
    if 'null_event_control' in all_results and all_results['null_event_control'].get('success'):
        n_total += 1
        if all_results['null_event_control'].get('assessment', {}).get('test_passes', False):
            n_pass += 1
            p_val = all_results['null_event_control'].get('comparison', {}).get('p_value', 1.0)
            cohens_d = all_results['null_event_control'].get('comparison', {}).get('cohens_d', 0)
            print_status(f"✓ Null Event Control: PASS (planetary > null, p={p_val:.4f}, d={cohens_d:.2f})", "SUCCESS")
        else:
            p_val = all_results['null_event_control'].get('comparison', {}).get('p_value', 1.0)
            print_status(f"⚠ Null Event Control: CONCERN (p={p_val:.4f}, planetary not > null)", "WARNING")
    
    print_status(f"", "INFO")
    print_status(f"Tests passed: {n_pass}/{n_total}", "INFO")
    
    all_results['summary'] = {
        'tests_passed': n_pass,
        'tests_total': n_total,
        'all_passed': n_pass == n_total
    }
    
    # Save results
    output_dir = ROOT / "results/outputs" / NAMESPACE
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"step_2_6_validation_tests_{args.center}.json"
    
    safe_json_write(all_results, output_file)
    print_status(f"Results saved: {output_file}", "SUCCESS")
    
    # Optionally update Step 2.2 results
    if args.update_step22:
        step22_file = output_dir / f"step_2_2_geospatial_temporal_analysis_{args.center}.json"
        if step22_file.exists():
            try:
                with open(step22_file, 'r') as f:
                    step22_results = json.load(f)
                step22_results['validation_tests'] = all_results
                safe_json_write(step22_results, step22_file)
                print_status(f"Updated Step 2.2 results: {step22_file}", "SUCCESS")
            except Exception as e:
                print_status(f"Failed to update Step 2.2 results: {e}", "WARNING")
        else:
            print_status(f"Step 2.2 results not found: {step22_file}", "WARNING")
    
    total_time = time.time() - start_time
    print_status(f"", "INFO")
    print_status(f"Step 2.6 completed in {total_time:.1f} seconds", "SUCCESS")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
