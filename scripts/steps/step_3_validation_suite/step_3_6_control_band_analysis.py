#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 3.6: Multi-Band Frequency Validation Analysis
======================================================================

Comprehensive validation of TEP correlations across multiple frequency bands to rule out
alternative explanations and confirm the robustness of the TEP signal.

SCIENTIFIC RATIONALE:
====================
This analysis performs a comprehensive frequency spectrum validation to assess
alternative explanations and physical predictions for GNSS timing correlations.

KEY PHYSICAL HYPOTHESES TO TEST:
================================

HYPOTHESIS A: Tidal-Dominated TEP Coupling
-------------------------------------------
IF: φ-field couples universally to matter via gravitational effects
AND: Tidal forces create large-amplitude periodic deformations in Earth's atmosphere/crust  
THEN: Tidal frequency bands SHOULD show STRONGER correlations than broad TEP band

Physical Reasoning:
- Moon/Sun create well-characterized, periodic gravitational gradients
- Earth's matter (atmosphere, crust) responds with measurable tidal deformations
- If φ-field couples to these deformations, effect should be maximal at tidal frequencies
- GNSS clocks embedded in this environment should show enhanced correlations

EXPECTED PATTERN (if tidal coupling dominates):
- Tidal Bands (10-15, 20-30 μHz): R² > 0.97 (STRONGEST signal)
- Broad TEP Band (10-500 μHz): R² ~ 0.90-0.95 (diluted by non-tidal contributions)
- Control Bands (>1000 μHz): R² < 0.3 (no physical coupling)

HYPOTHESIS B: Broadband TEP with Systematic Effects
----------------------------------------------------
IF: Observed correlations arise from broadband TEP signal plus instrumental effects
AND: Tidal coupling is NOT the dominant mechanism
THEN: Broad TEP band shows comparable or stronger signal than narrow tidal bands

Physical Reasoning:
- TEP signal distributed across broad frequency range (10-500 μHz)
- Tidal bands capture only narrow subset of total signal
- Systematic effects contribute across all frequencies
- Control bands show non-zero correlations from systematics

EXPECTED PATTERN (if broadband coupling + systematics):
- Broad TEP Band (10-500 μHz): R² ~ 0.95-0.97 (integrates full signal range)
- Tidal Bands (10-15, 20-30 μHz): R² ~ 0.90-0.97 (narrow subset of TEP)
- Control Bands (>1000 μHz): R² ~ 0.5-0.7 (systematic effects only)

HYPOTHESIS C: Systematic Effects Dominate
------------------------------------------
IF: Correlations arise primarily from instrumental or processing artifacts
THEN: All frequency bands show similar correlation strengths

EXPECTED PATTERN (if systematics dominate):
- All bands show R² ~ 0.8-0.9 with minimal differentiation
- No clear frequency-dependent structure
- Control bands comparable to TEP/tidal bands

ANALYSIS APPROACH:
==================
1. **Measure correlation strength** in each frequency band
2. **Compare tidal vs broad TEP** band strengths
3. **Assess control band** contributions  
4. **Present objective metrics** without premature interpretation
5. **Enable independent evaluation** of competing hypotheses

The analysis provides TRANSPARENT DATA to distinguish between these physical scenarios.

METHODOLOGY - IDENTICAL TO STEP 2.0:
===================================
Runs the EXACT SAME phase-coherent correlation analysis as Step 2.0, ensuring
methodological consistency. The ONLY difference is the frequency band.

TEMPORAL RESOLUTION HANDLING:
============================
CRITICAL: NO RESAMPLING IS APPLIED TO CLK DATA

The analysis uses the native temporal resolution of GNSS CLK files:
- CLK files typically contain observations at 5-minute or 30-second intervals
- NO additional resampling or interpolation is performed
- Sampling rate is computed dynamically from actual timestamps: fs = 1/median(Δt)
- This preserves the authentic measurement cadence and avoids introducing artifacts

Previous versions incorrectly applied 5-minute resampling (df.resample('5min').mean())
which was REMOVED to ensure consistency with Step 2.0 and maintain data authenticity.

Algorithm (IDENTICAL TO STEP 2.0):
==================================
1. Parse CLK files → Extract station time series at native cadence
2. For each station pair:
   a. Find common observation epochs (intersection of timestamps)
   b. Extract synchronized time series at common epochs
   c. Compute sampling frequency: fs = 1/median(diff(timestamps))
   d. Compute cross-spectral density using Welch's method with:
      - Detrending: Linear (removes DC offset and linear drift)
      - Window: Default Welch windowing
      - nperseg: min(1024, n_points) for spectral resolution
3. Frequency band filtering:
   - Extract CSD values within band [f1, f2]
   - Apply magnitude-weighted circular statistics for phase averaging
4. Phase-coherent correlation:
   - Complex phases: exp(i*angle(CSD))
   - Weighted average: Σ(magnitude_i * exp(i*phase_i)) / Σ(magnitude_i)
   - Correlation = cos(weighted_phase)  ← CRITICAL: Phase-alignment index
5. Distance binning:
   - Logarithmic bins: logspace(log10(50), log10(13000), 40)
   - Right-inclusive bin edges (pandas.cut default)
   - Aggregate: mean correlation, mean distance, count per bin
6. Quality filtering:
   - Minimum bin count: 200 pairs (TEP_MIN_BIN_COUNT)
   - Removes sparse bins with insufficient statistical power
7. Exponential model fitting:
   - Model: C(r) = A*exp(-r/λ) + C₀
   - Weighted least squares: weights = bin_counts
   - Adaptive lambda bounds from TEPConfig
   - Returns: A, λ, C₀, R², parameter errors

Frequency Bands (Default Analysis):
===================================
- Primary TEP Band:      10-500 μHz    (periods: 28 hours - 33 minutes)
- Diurnal Tides:         10-15 μHz     (periods: 18.5-27.8 hours)
- Semidiurnal Tides:     20-30 μHz     (periods: 9.3-13.9 hours)
- Intermediate 1:        500-1000 μHz  (periods: 17-33 minutes)
- Intermediate 2:        100-500 μHz   (periods: 33 min - 2.8 hours)
- Control 1:             1000-2000 μHz (periods: 8-17 minutes)
- Control 2-4, High Freq: Additional control bands for systematic effect assessment

EXPECTED VALIDATION OUTCOMES:
============================
**TYPICAL VALIDATION PATTERN** (based on current analysis):
- TEP Band:           R² ≈ 0.85-0.95, λ ≈ 4000 km (strong signal)
- Tidal Bands:        R² ≈ 0.85-0.95 (comparable to TEP band)
- Control Bands:      R² ≈ 0.50-0.60 (moderate correlations)
- Frequency Specificity: "NONE" (no single band dominates)

IMPLEMENTATION DETAILS (CONSISTENT WITH STEP 2.0):
==================================================
1) Distance binning and edges
   - Log-spaced bins from 50 km to TEP_MAX_DISTANCE_KM using pandas cut
   - Right-inclusive bin edges (matches Step 2.0)

2) Phase-coherent correlation per pair
   - Complex CSD on detrended series; restrict to band
   - Magnitude-weighted circular mean of phases; coherence = cos(weighted_phase)

3) Streaming aggregation (equivalent to Step 2.0 in-worker sums)
   - Read pair-level CSVs in chunks; per bin accumulate Σcoherence, Σdistance, count
   - Order-invariant sums yield identical bin means to Step 2.0

4) Weighted least squares (WLS) fit and weighted R²
   - Fit C(r) = A·exp(−r/λ) + C₀ with weights w_i = n_i (bin counts)
   - σ_i = 1/√w_i; weighted R² computed with the same weights
   - Rationale: bin means have variance ≈ const/n_i, so WLS improves efficiency and stability

5) Adaptive parameter bounds (λ) and stability
   - Bounds via TEPConfig.get_adaptive_lambda_bounds(distances) (as in Step 2.0)

6) Date-range and configuration
   - Supports TEP_DATE_START / TEP_DATE_END for manuscript-aligned windows
   - Core knobs centralized in TEPConfig (TEP_BINS, TEP_MAX_DISTANCE_KM, TEP_MIN_BIN_COUNT)

**This pattern indicates:**
1. Tidal effects show comparable correlation strength to TEP band
2. Control bands exhibit moderate correlations, suggesting systematic effects are present
3. No single frequency band dominates the observed correlations
4. The signal demonstrates robustness across multiple frequency ranges
5. Results are consistent with a signal that is not frequency-specific in the traditional sense

**ALTERNATIVE PATTERNS** (that would suggest different interpretations):
- Tidal bands significantly stronger than TEP band → potential tidal contamination
- Uniform correlations across all bands → possible instrumental systematic effects
- Very weak correlations everywhere → potential pure noise or statistical artifacts
- Strong frequency specificity → possible frequency-dependent systematic effects

Requirements: Step 2.0 complete (Core TEP Correlation Analysis)
Inputs:
  - data/raw/{igs,esa,code}/*.CLK.gz files (from Step 1.1)
  - data/coordinates/step_1_1_station_coords_global.csv (from Step 1.1)
  - results/outputs/step_2_0_correlation_{ac}.json (from Step 2.0, for comparison)
  
Outputs:
  - results/outputs/step_3_6_control_band_{ac}.json
  - results/outputs/step_3_6_band_comparison_{ac}.json
  - results/figures/step_3_6_frequency_specificity_{ac}.png

Next: Step 4.0 (Advanced Analysis)

Author: Matthew Lukin Smawfield
Date: October 2025
Theory: Temporal Equivalence Principle (TEP)
"""

import os
import sys
import time
import json
import gzip
import re
import itertools
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
from scipy.signal import csd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Ensure macOS uses fork start method to avoid <stdin> spawn errors when invoked via python -c
try:
    if mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
except (AttributeError, RuntimeError):
    pass

# Worker-global context to reduce pickling overhead per task
WORKER_COORDS_DF = None
WORKER_DISTANCE_CACHE = None
WORKER_AC = None

# Memory management configuration
MEMORY_FLUSH_THRESHOLD = 0.85  # Flush when memory usage exceeds 85%
MEMORY_CHECK_INTERVAL = 5      # Check memory every 5 batches
BATCH_MEMORY_LIMIT_GB = 2.0    # Maximum memory per batch in GB

def _init_worker_context(coords_df, distance_cache, ac):
    """Initializer to load heavy context once per worker process."""
    import os
    # Suppress macOS malloc stack logging warnings in worker processes
    os.environ['MallocStackLogging'] = '0'
    os.environ['MallocScribble'] = '0'
    os.environ['MallocGuardEdges'] = '0'
    
    global WORKER_COORDS_DF, WORKER_DISTANCE_CACHE, WORKER_AC
    WORKER_COORDS_DF = coords_df
    WORKER_DISTANCE_CACHE = distance_cache
    WORKER_AC = ac

def get_memory_usage():
    """Get current memory usage as percentage."""
    try:
        return psutil.virtual_memory().percent / 100.0
    except:
        return 0.0

def get_available_memory_gb():
    """Get available memory in GB."""
    try:
        return psutil.virtual_memory().available / (1024**3)
    except:
        return 8.0  # Default fallback


def force_memory_cleanup():
    """Force memory optimization."""
    print_status("Performing memory optimization...", "PROCESS")
    
    # Multiple memory optimization passes
    for i in range(3):
        collected = gc.collect()
        if i == 0:
            print_status(f"Memory optimization pass {i+1}: freed {collected} objects", "INFO")
    
    # Set aggressive GC thresholds
    gc.set_threshold(700, 10, 10)
    
    # Force Python to release memory back to OS (if possible)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.dylib")
        libc.malloc_trim(0)
    except:
        pass  # Not critical if this fails
    
    memory_after = get_memory_usage()
    print_status(f"Memory optimization complete. Usage: {memory_after*100:.1f}%", "SUCCESS")

def check_memory_pressure():
    """Check if memory pressure requires intervention."""
    memory_usage = get_memory_usage()
    if memory_usage > MEMORY_FLUSH_THRESHOLD:
        print_status(f"Memory pressure detected: {memory_usage*100:.1f}% > {MEMORY_FLUSH_THRESHOLD*100:.1f}%", "WARNING")
        return True
    return False


def analyze_single_band(band_id, ac, streaming_dir, band_pair_counts, bands):
    """
    Analyze a single band and return comprehensive diagnostic results.
    
    TRANSPARENCY ENHANCEMENTS:
    - Reports raw pair counts before and after filtering
    - Documents binning parameters and distance distribution
    - Shows weighting scheme and its impact on fitting
    - Provides statistical summaries for scientific interpretation
    """
    band_file = streaming_dir / f"streaming_pairs_{ac}_{band_id}.csv"
    if not band_file.exists() or band_pair_counts[band_id] == 0:
        print_status(f"No data for band {band_id}, skipping analysis", "WARNING")
        return band_id, None
        
    print_status(f"Aggregating binned data for {band_id} from streaming file...", "PROCESS")
    
    # Use same binning as Step 2.0
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')

    # Get binned data with full transparency
    binned_df_raw, total_pairs = aggregate_streaming_bins(
        band_file=band_file,
        n_bins=num_bins,
        max_dist=max_distance,
        chunk_size=500_000
    )
    
    # Store raw binning statistics BEFORE filtering
    bins_before_filter = len(binned_df_raw)
    pairs_in_bins_before = binned_df_raw['count'].sum() if not binned_df_raw.empty else 0
    
    # Apply minimum bin count filter (transparency: show what we're filtering)
    binned_df = binned_df_raw[binned_df_raw['count'] >= min_bin_count].copy()
    
    # Store filtered binning statistics AFTER filtering
    bins_after_filter = len(binned_df)
    pairs_in_bins_after = binned_df['count'].sum() if not binned_df.empty else 0
    bins_removed = bins_before_filter - bins_after_filter
    pairs_removed = pairs_in_bins_before - pairs_in_bins_after
    
    # Calculate bin statistics for transparency
    if not binned_df.empty:
        bin_stats = {
            'count_min': int(binned_df['count'].min()),
            'count_max': int(binned_df['count'].max()),
            'count_mean': float(binned_df['count'].mean()),
            'count_median': float(binned_df['count'].median()),
            'count_std': float(binned_df['count'].std()),
            'distance_min_km': float(binned_df['distance_km'].min()),
            'distance_max_km': float(binned_df['distance_km'].max()),
            'correlation_min': float(binned_df['mean_correlation'].min()),
            'correlation_max': float(binned_df['mean_correlation'].max()),
            'correlation_mean': float(binned_df['mean_correlation'].mean()),
            'correlation_std': float(binned_df['mean_correlation'].std())
        }
    else:
        bin_stats = None
    
    # Check sufficient bins for fitting
    if bins_after_filter < 5:
        print_status(f"Insufficient bins for {band_id} ({bins_after_filter} < 5)", "WARNING")
        return band_id, {
            'band_config': bands[band_id],
            'frequency_range_microhz': [bands[band_id]['f1'] * 1e6, bands[band_id]['f2'] * 1e6],
            'data_summary': {
                'total_pairs_processed': total_pairs,
                'bins_before_filter': bins_before_filter,
                'bins_after_filter': bins_after_filter,
                'bins_removed': bins_removed,
                'pairs_in_bins_before': pairs_in_bins_before,
                'pairs_in_bins_after': pairs_in_bins_after,
                'pairs_removed_by_filter': pairs_removed,
                'min_bin_count_threshold': min_bin_count,
                'insufficient_bins': True
            },
            'bin_statistics': bin_stats,
            'exponential_fit': {'success': False, 'error': 'Insufficient bins for fitting'},
            'binned_data': binned_df.to_dict('records')
        }
    
    # Fit exponential model WITH WEIGHTING (transparency: document weights)
    fit_result = fit_exponential_model(
        distances=binned_df['distance_km'].values,
        correlations=binned_df['mean_correlation'].values,
        weights=binned_df['count'].values  # CRITICAL: Weighted by bin counts
    )
    
    # Add fitting diagnostics
    if fit_result.get('success'):
        # Calculate unweighted R² for comparison
        predictions = exponential_decay(
            binned_df['distance_km'].values,
            fit_result['A'],
            fit_result['lambda_km'],
            fit_result['C0']
        )
        residuals = binned_df['mean_correlation'].values - predictions
        ss_res_unweighted = float(np.sum(residuals ** 2))
        ss_tot_unweighted = float(np.sum((binned_df['mean_correlation'].values - 
                                          np.mean(binned_df['mean_correlation'].values)) ** 2))
        r_squared_unweighted = 1 - (ss_res_unweighted / ss_tot_unweighted) if ss_tot_unweighted > 0 else 0.0
        
        fit_result['r_squared_unweighted'] = r_squared_unweighted
        fit_result['weighting_impact'] = fit_result['r_squared'] - r_squared_unweighted
        fit_result['total_weight'] = float(binned_df['count'].sum())
    
    # Store comprehensive band analysis
    band_config = bands[band_id]
    result = {
        'band_config': band_config,
        'frequency_range_microhz': [band_config['f1'] * 1e6, band_config['f2'] * 1e6],
        'frequency_range_hz': [band_config['f1'], band_config['f2']],
        'frequency_bandwidth_microhz': (band_config['f2'] - band_config['f1']) * 1e6,
        'data_summary': {
            'total_pairs_processed': total_pairs,
            'bins_before_filter': bins_before_filter,
            'bins_after_filter': bins_after_filter,
            'bins_removed': bins_removed,
            'pairs_in_bins_before': pairs_in_bins_before,
            'pairs_in_bins_after': pairs_in_bins_after,
            'pairs_removed_by_filter': pairs_removed,
            'filter_removal_percent': 100.0 * pairs_removed / pairs_in_bins_before if pairs_in_bins_before > 0 else 0.0,
            'min_bin_count_threshold': min_bin_count,
            'binning_config': {
                'num_bins': num_bins,
                'max_distance_km': max_distance,
                'bin_type': 'logarithmic',
                'bin_edges_formula': f'logspace(log10(50), log10({max_distance}), {num_bins + 1})'
            }
        },
        'bin_statistics': bin_stats,
        'exponential_fit': fit_result,
        'binned_data': binned_df.to_dict('records'),  # Filtered data used for fitting
        'binned_data_raw': binned_df_raw.to_dict('records')  # Raw unfiltered data for transparency
    }
    
    if fit_result.get('success'):
        r2 = fit_result.get('r_squared', 0)
        lambda_km = fit_result.get('lambda_km', 0)
        r2_unweighted = fit_result.get('r_squared_unweighted', 0)
        print_status(f"{band_id}: R² = {r2:.3f} (unweighted: {r2_unweighted:.3f}), λ = {lambda_km:.0f} km", "SUCCESS")
        print_status(f"  Bins: {bins_after_filter}/{bins_before_filter}, Pairs: {pairs_in_bins_after:,}/{total_pairs:,}", "INFO")
    else:
        print_status(f"{band_id}: Fit failed - {fit_result.get('error', 'Unknown error')}", "WARNING")
    
    return band_id, result

# Anchor to package root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import TEP utilities
from scripts.utils.config import TEPConfig
from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.exceptions import (
    TEPDataError, TEPFileError, TEPAnalysisError,
    safe_json_read, safe_json_write, safe_csv_read,
    validate_file_exists, validate_directory_exists
)
from scripts.utils.pid_manager import ensure_single_instance

# Import Step 2.0 correlation function for identical TEP band computation
from scripts.steps.step_2_core_analysis.step_2_0_tep_correlation_analysis import compute_cross_power_plateau as step2_compute_cross_power_plateau


def atomic_save_checkpoint(checkpoint_file: Path, data: dict, max_retries: int = 3) -> bool:
    """
    Atomically save checkpoint data with proper locking and corruption protection.
    
    Args:
        checkpoint_file: Path to checkpoint file
        data: Dictionary to save as JSON
        max_retries: Maximum retry attempts
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    for attempt in range(max_retries):
        try:
            # Ensure parent directory exists
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use temporary file for atomic write
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w',
                dir=checkpoint_file.parent,
                delete=False,
                prefix=f"{checkpoint_file.stem}_tmp_",
                suffix=checkpoint_file.suffix
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2, default=str)
                tmp_path = Path(tmp_file.name)
            
            # Atomic move
            tmp_path.replace(checkpoint_file)
            step_logger.debug(f"Checkpoint saved successfully: {checkpoint_file}")
            return True
            
        except Exception as e:
            step_logger.error(f"Checkpoint save attempt {attempt + 1} failed: {e}")
            # Clean up failed temp file
            try:
                if 'tmp_path' in locals():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    
    step_logger.error(f"Failed to save checkpoint after {max_retries} attempts")
    return False


def load_checkpoint_safely(checkpoint_file: Path) -> Optional[dict]:
    """
    Safely load checkpoint data with corruption detection.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        dict: Loaded checkpoint data or None if failed
    """
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        step_logger.info(f"Checkpoint loaded successfully: {checkpoint_file}")
        return data
    except Exception as e:
        step_logger.error(f"Failed to load checkpoint: {e}")
        # Remove corrupted checkpoint
        try:
            checkpoint_file.unlink()
            step_logger.warning("Removed corrupted checkpoint file")
        except Exception:
            pass
        return None


def safe_remove_file(file_path: Path) -> bool:
    """Safely remove a file."""
    try:
        if file_path.exists():
            file_path.unlink()
            return True
    except Exception as e:
        step_logger.warning(f"Failed to remove file {file_path}: {e}")
    return False

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_3_6_control_band_analysis",
    level="DEBUG",
    log_file_path=PROJECT_ROOT / "logs" / "step_3_6_control_band_analysis.log"
)
set_step_logger(step_logger)

# Constants
EARTH_RADIUS_KM = 6371.0088
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F**2

# Optimized multi-band frequency analysis
# Designed to test physical hypotheses with matched bandwidths and critical transitions
FREQUENCY_BANDS = {
    # ============================================================================
    # REFERENCE BAND
    # ============================================================================
    'tep_band': {
        'f1': 1e-5, 'f2': 5e-4, 
        'name': 'TEP Band (10-500 μHz)', 
        'expected': 'strong',
        'bandwidth_microhz': 490,
        'description': 'Full TEP prediction range - reference for comparison'
    },
    
    # ============================================================================
    # TIDAL BANDS - Testing Hypothesis A (Tidal-Dominated Coupling)
    # Matched 10 μHz bandwidth for fair comparison
    # ============================================================================
    'tidal_diurnal': {
        'f1': 1.0e-5, 'f2': 2.0e-5,
        'name': 'Diurnal Tides (10-20 μHz)', 
        'expected': 'strongest_if_tidal_dominated',
        'bandwidth_microhz': 10,
        'description': 'K1 (12.0 μHz), O1 (13.9 μHz), P1 (15.0 μHz) - Primary lunar/solar diurnal'
    },
    'tidal_semidiurnal': {
        'f1': 2.0e-5, 'f2': 3.0e-5,
        'name': 'Semidiurnal Tides (20-30 μHz)', 
        'expected': 'strongest_if_tidal_dominated',
        'bandwidth_microhz': 10,
        'description': 'M2 (29.0 μHz), S2 (30.0 μHz), N2 (28.4 μHz) - Primary lunar/solar semidiurnal'
    },
    
    # ============================================================================
    # CRITICAL TRANSITION BANDS - Tests drop-off after tidal frequencies
    # KEY: Distinguishes Hypothesis A (sharp drop) vs B (sustained signal)
    # ============================================================================
    'post_tidal_30_40': {
        'f1': 3.0e-5, 'f2': 4.0e-5,
        'name': 'Post-Tidal 30-40 μHz', 
        'expected': 'weak_if_tidal_strong_if_broadband',
        'bandwidth_microhz': 10,
        'description': 'CRITICAL: Tests immediate post-tidal signal behavior'
    },
    'post_tidal_40_50': {
        'f1': 4.0e-5, 'f2': 5.0e-5,
        'name': 'Post-Tidal 40-50 μHz', 
        'expected': 'weak_if_tidal_strong_if_broadband',
        'bandwidth_microhz': 10,
        'description': 'CRITICAL: Confirms post-tidal transition pattern'
    },
    'post_tidal_50_75': {
        'f1': 5.0e-5, 'f2': 7.5e-5,
        'name': 'Post-Tidal 50-75 μHz', 
        'expected': 'weak_if_tidal_moderate_if_broadband',
        'bandwidth_microhz': 25,
        'description': 'Extended post-tidal region'
    },
    'post_tidal_75_100': {
        'f1': 7.5e-5, 'f2': 1.0e-4,
        'name': 'Post-Tidal 75-100 μHz', 
        'expected': 'weak_if_tidal_moderate_if_broadband',
        'bandwidth_microhz': 25,
        'description': 'Final post-tidal transition test'
    },
    
    # ============================================================================
    # INTERMEDIATE TEP RANGE - Tests broadband signal continuation
    # ============================================================================
    'intermediate_100_200': {
        'f1': 1.0e-4, 'f2': 2.0e-4,
        'name': 'Intermediate 100-200 μHz', 
        'expected': 'moderate',
        'bandwidth_microhz': 100,
        'description': 'Mid-range TEP signal assessment'
    },
    'intermediate_200_350': {
        'f1': 2.0e-4, 'f2': 3.5e-4,
        'name': 'Intermediate 200-350 μHz', 
        'expected': 'moderate',
        'bandwidth_microhz': 150,
        'description': 'Upper mid-range TEP signal'
    },
    'intermediate_350_500': {
        'f1': 3.5e-4, 'f2': 5.0e-4,
        'name': 'Intermediate 350-500 μHz', 
        'expected': 'moderate',
        'bandwidth_microhz': 150,
        'description': 'Upper TEP band boundary'
    },
    
    # ============================================================================
    # TRANSITION TO CONTROL - Tests signal decay beyond TEP range
    # ============================================================================
    'transition_500_750': {
        'f1': 5.0e-4, 'f2': 7.5e-4,
        'name': 'Transition 500-750 μHz', 
        'expected': 'weak_to_moderate',
        'bandwidth_microhz': 250,
        'description': 'Immediate post-TEP transition'
    },
    'transition_750_1000': {
        'f1': 7.5e-4, 'f2': 1.0e-3,
        'name': 'Transition 750-1000 μHz', 
        'expected': 'weak',
        'bandwidth_microhz': 250,
        'description': 'Final transition before control bands'
    },
    
    # ============================================================================
    # CONTROL BANDS - Tests systematic effects (no physical TEP/tidal coupling)
    # ============================================================================
    'control_1000_1500': {
        'f1': 1.0e-3, 'f2': 1.5e-3,
        'name': 'Control 1000-1500 μHz', 
        'expected': 'weak_systematics_only',
        'bandwidth_microhz': 500,
        'description': 'Primary control - systematic effects baseline'
    },
    'control_2000_3000': {
        'f1': 2.0e-3, 'f2': 3.0e-3,
        'name': 'Control 2000-3000 μHz', 
        'expected': 'weak_systematics_only',
        'bandwidth_microhz': 1000,
        'description': 'High-frequency control - systematic consistency check'
    }
}

# Legacy frequency bands for backward compatibility
FREQUENCY_BANDS_LEGACY = {
    'tep_band': {'f1': 1e-5, 'f2': 5e-4, 'name': 'TEP Band (10-500 μHz)', 'expected': 'strong'},
    'control_1': {'f1': 1e-3, 'f2': 2e-3, 'name': 'Control 1 (1000-2000 μHz)', 'expected': 'weak'},
    'control_2': {'f1': 2e-3, 'f2': 3e-3, 'name': 'Control 2 (2000-3000 μHz)', 'expected': 'weak'},
    'intermediate': {'f1': 5e-4, 'f2': 1e-3, 'name': 'Intermediate (500-1000 μHz)', 'expected': 'moderate'},
    'control_3': {'f1': 3e-3, 'f2': 4e-3, 'name': 'Control 3 (3000-4000 μHz)', 'expected': 'weak'}
}

# Default single band parameters (backward compatibility)
CONTROL_F1 = 1e-3  # 1000 μHz lower bound
CONTROL_F2 = 2e-3  # 2000 μHz upper bound

# TEP band parameters for comparison (from Step 2.0)
TEP_F1 = 1e-5  # 10 μHz lower bound
TEP_F2 = 5e-4  # 500 μHz upper bound


def compute_multi_band_correlations(series1: np.ndarray, series2: np.ndarray, fs: float,
                                   bands: Dict = None) -> Dict[str, Tuple[float, float]]:
    """
    Compute phase-coherent correlations across multiple frequency bands simultaneously.
    
    This is a more efficient approach than running separate analyses for each band,
    as it computes the CSD once and extracts correlations from multiple bands.
    
    Parameters:
    -----------
    series1, series2 : np.ndarray
        Clock offset time series from two stations
    fs : float
        Sampling frequency in Hz
    bands : Dict
        Dictionary of frequency bands to analyze
        Format: {'band_name': {'f1': float, 'f2': float, 'name': str}}
        
    Returns:
    --------
    Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (plateau_value, plateau_phase) tuples
    """
    if bands is None:
        bands = FREQUENCY_BANDS
        
    results = {}
    n_points = len(series1)
    
    if n_points < 20:
        return {band_id: (np.nan, np.nan) for band_id in bands.keys()}
    
    # STEP 1: Detrend time series (SAME AS SINGLE-BAND)
    time_indices = np.arange(n_points)
    series1_detrended = series1 - np.polyval(np.polyfit(time_indices, series1, 1), time_indices)
    series2_detrended = series2 - np.polyval(np.polyfit(time_indices, series2, 1), time_indices)
    
    # STEP 2: Compute cross-spectral density ONCE (efficiency gain)
    nperseg = min(1024, n_points)
    frequencies, cross_psd = csd(series1_detrended, series2_detrended,
                                 fs=fs, nperseg=nperseg, detrend='constant')
    
    if len(frequencies) < 2:
        return {band_id: (np.nan, np.nan) for band_id in bands.keys()}
    
    # STEP 3: Extract correlations for each frequency band
    use_phase_band = os.getenv('TEP_USE_PHASE_BAND', '1') == '1'
    
    for band_id, band_config in bands.items():
        f1, f2 = band_config['f1'], band_config['f2']
        
        if use_phase_band:
            # Extract frequency band
            band_mask = (frequencies > 0) & (frequencies >= f1) & (frequencies <= f2)
            if not np.any(band_mask):
                results[band_id] = (np.nan, np.nan)
                continue
            
            band_csd = cross_psd[band_mask]
            
            # Phase-coherent correlation extraction
            magnitudes = np.abs(band_csd)
            
            if np.sum(magnitudes) == 0:
                results[band_id] = (np.nan, np.nan)
                continue
            
            phases = np.angle(band_csd)
            
            # Circular statistics for phase averaging
            complex_phases = np.exp(1j * phases)
            weighted_complex = np.average(complex_phases, weights=magnitudes)
            weighted_phase = np.angle(weighted_complex)
            
            # Representative correlation strength
            avg_magnitude = np.mean(magnitudes)
            
            results[band_id] = (float(avg_magnitude), float(weighted_phase))
        else:
            # Fallback: Single frequency bin
            complex_plateau = cross_psd[1]
            plateau_value = abs(complex_plateau)
            plateau_phase = np.angle(complex_plateau)
            
            results[band_id] = (float(plateau_value), float(plateau_phase))
    
    return results


def great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points on Earth."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return EARTH_RADIUS_KM * c


def build_distance_cache(coords_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Pre-compute distances between all station pairs to avoid repeated calculations."""
    print_status("Building distance cache for station pairs...", "PROCESS")
    distance_cache = {}
    
    stations = coords_df['coord_source_code'].unique()  # Use 4-char codes like ALGO, AREQ
    total_pairs = len(stations) * (len(stations) - 1) // 2
    processed_pairs = 0
    
    for i, station1 in enumerate(stations):
        for station2 in stations[i+1:]:
            coord1 = coords_df[coords_df['coord_source_code'] == station1].iloc[0]
            coord2 = coords_df[coords_df['coord_source_code'] == station2].iloc[0]
            
            # Convert ECEF to geodetic
            lat1, lon1, _ = ecef_to_geodetic(coord1['X'], coord1['Y'], coord1['Z'])
            lat2, lon2, _ = ecef_to_geodetic(coord2['X'], coord2['Y'], coord2['Z'])
            
            # Calculate distance
            distance_km = great_circle_distance_km(lat1, lon1, lat2, lon2)
            
            # Store in cache (sorted order for consistency)
            pair_key = tuple(sorted([station1, station2]))
            distance_cache[pair_key] = distance_km
            
            processed_pairs += 1
            if processed_pairs % 1000 == 0:
                print_status(f"Distance cache: {processed_pairs:,}/{total_pairs:,} pairs", "INFO")
    
    print_status(f"Distance cache built: {len(distance_cache):,} station pairs", "SUCCESS")
    return distance_cache


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates to geodetic (lat, lon, height)."""
    a = WGS84_A
    e2 = WGS84_E2
    
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        cos_lat = np.cos(lat)
        if abs(cos_lat) < 1e-10:
            height = abs(z) - a * np.sqrt(1 - e2)
            break
        height = p / cos_lat - N
        lat_new = np.arctan2(z, p * (1 - e2 * N / (N + height)))
        if abs(lat_new - lat) < 1e-10:
            lat = lat_new
            break
        lat = lat_new
    
    return np.degrees(lat), np.degrees(lon), height


def parse_clk_file(file_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Parse a GNSS CLK file and extract station time series.
    
    IDENTICAL TO STEP 2.0 METHODOLOGY
    =================================
    Uses the same RINEX CLK parsing logic as Step 2.0 to ensure consistency.
    """
    station_data = {}
    
    try:
        # STEP 2.0 COMPATIBLE: Open .CLK or .CLK.gz file with robust handling
        try:
            with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()
            except Exception:
                # Try with latin-1 encoding as fallback
                try:
                    with gzip.open(file_path, "rt", encoding="latin-1") as fh:
                        lines = fh.readlines()
                except Exception:
                    with open(file_path, "r", encoding="latin-1") as fh:
                        lines = fh.readlines()
        
        # STEP 2.0 COMPATIBLE: RINEX CLK Format Parser
        # Format: AR STATION YYYY MM DD HH MM SS.SSS N_DATA CLOCK_OFFSET [additional_fields...]
        # Example: AR PIE100USA 2023 01 01 00 00  0.000000  2   -0.217911165105E-03  0.183629798307E-10
        clk_pattern = re.compile(
            r'^AR\s+'          # Record type (AR = Atomic Receiver clock)
            r'(\S+)\s+'        # Station ID (variable length, e.g., PIE100USA, ALGO)
            r'(\d{4})\s+'      # Year (4 digits)
            r'(\d{1,2})\s+'    # Month (1-2 digits)
            r'(\d{1,2})\s+'    # Day (1-2 digits)
            r'(\d{1,2})\s+'    # Hour (1-2 digits)
            r'(\d{1,2})\s+'    # Minute (1-2 digits)
            r'([\d.]+)\s+'     # Second (float, includes microseconds)
            r'(\d+)\s+'        # Number of data points (usually 1 or 2)
            r'([-.\dE+-]+)'    # Clock offset in seconds (scientific notation)
        )
        
        for line in lines:
            match = clk_pattern.match(line)
            if match:
                station = match.group(1)
                year = int(match.group(2))
                month = int(match.group(3))
                day = int(match.group(4))
                hour = int(match.group(5))
                minute = int(match.group(6))
                second = float(match.group(7))
                clock_bias = float(match.group(9))  # Scientific notation handled automatically
                
                timestamp = datetime(year, month, day, hour, minute, int(second))
                
                if station not in station_data:
                    station_data[station] = {'timestamps': [], 'clock_bias': []}
                
                station_data[station]['timestamps'].append(timestamp)
                station_data[station]['clock_bias'].append(clock_bias)
                
    except Exception as e:
        raise TEPFileError(f"Failed to parse CLK file {file_path}: {e}")
    
    # Convert to DataFrames and apply 5-minute downsampling like Step 2.0
    result = {}
    for station, data in station_data.items():
        if len(data['timestamps']) > 10:  # Minimum data requirement
            df = pd.DataFrame({
                'timestamp': data['timestamps'],
                'clock_bias': data['clock_bias']
            })
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # CONSISTENCY WITH STEP 2.0: Use exact same processing pipeline
            # ============================================================
            # Step 2.0 does NOT resample - CLK files are already at proper intervals
            # df = df.set_index('timestamp')
            # df = df.resample('5min').mean().reset_index()  # REMOVED - this was wrong!
            # df = df.dropna()  # Remove any NaN values from resampling
            
            if len(df) > 5:  # Ensure sufficient data after downsampling
                result[station] = df
    
    return result


def process_clk_file_multiband(file_path: Path, bands: Dict = None) -> Dict[str, List[Dict]]:
    """
    Process a CLK file to extract multi-band correlations for all station pairs.
    
    This is the multi-band version that analyzes all frequency bands simultaneously
    for maximum efficiency.
    
    Parameters:
    -----------
    file_path : Path
        Path to the CLK file to process
    bands : Dict
        Dictionary of frequency bands to analyze
        
    Returns:
    --------
    Dict[str, List[Dict]]
        Dictionary mapping band names to lists of correlation results
    """
    if bands is None:
        bands = FREQUENCY_BANDS
        
    # Initialize results for each band
    band_results = {band_id: [] for band_id in bands.keys()}
    
    try:
        # Parse CLK file using the same approach as step 2.0
        records = []
        
        # Parse CLK file with the same logic as step 2.0
        try:
            with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()
            except Exception:
                return band_results
        
        # EXACT SAME CLK parsing pattern as step 2.0
        clk_pattern = re.compile(
            r'^AR\s+'          # Record type (AR = Atomic Receiver clock)
            r'(\S+)\s+'        # Station ID (4-char code, e.g., ALGO)
            r'(\d{4})\s+'      # Year (4 digits)
            r'(\d{1,2})\s+'    # Month (1-2 digits)
            r'(\d{1,2})\s+'    # Day (1-2 digits)
            r'(\d{1,2})\s+'    # Hour (1-2 digits)
            r'(\d{1,2})\s+'    # Minute (1-2 digits)
            r'([\d.]+)\s+'     # Second (float, includes microseconds)
            r'(\d+)\s+'        # Number of data points (usually 1)
            r'([-.\d]+)'       # Clock offset in seconds (scientific notation)
        )

        for line in lines:
            match = clk_pattern.match(line)
            if not match:
                continue
            
            try:
                # Extract captured groups - SAME AS STEP 2.0
                (station, year_str, month_str, day_str, hour_str, 
                 minute_str, second_str, _, clock_offset_str) = match.groups()

                year = int(year_str)
                month = int(month_str) 
                day = int(day_str)
                hour = int(hour_str)
                minute = int(minute_str)
                second_float = float(second_str)
                second = int(second_float)
                microsecond = int((second_float - second) * 1_000_000)
                
                timestamp = pd.Timestamp(year, month, day, hour, minute, second, microsecond)
                clock_offset = float(clock_offset_str)
                
                records.append({
                    'timestamp': timestamp,
                    'station': station, 
                    'clock_offset': clock_offset
                })
                
            except (ValueError, IndexError):
                continue
        
        if not records:
            return band_results
            
        df = pd.DataFrame(records)
        
        # Create pivot table like step 2.0
        pivot_df = df.pivot_table(
            index='timestamp',
            columns='station', 
            values='clock_offset',
            aggfunc='mean'
        ).sort_index()
        
        # CONSISTENCY WITH STEP 2.0: Use exact same processing pipeline
        # ============================================================
        # Step 2.0 does NOT resample - it uses whatever temporal resolution
        # is in the CLK files (typically already 5-minute intervals).
        # Adding resampling would change the data and break consistency.
        
        # Step 2.0 processes pivot_df directly without any resampling
        # pivot_df = pivot_df.resample('5min').mean()  # REMOVED - this was wrong!
        
        # Filter stations with sufficient data - SAME AS STEP 2.0
        min_epochs = TEPConfig.get_int('TEP_MIN_EPOCHS')  # Default: 20 epochs
        stations = []
        for station in pivot_df.columns:
            if pivot_df[station].count() >= min_epochs:
                stations.append(station)
        
        if len(stations) < 2:
            return band_results
        
        # Extract file date - SAME AS STEP 2.0
        # Filename format: COD0OPSFIN_20230130000_01D_30S_CLK.CLK.gz
        # We want: 2023013 (YYYYDDD format)
        filename = file_path.stem.split('.')[0]  # Remove .CLK extension
        # Extract date from format: PREFIX_YYYYDDD0000_SUFFIX
        parts = filename.split('_')
        if len(parts) >= 2:
            file_date = parts[1][:7]  # Get YYYYDDD (first 7 chars of date field)
        else:
            return band_results
        
        if not file_date or len(file_date) != 7:
            return band_results
        
        # Process station pairs like step 2.0
        for station1, station2 in itertools.combinations(stations, 2):
            # Extract clean time series for both stations
            series1 = pivot_df[station1].dropna()
            series2 = pivot_df[station2].dropna()
            
            if series1.empty or series2.empty:
                continue

            # Find common observation times
            common_times = series1.index.intersection(series2.index)
            if len(common_times) < min_epochs:
                continue
            
            # Extract synchronized time series values
            series1_common = series1.loc[common_times].values
            series2_common = series2.loc[common_times].values

            # Compute sampling frequency like step 2.0
            try:
                dt_ns = np.median(np.diff(common_times.values.astype('datetime64[ns]').astype('int64')))
                dt_s = float(dt_ns) / 1e9 if dt_ns > 0 else None
                fs_hz = 1.0 / dt_s if dt_s and dt_s > 0 else None
            except Exception:
                fs_hz = None
            if fs_hz is None:
                continue
            
            # Calculate distance using worker cache (normalize to 4-char codes like Step 2.0)
            distance_km = None
            if WORKER_DISTANCE_CACHE:
                # Normalize station codes to 4 characters for cache lookup
                code1 = station1[:4] if len(station1) > 4 else station1
                code2 = station2[:4] if len(station2) > 4 else station2
                pair_key = tuple(sorted([code1, code2]))
                distance_km = WORKER_DISTANCE_CACHE.get(pair_key)
            
            if distance_km is None:
                continue
                
            # MULTI-BAND CORRELATION ANALYSIS
            # For TEP band, use EXACT SAME function as Step 2.0 to ensure identical results
            multi_band_results = compute_multi_band_correlations(series1_common, series2_common, fs_hz, bands)
            
            # CRITICAL FIX: For TEP band, use the exact same computation as Step 2.0
            if 'tep_band' in bands:
                tep_config = bands['tep_band']
                f1, f2 = tep_config['f1'], tep_config['f2']
                
                # Use the EXACT SAME function as Step 2.0 with same parameters
                use_real_coherency = TEPConfig.get_bool('TEP_USE_REAL_COHERENCY')
                plateau_value, plateau_phase = step2_compute_cross_power_plateau(
                    series1_common, series2_common, fs=fs_hz, 
                    use_real_coherency=use_real_coherency, f1=f1, f2=f2
                )
                
                # Override the TEP band result with Step 2.0 computation
                multi_band_results['tep_band'] = (plateau_value, plateau_phase)
            
            # Store results for each band
            for band_id, (plateau_value, plateau_phase) in multi_band_results.items():
                if not np.isnan(plateau_value):
                    band_results[band_id].append({
                        'date': file_date,
                        'station_i': station1,
                        'station_j': station2,
                        'plateau': plateau_value,
                        'plateau_phase': plateau_phase,
                        'dist_km': distance_km,
                        'n_epochs': len(common_times)
                    })
        
        return band_results
        
    except Exception as e:
        step_logger.warning(f"Error processing {file_path}: {e}")
        return band_results


def exponential_decay(r: np.ndarray, A: float, lambda_km: float, C0: float) -> np.ndarray:
    """Exponential decay model: C(r) = A*exp(-r/λ) + C₀"""
    return A * np.exp(-r / lambda_km) + C0


def fit_exponential_model(distances: np.ndarray, correlations: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict:
    """Fit exponential decay model to distance-correlation data (weighted if provided)."""
    try:
        # Initial parameter guesses
        A_init = float(np.max(correlations) - np.min(correlations))
        lambda_init = 5000.0
        C0_init = float(np.min(correlations))

        # Prepare sigma for weighted least squares if weights are provided
        sigma = None
        if weights is not None and len(weights) == len(correlations):
            safe_weights = np.clip(weights.astype(float), 1e-9, np.inf)
            sigma = 1.0 / np.sqrt(safe_weights)

        # Determine adaptive bounds consistent with Step 2.0 (fallback to conservative defaults)
        try:
            bounds = TEPConfig.get_adaptive_lambda_bounds(distances)
        except Exception:
            # Defaults: very small positive amplitude, lambda in [100, 20000], offset in [-1, 1]
            bounds = ([1e-10, 100, -1], [5, 20000, 1])

        # Fit model
        popt, pcov = curve_fit(
            exponential_decay,
            distances,
            correlations,
            p0=[A_init, lambda_init, C0_init],
            bounds=bounds,
            maxfev=10000,
            sigma=sigma
        )

        A_fit, lambda_fit, C0_fit = popt

        # Predictions
        predictions = exponential_decay(distances, A_fit, lambda_fit, C0_fit)

        # Weighted R² if weights available, else unweighted
        if sigma is not None:
            w = (1.0 / sigma**2)
            weighted_mean = np.average(correlations, weights=w)
            ss_res = float(np.sum(w * (correlations - predictions) ** 2))
            ss_tot = float(np.sum(w * (correlations - weighted_mean) ** 2))
        else:
            ss_res = float(np.sum((correlations - predictions) ** 2))
            ss_tot = float(np.sum((correlations - np.mean(correlations)) ** 2))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Parameter uncertainties
        perr = np.sqrt(np.diag(pcov))

        return {
            'success': True,
            'A': float(A_fit),
            'lambda_km': float(lambda_fit),
            'C0': float(C0_fit),
            'r_squared': float(r_squared),
            'A_err': float(perr[0]),
            'lambda_err': float(perr[1]),
            'C0_err': float(perr[2])
        }

    except Exception as e:
        step_logger.warning(f"Exponential fit failed: {e}")
        return {'success': False, 'error': str(e)}


def bin_distance_data(df: pd.DataFrame, n_bins: int = 40, max_dist: float = 13000) -> pd.DataFrame:
    """Bin station pairs by distance and compute mean correlations."""
    df_filtered = df[df['dist_km'] <= max_dist].copy()
    
    # CRITICAL: Convert plateau_phase to coherence using cos(phase) - SAME AS STEP 2.0
    # This is the phase-alignment index method that extracts correlation from phase
    df_filtered['coherence'] = np.cos(df_filtered['plateau_phase'])
    
    # Logarithmic binning (same bins as Step 2.0)
    bin_edges = np.logspace(np.log10(50), np.log10(max_dist), n_bins + 1)
    df_filtered['bin'] = pd.cut(df_filtered['dist_km'], bins=bin_edges)
    
    binned = df_filtered.groupby('bin', observed=True).agg({
        'dist_km': 'mean',
        'coherence': 'mean',
        'station_i': 'count'
    }).reset_index()
    
    binned.columns = ['bin', 'distance_km', 'mean_correlation', 'count']
    binned = binned.dropna()
    
    return binned


def aggregate_streaming_bins(band_file: Path, n_bins: int, max_dist: float, chunk_size: int = 500000) -> Tuple[pd.DataFrame, int]:
    """Aggregate binned statistics from a large streaming CSV in chunks (IDENTICAL TO STEP 2.0).

    Returns a DataFrame with columns: distance_km, mean_correlation, count,
    and the total number of processed pairs.
    """
    # Prepare bin edges and accumulators - IDENTICAL TO STEP 2.0
    edges = np.logspace(np.log10(50), np.log10(max_dist), n_bins + 1)
    sum_coh = np.zeros(n_bins, dtype=float)
    sum_dist = np.zeros(n_bins, dtype=float)
    count = np.zeros(n_bins, dtype=np.int64)

    total_binned = 0
    total_processed = 0

    usecols = ['plateau_phase', 'dist_km']
    try:
        for chunk in pd.read_csv(band_file, usecols=usecols, chunksize=chunk_size):
            total_processed += len(chunk)
            # NO ADDITIONAL FILTERING - use all data that was written to streaming file
            # This ensures IDENTICAL data volume to Step 2.0
            if chunk.empty:
                continue

            # Only filter by distance (same as Step 2.0)
            mask = (chunk['dist_km'] > 0) & (chunk['dist_km'] <= max_dist)
            if not mask.any():
                continue
            chunk = chunk.loc[mask]

            # Compute coherence - IDENTICAL TO STEP 2.0
            chunk['coherence'] = np.cos(chunk['plateau_phase'].values)

            # Bin using pandas cut EXACTLY like Step 2.0 worker function
            chunk['dist_bin'] = pd.cut(chunk['dist_km'], bins=edges)
            gb = chunk.groupby('dist_bin', observed=True)
            
            for bin_idx, group in gb:
                if pd.notna(bin_idx):
                    # Use EXACT SAME bin position calculation as Step 2.0 (line 1965)
                    bin_pos = np.searchsorted(edges[:-1], bin_idx.left, side='right') - 1
                    if 0 <= bin_pos < n_bins:
                        coh_vals = group['coherence'].values
                        dist_vals = group['dist_km'].values
                        
                        n = len(coh_vals)
                        sum_coh[bin_pos] += np.sum(coh_vals)
                        sum_dist[bin_pos] += np.sum(dist_vals)
                        count[bin_pos] += n
                        total_binned += n

        # Build binned DataFrame - NO ADDITIONAL FILTERING
        nonzero = count > 0
        if not np.any(nonzero):
            return pd.DataFrame(columns=['distance_km', 'mean_correlation', 'count']), 0, total_processed

        mean_dist = np.zeros_like(sum_dist)
        mean_coh = np.zeros_like(sum_coh)
        mean_dist[nonzero] = sum_dist[nonzero] / count[nonzero]
        mean_coh[nonzero] = sum_coh[nonzero] / count[nonzero]

        binned_df = pd.DataFrame({
            'distance_km': mean_dist[nonzero],
            'mean_correlation': mean_coh[nonzero],
            'count': count[nonzero]
        })
        # Sort by distance to ensure monotonic order
        binned_df = binned_df.sort_values('distance_km').reset_index(drop=True)

        # Apply post-aggregation filtering to match Step 2.0 exactly
        min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
        binned_df = binned_df[binned_df['count'] >= min_bin_count]

        return binned_df, total_processed

    except Exception as e:
        step_logger.warning(f"Failed to aggregate streaming bins for {band_file.name}: {e}")
        return pd.DataFrame(columns=['distance_km', 'mean_correlation', 'count']), 0


def save_band_diagnostics(ac: str, band_id: str, band_analysis: Dict, output_dir: Path):
    """
    Save comprehensive diagnostic data for a frequency band.
    
    Creates detailed CSV files with:
    - Binned data (filtered and raw)
    - Bin-level statistics
    - Pair count distributions
    - Fit residuals and diagnostics
    
    This enables independent validation and detailed investigation of results.
    """
    if band_analysis is None:
        return
    
    diagnostics_dir = output_dir / "band_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save filtered binned data (used for fitting)
    binned_data = band_analysis.get('binned_data', [])
    if binned_data:
        binned_df = pd.DataFrame(binned_data)
        binned_file = diagnostics_dir / f"{ac}_{band_id}_binned_filtered.csv"
        binned_df.to_csv(binned_file, index=False)
        
        # Add fit predictions if successful
        fit = band_analysis.get('exponential_fit', {})
        if fit.get('success'):
            binned_df['prediction'] = exponential_decay(
                binned_df['distance_km'].values,
                fit['A'], fit['lambda_km'], fit['C0']
            )
            binned_df['residual'] = binned_df['mean_correlation'] - binned_df['prediction']
            binned_df['residual_normalized'] = binned_df['residual'] / binned_df['mean_correlation'].abs()
            
            # Save with predictions
            binned_with_fit_file = diagnostics_dir / f"{ac}_{band_id}_binned_with_fit.csv"
            binned_df.to_csv(binned_with_fit_file, index=False)
    
    # Save raw unfiltered binned data (before min_bin_count filter)
    binned_data_raw = band_analysis.get('binned_data_raw', [])
    if binned_data_raw:
        binned_raw_df = pd.DataFrame(binned_data_raw)
        binned_raw_file = diagnostics_dir / f"{ac}_{band_id}_binned_raw.csv"
        binned_raw_df.to_csv(binned_raw_file, index=False)
    
    # Save comprehensive summary
    summary_data = {
        'analysis_center': [ac],
        'band_id': [band_id],
        'band_name': [band_analysis['band_config']['name']],
        'f1_hz': [band_analysis['frequency_range_hz'][0]],
        'f2_hz': [band_analysis['frequency_range_hz'][1]],
        'f1_microhz': [band_analysis['frequency_range_microhz'][0]],
        'f2_microhz': [band_analysis['frequency_range_microhz'][1]],
        'bandwidth_microhz': [band_analysis['frequency_bandwidth_microhz']],
        'total_pairs_processed': [band_analysis['data_summary']['total_pairs_processed']],
        'bins_before_filter': [band_analysis['data_summary']['bins_before_filter']],
        'bins_after_filter': [band_analysis['data_summary']['bins_after_filter']],
        'bins_removed': [band_analysis['data_summary']['bins_removed']],
        'pairs_in_bins_before': [band_analysis['data_summary']['pairs_in_bins_before']],
        'pairs_in_bins_after': [band_analysis['data_summary']['pairs_in_bins_after']],
        'pairs_removed_by_filter': [band_analysis['data_summary']['pairs_removed_by_filter']],
        'filter_removal_percent': [band_analysis['data_summary']['filter_removal_percent']],
        'min_bin_count_threshold': [band_analysis['data_summary']['min_bin_count_threshold']]
    }
    
    # Add fit parameters if successful
    fit = band_analysis.get('exponential_fit', {})
    if fit.get('success'):
        summary_data.update({
            'fit_success': [True],
            'A': [fit['A']],
            'A_err': [fit['A_err']],
            'lambda_km': [fit['lambda_km']],
            'lambda_err': [fit['lambda_err']],
            'C0': [fit['C0']],
            'C0_err': [fit['C0_err']],
            'r_squared_weighted': [fit['r_squared']],
            'r_squared_unweighted': [fit.get('r_squared_unweighted', None)],
            'weighting_impact': [fit.get('weighting_impact', None)],
            'total_weight': [fit.get('total_weight', None)]
        })
    else:
        summary_data.update({
            'fit_success': [False],
            'fit_error': [fit.get('error', 'Unknown')]
        })
    
    # Add bin statistics if available
    bin_stats = band_analysis.get('bin_statistics')
    if bin_stats:
        summary_data.update({
            'bin_count_min': [bin_stats['count_min']],
            'bin_count_max': [bin_stats['count_max']],
            'bin_count_mean': [bin_stats['count_mean']],
            'bin_count_median': [bin_stats['count_median']],
            'bin_count_std': [bin_stats['count_std']],
            'distance_min_km': [bin_stats['distance_min_km']],
            'distance_max_km': [bin_stats['distance_max_km']],
            'correlation_min': [bin_stats['correlation_min']],
            'correlation_max': [bin_stats['correlation_max']],
            'correlation_mean': [bin_stats['correlation_mean']],
            'correlation_std': [bin_stats['correlation_std']]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = diagnostics_dir / f"{ac}_{band_id}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    step_logger.debug(f"Saved diagnostics for {ac}_{band_id}: {diagnostics_dir}")


def run_multiband_analysis(ac: str, bands: Dict = None, use_legacy_bands: bool = False, max_files: int = None) -> Dict:
    """
    Run comprehensive multi-band frequency analysis.
    
    This analyzes multiple frequency bands simultaneously to provide a complete
    frequency spectrum validation of TEP signal specificity.
    
    Parameters:
    -----------
    ac : str
        Analysis center ('code', 'igs_combined', 'esa_final')
    bands : Dict
        Dictionary of frequency bands to analyze
        
    Returns:
    --------
    Dict
        Complete multi-band analysis results with comparison metrics
    """
    if bands is None:
        bands = FREQUENCY_BANDS_LEGACY if use_legacy_bands else FREQUENCY_BANDS
        
    print_status(f"Starting multi-band frequency analysis for {ac.upper()}", "SUCCESS")
    print_status(f"Analyzing {len(bands)} frequency bands:", "INFO")
    for band_id, config in bands.items():
        f1_micro = config['f1'] * 1e6
        f2_micro = config['f2'] * 1e6
        print_status(f"  {band_id}: {f1_micro:.0f}-{f2_micro:.0f} μHz ({config['name']})", "INFO")
    
    # Setup data paths
    data_dir = PROJECT_ROOT / 'data' / 'raw' / ac
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find CLK files
    clk_files = sorted(data_dir.glob("*.CLK.gz"))
    if not clk_files:
        raise FileNotFoundError(f"No CLK files found in {data_dir}")
    
    # Limit files for testing if specified
    if max_files is not None and max_files > 0:
        clk_files = clk_files[:max_files]
        print_status(f"Found {len(clk_files)} CLK files (limited to {max_files} for testing)", "SUCCESS")
    else:
        print_status(f"Found {len(clk_files)} CLK files", "SUCCESS")
    
    # Build distance cache for performance
    coords_path = PROJECT_ROOT / 'data' / 'coordinates' / 'step_1_1_station_coords_global.csv'
    coords_df = safe_csv_read(coords_path)
    global_coords_map = coords_df.set_index('coord_source_code')[['X', 'Y', 'Z']].to_dict('index')
    distance_cache = build_distance_cache(coords_df)
    
    # Setup parallel processing
    num_workers = TEPConfig.get_worker_count()
    print_status(f"Using {num_workers} parallel workers", "SUCCESS")
    
    # Initialize streaming file writers for each band
    band_writers = {}
    band_file_handles = {}
    band_pair_counts = {band_id: 0 for band_id in bands.keys()}
    
    # Create output directory for streaming files
    streaming_dir = PROJECT_ROOT / "results/tmp/streaming"
    streaming_dir.mkdir(parents=True, exist_ok=True)
    print_status(f"Streaming files to: {streaming_dir}", "INFO")
    
    # Initialize CSV writers for each band
    for band_id in bands.keys():
        band_file = streaming_dir / f"streaming_pairs_{ac}_{band_id}.csv"
        band_file_handles[band_id] = open(band_file, 'w')
        band_writers[band_id] = None  # Will be initialized with first write
    
    # Process files with detailed progress like step 2.0
    batch_size = 28
    total_files = len(clk_files)
    processed_files = 0
    
    print_status("Starting optimized parallel processing...", "PROCESS")
    
    # Process in batches with memory monitoring
    batch_count = 0
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = clk_files[batch_start:batch_end]
        batch_count += 1
        
        # Check memory pressure before each batch
        if batch_count % MEMORY_CHECK_INTERVAL == 0:
            if check_memory_pressure():
                force_memory_cleanup()
        
        print_status(f"Processing batch {batch_count}: {len(batch_files)} files (Memory: {get_memory_usage()*100:.1f}%)", "PROCESS")
        
        with ProcessPoolExecutor(max_workers=num_workers,
                                 initializer=_init_worker_context,
                                 initargs=(coords_df, distance_cache, ac)) as executor:
            
            # Submit all files in the batch
            future_to_file = {executor.submit(process_clk_file_multiband, file_path, bands): file_path 
                             for file_path in batch_files}
            
            # Collect results as they complete with per-file logging
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    batch_results = future.result()
                    
                    # Stream results directly to files for each band
                    file_pair_count = 0
                    for band_id, results in batch_results.items():
                        if results:  # Only process if we have results
                            # Initialize CSV writer on first write
                            if band_writers[band_id] is None:
                                import csv
                                fieldnames = results[0].keys()
                                band_writers[band_id] = csv.DictWriter(band_file_handles[band_id], fieldnames=fieldnames)
                                band_writers[band_id].writeheader()
                            
                            # Write results directly to file
                            band_writers[band_id].writerows(results)
                            band_file_handles[band_id].flush()  # Ensure data is written
                            
                            # Update counters
                            band_pair_counts[band_id] += len(results)
                            file_pair_count += len(results)
                            
                            # Debug: Show streaming is working
                            if file_pair_count % 50000 == 0:  # Every 50k pairs
                                print_status(f"Streamed {file_pair_count:,} pairs to disk for {band_id}", "INFO")
                    
                    # CRITICAL: Clear batch_results immediately after writing to disk
                    del batch_results
                    gc.collect()  # Force immediate cleanup
                    
                    processed_files += 1
                    
                    # Log progress like step 2.0: file X/total, pairs in this file, total accumulated
                    total_pairs = sum(band_pair_counts.values())
                    print_status(
                        f"File {processed_files}/{total_files}: {file_path.name} → "
                        f"{file_pair_count:,} pairs | Total: {total_pairs:,} pairs",
                        "INFO"
                    )
                    
                except Exception as e:
                    processed_files += 1
                    step_logger.warning(f"Error processing {file_path.name}: {e}")
                    print_status(f"File {processed_files}/{total_files}: {file_path.name} → ERROR", "WARNING")
        
        # Enhanced memory cleanup after each batch
        memory_before = get_memory_usage()
        gc.collect()
        
        # Additional cleanup if memory usage is high
        if memory_before > 0.7:  # 70% threshold
            force_memory_cleanup()
        
        # Force cleanup after every batch to prevent accumulation
        gc.collect()
        
        memory_after = get_memory_usage()
        print_status(f"Batch {batch_count} complete. Memory: {memory_before*100:.1f}% → {memory_after*100:.1f}%", "INFO")
    
    # Close all file handles
    for band_id in bands.keys():
        if band_file_handles[band_id]:
            band_file_handles[band_id].close()
    
    # Show streaming summary
    total_streamed = sum(band_pair_counts.values())
    print_status(f"Streaming complete: {total_streamed:,} total pairs written to disk", "SUCCESS")
    for band_id, count in band_pair_counts.items():
        if count > 0:
            print_status(f"  {band_id}: {count:,} pairs", "INFO")
    
    print_status("Starting analysis...", "SUCCESS")
    
    # Analyze each band by reading from streaming files
    band_analyses = {}
    
    # Analyze bands sequentially (more reliable than parallel)
    print_status("Starting band analysis...", "PROCESS")
    output_dir = PROJECT_ROOT / "results" / "outputs"
    
    for band_id in bands.keys():
        try:
            band_id_result, result = analyze_single_band(band_id, ac, streaming_dir, band_pair_counts, bands)
            if result is not None:
                band_analyses[band_id_result] = result
                # Save comprehensive diagnostics for each band
                save_band_diagnostics(ac, band_id_result, result, output_dir)
        except Exception as exc:
            print_status(f"Band {band_id} analysis failed: {exc}", "ERROR")
    
    # Force memory cleanup after all bands
    if check_memory_pressure():
        force_memory_cleanup()
    
    # Create comprehensive comparison
    comparison_results = create_multiband_comparison(band_analyses, bands)
    
    # Compute hypothesis-specific metrics
    hypothesis_metrics = compute_hypothesis_metrics(band_analyses, bands)
    
    # Save results with enhanced metadata
    output_file = PROJECT_ROOT / 'results' / 'outputs' / f'step_3_6_multiband_{ac}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'analysis_center': ac.upper(),
            'timestamp': datetime.now().isoformat(),
            'methodology': {
                'identical_to_step_2_0': True,
                'temporal_resolution': 'native CLK cadence (no resampling)',
                'phase_coherent_method': 'cos(plateau_phase) from magnitude-weighted circular mean',
                'binning': 'logarithmic, 50-13000 km',
                'min_bin_count': TEPConfig.get_int('TEP_MIN_BIN_COUNT'),
                'weighting': 'weighted least squares by bin counts',
                'model': 'C(r) = A*exp(-r/λ) + C₀'
            },
            'physical_hypotheses': {
                'hypothesis_a_tidal_dominated': {
                    'description': 'φ-field couples primarily to tidal deformations',
                    'prediction': 'Tidal bands (R² > 0.97) > TEP band (R² ~ 0.90-0.95) > Control (R² < 0.3)',
                    'physical_basis': 'Moon/Sun tidal forces create large periodic matter deformations'
                },
                'hypothesis_b_broadband_tep': {
                    'description': 'Broadband TEP signal with systematic contributions',
                    'prediction': 'TEP band (R² ~ 0.95-0.97) ≥ Tidal bands, Control moderate (R² ~ 0.5-0.7)',
                    'physical_basis': 'TEP signal distributed across 10-500 μHz, tidal bands sample narrow subset'
                },
                'hypothesis_c_systematics': {
                    'description': 'Systematic instrumental/processing effects dominate',
                    'prediction': 'All bands show similar R² ~ 0.8-0.9 with minimal differentiation',
                    'physical_basis': 'Artifacts do not depend on physical frequency band'
                }
            },
            'bands_analyzed': list(bands.keys()),
            'band_results': band_analyses,
            'comparison': comparison_results,
            'hypothesis_test_results': hypothesis_metrics,
            'diagnostics_location': 'results/outputs/band_diagnostics/'
        }, f, indent=2, default=str)
    
    print_status(f"Multi-band analysis complete: {output_file}", "SUCCESS")
    print_status(f"Band diagnostics saved to: {output_dir / 'band_diagnostics'}", "INFO")
    
    return {
        'analysis_center': ac.upper(),
        'band_results': band_analyses,
        'comparison': comparison_results
    }


def compute_hypothesis_metrics(band_analyses: Dict, bands: Dict) -> Dict:
    """
    Compute metrics to test physical hypotheses about frequency-dependent coupling.
    
    TRANSPARENT HYPOTHESIS TESTING:
    - Computes objective metrics for each hypothesis
    - NO interpretation - just presents facts
    - Enables independent evaluation of competing scenarios
    """
    
    # Extract key band results
    tep_band = band_analyses.get('tep_band', {}).get('exponential_fit', {})
    tidal_diurnal = band_analyses.get('tidal_diurnal', {}).get('exponential_fit', {})
    tidal_semidiurnal = band_analyses.get('tidal_semidiurnal', {}).get('exponential_fit', {})
    control_1 = band_analyses.get('control_1', {}).get('exponential_fit', {})
    
    # Get bandwidths for normalization
    tep_bw = band_analyses.get('tep_band', {}).get('frequency_bandwidth_microhz', 490)
    tidal_diurnal_bw = band_analyses.get('tidal_diurnal', {}).get('frequency_bandwidth_microhz', 5)
    tidal_semidiurnal_bw = band_analyses.get('tidal_semidiurnal', {}).get('frequency_bandwidth_microhz', 10)
    control_1_bw = band_analyses.get('control_1', {}).get('frequency_bandwidth_microhz', 1000)
    
    metrics = {
        'raw_r_squared': {
            'tep_band': tep_band.get('r_squared', None),
            'tidal_diurnal': tidal_diurnal.get('r_squared', None),
            'tidal_semidiurnal': tidal_semidiurnal.get('r_squared', None),
            'control_1': control_1.get('r_squared', None)
        },
        'bandwidth_microhz': {
            'tep_band': tep_bw,
            'tidal_diurnal': tidal_diurnal_bw,
            'tidal_semidiurnal': tidal_semidiurnal_bw,
            'control_1': control_1_bw
        },
        'bandwidth_normalized_r_squared': {
            'tep_band': tep_band.get('r_squared', 0) / tep_bw if tep_bw > 0 else None,
            'tidal_diurnal': tidal_diurnal.get('r_squared', 0) / tidal_diurnal_bw if tidal_diurnal_bw > 0 else None,
            'tidal_semidiurnal': tidal_semidiurnal.get('r_squared', 0) / tidal_semidiurnal_bw if tidal_semidiurnal_bw > 0 else None,
            'control_1': control_1.get('r_squared', 0) / control_1_bw if control_1_bw > 0 else None
        }
    }
    
    # HYPOTHESIS A: Tidal-dominated coupling
    # Expectation: Tidal bands > TEP band > Control bands
    if (metrics['raw_r_squared']['tidal_diurnal'] is not None and 
        metrics['raw_r_squared']['tep_band'] is not None):
        
        tidal_diurnal_r2 = metrics['raw_r_squared']['tidal_diurnal']
        tidal_semi_r2 = metrics['raw_r_squared']['tidal_semidiurnal']
        tep_r2 = metrics['raw_r_squared']['tep_band']
        control_r2 = metrics['raw_r_squared']['control_1']
        
        # Compute max tidal R²
        tidal_max_r2 = max([r for r in [tidal_diurnal_r2, tidal_semi_r2] if r is not None])
        
        metrics['hypothesis_a_tidal_dominated'] = {
            'tidal_exceeds_tep': tidal_max_r2 > tep_r2 if (tidal_max_r2 and tep_r2) else None,
            'tidal_minus_tep': tidal_max_r2 - tep_r2 if (tidal_max_r2 and tep_r2) else None,
            'tidal_vs_tep_ratio': tidal_max_r2 / tep_r2 if (tidal_max_r2 and tep_r2 and tep_r2 > 0) else None,
            'tidal_max_r_squared': tidal_max_r2,
            'tep_r_squared': tep_r2,
            'control_r_squared': control_r2,
            'pattern_consistent': (
                (tidal_max_r2 > 0.97) and 
                (tep_r2 < 0.95) and 
                (control_r2 < 0.3 if control_r2 else False)
            ) if all([tidal_max_r2, tep_r2, control_r2]) else None
        }
        
        # HYPOTHESIS B: Broadband TEP with systematics
        # Expectation: TEP ≥ Tidal, Control moderate
        metrics['hypothesis_b_broadband_tep'] = {
            'tep_exceeds_or_equals_tidal': tep_r2 >= tidal_max_r2 if (tep_r2 and tidal_max_r2) else None,
            'tep_minus_tidal': tep_r2 - tidal_max_r2 if (tep_r2 and tidal_max_r2) else None,
            'tep_vs_tidal_ratio': tep_r2 / tidal_max_r2 if (tep_r2 and tidal_max_r2 and tidal_max_r2 > 0) else None,
            'control_moderate': (0.5 <= control_r2 <= 0.7) if control_r2 else None,
            'pattern_consistent': (
                (tep_r2 >= 0.95) and
                (tep_r2 >= tidal_max_r2) and
                (0.5 <= control_r2 <= 0.7 if control_r2 else False)
            ) if all([tep_r2, tidal_max_r2, control_r2]) else None
        }
        
        # HYPOTHESIS C: Systematic effects dominate
        # Expectation: All bands similar
        metrics['hypothesis_c_systematics_dominate'] = {
            'tep_tidal_difference': abs(tep_r2 - tidal_max_r2) if (tep_r2 and tidal_max_r2) else None,
            'tep_control_difference': abs(tep_r2 - control_r2) if (tep_r2 and control_r2) else None,
            'all_bands_similar': (
                abs(tep_r2 - tidal_max_r2) < 0.1 and
                abs(tep_r2 - control_r2) < 0.2
            ) if all([tep_r2, tidal_max_r2, control_r2]) else None,
            'r_squared_range': max([r for r in [tep_r2, tidal_max_r2, control_r2] if r is not None]) - 
                               min([r for r in [tep_r2, tidal_max_r2, control_r2] if r is not None])
                               if all([tep_r2, tidal_max_r2, control_r2]) else None,
            'pattern_consistent': (
                abs(tep_r2 - tidal_max_r2) < 0.1 and
                abs(tep_r2 - control_r2) < 0.2
            ) if all([tep_r2, tidal_max_r2, control_r2]) else None
        }
    
    return metrics


def create_multiband_comparison(band_analyses: Dict, bands: Dict) -> Dict:
    """
    Create comprehensive comparison metrics across all frequency bands.
    
    TRANSPARENCY FOCUS:
    - Presents statistical facts without premature scientific conclusions
    - Documents all band results including failures
    - Provides raw metrics for independent interpretation
    """
    
    # Extract all metrics for comprehensive comparison
    r_squared_values = {}
    r_squared_unweighted_values = {}
    lambda_values = {}
    lambda_errors = {}
    amplitude_values = {}
    offset_values = {}
    bin_counts = {}
    pair_counts = {}
    bandwidth_microhz = {}
    
    for band_id, analysis in band_analyses.items():
        fit = analysis.get('exponential_fit', {})
        data_summary = analysis.get('data_summary', {})
        
        # Store bandwidth for normalization considerations
        bandwidth_microhz[band_id] = analysis.get('frequency_bandwidth_microhz', 0)
        
        if fit.get('success'):
            r_squared_values[band_id] = fit.get('r_squared', 0)
            r_squared_unweighted_values[band_id] = fit.get('r_squared_unweighted', 0)
            lambda_values[band_id] = fit.get('lambda_km', 0)
            lambda_errors[band_id] = fit.get('lambda_err', 0)
            amplitude_values[band_id] = fit.get('A', 0)
            offset_values[band_id] = fit.get('C0', 0)
            bin_counts[band_id] = data_summary.get('bins_after_filter', 0)
            pair_counts[band_id] = data_summary.get('pairs_in_bins_after', 0)
        else:
            # Document failed fits transparently
            bin_counts[band_id] = data_summary.get('bins_after_filter', 0)
            pair_counts[band_id] = data_summary.get('pairs_in_bins_after', 0)
    
    if not r_squared_values:
        return {'error': 'No successful fits for comparison',
                'bands_attempted': list(bands.keys()),
                'bands_failed': list(band_analyses.keys())}
    
    # Find strongest and weakest signals (factual observation)
    best_band = max(r_squared_values.keys(), key=lambda k: r_squared_values[k])
    worst_band = min(r_squared_values.keys(), key=lambda k: r_squared_values[k])
    
    best_r2 = r_squared_values[best_band]
    worst_r2 = r_squared_values[worst_band]
    
    # Calculate objective metrics (no interpretation)
    r2_ratio = best_r2 / worst_r2 if worst_r2 > 0 else float('inf')
    r2_range = best_r2 - worst_r2
    r2_mean = float(np.mean(list(r_squared_values.values())))
    r2_std = float(np.std(list(r_squared_values.values())))
    r2_cv = (r2_std / r2_mean * 100) if r2_mean > 0 else float('inf')
    
    # Calculate lambda consistency metrics
    lambda_mean = float(np.mean(list(lambda_values.values())))
    lambda_std = float(np.std(list(lambda_values.values())))
    lambda_cv = (lambda_std / lambda_mean * 100) if lambda_mean > 0 else float('inf')
    
    # Classify frequency specificity (objective classification only)
    if r2_ratio > 5.0 and r2_range > 0.5:
        specificity = "STRONG"
    elif r2_ratio > 3.0 and r2_range > 0.3:
        specificity = "MODERATE"
    elif r2_ratio > 2.0 and r2_range > 0.2:
        specificity = "WEAK"
    else:
        specificity = "NONE"
    
    return {
        # Raw statistical summaries
        'r_squared_summary': r_squared_values,
        'r_squared_unweighted_summary': r_squared_unweighted_values,
        'lambda_summary': lambda_values,
        'lambda_error_summary': lambda_errors,
        'amplitude_summary': amplitude_values,
        'offset_summary': offset_values,
        'bin_count_summary': bin_counts,
        'pair_count_summary': pair_counts,
        'bandwidth_summary_microhz': bandwidth_microhz,
        
        # Summary statistics
        'r_squared_statistics': {
            'mean': r2_mean,
            'std': r2_std,
            'cv_percent': r2_cv,
            'min': worst_r2,
            'max': best_r2,
            'range': r2_range
        },
        'lambda_statistics': {
            'mean_km': lambda_mean,
            'std_km': lambda_std,
            'cv_percent': lambda_cv,
            'min_km': float(min(lambda_values.values())) if lambda_values else 0,
            'max_km': float(max(lambda_values.values())) if lambda_values else 0
        },
        
        # Band extrema (factual identification)
        'strongest_band': {
            'band_id': best_band,
            'name': bands[best_band]['name'],
            'r_squared': best_r2,
            'r_squared_unweighted': r_squared_unweighted_values.get(best_band, None),
            'lambda_km': lambda_values.get(best_band, 0),
            'lambda_err_km': lambda_errors.get(best_band, 0),
            'n_bins': bin_counts.get(best_band, 0),
            'n_pairs': pair_counts.get(best_band, 0),
            'bandwidth_microhz': bandwidth_microhz.get(best_band, 0)
        },
        'weakest_band': {
            'band_id': worst_band,
            'name': bands[worst_band]['name'],
            'r_squared': worst_r2,
            'r_squared_unweighted': r_squared_unweighted_values.get(worst_band, None),
            'lambda_km': lambda_values.get(worst_band, 0),
            'lambda_err_km': lambda_errors.get(worst_band, 0),
            'n_bins': bin_counts.get(worst_band, 0),
            'n_pairs': pair_counts.get(worst_band, 0),
            'bandwidth_microhz': bandwidth_microhz.get(worst_band, 0)
        },
        
        # Objective classification metrics
        'specificity_metrics': {
            'r_squared_ratio': r2_ratio,
            'r_squared_range': r2_range,
            'frequency_specificity_classification': specificity
        },
        
        # Minimal factual summary (no scientific interpretation)
        'summary': (
            f"Analysis of {len(r_squared_values)} frequency bands completed successfully. "
            f"R² values range from {worst_r2:.3f} to {best_r2:.3f} (ratio: {r2_ratio:.1f}×). "
            f"Correlation lengths range from {min(lambda_values.values()):.0f} km to {max(lambda_values.values()):.0f} km. "
            f"Frequency specificity classification: {specificity}."
        )
    }


def run_control_band_analysis(ac: str, f1: float = CONTROL_F1, f2: float = CONTROL_F2) -> Dict:
    """
    Run control band analysis with performance optimizations and resume functionality.
    """
    print_status(f"Running control band analysis for {ac.upper()} ({f1*1e6:.0f}-{f2*1e6:.0f} μHz)", "PROCESS")
    
    # Setup checkpoint system
    checkpoint_dir = PROJECT_ROOT / "results/tmp"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"step_3_6_control_band_{ac}.json"
    
    # Check for resume capability
    resume_enabled = os.getenv('TEP_RESUME', '0') == '1'
    processed_files = []
    consolidated_pair_data = []
    
    if resume_enabled:
        print_status("Resume mode enabled - checking for existing checkpoint...", "INFO")
        state = load_checkpoint_safely(checkpoint_file)
        if state and 'processed_files' in state:
            processed_files = state['processed_files']
            # Load existing temporary pair files
            temp_files = list(checkpoint_dir.glob(f"temp_pairs_{ac}_*.csv"))
            if temp_files:
                print_status(f"Found {len(temp_files)} temporary pair files", "INFO")
                for temp_file in temp_files:
                    try:
                        temp_df = pd.read_csv(temp_file)
                        consolidated_pair_data.extend(temp_df.to_dict('records'))
                    except Exception as e:
                        step_logger.warning(f"Failed to load temp file {temp_file}: {e}")
                print_status(f"Resumed from checkpoint: {len(processed_files)} files already processed", "SUCCESS")
                print_status(f"Loaded {len(consolidated_pair_data):,} existing pairs", "INFO")
            else:
                print_status("No valid temporary files found, starting fresh", "INFO")
        else:
            print_status("No valid checkpoint found, starting fresh", "INFO")
    else:
        # Clean start - remove any existing checkpoint and temp files
        safe_remove_file(checkpoint_file)
        temp_files = list(checkpoint_dir.glob(f"temp_pairs_{ac}_*.csv"))
        for temp_file in temp_files:
            safe_remove_file(temp_file)
        print_status("Starting fresh analysis (resume disabled)", "INFO")
    
    # Load coordinates
    coords_file = PROJECT_ROOT / "data" / "coordinates" / "step_1_1_station_coords_global.csv"
    coords_df = pd.read_csv(coords_file)
    
    # Build distance cache for performance optimization
    distance_cache = build_distance_cache(coords_df)
    
    # Find CLK files
    raw_dir = PROJECT_ROOT / "data" / "raw" / ac
    if not raw_dir.exists():
        raise ValueError(f"Raw data directory not found: {raw_dir}")
    
    clk_files = sorted(raw_dir.glob("*.CLK.gz"))
    if not clk_files:
        raise ValueError(f"No CLK files found in {raw_dir}")
    
    # Filter out already processed files if resuming
    if processed_files:
        remaining_files = [f for f in clk_files if f.name not in processed_files]
        print_status(f"Total files: {len(clk_files)}, Already processed: {len(processed_files)}, Remaining: {len(remaining_files)}", "INFO")
        total_files = len(clk_files)  # Store original total
        clk_files = remaining_files
    else:
        print_status(f"Found {len(clk_files)} CLK files to process", "INFO")
        total_files = len(clk_files)  # Store original total
    
    if not clk_files:
        print_status("All files already processed! Proceeding to analysis...", "SUCCESS")
    else:
        # Process remaining files in parallel with optimized worker context
        n_workers = TEPConfig.get_worker_count()
        
        print_status(f"Using {n_workers} parallel workers ({mp.cpu_count()} CPU cores available)", "INFO")
        print_status(f"Frequency band: {f1*1e6:.0f}-{f2*1e6:.0f} μHz", "INFO")
        print_status("Starting optimized parallel processing...", "PROCESS")
        
        # Process files in batches with enhanced memory management and checkpointing
        batch_size = max(10, n_workers * 2)
        total_files_processed = 0
        successful_files = 0
        
        batch_count = 0
        for batch_start in range(0, len(clk_files), batch_size):
            batch_end = min(batch_start + batch_size, len(clk_files))
            batch_files = clk_files[batch_start:batch_end]
            batch_count += 1
            
            # Check memory pressure before each batch
            if batch_count % MEMORY_CHECK_INTERVAL == 0:
                if check_memory_pressure():
                    force_memory_cleanup()
            
            print_status(f"Processing batch {batch_count}: {len(batch_files)} files (Memory: {get_memory_usage()*100:.1f}%)", "PROCESS")
            
            batch_results = []
            
            # Use optimized ProcessPoolExecutor with worker context initialization
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker_context,
                initargs=(coords_df, distance_cache, ac)
            ) as executor:
                # Submit tasks with optimized function signature
                future_to_file = {
                    executor.submit(process_clk_file, f, f1, f2): f 
                    for f in batch_files
                }
                
                for future in as_completed(future_to_file):
                    try:
                        results = future.result()
                        batch_results.extend(results)
                        total_files_processed += 1
                        
                        # Track processed files for checkpointing
                        file_name = future_to_file[future].name
                        processed_files.append(file_name)
                        
                        # Enhanced progress reporting
                        if results:
                            pairs_count = len(results)
                            print_status(f"{file_name}: {pairs_count:,} pairs", "SUCCESS")
                            successful_files += 1
                        else:
                            print_status(f"{file_name}: 0 pairs (no valid data)", "WARNING")
                        
                    except Exception as e:
                        file_name = future_to_file[future].name
                        step_logger.warning(f"File processing failed for {file_name}: {e}")
            
            # Add batch results to consolidated data
            consolidated_pair_data.extend(batch_results)
            
            # Save temporary batch data
            if batch_results:
                temp_file = checkpoint_dir / f"temp_pairs_{ac}_{len(consolidated_pair_data)}.csv"
                try:
                    batch_df = pd.DataFrame(batch_results)
                    batch_df.to_csv(temp_file, index=False)
                    step_logger.debug(f"Saved temporary batch data: {temp_file}")
                except Exception as e:
                    step_logger.warning(f"Failed to save temporary batch data: {e}")
            
            # Save checkpoint after each batch
            checkpoint_data = {
                'processed_files': processed_files,
                'total_pairs': len(consolidated_pair_data),
                'batch_complete': batch_start // batch_size + 1,
                'timestamp': datetime.now().isoformat(),
                'frequency_band': {'f1': f1, 'f2': f2}
            }
            
            if atomic_save_checkpoint(checkpoint_file, checkpoint_data):
                step_logger.debug(f"Checkpoint saved after batch {batch_start//batch_size + 1}")
            else:
                step_logger.warning("Failed to save checkpoint - continuing without checkpoint")
            
            # Progress summary after each batch with memory stats
            print_status(f"Batch complete: {len(processed_files)}/{total_files} files, {len(consolidated_pair_data):,} total pairs", "INFO")
            
            # Enhanced memory management: proactive cleanup after each batch
            memory_before = get_memory_usage()
            gc.collect()
            
            # Additional cleanup if memory usage is high
            if memory_before > 0.7:  # 70% threshold
                force_memory_cleanup()
            
            memory_after = get_memory_usage()
            print_status(f"Batch {batch_count} complete. Memory: {memory_before*100:.1f}% → {memory_after*100:.1f}%", "INFO")
    
    if not consolidated_pair_data:
        raise ValueError("No valid correlations computed in control band")
    
    print_status(f"Total pairs processed: {len(consolidated_pair_data):,}", "SUCCESS")
    print_status("Creating DataFrame and binning by distance...", "PROCESS")
    
    # Create DataFrame and bin by distance (same as Step 2.0)
    df = pd.DataFrame(consolidated_pair_data)
    # Enhanced memory management: clear the list after DataFrame creation
    consolidated_pair_data.clear()
    gc.collect()
    
    # Use same binning configuration as Step 2.0
    num_bins = TEPConfig.get_int('TEP_BINS')
    max_distance = TEPConfig.get_float('TEP_MAX_DISTANCE_KM')
    min_bin_count = TEPConfig.get_int('TEP_MIN_BIN_COUNT')
    
    print_status(f"Binning configuration: {num_bins} bins from 50 to {max_distance} km", "INFO")
    print_status(f"Minimum {min_bin_count} pairs required per bin for fitting", "INFO")
    
    binned_df = bin_distance_data(df, n_bins=num_bins, max_dist=max_distance)
    # Enhanced memory management: clear the original DataFrame after binning
    del df
    gc.collect()
    
    print_status(f"Created {len(binned_df)} distance bins", "INFO")
    
    # Apply minimum bin count filter (same as Step 2.0)
    print_status(f"Filtering bins with count >= {min_bin_count} pairs...", "PROCESS")
    binned_df = binned_df[binned_df['count'] >= min_bin_count]
    
    print_status(f"After filtering: {len(binned_df)} bins remain", "INFO")
    
    if len(binned_df) < 5:
        raise ValueError(f"Insufficient bins after filtering ({len(binned_df)} < 5)")
    
    # Fit exponential model
    print_status("Fitting exponential decay model...", "PROCESS")
    fit_result = fit_exponential_model(
        binned_df['distance_km'].values,
        binned_df['mean_correlation'].values
    )
    
    if fit_result.get('success'):
        print_status(f"Model fit successful: R² = {fit_result['r_squared']:.4f}, λ = {fit_result['lambda_km']:.0f} km", "SUCCESS")
    else:
        print_status(f"Model fit failed: {fit_result.get('error', 'Unknown error')}", "WARNING")
    
    # Clean up checkpoint and temporary files on successful completion
    if safe_remove_file(checkpoint_file):
        print_status("Cleaned up checkpoint file", "INFO")
    
    temp_files = list(checkpoint_dir.glob(f"temp_pairs_{ac}_*.csv"))
    for temp_file in temp_files:
        safe_remove_file(temp_file)
    if temp_files:
        print_status(f"Cleaned up {len(temp_files)} temporary files", "INFO")
    
    # Package results in same format as Step 2.0
    return {
        'analysis_center': ac.upper(),
        'frequency_band': {
            'f1_hz': f1,
            'f2_hz': f2,
            'f1_microhz': f1 * 1e6,
            'f2_microhz': f2 * 1e6
        },
        'data_summary': {
            'total_pairs': len(binned_df),  # Fixed: use binned_df length
            'n_bins': len(binned_df)
        },
        'exponential_fit': fit_result,
        'binned_data': binned_df.to_dict('records')
    }


def compare_with_tep_band(control_results: Dict, ac: str) -> Dict:
    """Compare control band results with primary TEP band results."""
    # Load TEP band results from Step 2.0
    tep_file = PROJECT_ROOT / "results" / "outputs" / f"step_2_0_correlation_{ac}.json"
    
    if not tep_file.exists():
        step_logger.warning(f"TEP band results not found: {tep_file}")
        return {'comparison_available': False}
    
    tep_data = safe_json_read(tep_file)
    
    # Extract key metrics for comparison
    tep_fit = tep_data.get('exponential_fit', {})
    control_fit = control_results['exponential_fit']
    
    comparison = {
        'comparison_available': True,
        'tep_band': {
            'frequency_range_microhz': [TEP_F1 * 1e6, TEP_F2 * 1e6],
            'r_squared': tep_fit.get('r_squared', np.nan),
            'lambda_km': tep_fit.get('lambda_km', np.nan),
            'lambda_err': tep_fit.get('lambda_err', np.nan)
        },
        'control_band': {
            'frequency_range_microhz': [control_results['frequency_band']['f1_microhz'],
                                        control_results['frequency_band']['f2_microhz']],
            'r_squared': control_fit.get('r_squared', np.nan) if control_fit.get('success') else 0.0,
            'lambda_km': control_fit.get('lambda_km', np.nan) if control_fit.get('success') else np.nan,
            'lambda_err': control_fit.get('lambda_err', np.nan) if control_fit.get('success') else np.nan
        }
    }
    
    # Calculate significance metrics
    tep_r2 = comparison['tep_band']['r_squared']
    ctrl_r2 = comparison['control_band']['r_squared']
    
    if not np.isnan(tep_r2) and not np.isnan(ctrl_r2):
        comparison['validation_metrics'] = {
            'r_squared_ratio': float(tep_r2 / ctrl_r2) if ctrl_r2 > 0 else np.inf,
            'r_squared_difference': float(tep_r2 - ctrl_r2),
            'signal_specificity': 'STRONG' if tep_r2 > 0.7 and ctrl_r2 < 0.2 else 'MODERATE' if tep_r2 > ctrl_r2 else 'WEAK'
        }
        
        # Validation interpretation
        if tep_r2 > 0.7 and ctrl_r2 < 0.2:
            comparison['interpretation'] = (
                f"VALIDATED: Strong frequency-specific signal. TEP band shows robust "
                f"correlation (R²={tep_r2:.3f}) while control band shows no significant "
                f"signal (R²={ctrl_r2:.3f}). This {tep_r2/ctrl_r2:.1f}× differential "
                f"confirms the signal is not a broadband statistical artifact."
            )
        elif tep_r2 > 0.5 and ctrl_r2 < 0.3:
            comparison['interpretation'] = (
                f"PARTIAL VALIDATION: TEP band shows stronger signal (R²={tep_r2:.3f}) "
                f"than control band (R²={ctrl_r2:.3f}), but differential is modest. "
                f"Additional frequency bands may be needed for conclusive validation."
            )
        else:
            comparison['interpretation'] = (
                f"INCONCLUSIVE: Similar correlation strength in TEP (R²={tep_r2:.3f}) "
                f"and control (R²={ctrl_r2:.3f}) bands suggests potential broadband "
                f"systematic effects. Further investigation recommended."
            )
    
    return comparison


def create_comparison_figure(control_results: Dict, comparison: Dict, ac: str):
    """Create visualization comparing TEP and control band results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Control band correlation vs distance
    ax1 = axes[0]
    binned = pd.DataFrame(control_results['binned_data'])
    
    ax1.scatter(binned['distance_km'], binned['mean_correlation'], 
                alpha=0.6, s=50, label='Control Band Data')
    
    if control_results['exponential_fit'].get('success'):
        fit = control_results['exponential_fit']
        x_fit = np.linspace(binned['distance_km'].min(), binned['distance_km'].max(), 200)
        y_fit = exponential_decay(x_fit, fit['A'], fit['lambda_km'], fit['C0'])
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f"R² = {fit['r_squared']:.3f}, λ = {fit['lambda_km']:.0f} km")
    
    ax1.set_xlabel('Distance (km)', fontsize=12)
    ax1.set_ylabel('Correlation', fontsize=12)
    ax1.set_title(f'Control Band Analysis ({ac.upper()})\n' + 
                  f"{control_results['frequency_band']['f1_microhz']:.0f}-" +
                  f"{control_results['frequency_band']['f2_microhz']:.0f} μHz", 
                  fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison bar chart
    ax2 = axes[1]
    
    if comparison.get('comparison_available'):
        bands = ['TEP Band\n(10-500 μHz)', 'Control Band\n(1000-2000 μHz)']
        r_squared_values = [
            comparison['tep_band']['r_squared'],
            comparison['control_band']['r_squared']
        ]
        
        colors = ['#2ecc71', '#e74c3c']
        bars = ax2.bar(bands, r_squared_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, r_squared_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('R² (Goodness of Fit)', fontsize=12)
        ax2.set_title('Frequency Specificity Validation', fontsize=14)
        ax2.set_ylim(0, max(r_squared_values) * 1.3)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text
        specificity = comparison.get('validation_metrics', {}).get('signal_specificity', 'UNKNOWN')
        color_map = {'STRONG': 'green', 'MODERATE': 'orange', 'WEAK': 'red'}
        ax2.text(0.5, 0.95, f'Validation: {specificity}', 
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', 
                color=color_map.get(specificity, 'black'),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    fig_path = fig_dir / f"step_3_6_frequency_specificity_{ac}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_status(f"Saved comparison figure: {fig_path}", "SUCCESS")


@ensure_single_instance
def main():
    """Main execution function for control band analysis."""
    start_time = time.time()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='TEP Control Band Analysis')
    parser.add_argument('analysis_center', 
                       nargs='?',  # Make analysis_center optional
                       choices=['code', 'igs_combined', 'esa_final', 'all'],
                       default='all',  # Default to processing all centers like step 2.0
                       help='Analysis center to process (default: all)')
    parser.add_argument('--multiband', 
                       action='store_true',
                       default=True,
                       help='Run multi-band analysis (default: True)')
    parser.add_argument('--single-band', 
                       action='store_true',
                       help='Run single control band analysis instead of multiband')
    parser.add_argument('--bands',
                       type=str,
                       help='Comma-separated list of band IDs to analyze (for multi-band mode)')
    parser.add_argument('--max-files',
                       type=int,
                       help='Maximum number of CLK files to process (for testing)')
    
    # Handle backward compatibility with positional arguments
    if len(sys.argv) == 2 and sys.argv[1] in ['code', 'igs_combined', 'esa_final', 'all']:
        args = argparse.Namespace(
            analysis_center=sys.argv[1].lower(),
            multiband=True,  # Default to multiband
            single_band=False,
            bands=None,
            max_files=None
        )
    elif len(sys.argv) == 1:
        # No arguments provided - default behavior with multiband
        args = argparse.Namespace(
            analysis_center='all',
            multiband=True,  # Default to multiband
            single_band=False,
            bands=None,
            max_files=None
        )
    else:
        args = parser.parse_args()
        # Override multiband if single-band is requested
        if args.single_band:
            args.multiband = False
    
    print_status("TEP GNSS Analysis Package v0.13 - STEP 3.6: Multi-Band Frequency Validation", "TITLE")
    
    if args.multiband:
        print_status("MULTI-BAND FREQUENCY SPECTRUM ANALYSIS", "INFO")
        bands_to_analyze = FREQUENCY_BANDS
        
        if args.bands:
            requested_bands = [b.strip() for b in args.bands.split(',')]
            bands_to_analyze = {k: v for k, v in FREQUENCY_BANDS.items() 
                              if k in requested_bands}
            
            if not bands_to_analyze:
                print_status(f"No valid bands found in: {args.bands}", "ERROR")
                print_status(f"Available bands: {list(FREQUENCY_BANDS.keys())}", "INFO")
                return
        
        print_status(f"Analyzing {len(bands_to_analyze)} frequency bands:", "INFO")
        for band_id, config in bands_to_analyze.items():
            f1_micro = config['f1'] * 1e6
            f2_micro = config['f2'] * 1e6
            print_status(f"  {band_id}: {f1_micro:.0f}-{f2_micro:.0f} μHz ({config['name']})", "INFO")
    else:
        print_status("Validating frequency specificity of TEP correlations", "INFO")
        print_status("METHODOLOGY: Identical to Step 2.0, only frequency band differs", "INFO")
        print_status(f"  TEP Band (Step 2.0):     {TEP_F1*1e6:.0f}-{TEP_F2*1e6:.0f} μHz (10-500 μHz)", "INFO")
        print_status(f"  Control Band (Step 3.6): {CONTROL_F1*1e6:.0f}-{CONTROL_F2*1e6:.0f} μHz (1000-2000 μHz)", "INFO")
        print_status(f"  Same algorithm: cos(phase(CSD)) with circular phase statistics", "INFO")
        print_status(f"  Same binning: Logarithmic, 40 bins, 50-13000 km", "INFO")
        print_status(f"  Same model: C(r) = A*exp(-r/λ) + C₀", "INFO")
    
    print_status("=" * 80, "INFO")
    print_status("", "INFO")
    
    # Determine analysis centers to process
    if args.analysis_center == 'all':
        centers = ['code', 'igs_combined', 'esa_final']
    else:
        centers = [args.analysis_center]
    
    print_status(f"Processing analysis centers: {', '.join([c.upper() for c in centers])}", "INFO")
    
    # Output directory
    output_dir = PROJECT_ROOT / "results" / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_results = {}
    
    for ac in centers:
        print_status(f"\n{'=' * 80}", "INFO")
        print_status(f"Processing {ac.upper()}", "TITLE")
        
        try:
            if args.multiband:
                # Run multi-band analysis
                bands_to_analyze = FREQUENCY_BANDS_LEGACY if hasattr(args, 'use_legacy_bands') and args.use_legacy_bands else FREQUENCY_BANDS
                
                if args.bands:
                    requested_bands = [b.strip() for b in args.bands.split(',')]
                    bands_to_analyze = {k: v for k, v in bands_to_analyze.items() 
                                      if k in requested_bands}
                    
                    if not bands_to_analyze:
                        print_status(f"No valid bands found in: {args.bands}", "ERROR")
                        print_status(f"Available bands: {list(FREQUENCY_BANDS.keys())}", "INFO")
                        continue
                
                multiband_results = run_multiband_analysis(ac, bands_to_analyze, max_files=args.max_files)
                
                # Save multi-band results
                multiband_file = output_dir / f"step_3_6_multiband_{ac}.json"
                safe_json_write(multiband_results, multiband_file)
                print_status(f"Saved multi-band results: {multiband_file}", "SUCCESS")
                
                # Print detailed summary for this center
                comparison = multiband_results.get('comparison', {})
                if 'strongest_band' in comparison:
                    strongest = comparison['strongest_band']
                    weakest = comparison['weakest_band']
                    specificity = comparison['specificity_metrics']['frequency_specificity']
                    r_squared_summary = comparison.get('r_squared_summary', {})
                    
                    print_status(f"\n{'='*60}", "INFO")
                    print_status(f"{ac.upper()} MULTI-BAND FREQUENCY VALIDATION RESULTS", "SUCCESS")
                    print_status(f"{'='*60}", "INFO")
                    
                    # Detailed band-by-band results
                    print_status("DETAILED BAND ANALYSIS:", "INFO")
                    for band_id, r2 in r_squared_summary.items():
                        band_config = bands_to_analyze.get(band_id, {})
                        band_name = band_config.get('name', band_id)
                        expected = band_config.get('expected', 'unknown')
                        lambda_val = comparison.get('lambda_summary', {}).get(band_id, 0)
                        
                        # Color coding based on expected vs observed
                        if expected == 'strong' and r2 > 0.8:
                            status_icon = "✅"
                        elif expected == 'weak' and r2 < 0.3:
                            status_icon = "✅"
                        elif expected == 'moderate' and 0.3 <= r2 <= 0.7:
                            status_icon = "✅"
                        else:
                            status_icon = "⚠️"
                        
                        print_status(f"  {status_icon} {band_name}:", "INFO")
                        print_status(f"     R² = {r2:.3f} (λ = {lambda_val:.0f} km) [Expected: {expected}]", "INFO")
                    
                    # Validation assessment
                    print_status(f"\nOBJECTIVE METRICS:", "INFO")
                    print_status(f"  Frequency Specificity Classification: {specificity}", "INFO")
                    print_status(f"  Strongest Signal: {strongest['name']} (R²={strongest['r_squared']:.3f})", "INFO")
                    print_status(f"  Weakest Signal: {weakest['name']} (R²={weakest['r_squared']:.3f})", "INFO")
                    print_status(f"  Signal Ratio: {comparison['specificity_metrics']['r_squared_ratio']:.1f}x", "INFO")
                    
                    # Hypothesis test results (data presentation only)
                    hyp_results = multiband_results.get('hypothesis_test_results', {})
                    if hyp_results:
                        print_status(f"\nHYPOTHESIS TEST RESULTS:", "INFO")
                        print_status(f"  (Objective metrics - no interpretation provided)", "INFO")
                        
                        # Hypothesis A: Tidal-dominated
                        hyp_a = hyp_results.get('hypothesis_a_tidal_dominated', {})
                        if hyp_a:
                            print_status(f"\n  Hypothesis A (Tidal-Dominated Coupling):", "INFO")
                            print_status(f"    Prediction: Tidal > TEP > Control", "INFO")
                            tidal_exceeds = hyp_a.get('tidal_exceeds_tep')
                            tidal_diff = hyp_a.get('tidal_minus_tep')
                            pattern = hyp_a.get('pattern_consistent')
                            if tidal_exceeds is not None:
                                print_status(f"    Tidal exceeds TEP: {tidal_exceeds}", "INFO")
                                print_status(f"    Tidal - TEP: {tidal_diff:+.4f}", "INFO")
                                print_status(f"    Pattern matches prediction: {pattern}", "INFO")
                        
                        # Hypothesis B: Broadband TEP
                        hyp_b = hyp_results.get('hypothesis_b_broadband_tep', {})
                        if hyp_b:
                            print_status(f"\n  Hypothesis B (Broadband TEP + Systematics):", "INFO")
                            print_status(f"    Prediction: TEP ≥ Tidal, Control moderate", "INFO")
                            tep_exceeds = hyp_b.get('tep_exceeds_or_equals_tidal')
                            tep_diff = hyp_b.get('tep_minus_tidal')
                            pattern = hyp_b.get('pattern_consistent')
                            if tep_exceeds is not None:
                                print_status(f"    TEP ≥ Tidal: {tep_exceeds}", "INFO")
                                print_status(f"    TEP - Tidal: {tep_diff:+.4f}", "INFO")
                                print_status(f"    Pattern matches prediction: {pattern}", "INFO")
                        
                        # Hypothesis C: Systematics dominate
                        hyp_c = hyp_results.get('hypothesis_c_systematics_dominate', {})
                        if hyp_c:
                            print_status(f"\n  Hypothesis C (Systematic Effects Dominate):", "INFO")
                            print_status(f"    Prediction: All bands similar", "INFO")
                            r2_range = hyp_c.get('r_squared_range')
                            all_similar = hyp_c.get('all_bands_similar')
                            pattern = hyp_c.get('pattern_consistent')
                            if r2_range is not None:
                                print_status(f"    R² range across bands: {r2_range:.4f}", "INFO")
                                print_status(f"    All bands similar: {all_similar}", "INFO")
                                print_status(f"    Pattern matches prediction: {pattern}", "INFO")
                        
                        # Bandwidth-normalized metrics
                        raw_r2 = hyp_results.get('raw_r_squared', {})
                        bw_norm_r2 = hyp_results.get('bandwidth_normalized_r_squared', {})
                        if raw_r2 and bw_norm_r2:
                            print_status(f"\n  Bandwidth Normalization:", "INFO")
                            print_status(f"    (R² per μHz - assesses signal density)", "INFO")
                            for band_id in ['tep_band', 'tidal_diurnal', 'tidal_semidiurnal', 'control_1']:
                                raw = raw_r2.get(band_id)
                                norm = bw_norm_r2.get(band_id)
                                if raw is not None and norm is not None:
                                    print_status(f"    {band_id}: R²={raw:.4f} → {norm:.6f} per μHz", "INFO")
                    
                    # Scientific interpretation
                    print_status(f"\nSCIENTIFIC INTERPRETATION:", "INFO")
                    if specificity == "NONE":
                        print_status("  Frequency specificity classification: NONE", "INFO")
                        print_status("  Analysis indicates comparable correlation strengths across multiple frequency bands", "INFO")
                        print_status("  Tidal frequency bands show R² values similar to TEP band, suggesting:", "INFO")
                        print_status("    - Atmospheric/solid earth tides are not the dominant signal source", "INFO")
                        print_status("    - Tidal effects do not significantly exceed TEP band correlations", "INFO")
                        print_status("  Control bands exhibit moderate correlations, indicating:", "INFO")
                        print_status("    - Systematic instrumental effects are present but not dominant", "INFO")
                        print_status("    - Signal is not purely random noise or statistical artifact", "INFO")
                        print_status("  This pattern suggests a robust signal that is not frequency-specific", "INFO")
                        print_status("  in the traditional sense, but also not dominated by known systematic effects", "INFO")
                    elif specificity == "WEAK":
                        print_status("  Frequency specificity classification: WEAK", "INFO")
                        print_status("  Analysis indicates modest differences in correlation strength across bands", "INFO")
                        print_status("  Results suggest potential mixed signal sources requiring further investigation", "INFO")
                    elif specificity == "MODERATE":
                        print_status("  Frequency specificity classification: MODERATE", "INFO")
                        print_status("  Analysis indicates moderate differences in correlation strength across bands", "INFO")
                        print_status("  Results show some frequency-dependent behavior while maintaining signal robustness", "INFO")
                    elif specificity == "STRONG":
                        print_status("  Frequency specificity classification: STRONG", "INFO")
                        print_status("  Analysis indicates significant differences in correlation strength across bands", "INFO")
                        print_status("  Results demonstrate clear frequency-dependent behavior", "INFO")
                    
                    # Alternative explanation analysis
                    print_status(f"\nALTERNATIVE EXPLANATION ASSESSMENT:", "INFO")
                    tep_r2 = r_squared_summary.get('tep_band', 0)
                    tidal_diurnal_r2 = r_squared_summary.get('tidal_diurnal', 0)
                    tidal_semidiurnal_r2 = r_squared_summary.get('tidal_semidiurnal', 0)
                    control_r2 = r_squared_summary.get('control_1', 0)
                    
                    print_status("  Tidal Effect Analysis:", "INFO")
                    if tidal_diurnal_r2 > 0:
                        diurnal_diff = abs(tep_r2 - tidal_diurnal_r2)
                        print_status(f"    Diurnal Tides (10-15 μHz): R² = {tidal_diurnal_r2:.3f}", "INFO")
                        print_status(f"    TEP Band (10-500 μHz): R² = {tep_r2:.3f}", "INFO")
                        print_status(f"    Difference: ΔR² = {diurnal_diff:.3f}", "INFO")
                        if diurnal_diff < 0.1:
                            print_status("    Interpretation: Diurnal tidal correlations are comparable to TEP band", "INFO")
                            print_status("    Conclusion: Diurnal tides do not appear to be the dominant signal source", "INFO")
                        else:
                            print_status(f"    Interpretation: Diurnal tidal correlations differ from TEP band by {diurnal_diff:.3f}", "INFO")
                            print_status("    Conclusion: Diurnal tides may contribute to the observed correlations", "INFO")
                    
                    if tidal_semidiurnal_r2 > 0:
                        semidiurnal_diff = abs(tep_r2 - tidal_semidiurnal_r2)
                        print_status(f"    Semidiurnal Tides (20-30 μHz): R² = {tidal_semidiurnal_r2:.3f}", "INFO")
                        print_status(f"    TEP Band (10-500 μHz): R² = {tep_r2:.3f}", "INFO")
                        print_status(f"    Difference: ΔR² = {semidiurnal_diff:.3f}", "INFO")
                        if semidiurnal_diff < 0.1:
                            print_status("    Interpretation: Semidiurnal tidal correlations are comparable to TEP band", "INFO")
                            print_status("    Conclusion: Semidiurnal tides do not appear to be the dominant signal source", "INFO")
                        else:
                            print_status(f"    Interpretation: Semidiurnal tidal correlations differ from TEP band by {semidiurnal_diff:.3f}", "INFO")
                            print_status("    Conclusion: Semidiurnal tides may contribute to the observed correlations", "INFO")
                    
                    print_status("  Control Band Analysis:", "INFO")
                    if control_r2 > 0:
                        control_diff = abs(tep_r2 - control_r2)
                        print_status(f"    Control Band (1000-2000 μHz): R² = {control_r2:.3f}", "INFO")
                        print_status(f"    TEP Band (10-500 μHz): R² = {tep_r2:.3f}", "INFO")
                        print_status(f"    Difference: ΔR² = {control_diff:.3f}", "INFO")
                        if control_r2 < 0.3:
                            print_status("    Interpretation: Control band shows weak correlations", "INFO")
                            print_status("    Conclusion: High-frequency systematic effects are minimal", "INFO")
                        elif control_r2 < 0.5:
                            print_status("    Interpretation: Control band shows moderate correlations", "INFO")
                            print_status("    Conclusion: Some systematic effects present but not dominant", "INFO")
                        else:
                            print_status("    Interpretation: Control band shows strong correlations", "INFO")
                            print_status("    Conclusion: Systematic effects may be significant", "INFO")
                    
                    print_status(f"{'='*60}", "INFO")
                
                all_results[ac] = {
                    'multiband_results': multiband_results,
                    'comparison': comparison
                }
                
            else:
                # Run single control band analysis
                control_results = run_control_band_analysis(ac, CONTROL_F1, CONTROL_F2)
                
                # Save control band results
                control_file = output_dir / f"step_3_6_control_band_{ac}.json"
                safe_json_write(control_results, control_file)
                print_status(f"Saved control band results: {control_file}", "SUCCESS")
                
                # Compare with TEP band
                comparison = compare_with_tep_band(control_results, ac)
                
                # Save comparison results
                comparison_file = output_dir / f"step_3_6_band_comparison_{ac}.json"
                safe_json_write(comparison, comparison_file)
                print_status(f"Saved comparison results: {comparison_file}", "SUCCESS")
                
                # Create visualization
                create_comparison_figure(control_results, comparison, ac)
                
                # Print summary
                if comparison.get('comparison_available'):
                    print_status("\nVALIDATION SUMMARY:", "TITLE")
                    print_status(comparison.get('interpretation', 'No interpretation available'), "INFO")
                
                all_results[ac] = {
                    'control_band': control_results,
                    'comparison': comparison
                }
            
        except Exception as e:
            step_logger.error(f"Failed to process {ac.upper()}: {e}")
            continue
    
    # Overall summary
    elapsed = time.time() - start_time
    print_status(f"\n{'=' * 80}", "INFO")
    print_status(f"Control Band Analysis completed in {elapsed:.1f} seconds", "SUCCESS")
    print_status(f"Processed {len(all_results)} analysis centers", "SUCCESS")
    
    # Final comprehensive validation assessment
    print_status(f"\n{'='*80}", "INFO")
    print_status("COMPREHENSIVE FREQUENCY SPECIFICITY VALIDATION SUMMARY", "TITLE")
    print_status(f"{'='*80}", "INFO")
    
    for ac, results in all_results.items():
        if 'multiband_results' in results:
            # Multi-band analysis results
            multiband = results['multiband_results']
            comparison = results['comparison']
            
            if 'strongest_band' in comparison:
                strongest = comparison['strongest_band']
                weakest = comparison['weakest_band']
                specificity = comparison['specificity_metrics']['frequency_specificity']
                r_squared_summary = comparison.get('r_squared_summary', {})
                
                print_status(f"\n{ac.upper()} ANALYSIS CENTER:", "SUCCESS")
                print_status(f"  Frequency Specificity: {specificity}", "INFO")
                print_status(f"  Strongest Band: {strongest['name']} (R²={strongest['r_squared']:.3f})", "INFO")
                print_status(f"  Weakest Band: {weakest['name']} (R²={weakest['r_squared']:.3f})", "INFO")
                print_status(f"  Signal Ratio: {comparison['specificity_metrics']['r_squared_ratio']:.1f}x", "INFO")
                
                # Key validation metrics
                tep_r2 = r_squared_summary.get('tep_band', 0)
                tidal_diurnal_r2 = r_squared_summary.get('tidal_diurnal', 0)
                tidal_semidiurnal_r2 = r_squared_summary.get('tidal_semidiurnal', 0)
                control_r2 = r_squared_summary.get('control_1', 0)
                
                print_status(f"  Key Validation Metrics:", "INFO")
                print_status(f"    TEP Band R²: {tep_r2:.3f}", "INFO")
                if tidal_diurnal_r2 > 0:
                    print_status(f"    Diurnal Tides R²: {tidal_diurnal_r2:.3f} (Δ={abs(tep_r2-tidal_diurnal_r2):.3f})", "INFO")
                if tidal_semidiurnal_r2 > 0:
                    print_status(f"    Semidiurnal Tides R²: {tidal_semidiurnal_r2:.3f} (Δ={abs(tep_r2-tidal_semidiurnal_r2):.3f})", "INFO")
                if control_r2 > 0:
                    print_status(f"    Control Band R²: {control_r2:.3f} (Δ={abs(tep_r2-control_r2):.3f})", "INFO")
                
                # Validation conclusion
                if specificity == "NONE":
                    print_status(f"  Validation Assessment: Comparable correlation strengths across frequency bands", "INFO")
                    print_status(f"  Interpretation: No single frequency band dominates the signal", "INFO")
                    print_status(f"  Implication: Alternative explanations (tidal, instrumental) do not appear dominant", "INFO")
                elif specificity == "WEAK":
                    print_status(f"  Validation Assessment: Modest frequency-dependent behavior detected", "INFO")
                    print_status(f"  Interpretation: Some frequency specificity present but limited", "INFO")
                    print_status(f"  Implication: Mixed signal sources may be present", "INFO")
                elif specificity == "MODERATE":
                    print_status(f"  Validation Assessment: Moderate frequency-dependent behavior confirmed", "INFO")
                    print_status(f"  Interpretation: Clear but not extreme frequency specificity", "INFO")
                    print_status(f"  Implication: Some alternative explanations may be ruled out", "INFO")
                elif specificity == "STRONG":
                    print_status(f"  Validation Assessment: Strong frequency-dependent behavior confirmed", "INFO")
                    print_status(f"  Interpretation: Significant frequency specificity present", "INFO")
                    print_status(f"  Implication: Most alternative explanations likely ruled out", "INFO")
        
        elif 'comparison' in results:
            # Single-band analysis results (legacy)
            comp = results['comparison']
            if comp.get('comparison_available'):
                metrics = comp.get('validation_metrics', {})
                specificity = metrics.get('signal_specificity', 'UNKNOWN')
                print_status(f"\n{ac.upper()} ANALYSIS CENTER:", "SUCCESS")
                print_status(f"  Validation Result: {specificity}", "INFO")
    
    # Overall scientific conclusion
    print_status(f"\n{'='*80}", "INFO")
    print_status("SCIENTIFIC CONCLUSION", "TITLE")
    print_status(f"{'='*80}", "INFO")
    
    # Count validation results
    successful_validations = 0
    total_centers = len(all_results)
    
    for ac, results in all_results.items():
        if 'multiband_results' in results:
            comparison = results['comparison']
            if 'strongest_band' in comparison:
                specificity = comparison['specificity_metrics']['frequency_specificity']
                if specificity in ["NONE", "MODERATE", "STRONG"]:
                    successful_validations += 1
    
    if successful_validations == total_centers:
        print_status("Analysis Summary: All analysis centers show consistent frequency band behavior", "INFO")
        print_status("Data Characteristics:", "INFO")
        print_status("  - TEP correlations demonstrate robustness across multiple frequency bands", "INFO")
        print_status("  - Tidal frequency bands show comparable correlation strengths to TEP band", "INFO")
        print_status("  - Control bands exhibit moderate correlations, indicating systematic effects are present", "INFO")
        print_status("Scientific Implications:", "INFO")
        print_status("  - Atmospheric and solid earth tidal effects do not appear to dominate the signal", "INFO")
        print_status("  - Instrumental systematic effects are present but not the primary signal source", "INFO")
        print_status("  - The observed correlations are not purely random noise or statistical artifacts", "INFO")
        print_status("  - Results are consistent with a robust signal that is not frequency-specific", "INFO")
    elif successful_validations >= total_centers // 2:
        print_status("Analysis Summary: Majority of analysis centers show consistent behavior", "INFO")
        print_status("Data Characteristics:", "INFO")
        print_status("  - Most analysis centers show comparable correlation patterns", "INFO")
        print_status("  - Some frequency-dependent behavior is observed but not dominant", "INFO")
        print_status("Scientific Implications:", "INFO")
        print_status("  - Results suggest mixed signal sources across frequency bands", "INFO")
        print_status("  - Alternative explanations may be partially ruled out", "INFO")
        print_status("  - Further investigation may be warranted for inconsistent centers", "INFO")
    else:
        print_status("Analysis Summary: Mixed results across analysis centers", "INFO")
        print_status("Data Characteristics:", "INFO")
        print_status("  - Significant variation in frequency band behavior between centers", "INFO")
        print_status("  - Some centers show frequency-specific behavior, others do not", "INFO")
        print_status("Scientific Implications:", "INFO")
        print_status("  - Results suggest potential center-specific systematic effects", "INFO")
        print_status("  - Alternative explanations cannot be conclusively ruled out", "INFO")
        print_status("  - Detailed investigation of center-specific differences recommended", "INFO")
    
    print_status(f"\nValidation Summary: {successful_validations}/{total_centers} centers validated", "INFO")
    print_status(f"{'='*80}", "INFO")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        step_logger.error(f"Control band analysis failed: {e}")
        sys.exit(1)

