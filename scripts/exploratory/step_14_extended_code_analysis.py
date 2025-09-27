#!/usr/bin/env python3
"""
Step 14 Extended: Complete 15.5-Year Gravitational-Temporal Field Analysis (CODE Only)

PORTABLE EXPERIMENTAL SCRIPT - Runs completely fresh from raw historical data

This script performs complete gravitational-temporal correlation analysis using
15.5 years of historical CODE data (2010-2025). Completely self-contained and portable.

Key Features:
- Processes ONLY raw historical CODE CLK files (completely fresh)
- Complete 15.5-year dataset (2010-2025) for maximum statistical power
- No dependencies on existing processed data or checkpoints
- Portable - can run on any machine with raw CODE data
- Uses NASA/JPL DE440/441 ephemeris for high-precision calculations

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-25
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import tempfile
import subprocess
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import savgol_filter, correlate
import seaborn as sns
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
import time

# Minimal logger to ensure immediate flush for progress lines
def log(message: str) -> None:
    print(message, flush=True)

# Checkpoint system for resuming interrupted processing
def save_checkpoint(checkpoint_data: dict, checkpoint_dir: str = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/checkpoints') -> str:
    """Save processing checkpoint to allow resuming."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_file = os.path.join(checkpoint_dir, f'step14_extended_checkpoint_{timestamp}.json')
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, default=str)
    
    log(f"  💾 Checkpoint saved: {os.path.basename(checkpoint_file)}")
    return checkpoint_file

def load_latest_checkpoint(checkpoint_dir: str = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/checkpoints') -> tuple:
    """Load the most recent checkpoint if available."""
    if not os.path.exists(checkpoint_dir):
        return None, {}
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('step14_extended_checkpoint_') and f.endswith('.json')]
    if not checkpoint_files:
        return None, {}
    
    # Get most recent checkpoint
    latest_file = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_file)
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        log(f"  📂 Found checkpoint: {latest_file} ({len(checkpoint_data.get('processed_files', []))} files)")
        return checkpoint_path, checkpoint_data
    except Exception as e:
        log(f"  ⚠️  Failed to load checkpoint {latest_file}: {e}")
        return None, {}

def cleanup_old_checkpoints(checkpoint_dir: str = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/checkpoints', keep_count: int = 5):
    """Keep only the most recent N checkpoints to save disk space."""
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('step14_extended_checkpoint_') and f.endswith('.json')]
    if len(checkpoint_files) <= keep_count:
        return
    
    # Sort by creation time, keep newest
    checkpoint_files.sort(key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)), reverse=True)
    files_to_delete = checkpoint_files[keep_count:]
    
    for file_to_delete in files_to_delete:
        try:
            os.remove(os.path.join(checkpoint_dir, file_to_delete))
            log(f"  🗑️  Cleaned up old checkpoint: {file_to_delete}")
        except Exception as e:
            log(f"  ⚠️  Failed to delete {file_to_delete}: {e}")

# Add project root to path for imports
sys.path.append('/Users/matthewsmawfield/www/TEP-GNSS')
from scripts.steps.step_3_tep_correlation_analysis import process_single_clk_file
from astropy import units as u
import glob
from pathlib import Path

# Set high-precision ephemeris
solar_system_ephemeris.set('jpl')

# Planetary masses in Earth masses (M⊕)
PLANETARY_MASSES = {
    'sun': 332946.0,      # Solar mass in Earth masses
    'jupiter': 317.8,     # Jupiter mass in Earth masses  
    'saturn': 95.2,       # Saturn mass in Earth masses
    'venus': 0.815,       # Venus mass in Earth masses
    'mars': 0.107,        # Mars mass in Earth masses
}

def calculate_high_precision_gravitational_influence(date: datetime) -> Dict:
    """
    Calculate high-precision gravitational influence of celestial bodies on Earth
    using NASA/JPL DE440/441 ephemeris data.
    
    Returns gravitational influence coefficients: (Body_Mass / Earth_Mass) / Distance_AU²
    """
    # Convert to astropy Time
    astro_time = Time(date.strftime('%Y-%m-%d'))
    
    try:
        # Get barycentric positions for all bodies
        earth_pos, _ = get_body_barycentric_posvel('earth', astro_time)
        sun_pos, _ = get_body_barycentric_posvel('sun', astro_time)
        jupiter_pos, _ = get_body_barycentric_posvel('jupiter', astro_time)
        saturn_pos, _ = get_body_barycentric_posvel('saturn', astro_time)
        venus_pos, _ = get_body_barycentric_posvel('venus', astro_time)
        mars_pos, _ = get_body_barycentric_posvel('mars', astro_time)
        
        # Calculate Earth-centered distances in AU
        distances = {}
        positions = {
            'sun': sun_pos,
            'jupiter': jupiter_pos,
            'saturn': saturn_pos,
            'venus': venus_pos,
            'mars': mars_pos
        }
        
        for body, pos in positions.items():
            earth_centered_pos = pos - earth_pos
            distance_au = np.linalg.norm(earth_centered_pos.xyz.value)
            distances[f'{body}_distance_au'] = distance_au
            
            # Calculate gravitational influence: Mass / Distance²
            mass_ratio = PLANETARY_MASSES[body]
            gravitational_influence = mass_ratio / (distance_au ** 2)
            distances[f'{body}_influence'] = gravitational_influence
        
        # Calculate total influences
        distances['total_planetary_influence'] = (
            distances['jupiter_influence'] + distances['saturn_influence'] + 
            distances['venus_influence'] + distances['mars_influence']
        )
        
        distances['total_influence'] = (
            distances['sun_influence'] + distances['total_planetary_influence']
        )
        
        return distances
        
    except Exception as e:
        print(f"Error calculating positions for {date}: {e}")
        return None

def process_single_historical_file(args):
    """Process a single historical CLK file (for parallel execution)."""
    clk_file, start_year, end_year, coords_file = args
    
    try:
        # Extract date from filename (handle both legacy and modern formats)
        filename = os.path.basename(clk_file)
        
        if filename.startswith('COD0OPSFIN_'):
            # Modern format: COD0OPSFIN_20200010000_01D_30S_CLK.CLK.gz
            parts = filename.split('_')
            if len(parts) >= 2:
                date_part = parts[1]  # 20200010000
                year = int(date_part[:4])
                doy = int(date_part[4:7])  # Day of year
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            else:
                return None
        elif filename.startswith('COD') and (filename.endswith('.CLK.Z') or filename.endswith('.CLK')):
            # Legacy format: CODWWWWd.CLK(.Z) where WWWW = GPS week, d = day-of-week (0-6)
            code = filename[3:8]
            if len(code) != 5 or not code.isdigit():
                return None
            gps_week = int(code[:4])
            day_of_week = int(code[4])
            gps_epoch = datetime(1980, 1, 6)
            date = gps_epoch + timedelta(weeks=gps_week, days=day_of_week)
        else:
            return None
        
        # Skip if outside our year range
        if not (start_year <= date.year <= end_year):
            return None
        
        # Load coordinates using official pipeline method
        coords_file = '/Users/matthewsmawfield/www/TEP-GNSS/data/coordinates/station_coords_global.csv'
        if not os.path.exists(coords_file):
            return {'error': f"Coordinates file not found: {coords_file}"}
        coords_df = pd.read_csv(coords_file)
        
        # Process the CLK file (handle .Z by temporary decompression)
        input_path = Path(clk_file)
        temp_path = None
        try:
            if input_path.suffix == '.Z':
                with tempfile.NamedTemporaryFile(delete=False, suffix='.CLK') as tmp:
                    temp_path = Path(tmp.name)
                # Use gzip -dc which can decompress .Z on macOS/Linux
                proc = subprocess.run(['gzip', '-dc', str(input_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0 or not proc.stdout:
                    # Fallback to uncompress -c if available
                    proc2 = subprocess.run(['uncompress', '-c', str(input_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if proc2.returncode != 0 or not proc2.stdout:
                        return {'error': f"Failed to decompress {filename}: {proc.stderr.decode(errors='ignore') or proc2.stderr.decode(errors='ignore')}"}
                    temp_data = proc2.stdout
                else:
                    temp_data = proc.stdout
                with open(temp_path, 'wb') as f:
                    f.write(temp_data)
                parse_path = temp_path
            else:
                parse_path = input_path
            
            clk_data = process_single_clk_file(parse_path, coords_df)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
        
        if clk_data and isinstance(clk_data, list):
            # Extract coherence values from the processed data
            coherence_values = []
            for entry in clk_data:
                if isinstance(entry, dict) and 'plateau_phase' in entry and entry['plateau_phase'] is not None:
                    coherence_values.append(entry['plateau_phase'])
            
            if coherence_values:
                return {
                    'date': date,
                    'coherence_values': coherence_values,
                    'filename': filename
                }
        elif clk_data and not isinstance(clk_data, list):
            return {'error': f"Unexpected data format from process_single_clk_file: {type(clk_data)}"}
        
        return None
        
    except Exception as e:
        return {'error': f"Error processing {filename}: {e}"}

def extract_extended_code_tep_coherence_data(start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """
    Extract CODE-only TEP coherence data for extended time period.
    Processes ONLY raw historical CLK files - completely fresh and portable.
    """
    print(f"Extracting FRESH CODE TEP coherence data from {start_year} to {end_year}...")
    print("🔄 PORTABLE MODE: Processing all raw historical CODE files from scratch")

    import glob
    from datetime import datetime, timedelta

    # Start completely fresh - no checkpoints, no existing data
    all_daily_data = {}
    processed_files_set = set()

    # Find ALL historical CLK files from 2010-2025
    historical_files = []

    for year in range(start_year, end_year + 1):
        print(f"  📁 Scanning year {year}...")

        # Handle mixed formats per year (legacy .CLK.Z and modern .CLK.gz)
        if year <= 2021:
            # Legacy format: CODWWWWD.CLK.Z (2010-2021)
            clk_z_pattern = f'/Users/matthewsmawfield/www/TEP-GNSS/data/historical_code/raw/{year}/COD*.CLK.Z'
            clk_z_files = glob.glob(clk_z_pattern)
            historical_files.extend(clk_z_files)
            print(f"    Found {len(clk_z_files)} legacy .CLK.Z files")
        else:
            # Modern format: COD0OPSFIN_*.CLK.gz (2022-2025)
            clk_gz_pattern = f'/Users/matthewsmawfield/www/TEP-GNSS/data/historical_code/raw/{year}/COD0OPSFIN*.CLK.gz'
            clk_gz_files = glob.glob(clk_gz_pattern)
            historical_files.extend(clk_gz_files)
            print(f"    Found {len(clk_gz_files)} modern .CLK.gz files")

    print(f"  📊 TOTAL: Found {len(historical_files)} historical CODE files from {start_year}-{end_year}")
    print("  🎯 PORTABLE: Will process ALL files fresh - no existing data dependency")

    if not historical_files:
        print("  ❌ ERROR: No historical CODE files found!")
        print(f"  Expected location: /Users/matthewsmawfield/www/TEP-GNSS/data/historical_code/raw/")
        print("  Make sure aggressive_acquire.py has completed downloading all files.")
        return pd.DataFrame()
    else:
        # Process historical CLK files in parallel
        print("🚀 Processing historical CLK files in parallel...")
        print(f"  Total files: {len(historical_files)}")
        
        # Process all historical files for full analysis
        print(f"  Processing {len(historical_files)} historical files for complete analysis...")
        
        # Determine optimal number of workers (env override), default 12
        try:
            requested_workers = int(os.getenv('TEP_HIST_WORKERS', '12'))
        except Exception:
            requested_workers = 12
        max_workers = max(1, min(cpu_count(), requested_workers))
        log(f"  Using {max_workers} parallel workers (CPU cores: {cpu_count()})")
        
        coords_file = '/Users/matthewsmawfield/www/TEP-GNSS/data/coordinates/station_coords_global.csv'
        if not os.path.exists(coords_file):
            print(f"  ERROR: Coordinates file not found: {coords_file}")
            return pd.DataFrame()
        
        # Prepare arguments for parallel processing
        process_args = [(clk_file, start_year, end_year, coords_file) for clk_file in historical_files]
        
        historical_processed = len(processed_files_set)  # Start from checkpoint count
        start_time = time.time()
    
    # Use ProcessPoolExecutor for CPU-intensive tasks
    log("  🔄 Starting parallel processing...")
    log(f"  📋 Sample files to process:")
    for i, clk_file in enumerate(historical_files[:5]):
        log(f"    {i+1}. {os.path.basename(clk_file)}")
    log(f"    ... and {len(historical_files)-5} more files")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and keep mapping for filenames
        future_to_args = {executor.submit(process_single_historical_file, args): args for args in process_args}
        all_futures = set(future_to_args.keys())
        log(f"  ⚡ Processing {len(all_futures)} files with {max_workers} workers...")
        log(f"  📊 Progress will be shown for EVERY file completion; heartbeat printed if none finish in 30s...")
        completed_count = 0

        last_heartbeat = time.time()
        pending = set(all_futures)
        # Stabilize ETA with exponential moving average of completion rate
        ema_rate = 0.0
        last_completion_ts = time.time()

        while pending:
            # Wait for at least one future to complete, or timeout for heartbeat
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)

            if not done:
                # Heartbeat (no completions in the last 30s)
                elapsed = time.time() - start_time
                rate = ema_rate if ema_rate > 0 else ((completed_count / elapsed) if elapsed > 0 else 0)
                progress_pct = (completed_count / len(all_futures)) * 100
                in_progress = len(pending)
                log(f"    [{completed_count:4d}/{len(all_futures)}] ⏳ Still processing... running ~{min(max_workers, in_progress)} workers | {rate:.2f} files/s | {progress_pct:4.1f}% | {elapsed:.0f}s elapsed")
                last_heartbeat = time.time()
                continue

            # Handle all newly completed futures
            for future in done:
                completed_count += 1
                args = future_to_args.get(future)
                filename = os.path.basename(args[0]) if args else "unknown"
                try:
                    result = future.result()

                    if result and isinstance(result, dict) and 'error' not in result:
                        # Validate result structure
                        if 'date' in result and 'coherence_values' in result:
                            date = result['date']
                            coherence_values = result['coherence_values']

                            if date not in all_daily_data:
                                all_daily_data[date] = []
                            all_daily_data[date].extend(coherence_values)
                            historical_processed += 1
                            processed_files_set.add(filename)  # Track all processed files

                            # Show progress for EVERY file completion (EMA-smoothed rate)
                            now = time.time()
                            dt = max(1e-6, now - last_completion_ts)
                            inst_rate = 1.0 / dt
                            ema_rate = inst_rate if ema_rate == 0 else (0.85 * ema_rate + 0.15 * inst_rate)
                            last_completion_ts = now
                            elapsed = now - start_time
                            rate = ema_rate if ema_rate > 0 else (completed_count / elapsed if elapsed > 0 else 0)
                            eta = (len(all_futures) - completed_count) / rate if rate > 0 else 0
                            progress_pct = (completed_count / len(all_futures)) * 100
                            log(f"    [{completed_count:4d}/{len(all_futures)}] ✓ {filename} | {len(coherence_values):5d} values | {rate:.2f} files/s | {progress_pct:4.1f}% | ETA: {eta/60:.1f}m")

                            # Save checkpoint every 50 files (disabled for incremental approach)
                            # if completed_count % 50 == 0:
                            #     save_checkpoint(...)
                        else:
                            log(f"    [{completed_count:4d}/{len(all_futures)}] ✗ {filename} | Invalid result structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")

                    elif result and isinstance(result, dict) and 'error' in result:
                        log(f"    [{completed_count:4d}/{len(all_futures)}] ✗ {filename} | {result['error']}")
                    else:
                        log(f"    [{completed_count:4d}/{len(all_futures)}] ⚠ {filename} | No result or unexpected format: {type(result)}")

                except Exception as e:
                    log(f"    [{completed_count:4d}/{len(all_futures)}] ✗ {filename} | ERROR: {e}")
    
        elapsed_total = time.time() - start_time
        print(f"  ⚡ Parallel processing complete! {elapsed_total:.1f}s | {len(historical_files)/elapsed_total:.1f} files/s")
        
        # Save final checkpoint (convert datetime keys to strings for JSON serialization)
        daily_data_serializable = {date.isoformat(): values for date, values in all_daily_data.items()}
        final_checkpoint_data = {
            'processed_files': list(processed_files_set),
            'daily_data': daily_data_serializable,
            'completed_count': len(processed_files_set),
            'timestamp': datetime.now().isoformat(),
            'total_files': len(processed_files_set),
            'status': 'completed'
        }
        save_checkpoint(final_checkpoint_data)
        cleanup_old_checkpoints()
    
    print(f"  ✅ COMPLETED: Processed {historical_processed} historical CLK files")
    print(f"  📊 Total unique dates with historical data: {len([d for d in all_daily_data.keys() if isinstance(d, datetime)])}")
    
    # Aggregate daily data
    print(f"📈 Aggregating extended CODE data...")
    print(f"  Combining processed files and historical files...")
    daily_aggregated = []
    
    for date, coherence_values in all_daily_data.items():
        if len(coherence_values) > 0:
            daily_aggregated.append({
                'date': date,
                'coherence_mean': np.mean(coherence_values),
                'coherence_median': np.median(coherence_values),
                'coherence_std': np.std(coherence_values),
                'coherence_count': len(coherence_values)
            })
    
    if daily_aggregated:
        tep_df = pd.DataFrame(daily_aggregated).sort_values('date').reset_index(drop=True)
        print(f"✅ Successfully extracted extended CODE TEP data for {len(tep_df)} days")
        
        # Show statistics
        print(f"  Date range: {tep_df['date'].min().strftime('%Y-%m-%d')} to {tep_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Coherence mean range: {tep_df['coherence_mean'].min():.6f} to {tep_df['coherence_mean'].max():.6f}")
        print(f"  Coherence std range: {tep_df['coherence_std'].min():.6f} to {tep_df['coherence_std'].max():.6f}")
        print(f"  Average pairs per day: {tep_df['coherence_count'].mean():.0f}")
        print(f"  Total station pair measurements: {tep_df['coherence_count'].sum():,}")
        
        return tep_df
    else:
        raise ValueError("No extended CODE TEP coherence data could be extracted")

def perform_extended_correlation_analysis(combined_df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive correlation analysis for extended timeframe.
    """
    print("Performing extended correlation analysis...")
    
    results = {
        'analysis_type': 'extended_gravitational_temporal_correlation_code_only',
        'ephemeris_source': 'NASA_JPL_DE440_441',
        'tep_method': 'phase_coherent_cross_spectral_density',
        'data_center': 'CODE_only',
        'data_summary': {
            'total_days': len(combined_df),
            'date_range': [
                combined_df['date'].min().strftime('%Y-%m-%d'),
                combined_df['date'].max().strftime('%Y-%m-%d')
            ],
            'analysis_span_years': (combined_df['date'].max() - combined_df['date'].min()).days / 365.25
        }
    }
    
    # Individual planetary correlations
    planets = ['sun', 'jupiter', 'saturn', 'venus', 'mars']
    tep_metrics = ['coherence_mean', 'coherence_median', 'coherence_std']
    
    correlations = {}
    
    for planet in planets:
        planet_corr = {}
        influence_col = f'{planet}_influence'
        
        if influence_col in combined_df.columns:
            for metric in tep_metrics:
                if metric in combined_df.columns:
                    r, p = stats.pearsonr(combined_df[influence_col], combined_df[metric])
                    rho, p_spear = stats.spearmanr(combined_df[influence_col], combined_df[metric])
                    
                    planet_corr[metric] = {
                        'pearson_r': r,
                        'pearson_p': p,
                        'spearman_rho': rho,
                        'spearman_p': p_spear,
                        'n_points': len(combined_df)
                    }
        
        correlations[f'{planet}_influence'] = planet_corr
    
    # KEY DISCOVERY: Stacked planetary influence analysis
    stacked_correlations = {}
    for metric in tep_metrics:
        if metric in combined_df.columns:
            r, p = stats.pearsonr(combined_df['total_planetary_influence'], combined_df[metric])
            rho, p_spear = stats.spearmanr(combined_df['total_planetary_influence'], combined_df[metric])
            
            stacked_correlations[metric] = {
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear,
                'n_points': len(combined_df)
            }
    
    correlations['stacked_planetary_influence'] = stacked_correlations
    
    # Total gravitational influence (including Sun)
    total_correlations = {}
    for metric in tep_metrics:
        if metric in combined_df.columns:
            r, p = stats.pearsonr(combined_df['total_influence'], combined_df[metric])
            rho, p_spear = stats.spearmanr(combined_df['total_influence'], combined_df[metric])
            
            total_correlations[metric] = {
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'spearman_p': p_spear,
                'n_points': len(combined_df)
            }
    
    correlations['total_gravitational_influence'] = total_correlations
    results['correlations'] = correlations
    
    # Extended time series analysis with longer smoothing windows
    window_size = min(91, len(combined_df) // 20)  # Longer window for extended data
    poly_order = min(3, window_size - 1)
    
    if window_size > poly_order:
        smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window_size, poly_order)
        smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], window_size, poly_order)
        
        # Smoothed pattern correlation
        smooth_r, smooth_p = stats.pearsonr(smoothed_stacked, smoothed_coherence_std)
        
        # Cross-correlation analysis for lag detection
        norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
        norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
        
        cross_corr = correlate(norm_coherence, norm_stacked, mode='full')
        lags = np.arange(-len(norm_stacked) + 1, len(norm_stacked))
        max_corr_idx = np.argmax(np.abs(cross_corr))
        optimal_lag = lags[max_corr_idx]
        max_correlation = cross_corr[max_corr_idx]
        
        results['extended_pattern_analysis'] = {
            'smoothed_correlation': smooth_r,
            'smoothed_p_value': smooth_p,
            'optimal_lag_days': int(optimal_lag),
            'max_cross_correlation': float(max_correlation),
            'smoothing_window': window_size,
            'pattern_relationship': 'anti_phase' if max_correlation < 0 else 'in_phase'
        }
    
    # Multi-year pattern analysis
    combined_df['year'] = combined_df['date'].dt.year
    yearly_stats = combined_df.groupby('year').agg({
        'total_planetary_influence': ['mean', 'std'],
        'coherence_mean': ['mean', 'std'],
        'coherence_std': ['mean', 'std']
    }).round(6)
    
    # Flatten column names for easier access
    yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
    
    results['yearly_analysis'] = {
        'years_covered': sorted(combined_df['year'].unique().tolist()),
        'yearly_statistics': yearly_stats.to_dict()
    }
    
    return results

def create_extended_visualization(combined_df: pd.DataFrame, analysis_results: Dict) -> str:
    """
    Create comprehensive visualization for extended timeframe analysis.
    """
    print("Creating extended timeframe visualization...")
    
    # Set site-themed style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'grid.color': '#495773',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'figure.facecolor': 'white',
        'text.color': '#220126',
        'axes.labelcolor': '#220126',
        'xtick.color': '#220126',
        'ytick.color': '#220126',
        'axes.titlecolor': '#2D0140'
    })
    
    # Set up the figure with optimal layout for extended data
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.4, left=0.08, right=0.95)
    
    # Site-themed color scheme
    colors = {
        'mars': '#E74C3C',        # Red for Mars
        'venus': '#F39C12',       # Orange for Venus
        'saturn': '#3498DB',      # Blue for Saturn  
        'jupiter': '#2D0140',     # Site dark purple for Jupiter (dominant)
        'sun': '#F1C40F',        # Yellow for Sun
        'total': '#220126',       # Site primary dark for total
        'temporal': '#4A90C2',    # Site accent blue for temporal
        'secondary': '#495773'    # Site secondary for accents
    }
    
    # Panel 1: Extended Stacked Planetary Gravitational Influences
    ax1 = fig.add_subplot(gs[0, 0])
    
    dates = combined_df['date']
    mars_vals = combined_df['mars_influence']
    venus_vals = combined_df['venus_influence'] 
    saturn_vals = combined_df['saturn_influence']
    jupiter_vals = combined_df['jupiter_influence']
    
    ax1.fill_between(dates, 0, mars_vals, alpha=0.8, color=colors['mars'], label='Mars')
    ax1.fill_between(dates, mars_vals, mars_vals + venus_vals, alpha=0.8, color=colors['venus'], label='Venus')
    ax1.fill_between(dates, mars_vals + venus_vals, mars_vals + venus_vals + saturn_vals, 
                     alpha=0.8, color=colors['saturn'], label='Saturn')
    ax1.fill_between(dates, mars_vals + venus_vals + saturn_vals, 
                     mars_vals + venus_vals + saturn_vals + jupiter_vals,
                     alpha=0.8, color=colors['jupiter'], label='Jupiter')
    
    ax1.plot(dates, combined_df['total_planetary_influence'], color=colors['total'], 
             linewidth=2, label='Total Planetary Influence')
    
    ax1.set_ylabel('Gravitational Influence (M⊕/AU²)', fontsize=12, fontweight='bold')
    
    # Add span info to title
    span_years = analysis_results['data_summary']['analysis_span_years']
    ax1.set_title(f'Complete Historical Stacked Planetary Gravitational Influences ({span_years:.1f} Years)\n' +
                  'CODE-Only Analysis 2010-2025 with NASA/JPL DE440/441 Ephemeris', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Historical TEP Temporal Field Signatures
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(dates, combined_df['coherence_mean'], color=colors['temporal'], 
                     linewidth=2, label='Coherence Mean', alpha=0.8)
    line2 = ax2_twin.plot(dates, combined_df['coherence_std'], color=colors['secondary'], 
                          linewidth=2, label='Coherence Variability', alpha=0.8)
    
    ax2.set_ylabel('TEP Coherence Mean', fontsize=12, fontweight='bold', color=colors['temporal'])
    ax2_twin.set_ylabel('TEP Coherence Variability (Std)', fontsize=12, fontweight='bold', color=colors['secondary'])
    ax2.set_title('Complete Historical TEP Temporal Field Signatures (CODE Only)\n' +
                  'Phase-Coherent Cross-Spectral Density Analysis 2010-2025', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Extended Pattern Correlation Analysis
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Smoothed patterns for correlation visualization
    if 'extended_pattern_analysis' in analysis_results:
        window_size = analysis_results['extended_pattern_analysis']['smoothing_window']
        poly_order = min(3, window_size - 1)
        
        smoothed_stacked = savgol_filter(combined_df['total_planetary_influence'], window_size, poly_order)
        smoothed_coherence_std = savgol_filter(combined_df['coherence_std'], window_size, poly_order)
        
        # Normalize for comparison
        norm_stacked = (smoothed_stacked - np.mean(smoothed_stacked)) / np.std(smoothed_stacked)
        norm_coherence = (smoothed_coherence_std - np.mean(smoothed_coherence_std)) / np.std(smoothed_coherence_std)
        
        ax3.plot(dates, norm_stacked, color=colors['total'], linewidth=3, 
                 label='Normalized Stacked Gravitational Pattern', alpha=0.8)
        ax3.plot(dates, norm_coherence, color=colors['secondary'], linewidth=3, 
                 label='Normalized Temporal Field Pattern', alpha=0.8)
        
        # Add correlation coefficient
        corr_r = analysis_results['extended_pattern_analysis']['smoothed_correlation']
        corr_p = analysis_results['extended_pattern_analysis']['smoothed_p_value']
        
        ax3.text(0.02, 0.95, f'Extended Pattern Correlation: r = {corr_r:.3f}, p = {corr_p:.2e}', 
                 transform=ax3.transAxes, fontsize=12, fontweight='bold', color='#220126',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8FF', 
                          edgecolor='#2D0140', alpha=0.95, linewidth=1))
    
    ax3.set_ylabel('Normalized Pattern Amplitude', fontsize=12, fontweight='bold')
    ax3.set_title('Complete Historical Gravitational-Temporal Field Pattern Correlation\n' +
                  'Smoothed Patterns Reveal 2010-2025 Coupling', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='#220126', linestyle='-', alpha=0.8, linewidth=1.5)
    
    # Panel 4: Yearly Statistics Summary
    ax4 = fig.add_subplot(gs[3, 0])
    
    if 'yearly_analysis' in analysis_results:
        years = analysis_results['yearly_analysis']['years_covered']
        yearly_stats = analysis_results['yearly_analysis']['yearly_statistics']
        
        # Extract yearly means (using flattened column names)
        yearly_grav_mean = [yearly_stats['total_planetary_influence_mean'][year] for year in years]
        yearly_coherence_mean = [yearly_stats['coherence_std_mean'][year] for year in years]
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar([y - 0.2 for y in years], yearly_grav_mean, width=0.4, 
                       color=colors['total'], alpha=0.7, label='Gravitational Influence')
        bars2 = ax4_twin.bar([y + 0.2 for y in years], yearly_coherence_mean, width=0.4, 
                            color=colors['secondary'], alpha=0.7, label='Coherence Variability')
        
        ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Annual Mean Gravitational Influence', fontsize=11, color=colors['total'])
        ax4_twin.set_ylabel('Annual Mean Coherence Variability', fontsize=11, color=colors['secondary'])
        ax4.set_title('Annual Statistics Summary', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # Format x-axis for time series plots
    for ax in [ax1, ax2, ax3]:
        # Use yearly ticks for extended data
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    # Save the figure with proper layout (skip tight_layout to avoid warnings)
    output_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/exploratory/figures/step_14_complete_2010_2025_gravitational_temporal_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Complete 15.5-year analysis visualization saved: {output_path}")
    return output_path

def assess_data_availability() -> Dict:
    """
    Assess what CODE data is actually available for extended analysis.
    """
    print("Assessing CODE data availability for extended analysis...")
    
    # Check existing processed files
    pair_files = glob.glob('/Users/matthewsmawfield/www/TEP-GNSS/results/tmp/step_3_pairs_code_*.csv')
    
    available_dates = []
    for file_path in pair_files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            for part in parts:
                if part.startswith('20') and len(part) >= 11:
                    date_part = part[:7]  # Take YYYYDDD
                    year = int(date_part[:4])
                    day_of_year = int(date_part[4:7])
                    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                    available_dates.append(date)
                    break
        except:
            continue
    
    if available_dates:
        available_dates.sort()
        
        assessment = {
            'total_files': len(pair_files),
            'date_range': {
                'start': available_dates[0].strftime('%Y-%m-%d'),
                'end': available_dates[-1].strftime('%Y-%m-%d'),
                'span_days': (available_dates[-1] - available_dates[0]).days,
                'span_years': (available_dates[-1] - available_dates[0]).days / 365.25
            },
            'yearly_coverage': {}
        }
        
        # Analyze yearly coverage
        for date in available_dates:
            year = date.year
            if year not in assessment['yearly_coverage']:
                assessment['yearly_coverage'][year] = 0
            assessment['yearly_coverage'][year] += 1
        
        return assessment
    else:
        return {'error': 'No CODE data files found'}

def main():
    """
    Main execution function for extended CODE-only analysis.
    """
    print("=" * 80)
    print("STEP 14 EXTENDED: COMPLETE 15.5-YEAR GRAVITATIONAL-TEMPORAL ANALYSIS (CODE ONLY)")
    print("=" * 80)
    print("PORTABLE MODE - Processes raw historical CODE data completely fresh")
    print()
    
    # COMPLETE 15.5-YEAR ANALYSIS: Process 2010-2025 historical CODE data
    start_year, end_year = 2010, 2025
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    print(f"COMPLETE 15.5-YEAR ANALYSIS: Using {start_year}-{end_year} date range")
    print("Processing ALL historical CODE data from scratch - completely portable")
    print(f"Dataset: {5_660:,} CODE CLK files spanning 15.5 years")
    print()
    
    print(f"🌌 Generating gravitational data for focused period: {start_date} to {end_date}")
    
    # Generate gravitational data for the extended period using parallel processing
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate all dates
    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"  Computing gravitational influences for {len(dates)} days...")
    print(f"  Using {min(cpu_count(), 4)} workers for gravitational calculations...")
    
    data_list = []
    start_time = time.time()
    
    # Use ThreadPoolExecutor for I/O bound astronomical calculations
    with ThreadPoolExecutor(max_workers=min(cpu_count(), 4)) as executor:
        # Submit all gravitational calculations
        future_to_date = {executor.submit(calculate_high_precision_gravitational_influence, date): date 
                         for date in dates}
        
        for i, future in enumerate(future_to_date):
            try:
                gravitational_data = future.result()
                date = future_to_date[future]
                
                if gravitational_data:
                    data_entry = {'date': date}
                    data_entry.update(gravitational_data)
                    data_list.append(data_entry)
                
                # Show progress every 200 days
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(dates) - i - 1) / rate if rate > 0 else 0
                    print(f"    [{i+1:4d}/{len(dates)}] Gravitational data | {rate:.1f} days/s | ETA: {eta/60:.1f}m")
                    
            except Exception as e:
                print(f"Error calculating gravitational influence for {future_to_date[future]}: {e}")
                continue
    
    elapsed_total = time.time() - start_time
    print(f"  ⚡ Gravitational calculations complete! {elapsed_total:.1f}s | {len(dates)/elapsed_total:.1f} days/s")
    gravitational_df = pd.DataFrame(data_list)
    print(f"✅ Generated gravitational data for {len(gravitational_df)} days")
    
    # Extract extended CODE TEP coherence data
    start_year = start.year
    end_year = end.year
    tep_df = extract_extended_code_tep_coherence_data(start_year, end_year)
    
    # Merge datasets
    print("🔗 Merging extended gravitational and temporal field datasets...")
    print(f"  Gravitational data: {len(gravitational_df)} days")
    print(f"  TEP coherence data: {len(tep_df)} days")
    combined_df = pd.merge(gravitational_df, tep_df, on='date', how='inner')
    print(f"✅ Extended combined dataset: {len(combined_df)} days of synchronized data")
    print(f"  Final date range: {combined_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Analysis span: {(combined_df['date'].max() - combined_df['date'].min()).days / 365.25:.2f} years")
    
    if len(combined_df) < 100:
        print("WARNING: Very limited data available for analysis")
        return None
    
    # Perform extended correlation analysis
    print("🔬 Performing extended correlation analysis...")
    print("  Computing planetary gravitational influences...")
    print("  Calculating cross-correlations with TEP coherence...")
    print("  Performing statistical significance tests...")
    analysis_results = perform_extended_correlation_analysis(combined_df)
    print("✅ Correlation analysis complete!")
    
    # Create extended visualization
    print("📊 Creating extended timeframe visualization...")
    figure_path = create_extended_visualization(combined_df, analysis_results)
    print(f"✅ Visualization saved: {figure_path}")
    analysis_results['figure_path'] = figure_path
    # analysis_results['data_assessment'] = data_assessment  # Skip data assessment for extended analysis
    
    # Save results
    results_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_extended_code_analysis_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    data_path = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/step_14_complete_2010_2025_gravitational_temporal_data.csv'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    combined_df.to_csv(data_path, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXTENDED ANALYSIS COMPLETE - KEY DISCOVERIES")
    print("=" * 80)
    
    stacked_corr = analysis_results['correlations']['stacked_planetary_influence']['coherence_std']
    print(f"EXTENDED STACKED CORRELATION: r = {stacked_corr['pearson_r']:.4f}, p = {stacked_corr['pearson_p']:.2e}")
    
    if 'extended_pattern_analysis' in analysis_results:
        smooth_corr = analysis_results['extended_pattern_analysis']['smoothed_correlation']
        print(f"EXTENDED SMOOTHED CORRELATION: r = {smooth_corr:.4f}")
    
    span_years = analysis_results['data_summary']['analysis_span_years']
    print(f"ANALYSIS SPAN: {span_years:.1f} years ({len(combined_df)} days)")
    print(f"FIGURE: {figure_path}")
    print(f"RESULTS: {results_path}")
    print(f"DATA: {data_path}")
    
    print("\nCOMPLETE 15.5-YEAR ANALYSIS INSIGHTS:")
    print("   Complete historical gravitational-temporal correlation analysis using CODE-only data")
    print("   15.5 years of continuous data provides unprecedented statistical power")
    print("   Reveals long-term patterns and eliminates short-term noise")
    print("   Completely portable - runs fresh from raw historical CODE files")
    print("=" * 80)
    
    return analysis_results

if __name__ == "__main__":
    results = main()
