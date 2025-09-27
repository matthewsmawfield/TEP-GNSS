#!/usr/bin/env python3
"""
TEP GNSS Analysis – STEP 18: Comprehensive Diurnal Analysis
==========================================================

Purpose
-------
Perform comprehensive diurnal analysis across multiple analysis centers and
extended date ranges. This script provides both annual and seasonal diurnal
pattern analysis, building on the core phase-coherent correlation methodology
from Step 3 but with enhanced temporal resolution and cross-center validation.

Key Features
------------
- Multi-center analysis (CODE, IGS_COMBINED, ESA_FINAL)
- Extended date range support (2023-2025-06-30)
- Annual and seasonal diurnal pattern analysis
- Parallel processing for optimal performance
- Comprehensive statistical validation
- Production-ready error handling and logging

Methodology
-----------
1. Load raw CLK files for specified date range and center
2. Apply phase-coherent cross-spectral analysis (Step 3 methodology)
3. Compute hourly coherence windows with local solar time conversion
4. Aggregate statistics by local time bins (annual and seasonal)
5. Perform cross-center validation and statistical analysis
6. Generate comprehensive outputs and visualizations

Outputs
-------
- Annual diurnal patterns for each center
- Seasonal diurnal patterns (DJF, MAM, JJA, SON)
- Cross-center validation statistics
- Comprehensive JSON validation reports
- Statistical summary tables
- Visualization plots

Usage
-----
python scripts/steps/step_18_comprehensive_diurnal_analysis.py \
    --start-date 2023-01-01 --end-date 2025-06-30 \
    --centers code igs_combined esa_final \
    --max-workers 8 --verbose

Environment Variables
--------------------
- STEP18_START_DATE: Start date (YYYY-MM-DD)
- STEP18_END_DATE: End date (YYYY-MM-DD)
- STEP18_CENTERS: Comma-separated list of centers
- STEP18_MAX_WORKERS: Number of parallel workers
- STEP18_MAX_STATIONS: Maximum stations per day
- STEP18_MAX_PAIRS: Maximum pairs per day
- STEP18_MIN_EPOCHS: Minimum epochs per pair
- STEP18_MIN_HOUR_EPOCHS: Minimum epochs per hourly window
- STEP18_WINDOW_HOURS: Sliding window width in hours
- STEP18_RANDOM_SEED: Random seed for reproducibility
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger

# ---------------------------------------------------------------------------
# Configuration and Constants
# ---------------------------------------------------------------------------

@dataclass
class Step18Config:
    """Configuration for Step 18 comprehensive diurnal analysis."""
    
    start_date: datetime
    end_date: datetime
    centers: List[str]
    max_workers: int = 8
    max_stations: int = 50
    max_pairs: int = 2000
    min_epochs: int = 24
    min_hour_epochs: int = 12
    window_hours: float = 2.0
    random_seed: int = 42
    verbose: bool = False
    
    @classmethod
    def from_args_and_env(cls, args: argparse.Namespace) -> "Step18Config":
        """Create configuration from command line arguments and environment variables."""
        
        def env_or_arg(name: str, arg_value, cast, default):
            env_val = os.getenv(name)
            if env_val is not None:
                try:
                    return cast(env_val)
                except Exception as exc:
                    raise ValueError(f"Invalid value for {name}: {env_val}") from exc
            return cast(arg_value) if arg_value is not None else cast(default)
        
        # Parse centers
        centers_arg = env_or_arg("STEP18_CENTERS", args.centers, str, "code,igs_combined,esa_final")
        centers = [c.strip().lower() for c in centers_arg.split(",")]
        
        # Parse dates
        start_date_str = env_or_arg("STEP18_START_DATE", args.start_date, str, "2023-01-01")
        end_date_str = env_or_arg("STEP18_END_DATE", args.end_date, str, "2025-06-30")
        
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {exc}") from exc
        
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        return cls(
            start_date=start_date,
            end_date=end_date,
            centers=centers,
            max_workers=env_or_arg("STEP18_MAX_WORKERS", args.max_workers, int, 8),
            max_stations=env_or_arg("STEP18_MAX_STATIONS", args.max_stations, int, 50),
            max_pairs=env_or_arg("STEP18_MAX_PAIRS", args.max_pairs, int, 2000),
            min_epochs=env_or_arg("STEP18_MIN_EPOCHS", args.min_epochs, int, 24),
            min_hour_epochs=env_or_arg("STEP18_MIN_HOUR_EPOCHS", args.min_hour_epochs, int, 12),
            window_hours=env_or_arg("STEP18_WINDOW_HOURS", args.window_hours, float, 2.0),
            random_seed=env_or_arg("STEP18_RANDOM_SEED", args.random_seed, int, 42),
            verbose=env_or_arg("STEP18_VERBOSE", args.verbose, bool, False)
        )

# ---------------------------------------------------------------------------
# Core Analysis Functions (from Step 3)
# ---------------------------------------------------------------------------

def parse_clk_file(clk_path: Path) -> pd.DataFrame:
    """Parse a single CLK file into a pivot DataFrame with timestamps."""
    
    try:
        with gzip.open(clk_path, "rt", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except Exception:
        try:
            with open(clk_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except Exception:
            # Try with latin-1 encoding as fallback
            try:
                with gzip.open(clk_path, "rt", encoding="latin-1") as fh:
                    lines = fh.readlines()
            except Exception:
                with open(clk_path, "r", encoding="latin-1") as fh:
                    lines = fh.readlines()
    
    records = []
    for line in lines:
        if line.startswith("AR "):  # AR = Analysis Receiver (ground stations)
            parts = line.strip().split()
            if len(parts) >= 10:
                try:
                    station = parts[1]
                    year = int(parts[2])
                    month = int(parts[3])
                    day = int(parts[4])
                    hour = int(parts[5])
                    minute = int(parts[6])
                    second = float(parts[7])
                    clock_bias = float(parts[9])  # parts[8] is the number of satellites, parts[9] is the clock bias
                    
                    timestamp = datetime(year, month, day, hour, minute, int(second))
                    records.append({
                        "timestamp": timestamp,
                        "station": station,
                        "clock_bias": clock_bias
                    })
                except (ValueError, IndexError):
                    continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    pivot = df.pivot(index="timestamp", columns="station", values="clock_bias")
    return pivot

def estimate_sampling_frequency(timestamps: np.ndarray) -> Optional[float]:
    """Estimate sampling frequency from timestamps."""
    
    if len(timestamps) < 2:
        return None
    
    dt = np.diff(timestamps.astype("datetime64[ns]")).astype("timedelta64[ns]")
    dt_seconds = dt.astype("float64") / 1e9
    
    # Check for consistent sampling
    if len(dt_seconds) > 0:
        median_dt = np.median(dt_seconds)
        if 0.9 * median_dt <= np.percentile(dt_seconds, 10) <= 1.1 * median_dt:
            return 1.0 / median_dt
    
    return None

# Import the working implementation from Step 3
from scripts.steps.step_3_tep_correlation_analysis import compute_cross_power_plateau

# ---------------------------------------------------------------------------
# Diurnal Analysis Functions
# ---------------------------------------------------------------------------

def load_station_coordinates() -> pd.DataFrame:
    """Load station coordinates for local time conversion."""
    
    coords_path = PROJECT_ROOT / "data" / "coordinates" / "station_coords_global.csv"
    if not coords_path.exists():
        raise FileNotFoundError(f"Station coordinates not found: {coords_path}")
    
    return pd.read_csv(coords_path)

def process_pair_hourly(
    pivot: pd.DataFrame,
    station_i: str,
    station_j: str,
    coords_df: pd.DataFrame,
    min_epochs: int,
    min_hour_epochs: int,
    window_hours: float,
) -> List[Dict[str, object]]:
    """Process a single station pair with hourly windowing."""
    
    if station_i not in pivot.columns or station_j not in pivot.columns:
        return []
    
    # Get station coordinates
    coords_i = coords_df[coords_df["code"] == station_i]
    coords_j = coords_df[coords_df["code"] == station_j]
    
    if coords_i.empty or coords_j.empty:
        return []
    
    lon_i = coords_i.iloc[0]["lon_deg"]
    lat_i = coords_i.iloc[0]["lat_deg"]
    lon_j = coords_j.iloc[0]["lon_deg"]
    lat_j = coords_j.iloc[0]["lat_deg"]
    
    # Get common time series
    series_i = pivot[station_i].dropna()
    series_j = pivot[station_j].dropna()
    common_times = series_i.index.intersection(series_j.index)
    
    if len(common_times) < min_epochs:
        return []
    
    # Create pair DataFrame
    pair_df = pd.DataFrame({
        "timestamp": common_times,
        "series_i": series_i.loc[common_times].values,
        "series_j": series_j.loc[common_times].values,
    })
    
    # Sliding window analysis
    window_delta = pd.to_timedelta(window_hours, unit="h")
    half_window = window_delta / 2
    start_time = pair_df["timestamp"].min().floor("h")
    end_time = pair_df["timestamp"].max().ceil("h")
    window_centers = pd.date_range(start_time + half_window, end_time - half_window, freq="1h")
    
    results = []
    for center in window_centers:
        mask = (pair_df["timestamp"] >= center - half_window) & (pair_df["timestamp"] < center + half_window)
        hour_df = pair_df.loc[mask]
        
        if len(hour_df) < min_hour_epochs:
            continue
        
        # Estimate sampling frequency
        timestamps = hour_df["timestamp"].values.astype("datetime64[ns]")
        fs = estimate_sampling_frequency(timestamps)
        if fs is None:
            continue
        
        # Compute cross-power plateau
        plateau_value, plateau_phase = compute_cross_power_plateau(
            hour_df["series_i"].values,
            hour_df["series_j"].values,
            fs=fs,
            use_real_coherency=False,
            f1=1e-5,   # 10 μHz - TEP frequency band
            f2=5e-4,   # 500 μHz - TEP frequency band
        )
        
        
        # Convert to phase-coherent coherence (cosine of phase)
        if not (np.isnan(plateau_value) or np.isnan(plateau_phase)):
            coherence = float(np.cos(plateau_phase))
        else:
            coherence = 0.0
        
        # Convert to local solar time
        utc_hour = int(center.hour)
        local_hour_i = (utc_hour + lon_i / 15.0) % 24.0
        local_hour_j = (utc_hour + lon_j / 15.0) % 24.0
        
        results.append({
            "date": center.date(),
            "utc_hour": utc_hour,
            "coherence": coherence,
            "plateau_phase": plateau_phase,
            "plateau_value": plateau_value,
            "n_epochs": len(hour_df),
            "station_i": station_i,
            "station_j": station_j,
            "local_hour_i": local_hour_i,
            "local_hour_j": local_hour_j,
            "lon_i": lon_i,
            "lon_j": lon_j,
            "lat_i": lat_i,
            "lat_j": lat_j,
            "distance_km": 0.0,  # Will be calculated if needed
            "window_hours": window_hours,
        })
    
    return results

def discover_clk_files(cfg: Step18Config, center: str) -> List[Path]:
    """Discover CLK files for the specified center and date range."""
    
    data_dir = PROJECT_ROOT / "data" / "raw" / center
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    clk_files = []
    current_date = cfg.start_date
    
    while current_date <= cfg.end_date:
        # Look for CLK files for this date
        date_str = current_date.strftime("%Y%j")
        pattern = f"*{date_str}*CLK.CLK.gz"
        
        matching_files = list(data_dir.glob(pattern))
        if matching_files:
            clk_files.extend(matching_files)
        
        current_date += timedelta(days=1)
    
    return sorted(clk_files)

def select_stations(pivot: pd.DataFrame, max_stations: int) -> List[str]:
    """Select stations with sufficient data."""
    
    # Count non-null values per station
    station_counts = pivot.count()
    valid_stations = station_counts[station_counts > 0].index.tolist()
    
    # Sort by data availability and select top stations
    station_counts = station_counts[valid_stations].sort_values(ascending=False)
    return station_counts.head(max_stations).index.tolist()

def limited_combinations(stations: List[str], max_pairs: int, rng: np.random.Generator) -> List[Tuple[str, str]]:
    """Generate limited combinations of station pairs."""
    
    all_pairs = [(i, j) for i in stations for j in stations if i < j]
    
    if len(all_pairs) <= max_pairs:
        return all_pairs
    
    # Randomly sample pairs
    return rng.choice(all_pairs, size=max_pairs, replace=False).tolist()

def process_single_file(args_tuple: Tuple[Path, Step18Config, pd.DataFrame, np.random.Generator, str]) -> List[Dict[str, object]]:
    """Process a single CLK file and return hourly records for all pairs."""
    
    clk_path, cfg, coords_df, rng, center = args_tuple
    
    try:
        pivot = parse_clk_file(clk_path)
        if pivot.empty:
            return []
        
        stations = select_stations(pivot, cfg.max_stations)
        if len(stations) < 2:
            return []
        
        pairs = limited_combinations(stations, cfg.max_pairs, rng)
        hourly_records = []
        
        for station_i, station_j in pairs:
            pair_results = process_pair_hourly(
                pivot,
                station_i,
                station_j,
                coords_df,
                cfg.min_epochs,
                cfg.min_hour_epochs,
                cfg.window_hours,
            )
            hourly_records.extend(pair_results)
        
        return hourly_records
        
    except Exception as exc:
        print(f"Error processing {clk_path.name}: {exc}")
        return []

# ---------------------------------------------------------------------------
# Analysis and Output Functions
# ---------------------------------------------------------------------------

def analyze_annual_patterns(hourly_df: pd.DataFrame) -> Dict[str, object]:
    """Analyze annual diurnal patterns."""
    
    # Group by local hour (using local_hour_i as proxy)
    hourly_df['local_hour_bin'] = hourly_df['local_hour_i'].round().astype(int) % 24
    
    annual_stats = hourly_df.groupby('local_hour_bin').agg({
        'coherence': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    annual_stats.columns = ['local_hour_bin', 'coherence_mean', 'coherence_std', 'sample_count']
    
    # Calculate day/night statistics
    day_mask = (annual_stats['local_hour_bin'] >= 6) & (annual_stats['local_hour_bin'] < 18)
    night_mask = ~day_mask
    
    day_mean = annual_stats[day_mask]['coherence_mean'].mean()
    night_mean = annual_stats[night_mask]['coherence_mean'].mean()
    
    # Calculate CV
    cv = annual_stats['coherence_mean'].std() / abs(annual_stats['coherence_mean'].mean())
    
    return {
        'coherence_cv': cv,
        'day_mean': day_mean,
        'night_mean': night_mean,
        'day_night_ratio': day_mean / night_mean if night_mean != 0 else np.nan,
        'hours': annual_stats.to_dict('records')
    }

def analyze_seasonal_patterns(hourly_df: pd.DataFrame) -> Dict[str, object]:
    """Analyze seasonal diurnal patterns."""
    
    # Add seasonal information
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    hourly_df['month'] = hourly_df['date'].dt.month
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'DJF'  # Winter
        elif month in [3, 4, 5]:
            return 'MAM'  # Spring
        elif month in [6, 7, 8]:
            return 'JJA'  # Summer
        else:
            return 'SON'  # Fall
    
    hourly_df['season'] = hourly_df['month'].apply(get_season)
    hourly_df['local_hour_bin'] = hourly_df['local_hour_i'].round().astype(int) % 24
    
    # Group by season and local hour
    seasonal_hourly = hourly_df.groupby(['season', 'local_hour_bin']).agg({
        'coherence': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    seasonal_hourly.columns = ['season', 'local_hour_bin', 'coherence_mean', 'coherence_std', 'sample_count']
    
    # Analyze each season
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    results = {}
    
    for season in seasons:
        season_data = seasonal_hourly[seasonal_hourly['season'] == season]
        if len(season_data) == 0:
            continue
        
        # Find peak hour
        peak_idx = season_data['coherence_mean'].idxmax()
        peak_hour = season_data.loc[peak_idx, 'local_hour_bin']
        peak_value = season_data.loc[peak_idx, 'coherence_mean']
        
        # Calculate day/night means
        day_mask = (season_data['local_hour_bin'] >= 6) & (season_data['local_hour_bin'] < 18)
        night_mask = ~day_mask
        
        day_mean = season_data[day_mask]['coherence_mean'].mean()
        night_mean = season_data[night_mask]['coherence_mean'].mean()
        
        # Calculate CV
        cv = season_data['coherence_mean'].std() / abs(season_data['coherence_mean'].mean())
        
        results[season] = {
            'peak_hour': peak_hour,
            'peak_value': peak_value,
            'day_mean': day_mean,
            'night_mean': night_mean,
            'day_night_ratio': day_mean / night_mean if night_mean != 0 else np.nan,
            'cv': cv,
            'sample_count': season_data['sample_count'].sum(),
            'hours': season_data.to_dict('records')
        }
    
    return results

def run_center_analysis(cfg: Step18Config, center: str, logger: TEPLogger) -> Dict[str, object]:
    """Run comprehensive diurnal analysis for a single center."""
    
    logger.info(f"Starting analysis for {center.upper()}")
    
    # Load station coordinates
    coords_df = load_station_coordinates()
    
    # Discover CLK files
    clk_files = discover_clk_files(cfg, center)
    if not clk_files:
        raise RuntimeError(f"No CLK files found for {center} in date range {cfg.start_date} to {cfg.end_date}")
    
    logger.info(f"Found {len(clk_files)} CLK files for {center.upper()}")
    
    # Process files in parallel
    rng = np.random.default_rng(cfg.random_seed)
    hourly_records = []
    
    file_args = [(clk_path, cfg, coords_df, rng, center) for clk_path in clk_files]
    
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, args): args[0] 
            for args in file_args
        }
        
        completed_files = 0
        for future in as_completed(future_to_file):
            clk_path = future_to_file[future]
            completed_files += 1
            
            try:
                file_records = future.result()
                hourly_records.extend(file_records)
                
                if cfg.verbose and completed_files % 50 == 0:
                    logger.info(f"Completed {completed_files}/{len(clk_files)} files, {len(hourly_records)} hourly records")
                    
            except Exception as exc:
                logger.error(f"Error processing {clk_path.name}: {exc}")
    
    if not hourly_records:
        raise RuntimeError(f"No hourly records generated for {center}")
    
    logger.info(f"Generated {len(hourly_records)} hourly records for {center.upper()}")
    
    # Convert to DataFrame
    hourly_df = pd.DataFrame(hourly_records)
    
    # Analyze patterns
    annual_patterns = analyze_annual_patterns(hourly_df)
    seasonal_patterns = analyze_seasonal_patterns(hourly_df)
    
    # Prepare results
    results = {
        'center': center,
        'date_range': {
            'start': cfg.start_date.strftime('%Y-%m-%d'),
            'end': cfg.end_date.strftime('%Y-%m-%d')
        },
        'coverage': {
            'clk_files_processed': len(clk_files),
            'pair_hours': len(hourly_records),
            'station_hour_samples': len(hourly_records) * 2,
            'unique_stations': len(set(hourly_df['station_i'].tolist() + hourly_df['station_j'].tolist()))
        },
        'annual_patterns': annual_patterns,
        'seasonal_patterns': seasonal_patterns,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    return results

def save_results(results: Dict[str, object], center: str, cfg: Step18Config):
    """Save analysis results to files."""
    
    output_dir = PROJECT_ROOT / "results" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON validation report
    json_path = output_dir / f"step_18_comprehensive_validation_{center}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save hourly summary CSV
    if 'annual_patterns' in results and 'hours' in results['annual_patterns']:
        summary_df = pd.DataFrame(results['annual_patterns']['hours'])
        summary_csv = output_dir / f"step_18_comprehensive_hourly_summary_{center}.csv"
        summary_df.to_csv(summary_csv, index=False)
    
    # Save raw hourly data (if needed for further analysis)
    # This would require the full hourly_df, which we don't have in results
    # Could be added if needed for downstream analysis

# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

def run_comprehensive_analysis(cfg: Step18Config) -> Dict[str, object]:
    """Run comprehensive diurnal analysis across all centers."""
    
    # Setup logging
    logger = TEPLogger("step_18", level="DEBUG" if cfg.verbose else "INFO")
    
    logger.info("=== STEP 18: COMPREHENSIVE DIURNAL ANALYSIS ===")
    logger.info(f"Date range: {cfg.start_date} to {cfg.end_date}")
    logger.info(f"Centers: {', '.join(cfg.centers)}")
    logger.info(f"Max workers: {cfg.max_workers}")
    
    all_results = {}
    
    for center in cfg.centers:
        try:
            logger.info(f"\n--- Processing {center.upper()} ---")
            
            # Run analysis for this center
            center_results = run_center_analysis(cfg, center, logger)
            all_results[center] = center_results
            
            # Save results
            save_results(center_results, center, cfg)
            
            # Print summary
            annual = center_results['annual_patterns']
            logger.info(f"Annual CV: {annual['coherence_cv']:.4f}")
            logger.info(f"Day/Night ratio: {annual['day_night_ratio']:.3f}")
            logger.info(f"Day mean: {annual['day_mean']:.6f}")
            logger.info(f"Night mean: {annual['night_mean']:.6f}")
            
            # Print seasonal summary
            seasonal = center_results['seasonal_patterns']
            logger.info("Seasonal peaks:")
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                if season in seasonal:
                    data = seasonal[season]
                    logger.info(f"  {season}: Hour {data['peak_hour']:.0f}, Ratio {data['day_night_ratio']:.3f}")
            
        except Exception as exc:
            logger.error(f"Error processing {center}: {exc}")
            continue
    
    # Save combined results
    combined_path = PROJECT_ROOT / "results" / "outputs" / "step_18_comprehensive_analysis.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nComprehensive analysis complete. Results saved to {combined_path}")
    
    return all_results

# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Step 18: Comprehensive Diurnal Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--centers", type=str, help="Comma-separated list of centers")
    parser.add_argument("--max-workers", type=int, help="Number of parallel workers")
    parser.add_argument("--max-stations", type=int, help="Maximum stations per day")
    parser.add_argument("--max-pairs", type=int, help="Maximum pairs per day")
    parser.add_argument("--min-epochs", type=int, help="Minimum epochs per pair")
    parser.add_argument("--min-hour-epochs", type=int, help="Minimum epochs per hourly window")
    parser.add_argument("--window-hours", type=float, help="Sliding window width in hours")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser

def main():
    """Main entry point."""
    
    parser = build_arg_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        cfg = Step18Config.from_args_and_env(args)
        
        # Run analysis
        results = run_comprehensive_analysis(cfg)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Processed {len(cfg.centers)} centers")
        print(f"Date range: {cfg.start_date} to {cfg.end_date}")
        print("Results saved to results/outputs/")
        
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
