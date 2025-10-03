#!/usr/bin/env python3
"""
TEP GNSS Analysis – STEP 18: Seasonal Diurnal Analysis
=====================================================

Purpose
-------
Analyze the seasonal variations in diurnal patterns from the full 2024 data
to understand whether the early morning peaks are consistent year-round or
vary with solar angle and seasonal effects.

Key Features
------------
- Loads hourly data from all three centers (CODE, IGS, ESA)
- Groups data by season (DJF, MAM, JJA, SON)
- Analyzes diurnal patterns for each season
- Compares peak timing and magnitude across seasons
- Identifies seasonal dependencies in the early morning peaks

Outputs
-------
- Seasonal diurnal patterns for each center
- Peak timing analysis by season
- Statistical comparison across seasons
- Visualization of seasonal variations

Usage
-----
python scripts/experimental/step_18_seasonal_analysis.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"

def load_hourly_data(center: str) -> pd.DataFrame:
    """Load hourly summary data for a center."""
    csv_path = RESULTS_DIR / f"step_18_diurnal_hourly_summary_{center}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Hourly data not found for {center}: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} hourly records for {center}")
    return df

def analyze_seasonal_patterns(df: pd.DataFrame, center: str) -> dict:
    """Analyze seasonal patterns in the hourly data."""
    
    # Since we don't have date information in the hourly summary,
    # we'll need to load the raw data to get seasonal information
    raw_csv_path = RESULTS_DIR / f"step_18_diurnal_hourly_raw_{center}.csv"
    
    if not raw_csv_path.exists():
        print(f"Raw data not found for {center}, using summary data only")
        return analyze_summary_patterns(df, center)
    
    # Load raw data with timestamps
    raw_df = pd.read_csv(raw_csv_path)
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df['month'] = raw_df['date'].dt.month
    
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'DJF'  # Winter
        elif month in [3, 4, 5]:
            return 'MAM'  # Spring
        elif month in [6, 7, 8]:
            return 'JJA'  # Summer
        else:
            return 'SON'  # Fall
    
    raw_df['season'] = raw_df['month'].apply(get_season)
    
    # Group by season and local hour (using local_hour_i as proxy for local time)
    seasonal_hourly = raw_df.groupby(['season', 'local_hour_i']).agg({
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
        
        # Calculate day/night means (6 AM - 6 PM vs 6 PM - 6 AM)
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
            'sample_count': season_data['sample_count'].sum()
        }
    
    return results

def analyze_summary_patterns(df: pd.DataFrame, center: str) -> dict:
    """Analyze patterns from summary data when raw data is not available."""
    
    # Find peak hour
    peak_idx = df['coherence_mean'].idxmax()
    peak_hour = df.loc[peak_idx, 'local_hour_bin']
    peak_value = df.loc[peak_idx, 'coherence_mean']
    
    # Calculate day/night means
    day_mask = (df['local_hour_bin'] >= 6) & (df['local_hour_bin'] < 18)
    night_mask = ~day_mask
    
    day_mean = df[day_mask]['coherence_mean'].mean()
    night_mean = df[night_mask]['coherence_mean'].mean()
    
    # Calculate CV
    cv = df['coherence_mean'].std() / abs(df['coherence_mean'].mean())
    
    return {
        'annual': {
            'peak_hour': peak_hour,
            'peak_value': peak_value,
            'day_mean': day_mean,
            'night_mean': night_mean,
            'day_night_ratio': day_mean / night_mean if night_mean != 0 else np.nan,
            'cv': cv,
            'sample_count': df['sample_count'].sum()
        }
    }

def create_seasonal_plot(center: str, seasonal_data: dict):
    """Create a plot showing seasonal variations."""
    
    if 'annual' in seasonal_data:
        # Only annual data available
        return
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    season_names = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Peak hours by season
    peak_hours = [seasonal_data.get(season, {}).get('peak_hour', np.nan) for season in seasons]
    peak_values = [seasonal_data.get(season, {}).get('peak_value', np.nan) for season in seasons]
    
    bars = ax1.bar(season_names, peak_hours, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Peak Hour (Local Time)')
    ax1.set_title(f'{center.upper()} - Peak Hour by Season')
    ax1.set_ylim(0, 24)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, peak_hours):
        if not np.isnan(value):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                    f'{value:.0f}', ha='center', va='bottom')
    
    # Plot 2: Day/Night ratios by season
    ratios = [seasonal_data.get(season, {}).get('day_night_ratio', np.nan) for season in seasons]
    
    bars2 = ax2.bar(season_names, ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Day/Night Ratio')
    ax2.set_title(f'{center.upper()} - Day/Night Ratio by Season')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal Day/Night')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars2, ratios):
        if not np.isnan(value):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_path = PROJECT_ROOT / "results" / "figures" / f"step_18_seasonal_analysis_{center}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Seasonal plot saved: {output_path}")
    
    plt.show()

def main():
    """Main analysis function."""
    
    centers = ['code', 'igs_combined', 'esa_final']
    all_results = {}
    
    print("=== STEP 18: SEASONAL DIURNAL ANALYSIS ===\n")
    
    for center in centers:
        print(f"Analyzing {center.upper()}...")
        
        try:
            # Load hourly data
            df = load_hourly_data(center)
            
            # Analyze seasonal patterns
            seasonal_data = analyze_seasonal_patterns(df, center)
            all_results[center] = seasonal_data
            
            # Create seasonal plot
            create_seasonal_plot(center, seasonal_data)
            
            # Print summary
            print(f"\n{center.upper()} Seasonal Summary:")
            if 'annual' in seasonal_data:
                data = seasonal_data['annual']
                print(f"  Annual Peak: Hour {data['peak_hour']:.0f} ({data['peak_value']:.5f})")
                print(f"  Day/Night Ratio: {data['day_night_ratio']:.3f}")
                print(f"  CV: {data['cv']:.3f}")
            else:
                for season in ['DJF', 'MAM', 'JJA', 'SON']:
                    if season in seasonal_data:
                        data = seasonal_data[season]
                        print(f"  {season}: Peak Hour {data['peak_hour']:.0f}, Ratio {data['day_night_ratio']:.3f}, CV {data['cv']:.3f}")
            
            print()
            
        except Exception as e:
            print(f"Error analyzing {center}: {e}")
            continue
    
    # Save results
    output_path = RESULTS_DIR / "step_18_seasonal_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Seasonal analysis results saved: {output_path}")
    
    # Print cross-center comparison
    print("\n=== CROSS-CENTER SEASONAL COMPARISON ===")
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        print(f"\n{season} Season:")
        for center in centers:
            if center in all_results and season in all_results[center]:
                data = all_results[center][season]
                print(f"  {center.upper()}: Peak Hour {data['peak_hour']:.0f}, Ratio {data['day_night_ratio']:.3f}")

if __name__ == "__main__":
    main()
