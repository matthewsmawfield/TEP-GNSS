#!/usr/bin/env python3
"""
Step 14 2022 Final Assessment
Analyzes the final state of 2022 data after comprehensive redownload

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-27
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter, defaultdict
import glob
from typing import Dict, List, Tuple
import re

def log(message: str) -> None:
    print(message, flush=True)

def analyze_final_2022_status() -> Dict:
    """Analyze the final state of 2022 data after comprehensive redownload"""
    
    log("🔍 Analyzing final 2022 data status...")
    
    analysis = {
        'file_inventory': {},
        'temporal_coverage': {},
        'data_quality': {},
        'gaps_analysis': {},
        'summary': {}
    }
    
    # Count files by type
    code_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    
    # Count all 2022 files
    all_2022_files = glob.glob(f"{code_dir}/*2022*")
    analysis['file_inventory']['total_files'] = len(all_2022_files)
    
    # Count by format
    modern_files = glob.glob(f"{code_dir}/COD0OPSFIN_2022*_30S_CLK.CLK.gz")
    legacy_files = glob.glob(f"{code_dir}/COD0OPSFIN_2022*_30S_CLK.CLK")
    
    analysis['file_inventory']['modern_gz'] = len(modern_files)
    analysis['file_inventory']['legacy_uncompressed'] = len(legacy_files)
    
    # Extract dates from filenames
    dates = []
    for file_path in all_2022_files:
        filename = os.path.basename(file_path)
        # Extract date from COD0OPSFIN_YYYYMMDDHHMMSS format
        match = re.search(r'(\d{8})', filename)
        if match:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date)
            except ValueError:
                continue
    
    dates = sorted(set(dates))
    analysis['temporal_coverage']['unique_dates'] = len(dates)
    analysis['temporal_coverage']['date_range'] = {
        'start': dates[0].strftime('%Y-%m-%d') if dates else None,
        'end': dates[-1].strftime('%Y-%m-%d') if dates else None
    }
    
    # Calculate coverage
    expected_days = 365  # 2022 is not a leap year
    coverage_percent = (len(dates) / expected_days) * 100
    analysis['temporal_coverage']['coverage_percent'] = coverage_percent
    
    # Find gaps
    if dates:
        all_2022_dates = set()
        current_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        
        while current_date <= end_date:
            all_2022_dates.add(current_date)
            current_date += timedelta(days=1)
        
        available_dates = set(dates)
        missing_dates = sorted(all_2022_dates - available_dates)
        
        analysis['gaps_analysis']['missing_dates'] = len(missing_dates)
        analysis['gaps_analysis']['missing_percent'] = (len(missing_dates) / expected_days) * 100
        
        # Find gap periods
        gap_periods = []
        if missing_dates:
            gap_start = missing_dates[0]
            gap_end = missing_dates[0]
            
            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i-1]).days == 1:
                    gap_end = missing_dates[i]
                else:
                    gap_periods.append({
                        'start': gap_start.strftime('%Y-%m-%d'),
                        'end': gap_end.strftime('%Y-%m-%d'),
                        'duration_days': (gap_end - gap_start).days + 1
                    })
                    gap_start = missing_dates[i]
                    gap_end = missing_dates[i]
            
            # Add the last gap
            gap_periods.append({
                'start': gap_start.strftime('%Y-%m-%d'),
                'end': gap_end.strftime('%Y-%m-%d'),
                'duration_days': (gap_end - gap_start).days + 1
            })
        
        analysis['gaps_analysis']['gap_periods'] = gap_periods
        analysis['gaps_analysis']['largest_gap_days'] = max([gap['duration_days'] for gap in gap_periods]) if gap_periods else 0
    
    # Summary
    analysis['summary'] = {
        'total_files_available': len(all_2022_files),
        'unique_dates_available': len(dates),
        'coverage_percentage': coverage_percent,
        'missing_days': len(missing_dates) if dates else expected_days,
        'status': 'EXCELLENT' if coverage_percent >= 95 else 'GOOD' if coverage_percent >= 80 else 'POOR'
    }
    
    return analysis

def create_final_assessment_plot(analysis: Dict) -> str:
    """Create visualization of final 2022 data status"""
    
    log("📊 Creating final assessment visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('2022 Data Final Assessment - After Comprehensive Redownload', fontsize=16, fontweight='bold')
    
    # 1. File inventory
    ax1 = axes[0, 0]
    file_types = ['Modern (.gz)', 'Legacy (.CLK)', 'Total Files']
    file_counts = [
        analysis['file_inventory']['modern_gz'],
        analysis['file_inventory']['legacy_uncompressed'],
        analysis['file_inventory']['total_files']
    ]
    
    bars = ax1.bar(file_types, file_counts, color=['#2E8B57', '#4169E1', '#DC143C'])
    ax1.set_title('File Inventory by Type')
    ax1.set_ylabel('Number of Files')
    
    # Add value labels on bars
    for bar, count in zip(bars, file_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Coverage summary
    ax2 = axes[0, 1]
    coverage = analysis['temporal_coverage']['coverage_percent']
    missing = 100 - coverage
    
    wedges, texts, autotexts = ax2.pie([coverage, missing], 
                                      labels=[f'Available\n{coverage:.1f}%', f'Missing\n{missing:.1f}%'],
                                      colors=['#2E8B57', '#DC143C'],
                                      autopct='%1.1f%%',
                                      startangle=90)
    ax2.set_title('2022 Data Coverage')
    
    # 3. Monthly distribution
    ax3 = axes[1, 0]
    if analysis['temporal_coverage']['unique_dates'] > 0:
        # Count files by month
        monthly_counts = defaultdict(int)
        for date in analysis.get('dates', []):
            monthly_counts[date.month] += 1
        
        months = list(range(1, 13))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        counts = [monthly_counts[month] for month in months]
        
        bars = ax3.bar(month_names, counts, color='#4169E1')
        ax3.set_title('Files Available by Month')
        ax3.set_ylabel('Number of Days')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Gap analysis
    ax4 = axes[1, 1]
    if analysis['gaps_analysis']['gap_periods']:
        gap_periods = analysis['gaps_analysis']['gap_periods']
        gap_durations = [gap['duration_days'] for gap in gap_periods]
        
        ax4.bar(range(len(gap_durations)), gap_durations, color='#DC143C')
        ax4.set_title('Gap Periods (Missing Data)')
        ax4.set_ylabel('Duration (Days)')
        ax4.set_xlabel('Gap Number')
        
        # Add value labels
        for i, duration in enumerate(gap_durations):
            ax4.text(i, duration + 0.5, f'{duration}d', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Gaps Found!\nComplete Coverage', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, fontweight='bold', color='#2E8B57')
        ax4.set_title('Gap Analysis')
    
    plt.tight_layout()
    
    # Save plot
    output_path = "/Users/matthewsmawfield/www/TEP-GNSS/results/exploratory/figures/step_14_2022_final_assessment.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    """Main execution function"""
    
    log("🚀 Starting 2022 Final Assessment")
    log("=" * 80)
    
    # Analyze final status
    analysis = analyze_final_2022_status()
    
    # Create visualization
    plot_path = create_final_assessment_plot(analysis)
    
    # Save results
    results_path = "/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_2022_final_assessment_results.json"
    with open(results_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Print summary
    log("\n" + "=" * 80)
    log("🎯 2022 FINAL ASSESSMENT COMPLETE")
    log("=" * 80)
    
    summary = analysis['summary']
    log(f"📊 Total files available: {summary['total_files_available']:,}")
    log(f"📅 Unique dates available: {summary['unique_dates_available']:,}")
    log(f"📈 Coverage percentage: {summary['coverage_percentage']:.1f}%")
    log(f"❌ Missing days: {summary['missing_days']:,}")
    log(f"🎯 Status: {summary['status']}")
    
    if analysis['gaps_analysis']['gap_periods']:
        log(f"\n🔍 Gap Analysis:")
        log(f"   Number of gap periods: {len(analysis['gaps_analysis']['gap_periods'])}")
        log(f"   Largest gap: {analysis['gaps_analysis']['largest_gap_days']} days")
        
        log(f"\n📋 Gap Periods:")
        for i, gap in enumerate(analysis['gaps_analysis']['gap_periods'][:5]):  # Show first 5
            log(f"   {i+1}. {gap['start']} to {gap['end']} ({gap['duration_days']} days)")
        
        if len(analysis['gaps_analysis']['gap_periods']) > 5:
            log(f"   ... and {len(analysis['gaps_analysis']['gap_periods']) - 5} more periods")
    
    log(f"\n📊 Results saved to: {results_path}")
    log(f"📈 Visualization saved to: {plot_path}")
    
    # Final recommendation
    if summary['coverage_percentage'] >= 95:
        log(f"\n🎯 EXCELLENT! Ready for full TEP correlation analysis")
    elif summary['coverage_percentage'] >= 80:
        log(f"\n✅ GOOD! Suitable for TEP correlation analysis with minor gaps")
    else:
        log(f"\n⚠️  POOR coverage. Consider additional data recovery")
    
    log("=" * 80)

if __name__ == "__main__":
    main()
