#!/usr/bin/env python3
"""
Step 14 2022 Completeness Assessment
Analyzes the completeness of 2022 data after redownload and identifies any remaining gaps

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

def analyze_2022_file_completeness() -> Dict:
    """Comprehensive analysis of 2022 file completeness after redownload"""
    
    log("🔍 Analyzing 2022 file completeness after redownload...")
    
    analysis = {
        'file_inventory': {},
        'temporal_coverage': {},
        'file_integrity': {},
        'remaining_gaps': {},
        'quality_assessment': {}
    }
    
    # Get all 2022 files
    code_dir = '/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code'
    all_2022_files = glob.glob(os.path.join(code_dir, '*2022*'))
    
    log(f"  Found {len(all_2022_files)} total 2022 files")
    
    # Parse file information
    file_info = []
    for file_path in all_2022_files:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Extract date from filename
        date_extracted = None
        doy_extracted = None
        
        # Handle both formats
        if 'COD0OPSFIN_' in filename:
            # Modern format: COD0OPSFIN_20220100000_01D_30S_CLK.CLK.gz
            match = re.search(r'COD0OPSFIN_(\d{7})', filename)
            if match:
                year_doy = match.group(1)
                year = int(year_doy[:4])
                doy = int(year_doy[4:])
                date_extracted = datetime(year, 1, 1) + timedelta(days=doy - 1)
                doy_extracted = doy
        else:
            # Legacy format converted: should have been converted to modern
            log(f"    Warning: Unexpected filename format: {filename}")
        
        if date_extracted:
            file_info.append({
                'filename': filename,
                'file_path': file_path,
                'date': date_extracted,
                'doy': doy_extracted,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024)
            })
    
    # Sort by date
    file_info.sort(key=lambda x: x['date'])
    
    analysis['file_inventory'] = {
        'total_files': len(file_info),
        'size_range_mb': {
            'min': min(f['size_mb'] for f in file_info) if file_info else 0,
            'max': max(f['size_mb'] for f in file_info) if file_info else 0,
            'mean': np.mean([f['size_mb'] for f in file_info]) if file_info else 0
        },
        'total_size_gb': sum(f['size_bytes'] for f in file_info) / (1024**3)
    }
    
    # Temporal coverage analysis
    if file_info:
        dates_covered = [f['date'] for f in file_info]
        date_strings = [d.strftime('%Y-%m-%d') for d in dates_covered]
        
        # Generate expected 2022 dates
        expected_dates = []
        current = datetime(2022, 1, 1)
        while current.year == 2022:
            expected_dates.append(current)
            current += timedelta(days=1)
        
        expected_date_strings = [d.strftime('%Y-%m-%d') for d in expected_dates]
        
        # Find gaps
        covered_set = set(date_strings)
        expected_set = set(expected_date_strings)
        missing_dates = expected_set - covered_set
        
        # Analyze temporal patterns
        monthly_coverage = {}
        for month in range(1, 13):
            month_dates = [d for d in expected_dates if d.month == month]
            month_covered = [d for d in dates_covered if d.month == month]
            monthly_coverage[month] = {
                'expected': len(month_dates),
                'covered': len(month_covered),
                'coverage_pct': (len(month_covered) / len(month_dates)) * 100 if month_dates else 0
            }
        
        analysis['temporal_coverage'] = {
            'date_range': {
                'start': min(dates_covered).strftime('%Y-%m-%d'),
                'end': max(dates_covered).strftime('%Y-%m-%d')
            },
            'total_expected': len(expected_dates),
            'total_covered': len(dates_covered),
            'coverage_percentage': (len(dates_covered) / len(expected_dates)) * 100,
            'missing_count': len(missing_dates),
            'monthly_coverage': monthly_coverage
        }
        
        # Identify gap patterns
        missing_date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in sorted(missing_dates)]
        gap_periods = []
        
        if missing_date_objs:
            current_gap_start = missing_date_objs[0]
            current_gap_end = missing_date_objs[0]
            
            for i in range(1, len(missing_date_objs)):
                if (missing_date_objs[i] - current_gap_end).days == 1:
                    # Consecutive missing day
                    current_gap_end = missing_date_objs[i]
                else:
                    # Gap ended, record it
                    gap_periods.append({
                        'start': current_gap_start.strftime('%Y-%m-%d'),
                        'end': current_gap_end.strftime('%Y-%m-%d'),
                        'duration_days': (current_gap_end - current_gap_start).days + 1
                    })
                    current_gap_start = missing_date_objs[i]
                    current_gap_end = missing_date_objs[i]
            
            # Don't forget the last gap
            gap_periods.append({
                'start': current_gap_start.strftime('%Y-%m-%d'),
                'end': current_gap_end.strftime('%Y-%m-%d'),
                'duration_days': (current_gap_end - current_gap_start).days + 1
            })
        
        analysis['remaining_gaps'] = {
            'total_missing_days': len(missing_dates),
            'gap_periods': gap_periods,
            'largest_gap_days': max([g['duration_days'] for g in gap_periods]) if gap_periods else 0,
            'missing_dates': sorted(list(missing_dates))
        }
    
    # File integrity assessment
    suspicious_files = []
    size_threshold_mb = 1.0  # Files smaller than 1MB are suspicious
    
    for f in file_info:
        if f['size_mb'] < size_threshold_mb:
            suspicious_files.append({
                'filename': f['filename'],
                'size_mb': f['size_mb'],
                'date': f['date'].strftime('%Y-%m-%d'),
                'issue': 'File too small'
            })
    
    # Check for duplicates (same date, different files)
    date_counts = Counter([f['date'].strftime('%Y-%m-%d') for f in file_info])
    duplicate_dates = [date for date, count in date_counts.items() if count > 1]
    
    analysis['file_integrity'] = {
        'suspicious_files': suspicious_files,
        'duplicate_dates': duplicate_dates,
        'size_distribution': {
            'files_under_1mb': len([f for f in file_info if f['size_mb'] < 1.0]),
            'files_1_to_3mb': len([f for f in file_info if 1.0 <= f['size_mb'] < 3.0]),
            'files_3_to_5mb': len([f for f in file_info if 3.0 <= f['size_mb'] < 5.0]),
            'files_over_5mb': len([f for f in file_info if f['size_mb'] >= 5.0])
        }
    }
    
    # Quality assessment and recommendations
    coverage_pct = analysis['temporal_coverage']['coverage_percentage']
    
    if coverage_pct >= 90:
        quality_rating = "EXCELLENT"
        recommendation = "Proceed with full 2022 analysis"
    elif coverage_pct >= 70:
        quality_rating = "GOOD"
        recommendation = "Suitable for analysis with gap documentation"
    elif coverage_pct >= 50:
        quality_rating = "MODERATE"
        recommendation = "Consider for supplementary analysis only"
    elif coverage_pct >= 25:
        quality_rating = "POOR"
        recommendation = "Use with caution, document limitations"
    else:
        quality_rating = "VERY POOR"
        recommendation = "Exclude from main analysis"
    
    analysis['quality_assessment'] = {
        'overall_rating': quality_rating,
        'coverage_percentage': coverage_pct,
        'recommendation': recommendation,
        'data_usability': {
            'suitable_for_main_analysis': coverage_pct >= 70,
            'suitable_for_supplementary': coverage_pct >= 25,
            'requires_gap_documentation': len(analysis['remaining_gaps']['gap_periods']) > 5
        }
    }
    
    return analysis

def create_completeness_visualization(analysis: Dict) -> str:
    """Create comprehensive visualization of 2022 data completeness"""
    
    log("📈 Creating 2022 completeness visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('2022 Data Completeness Assessment: Post-Redownload Analysis', 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Monthly coverage
    ax1 = axes[0, 0]
    
    monthly_data = analysis['temporal_coverage']['monthly_coverage']
    months = list(monthly_data.keys())
    coverage_pcts = [monthly_data[m]['coverage_pct'] for m in months]
    
    bars = ax1.bar(months, coverage_pcts, alpha=0.7, 
                   color=['green' if pct >= 70 else 'orange' if pct >= 25 else 'red' for pct in coverage_pcts])
    
    ax1.set_title('Monthly Coverage Percentage', fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, coverage_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel 2: Gap analysis
    ax2 = axes[0, 1]
    
    gap_periods = analysis['remaining_gaps']['gap_periods']
    if gap_periods:
        gap_durations = [g['duration_days'] for g in gap_periods]
        
        # Create histogram of gap durations
        bins = [1, 2, 5, 10, 20, 50, max(gap_durations)+1] if gap_durations else [1, 2]
        hist, bin_edges = np.histogram(gap_durations, bins=bins)
        
        ax2.bar(range(len(hist)), hist, alpha=0.7, color='red')
        ax2.set_title('Gap Duration Distribution', fontweight='bold')
        ax2.set_xlabel('Gap Duration (days)')
        ax2.set_ylabel('Number of Gaps')
        
        # Custom x-tick labels
        labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}' for i in range(len(hist))]
        ax2.set_xticks(range(len(hist)))
        ax2.set_xticklabels(labels, rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No gaps found!', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, color='green')
        ax2.set_title('Gap Duration Distribution', fontweight='bold')
    
    # Panel 3: File size distribution
    ax3 = axes[1, 0]
    
    size_dist = analysis['file_integrity']['size_distribution']
    categories = ['<1MB', '1-3MB', '3-5MB', '>5MB']
    counts = [
        size_dist['files_under_1mb'],
        size_dist['files_1_to_3mb'],
        size_dist['files_3_to_5mb'],
        size_dist['files_over_5mb']
    ]
    
    colors = ['red', 'orange', 'green', 'blue']
    bars = ax3.bar(categories, counts, color=colors, alpha=0.7)
    
    ax3.set_title('File Size Distribution', fontweight='bold')
    ax3.set_xlabel('File Size Category')
    ax3.set_ylabel('Number of Files')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom')
    
    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    coverage_pct = analysis['temporal_coverage']['coverage_percentage']
    total_files = analysis['file_inventory']['total_files']
    missing_days = analysis['remaining_gaps']['total_missing_days']
    quality_rating = analysis['quality_assessment']['overall_rating']
    recommendation = analysis['quality_assessment']['recommendation']
    
    summary_text = f"""
2022 DATA COMPLETENESS SUMMARY

Coverage Statistics:
• Total files: {total_files} / 365 days
• Coverage: {coverage_pct:.1f}%
• Missing days: {missing_days}
• Data size: {analysis['file_inventory']['total_size_gb']:.2f} GB

Quality Assessment:
• Rating: {quality_rating}
• Largest gap: {analysis['remaining_gaps']['largest_gap_days']} days
• Suspicious files: {len(analysis['file_integrity']['suspicious_files'])}

File Integrity:
• Average size: {analysis['file_inventory']['size_range_mb']['mean']:.1f} MB
• Size range: {analysis['file_inventory']['size_range_mb']['min']:.1f} - {analysis['file_inventory']['size_range_mb']['max']:.1f} MB

RECOMMENDATION:
{recommendation}

NEXT STEPS:
{'✓ Proceed with 2022 analysis' if coverage_pct >= 70 else '⚠ Document gaps before analysis' if coverage_pct >= 25 else '✗ Exclude from main analysis'}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the completeness report
    output_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/exploratory/figures/step_14_2022_completeness_assessment.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"✅ Completeness assessment saved: {output_path}")
    
    return output_path

def main():
    """Main completeness assessment function"""
    
    log("🚀 Starting 2022 Data Completeness Assessment")
    log("   (Post-redownload analysis and gap identification)")
    log("=" * 80)
    
    # Perform comprehensive analysis
    analysis = analyze_2022_file_completeness()
    
    # Create visualization
    report_path = create_completeness_visualization(analysis)
    
    # Save detailed results
    results = {
        'assessment_date': datetime.now().isoformat(),
        'completeness_analysis': analysis,
        'report_figure': report_path
    }
    
    results_path = '/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_2022_completeness_assessment.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary report
    log("\n" + "="*80)
    log("2022 DATA COMPLETENESS ASSESSMENT COMPLETE")
    log("="*80)
    
    coverage_pct = analysis['temporal_coverage']['coverage_percentage']
    total_files = analysis['file_inventory']['total_files']
    missing_days = analysis['remaining_gaps']['total_missing_days']
    
    log(f"COVERAGE SUMMARY:")
    log(f"  Files available: {total_files} / 365 days")
    log(f"  Coverage percentage: {coverage_pct:.1f}%")
    log(f"  Missing days: {missing_days}")
    log(f"  Total data size: {analysis['file_inventory']['total_size_gb']:.2f} GB")
    
    log(f"\nGAP ANALYSIS:")
    log(f"  Number of gap periods: {len(analysis['remaining_gaps']['gap_periods'])}")
    log(f"  Largest gap: {analysis['remaining_gaps']['largest_gap_days']} days")
    
    if analysis['remaining_gaps']['gap_periods']:
        log(f"  Major gaps (>7 days):")
        for gap in analysis['remaining_gaps']['gap_periods']:
            if gap['duration_days'] > 7:
                log(f"    {gap['start']} to {gap['end']} ({gap['duration_days']} days)")
    
    log(f"\nFILE INTEGRITY:")
    log(f"  Suspicious files: {len(analysis['file_integrity']['suspicious_files'])}")
    log(f"  Duplicate dates: {len(analysis['file_integrity']['duplicate_dates'])}")
    log(f"  Average file size: {analysis['file_inventory']['size_range_mb']['mean']:.1f} MB")
    
    log(f"\nQUALITY ASSESSMENT:")
    log(f"  Overall rating: {analysis['quality_assessment']['overall_rating']}")
    log(f"  Recommendation: {analysis['quality_assessment']['recommendation']}")
    
    # Determine next steps
    if coverage_pct >= 70:
        log(f"\n✅ EXCELLENT RECOVERY!")
        log(f"  • {coverage_pct:.1f}% coverage is sufficient for reliable analysis")
        log(f"  • Proceed with full 2022 integration into TEP analysis")
        log(f"  • Document any remaining gaps in methodology")
    elif coverage_pct >= 25:
        log(f"\n⚠️  PARTIAL RECOVERY")
        log(f"  • {coverage_pct:.1f}% coverage allows supplementary analysis")
        log(f"  • Use 2022 data with documented limitations")
        log(f"  • Consider gap-filling techniques if appropriate")
    else:
        log(f"\n❌ INSUFFICIENT RECOVERY")
        log(f"  • {coverage_pct:.1f}% coverage too low for reliable analysis")
        log(f"  • Recommend excluding 2022 from main correlation study")
        log(f"  • Focus on 2014-2021 period for robust results")
    
    log(f"\n📊 Results saved to: {results_path}")
    log(f"📈 Assessment report: {report_path}")
    
    log("\n✅ 2022 completeness assessment complete!")

if __name__ == "__main__":
    main()
