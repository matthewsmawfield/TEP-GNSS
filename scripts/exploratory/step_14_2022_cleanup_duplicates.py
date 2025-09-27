#!/usr/bin/env python3
"""
Step 14 2022 Cleanup Duplicates
Removes duplicate 2022 files to have exactly one file per date
Keeps compressed .gz files and removes uncompressed .CLK files

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-27
"""

import os
import sys
import glob
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set

def log(message: str) -> None:
    print(message, flush=True)

def analyze_2022_duplicates() -> Dict:
    """Analyze the current state of 2022 files and identify duplicates"""
    
    log("🔍 Analyzing 2022 file duplicates...")
    
    code_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    
    # Find all 2022 files
    all_2022_files = glob.glob(f"{code_dir}/*2022*")
    
    # Group files by date
    files_by_date = defaultdict(list)
    date_pattern = re.compile(r'(\d{8})')  # Extract YYYYMMDD from filename
    
    for file_path in all_2022_files:
        filename = os.path.basename(file_path)
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)
            files_by_date[date_str].append(file_path)
    
    # Analyze duplicates
    analysis = {
        'total_files': len(all_2022_files),
        'unique_dates': len(files_by_date),
        'duplicate_dates': 0,
        'files_to_remove': [],
        'files_to_keep': [],
        'date_coverage': {}
    }
    
    for date_str, file_list in files_by_date.items():
        if len(file_list) > 1:
            analysis['duplicate_dates'] += 1
            
            # Sort files: prefer .gz files over .CLK files
            gz_files = [f for f in file_list if f.endswith('.gz')]
            clk_files = [f for f in file_list if f.endswith('.CLK')]
            
            if gz_files:
                # Keep the first .gz file, remove others
                analysis['files_to_keep'].append(gz_files[0])
                analysis['files_to_remove'].extend(gz_files[1:])
                analysis['files_to_remove'].extend(clk_files)
            elif clk_files:
                # Keep the first .CLK file, remove others
                analysis['files_to_keep'].append(clk_files[0])
                analysis['files_to_remove'].extend(clk_files[1:])
        else:
            # Single file for this date
            analysis['files_to_keep'].append(file_list[0])
        
        analysis['date_coverage'][date_str] = {
            'total_files': len(file_list),
            'files': file_list
        }
    
    return analysis

def cleanup_duplicates(analysis: Dict) -> Dict:
    """Remove duplicate files and keep only one file per date"""
    
    log("🧹 Cleaning up duplicate files...")
    
    cleanup_results = {
        'files_removed': 0,
        'files_kept': 0,
        'errors': [],
        'removed_files': [],
        'kept_files': []
    }
    
    # Remove duplicate files
    for file_path in analysis['files_to_remove']:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleanup_results['files_removed'] += 1
                cleanup_results['removed_files'].append(file_path)
                log(f"  ❌ Removed: {os.path.basename(file_path)}")
        except Exception as e:
            error_msg = f"Error removing {file_path}: {e}"
            cleanup_results['errors'].append(error_msg)
            log(f"  ⚠️  {error_msg}")
    
    # Count kept files
    for file_path in analysis['files_to_keep']:
        if os.path.exists(file_path):
            cleanup_results['files_kept'] += 1
            cleanup_results['kept_files'].append(file_path)
    
    return cleanup_results

def verify_final_coverage() -> Dict:
    """Verify the final coverage after cleanup"""
    
    log("✅ Verifying final coverage...")
    
    code_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    
    # Find all remaining 2022 files
    remaining_files = glob.glob(f"{code_dir}/*2022*")
    
    # Extract unique dates
    date_pattern = re.compile(r'(\d{8})')
    unique_dates = set()
    
    for file_path in remaining_files:
        filename = os.path.basename(file_path)
        match = date_pattern.search(filename)
        if match:
            unique_dates.add(match.group(1))
    
    # Calculate coverage
    expected_days = 365  # 2022 is not a leap year
    coverage_percent = (len(unique_dates) / expected_days) * 100
    
    verification = {
        'total_files': len(remaining_files),
        'unique_dates': len(unique_dates),
        'expected_days': expected_days,
        'coverage_percent': coverage_percent,
        'status': 'COMPLETE' if coverage_percent >= 95 else 'INCOMPLETE'
    }
    
    return verification

def main():
    """Main execution function"""
    
    log("🚀 Starting 2022 Duplicate Cleanup")
    log("=" * 80)
    
    # Analyze current state
    analysis = analyze_2022_duplicates()
    
    log(f"📊 Current State:")
    log(f"  Total files: {analysis['total_files']:,}")
    log(f"  Unique dates: {analysis['unique_dates']:,}")
    log(f"  Dates with duplicates: {analysis['duplicate_dates']:,}")
    log(f"  Files to remove: {len(analysis['files_to_remove']):,}")
    log(f"  Files to keep: {len(analysis['files_to_keep']):,}")
    
    if analysis['duplicate_dates'] == 0:
        log("✅ No duplicates found. Nothing to clean up.")
        return
    
    log("")
    log("🧹 Starting cleanup...")
    
    # Clean up duplicates
    cleanup_results = cleanup_duplicates(analysis)
    
    log("")
    log("📊 Cleanup Results:")
    log(f"  Files removed: {cleanup_results['files_removed']:,}")
    log(f"  Files kept: {cleanup_results['files_kept']:,}")
    log(f"  Errors: {len(cleanup_results['errors']):,}")
    
    if cleanup_results['errors']:
        log("")
        log("⚠️  Errors encountered:")
        for error in cleanup_results['errors']:
            log(f"  {error}")
    
    # Verify final coverage
    log("")
    verification = verify_final_coverage()
    
    log("")
    log("=" * 80)
    log("🎯 CLEANUP COMPLETE")
    log("=" * 80)
    
    log(f"📊 Final State:")
    log(f"  Total files: {verification['total_files']:,}")
    log(f"  Unique dates: {verification['unique_dates']:,}")
    log(f"  Expected days: {verification['expected_days']:,}")
    log(f"  Coverage: {verification['coverage_percent']:.1f}%")
    log(f"  Status: {verification['status']}")
    
    if verification['status'] == 'COMPLETE':
        log("")
        log("🎯 SUCCESS! 2022 data is now properly cleaned up")
        log("  Ready for TEP correlation analysis")
    else:
        log("")
        log("⚠️  WARNING: Coverage is incomplete")
        log("  Consider investigating missing dates")
    
    log("=" * 80)

if __name__ == "__main__":
    main()
