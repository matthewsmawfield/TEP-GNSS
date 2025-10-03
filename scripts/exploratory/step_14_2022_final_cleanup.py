#!/usr/bin/env python3
"""
Step 14 2022 Final Cleanup
Removes files with invalid dates and keeps only valid 2022 calendar dates
One file per valid date only

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

def is_valid_2022_date(date_str: str) -> bool:
    """Check if a date string represents a valid 2022 calendar date"""
    try:
        # Parse YYYYMMDD format
        if len(date_str) != 8:
            return False
        
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        
        # Must be 2022
        if year != 2022:
            return False
        
        # Check if it's a valid calendar date
        datetime(year, month, day)
        return True
        
    except (ValueError, TypeError):
        return False

def analyze_2022_files() -> Dict:
    """Analyze all 2022 files and identify valid vs invalid dates"""
    
    log("🔍 Analyzing 2022 files for valid dates...")
    
    code_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    
    # Find all 2022 files
    all_2022_files = glob.glob(f"{code_dir}/*2022*")
    
    # Group files by extracted date
    files_by_date = defaultdict(list)
    date_pattern = re.compile(r'(\d{8})')  # Extract YYYYMMDD from filename
    
    valid_files = []
    invalid_files = []
    
    for file_path in all_2022_files:
        filename = os.path.basename(file_path)
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)
            if is_valid_2022_date(date_str):
                files_by_date[date_str].append(file_path)
                valid_files.append(file_path)
            else:
                invalid_files.append(file_path)
        else:
            invalid_files.append(file_path)
    
    # Analyze duplicates among valid files
    duplicate_dates = 0
    files_to_remove = []
    files_to_keep = []
    
    for date_str, file_list in files_by_date.items():
        if len(file_list) > 1:
            duplicate_dates += 1
            # Keep the first file, remove others
            files_to_keep.append(file_list[0])
            files_to_remove.extend(file_list[1:])
        else:
            files_to_keep.append(file_list[0])
    
    analysis = {
        'total_files': len(all_2022_files),
        'valid_files': len(valid_files),
        'invalid_files': len(invalid_files),
        'unique_valid_dates': len(files_by_date),
        'duplicate_dates': duplicate_dates,
        'files_to_remove': files_to_remove + invalid_files,
        'files_to_keep': files_to_keep,
        'invalid_file_list': invalid_files
    }
    
    return analysis

def cleanup_files(analysis: Dict) -> Dict:
    """Remove invalid and duplicate files"""
    
    log("🧹 Cleaning up invalid and duplicate files...")
    
    cleanup_results = {
        'files_removed': 0,
        'files_kept': 0,
        'errors': [],
        'removed_files': [],
        'kept_files': []
    }
    
    # Remove files
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

def verify_final_state() -> Dict:
    """Verify the final state after cleanup"""
    
    log("✅ Verifying final state...")
    
    code_dir = "/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code"
    
    # Find all remaining 2022 files
    remaining_files = glob.glob(f"{code_dir}/*2022*")
    
    # Extract and validate dates
    date_pattern = re.compile(r'(\d{8})')
    valid_dates = set()
    invalid_files = []
    
    for file_path in remaining_files:
        filename = os.path.basename(file_path)
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)
            if is_valid_2022_date(date_str):
                valid_dates.add(date_str)
            else:
                invalid_files.append(file_path)
        else:
            invalid_files.append(file_path)
    
    # Calculate coverage
    expected_days = 365  # 2022 is not a leap year
    coverage_percent = (len(valid_dates) / expected_days) * 100
    
    verification = {
        'total_files': len(remaining_files),
        'valid_dates': len(valid_dates),
        'invalid_files': len(invalid_files),
        'expected_days': expected_days,
        'coverage_percent': coverage_percent,
        'status': 'COMPLETE' if coverage_percent >= 95 else 'INCOMPLETE',
        'invalid_file_list': invalid_files
    }
    
    return verification

def main():
    """Main execution function"""
    
    log("🚀 Starting 2022 Final Cleanup")
    log("=" * 80)
    
    # Analyze current state
    analysis = analyze_2022_files()
    
    log(f"📊 Current State:")
    log(f"  Total files: {analysis['total_files']:,}")
    log(f"  Valid files: {analysis['valid_files']:,}")
    log(f"  Invalid files: {analysis['invalid_files']:,}")
    log(f"  Unique valid dates: {analysis['unique_valid_dates']:,}")
    log(f"  Dates with duplicates: {analysis['duplicate_dates']:,}")
    log(f"  Files to remove: {len(analysis['files_to_remove']):,}")
    log(f"  Files to keep: {len(analysis['files_to_keep']):,}")
    
    if analysis['invalid_files'] > 0:
        log("")
        log("⚠️  Invalid files found (first 10):")
        for i, file_path in enumerate(analysis['invalid_file_list'][:10]):
            log(f"  {os.path.basename(file_path)}")
        if len(analysis['invalid_file_list']) > 10:
            log(f"  ... and {len(analysis['invalid_file_list']) - 10} more")
    
    if len(analysis['files_to_remove']) == 0:
        log("✅ No files to remove. Nothing to clean up.")
        return
    
    log("")
    log("🧹 Starting cleanup...")
    
    # Clean up files
    cleanup_results = cleanup_files(analysis)
    
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
    
    # Verify final state
    log("")
    verification = verify_final_state()
    
    log("")
    log("=" * 80)
    log("🎯 FINAL CLEANUP COMPLETE")
    log("=" * 80)
    
    log(f"📊 Final State:")
    log(f"  Total files: {verification['total_files']:,}")
    log(f"  Valid dates: {verification['valid_dates']:,}")
    log(f"  Invalid files: {verification['invalid_files']:,}")
    log(f"  Expected days: {verification['expected_days']:,}")
    log(f"  Coverage: {verification['coverage_percent']:.1f}%")
    log(f"  Status: {verification['status']}")
    
    if verification['invalid_files'] > 0:
        log("")
        log("⚠️  WARNING: Invalid files still present (first 5):")
        for file_path in verification['invalid_file_list'][:5]:
            log(f"  {os.path.basename(file_path)}")
    
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
