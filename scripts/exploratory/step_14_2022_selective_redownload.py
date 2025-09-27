#!/usr/bin/env python3
"""
Step 14 2022 Selective Redownload
Attempts to redownload missing 2022 CODE files using aggressive acquisition strategy

Based on aggressive_acquire.py but focused specifically on 2022 missing data
Uses parallel downloads with retry logic and proper error handling

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-27
"""

import os
import sys
import requests
import time
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Global progress tracking
progress_lock = threading.Lock()
progress_stats = {
    'total_files': 0,
    'downloaded': 0,
    'failed': 0,
    'skipped': 0,
    'total_size': 0,
    'start_time': time.time()
}

def log(message: str) -> None:
    print(message, flush=True)

def gps_week_day_from_date(date: datetime) -> Tuple[int, int]:
    """Convert date to GPS week and day."""
    gps_epoch = datetime(1980, 1, 6)
    delta = date - gps_epoch
    gps_week = delta.days // 7
    gps_day = delta.days % 7
    return gps_week, gps_day

def convert_legacy_to_modern(legacy_file: Path, modern_file: Path) -> bool:
    """Convert legacy .CLK.Z file to modern .CLK.gz format."""
    try:
        # Try uncompress first
        temp_clk = legacy_file.with_suffix('')
        result = subprocess.run(['uncompress', '-c', str(legacy_file)], 
                              stdout=open(temp_clk, 'wb'), 
                              stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            # Compress to gzip
            with open(temp_clk, 'rb') as f_in:
                with gzip.open(modern_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            temp_clk.unlink()
            return modern_file.exists()
        else:
            # Fallback: try gzip decompression
            try:
                with gzip.open(legacy_file, 'rb') as f_in:
                    with gzip.open(modern_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return modern_file.exists()
            except:
                return False
                
    except Exception:
        return False

def download_file_2022(url: str, destination: Path, year: int, date_str: str, 
                      is_legacy: bool = False) -> Dict:
    """
    Download a single 2022 file with aggressive retry logic and progress tracking.
    """
    result = {
        'url': url,
        'destination': str(destination),
        'year': year,
        'date': date_str,
        'success': False,
        'size_bytes': 0,
        'download_time': 0,
        'error': None,
        'is_legacy': is_legacy
    }
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Create destination directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with streaming
            response = requests.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify download
                if destination.exists() and destination.stat().st_size > 0:
                    result['size_bytes'] = destination.stat().st_size
                    result['download_time'] = time.time() - start_time
                    
                    # Convert legacy format if needed
                    if is_legacy:
                        modern_filename = f"COD0OPSFIN_{year}{date_str.replace('-', '')[4:7]}0000_01D_30S_CLK.CLK.gz"
                        modern_destination = destination.parent / modern_filename
                        
                        if convert_legacy_to_modern(destination, modern_destination):
                            # Remove legacy file, keep modern
                            destination.unlink()
                            destination = modern_destination
                            result['destination'] = str(destination)
                        else:
                            result['error'] = "Legacy conversion failed"
                            return result
                    
                    result['success'] = True
                    
                    # Update global progress
                    with progress_lock:
                        progress_stats['downloaded'] += 1
                        progress_stats['total_size'] += result['size_bytes']
                    
                    # Print success
                    size_mb = result['size_bytes'] / (1024*1024)
                    speed_mbps = size_mb / result['download_time'] if result['download_time'] > 0 else 0
                    
                    with progress_lock:
                        total_done = progress_stats['downloaded'] + progress_stats['failed'] + progress_stats['skipped']
                        pct = (total_done / progress_stats['total_files']) * 100 if progress_stats['total_files'] > 0 else 0
                        elapsed = time.time() - progress_stats['start_time']
                        rate = total_done / elapsed if elapsed > 0 else 0
                        eta = (progress_stats['total_files'] - total_done) / rate if rate > 0 else 0
                    
                    log(f"✓ {date_str} | {size_mb:.1f}MB | {speed_mbps:.1f}MB/s | {pct:.1f}% | ETA:{eta/60:.0f}m | SUCCESS | {destination.name}")
                    return result
                else:
                    result['error'] = "File not created or empty"
            else:
                result['error'] = f"HTTP {response.status_code}"
                
        except Exception as e:
            result['error'] = str(e)
        
        # Retry logic
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
    
    # All retries failed
    with progress_lock:
        progress_stats['failed'] += 1
    
    log(f"✗ {date_str} | FAILED | {result['error']} | {destination.name}")
    return result

def generate_2022_missing_tasks() -> List[Dict]:
    """Generate download tasks for missing 2022 dates only."""
    
    log("🔍 Analyzing existing 2022 files to determine missing dates...")
    
    # Get existing 2022 files
    existing_files = []
    code_dir = Path('/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code')
    
    for file_path in code_dir.glob('*2022*'):
        # Extract date from filename like COD0OPSFIN_20223310000_01D_30S_CLK.CLK.gz
        match = re.search(r'(\d{7})', file_path.name)
        if match:
            year_doy = match.group(1)
            year = int(year_doy[:4])
            doy = int(year_doy[4:])
            
            # Convert day of year to date
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            existing_files.append(date)
    
    existing_files.sort()
    log(f"  Found {len(existing_files)} existing 2022 files")
    
    # Generate all 2022 dates
    all_2022_dates = []
    current_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    while current_date <= end_date:
        all_2022_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Find missing dates
    existing_date_strs = {d.strftime('%Y-%m-%d') for d in existing_files}
    all_date_strs = {d.strftime('%Y-%m-%d') for d in all_2022_dates}
    missing_date_strs = all_date_strs - existing_date_strs
    
    missing_dates = [datetime.strptime(d, '%Y-%m-%d') for d in missing_date_strs]
    missing_dates.sort()
    
    log(f"  Missing {len(missing_dates)} out of 365 total 2022 dates")
    log(f"  Will attempt to download {len(missing_dates)} missing files")
    
    # Generate download tasks for missing dates
    tasks = []
    
    for date in missing_dates:
        date_str = date.strftime('%Y-%m-%d')
        doy = date.timetuple().tm_yday
        year = 2022
        
        # Determine format and URL
        # Modern format started mid-2022 (around November 27, day 331)
        if doy >= 331:
            # Modern format
            filename = f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
            url = f"http://ftp.aiub.unibe.ch/CODE/{year}/{filename}"
            is_legacy = False
        else:
            # Legacy format
            gps_week, gps_day = gps_week_day_from_date(date)
            filename = f"COD{gps_week:04d}{gps_day}.CLK.Z"
            url = f"http://ftp.aiub.unibe.ch/CODE/{year}/{filename}"
            is_legacy = True
        
        tasks.append({
            'year': year,
            'date': date,
            'date_str': date_str,
            'url': url,
            'filename': filename,
            'is_legacy': is_legacy
        })
    
    return tasks

def download_worker_2022(task: Dict, output_dir: Path) -> Dict:
    """Worker function for parallel 2022 downloads."""
    year = task['year']
    date_str = task['date_str']
    url = task['url']
    filename = task['filename']
    is_legacy = task['is_legacy']
    
    # Use main CODE directory directly
    destination = output_dir / filename
    
    # Check if already exists and has reasonable size (>1MB)
    if destination.exists() and destination.stat().st_size > 1048576:  # 1MB
        with progress_lock:
            progress_stats['skipped'] += 1
        size_mb = destination.stat().st_size / (1024*1024)
        log(f"⏭ {date_str} | {size_mb:.1f}MB | SKIPPED | {filename}")
        return {'success': True, 'skipped': True, 'size_bytes': destination.stat().st_size}
    elif destination.exists() and destination.stat().st_size <= 1048576:
        # File exists but is too small, remove it and re-download
        destination.unlink()
        log(f"🔄 {date_str} | RE-DOWNLOADING (file too small) | {filename}")
    
    # Download file
    log(f"📥 {date_str} | DOWNLOADING | {filename}")
    return download_file_2022(url, destination, year, date_str, is_legacy)

def main():
    """Main 2022 selective redownload function."""
    
    log("🚀 Starting 2022 Selective Redownload")
    log("   (Attempting to recover missing 2022 CODE files)")
    log("=" * 80)
    
    # Configuration
    max_workers = 8  # Slightly reduced for focused download
    output_dir = Path('/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code')
    
    log(f"Target: Missing 2022 CODE files only")
    log(f"Parallel workers: {max_workers}")
    log(f"Output directory: {output_dir}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate download tasks for missing 2022 files only
    tasks = generate_2022_missing_tasks()
    progress_stats['total_files'] = len(tasks)
    
    if len(tasks) == 0:
        log("✅ No missing 2022 files found - all data already downloaded!")
        return {'success': True, 'message': 'No missing files'}
    
    log(f"📊 Total missing files to download: {len(tasks):,}")
    log("")
    
    # Start parallel downloads
    log("🚀 Starting selective 2022 downloads...")
    log("=" * 80)
    
    results = []
    start_time = time.time()
    progress_stats['start_time'] = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(download_worker_2022, task, output_dir): task 
            for task in tasks
        }
        
        # Process completed downloads
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = future_to_task[future]
                log(f"✗ {task['date_str']} | EXCEPTION | {str(e)[:50]}...")
                results.append({'success': False, 'error': str(e)})
    
    # Final summary
    total_time = time.time() - start_time
    
    log("")
    log("=" * 80)
    log("🎯 2022 SELECTIVE REDOWNLOAD COMPLETE")
    log("=" * 80)
    
    successful = sum(1 for r in results if r.get('success', False) and not r.get('skipped', False))
    failed = sum(1 for r in results if not r.get('success', False))
    skipped = sum(1 for r in results if r.get('skipped', False))
    total_size_gb = progress_stats['total_size'] / (1024*1024*1024)
    
    log(f"Total missing files attempted: {len(tasks):,}")
    log(f"Successfully downloaded: {successful:,}")
    log(f"Failed: {failed:,}")
    log(f"Skipped (already exists): {skipped:,}")
    log(f"Total size downloaded: {total_size_gb:.2f} GB")
    log(f"Total time: {total_time/60:.1f} minutes")
    
    if len(tasks) > 0:
        log(f"Success rate: {(successful/len(tasks))*100:.1f}%")
        log(f"Recovery rate: {((successful + skipped)/len(tasks))*100:.1f}%")
    
    # Calculate final 2022 coverage
    final_existing = len([f for f in output_dir.glob('*2022*')])
    final_coverage = (final_existing / 365) * 100
    
    log(f"\n📊 FINAL 2022 STATUS:")
    log(f"  Files now available: {final_existing} / 365")
    log(f"  Coverage: {final_coverage:.1f}%")
    log(f"  Improvement: +{successful} files")
    
    # Save results
    results_summary = {
        'redownload_date': datetime.now().isoformat(),
        'target_year': 2022,
        'missing_files_attempted': len(tasks),
        'successful_downloads': successful,
        'failed_downloads': failed,
        'skipped_files': skipped,
        'total_size_gb': total_size_gb,
        'total_time_minutes': total_time/60,
        'success_rate': (successful/len(tasks))*100 if len(tasks) > 0 else 0,
        'final_coverage_percent': final_coverage,
        'parallel_workers': max_workers
    }
    
    results_file = Path('/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_2022_redownload_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    log(f"\n📊 Results saved to: {results_file}")
    
    # Recommendations based on results
    if successful > 0:
        log(f"\n🎯 NEXT STEPS:")
        log(f"1. Rerun processing pipeline with recovered data:")
        log(f"   python scripts/experimental/step_14_extended_code_analysis.py")
        log(f"2. Update analysis to include recovered 2022 data")
        log(f"3. Compare results with/without 2022 data")
        
        if final_coverage > 50:
            log(f"\n✅ GOOD NEWS: {final_coverage:.1f}% coverage may be sufficient for analysis!")
        else:
            log(f"\n⚠️  NOTE: {final_coverage:.1f}% coverage still quite low for reliable analysis")
    else:
        log(f"\n❌ NO FILES RECOVERED:")
        log(f"   • 2022 files may no longer be available in CODE archive")
        log(f"   • Server may be temporarily unavailable")
        log(f"   • Network connectivity issues")
        log(f"\n💡 RECOMMENDATION: Proceed with 2014-2021 analysis as originally planned")
    
    log("=" * 80)
    
    return results_summary

if __name__ == "__main__":
    main()
