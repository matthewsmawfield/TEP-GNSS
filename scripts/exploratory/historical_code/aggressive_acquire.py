#!/usr/bin/env python3
"""
Aggressive Historical CODE Data Acquisition (2010-2025)

High-performance parallel download of 15.5 years of CODE data with:
- 10 parallel downloads
- Real-time progress tracking
- Detailed success/failure reporting
- Automatic retry logic
- Format detection and conversion

Author: TEP-GNSS Analysis Pipeline
Date: 2025-09-25
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

def print_progress_header():
    """Print aggressive acquisition header."""
    print("=" * 120)
    print("🚀 AGGRESSIVE HISTORICAL CODE DATA ACQUISITION")
    print("=" * 120)
    cutoff_date = datetime(2025, 6, 30)
    print(f"Target: 2010-{cutoff_date.strftime('%Y-%m-%d')} (through end of June 2025)")
    print(f"Expected files: ~5,650")
    print(f"Parallel downloads: 10")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

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
                              stderr=subprocess.PIPE, capture_output=True)
        
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

def download_file_aggressive(url: str, destination: Path, year: int, date_str: str, 
                           is_legacy: bool = False) -> Dict:
    """
    Download a single file with aggressive retry logic and progress tracking.
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
                    
                    print(f"✓ {date_str} | {size_mb:.1f}MB | {speed_mbps:.1f}MB/s | {pct:.1f}% | ETA:{eta/60:.0f}m | SUCCESS | {destination.name}", flush=True)
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
    
    print(f"✗ {date_str} | FAILED | {result['error']} | {destination.name}", flush=True)
    return result

def generate_download_tasks(start_year: int, end_year: int, cutoff_date: datetime = None) -> List[Dict]:
    """Generate all download tasks for the specified year range."""
    tasks = []
    
    for year in range(start_year, end_year + 1):
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        
        # For the final year, don't go beyond the cutoff date
        if year == end_year and cutoff_date:
            year_end = min(year_end, cutoff_date)
        
        current_date = year_start
        while current_date <= year_end:
            date_str = current_date.strftime('%Y-%m-%d')
            doy = current_date.timetuple().tm_yday
            
            # Determine format and URL
            # Modern format started mid-2022 (around November 27, day 331)
            if year >= 2023 or (year == 2022 and doy >= 331):
                # Modern format
                filename = f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
                url = f"http://ftp.aiub.unibe.ch/CODE/{year}/{filename}"
                is_legacy = False
            else:
                # Legacy format
                gps_week, gps_day = gps_week_day_from_date(current_date)
                filename = f"COD{gps_week:04d}{gps_day}.CLK.Z"
                url = f"http://ftp.aiub.unibe.ch/CODE/{year}/{filename}"
                is_legacy = True
            
            tasks.append({
                'year': year,
                'date': current_date,
                'date_str': date_str,
                'url': url,
                'filename': filename,
                'is_legacy': is_legacy
            })
            
            current_date += timedelta(days=1)
    
    return tasks

def download_worker(task: Dict, output_dir: Path) -> Dict:
    """Worker function for parallel downloads."""
    year = task['year']
    date_str = task['date_str']
    url = task['url']
    filename = task['filename']
    is_legacy = task['is_legacy']
    
    # Create year directory
    year_dir = output_dir / str(year)
    destination = year_dir / filename
    
    # Check if already exists and has reasonable size (>1MB)
    if destination.exists() and destination.stat().st_size > 1048576:  # 1MB
        with progress_lock:
            progress_stats['skipped'] += 1
        size_mb = destination.stat().st_size / (1024*1024)
        print(f"⏭ {date_str} | {size_mb:.1f}MB | SKIPPED | {filename}", flush=True)
        return {'success': True, 'skipped': True, 'size_bytes': destination.stat().st_size}
    elif destination.exists() and destination.stat().st_size <= 1048576:
        # File exists but is too small, remove it and re-download
        destination.unlink()
        print(f"🔄 {date_str} | RE-DOWNLOADING (file too small) | {filename}", flush=True)
    
    # Download file
    print(f"📥 {date_str} | DOWNLOADING | {filename}", flush=True)
    return download_file_aggressive(url, destination, year, date_str, is_legacy)

def main():
    """Main aggressive acquisition function."""
    print_progress_header()
    
    # Configuration
    start_year = 2010
    # Use end of June 2025 as specified
    cutoff_date = datetime(2025, 6, 30)
    end_year = cutoff_date.year
    max_workers = 10
    output_dir = Path('data/historical_code/raw')
    
    print(f"Years: {start_year}-{end_year}")
    print(f"Parallel workers: {max_workers}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all download tasks
    print("Generating download tasks...")
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')} (end of June 2025)")
    tasks = generate_download_tasks(start_year, end_year, cutoff_date)
    progress_stats['total_files'] = len(tasks)
    
    print(f"Total files to process: {len(tasks):,}")
    print()
    
    # Start parallel downloads
    print("🚀 Starting aggressive parallel downloads...")
    print("=" * 120)
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(download_worker, task, output_dir): task 
            for task in tasks
        }
        
        # Process completed downloads
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = future_to_task[future]
                print(f"✗ {task['date_str']} | EXCEPTION | {str(e)[:50]}...")
                results.append({'success': False, 'error': str(e)})
    
    # Final summary
    total_time = time.time() - start_time
    
    print()
    print("=" * 120)
    print("🎯 AGGRESSIVE ACQUISITION COMPLETE")
    print("=" * 120)
    
    successful = sum(1 for r in results if r.get('success', False) and not r.get('skipped', False))
    failed = sum(1 for r in results if not r.get('success', False))
    skipped = sum(1 for r in results if r.get('skipped', False))
    total_size_gb = progress_stats['total_size'] / (1024*1024*1024)
    
    print(f"Total files: {len(tasks):,}")
    print(f"Successfully downloaded: {successful:,}")
    print(f"Failed: {failed:,}")
    print(f"Skipped (already exists): {skipped:,}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {len(tasks)/(total_time/60):.1f} files/minute")
    print(f"Success rate: {(successful/len(tasks))*100:.1f}%")
    
    # Save results
    results_summary = {
        'acquisition_date': datetime.now().isoformat(),
        'years': f"{start_year}-{end_year}",
        'total_files': len(tasks),
        'successful_downloads': successful,
        'failed_downloads': failed,
        'skipped_files': skipped,
        'total_size_gb': total_size_gb,
        'total_time_minutes': total_time/60,
        'success_rate': (successful/len(tasks))*100,
        'parallel_workers': max_workers
    }
    
    results_file = Path('results/historical_code/aggressive_acquisition_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    if successful > 0:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Copy files to main pipeline:")
        print(f"   find {output_dir} -name '*.gz' -exec cp {{}} data/raw/code/ \\;")
        print(f"2. Update date range:")
        print(f"   export TEP_DATE_START=\"{start_year}-01-01\"")
        print(f"   export TEP_DATE_END=\"2025-06-30\"")
        print(f"3. Run extended analysis:")
        print(f"   python scripts/experimental/step_14_extended_code_analysis.py")
        print(f"\n🚀 Ready for 15.5-year TEP analysis!")
    
    print("=" * 120)
    
    return results_summary

if __name__ == "__main__":
    main()
