#!/usr/bin/env python3
"""
Step 14 2022 Comprehensive Redownload
Downloads ALL available 2022 CODE files in multiple formats discovered in archive

Based on archive investigation:
- 330 legacy format files: COD#####.CLK.Z 
- 35 modern format files: COD0OPSFIN_*_30S_CLK.CLK.gz
- Plus _v3 variants and other formats

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

def download_file_comprehensive(url: str, destination: Path, year: int, date_str: str, 
                               file_type: str = "unknown") -> Dict:
    """Download a single 2022 file with comprehensive retry logic."""
    result = {
        'url': url,
        'destination': str(destination),
        'year': year,
        'date': date_str,
        'file_type': file_type,
        'success': False,
        'size_bytes': 0,
        'download_time': 0,
        'error': None
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
                    
                    log(f"✓ {date_str} | {size_mb:.1f}MB | {speed_mbps:.1f}MB/s | {pct:.1f}% | ETA:{eta/60:.0f}m | {file_type} | {destination.name}")
                    return result
                else:
                    result['error'] = "File not created or empty"
            else:
                result['error'] = f"HTTP {response.status_code}"
                
        except Exception as e:
            result['error'] = str(e)
        
        # Retry logic
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))
    
    # All retries failed
    with progress_lock:
        progress_stats['failed'] += 1
    
    log(f"✗ {date_str} | FAILED | {result['error']} | {file_type} | {destination.name}")
    return result

def discover_2022_files() -> List[Dict]:
    """Discover all available 2022 files from CODE archive."""
    
    log("🔍 Discovering all available 2022 CODE files...")
    
    # Get archive listing
    try:
        response = requests.get("http://ftp.aiub.unibe.ch/CODE/2022/", timeout=30)
        if response.status_code != 200:
            log(f"  Error accessing archive: HTTP {response.status_code}")
            return []
        
        archive_html = response.text
    except Exception as e:
        log(f"  Error accessing archive: {e}")
        return []
    
    # Parse available files
    discovered_files = []
    
    # Pattern 1: Legacy format COD#####.CLK.Z
    legacy_pattern = r'<a href="(COD\d{5}\.CLK\.Z)">'
    legacy_matches = re.findall(legacy_pattern, archive_html)
    
    for filename in legacy_matches:
        # Extract GPS week and day from filename
        match = re.match(r'COD(\d{4})(\d)\.CLK\.Z', filename)
        if match:
            gps_week = int(match.group(1))
            gps_day = int(match.group(2))
            
            # Convert to date
            gps_epoch = datetime(1980, 1, 6)
            date = gps_epoch + timedelta(weeks=gps_week, days=gps_day)
            
            if date.year == 2022:
                discovered_files.append({
                    'filename': filename,
                    'url': f"http://ftp.aiub.unibe.ch/CODE/2022/{filename}",
                    'date': date,
                    'date_str': date.strftime('%Y-%m-%d'),
                    'file_type': 'legacy_clk_z',
                    'format': 'CLK.Z'
                })
    
    # Pattern 2: Modern format COD0OPSFIN_*_30S_CLK.CLK.gz
    modern_pattern = r'<a href="(COD0OPSFIN_(\d{7})0000_01D_30S_CLK\.CLK\.gz)">'
    modern_matches = re.findall(modern_pattern, archive_html)
    
    for filename, year_doy in modern_matches:
        year = int(year_doy[:4])
        doy = int(year_doy[4:])
        
        if year == 2022:
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            discovered_files.append({
                'filename': filename,
                'url': f"http://ftp.aiub.unibe.ch/CODE/2022/{filename}",
                'date': date,
                'date_str': date.strftime('%Y-%m-%d'),
                'file_type': 'modern_30s_clk_gz',
                'format': '30S_CLK.gz'
            })
    
    # Pattern 3: V3 variants COD#####_v3.CLK.Z
    v3_pattern = r'<a href="(COD(\d{5})_v3\.CLK\.Z)">'
    v3_matches = re.findall(v3_pattern, archive_html)
    
    for filename, week_day in v3_matches:
        gps_week = int(week_day[:4])
        gps_day = int(week_day[4])
        
        # Convert to date
        gps_epoch = datetime(1980, 1, 6)
        date = gps_epoch + timedelta(weeks=gps_week, days=gps_day)
        
        if date.year == 2022:
            discovered_files.append({
                'filename': filename,
                'url': f"http://ftp.aiub.unibe.ch/CODE/2022/{filename}",
                'date': date,
                'date_str': date.strftime('%Y-%m-%d'),
                'file_type': 'legacy_v3_clk_z',
                'format': 'v3_CLK.Z'
            })
    
    # Sort by date
    discovered_files.sort(key=lambda x: x['date'])
    
    # Remove duplicates (prefer modern format, then v3, then legacy)
    unique_files = {}
    for file_info in discovered_files:
        date_key = file_info['date_str']
        
        if date_key not in unique_files:
            unique_files[date_key] = file_info
        else:
            # Priority: modern > v3 > legacy
            current = unique_files[date_key]
            if (file_info['file_type'] == 'modern_30s_clk_gz' or
                (file_info['file_type'] == 'legacy_v3_clk_z' and current['file_type'] == 'legacy_clk_z')):
                unique_files[date_key] = file_info
    
    final_files = list(unique_files.values())
    final_files.sort(key=lambda x: x['date'])
    
    log(f"  Discovered files by type:")
    log(f"    Legacy CLK.Z: {len(legacy_matches)} files")
    log(f"    Modern 30S: {len(modern_matches)} files") 
    log(f"    V3 variants: {len(v3_matches)} files")
    log(f"    Unique dates after deduplication: {len(final_files)} files")
    
    return final_files

def download_worker_comprehensive(file_info: Dict, output_dir: Path) -> Dict:
    """Worker function for comprehensive 2022 downloads."""
    
    filename = file_info['filename']
    url = file_info['url']
    date_str = file_info['date_str']
    file_type = file_info['file_type']
    
    # Determine output filename (normalize to modern format)
    if file_type == 'modern_30s_clk_gz':
        output_filename = filename
    else:
        # Convert legacy to modern naming
        date = file_info['date']
        doy = date.timetuple().tm_yday
        output_filename = f"COD0OPSFIN_{date.year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
    
    destination = output_dir / output_filename
    
    # Check if already exists and has reasonable size (>1MB)
    if destination.exists() and destination.stat().st_size > 1048576:
        with progress_lock:
            progress_stats['skipped'] += 1
        size_mb = destination.stat().st_size / (1024*1024)
        log(f"⏭ {date_str} | {size_mb:.1f}MB | SKIPPED | {file_type} | {output_filename}")
        return {'success': True, 'skipped': True, 'size_bytes': destination.stat().st_size}
    elif destination.exists() and destination.stat().st_size <= 1048576:
        destination.unlink()
        log(f"🔄 {date_str} | RE-DOWNLOADING (file too small) | {file_type} | {output_filename}")
    
    # Download to temporary location first
    temp_destination = destination.with_suffix('.tmp')
    
    log(f"📥 {date_str} | DOWNLOADING | {file_type} | {filename}")
    result = download_file_comprehensive(url, temp_destination, 2022, date_str, file_type)
    
    if result['success']:
        # Convert if needed
        if file_type in ['legacy_clk_z', 'legacy_v3_clk_z']:
            # Convert from .CLK.Z to .CLK.gz
            if convert_legacy_to_modern(temp_destination, destination):
                temp_destination.unlink()  # Remove temp file
                result['destination'] = str(destination)
                log(f"🔄 {date_str} | CONVERTED | {file_type} -> modern format")
            else:
                result['success'] = False
                result['error'] = "Legacy conversion failed"
                if temp_destination.exists():
                    temp_destination.unlink()
        else:
            # Just rename temp to final
            temp_destination.rename(destination)
    else:
        # Clean up failed download
        if temp_destination.exists():
            temp_destination.unlink()
    
    return result

def main():
    """Main comprehensive 2022 redownload function."""
    
    log("🚀 Starting 2022 COMPREHENSIVE Redownload")
    log("   (All available formats: legacy, modern, v3 variants)")
    log("=" * 80)
    
    # Configuration
    max_workers = 8
    output_dir = Path('/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code')
    
    log(f"Target: ALL available 2022 CODE files")
    log(f"Parallel workers: {max_workers}")
    log(f"Output directory: {output_dir}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all available files
    discovered_files = discover_2022_files()
    progress_stats['total_files'] = len(discovered_files)
    
    if len(discovered_files) == 0:
        log("❌ No 2022 files discovered in archive!")
        return {'success': False, 'message': 'No files found'}
    
    log(f"📊 Total files to download: {len(discovered_files):,}")
    log("")
    
    # Start parallel downloads
    log("🚀 Starting comprehensive 2022 downloads...")
    log("=" * 80)
    
    results = []
    start_time = time.time()
    progress_stats['start_time'] = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(download_worker_comprehensive, file_info, output_dir): file_info 
            for file_info in discovered_files
        }
        
        # Process completed downloads
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                file_info = future_to_file[future]
                log(f"✗ {file_info['date_str']} | EXCEPTION | {str(e)[:50]}...")
                results.append({'success': False, 'error': str(e)})
    
    # Final summary
    total_time = time.time() - start_time
    
    log("")
    log("=" * 80)
    log("🎯 2022 COMPREHENSIVE REDOWNLOAD COMPLETE")
    log("=" * 80)
    
    successful = sum(1 for r in results if r.get('success', False) and not r.get('skipped', False))
    failed = sum(1 for r in results if not r.get('success', False))
    skipped = sum(1 for r in results if r.get('skipped', False))
    total_size_gb = progress_stats['total_size'] / (1024*1024*1024)
    
    log(f"Total files attempted: {len(discovered_files):,}")
    log(f"Successfully downloaded: {successful:,}")
    log(f"Failed: {failed:,}")
    log(f"Skipped (already exists): {skipped:,}")
    log(f"Total size downloaded: {total_size_gb:.2f} GB")
    log(f"Total time: {total_time/60:.1f} minutes")
    
    if len(discovered_files) > 0:
        log(f"Success rate: {(successful/len(discovered_files))*100:.1f}%")
        log(f"Recovery rate: {((successful + skipped)/len(discovered_files))*100:.1f}%")
    
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
        'discovered_files': len(discovered_files),
        'successful_downloads': successful,
        'failed_downloads': failed,
        'skipped_files': skipped,
        'total_size_gb': total_size_gb,
        'total_time_minutes': total_time/60,
        'success_rate': (successful/len(discovered_files))*100 if len(discovered_files) > 0 else 0,
        'final_coverage_percent': final_coverage,
        'parallel_workers': max_workers,
        'file_types_processed': list(set([f['file_type'] for f in discovered_files]))
    }
    
    results_file = Path('/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_2022_comprehensive_redownload_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    log(f"\n📊 Results saved to: {results_file}")
    
    # Recommendations based on results
    if final_coverage >= 70:
        log(f"\n🎯 EXCELLENT RECOVERY!")
        log(f"  Coverage: {final_coverage:.1f}% - Suitable for full analysis")
        log(f"  Next: Rerun TEP correlation analysis with complete 2022 data")
    elif final_coverage >= 50:
        log(f"\n✅ GOOD RECOVERY!")
        log(f"  Coverage: {final_coverage:.1f}% - Suitable for analysis with documentation")
        log(f"  Next: Include 2022 in analysis with gap documentation")
    elif final_coverage >= 25:
        log(f"\n⚠️  MODERATE RECOVERY")
        log(f"  Coverage: {final_coverage:.1f}% - Use with caution")
        log(f"  Next: Consider supplementary analysis only")
    else:
        log(f"\n❌ INSUFFICIENT RECOVERY")
        log(f"  Coverage: {final_coverage:.1f}% - Still too low")
        log(f"  Next: Exclude 2022 from main analysis")
    
    log("=" * 80)
    
    return results_summary

if __name__ == "__main__":
    main()
