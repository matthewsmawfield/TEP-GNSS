#!/usr/bin/env python3
"""
Step 14 2022 Alternative Archive Download
Downloads 2022 data from the 2022_M archive (Multi-GNSS format)
This archive contains 330 CLK files covering the full year

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
import glob
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

def download_file(url: str, output_path: str, file_type: str) -> Dict:
    """Download a single file with retry logic"""
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            downloaded = 0
            start_time = time.time()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            # Verify download
            if os.path.getsize(output_path) > 0:
                download_time = time.time() - start_time
                speed = (downloaded / 1024 / 1024) / download_time if download_time > 0 else 0
                
                with progress_lock:
                    progress_stats['downloaded'] += 1
                    progress_stats['total_size'] += downloaded
                
                return {
                    'success': True,
                    'size': downloaded,
                    'speed': speed,
                    'time': download_time
                }
            else:
                os.remove(output_path)
                raise Exception("Downloaded file is empty")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                with progress_lock:
                    progress_stats['failed'] += 1
                return {
                    'success': False,
                    'error': str(e)
                }
    
    return {'success': False, 'error': 'Max retries exceeded'}

def discover_2022_m_files() -> List[Dict]:
    """Discover all available 2022_M CLK files"""
    
    log("🔍 Discovering 2022_M archive files...")
    
    try:
        response = requests.get("http://ftp.aiub.unibe.ch/CODE/2022_M/", timeout=30)
        response.raise_for_status()
        
        # Parse HTML to find CLK_M.Z files
        clk_files = []
        for line in response.text.split('\n'):
            if 'CLK_M.Z' in line:
                # Extract filename and date from HTML
                filename_match = re.search(r'href="([^"]*\.CLK_M\.Z)"', line)
                date_match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', line)
                
                if filename_match and date_match:
                    filename = filename_match.group(1)
                    date_str = date_match.group(1)
                    
                    # Parse date (02-Jan-2022 -> 2022-01-02)
                    try:
                        date = datetime.strptime(date_str, '%d-%b-%Y')
                        
                        clk_files.append({
                            'filename': filename,
                            'date': date.strftime('%Y-%m-%d'),
                            'url': f"http://ftp.aiub.unibe.ch/CODE/2022_M/{filename}",
                            'output_name': f"COD0OPSFIN_{date.strftime('%Y%m%d')}0000_01D_30S_CLK.CLK.gz"
                        })
                    except ValueError:
                        continue
        
        log(f"  Discovered {len(clk_files)} CLK_M.Z files")
        return clk_files
        
    except Exception as e:
        log(f"❌ Error discovering files: {e}")
        return []

def download_worker(task: Dict) -> Dict:
    """Worker function for parallel downloads"""
    
    url = task['url']
    output_path = task['output_path']
    date = task['date']
    filename = task['filename']
    
    # Check if file already exists
    if os.path.exists(output_path):
        with progress_lock:
            progress_stats['skipped'] += 1
        return {
            'date': date,
            'status': 'SKIPPED',
            'reason': 'File already exists'
        }
    
    # Download file
    result = download_file(url, output_path, 'clk_m_z')
    
    if result['success']:
        # Convert .Z to .gz format for consistency
        if output_path.endswith('.Z'):
            gz_path = output_path.replace('.Z', '.gz')
            try:
                # Decompress .Z and recompress as .gz
                with open(output_path, 'rb') as f_in:
                    with gzip.open(gz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(output_path)
                output_path = gz_path
            except Exception as e:
                log(f"⚠️  Warning: Could not convert {filename}: {e}")
        
        return {
            'date': date,
            'status': 'DOWNLOADED',
            'size': result['size'],
            'speed': result['speed'],
            'output_path': output_path
        }
    else:
        return {
            'date': date,
            'status': 'FAILED',
            'error': result['error']
        }

def main():
    """Main execution function"""
    
    log("🚀 Starting 2022 Alternative Archive Download")
    log("   (2022_M Multi-GNSS format)")
    log("=" * 80)
    
    # Setup
    output_dir = Path("/Users/matthewsmawfield/www/TEP-GNSS/data/raw/code")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    max_workers = 8
    start_time = time.time()
    
    log(f"Output directory: {output_dir}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    
    # Discover files
    clk_files = discover_2022_m_files()
    if not clk_files:
        log("❌ No files discovered. Exiting.")
        return
    
    # Prepare download tasks
    tasks = []
    for file_info in clk_files:
        output_path = output_dir / file_info['output_name']
        tasks.append({
            'url': file_info['url'],
            'output_path': str(output_path),
            'date': file_info['date'],
            'filename': file_info['filename']
        })
    
    progress_stats['total_files'] = len(tasks)
    
    log(f"📊 Total files to download: {len(tasks):,}")
    log("")
    
    # Download files in parallel
    log("🚀 Starting 2022_M downloads...")
    log("=" * 80)
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(download_worker, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            
            # Progress update
            completed = len(results)
            progress = (completed / len(tasks)) * 100
            
            if result['status'] == 'DOWNLOADED':
                size_mb = result['size'] / 1024 / 1024
                speed = result['speed']
                log(f"✓ {result['date']} | {size_mb:.1f}MB | {speed:.1f}MB/s | {progress:.1f}% | {result['output_path']}")
            elif result['status'] == 'SKIPPED':
                log(f"⏭ {result['date']} | SKIPPED | {progress:.1f}% | {result['reason']}")
            else:
                log(f"❌ {result['date']} | FAILED | {progress:.1f}% | {result['error']}")
    
    # Final summary
    total_time = time.time() - start_time
    
    log("")
    log("=" * 80)
    log("🎯 2022_M ALTERNATIVE DOWNLOAD COMPLETE")
    log("=" * 80)
    
    downloaded = sum(1 for r in results if r['status'] == 'DOWNLOADED')
    failed = sum(1 for r in results if r['status'] == 'FAILED')
    skipped = sum(1 for r in results if r['status'] == 'SKIPPED')
    
    log(f"Total files attempted: {len(tasks):,}")
    log(f"Successfully downloaded: {downloaded:,}")
    log(f"Failed: {failed:,}")
    log(f"Skipped (already exists): {skipped:,}")
    log(f"Total size downloaded: {progress_stats['total_size'] / 1024 / 1024 / 1024:.2f} GB")
    log(f"Total time: {total_time / 60:.1f} minutes")
    log(f"Success rate: {(downloaded / len(tasks)) * 100:.1f}%")
    
    # Check final coverage
    final_files = len(glob.glob(f"{output_dir}/*2022*"))
    log(f"Recovery rate: {(final_files / 365) * 100:.1f}%")
    
    log("")
    log("📊 FINAL 2022 STATUS:")
    log(f"  Files now available: {final_files:,} / 365")
    log(f"  Coverage: {(final_files / 365) * 100:.1f}%")
    log(f"  Improvement: +{downloaded:,} files")
    
    # Save results
    results_data = {
        'download_summary': {
            'total_attempted': len(tasks),
            'downloaded': downloaded,
            'failed': failed,
            'skipped': skipped,
            'total_size_gb': progress_stats['total_size'] / 1024 / 1024 / 1024,
            'total_time_minutes': total_time / 60,
            'success_rate': (downloaded / len(tasks)) * 100
        },
        'final_status': {
            'total_files': final_files,
            'coverage_percent': (final_files / 365) * 100,
            'improvement': downloaded
        },
        'detailed_results': results
    }
    
    results_path = "/Users/matthewsmawfield/www/TEP-GNSS/results/experimental/step_14_2022_alternative_archive_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    log(f"📊 Results saved to: {results_path}")
    
    if downloaded > 0:
        log("")
        log("🎯 EXCELLENT RECOVERY!")
        log(f"  Coverage: {(final_files / 365) * 100:.1f}% - Suitable for full analysis")
        log("  Next: Rerun TEP correlation analysis with complete 2022 data")
    else:
        log("")
        log("⚠️  No new files downloaded")
        log("  Consider investigating other archive sources")
    
    log("=" * 80)

if __name__ == "__main__":
    main()
