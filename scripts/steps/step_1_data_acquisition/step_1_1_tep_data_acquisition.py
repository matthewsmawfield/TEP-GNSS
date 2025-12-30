#!/usr/bin/env python3
"""
TEP-GNSS Data Acquisition - STEP 1.1: Robust Implementation

Based on the proven aggressive_acquire.py approach with:
- Proper file existence checking with size validation
- Parallel downloads with real-time progress tracking
- Comprehensive error handling and retry logic
- Complete date range coverage (2023-2025)

Inputs:
  - None (fetches data from external sources)

Outputs:
  - data/coordinates/step_1_1_station_coords_global.csv
  - data/raw/igs_combined/*.CLK.gz
  - data/raw/code/*.CLK.gz
  - data/raw/esa_final/*.CLK.gz
  - results/outputs/step_1_1_data_acquisition.json

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import sys
import os
import time
import urllib.request
import urllib.error
import ssl
import json
import math
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Ensure the directory containing 'scripts' is on the path
script_dir = Path(__file__).resolve()
project_root = script_dir.parents[3] # Corrected to point to project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import TEP utilities after path setup
from scripts.utils.pid_manager import ensure_single_instance

# Define PACKAGE_ROOT for consistent usage throughout the script
PACKAGE_ROOT = project_root

# Import TEP utilities
from scripts.utils.config import TEPConfig
from scripts.utils.logger import print_status, TEPLogger, set_step_logger
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPNetworkError, TEPFileError, TEPAnalysisError,
    safe_csv_read, safe_json_read, safe_json_write
)

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_1_1_data_acquisition",
    level="DEBUG",
    log_file_path=PACKAGE_ROOT / "logs" / "step_1_1_data_acquisition.log"
)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float):
    """Convert geodetic coordinates to ECEF"""
    # WGS84 constants
    a = 6378137.0  # Semi-major axis
    e2 = 6.69437999014e-3  # First eccentricity squared
    
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    X = (N + h_m) * math.cos(lat) * math.cos(lon)
    Y = (N + h_m) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - e2) + h_m) * sin_lat
    return X, Y, Z

def calculate_geomagnetic_coordinates(lat_deg: float, lon_deg: float, height_m: float, date: datetime):
    """Calculate geomagnetic coordinates (Apex coordinates) using the ApexPy library.
    
    This function utilizes the ApexPy library to perform a more accurate transformation
    from geographic coordinates to magnetic apex coordinates, which are a robust
    approximation of geomagnetic dipole coordinates.
    
    Args:
        lat_deg (float): Geographic latitude in degrees.
        lon_deg (float): Geographic longitude in degrees.
        height_m (float): Altitude in meters.
        date (datetime): The date for which to calculate the geomagnetic coordinates.
        
    Returns:
        Tuple[float, float]: A tuple containing (apex_latitude, apex_longitude) in degrees.
    """
    try:
        from apexpy import Apex
        
        height_km = height_m / 1000.0
        
        # Initialize Apex object for the given date
        apex = Apex(date=date)
        
        # Convert geographic coordinates to apex coordinates
        apex_lat, apex_lon = apex.geo2apex(lat_deg, lon_deg, height_km)
        
        return apex_lat, apex_lon
        
    except Exception as e:
        # If apexpy is not installed, raise an ImportError which will be caught by main()
        if "No module named 'apexpy'" in str(e):
            raise ImportError("apexpy module not found. Please install it with: pip install apexpy") from e
        else:
            # Log other unexpected errors but still return None, None to allow partial operation
            step_logger.error(f"An unexpected error occurred during geomagnetic calculation: {e}")
            return None, None

def fetch_igs_coordinates():
    """Fetch coordinates from IGS network JSON"""
    def _fetch_operation():
        url = TEPConfig.get_str('TEP_IGS_COORDS_URL')
        print_status("Fetching IGS network coordinates...", "PROCESS")
        
        ssl_context = ssl.create_default_context()
        timeout = TEPConfig.get_int('TEP_NETWORK_TIMEOUT')
        
        with urllib.request.urlopen(url, context=ssl_context, timeout=timeout) as response:
            data = json.load(response)
        
        rows = []
        for code9, meta in data.items():
            code = code9[:4].upper()
            try:
                X = float(meta["X"])
                Y = float(meta["Y"])
                Z = float(meta["Z"])
                lat = float(meta.get("Latitude", 0))
                lon = float(meta.get("Longitude", 0))
                h = float(meta.get("Height", 0))
                
                # Normalize longitude to -180 to +180 range
                if lon > 180:
                    lon = lon - 360
                
                rows.append({
                    'code': code9,
                    'coord_source_code': code,
                    'lat_deg': lat,
                    'lon_deg': lon,
                    'height_m': h,
                    'X': X,
                    'Y': Y,
                    'Z': Z,
                    'source': 'IGS'
                })
            except (KeyError, ValueError, TypeError):
                continue
        
        print_status(f"Retrieved {len(rows)} stations from IGS network", "SUCCESS")
        return pd.DataFrame(rows)
    
    result = SafeErrorHandler.safe_network_operation(
        _fetch_operation,
        error_message="IGS coordinate fetch failed",
        logger_func=step_logger.warning,
        return_on_error=pd.DataFrame(),
        max_retries=2
    )
    return result if result is not None else pd.DataFrame()

def add_geomagnetic_coordinates(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Add geomagnetic coordinates to station coordinate dataframe."""
    print_status("Calculating geomagnetic coordinates for all stations...", "PROCESS")
    
    required_cols = ['lat_deg', 'lon_deg', 'height_m']
    missing_cols = [col for col in required_cols if col not in coords_df.columns]
    
    if missing_cols:
        print_status(f"Missing required columns for geomagnetic calculation: {missing_cols}", "ERROR")
        return coords_df
    
    coords_df['geomag_lat'] = None
    coords_df['geomag_lon'] = None
    
    successful_calculations = 0
    failed_calculations = 0
    
    for idx, row in coords_df.iterrows():
        if pd.notna(row['lat_deg']) and pd.notna(row['lon_deg']) and pd.notna(row['height_m']):
            geomag_lat, geomag_lon = calculate_geomagnetic_coordinates(
                row['lat_deg'], row['lon_deg'], row['height_m'], row['date']
            )
            
            if geomag_lat is not None and geomag_lon is not None:
                coords_df.at[idx, 'geomag_lat'] = geomag_lat
                coords_df.at[idx, 'geomag_lon'] = geomag_lon
                successful_calculations += 1
            else:
                failed_calculations += 1
        else:
            failed_calculations += 1
    
    if successful_calculations > 0:
        print_status(f"Geomagnetic coordinate calculation complete: {successful_calculations} successful, {failed_calculations} failed", "SUCCESS")
    else:
        print_status(f"Geomagnetic coordinate calculation: {failed_calculations} failed (apexpy not installed)", "WARNING")
    
    return coords_df

def build_station_catalogue():
    """Build comprehensive station catalogue from IGS"""
    print_status("Building comprehensive coordinate catalogue...", "PROCESS")
    
    # Fetch from IGS
    igs_df = fetch_igs_coordinates()
    if len(igs_df) == 0:
        print_status("No coordinate sources available", "ERROR")
        return None
    
    # Deduplicate by code
    dedup = igs_df.drop_duplicates(subset=['code'], keep='first')
    
    # Add a date column for geomagnetic calculation, using the start date of the analysis period
    from datetime import datetime
    start_date_str = TEPConfig.get_str('TEP_DATE_START')
    analysis_date = datetime.fromisoformat(start_date_str)
    dedup['date'] = analysis_date

    # Add geomagnetic coordinates
    dedup_with_geomag = add_geomagnetic_coordinates(dedup)
    
    # Add coordinate validation flag
    dedup_with_geomag['has_coordinates'] = (
        dedup_with_geomag['X'].apply(lambda x: pd.notna(x) and np.isfinite(x) and x != 0) &
        dedup_with_geomag['Y'].apply(lambda x: pd.notna(x) and np.isfinite(x) and x != 0) &
        dedup_with_geomag['Z'].apply(lambda x: pd.notna(x) and np.isfinite(x) and x != 0)
    )

    # Reorder columns
    columns = [
        'code', 'coord_source_code', 'lat_deg', 'lon_deg', 'height_m', 'X', 'Y', 'Z',
        'has_coordinates', 'geomag_lat', 'geomag_lon'
    ]

    for col in columns:
        if col not in dedup_with_geomag.columns:
            if col == 'coord_source_code':
                dedup_with_geomag[col] = dedup_with_geomag['code'].str[:4]
            else:
                dedup_with_geomag[col] = None

    result_df = dedup_with_geomag[columns].copy()
    
    # Report statistics
    valid_geomag = result_df['geomag_lat'].notna().sum()
    stations_with_coords = result_df['has_coordinates'].sum()
    print_status(f"Built coordinate catalogue: {len(result_df)} unique stations", "SUCCESS")
    print_status(f"Stations with valid coordinates: {stations_with_coords}/{len(result_df)} ({100*stations_with_coords/len(result_df):.1f}%)", "SUCCESS")
    print_status(f"Geomagnetic coordinates: {valid_geomag}/{len(result_df)} stations ({100*valid_geomag/len(result_df):.1f}%)", "SUCCESS")

    return result_df

def download_file_robust(url: str, destination: Path, min_size_mb: float = 1.0) -> Dict:
    """
    Download a file with robust retry logic and size validation.
    Based on aggressive_acquire.py approach.
    """
    result = {
        'url': url,
        'destination': str(destination),
        'success': False,
        'size_bytes': 0,
        'download_time': 0,
        'error': None,
        'skipped': False
    }
    
    # Use configurable minimum file size
    min_size_bytes = int(TEPConfig.get_float('TEP_MIN_FILE_SIZE_MB', min_size_mb) * 1024 * 1024)
    
    # Check if file already exists and has sufficient size
    if destination.exists() and destination.stat().st_size >= min_size_bytes:
        result['success'] = True
        result['skipped'] = True
        result['size_bytes'] = destination.stat().st_size
        return result
    elif destination.exists() and destination.stat().st_size < min_size_bytes:
        # File exists but is too small, remove and re-download
        destination.unlink()
    
    max_retries = 5  # Increased from 3 to 5
    retry_delay = 2  # Increased base delay
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Create destination directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Create SSL context for HTTPS URLs
            ssl_context = ssl.create_default_context() if url.startswith('https') else None
            
            # Increase timeout for retries
            timeout = TEPConfig.get_int('TEP_DOWNLOAD_TIMEOUT', 60) + (attempt * 30)
            
            # Download with urllib
            with urllib.request.urlopen(url, context=ssl_context, timeout=timeout) as response:
                data = response.read()
            
            with open(destination, 'wb') as f:
                f.write(data)
            
            # Verify download size
            if destination.exists() and destination.stat().st_size >= min_size_bytes:
                result['size_bytes'] = destination.stat().st_size
                result['download_time'] = time.time() - start_time
                result['success'] = True
                return result
            else:
                result['error'] = f"File too small: {destination.stat().st_size if destination.exists() else 0} bytes"
                
        except Exception as e:
            result['error'] = str(e)
            # Clean up partial download
            if destination.exists():
                destination.unlink()
        
        # Enhanced retry logic with exponential backoff and jitter
        if attempt < max_retries - 1:
            # Add jitter to prevent thundering herd
            import random
            jitter = random.uniform(0.5, 1.5)
            delay = retry_delay * (2 ** attempt) * jitter
            time.sleep(min(delay, 120))  # Cap at 2 minutes max delay
    
    return result

def download_worker(task: Dict) -> Dict:
    """Worker function for parallel downloads - based on aggressive_acquire.py"""
    url = task['url']
    destination = task['destination']
    date_str = task['date_str']
    
    # Check if already exists and has reasonable size (>1MB)
    min_file_size_mb = TEPConfig.get_float('TEP_MIN_FILE_SIZE_MB', 1.0)
    if destination.exists() and destination.stat().st_size > int(min_file_size_mb * 1024 * 1024):  # 1MB
        size_mb = destination.stat().st_size / (1024*1024)
        return {
            'success': True, 
            'skipped': True, 
            'size_bytes': destination.stat().st_size,
            'destination': str(destination),
            'date_str': date_str
        }
    elif destination.exists() and destination.stat().st_size <= int(min_file_size_mb * 1024 * 1024):
        # File exists but is too small, remove it and re-download
        destination.unlink()
    
    # Download file
    return download_file_robust(url, destination)

def gps_week_from_date(date: datetime) -> int:
    """Convert UTC date to GPS week number."""
    gps_epoch = datetime(1980, 1, 6)
    return int((date - gps_epoch).days // 7)

def day_of_year(date: datetime) -> int:
    """Get day of year from date."""
    return date.timetuple().tm_yday

def generate_download_tasks() -> List[Dict]:
    """Generate download tasks for all three analysis centers"""
    # Get date range from configuration (TEP_DATE_START, TEP_DATE_END in scripts/utils/config.py or environment variables)
    try:
        date_start_s, date_end_s = TEPConfig.get_date_range()
        ds = datetime.fromisoformat(date_start_s)
        de = datetime.fromisoformat(date_end_s)
        if de < ds:
            ds, de = de, ds
        date_list = [ds + timedelta(days=i) for i in range((de - ds).days + 1)]
        print_status(f"Using date filter {ds.date()} → {de.date()} ({len(date_list)} days)", "INFO")
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Invalid date configuration: {e}")
    
    raw_dir = PACKAGE_ROOT / "data" / "raw"
    tasks = []
    
    # Generate tasks for each analysis center
    for date in date_list:
        year = date.year
        doy = day_of_year(date)
        week = gps_week_from_date(date)
        
        # IGS Combined
        igs_url = TEPConfig.get_str('TEP_IGS_CLK_URL_TEMPLATE').format(week=week, year=year, doy=doy)
        igs_dst = raw_dir / "igs_combined" / f"IGS0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        tasks.append({
            'center': 'IGS',
            'url': igs_url,
            'destination': igs_dst,
            'date_str': date.strftime('%Y-%m-%d')
        })
        
        # CODE
        code_url = TEPConfig.get_str('TEP_CODE_CLK_URL_TEMPLATE').format(year=year, doy=doy)
        code_dst = raw_dir / "code" / f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        tasks.append({
            'center': 'CODE',
            'url': code_url,
            'destination': code_dst,
            'date_str': date.strftime('%Y-%m-%d')
        })
        
        # ESA Final
        esa_url = TEPConfig.get_str('TEP_ESA_CLK_URL_TEMPLATE').format(week=week, year=year, doy=doy)
        esa_dst = raw_dir / "esa_final" / f"ESA0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        tasks.append({
            'center': 'ESA',
            'url': esa_url,
            'destination': esa_dst,
            'date_str': date.strftime('%Y-%m-%d')
        })
    
    return tasks

def download_clock_files():
    """Download clock files using robust parallel approach with enhanced progress tracking"""
    import threading
    
    # Create directories
    raw_dir = PACKAGE_ROOT / "data" / "raw"
    (raw_dir / "igs_combined").mkdir(parents=True, exist_ok=True)
    (raw_dir / "code").mkdir(parents=True, exist_ok=True)
    (raw_dir / "esa_final").mkdir(parents=True, exist_ok=True)
    
    # Check existing files
    existing_igs = len(list((raw_dir / "igs_combined").glob("*.CLK.gz")))
    existing_code = len(list((raw_dir / "code").glob("*.CLK.gz")))
    existing_esa = len(list((raw_dir / "esa_final").glob("*.CLK.gz")))
    
    print_status(f"Existing clock files: IGS:{existing_igs} CODE:{existing_code} ESA:{existing_esa}", "INFO")
    
    # Generate all download tasks
    print_status("Generating download tasks...", "PROCESS")
    all_tasks = generate_download_tasks()
    
    # Group tasks by center
    igs_tasks = [t for t in all_tasks if t['center'] == 'IGS']
    code_tasks = [t for t in all_tasks if t['center'] == 'CODE']
    esa_tasks = [t for t in all_tasks if t['center'] == 'ESA']
    
    print_status(f"Tasks generated: IGS:{len(igs_tasks)} CODE:{len(code_tasks)} ESA:{len(esa_tasks)}", "SUCCESS")
    
    # Global progress tracking
    progress_lock = threading.Lock()
    global_progress = {
        'total_files': 0,
        'total_missing': 0,
        'downloaded': 0,
        'failed': 0,
        'start_time': time.time()
    }
    
    # Download each center with enhanced progress tracking
    max_workers = TEPConfig.get_int('TEP_MAX_PARALLEL_DOWNLOADS', 14)
    results = {'IGS': [], 'CODE': [], 'ESA': []}
    
    # Print header
    print_status("PARALLEL CLOCK FILE ACQUISITION", "TITLE")
    
    for center, tasks in [('IGS', igs_tasks), ('CODE', code_tasks), ('ESA', esa_tasks)]:
        if not tasks:
            continue
            
        print_status(f"Processing {center}: {len(tasks)} files with {max_workers} workers", "PROCESS")
        
        # Filter to only missing files
        missing_tasks = []
        existing_count = 0
        
        for task in tasks:
            if task['destination'].exists() and task['destination'].stat().st_size > int(TEPConfig.get_float('TEP_MIN_FILE_SIZE_MB', 1.0) * 1024 * 1024):
                existing_count += 1
            else:
                missing_tasks.append(task)
        
        # Update global progress
        with progress_lock:
            global_progress['total_files'] += len(tasks)
            global_progress['total_missing'] += len(missing_tasks)
        
        if existing_count > 0:
            print_status(f"{center}: {existing_count} files already exist", "SUCCESS")
        
        if missing_tasks:
            print_status(f"{center}: Downloading {len(missing_tasks)} missing files...", "PROCESS")
            
            # Parallel download of missing files
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(download_worker, task): task for task in missing_tasks}
                
                downloaded = 0
                failed = 0
                
                failed_tasks = []
                
                for future in as_completed(future_to_task):
                    original_task = future_to_task[future]
                    try:
                        result = future.result()
                        results[center].append(result)
                        
                        if result['success']:
                            if not result.get('skipped', False):
                                downloaded += 1
                                size_mb = result['size_bytes'] / (1024*1024)
                                
                                # Update global progress
                                with progress_lock:
                                    global_progress['downloaded'] += 1
                                    total_done = global_progress['downloaded'] + global_progress['failed']
                                    elapsed = time.time() - global_progress['start_time']
                                    rate = total_done / elapsed if elapsed > 0 else 0
                                    eta_seconds = (global_progress['total_missing'] - total_done) / rate if rate > 0 else 0
                                    eta_minutes = eta_seconds / 60
                                    
                                    progress_pct = (total_done / global_progress['total_missing']) * 100 if global_progress['total_missing'] > 0 else 0
                                
                                # Progress reporting with scientific precision
                                print_status(f"{center}: {Path(result['destination']).name} ({size_mb:.1f}MB) | Progress: {progress_pct:.1f}% | ETA: {eta_minutes:.0f}min", "SUCCESS")
                                
                                # Overall progress update every 25 files
                                if downloaded % 25 == 0:
                                    with progress_lock:
                                        total_done = global_progress['downloaded'] + global_progress['failed']
                                        print_status(f"[PROGRESS] Overall: {total_done}/{global_progress['total_missing']} files ({progress_pct:.1f}%) | Rate: {rate:.1f} files/sec", "INFO")
                        else:
                            failed += 1
                            failed_tasks.append(original_task)  # Store failed task for retry
                            with progress_lock:
                                global_progress['failed'] += 1
                            print_status(f"{center} download failed: {result.get('error', 'Unknown error')}", "DEBUG")
                            
                    except Exception as e:
                        failed += 1
                        failed_tasks.append(original_task)  # Store failed task for retry
                        with progress_lock:
                            global_progress['failed'] += 1
                        print_status(f"{center} processing exception: {e}", "DEBUG")
                
                print_status(f"{center} initial acquisition phase: {downloaded} files acquired, {failed} files failed", "SUCCESS")
                
                # Retry failed downloads with enhanced retry logic
                if failed_tasks:
                    print_status(f"{center}: Initiating retry sequence for {len(failed_tasks)} failed downloads", "PROCESS")
                    retry_downloaded = 0
                    retry_failed = 0
                    
                    # Use fewer workers for retries to be more conservative
                    retry_workers = min(max_workers // 2, 4)
                    
                    with ThreadPoolExecutor(max_workers=retry_workers) as retry_executor:
                        retry_futures = {retry_executor.submit(download_worker, task): task for task in failed_tasks}
                        
                        for future in as_completed(retry_futures):
                            try:
                                result = future.result()
                                results[center].append(result)
                                
                                if result['success']:
                                    retry_downloaded += 1
                                    size_mb = result['size_bytes'] / (1024*1024)
                                    print_status(f"{center} retry successful: {Path(result['destination']).name} ({size_mb:.1f}MB)", "SUCCESS")
                                    
                                    # Update global progress
                                    with progress_lock:
                                        global_progress['downloaded'] += 1
                                        global_progress['failed'] -= 1  # Remove from failed count
                                else:
                                    retry_failed += 1
                                    print_status(f"{center} retry unsuccessful: {result.get('error', 'Unknown error')}", "WARNING")
                                    
                            except Exception as e:
                                retry_failed += 1
                                print_status(f"{center} retry exception: {e}", "WARNING")
                    
                    print_status(f"{center} retry phase complete: {retry_downloaded} files recovered, {retry_failed} files remain failed", "SUCCESS")
                    downloaded += retry_downloaded
                    failed = retry_failed  # Update failed count to only remaining failures
                
                print_status(f"{center} acquisition summary: {downloaded} files successfully acquired, {failed} files failed", "SUCCESS")
        else:
            print_status(f"{center}: All files already exist", "SUCCESS")
    
    # Final summary with clean formatting
    final_igs = len(list((raw_dir / "igs_combined").glob("*.CLK.gz")))
    final_code = len(list((raw_dir / "code").glob("*.CLK.gz")))
    final_esa = len(list((raw_dir / "esa_final").glob("*.CLK.gz")))
    
    total_time = time.time() - global_progress['start_time']
    
    print_status("ACQUISITION COMPLETE", "INFO")
    print_status(f"Final clock files: IGS:{final_igs} CODE:{final_code} ESA:{final_esa}", "SUCCESS")
    print_status(f"Total time: {total_time/60:.1f} minutes", "SUCCESS")
    print_status(f"Downloaded: {global_progress['downloaded']} files", "SUCCESS")
    
    if global_progress['failed'] > 0:
        print_status(f"Failed: {global_progress['failed']} files", "WARNING")
    
    if global_progress['downloaded'] > 0:
        avg_speed = global_progress['downloaded'] / total_time if total_time > 0 else 0
        print_status(f"Average speed: {avg_speed:.1f} files/sec", "SUCCESS")
    
    # Validate minimum requirements
    if final_igs < 1 or final_code < 1 or final_esa < 1:
        print_status("CRITICAL: Insufficient clock files downloaded", "ERROR")
        return False

    # Return progress information for logging
    return {
        'success': True,
        'downloaded': global_progress['downloaded'],
        'failed': global_progress['failed'],
        'total_time': total_time,
        'final_igs': final_igs,
        'final_code': final_code,
        'final_esa': final_esa
    }

@ensure_single_instance
def main():
    """Main data acquisition function"""
    set_step_logger(step_logger)
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING} - STEP 1.1: Data Acquisition", "TITLE")

    # Verify apexpy dependency
    try:
        from apexpy import Apex
    except ImportError:
        print_status("apexpy module not found. Please install it with: pip install apexpy", "ERROR")
        print_status("Geomagnetic coordinate calculation will be skipped.", "INFO")
        return False # This is now a critical dependency

    # Create logs directory
    (PACKAGE_ROOT / "logs").mkdir(exist_ok=True)

    # Build station catalogue
    print_status("Building comprehensive station catalogue from authoritative sources", "PROCESS")
    print_status("Fetching coordinates from authoritative sources...", "PROCESS")
    
    coords_df = build_station_catalogue()
    if coords_df is None or len(coords_df) == 0:
        print_status("Station catalogue building failed", "ERROR")
        return False

    # Validate minimum station count
    min_stations = TEPConfig.get_int('TEP_MIN_STATIONS', 0)
    if len(coords_df) < min_stations:
        print_status(f"CRITICAL: Insufficient stations ({len(coords_df)}) after catalogue build. Minimum required: {min_stations}", "ERROR")
        return False

    # Save station catalogue
    coord_path = PACKAGE_ROOT / "data" / "coordinates" / "step_1_1_station_coords_global.csv"
    coord_path.parent.mkdir(parents=True, exist_ok=True)
    coords_df.to_csv(coord_path, index=False)
    
    stations_with_coords = coords_df['has_coordinates'].sum()
    print_status(f"Station catalogue built: {len(coords_df)} stations saved to {Path(coord_path).name}", "SUCCESS")
    print_status(f"Coordinate verification summary:", "INFO")
    print_status(f"  Total stations in catalogue: {len(coords_df)}", "INFO")
    print_status(f"  Stations with valid coordinates: {stations_with_coords}", "SUCCESS")
    print_status(f"  Verified stations for analysis: {stations_with_coords}", "SUCCESS")
    print_status(f"Final station catalogue: {len(coords_df)} stations ({stations_with_coords} with valid coordinates)", "SUCCESS")
    
    # Download clock files
    download_result = download_clock_files()
    if not download_result or not download_result.get('success', False):
        print_status("Clock file download failed", "ERROR")
        return False

    print_status("Data acquisition completed successfully", "SUCCESS")
    print_status("Ready for coordinate validation (Step 1.2)", "INFO")

    # Create and save completion log
    total_time = download_result['total_time']
    avg_speed = download_result['downloaded'] / total_time if total_time > 0 else 0
    
    completion_log = {
        "script_name": "step_1_1_tep_data_acquisition.py",
        "timestamp": datetime.now().isoformat(),
        "total_stations_in_catalogue": int(len(coords_df)),
        "stations_with_valid_coordinates": int(stations_with_coords),
        "total_clock_files_downloaded": int(download_result['downloaded']),
        "total_clock_files_failed": int(download_result['failed']),
        "total_time_minutes": float(total_time / 60),
        "average_download_speed_files_per_second": float(avg_speed),
        "final_clock_files_count": {
            "igs": int(download_result['final_igs']),
            "code": int(download_result['final_code']),
            "esa": int(download_result['final_esa'])
        }
    }
    # Save to results/outputs directory (for pipeline consistency)
    results_path = PACKAGE_ROOT / "results" / "outputs" / "step_1_1_data_acquisition.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(completion_log, f, indent=4)
    print_status(f"Results saved to {Path(results_path).name}", "SUCCESS")
    print_status(f"Detailed logs available in step_1_1_data_acquisition.log", "SUCCESS")

    # Update provenance snapshot
    try:
        from scripts.utils.provenance import update_provenance_snapshot
        provenance_success = update_provenance_snapshot('step_1_1_tep_data_acquisition', {
            'step_1_1_completion': completion_log
        })
        if provenance_success:
            print_status("Provenance snapshot updated", "SUCCESS")
        else:
            print_status("Failed to update provenance snapshot", "WARNING")
    except ImportError:
        print_status("Provenance utility not available - skipping provenance update", "WARNING")
    except Exception as e:
        print_status(f"Error updating provenance: {e}", "WARNING")

    return True

if __name__ == "__main__":
    main()