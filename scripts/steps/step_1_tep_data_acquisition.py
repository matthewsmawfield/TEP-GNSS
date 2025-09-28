#!/usr/bin/env python3
"""
TEP-GNSS Data Acquisition - Robust Implementation

Based on the proven aggressive_acquire.py approach with:
- Proper file existence checking with size validation
- Parallel downloads with real-time progress tracking
- Comprehensive error handling and retry logic
- Complete date range coverage (2023-2025)

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

# Ensure package root on sys.path for intra-package imports
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Import TEP utilities
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPNetworkError, TEPFileError, 
    safe_csv_read, safe_json_read, safe_json_write
)
from scripts.utils.logger import TEPLogger

# Instantiate the logger
logger = TEPLogger()

def print_status(message, level="INFO"):
    """Enhanced status printing with timestamp and color coding."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Color coding for different levels
    colors = {
        "TITLE": "\033[1;36m",    # Cyan bold
        "SUCCESS": "\033[1;32m",  # Green bold
        "WARNING": "\033[1;33m",  # Yellow bold
        "ERROR": "\033[1;31m",    # Red bold
        "INFO": "\033[0;37m",     # White
        "DEBUG": "\033[0;90m",    # Dark gray
        "PROCESS": "\033[0;34m"   # Blue
    }
    reset = "\033[0m"

    color = colors.get(level, colors["INFO"])

    if level == "TITLE":
        print(f"\n{color}{'='*80}")
        print(f"[{timestamp}] {message}")
        print(f"{'='*80}{reset}\n")
    else:
        print(f"{color}[{timestamp}] [{level}] {message}{reset}")

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

def calculate_geomagnetic_coordinates(lat_deg: float, lon_deg: float, height_m: float, date_decimal: float = 2024.0):
    """Calculate geomagnetic coordinates using IGRF model."""
    try:
        import pyIGRF
        
        height_km = height_m / 1000.0
        mag_components = pyIGRF.igrf_value(lat_deg, lon_deg, height_km, date_decimal)
        inclination = mag_components[1]  # In degrees
        
        # Calculate geomagnetic latitude from inclination using dipole approximation
        geomag_lat = math.degrees(math.atan(math.tan(math.radians(inclination)) / 2.0))
        geomag_lon = lon_deg  # Simplified approach
        
        return geomag_lat, geomag_lon
        
    except Exception as e:
        if not hasattr(calculate_geomagnetic_coordinates, '_error_logged'):
            logger.warning(f"Geomagnetic calculation failed: {e}")
            if "No module named 'pyIGRF'" in str(e):
                logger.info("Install pyIGRF with: pip install pyigrf==0.3.3")
            calculate_geomagnetic_coordinates._error_logged = True
        return None, None

def fetch_igs_coordinates():
    """Fetch coordinates from IGS network JSON"""
    def _fetch_operation():
        url = "https://files.igs.org/pub/station/general/IGSNetworkWithFormer.json"
        logger.process("Fetching IGS network coordinates...")
        
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
        
        logger.success(f"Retrieved {len(rows)} stations from IGS network")
        return pd.DataFrame(rows)
    
    result = SafeErrorHandler.safe_network_operation(
        _fetch_operation,
        error_message="IGS coordinate fetch failed",
        logger_func=logger.warning,
        return_on_error=pd.DataFrame(),
        max_retries=2
    )
    return result if result is not None else pd.DataFrame()

def add_geomagnetic_coordinates(coords_df: pd.DataFrame) -> pd.DataFrame:
    """Add geomagnetic coordinates to station coordinate dataframe."""
    logger.process("Calculating geomagnetic coordinates for all stations...")
    
    required_cols = ['lat_deg', 'lon_deg', 'height_m']
    missing_cols = [col for col in required_cols if col not in coords_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns for geomagnetic calculation: {missing_cols}")
        return coords_df
    
    coords_df['geomag_lat'] = None
    coords_df['geomag_lon'] = None
    
    successful_calculations = 0
    failed_calculations = 0
    
    for idx, row in coords_df.iterrows():
        if pd.notna(row['lat_deg']) and pd.notna(row['lon_deg']) and pd.notna(row['height_m']):
            geomag_lat, geomag_lon = calculate_geomagnetic_coordinates(
                row['lat_deg'], row['lon_deg'], row['height_m']
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
        logger.success(f"Geomagnetic coordinate calculation complete: {successful_calculations} successful, {failed_calculations} failed")
    else:
        logger.warning(f"Geomagnetic coordinate calculation: {failed_calculations} failed (pyIGRF not installed)")
    
    return coords_df

def build_station_catalogue():
    """Build comprehensive station catalogue from IGS"""
    logger.process("Building comprehensive coordinate catalogue...")
    
    # Fetch from IGS
    igs_df = fetch_igs_coordinates()
    if len(igs_df) == 0:
        logger.error("No coordinate sources available")
        return None
    
    # Deduplicate by code
    dedup = igs_df.drop_duplicates(subset=['code'], keep='first')
    
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
    logger.success(f"Built coordinate catalogue: {len(result_df)} unique stations")
    logger.success(f"Stations with valid coordinates: {stations_with_coords}/{len(result_df)} ({100*stations_with_coords/len(result_df):.1f}%)")
    logger.success(f"Geomagnetic coordinates: {valid_geomag}/{len(result_df)} stations ({100*valid_geomag/len(result_df):.1f}%)")

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
    
    min_size_bytes = int(min_size_mb * 1024 * 1024)
    
    # Check if file already exists and has sufficient size
    if destination.exists() and destination.stat().st_size >= min_size_bytes:
        result['success'] = True
        result['skipped'] = True
        result['size_bytes'] = destination.stat().st_size
        return result
    elif destination.exists() and destination.stat().st_size < min_size_bytes:
        # File exists but is too small, remove and re-download
        destination.unlink()
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Create destination directory
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Create SSL context for HTTPS URLs
            ssl_context = ssl.create_default_context() if url.startswith('https') else None
            
            # Download with urllib
            with urllib.request.urlopen(url, context=ssl_context, 
                                      timeout=TEPConfig.get_int('TEP_DOWNLOAD_TIMEOUT', 60)) as response:
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
        
        # Retry logic with exponential backoff
        if attempt < max_retries - 1:
            time.sleep(retry_delay * (2 ** attempt))
    
    return result

def download_worker(task: Dict) -> Dict:
    """Worker function for parallel downloads - based on aggressive_acquire.py"""
    url = task['url']
    destination = task['destination']
    date_str = task['date_str']
    
    # Check if already exists and has reasonable size (>1MB)
    if destination.exists() and destination.stat().st_size > 1048576:  # 1MB
        size_mb = destination.stat().st_size / (1024*1024)
        return {
            'success': True, 
            'skipped': True, 
            'size_bytes': destination.stat().st_size,
            'destination': str(destination),
            'date_str': date_str
        }
    elif destination.exists() and destination.stat().st_size <= 1048576:
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
    # Get date range from configuration
    try:
        date_start_s, date_end_s = TEPConfig.get_date_range()
        ds = datetime.fromisoformat(date_start_s)
        de = datetime.fromisoformat(date_end_s)
        if de < ds:
            ds, de = de, ds
        date_list = [ds + timedelta(days=i) for i in range((de - ds).days + 1)]
        logger.info(f"Using date filter {ds.date()} → {de.date()} ({len(date_list)} days)")
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
        igs_url = f"https://igs.bkg.bund.de/root_ftp/IGS/products/{week:04d}/IGS0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        igs_dst = raw_dir / "igs_combined" / f"IGS0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        tasks.append({
            'center': 'IGS',
            'url': igs_url,
            'destination': igs_dst,
            'date_str': date.strftime('%Y-%m-%d')
        })
        
        # CODE
        code_url = f"http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        code_dst = raw_dir / "code" / f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        tasks.append({
            'center': 'CODE',
            'url': code_url,
            'destination': code_dst,
            'date_str': date.strftime('%Y-%m-%d')
        })
        
        # ESA Final
        esa_url = f"http://navigation-office.esa.int/products/gnss-products/{week}/ESA0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
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
    
    logger.info(f"Existing clock files: IGS:{existing_igs} CODE:{existing_code} ESA:{existing_esa}")
    
    # Generate all download tasks
    logger.process("Generating download tasks...")
    all_tasks = generate_download_tasks()
    
    # Group tasks by center
    igs_tasks = [t for t in all_tasks if t['center'] == 'IGS']
    code_tasks = [t for t in all_tasks if t['center'] == 'CODE']
    esa_tasks = [t for t in all_tasks if t['center'] == 'ESA']
    
    logger.success(f"Tasks generated: IGS:{len(igs_tasks)} CODE:{len(code_tasks)} ESA:{len(esa_tasks)}")
    
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
    logger.info("PARALLEL CLOCK FILE ACQUISITION")
    
    for center, tasks in [('IGS', igs_tasks), ('CODE', code_tasks), ('ESA', esa_tasks)]:
        if not tasks:
            continue
            
        logger.process(f"Processing {center}: {len(tasks)} files with {max_workers} workers")
        
        # Filter to only missing files
        missing_tasks = []
        existing_count = 0
        
        for task in tasks:
            if task['destination'].exists() and task['destination'].stat().st_size > 1048576:
                existing_count += 1
            else:
                missing_tasks.append(task)
        
        # Update global progress
        with progress_lock:
            global_progress['total_files'] += len(tasks)
            global_progress['total_missing'] += len(missing_tasks)
        
        if existing_count > 0:
            logger.success(f"{center}: {existing_count} files already exist")
        
        if missing_tasks:
            logger.process(f"{center}: Downloading {len(missing_tasks)} missing files...")
            
            # Parallel download of missing files
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(download_worker, task): task for task in missing_tasks}
                
                downloaded = 0
                failed = 0
                
                for future in as_completed(future_to_task):
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
                                
                                # Clean progress logging
                                logger.success(f"{center}: {Path(result['destination']).name} ({size_mb:.1f}MB) | Progress: {progress_pct:.1f}% | ETA: {eta_minutes:.0f}m")
                                
                                # Overall progress update every 25 files
                                if downloaded % 25 == 0:
                                    with progress_lock:
                                        total_done = global_progress['downloaded'] + global_progress['failed']
                                        logger.info(f"[PROGRESS] Overall: {total_done}/{global_progress['total_missing']} files ({progress_pct:.1f}%) | Rate: {rate:.1f} files/sec")
                        else:
                            failed += 1
                            with progress_lock:
                                global_progress['failed'] += 1
                            logger.debug(f"{center} failed: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        failed += 1
                        with progress_lock:
                            global_progress['failed'] += 1
                        logger.debug(f"{center} exception: {e}")
                
                logger.success(f"{center} complete: {downloaded} downloaded, {failed} failed")
        else:
            logger.success(f"{center}: All files already exist")
    
    # Final summary with clean formatting
    final_igs = len(list((raw_dir / "igs_combined").glob("*.CLK.gz")))
    final_code = len(list((raw_dir / "code").glob("*.CLK.gz")))
    final_esa = len(list((raw_dir / "esa_final").glob("*.CLK.gz")))
    
    total_time = time.time() - global_progress['start_time']
    
    logger.info("ACQUISITION COMPLETE")
    logger.success(f"Final clock files: IGS:{final_igs} CODE:{final_code} ESA:{final_esa}")
    logger.success(f"Total time: {total_time/60:.1f} minutes")
    logger.success(f"Downloaded: {global_progress['downloaded']} files")
    
    if global_progress['failed'] > 0:
        logger.warning(f"Failed: {global_progress['failed']} files")
    
    if global_progress['downloaded'] > 0:
        avg_speed = global_progress['downloaded'] / total_time if total_time > 0 else 0
        logger.success(f"Average speed: {avg_speed:.1f} files/sec")
    
    # Validate minimum requirements
    if final_igs < 1 or final_code < 1 or final_esa < 1:
        logger.error("CRITICAL: Insufficient clock files downloaded")
        return False

    return True

def main():
    """Main data acquisition function"""
    print_status("TEP GNSS Analysis Package v0.13 - STEP 1: Data Acquisition", "TITLE")

    # Create logs directory
    (PACKAGE_ROOT / "logs").mkdir(exist_ok=True)

    # Build station catalogue
    logger.process("Building comprehensive station catalogue from authoritative sources")
    logger.process("Fetching coordinates from authoritative sources...")
    
    coords_df = build_station_catalogue()
    if coords_df is None or len(coords_df) == 0:
        logger.error("Station catalogue building failed")
        return False

    # Save station catalogue
    coord_path = PACKAGE_ROOT / "data" / "coordinates" / "station_coords_global.csv"
    coord_path.parent.mkdir(parents=True, exist_ok=True)
    coords_df.to_csv(coord_path, index=False)
    
    stations_with_coords = coords_df['has_coordinates'].sum()
    logger.success(f"Station catalogue built: {len(coords_df)} stations saved to {Path(coord_path).name}")
    logger.info(f"Coordinate verification summary:")
    logger.info(f"  Total stations in catalogue: {len(coords_df)}")
    logger.success(f"  Stations with valid coordinates: {stations_with_coords}")
    logger.success(f"  Verified stations for analysis: {stations_with_coords}")
    logger.success(f"Final station catalogue: {len(coords_df)} stations ({stations_with_coords} with valid coordinates)")
    
    # Download clock files
    success = download_clock_files()
    if not success:
        logger.error("Clock file download failed")
        return False

    logger.success("Data acquisition completed successfully")
    logger.info("Ready for coordinate validation (Step 2)")
    return True

if __name__ == "__main__":
    main()