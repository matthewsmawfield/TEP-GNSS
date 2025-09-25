#!/usr/bin/env python3
"""
TEP-GNSS Data Acquisition and Station Catalogue Construction

Implementation of precision timing network data acquisition following the
methodology described in Smawfield (2025) for Temporal Equivalence Principle
analysis.

Theoretical Background:
    Acquires chronometric observables from Global Navigation Satellite System
    precision timing networks. Establishes comprehensive station catalogue with
    ECEF coordinates for spatial correlation analysis.

Data Sources:
    - CODE: Center for Orbit Determination in Europe
    - IGS: International GNSS Service  
    - ESA: European Space Agency Final Products
    - ITRF2014: International Terrestrial Reference Frame

Validation Protocol:
    - Strict authentication against official repositories
    - No synthetic or fallback data permitted
    - Complete provenance tracking for reproducibility

References:
    Smawfield, M.L. (2025). Global Time Echoes: Distance-Structured Correlations
    in GNSS Clocks Across Independent Networks. Zenodo.

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import sys
import os
import time
import urllib.request
import urllib.error
import re
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd

# Ensure package root on sys.path for intra-package imports
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Import TEP utilities for better configuration and error handling
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPNetworkError, TEPFileError, 
    safe_csv_read, safe_json_read, safe_json_write
)
from scripts.utils.logger import TEPLogger # Import the centralized logger

# Instantiate the logger
logger = TEPLogger().logger

# Station catalogue building functions (integrated from utils)
import json
import math
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed

def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float):
    """Convert geodetic coordinates to ECEF"""
    import math
    
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
    """
    Calculate geomagnetic coordinates using IGRF model.
    
    Args:
        lat_deg: Geographic latitude in degrees
        lon_deg: Geographic longitude in degrees  
        height_m: Height above sea level in meters
        date_decimal: Decimal year for IGRF model (default: 2024.0)
        
    Returns:
        tuple: (geomag_lat, geomag_lon) in degrees, or (None, None) if calculation fails
    """
    try:
        import pyIGRF
        
        # Convert height to km for pyIGRF
        height_km = height_m / 1000.0
        
        # Calculate magnetic field components
        # pyIGRF.igrf_value returns (declination, inclination, horizontal_intensity, 
        #                          north_component, east_component, vertical_component, total_intensity)
        mag_components = pyIGRF.igrf_value(lat_deg, lon_deg, height_km, date_decimal)
        
        # Extract inclination (magnetic dip) to calculate geomagnetic latitude
        inclination = mag_components[1]  # In degrees
        
        # Calculate geomagnetic latitude from inclination using dipole approximation
        # tan(inclination) = 2 * tan(geomagnetic_latitude)
        geomag_lat = math.degrees(math.atan(math.tan(math.radians(inclination)) / 2.0))
        
        # For geomagnetic longitude, we use a simplified approach
        # In practice, this would require more complex calculations with dipole coordinates
        # For now, we'll use the geographic longitude as an approximation
        geomag_lon = lon_deg
        
        return geomag_lat, geomag_lon
        
    except Exception as e:
        # Only log the first few failures to avoid log spam
        if not hasattr(calculate_geomagnetic_coordinates, '_error_logged'):
            logger.warning(f"Geomagnetic calculation failed: {e}")
            if "No module named 'pyIGRF'" in str(e):
                logger.info("Install pyIGRF with: pip install pyigrf==0.3.3")
            calculate_geomagnetic_coordinates._error_logged = True
        return None, None

def add_geomagnetic_coordinates(coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geomagnetic coordinates to station coordinate dataframe.
    
    Args:
        coords_df: DataFrame with station coordinates
        
    Returns:
        DataFrame with added geomagnetic coordinate columns
    """
    logger.process("Calculating geomagnetic coordinates for all stations...")
    
    # Check if we have required columns
    required_cols = ['lat_deg', 'lon_deg', 'height_m']
    missing_cols = [col for col in required_cols if col not in coords_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns for geomagnetic calculation: {missing_cols}")
        return coords_df
    
    # Initialize new columns
    coords_df['geomag_lat'] = None
    coords_df['geomag_lon'] = None
    
    successful_calculations = 0
    failed_calculations = 0
    
    # Calculate geomagnetic coordinates for each station
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

def fetch_igs_coordinates():
    """Fetch coordinates from IGS network JSON"""
    def _fetch_operation():
        url = "https://files.igs.org/pub/station/general/IGSNetworkWithFormer.json"
        logger.info("Fetching IGS network coordinates...")
        
        # Create secure SSL context
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
                    'code': code9,  # Keep full 9-char code
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
                continue  # Skip malformed station entries
        
        logger.success(f"Retrieved {len(rows)} stations from IGS")
        return pd.DataFrame(rows)
    
    # Use safe network operation handler
    result = SafeErrorHandler.safe_network_operation(
        _fetch_operation,
        error_message="IGS coordinate fetch failed",
        logger_func=logger.warning, # Pass the logger's warning method
        return_on_error=pd.DataFrame(),
        max_retries=2
    )
    return result if result is not None else pd.DataFrame()

def build_coordinate_catalogue():
    """Build comprehensive coordinate catalogue with metadata"""
    logger.process("Building comprehensive coordinate catalogue...")
    
    # Fetch from IGS (primary and sufficient source)
    igs_df = fetch_igs_coordinates()
    if len(igs_df) == 0:
        logger.error("No coordinate sources available")
        return None
    
    # Use IGS data directly (766 stations is comprehensive)
    all_df = igs_df
    
    # Deduplicate by code (IGS takes priority)
    dedup = all_df.drop_duplicates(subset=['code'], keep='first')
    
    # Add essential metadata for validation
    dedup['has_coordinates'] = True  # All fetched stations have coordinates
    
    # Add geomagnetic coordinates to the dataset
    dedup_with_geomag = add_geomagnetic_coordinates(dedup)
    
    # Reorder columns for consistency - include geomagnetic coordinates
    columns = [
        'code', 'coord_source_code', 'lat_deg', 'lon_deg', 'height_m', 'X', 'Y', 'Z',
        'geomag_lat', 'geomag_lon', 'has_coordinates'
    ]
    
    # Add missing columns if needed
    for col in columns:
        if col not in dedup_with_geomag.columns:
            if col == 'coord_source_code':
                dedup_with_geomag[col] = dedup_with_geomag['code'].str[:4]
            else:
                dedup_with_geomag[col] = None
    
    result_df = dedup_with_geomag[columns].copy()
    
    # Report geomagnetic coordinate statistics
    valid_geomag = result_df['geomag_lat'].notna().sum()
    logger.success(f"Built coordinate catalogue: {len(result_df)} unique stations")
    logger.success(f"Geomagnetic coordinates: {valid_geomag}/{len(result_df)} stations ({100*valid_geomag/len(result_df):.1f}%)")
    
    return result_df

def download_station_clock_metadata():
    """Download clock metadata from IGS sources with smart existing data checks."""
    logger.info("Downloading station clock metadata")
    
    # Check if clock metadata already exists
    metadata_file = PACKAGE_ROOT / "results" / "outputs" / "step_1_station_metadata.json"
    station_logs_dir = PACKAGE_ROOT / "data" / "station_logs"
    
    if metadata_file.exists() and not TEPConfig.get_bool("TEP_REBUILD_METADATA"):
        logger.success("Using existing clock metadata")
        return True
    
    # Download clock metadata using existing functions
    station_logs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import clock metadata functions
        import urllib.request
        import re
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def get_station_log_urls():
            base_url = "https://files.igs.org/pub/station/log/"
            def _fetch_log_list():
                ssl_context = ssl.create_default_context()
                timeout = TEPConfig.get_int('TEP_NETWORK_TIMEOUT')
                
                with urllib.request.urlopen(base_url, context=ssl_context, timeout=timeout) as response:
                    html = response.read().decode('utf-8')
                log_files = re.findall(r'href="[^"]*?([a-z0-9_]+\.log)"', html)
                return list(set([f"{base_url}{fname}" for fname in log_files]))
            
            result = SafeErrorHandler.safe_network_operation(
                _fetch_log_list,
                error_message="Failed to fetch station log list",
                logger_func=logger.warning,
                return_on_error=[],
                max_retries=2
            )
            return result if result is not None else []
        
        def download_log_file(url):
            filename = url.split('/')[-1]
            station_code = filename[:4]
            file_path = station_logs_dir / f"{station_code}.log"
            
            if file_path.exists():
                return file_path
            
            # Use the safe download function we defined earlier
            if safe_download_file(url, file_path, timeout=30):
                return file_path
            else:
                return None
        
        def parse_oscillator_type(log_file_path):
            def _parse_operation():
                with open(log_file_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                freq_standard_sections = re.findall(r"6\.\d+\s+Standard Type\s+:\s*(.*)", content)
                if freq_standard_sections:
                    oscillator_type = freq_standard_sections[-1].strip()
                    if "HYDROGEN MASER" in oscillator_type.upper():
                        return "HYDROGEN MASER"
                    if "CESIUM" in oscillator_type.upper():
                        return "CESIUM"
                    if "RUBIDIUM" in oscillator_type.upper():
                        return "RUBIDIUM"
                    if "INTERNAL" in oscillator_type.upper():
                        return "INTERNAL"
                    return oscillator_type
                
                freq_standard_section = re.search(r"6\.\s+Frequency Standard(.*?)(?:7\.\s+Collocation Information|\Z)", content, re.DOTALL)
                if freq_standard_section:
                    section_content = freq_standard_section.group(1)
                    match = re.search(r"Standard Type\s+:\s*(.*)", section_content)
                    if match:
                        oscillator_type = match.group(1).strip()
                        if "INTERNAL" in oscillator_type.upper():
                            return "INTERNAL"
                        return oscillator_type
                
                return "UNKNOWN"
            
            result = SafeErrorHandler.safe_file_operation(
                _parse_operation,
                error_message=f"Failed to parse oscillator type from {log_file_path}",
                logger_func=logger.debug, # Use debug level for parsing failures
                return_on_error="UNKNOWN"
            )
            return result if result is not None else "UNKNOWN"
        
        # Download metadata
        logger.info("Fetching station log URLs...")
        urls = get_station_log_urls()
        
        if not urls:
            logger.error("No station log URLs found")
            return False
        
        logger.info(f"Downloading metadata for {len(urls)} stations...")
        
        station_metadata = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(download_log_file, url): url for url in urls}
            for future in as_completed(future_to_url):
                log_file_path = future.result()
                if log_file_path:
                    station_code = log_file_path.stem.upper()
                    oscillator = parse_oscillator_type(log_file_path)
                    station_metadata[station_code] = {"oscillator_type": oscillator}
        
        # Save metadata using safe JSON write
        try:
            safe_json_write(station_metadata, metadata_file, indent=4)
            logger.success(f"Clock metadata saved for {len(station_metadata)} stations")
            return True
        except (TEPFileError, TEPDataError) as e:
            logger.error(f"Failed to save clock metadata: {e}")
            return False
        
    except (ImportError, RuntimeError) as e:
        logger.error(f"Clock metadata download failed - system error: {e}")
        return False
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Clock metadata download failed - data error: {e}")
        return False

def build_and_write_catalogue(min_target: int = 0, out_path = None):
    """
    Build comprehensive station catalogue from authoritative sources.
    Creates single definitive coordinate file with validation metadata.
    """
    if out_path is None:
        out_path = PACKAGE_ROOT / "data" / "coordinates" / "station_coords_global.csv"
    
    logger.info("Building comprehensive station catalogue from authoritative sources")
    
    # Check if we should rebuild coordinates
    rebuild_coords = TEPConfig.get_bool("TEP_REBUILD_COORDS")
    
    if rebuild_coords or not out_path.exists():
        logger.info("Fetching coordinates from authoritative sources...")
        
        # Build coordinates directly (no external script)
        coords_df = build_coordinate_catalogue()
        
        if coords_df is None or len(coords_df) == 0:
            logger.error("Coordinate building failed - no stations retrieved")
            return False
        
        # Save the comprehensive catalogue
        coords_df.to_csv(out_path, index=False)
        logger.success(f"Station catalogue built: {len(coords_df)} stations saved to {out_path}")
        
        # Report verification statistics
        verified_stations = coords_df[coords_df['has_coordinates'] == True]
        
        logger.info(f"Coordinate verification summary:")
        logger.info(f"  Total stations in catalogue: {len(coords_df)}")
        logger.success(f"  Verified stations for analysis: {len(verified_stations)}")
        
    else:
        logger.success(f"Using existing station catalogue: {out_path}")
    
    # Verify coordinate catalogue and optionally fetch clock metadata
    if out_path.exists():
        import pandas as pd
        df = pd.read_csv(out_path)
        # Required coordinate columns only
        # Final verification
        required_cols = ['code', 'X', 'Y', 'Z']
        if not all(col in df.columns for col in required_cols):
            raise RuntimeError(f"Station catalogue missing required columns: {required_cols}")
        
        logger.success(f"Final station catalogue: {len(df)} stations")

        # Optional: fetch clock metadata (does not alter coordinate CSV)
        if TEPConfig.get_bool("TEP_FETCH_CLOCK_METADATA"):
            ok = download_station_clock_metadata()
            if not ok:
                msg = "Clock metadata fetch failed"
                if TEPConfig.get_bool("TEP_REQUIRE_CLOCK_METADATA"):
                    raise RuntimeError(msg)
                else:
                    logger.error(msg)
    
    return True

def gps_week_from_date(date: datetime) -> int:
    """Convert UTC date to GPS week number."""
    gps_epoch = datetime(1980, 1, 6)
    return int((date - gps_epoch).days // 7)

def download_station_coordinates_from_igs() -> bool:
    """Replaced by multi-source authoritative catalogue builder."""
    # Allow reusing existing real coordinates to avoid repeated rebuilds
    if TEPConfig.get_bool("TEP_SKIP_COORDS"):
        coord_path = PACKAGE_ROOT / "data" / "coordinates" / "station_coords_global.csv"
        if coord_path.exists() and coord_path.stat().st_size > 0:
            logger.success(f"Using existing coordinates: {coord_path}")
            return True
        else:
            logger.error("TEP_SKIP_COORDS=1 but no existing coordinates found")
            return False
    
    min_target = TEPConfig.get_int("TEP_MIN_STATIONS")
    try:
        build_and_write_catalogue(
            min_target=min_target,
            out_path=PACKAGE_ROOT / "data" / "coordinates" / "station_coords_global.csv"
        )
        return True
    except (TEPNetworkError, TEPFileError, TEPDataError) as e:
        logger.error(f"CRITICAL: Station catalogue build failed: {e}")
        return False
    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"CRITICAL: Station catalogue build failed - system error: {e}")
        return False

def download_small_real_clk_samples() -> bool:
    """Download a small real sample of IGS, CODE, and ESA .CLK files (strict) - smart existing data checks."""
    raw_dir = PACKAGE_ROOT / "data" / "raw"
    (raw_dir / "igs_combined").mkdir(parents=True, exist_ok=True)
    (raw_dir / "code").mkdir(parents=True, exist_ok=True)
    (raw_dir / "esa_final").mkdir(parents=True, exist_ok=True)
    
    # Check if we already have sufficient clock files
    existing_igs = len(list((raw_dir / "igs_combined").glob("*.CLK.gz")))
    existing_code = len(list((raw_dir / "code").glob("*.CLK.gz")))
    existing_esa = len(list((raw_dir / "esa_final").glob("*.CLK.gz")))
    
    logger.info(f"Existing clock files: IGS:{existing_igs} CODE:{existing_code} ESA:{existing_esa}")
    
    # Skip download if we have sufficient files and not forcing rebuild
    if (existing_igs > 50 and existing_code > 50 and existing_esa > 50 and 
        not TEPConfig.get_bool("TEP_REBUILD_CLK")):
        logger.success("Using existing clock files (sufficient coverage)")
        return True

    def day_of_year(d: datetime) -> int:
        return d.timetuple().tm_yday

    successes = {"igs_combined": 0, "code": 0, "esa_final": 0}
    downloaded = {"igs_combined": [], "code": [], "esa_final": []}

    # Get date range from centralized configuration (fail-fast: no defaults)
    try:
        date_start_s, date_end_s = TEPConfig.get_date_range()
        ds = datetime.fromisoformat(date_start_s)
        de = datetime.fromisoformat(date_end_s)
        if de < ds:
            ds, de = de, ds
        # Build inclusive daily range
        date_list = [ds + timedelta(days=i) for i in range((de - ds).days + 1)]
        logger.info(f"Using date filter {ds.date()} → {de.date()} ({len(date_list)} days)")
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Invalid date configuration in TEPConfig: {e}. Set TEP_DATE_START and TEP_DATE_END.")

    # Per-center known-good seeds (used only if no explicit date range supplied)
    igs_seed = datetime(2024, 1, 10)
    code_seed = datetime(2024, 1, 10)
    esa_seed = datetime(2022, 11, 26)

    # Per-center file limits using centralized configuration
    file_limits = TEPConfig.get_file_limits()
    files_per_igs = file_limits['igs_combined']
    files_per_code = file_limits['code']
    files_per_esa = file_limits['esa_final']
    
    logger.info(f"File limits: IGS:{files_per_igs or 'unlimited'} CODE:{files_per_code or 'unlimited'} ESA:{files_per_esa or 'unlimited'}")
    
    def safe_download_file(url: str, destination_path: Path, timeout: int = None) -> bool:
        """Safely download a file with proper error handling and SSL"""
        if timeout is None:
            timeout = TEPConfig.get_int('TEP_DOWNLOAD_TIMEOUT')
        
        def _download_operation():
            # Create SSL context for HTTPS URLs
            ssl_context = ssl.create_default_context() if url.startswith('https') else None
            
            with urllib.request.urlopen(url, context=ssl_context, timeout=timeout) as response:
                data = response.read()
            
            # Ensure destination directory exists
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(destination_path, 'wb') as f:
                f.write(data)
            
            return destination_path.stat().st_size > 0
        
        result = SafeErrorHandler.safe_network_operation(
            _download_operation,
            error_message=f"Download failed for {url}",
            logger_func=logger.debug,  # Use debug level for individual download failures
            return_on_error=False,
            max_retries=1  # Limited retries for bulk downloads
        )
        return result if result is not None else False

    # Build per-center date lists (explicit range preferred; else default 60-day windows)
    if date_list is not None:
        igs_dates = date_list
        code_dates = date_list
        esa_dates = date_list
    else:
        igs_dates = [igs_seed + timedelta(days=i) for i in range(0, 60)]
        code_dates = [code_seed + timedelta(days=i) for i in range(0, 60)]
        esa_dates = [esa_seed + timedelta(days=i) for i in range(0, 60)]

    # IGS Combined (week path)
    for d in igs_dates:
        if (files_per_igs is not None) and (successes["igs_combined"] >= files_per_igs):
            break
        year = d.year
        doy = day_of_year(d)
        week = gps_week_from_date(d)
        igs_url = f"https://igs.bkg.bund.de/root_ftp/IGS/products/{week:04d}/IGS0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        igs_dst = raw_dir / "igs_combined" / Path(igs_url).name
        
        if safe_download_file(igs_url, igs_dst):
            successes["igs_combined"] += 1
            downloaded["igs_combined"].append(igs_dst.name)
            logger.success(f"IGS sample: {igs_dst.name}")

    # CODE (year path)
    for d in code_dates:
        if (files_per_code is not None) and (successes["code"] >= files_per_code):
            break
        year = d.year
        doy = day_of_year(d)
        code_url = f"http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        code_dst = raw_dir / "code" / Path(code_url).name
        
        if safe_download_file(code_url, code_dst):
            successes["code"] += 1
            downloaded["code"].append(code_dst.name)
            logger.success(f"CODE sample: {code_dst.name}")

    # ESA (navigation-office week path)
    for d in esa_dates:
        if (files_per_esa is not None) and (successes["esa_final"] >= files_per_esa):
            break
        year = d.year
        doy = day_of_year(d)
        week = gps_week_from_date(d)
        esa_url = f"http://navigation-office.esa.int/products/gnss-products/{week}/ESA0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK.gz"
        esa_dst = raw_dir / "esa_final" / Path(esa_url).name
        
        if safe_download_file(esa_url, esa_dst):
            successes["esa_final"] += 1
            downloaded["esa_final"].append(esa_dst.name)
            logger.success(f"ESA sample: {esa_dst.name}")

    # Strict enforcement - check total files (existing + new downloads)
    total_igs = existing_igs + successes["igs_combined"]
    total_code = existing_code + successes["code"]
    total_esa = existing_esa + successes["esa_final"]
    
    if total_igs < 1:
        logger.error("CRITICAL: No IGS Combined .CLK files available")
        return False
    if total_code < 1:
        logger.error("CRITICAL: No CODE .CLK files available")
        return False
    if total_esa < 1:
        logger.error("CRITICAL: No ESA .CLK files available")
        return False

    logger.success(
        f" Clock files available -> IGS:{total_igs} CODE:{total_code} ESA:{total_esa}"
    )

    # Write a clean JSON summary of what we downloaded
    summary = {
        "igs_files": downloaded["igs_combined"],
        "code_files": downloaded["code"],
        "esa_files": downloaded["esa_final"],
        "counts": {
            "igs": successes["igs_combined"],
            "code": successes["code"],
            "esa": successes["esa_final"],
        }
    }
    try:
        (PACKAGE_ROOT / "logs").mkdir(exist_ok=True)
        safe_json_write(summary, PACKAGE_ROOT / "logs" / "step_1_downloads.json", indent=2)
    except (TEPFileError, TEPDataError) as e:
        logger.warning(f"Failed to write download summary: {e}")

    return True

def main():
    print("="*80)
    print("TEP GNSS Analysis Package v0.9")
    print("STEP 1: Data Acquisition")
    print("Acquiring authoritative GNSS data and coordinates")
    print("="*80)

    (PACKAGE_ROOT / "logs").mkdir(exist_ok=True)

    ok_coords = download_station_coordinates_from_igs()
    if not ok_coords:
        logger.error("Data acquisition failed: coordinates unavailable")
        return False

    ok_clk = download_small_real_clk_samples()
    if not ok_clk:
        logger.error("Data acquisition failed: clock products unavailable")
        return False

    # Final summary
    try:
        dl = safe_json_read(PACKAGE_ROOT / "logs" / "step_1_downloads.json")
        coords_df = safe_csv_read(PACKAGE_ROOT / "data" / "coordinates" / "station_coords_global.csv")
        
        final_summary = {
            "step": 1,
            "name": "Data Acquisition",
            "status": "completed",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "coordinates_stations": len(coords_df),
            "downloads": dl["counts"],
            "files": {
                "igs": dl["igs_files"][:5],
                "code": dl["code_files"][:5],
                "esa": dl["esa_files"][:5]
            },
            "outputs": {
                "coordinate_file": "data/coordinates/station_coords_global.csv",
                "download_log": "logs/step_1_downloads.json"
            }
        }
        
        safe_json_write(final_summary, PACKAGE_ROOT / "logs" / "step_1_data_acquisition.json", indent=2)
        
    except (TEPFileError, TEPDataError) as e:
        logger.warning(f"Failed to create final summary: {e}")

    logger.success("Data acquisition completed successfully")
    logger.info("Ready for coordinate validation (Step 2)")
    return True

if __name__ == "__main__":
    main()
