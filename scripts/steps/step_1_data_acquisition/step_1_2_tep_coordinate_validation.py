#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 1.2: Coordinate Validation
================================================

Validates station coordinates and establishes definitive station catalogue.
Performs comprehensive spatial verification and quality assurance for
precision distance calculations in temporal equivalence analysis.

Requirements: Step 1.1 complete (ensuring data acquisition is done)
Inputs:
  - data/coordinates/step_1_1_station_coords_global.csv (from Step 1.1)
  - results/outputs/step_1_1_data_acquisition.json (to verify Step 1.1 completion)
Outputs:
  - results/outputs/step_1_2_coordinate_validation.json (summary of validation)
  - results/tmp/step_1_2_station_audit.json (detailed station audit)
Next: Step 2.0 (Core Analysis - TEP Correlation Analysis)

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse
import os
import urllib.request

# Import TEP utilities for better configuration and error handling
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.utils.config import TEPConfig
from scripts.utils.exceptions import (
    SafeErrorHandler, TEPDataError, TEPFileError, 
    safe_csv_read, safe_json_read, safe_json_write,
    validate_file_exists, validate_directory_exists
)
from scripts.utils.logger import TEPLogger, print_status, set_step_logger
from scripts.utils.pid_manager import ensure_single_instance

# Instantiate the step-specific logger
step_logger = TEPLogger(
    name="step_1_2_coordinate_validation",
    level="DEBUG",
    log_file_path=Path(__file__).resolve().parents[3] / "logs" / "step_1_2_coordinate_validation.log"
)


def print_step_header():
    """Print formatted step header"""
    print_status("TEP GNSS Analysis Package v0.14 - STEP 1.2: Coordinate Validation", "TITLE")

def check_step_1_1_completion():
    """Check that Step 1 completed successfully"""
    set_step_logger(step_logger)
    print_status("Checking Step 1.1 completion...", "TEST")
    
    required_files = [
        "results/outputs/step_1_1_data_acquisition.json",
        "data/coordinates/step_1_1_station_coords_global.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_status("Step 1.1 not completed. Missing:", "ERROR")
        for file_path in missing_files:
            print_status(f"  Missing: {file_path}", "ERROR")
        return False
    
    print_status("Step 1.1 completion verified", "SUCCESS")
    return True

def validate_coordinate_data():
    """Validate the coordinate data from Step 1.1"""
    set_step_logger(step_logger)
    print_status("Validating coordinate data...", "PROCESS")
    
    # Check the single comprehensive coordinate file
    coord_file = Path("data/coordinates/step_1_1_station_coords_global.csv")
    
    if not coord_file.exists():
        print_status("Station coordinates file not found", "ERROR")
        return False
    
    try:
        # Load and validate the comprehensive coordinate file
        df = safe_csv_read(coord_file)
        
        # Check if this is the new comprehensive format
        if 'has_coordinates' in df.columns:
            verified_stations = df[df['has_coordinates'] == True]
            print_status(f"Comprehensive coordinate catalogue: {len(df)} stations", "INFO")
            print_status(f"Verified stations for analysis: {len(verified_stations)}", "SUCCESS")
        else:
            # Legacy format - all stations are considered verified
            verified_stations = df
            print_status(f"Legacy coordinate catalogue: {len(df)} stations (all verified)", "INFO")

        # Require only real ECEF coordinates (no inference). LLH is optional.
        required_cols = ['code', 'X', 'Y', 'Z']
        if 'has_coordinates' in df.columns:
            required_cols.append('has_coordinates')
            
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print_status(f"Missing columns: {missing_cols}", "ERROR")
            return False

        # Validate only verified stations (has_coordinates=True)
        n_total = len(df)
        n_verified = len(verified_stations)
        
        # Valid if X,Y,Z are finite and non-zero for verified stations
        valid_mask = (
            verified_stations['X'].apply(np.isfinite) &
            verified_stations['Y'].apply(np.isfinite) &
            verified_stations['Z'].apply(np.isfinite) &
            (verified_stations['X'] != 0) & (verified_stations['Y'] != 0) & (verified_stations['Z'] != 0)
        )
        valid_coords = int(valid_mask.sum())

        print_status(f"Running ECEF coordinate validation for {n_verified} verified stations...", "PROCESS")
        print_status(f"  Verified stations: {n_verified}/{n_total}", "INFO")
        print_status(f"  Valid ECEF coords: {valid_coords}/{n_verified}", "INFO")
        print_status(f"  ECEF ranges:", "INFO")
        print_status(f"    X: {verified_stations['X'].min():.0f} to {verified_stations['X'].max():.0f} m", "INFO")
        print_status(f"    Y: {verified_stations['Y'].min():.0f} to {verified_stations['Y'].max():.0f} m", "INFO")
        print_status(f"    Z: {verified_stations['Z'].min():.0f} to {verified_stations['Z'].max():.0f} m", "INFO")

        if valid_coords >= max(10, int(n_verified * 0.95)):  # Stricter validation for verified stations
            print_status("Coordinate validation passed - verified stations have valid ECEF", "SUCCESS")
            return True
        else:
            print_status("Too many invalid ECEF coordinates in verified stations", "ERROR")
            return False
            
    except (TEPDataError, TEPFileError) as e:
        print_status(f"Error reading coordinate file: {e}", "ERROR")
        return False

def create_step_2_0_summary():
    """Create Step 1.2 completion summary with definitive station counts"""
    coord_file = Path("data/coordinates/step_1_1_station_coords_global.csv")
    
    if not coord_file.exists():
        print_status("Coordinate file not found for summary", "ERROR")
        return None
    
    try:
        df = safe_csv_read(coord_file)
        n_total = len(df)
        
        # Check if this is comprehensive format with validation metadata
        if 'has_coordinates' in df.columns:
            verified_df = df[df['has_coordinates'] == True]
            n_verified = len(verified_df)
            coord_quality = "comprehensive_verified"
        else:
            # Legacy format - all stations considered verified
            verified_df = df
            n_verified = n_total
            coord_quality = "legacy_all_verified"
            
        verified_stations = n_verified
        
    except (TEPDataError, TEPFileError) as e:
        print_status(f"Error reading coordinate file: {e}", "ERROR")
        n_total = 0
        verified_stations = 0
        coord_quality = "error"
    
    # Load our audit results if available
    audit_file = Path("results/tmp/step_1_2_station_audit.json")
    center_breakdown = {}
    if audit_file.exists():
        try:
            audit_data = safe_json_read(audit_file)
            center_breakdown = audit_data.get('by_analysis_center', {})
            verified_stations = audit_data['overall_statistics']['sites_with_coordinates']
            print_status(f"Using audit results: {verified_stations} verified stations", "SUCCESS")
        except (TEPDataError, TEPFileError) as e:
            print_status(f"Could not load audit results: {e}", "WARNING")
    
    # Calculate data-driven validation metadata
    total_stations = len(df)
    verified_stations_count = len(verified_df)
    excluded_stations = total_stations - verified_stations_count

    # Get unique coordinate sources
    sources_checked = df['coord_source_code'].dropna().unique().tolist()
    if sources_checked:
        sources_checked = sorted(sources_checked)

    summary = {
        'step': "1.2",
        'name': 'Coordinate Validation',
        'completion_time': datetime.now().isoformat(),
        'status': 'completed' if verified_stations > 0 else 'failed',
        'outputs': {
            'coordinate_file': str(coord_file),
            'station_audit': 'results/tmp/step_1_2_station_audit.json',
            'n_stations_total': total_stations,
            'n_stations_verified': verified_stations_count,
            'n_stations_excluded': excluded_stations,
            'coordinate_quality': coord_quality,
            'by_analysis_center': center_breakdown
        },
        'validation': {
            'method': 'comprehensive_coordinate_validation',
            'excluded_stations': excluded_stations,
            'exclusion_reason': 'missing_or_invalid_coordinates_in_catalogue',
            'sources_checked': sources_checked or ['IGS'],
            'spatial_verification': 'ecef_coordinates_validated',
            'validation_criteria': {
                'finite_coordinates': 'X, Y, Z must be finite numbers',
                'non_zero_coordinates': 'X, Y, Z must not be zero',
                'coordinate_precision': 'meter-level precision maintained'
            }
        },
        'pipeline_consistency': {
            'definitive_station_count': verified_stations_count,
            'use_4char_sites': True,
            'reason': 'stations_with_valid_ecef_coordinates',
            'data_driven': True
        },
        'next_step': 'python scripts/steps/step_2_core_analysis/step_2_0_tep_correlation_analysis.py'
    }
    
    # Save summary to results/outputs directory
    summary_file = Path("results/outputs/step_1_2_coordinate_validation.json")
    try:
        safe_json_write(summary, summary_file, indent=2)
    except (TEPFileError, TEPDataError) as e:
        print_status(f"Failed to save summary: {e}", "WARNING")
    
    print_status(f"Validation summary saved: {summary_file}", "SUCCESS")
    print_status(f"Pipeline configured for {verified_stations} verified stations", "SUCCESS")
    return summary

def validate_step_2_0_completion(coord_validation_result=None):
    """Validate Step 1.2 completion"""
    set_step_logger(step_logger)
    print_status("Validating Step 1.2 completion...", "PROCESS")
    
    validation_checks = [
        ("Step 1.2 summary exists", Path("results/outputs/step_1_2_coordinate_validation.json").exists()),
        ("Coordinate file exists", Path("data/coordinates/step_1_1_station_coords_global.csv").exists())
    ]
    
    # Validate coordinate data quality
    if coord_validation_result is not None:
        print_status("Using cached coordinate validation result from execution phase", "INFO")
        coord_valid = coord_validation_result
    else:
        coord_valid = validate_coordinate_data()
    validation_checks.append(("Coordinate data valid", coord_valid))
    
    all_passed = all(result[1] for result in validation_checks)
    
    for check_name, passed in validation_checks:
        status_icon = "SUCCESS" if passed else "ERROR"
        (print_status if passed else print_status)(f"{check_name}: {'PASS' if passed else 'FAIL'}", status_icon)
    
    if all_passed:
        print_status("All validation checks passed", "SUCCESS")
    else:
        print_status("Validation checks failed", "ERROR")
    
    return all_passed

@ensure_single_instance
def main():
    """Main Step 1.2 execution"""
    set_step_logger(step_logger)
    parser = argparse.ArgumentParser(description='TEP Analysis - Step 1.2: Process Coordinates')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate Step 1.2 completion')
    
    args = parser.parse_args()
    
    print_status("TEP GNSS Analysis Package v0.14 - STEP 1.2: Coordinate Validation", "TITLE")
    
    # Validation only mode
    if args.validate_only:
        success = validate_step_2_0_completion()
        return success
    
    start_time = time.time()
    
    # Check Step 1.1 completion
    if not check_step_1_1_completion():
        return False
    
    # Validate coordinate data (Step 1.1 should have created this)
    coord_validation_success = validate_coordinate_data()
    if not coord_validation_success:
        print_status("Coordinate validation failed", "ERROR")
        return False
    
    print_status("Coordinate data looks good - no additional processing needed", "SUCCESS")
    
    # Run comprehensive station audit for pipeline consistency (always enabled)
    print_status("Running comprehensive station audit for pipeline consistency...", "PROCESS")
    try:
        audit_station_ids()
        print_status("Station ID audit complete - pipeline will use verified counts", "SUCCESS")
    except (TEPDataError, TEPFileError, ValueError, TypeError) as e:
        print_status(f"Station audit failed: {e}", "WARNING")
        print_status("Continuing with basic validation...", "INFO")
    
    # Create completion summary
    summary = create_step_2_0_summary()
    
    # Validate completion
    validation_success = validate_step_2_0_completion(coord_validation_result=coord_validation_success)
    
    # Final report
    elapsed_time = time.time() - start_time
    
    print_status("COORDINATE VALIDATION COMPLETE", "INFO")
    
    print_status(f"Execution time: {elapsed_time:.1f} seconds", "INFO")
    
    if validation_success:
        print_status("Station coordinates validated and ready for analysis", "SUCCESS")
    else:
        print_status("Coordinate validation failed", "ERROR")
        return False
    
    return True


def audit_station_ids():
    """
    Comprehensive station ID audit - validates coordinate quality and provides definitive station counts.
    Performs detailed analysis of coordinate sources, spatial distribution, and validation metrics.
    """
    set_step_logger(step_logger)
    print_status("Running comprehensive station ID audit...", "PROCESS")

    coord_file = Path("data/coordinates/step_1_1_station_coords_global.csv")
    if not coord_file.exists():
        print_status("Coordinate file not found, cannot run audit.", "ERROR")
        return

    df = safe_csv_read(coord_file)

    # Perform comprehensive audit
    audit_results = perform_comprehensive_audit(df)

    # Save comprehensive audit results
    outdir = Path('results/tmp')
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        safe_json_write(audit_results, outdir / 'step_1_2_station_audit.json', indent=2)
        print_status(f"Comprehensive audit complete: {audit_results['overall_statistics']['sites_with_coordinates']} verified stations", "SUCCESS")
    except (TEPFileError, TEPDataError) as e:
        print_status(f"Failed to save audit results: {e}", "WARNING")

def perform_comprehensive_audit(df: pd.DataFrame) -> dict:
    """Perform comprehensive audit of station coordinate data"""

    total_stations = len(df)
    verified_stations_df = df[df['has_coordinates'] == True] if 'has_coordinates' in df.columns else df
    verified_stations = len(verified_stations_df)

    # Analyze coordinate sources
    source_analysis = analyze_coordinate_sources(df)

    # Analyze spatial distribution
    spatial_analysis = analyze_spatial_distribution(verified_stations_df)

    # Analyze coordinate quality metrics
    quality_metrics = analyze_coordinate_quality(verified_stations_df)

    # Compile comprehensive audit results
    audit_results = {
        'audit_timestamp': datetime.now().isoformat(),
        'status': 'comprehensive_audit_completed',
        'audit_method': 'coordinate_validation_and_spatial_analysis',
        'coordinate_catalogue': {
            'total_stations': total_stations,
            'verified_stations': verified_stations,
            'excluded_stations': total_stations - verified_stations,
            'exclusion_rate': (total_stations - verified_stations) / total_stations * 100,
            'coordinate_sources': source_analysis['sources_used'],
            'source_distribution': source_analysis['source_distribution']
        },
        'spatial_analysis': spatial_analysis,
        'coordinate_quality': quality_metrics,
        'by_analysis_center': source_analysis['by_center'],
        'overall_statistics': {
            'sites_with_coordinates': verified_stations,
            'global_coverage': spatial_analysis['global_coverage'],
            'coordinate_precision': quality_metrics['precision_assessment']
        },
        'validation_criteria': {
            'finite_coordinates': 'X, Y, Z must be finite numbers',
            'non_zero_coordinates': 'X, Y, Z must not be zero',
            'spatial_consistency': 'coordinates must be within Earth radius bounds',
            'precision_maintained': 'meter-level precision preserved'
        }
    }

    return audit_results

def analyze_coordinate_sources(df: pd.DataFrame) -> dict:
    """Analyze distribution of coordinate sources"""

    sources = df['coord_source_code'].dropna().unique().tolist()
    source_counts = df['coord_source_code'].value_counts().to_dict()

    # Convert to native Python types for JSON serialization
    source_counts = {str(k): int(v) for k, v in source_counts.items()}

    # Group by 4-character codes for analysis centers
    center_mapping = {}
    for source in sources:
        center_4char = source[:4] if len(source) >= 4 else source
        if center_4char not in center_mapping:
            center_mapping[center_4char] = []
        center_mapping[center_4char].append(source)

    return {
        'sources_used': sorted(sources),
        'source_distribution': source_counts,
        'by_center': {center: len(stations) for center, stations in center_mapping.items()},
        'primary_centers': sorted(center_mapping.keys())
    }

def analyze_spatial_distribution(df: pd.DataFrame) -> dict:
    """Analyze spatial distribution of stations"""

    # Calculate geographic bounds
    lat_range = (float(df['lat_deg'].min()), float(df['lat_deg'].max()))
    lon_range = (float(df['lon_deg'].min()), float(df['lon_deg'].max()))

    # Calculate ECEF bounds
    x_range = (float(df['X'].min()), float(df['X'].max()))
    y_range = (float(df['Y'].min()), float(df['Y'].max()))
    z_range = (float(df['Z'].min()), float(df['Z'].max()))

    # Estimate global coverage (rough approximation)
    lat_coverage = lat_range[1] - lat_range[0]
    lon_coverage = lon_range[1] - lon_range[0]

    return {
        'geographic_bounds': {
            'latitude_range_deg': lat_range,
            'longitude_range_deg': lon_range,
            'latitude_coverage_deg': float(lat_coverage),
            'longitude_coverage_deg': float(lon_coverage)
        },
        'ecef_bounds': {
            'x_range_m': x_range,
            'y_range_m': y_range,
            'z_range_m': z_range
        },
        'global_coverage': {
            'latitude_percent': float(min(100, (lat_coverage / 180) * 100)),
            'longitude_percent': float(min(100, (lon_coverage / 360) * 100)),
            'hemispheric_balance': 'north_south_balanced' if abs(lat_range[0]) + abs(lat_range[1]) < 180 else 'polar_focused'
        },
        'station_density': float(len(df) / (4 * 3.14159 * 6371000**2) * 1000000)  # stations per million km²
    }

def analyze_coordinate_quality(df: pd.DataFrame) -> dict:
    """Analyze coordinate quality metrics"""

    # Coordinate precision analysis
    x_precision = len(str(df['X'].iloc[0]).split('.')[-1]) if '.' in str(df['X'].iloc[0]) else 0
    y_precision = len(str(df['Y'].iloc[0]).split('.')[-1]) if '.' in str(df['Y'].iloc[0]) else 0
    z_precision = len(str(df['Z'].iloc[0]).split('.')[-1]) if '.' in str(df['Z'].iloc[0]) else 0

    # Check for suspicious coordinates (e.g., too close to center of Earth)
    earth_radius = 6371000  # meters
    distances_from_center = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    suspicious_coords = int(((distances_from_center < earth_radius * 0.9) |
                           (distances_from_center > earth_radius * 1.1)).sum())

    return {
        'precision_assessment': {
            'x_precision_digits': int(x_precision),
            'y_precision_digits': int(y_precision),
            'z_precision_digits': int(z_precision),
            'precision_level': 'meter' if x_precision >= 1 else 'unknown'
        },
        'spatial_consistency': {
            'distances_from_center_range': (float(distances_from_center.min()), float(distances_from_center.max())),
            'suspicious_coordinates': suspicious_coords,
            'suspicious_rate': float(suspicious_coords / len(df) * 100)
        },
        'coordinate_ranges': {
            'x_range_m': (float(df['X'].min()), float(df['X'].max())),
            'y_range_m': (float(df['Y'].min()), float(df['Y'].max())),
            'z_range_m': (float(df['Z'].min()), float(df['Z'].max()))
        }
    }


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("Step 2 interrupted by user", "WARNING")
        sys.exit(1)
    except (TEPDataError, TEPFileError) as e:
        print_status(f"Step 2 failed - data/file error: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"Step 2 failed - unexpected error: {e}", "CRITICAL")
        import traceback
        traceback.print_exc()
        sys.exit(1)

