#!/usr/bin/env python3
"""
TEP-GNSS Step 2.0 Only Runner
============================

Runs ONLY Step 2.0 TEP Correlation Analysis without any cleanup or Step 1 dependencies.
This script assumes Step 1.x has already been completed and data exists.

Usage:
    python scripts/run_step2_only.py

Author: Matthew Lukin Smawfield
Date: October 2025
"""

import os
import sys
import time
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import print_status
from scripts.utils.pid_manager import ensure_single_instance

@ensure_single_instance
def main():
    """Run only Step 2.0 TEP Correlation Analysis."""
    
    print_status("TEP-GNSS Step 2.0 Only Runner", "TITLE")
    print_status("Running ONLY Step 2.0 without cleanup or Step 1 dependencies", "INFO")
    
    parser = argparse.ArgumentParser(description="Run Step 2.0 TEP Correlation Analysis only")
    parser.add_argument("--force", action="store_true", help="Delete existing Step 2.0 outputs before running")
    args = parser.parse_args()

    if args.force:
        cleanup_paths = [
            PROJECT_ROOT / "results/outputs/step_2_0_correlation_analysis_summary.json",
            PROJECT_ROOT / "results/outputs/step_2_0_correlation_code.json",
            PROJECT_ROOT / "results/outputs/step_2_0_correlation_igs_combined.json",
            PROJECT_ROOT / "results/outputs/step_2_0_correlation_esa_final.json",
        ]
        for path in cleanup_paths:
            if path.exists():
                path.unlink()
                print_status(f"Deleted existing output: {path}", "INFO")

    # Check prerequisites
    coords_file = PROJECT_ROOT / "data/coordinates/step_1_1_station_coords_global.csv"
    if not coords_file.exists():
        print_status(f"ERROR: Coordinate file not found: {coords_file}", "ERROR")
        print_status("Please run Step 1.x first to generate coordinate data", "ERROR")
        return False
    
    # Check for raw data
    raw_dirs = [
        PROJECT_ROOT / "data/raw/code",
        PROJECT_ROOT / "data/raw/esa_final", 
        PROJECT_ROOT / "data/raw/igs_combined"
    ]
    
    missing_dirs = [d for d in raw_dirs if not d.exists()]
    if missing_dirs:
        print_status("ERROR: Missing raw data directories:", "ERROR")
        for d in missing_dirs:
            print_status(f"  - {d}", "ERROR")
        print_status("Please run Step 1.1 first to download raw data", "ERROR")
        return False
    
    # Run Step 2.0
    step2_script = PROJECT_ROOT / "scripts/steps/step_2_core_analysis/step_2_0_tep_correlation_analysis.py"
    
    if not step2_script.exists():
        print_status(f"ERROR: Step 2.0 script not found: {step2_script}", "ERROR")
        return False
    
    print_status(f"Running Step 2.0: {step2_script.name}", "INFO")
    start_time = time.time()
    
    # Execute Step 2.0
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, str(step2_script)
        ], cwd=str(PROJECT_ROOT), capture_output=False)
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print_status(f"Step 2.0 completed successfully in {elapsed:.1f} seconds", "SUCCESS")
            return True
        else:
            print_status(f"Step 2.0 failed with return code {result.returncode}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Error running Step 2.0: {e}", "ERROR")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
