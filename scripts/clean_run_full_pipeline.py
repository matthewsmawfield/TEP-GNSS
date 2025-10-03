#!/usr/bin/env python3
"""
TEP-GNSS Full Pipeline Clean Run Script
======================================

Performs a complete clean run of the entire TEP-GNSS pipeline:
- Removes ALL data, processed files, logs, outputs, and figures
- Executes Step 1: Data Acquisition (1.0, 1.1, 1.2)
- Executes Step 2: Core Analysis (2.0, 2.1, 2.2)
- Executes Step 3: Validation Suite (3.0-3.5)
- Executes Step 4: Advanced Analysis & Visualization (4.0-4.7)

This script ensures a completely fresh start for the entire TEP analysis pipeline
from raw data acquisition through final visualizations and validation.

Usage:
    python scripts/clean_run_full_pipeline.py [--dry-run] [--skip-cleanup] [--start-step STEP]

Options:
    --dry-run       Show what would be cleaned without actually cleaning
    --skip-cleanup  Skip cleanup and only run the steps
    --start-step    Start from specific step (e.g., "2.0", "3.0"). Default: "1.0"

Examples:
    python scripts/clean_run_full_pipeline.py                    # Run everything from Step 1.0
    python scripts/clean_run_full_pipeline.py --start-step 2.0   # Run from Step 2.0 onwards
    python scripts/clean_run_full_pipeline.py --start-step 3.0   # Run from Step 3.0 onwards
    python scripts/clean_run_full_pipeline.py --dry-run          # See what would be cleaned
    python scripts/clean_run_full_pipeline.py --skip-cleanup     # Skip cleanup phase

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import sys
import os
import shutil
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.config import TEPConfig
from scripts.utils.logger import TEPLogger, print_status, reset_master_log
from scripts.utils.exceptions import SafeErrorHandler
from scripts.utils.pid_manager import ensure_single_instance

# Initialize logger
logger = TEPLogger()

def get_cleanup_targets(start_step: str = '1.0') -> Dict[str, List[Path]]:
    """
    Define all directories and files that need to be cleaned for a fresh full pipeline run.
    
    Args:
        start_step: Step to start from - determines what to preserve
    
    Returns:
        Dict with categories of cleanup targets
    """
    targets = {
        # Raw data directories (keep structure, remove files)
        'raw_data': [
            PROJECT_ROOT / "data" / "raw" / "igs_combined",
            PROJECT_ROOT / "data" / "raw" / "code", 
            PROJECT_ROOT / "data" / "raw" / "esa_final"
        ],
        
        # Processed data files
        'processed_data': [
            PROJECT_ROOT / "data" / "processed"
        ],
        
        # Coordinate files
        'coordinate_files': [
            PROJECT_ROOT / "data" / "coordinates"
        ],
        
        # All log files
        'log_files': [
            PROJECT_ROOT / "logs"
        ],
        
        # All output files
        'output_files': [
            PROJECT_ROOT / "results" / "outputs"
        ],
        
        # All figure files
        'figure_files': [
            PROJECT_ROOT / "results" / "figures",
            PROJECT_ROOT / "site" / "figures"
        ],
        
        # All temporary files
        'temp_files': [
            PROJECT_ROOT / "results" / "tmp",
            PROJECT_ROOT / "results" / "exploratory",
            PROJECT_ROOT / "results" / "validation"
        ]
    }
    
    return targets

def calculate_cleanup_size(targets: Dict[str, List[Path]]) -> Dict[str, int]:
    """
    Calculate the size of files that would be cleaned up.
    
    Args:
        targets: Cleanup targets dictionary
        
    Returns:
        Dict with size information by category
    """
    sizes = {}
    
    for category, paths in targets.items():
        total_size = 0
        file_count = 0
        
        for path in paths:
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
            elif path.is_dir():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        try:
                            total_size += file_path.stat().st_size
                            file_count += 1
                        except (OSError, FileNotFoundError):
                            # Skip files that can't be accessed
                            pass
        
        sizes[category] = {
            'size_bytes': total_size,
            'file_count': file_count
        }
    
    return sizes

def format_size(size_bytes: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def cleanup_directory_contents(dir_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Clean all contents of a directory while preserving the directory structure.
    
    Args:
        dir_path: Directory to clean
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dict with cleanup statistics
    """
    stats = {'files_removed': 0, 'size_freed': 0}
    
    if not dir_path.exists():
        return stats
    
    for item in dir_path.iterdir():
        if item.is_file():
            try:
                size = item.stat().st_size
                if not dry_run:
                    item.unlink()
                stats['files_removed'] += 1
                stats['size_freed'] += size
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Could not remove {item}: {e}")
        elif item.is_dir():
            try:
                if not dry_run:
                    shutil.rmtree(item)
                # Count files in directory
                for file_path in item.rglob('*'):
                    if file_path.is_file():
                        try:
                            stats['size_freed'] += file_path.stat().st_size
                            stats['files_removed'] += 1
                        except (OSError, FileNotFoundError):
                            pass
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Could not remove directory {item}: {e}")
    
    return stats

def cleanup_files(file_paths: List[Path], dry_run: bool = False) -> Dict[str, int]:
    """
    Clean specific files.
    
    Args:
        file_paths: List of files to remove
        dry_run: If True, only report what would be deleted
        
    Returns:
        Dict with cleanup statistics
    """
    stats = {'files_removed': 0, 'size_freed': 0}
    
    for file_path in file_paths:
        if file_path.exists() and file_path.is_file():
            try:
                size = file_path.stat().st_size
                if not dry_run:
                    file_path.unlink()
                stats['files_removed'] += 1
                stats['size_freed'] += size
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Could not remove {file_path}: {e}")
    
    return stats

def perform_cleanup(dry_run: bool = False) -> Dict[str, any]:
    """
    Perform complete cleanup of all pipeline data.
    
    Args:
        dry_run: If True, only report what would be cleaned
        
    Returns:
        Dict with cleanup summary
    """
    print_status("TEP-GNSS Full Pipeline Clean Run - Complete Cleanup Phase", "TITLE")
    
    targets = get_cleanup_targets()
    total_stats = {'files_removed': 0, 'size_freed': 0, 'categories': {}}
    
    if dry_run:
        print_status("DRY RUN MODE - No files will actually be deleted", "WARNING")
        
        # Calculate sizes for dry run
        sizes = calculate_cleanup_size(targets)
        total_size = sum(cat['size_bytes'] for cat in sizes.values())
        total_files = sum(cat['file_count'] for cat in sizes.values())
        
        print_status(f"Would clean {total_files} files ({format_size(total_size)})", "INFO")
        
        for category, size_info in sizes.items():
            if size_info['file_count'] > 0:
                print_status(f"  {category}: {size_info['file_count']} files ({format_size(size_info['size_bytes'])})", "INFO")
        
        return {'dry_run': True, 'total_files': total_files, 'total_size': total_size}
    
    # Perform actual cleanup
    for category, paths in targets.items():
        logger.process(f"Cleaning {category}...")
        category_stats = {'files_removed': 0, 'size_freed': 0}
        
        for path in paths:
            if path.is_dir():
                stats = cleanup_directory_contents(path, dry_run)
            else:
                stats = cleanup_files([path], dry_run)
            
            category_stats['files_removed'] += stats['files_removed']
            category_stats['size_freed'] += stats['size_freed']
        
        total_stats['files_removed'] += category_stats['files_removed']
        total_stats['size_freed'] += category_stats['size_freed']
        total_stats['categories'][category] = category_stats
        
        if category_stats['files_removed'] > 0:
            logger.success(f"Cleaned {category}: {category_stats['files_removed']} files ({format_size(category_stats['size_freed'])})")
        else:
            logger.info(f"No files to clean in {category}")
    
    print_status(f"Complete cleanup finished: {total_stats['files_removed']} files removed ({format_size(total_stats['size_freed'])})", "SUCCESS")
    
    return total_stats

def run_step_script(step_script: Path, step_name: str) -> bool:
    """
    Run a single step script and handle errors.
    
    Args:
        step_script: Path to the step script
        step_name: Human readable name for logging
        
    Returns:
        True if step completed successfully, False otherwise
    """
    logger.process(f"Executing {step_name}...")
    
    try:
        # Construct the environment for the subprocess to ensure sys.path is correct
        # Add PROJECT_ROOT to PYTHONPATH for subprocesses
        current_env = os.environ.copy()
        python_path = str(PROJECT_ROOT)
        if current_env.get('PYTHONPATH'):
            python_path = f"{python_path}:{current_env['PYTHONPATH']}"
        current_env['PYTHONPATH'] = python_path
        
        # Ensure subprocesses use the same master.log file
        # TEP_LOG_FILE is deprecated as each step now has its own log file.
        # The child script will initialize its own logger.
        current_env['PYTHONUNBUFFERED'] = '1'

        # Run the step script
        result = subprocess.run(
            [sys.executable, str(step_script)],
            cwd=PROJECT_ROOT,
            capture_output=False,  # Allow verbose output to console
            text=True,
            timeout=None,  # No timeout - let scientific processing complete
            env=current_env # Pass the modified environment
        )
        
        if result.returncode == 0:
            logger.success(f"{step_name} completed successfully")
            return True
        else:
            logger.error(f"{step_name} failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"{step_name} timed out after 3 hours")
        return False
    except Exception as e:
        logger.error(f"Failed to execute {step_name}: {e}")
        return False

def execute_full_pipeline_steps(start_step: str = '1.0') -> bool:
    """
    Execute the complete TEP-GNSS pipeline with all steps sequentially.
    
    Args:
        start_step: Step to start from (e.g., '2.0', '3.0')
    
    Returns:
        True if all steps completed successfully, False otherwise
    """
    print_status("TEP-GNSS Full Pipeline Execution", "TITLE")
    
    # Define all steps in execution order
    all_steps = [
        # Step 1: Data Acquisition
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_1_data_acquisition" / "step_1_0_provenance_snapshot.py",
            'name': "Step 1.0: Provenance Snapshot"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_1_data_acquisition" / "step_1_1_tep_data_acquisition.py",
            'name': "Step 1.1: Data Acquisition"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_1_data_acquisition" / "step_1_2_tep_coordinate_validation.py",
            'name': "Step 1.2: Coordinate Validation"
        },
        # Step 2: Core Analysis
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_2_core_analysis" / "step_2_0_tep_correlation_analysis.py",
            'name': "Step 2.0: TEP Correlation Analysis"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_2_core_analysis" / "step_2_1_data_quality_validation.py",
            'name': "Step 2.1: Data Quality Validation"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_2_core_analysis" / "step_2_2_tep_geospatial_temporal_analysis.py",
            'name': "Step 2.2: Geospatial Temporal Analysis"
        },
        # Step 3: Validation Suite
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_0_tep_cross_validation_suite.py",
            'name': "Step 3.0: TEP Cross Validation Suite"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_1_robust_block_bootstrap.py",
            'name': "Step 3.1: Robust Block Bootstrap"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_2_tep_null_tests.py",
            'name': "Step 3.2: TEP Null Tests"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_3_methodology_validation.py",
            'name': "Step 3.3: Methodology Validation"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_4_geographic_bias_validation.py",
            'name': "Step 3.4: Geographic Bias Validation"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_5_realistic_ionospheric_validation.py",
            'name': "Step 3.5: Realistic Ionospheric Validation"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_6_control_band_analysis.py",
            'name': "Step 3.6: Control Band Analysis"
        },
        # Step 4: Advanced Analysis & Visualization
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_0_tep_advanced_analysis.py",
            'name': "Step 4.0: TEP Advanced Analysis"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_1_tep_visualization.py",
            'name': "Step 4.1: TEP Visualization"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_2_tep_synthesis_figure.py",
            'name': "Step 4.2: TEP Synthesis Figure"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_3_high_resolution_astronomical_events.py",
            'name': "Step 4.3: High Resolution Astronomical Events"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_4_gravitational_temporal_field_analysis.py",
            'name': "Step 4.4: Gravitational Temporal Field Analysis"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_5_comprehensive_diurnal_analysis.py",
            'name': "Step 4.5: Comprehensive Diurnal Analysis"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_6_tid_exclusion_analysis.py",
            'name': "Step 4.6: TID Exclusion Analysis"
        },
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_4_advanced_analysis_and_visualization" / "step_4_7_multiple_comparison_corrections.py",
            'name': "Step 4.7: Multiple Comparison Corrections"
        }
    ]
    
    # Filter steps based on start_step
    start_index = 0
    for i, step in enumerate(all_steps):
        if step['name'].startswith(f"Step {start_step}:"):
            start_index = i
            break
    
    if start_index == 0 and start_step != '1.0':
        logger.error(f"Invalid start step: {start_step}. Available steps: 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7")
        return False
    
    filtered_steps = all_steps[start_index:]
    logger.info(f"Starting from Step {start_step} - will execute {len(filtered_steps)} steps")
    
    start_time = time.time()
    successful_steps = 0
    
    for i, step in enumerate(filtered_steps, 1):
        actual_step_num = start_index + i
        logger.info(f"Starting step {actual_step_num}/{len(all_steps)}: {step['name']}")
        
        if not step['script'].exists():
            logger.error(f"Step {actual_step_num} script not found: {step['script']}")
            return False
        
        step_start = time.time()
        success = run_step_script(step['script'], step['name'])
        step_elapsed = time.time() - step_start
        
        if success:
            successful_steps += 1
            actual_step_num = start_index + i
            logger.success(f"Step {actual_step_num} completed in {step_elapsed:.1f} seconds")
        else:
            logger.error(f"Step {actual_step_num} failed after {step_elapsed:.1f} seconds")
            logger.error("Full pipeline execution stopped due to step failure")
            return False
    
    total_elapsed = time.time() - start_time
    
    if successful_steps == len(filtered_steps):
        print_status(f"Full pipeline completed successfully in {total_elapsed:.1f} seconds", "SUCCESS")
        return True
    else:
        logger.error(f"Pipeline failed: {successful_steps}/{len(filtered_steps)} steps completed (steps {start_index + 1} to {start_index + successful_steps})")
        return False


def validate_full_pipeline() -> bool:
    """
    Validate that the full pipeline run was successful by checking for expected outputs.
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.process("Validating full pipeline results...")
    
    # Check for key outputs from each major step
    validation_checks = [
        # Step 1 outputs
        (PROJECT_ROOT / "data" / "coordinates" / "step_1_1_station_coords_global.csv", "Step 1: Station coordinates"),
        (PROJECT_ROOT / "results" / "outputs" / "step_1_1_data_acquisition.json", "Step 1: Data acquisition results"),
        
        # Step 2 outputs (at least one analysis center)
        (PROJECT_ROOT / "results" / "outputs" / "step_2_0_correlation_code.json", "Step 2: CODE correlation analysis"),
        (PROJECT_ROOT / "results" / "outputs" / "step_2_0_correlation_igs_combined.json", "Step 2: IGS correlation analysis"),
        (PROJECT_ROOT / "results" / "outputs" / "step_2_0_correlation_esa_final.json", "Step 2: ESA correlation analysis"),
        
        # Step 3 outputs (at least one validation)
        (PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_code.json", "Step 3: Cross-validation results"),
        (PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_code.json", "Step 3.6: Control band analysis"),
        
        # Step 4 outputs (at least one advanced analysis)
        (PROJECT_ROOT / "results" / "outputs" / "step_4_0_advanced_analysis.json", "Step 4: Advanced analysis results"),
        (PROJECT_ROOT / "results" / "outputs" / "step_4_7_multiple_comparison_corrections.json", "Step 4: Multiple Comparison Corrections results"),
        (PROJECT_ROOT / "results" / "outputs" / "step_4_7_corrected_significance_summary.json", "Step 4: Corrected Significance Summary"),
        (PROJECT_ROOT / "results" / "outputs" / "step_4_7_correction_impact_analysis.csv", "Step 4: Correction Impact Analysis"),
    ]
    
    missing_outputs = []
    for output_path, description in validation_checks:
        if not output_path.exists():
            missing_outputs.append(f"  {description}: {output_path}")
    
    if missing_outputs:
        logger.error("Validation failed - missing expected outputs:")
        for missing in missing_outputs:
            logger.error(missing)
        return False
    
    # Check for raw data files
    raw_dirs = [
        PROJECT_ROOT / "data" / "raw" / "igs_combined",
        PROJECT_ROOT / "data" / "raw" / "code",
        PROJECT_ROOT / "data" / "raw" / "esa_final"
    ]
    
    total_raw_files = 0
    for raw_dir in raw_dirs:
        if raw_dir.exists():
            total_raw_files += len(list(raw_dir.glob("*.CLK.gz")))
    
    if total_raw_files == 0:
        logger.warning("No raw data files found - data acquisition may have failed")
        return False
    
    logger.success(f"Full pipeline validation passed - found {total_raw_files} raw data files and all expected outputs")
    return True

@ensure_single_instance
def main():
    """Main full pipeline clean run execution."""
    parser = argparse.ArgumentParser(description='TEP-GNSS Full Pipeline Clean Run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without actually cleaning')
    parser.add_argument('--skip-cleanup', action='store_true',
                       help='Skip cleanup and only run the steps')
    parser.add_argument('--start-step', type=str, default='1.0',
                       help='Start from specific step (e.g., "2.0", "3.0"). Default: "1.0"')
    
    args = parser.parse_args()
    
    # Reset master log file for fresh start
    # reset_master_log() # DEPRECATED: Each step now has its own log file
    
    print_status("TEP-GNSS Full Pipeline Clean Run Script v1.0", "TITLE")
    print_status("Complete Pipeline Clean Run - All Steps (1.0-4.7)", "TITLE")
    
    # Phase 1: Cleanup (unless skipped)
    if not args.skip_cleanup:
        cleanup_stats = perform_cleanup(dry_run=args.dry_run)
        
        if args.dry_run:
            print_status("Dry run completed - no changes made", "SUCCESS")
            return True
        
        if cleanup_stats['files_removed'] == 0:
            logger.info("No files needed cleaning - starting fresh")
    else:
        logger.info("Skipping cleanup phase")
    
    # Phase 2: Execute Full Pipeline
    if not args.dry_run:
        pipeline_success = execute_full_pipeline_steps(args.start_step)
        
        if not pipeline_success:
            logger.error("Full pipeline execution failed")
            return False
        
        # Phase 3: Validate Results
        validation_success = validate_full_pipeline()
        
        if validation_success:
            print_status("Full pipeline clean run completed successfully!", "SUCCESS")
            return True
        else:
            logger.error("Full pipeline validation failed")
            return False
    else:
        print_status("Dry run completed - no pipeline execution", "SUCCESS")
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
