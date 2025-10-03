#!/usr/bin/env python3
"""
TEP-GNSS Clean Run Script - Step 3 & 4 Pipeline
==============================================

Performs a clean run of Step 3 (Validation Suite) and Step 4 (Advanced Analysis & Visualization):
- Removes ONLY Step 3 & 4 outputs, logs, and temporary files
- Preserves Step 1 & 2 outputs (data acquisition and core analysis results)
- Executes Step 3: Validation Suite (3.0-3.6)
- Executes Step 4: Advanced Analysis & Visualization (4.0-4.7)

This script is ideal for re-running validation and advanced analysis without
re-downloading data or re-running the core correlation analysis.

Usage:
    python scripts/clean_run_step3_4.py [--dry-run] [--skip-cleanup]

Options:
    --dry-run       Show what would be cleaned without actually cleaning
    --skip-cleanup  Skip cleanup and only run the steps

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

def get_cleanup_targets() -> Dict[str, List[Path]]:
    """
    Define Step 3 & 4 specific directories and files that need to be cleaned.
    Preserves Step 1 & 2 outputs.
    
    Returns:
        Dict with categories of cleanup targets
    """
    targets = {
        # Step 3 log files only
        'step3_log_files': [
            PROJECT_ROOT / "logs" / "step_3_0_cross_validation_suite.json",
            PROJECT_ROOT / "logs" / "step_3_1_robust_block_bootstrap.json",
            PROJECT_ROOT / "logs" / "step_3_2_tep_null_tests.json",
            PROJECT_ROOT / "logs" / "step_3_3_methodology_validation.json",
            PROJECT_ROOT / "logs" / "step_3_4_geographic_bias_validation.json",
            PROJECT_ROOT / "logs" / "step_3_5_realistic_ionospheric_validation.json",
            PROJECT_ROOT / "logs" / "step_3_6_control_band_analysis.log",
            # PROJECT_ROOT / "logs" / "step_3_7_multiple_comparison_corrections.json" # Moved to step_4_7
        ],
        
        # Step 4 log files only
        'step4_log_files': [
            PROJECT_ROOT / "logs" / "step_4_0_tep_advanced_analysis.json",
            PROJECT_ROOT / "logs" / "step_4_1_tep_visualization.json",
            PROJECT_ROOT / "logs" / "step_4_2_tep_synthesis_figure.json",
            PROJECT_ROOT / "logs" / "step_4_3_high_resolution_astronomical_events.json",
            PROJECT_ROOT / "logs" / "step_4_4_gravitational_temporal_field_analysis.json",
            PROJECT_ROOT / "logs" / "step_4_5_comprehensive_diurnal_analysis.json",
            PROJECT_ROOT / "logs" / "step_4_6_tid_exclusion_analysis.json",
            PROJECT_ROOT / "logs" / "step_4_7_multiple_comparison_corrections.json"
        ],
        
        # Step 3 output files only
        'step3_output_files': [
            PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_1_robust_block_bootstrap_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_1_robust_block_bootstrap_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_1_robust_block_bootstrap_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_2_tep_null_tests_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_2_tep_null_tests_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_2_tep_null_tests_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_methodology_validation.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_methodology_validation_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_methodology_validation_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_methodology_validation_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_multi_center_consistency.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_real_data_zero_lag_test_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_3_zero_lag_leakage_test.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_4_geographic_bias_validation_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_4_geographic_bias_validation_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_4_geographic_bias_validation_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_5_realistic_ionospheric_validation_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_5_realistic_ionospheric_validation_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_5_realistic_ionospheric_validation_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_6_band_comparison_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_6_band_comparison_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_3_6_band_comparison_esa_final.json",
            # PROJECT_ROOT / "results" / "outputs" / "step_3_7_multiple_comparison_corrections_code.json", # Moved to step_4_7
            # PROJECT_ROOT / "results" / "outputs" / "step_3_7_multiple_comparison_corrections_igs_combined.json", # Moved to step_4_7
            # PROJECT_ROOT / "results" / "outputs" / "step_3_7_multiple_comparison_corrections_esa_final.json" # Moved to step_4_7
        ],
        
        # Step 4 output files only
        'step4_output_files': [
            PROJECT_ROOT / "results" / "outputs" / "step_4_0_advanced_analysis.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_0_tep_advanced_analysis_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_0_tep_advanced_analysis_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_0_tep_advanced_analysis_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_0_circular_statistics_streamlined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_1_tep_visualization_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_1_tep_visualization_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_1_tep_visualization_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_2_tep_synthesis_figure.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_3_comprehensive_eclipses_all-centers.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_3_high_resolution_astronomical_events_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_3_high_resolution_astronomical_events_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_3_high_resolution_astronomical_events_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_4_gravitational_temporal_field_analysis.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_4_gravitational_temporal_field_analysis_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_4_gravitational_temporal_field_analysis_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_4_gravitational_temporal_field_analysis_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_analysis.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_diurnal_analysis_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_diurnal_analysis_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_diurnal_analysis_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_hourly_summary_code.csv",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_hourly_summary_esa_final.csv",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_hourly_summary_igs_combined.csv",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_validation_code.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_validation_esa_final.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_5_comprehensive_validation_igs_combined.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_6_tid_exclusion_analysis.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_7_multiple_comparison_corrections.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_7_corrected_significance_summary.json",
            PROJECT_ROOT / "results" / "outputs" / "step_4_7_correction_impact_analysis.csv"
        ],
        
        # Step 3 & 4 figure files only
        'step3_4_figure_files': [
            PROJECT_ROOT / "results" / "figures" / "step_3_*",
            PROJECT_ROOT / "results" / "figures" / "step_4_*",
            PROJECT_ROOT / "site" / "figures" / "step_3_*",
            PROJECT_ROOT / "site" / "figures" / "step_4_*"
        ],
        
        # Step 3 & 4 temporary files only
        'step3_4_temp_files': [
            PROJECT_ROOT / "results" / "tmp" / "step_3_*",
            PROJECT_ROOT / "results" / "tmp" / "step_4_*",
            PROJECT_ROOT / "results" / "exploratory" / "step_3_*",
            PROJECT_ROOT / "results" / "exploratory" / "step_4_*",
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
    Perform cleanup of Step 3 & 4 related data only.
    Preserves Step 1 & 2 outputs.
    
    Args:
        dry_run: If True, only report what would be cleaned
        
    Returns:
        Dict with cleanup summary
    """
    print_status("TEP-GNSS Clean Run - Step 3 & 4 Selective Cleanup Phase", "TITLE")
    
    targets = get_cleanup_targets()
    total_stats = {'files_removed': 0, 'size_freed': 0, 'categories': {}}
    
    if dry_run:
        print_status("DRY RUN MODE - No files will actually be deleted", "WARNING")
        
        # Calculate sizes for dry run
        sizes = calculate_cleanup_size(targets)
        total_size = sum(cat['size_bytes'] for cat in sizes.values())
        total_files = sum(cat['file_count'] for cat in sizes.values())
        
        print_status(f"Would clean {total_files} files ({format_size(total_size)})", "INFO")
        print_status("PRESERVING: Step 1 & 2 outputs (data acquisition and core analysis)", "INFO")
        
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
    
    print_status(f"Selective cleanup complete: {total_stats['files_removed']} files removed ({format_size(total_stats['size_freed'])})", "SUCCESS")
    print_status("PRESERVED: Step 1 & 2 outputs (data acquisition and core analysis)", "SUCCESS")
    
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

def execute_step3_4_pipeline() -> bool:
    """
    Execute the Step 3 & 4 pipeline sequentially.
    
    Returns:
        True if all steps completed successfully, False otherwise
    """
    print_status("TEP-GNSS Clean Run - Step 3 & 4 Pipeline Execution", "TITLE")
    
    # Define step execution order
    steps = [
        # Step 3: Validation Suite
        {
            'script': PROJECT_ROOT / "scripts" / "clean_run_step3.py",
            'name': "Step 3: Validation Suite Pipeline"
        },
        # Step 4: Advanced Analysis & Visualization
        {
            'script': PROJECT_ROOT / "scripts" / "clean_run_step4.py",
            'name': "Step 4: Advanced Analysis & Visualization Pipeline"
        }
    ]
    
    start_time = time.time()
    successful_steps = 0
    
    for i, step in enumerate(steps, 1):
        logger.info(f"Starting pipeline phase {i}/{len(steps)}: {step['name']}")
        
        if not step['script'].exists():
            logger.error(f"Step script not found: {step['script']}")
            return False
        
        step_start = time.time()
        success = run_step_script(step['script'], step['name'])
        step_elapsed = time.time() - step_start
        
        if success:
            successful_steps += 1
            logger.success(f"Phase {i} completed in {step_elapsed:.1f} seconds")
        else:
            logger.error(f"Phase {i} failed after {step_elapsed:.1f} seconds")
            logger.error("Step 3 & 4 pipeline execution stopped due to step failure")
            return False
    
    total_elapsed = time.time() - start_time
    
    if successful_steps == len(steps):
        print_status(f"Step 3 & 4 pipeline completed successfully in {total_elapsed:.1f} seconds", "SUCCESS")
        return True
    else:
        logger.error(f"Pipeline failed: {successful_steps}/{len(steps)} phases completed")
        return False

def validate_step3_4_pipeline() -> bool:
    """
    Validate that the Step 3 & 4 pipeline run was successful by checking for expected outputs.
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.process("Validating Step 3 & 4 pipeline results...")
    
    # Check for key outputs from Step 3 & 4
    validation_checks = [
        # Step 3 outputs (at least one validation)
        (PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_code.json", "Step 3: Cross-validation results"),
        
        # Step 4 outputs (at least one advanced analysis)
        (PROJECT_ROOT / "results" / "outputs" / "step_4_0_advanced_analysis.json", "Step 4: Advanced analysis results"),
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
    
    # Verify that Step 1 & 2 outputs are still preserved
    preserved_checks = [
        (PROJECT_ROOT / "data" / "coordinates" / "step_1_1_station_coords_global.csv", "Step 1: Station coordinates"),
        (PROJECT_ROOT / "results" / "outputs" / "step_2_0_correlation_code.json", "Step 2: Core correlation analysis"),
    ]
    
    missing_preserved = []
    for output_path, description in preserved_checks:
        if not output_path.exists():
            missing_preserved.append(f"  {description}: {output_path}")
    
    if missing_preserved:
        logger.warning("Some Step 1 & 2 outputs are missing (this may be expected if not previously run):")
        for missing in missing_preserved:
            logger.warning(missing)
    
    logger.success("Step 3 & 4 pipeline validation passed - found expected outputs")
    return True

@ensure_single_instance
def main():
    """Main Step 3 & 4 pipeline clean run execution."""
    parser = argparse.ArgumentParser(description='TEP-GNSS Clean Run - Step 3 & 4 Pipeline')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without actually cleaning')
    parser.add_argument('--skip-cleanup', action='store_true',
                       help='Skip cleanup and only run the steps')
    
    args = parser.parse_args()
    
    # Reset master log file for fresh start
    # reset_master_log() # DEPRECATED: Each step now has its own log file
    
    print_status("TEP-GNSS Clean Run Script v1.0", "TITLE")
    print_status("Step 3 & 4 Pipeline Clean Run (Preserves Step 1 & 2 Outputs)", "TITLE")
    
    # Phase 1: Selective Cleanup (unless skipped)
    if not args.skip_cleanup:
        cleanup_stats = perform_cleanup(dry_run=args.dry_run)
        
        if args.dry_run:
            print_status("Dry run completed - no changes made", "SUCCESS")
            return True
        
        if cleanup_stats['files_removed'] == 0:
            logger.info("No Step 3 & 4 files needed cleaning - starting fresh")
    else:
        logger.info("Skipping cleanup phase")
    
    # Phase 2: Execute Step 3 & 4 Pipeline
    if not args.dry_run:
        pipeline_success = execute_step3_4_pipeline()
        
        if not pipeline_success:
            logger.error("Step 3 & 4 pipeline execution failed")
            return False
        
        # Phase 3: Validate Results
        validation_success = validate_step3_4_pipeline()
        
        if validation_success:
            print_status("Step 3 & 4 pipeline clean run completed successfully!", "SUCCESS")
            return True
        else:
            logger.error("Step 3 & 4 pipeline validation failed")
            return False
    else:
        print_status("Dry run completed - no pipeline execution", "SUCCESS")
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
