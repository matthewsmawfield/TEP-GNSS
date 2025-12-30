#!/usr/bin/env python3
"""
TEP-GNSS Clean Run Script - Step 3 Validation Suite
==================================================

Performs a complete clean run of Step 3 Validation Suite with all substeps:
- Removes validation outputs, logs, and temporary files
- Executes Step 3.0 (Cross-Validation Suite)
- Executes Step 3.1 (Robust Block Bootstrap)
- Executes Step 3.2 (TEP Null Tests)
- Executes Step 3.3 (Methodology Validation)
- Executes Step 3.4 (Geographic Bias Validation)
- Executes Step 3.5 (Realistic Ionospheric Validation)
- Executes Step 3.6 (Control Band Analysis)
- Executes Step 3.7 (Multiple Comparison Corrections)

This script ensures a completely fresh start for the TEP validation pipeline.

Usage:
    python scripts/clean_run_step3.py [--dry-run] [--skip-cleanup]

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
from typing import List, Dict, Set, Tuple
import json
import multiprocessing as mp

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set multiprocessing start method for robustness
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        pass # Do not print warning here, as it's handled by individual scripts

from scripts.utils.config import TEPConfig
from scripts.utils.logger import TEPLogger, print_status, reset_master_log
from scripts.utils.exceptions import SafeErrorHandler
from scripts.utils.pid_manager import ensure_single_instance

# Initialize logger
logger = TEPLogger(name="clean_run_step3", level="DEBUG", log_file_path=PROJECT_ROOT / "logs" / "clean_run_step3.log") # Directly use TEPLogger instance

def get_cleanup_targets() -> Dict[str, List[Path]]:
    """
    Define all directories and files that need to be cleaned for a fresh Step 3 run.
    
    Returns:
        Dict with categories of cleanup targets
    """
    targets = {
        # Step 3 log files
        'log_files': [
            PROJECT_ROOT / "logs" / "step_3_0_cross_validation_suite.json",
            PROJECT_ROOT / "logs" / "step_3_1_robust_block_bootstrap.json",
            PROJECT_ROOT / "logs" / "step_3_2_tep_null_tests.json",
            PROJECT_ROOT / "logs" / "step_3_3_methodology_validation.json",
            PROJECT_ROOT / "logs" / "step_3_4_geographic_bias_validation.json",
            PROJECT_ROOT / "logs" / "step_3_5_realistic_ionospheric_validation.json",
            PROJECT_ROOT / "logs" / "step_3_6_control_band_analysis.log",
            # PROJECT_ROOT / "logs" / "step_3_7_multiple_comparison_corrections.json"
        ],
        
        # Output files for each step, organized by step name
        'step_outputs': {
            "Step 3.0: Cross-Validation Suite": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_code.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_igs_combined.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_0_cross_validation_suite_esa_final.json",
            ],
            "Step 3.1: Robust Block Bootstrap": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_1_robust_block_bootstrap_code.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_1_robust_block_bootstrap_igs_combined.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_1_robust_block_bootstrap_esa_final.json",
            ],
            "Step 3.2: TEP Null Tests": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_2_null_tests_code.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_2_null_tests_igs_combined.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_2_null_tests_esa_final.json",
            ],
            "Step 3.3: Methodology Validation": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_3_validation_report.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_3_bias_characterization.json",
            ],
            "Step 3.4: Geographic Bias Validation": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_4_geographic_bias_validation.json", # No AC suffix for this one
            ],
            "Step 3.5: Realistic Ionospheric Validation": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_5_realistic_ionospheric_validation.json", # No AC suffix for this one
            ],
            "Step 3.6: Control Band Analysis": [
                PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_code.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_igs_combined.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_6_control_band_esa_final.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_6_band_comparison_code.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_6_band_comparison_igs_combined.json",
                PROJECT_ROOT / "results" / "outputs" / "step_3_6_band_comparison_esa_final.json",
            ],
            # "Step 3.7: Multiple Comparison Corrections": [
            #     PROJECT_ROOT / "results" / "outputs" / "step_3_7_multiple_comparison_corrections.json", # No AC suffix for this one
            #     PROJECT_ROOT / "results" / "outputs" / "step_3_7_corrected_significance_summary.json",
            #     PROJECT_ROOT / "results" / "outputs" / "step_3_7_correction_impact_analysis.csv",
            # ],
        },
        
        # Validation-specific temporary files (globs)
        'temp_files': [
            PROJECT_ROOT / "results" / "tmp" / "step_3_*",
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
        
        if category == 'step_outputs':
            for step_name, output_paths in paths.items():
                for path in output_paths:
                    if path.is_file():
                        total_size += path.stat().st_size
                        file_count += 1
        else:
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
    Perform complete cleanup of all Step 3 related data.
    
    Args:
        dry_run: If True, only report what would be cleaned
        
    Returns:
        Dict with cleanup summary
    """
    print_status("TEP-GNSS Clean Run - Step 3 Validation Suite Cleanup Phase", "TITLE")
    
    targets = get_cleanup_targets()
    total_stats = {'files_removed': 0, 'size_freed': 0, 'categories': {}}
    
    if dry_run:
        print_status("DRY RUN MODE - No files will actually be deleted", "WARNING")
        
        # Calculate sizes for dry run
        sizes = calculate_cleanup_size(targets)
        total_size = sum(cat['size_bytes'] for cat in sizes.values() if isinstance(cat, dict) and 'size_bytes' in cat)
        total_files = sum(cat['file_count'] for cat in sizes.values() if isinstance(cat, dict) and 'file_count' in cat)
        
        print_status(f"Would clean {total_files} files ({format_size(total_size)})", "INFO")
        
        for category, size_info in sizes.items():
            if size_info['file_count'] > 0:
                print_status(f"  {category}: {size_info['file_count']} files ({format_size(size_info['size_bytes'])})", "INFO")
        
        return {'dry_run': True, 'total_files': total_files, 'total_size': total_size}
    
    # Perform actual cleanup
    for category, paths in targets.items():
        logger.process(f"Cleaning {category}...")
        category_stats = {'files_removed': 0, 'size_freed': 0}

        if category == 'step_outputs': # Handle step outputs separately as they are a dict
            for step_name, output_paths in paths.items():
                for path in output_paths:
                    stats = cleanup_files([path], dry_run)
                    category_stats['files_removed'] += stats['files_removed']
                    category_stats['size_freed'] += stats['size_freed']
        else: # Handle other categories (log_files, temp_files)
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
    
    print_status(f"Cleanup complete: {total_stats['files_removed']} files removed ({format_size(total_stats['size_freed'])})", "SUCCESS")
    
    return total_stats

def run_step(step_script: Path, step_name: str) -> bool:
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

        # Run the step script with real-time output
        result = subprocess.run(
            [sys.executable, "-u", str(step_script)],
            cwd=PROJECT_ROOT,
            capture_output=False,  # Don't capture output - show it in real-time
            text=True, # Decode stdout/stderr as text
            timeout=None,  # No timeout - let scientific processing complete
            env=current_env,  # Pass the modified environment
        )
        
        if result.returncode == 0:
            logger.success(f"{step_name} completed successfully")
            return True
        else:
            logger.error(f"{step_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"{step_name} timed out after 2 hours")
        return False
    except Exception as e:
        logger.error(f"Failed to execute {step_name}: {e}")
        return False

def execute_step3_pipeline(start_step: str) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Execute the complete Step 3 Validation Suite pipeline sequentially.
    
    Returns:
        True if all steps completed successfully, False otherwise, and a list of executed steps.
    """
    print_status("TEP-GNSS Clean Run - Step 3 Validation Suite Pipeline Execution", "TITLE")
    
    # Define step execution order
    steps = [
        # Step 3: Validation Suite
        {
            'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_0_tep_cross_validation_suite.py",
            'name': "Step 3.0: Cross-Validation Suite"
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
        # {
        #     'script': PROJECT_ROOT / "scripts" / "steps" / "step_3_validation_suite" / "step_3_7_multiple_comparison_corrections.py",
        #     'name': "Step 3.7: Multiple Comparison Corrections"
        # }
    ]
    
    start_time = time.time()
    successful_steps = 0
    
    # Determine the starting step index based on the provided start_step
    start_step_index = -1
    executed_steps_list = [] # Renamed to avoid conflict with function return
    for i, step in enumerate(steps):
        if step['name'].startswith(f"Step {start_step}"):
            start_step_index = i
            break
    
    if start_step_index == -1:
        logger.error(f"Could not find step '{start_step}' in the pipeline.")
        return False, [] # Return False and empty list on error

    for i, step in enumerate(steps[start_step_index:], start_step_index + 1):
        logger.info(f"Starting step {i}/{len(steps)}: {step['name']}")
        executed_steps_list.append(step) # Add executed step to the list
        
        if not step['script'].exists():
            logger.error(f"Step script not found: {step['script']}")
            return False, []
        
        step_start = time.time()
        success = run_step(step['script'], step['name'])
        step_elapsed = time.time() - step_start
        
        if success:
            successful_steps += 1
            logger.success(f"Step {i} completed in {step_elapsed:.1f} seconds")
        else:
            logger.error(f"Step {i} failed after {step_elapsed:.1f} seconds")
            logger.error("Pipeline execution stopped due to step failure")
            return False, []
    
    total_elapsed = time.time() - start_time
    
    # Calculate number of steps actually executed in this run
    num_executed_steps = len(steps[start_step_index:])

    if successful_steps == num_executed_steps:
        print_status(f"Step 3 Validation Suite completed successfully in {total_elapsed:.1f} seconds", "SUCCESS")
        return True, executed_steps_list # Return True and executed_steps_list on success
    else:
        logger.error(f"Pipeline failed: {successful_steps}/{num_executed_steps} steps completed")
        return False, []

def validate_clean_run(executed_steps: List[Dict[str, str]]) -> bool:
    """
    Validate that the Step 3 clean run was successful by checking for expected outputs.
    
    Args:
        executed_steps: A list of dictionaries, each containing 'script' and 'name' of the executed steps.
    Returns:
        True if validation passes, False otherwise
    """
    logger.process("Validating Step 3 clean run results...")
    
    # Expected outputs from Step 3
    expected_files_to_check = []
    targets = get_cleanup_targets()
    
    for step_info in executed_steps:
        step_name = step_info['name']
        
        # Add expected output files from the step_outputs dictionary
        if step_name in targets['step_outputs']:
            expected_files_to_check.extend(targets['step_outputs'][step_name])
        
        # Add expected log file (only if it exists in logs directory)
        step_name_for_file = step_name.replace(' ', '_').replace(':', '').replace('.', '_').lower()
        log_file_path = PROJECT_ROOT / "logs" / f"{step_name_for_file}.json"
        if log_file_path.exists():
            expected_files_to_check.append(log_file_path)

    missing_outputs = []
    for output_path in expected_files_to_check:
        if not output_path.exists():
            missing_outputs.append(output_path)
    
    if missing_outputs:
        logger.error("Validation failed - missing expected files:")
        for output in missing_outputs:
            logger.error(f"  {output}")
        return False
    
    logger.success("Step 3 validation passed - found all expected output files")
    return True

@ensure_single_instance
def main():
    """Main clean run execution."""
    parser = argparse.ArgumentParser(description='TEP-GNSS Clean Run - Step 3 Validation Suite')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without actually cleaning')
    parser.add_argument('--skip-cleanup', action='store_true',
                       help='Skip cleanup and only run the steps')
    parser.add_argument('--start-step', type=str, default='3.0',
                       help='Specify the starting step (e.g., "3.3")')
    # Add an argument to pass the log file path to subprocesses
    parser.add_argument('--log-file', type=str, help='Path to the log file for this run')

    args = parser.parse_args()
    
    # Reset master log file for fresh start
    # reset_master_log() # DEPRECATED: Each step now has its own log file
    
    # The global logger instance is sufficient for clean_run_step3.py
    # No need to re-initialize or set global here based on args.log_file
    # The log file for clean_run_step3.py is not explicitly handled by TEPLogger in its current design for this script

    # Extract the numerical part of the step name for comparison
    start_step_num = float(args.start_step.replace('Step ', '').split(':')[0])

    print_status("TEP-GNSS Clean Run Script v1.0", "TITLE")
    print_status("Complete Step 3 Validation Suite Clean Run", "TITLE")
    
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
    
    # Phase 2: Execute Step 3 Validation Suite Pipeline
    if not args.dry_run:
        pipeline_success, executed_steps = execute_step3_pipeline(start_step=args.start_step)
        
        if not pipeline_success:
            logger.error("Step 3 Validation Suite pipeline execution failed")
            return False
        
        # Phase 3: Validate Results
        validation_success = validate_clean_run(executed_steps=executed_steps)
        
        if validation_success:
            print_status("Step 3 Validation Suite clean run completed successfully!", "SUCCESS")
            return True
        else:
            logger.error("Step 3 clean run validation failed")
            return False
    else:
        print_status("Dry run completed - no pipeline execution", "SUCCESS")
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
