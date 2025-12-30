#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 1.0: Provenance Documentation
===================================================

Establishes complete computational provenance for reproducible research.
Documents analysis environment, data sources, and processing state to ensure
full transparency and scientific reproducibility.

Outputs: results/outputs/step_1_0_provenance_snapshot.json

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add utils to path for imports
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.utils.provenance import update_provenance_snapshot, verify_data_integrity
from scripts.utils.logger import print_status, TEPLogger, set_step_logger
from scripts.utils.pid_manager import ensure_single_instance
from scripts.utils.exceptions import TEPFileError, TEPDataError, TEPAnalysisError

# Initialize step-specific logger
step_logger = TEPLogger(
    name="step_1_0_provenance_snapshot",
    level="DEBUG",
    log_file_path=ROOT / "logs" / "step_1_0_provenance_snapshot.log"
)

def setup_logging():
    """Setup logging - no-op since we're using TEPLogger."""
    pass

@ensure_single_instance
def main():
    setup_logging()
    set_step_logger(step_logger)
    from scripts.utils.version_utils import VERSION_STRING
    print_status(f"TEP GNSS Analysis Package {VERSION_STRING} - STEP 1.0: Provenance Documentation", "INFO")
    print_status("="*80, "INFO")
    
    try:
        # Update provenance snapshot
        success = update_provenance_snapshot('step_1_0_provenance_snapshot')
        
        if success:
            print_status("Successfully updated provenance snapshot", "SUCCESS")
            
            # Verify data integrity
            verification = verify_data_integrity()
            if verification.get("status") == "success":
                print_status("Data integrity verification passed", "SUCCESS")
            else:
                raise TEPDataError(f"Data integrity verification failed: {verification.get('message', 'Unknown error')}")
        else:
            raise TEPAnalysisError("Failed to update provenance snapshot")
        
        return True
    
    except TEPFileError as e:
        print_status(f"Provenance snapshot failed due to file error: {e}", "ERROR")
        sys.exit(1)
    except TEPDataError as e:
        print_status(f"Provenance snapshot failed due to data error: {e}", "ERROR")
        sys.exit(1)
    except TEPAnalysisError as e:
        print_status(f"Provenance snapshot failed due to analysis error: {e}", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred during provenance snapshot: {e}", "CRITICAL")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()


