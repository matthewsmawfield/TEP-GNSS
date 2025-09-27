#!/usr/bin/env python3
"""
Step 18 Test Configuration
=========================

Test configuration for Step 18 comprehensive diurnal analysis.
This script runs a small test to validate the full pipeline before
running the complete 2023-2025-06-30 analysis.

Test Parameters:
- Date range: 2024-01-01 to 2024-01-07 (1 week)
- Centers: CODE only (fastest to process)
- Limited stations and pairs for quick validation
- Full seasonal analysis to test all functionality
"""

import subprocess
import sys
from pathlib import Path

def run_test():
    """Run Step 18 test with small dataset."""
    
    print("=== STEP 18 TEST RUN ===")
    print("Testing comprehensive diurnal analysis with small dataset...")
    print("Date range: 2024-01-01 to 2024-01-07 (1 week)")
    print("Center: CODE only")
    print("Max stations: 20")
    print("Max pairs: 500")
    print()
    
    # Test command
    cmd = [
        sys.executable,
        "scripts/steps/step_18_comprehensive_diurnal_analysis.py",
        "--start-date", "2024-01-01",
        "--end-date", "2024-01-07",
        "--centers", "code",
        "--max-workers", "4",
        "--max-stations", "20",
        "--max-pairs", "500",
        "--min-epochs", "12",
        "--min-hour-epochs", "6",
        "--window-hours", "2.0",
        "--random-seed", "42",
        "--verbose"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the test
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent, check=True)
        
        print("\n=== TEST COMPLETED SUCCESSFULLY ===")
        print("Step 18 comprehensive analysis is ready for full run!")
        print()
        print("Next steps:")
        print("1. Review test results in results/outputs/")
        print("2. Run full analysis for 2023-2025-06-30")
        print("3. Process all three centers (CODE, IGS, ESA)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Exit code: {e.returncode}")
        print("Please check the error messages above and fix any issues.")
        return False
    except Exception as e:
        print(f"\n=== TEST ERROR ===")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
