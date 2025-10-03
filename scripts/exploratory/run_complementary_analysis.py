#!/usr/bin/env python3
"""
Run TEP Complementary Metrics Analysis
=====================================

Simple execution script for the complementary metrics validation analysis.

Usage:
    python scripts/exploratory/run_complementary_analysis.py

This will:
1. Analyze existing correlation data
2. Compute theoretical complementary metrics
3. Generate comparison plots
4. Create detailed analysis report

Author: Matthew Lukin Smawfield
Date: January 2025
"""

import os
import sys
from pathlib import Path

# Add the exploratory directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import and run the analysis
from tep_complementary_analysis import main

if __name__ == "__main__":
    print("=" * 60)
    print("TEP Circular Statistics Theoretical Foundation Analysis")
    print("=" * 60)
    print()
    print("IMPORTANT: This provides theoretical interpretation,")
    print("not independent empirical validation.")
    print()
    
    success = main()
    
    if success:
        print()
        print("=" * 60)
        print("Analysis completed successfully!")
        print("Check results/exploratory/ for outputs:")
        print("- tep_circular_statistics_foundation.png")
        print("- tep_circular_statistics_foundation_report.md")
        print("=" * 60)
    else:
        print()
        print("=" * 60)
        print("Analysis failed - check error messages above")
        print("=" * 60)
        sys.exit(1)
