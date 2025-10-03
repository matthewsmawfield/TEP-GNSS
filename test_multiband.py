#!/usr/bin/env python3
"""
Test script for multi-band frequency analysis
"""

import sys
import os
sys.path.append('scripts')

from steps.step_3_validation_suite.step_3_6_control_band_analysis import (
    FREQUENCY_BANDS, 
    run_multiband_analysis,
    print_status
)

def test_multiband_analysis():
    """Test the multi-band analysis functionality with a subset of bands."""
    
    # Test with a smaller subset of bands for faster testing
    test_bands = {
        'tep_band': FREQUENCY_BANDS['tep_band'],
        'control_1': FREQUENCY_BANDS['control_1'],
        'intermediate': FREQUENCY_BANDS['intermediate']
    }
    
    print_status("Testing Multi-Band Analysis", "TITLE")
    print_status("=" * 60, "INFO")
    
    try:
        # Run analysis on CODE center with test bands
        result = run_multiband_analysis('code', test_bands)
        
        print_status("Test Results:", "SUCCESS")
        comparison = result.get('comparison', {})
        
        if 'strongest_band' in comparison:
            strongest = comparison['strongest_band']
            weakest = comparison['weakest_band']
            specificity = comparison['specificity_metrics']['frequency_specificity']
            
            print_status(f"Frequency Specificity: {specificity}", "INFO")
            print_status(f"Strongest Signal: {strongest['name']} (R²={strongest['r_squared']:.3f})", "INFO")
            print_status(f"Weakest Signal: {weakest['name']} (R²={weakest['r_squared']:.3f})", "INFO")
            print_status(f"Signal Ratio: {comparison['specificity_metrics']['r_squared_ratio']:.1f}x", "INFO")
            
            print_status("", "INFO")
            print_status("Multi-band analysis test completed successfully!", "SUCCESS")
        else:
            print_status("No comparison results available", "WARNING")
            
    except Exception as e:
        print_status(f"Test failed: {e}", "ERROR")
        raise

if __name__ == "__main__":
    test_multiband_analysis()

