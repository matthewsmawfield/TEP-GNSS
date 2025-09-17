#!/usr/bin/env python3
"""
TEP GNSS Analysis - Globe Connection Visualization Generator
===========================================================

Standalone script to generate globe visualizations showing GNSS station 
connections based on temporal correlation strength. Creates publication-ready
figures with great circle arcs connecting correlated station pairs.

Usage:
    python scripts/generate_globe_connections.py [options]

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.steps.step_8_tep_visualization import create_three_globe_views_with_connections

def main():
    parser = argparse.ArgumentParser(
        description='Generate globe visualizations with GNSS station connections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for CODE analysis center with default settings
    python scripts/generate_globe_connections.py --ac code
    
    # High-coherence connections only
    python scripts/generate_globe_connections.py --ac igs_combined --threshold 0.8 --max-connections 1000
    
    # Generate for all analysis centers
    python scripts/generate_globe_connections.py --all
        """
    )
    
    parser.add_argument('--ac', '--analysis-center', 
                       choices=['code', 'igs_combined', 'esa_final'],
                       help='Analysis center to process')
    
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Minimum coherence threshold for connections (default: 0.7)')
    
    parser.add_argument('--max-connections', type=int, default=2000,
                       help='Maximum number of connections to display (default: 2000)')
    
    parser.add_argument('--all', action='store_true',
                       help='Generate for all analysis centers with optimized settings')
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent
    
    if args.all:
        # Optimized configurations for each analysis center
        configs = [
            ('code', 0.7, 1500),
            ('igs_combined', 0.6, 2000), 
            ('esa_final', 0.8, 1000)
        ]
        
        print("Generating globe visualizations for all analysis centers...")
        results = []
        
        for ac, threshold, max_conn in configs:
            print(f"\n{'='*60}")
            print(f"Processing {ac.upper()}")
            print(f"Threshold: {threshold}, Max connections: {max_conn}")
            print('='*60)
            
            try:
                result = create_three_globe_views_with_connections(
                    root_dir, 
                    analysis_center=ac,
                    coherence_threshold=threshold,
                    max_connections=max_conn
                )
                results.append(result)
                print(f"✅ SUCCESS: {result}")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"Generated {len(results)} globe visualizations:")
        for result in results:
            print(f"  - {result}")
            
    elif args.ac:
        print(f"Generating globe visualization for {args.ac.upper()}")
        print(f"Coherence threshold: {args.threshold}")
        print(f"Max connections: {args.max_connections}")
        
        try:
            result = create_three_globe_views_with_connections(
                root_dir,
                analysis_center=args.ac,
                coherence_threshold=args.threshold,
                max_connections=args.max_connections
            )
            print(f"✅ SUCCESS: {result}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
