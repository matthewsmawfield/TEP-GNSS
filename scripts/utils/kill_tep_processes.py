#!/usr/bin/env python3
"""
TEP-GNSS Process Cleanup Utility

This script provides a simple way to kill all TEP-GNSS related Python processes.
Use this when you need to ensure all TEP processes are terminated.

Usage:
    python scripts/utils/kill_tep_processes.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.pid_manager import kill_all_tep_processes

def main():
    """Kill all TEP-GNSS processes."""
    print("TEP-GNSS Process Cleanup Utility")
    print("=" * 50)
    
    killed_count = kill_all_tep_processes()
    
    if killed_count > 0:
        print(f"✅ Successfully killed {killed_count} TEP-GNSS process(es)")
    else:
        print("✅ No TEP-GNSS processes were running")
    
    return killed_count

if __name__ == "__main__":
    try:
        killed = main()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
