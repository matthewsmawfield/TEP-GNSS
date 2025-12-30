#!/usr/bin/env python3
"""
TEP GNSS Analysis - Provenance Utility
=====================================

Provides provenance tracking functionality for reproducible research.
Can be imported and used by any pipeline step to update provenance documentation.

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Get package root
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

def sha256sum(p: Path, max_bytes: int | None = None) -> str:
    """Calculate SHA256 hash of file, optionally limited to first max_bytes."""
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            if max_bytes is None:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    h.update(chunk)
            else:
                h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return ""

def list_dir(d: Path, glob: str) -> list[dict]:
    """List files in directory with metadata."""
    items = []
    for p in sorted(d.glob(glob)):
        try:
            items.append({
                "path": str(p.relative_to(PACKAGE_ROOT)),
                "size_bytes": p.stat().st_size,
                "sha256_1mb": sha256sum(p, max_bytes=1024*1024)
            })
        except FileNotFoundError:
            continue
    return items

def csv_count(p: Path) -> int:
    """Count lines in CSV file (excluding header)."""
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return -1

def update_provenance_snapshot(step_name: str, additional_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update the provenance snapshot with current state.
    
    Args:
        step_name: Name of the step that triggered the update
        additional_data: Optional additional data to include in snapshot
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        out_dir = PACKAGE_ROOT / 'results' / 'outputs'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment variables
        env_keys = [
            'TEP_MIN_STATIONS', 'TEP_SKIP_COORDS',
            'TEP_FILES_PER_CENTER', 'TEP_FILES_PER_CENTER_IGS', 
            'TEP_FILES_PER_CENTER_CODE', 'TEP_FILES_PER_CENTER_ESA',
            'TEP_INCLUDE_LOGS', 'TEP_LOGS_MAX', 'TEP_LOGS_CONCURRENCY'
        ]
        env = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
        
        # Build snapshot
        snapshot = {
            "last_updated_by": step_name,
            "last_updated": datetime.now().isoformat(),
            "env": env,
            "raw_files": (list_dir(PACKAGE_ROOT / 'data' / 'raw' / 'igs_combined', '*.gz') +
                         list_dir(PACKAGE_ROOT / 'data' / 'raw' / 'code', '*.gz') +
                         list_dir(PACKAGE_ROOT / 'data' / 'raw' / 'esa_final', '*.gz')),
            "processed_files": list_dir(PACKAGE_ROOT / 'data' / 'coordinates', '*.csv'),
            "results_files": (list_dir(PACKAGE_ROOT / 'results' / 'outputs', '*.json') + 
                            list_dir(PACKAGE_ROOT / 'results' / 'outputs', '*.csv')),
            "counts": {
                "coords_stations": csv_count(PACKAGE_ROOT / 'data' / 'coordinates' / 'step_1_1_station_coords_global.csv')
            }
        }
        
        # Add any additional data
        if additional_data:
            snapshot.update(additional_data)
        
        # Write snapshot to results/outputs directory only
        snapshot_path = out_dir / 'step_1_0_provenance_snapshot.json'
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        return True
        
    except Exception as e:
        print(f"Error updating provenance snapshot: {e}")
        return False

def get_provenance_snapshot() -> Optional[Dict[str, Any]]:
    """Load and return the current provenance snapshot."""
    try:
        snapshot_path = PACKAGE_ROOT / 'results' / 'outputs' / 'step_1_0_provenance_snapshot.json'
        if snapshot_path.exists():
            with open(snapshot_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading provenance snapshot: {e}")
    return None

def verify_data_integrity() -> Dict[str, Any]:
    """
    Verify data integrity by checking file checksums and counts.
    
    Returns:
        Dict with verification results
    """
    snapshot = get_provenance_snapshot()
    if not snapshot:
        return {"status": "error", "message": "No provenance snapshot found"}
    
    verification = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check raw files
    current_raw = (list_dir(PACKAGE_ROOT / 'data' / 'raw' / 'igs_combined', '*.gz') +
                  list_dir(PACKAGE_ROOT / 'data' / 'raw' / 'code', '*.gz') +
                  list_dir(PACKAGE_ROOT / 'data' / 'raw' / 'esa_final', '*.gz'))
    
    verification["checks"]["raw_files"] = {
        "expected_count": len(snapshot.get("raw_files", [])),
        "current_count": len(current_raw),
        "match": len(snapshot.get("raw_files", [])) == len(current_raw)
    }
    
    # Check processed files
    current_processed = list_dir(PACKAGE_ROOT / 'data' / 'coordinates', '*.csv')
    verification["checks"]["processed_files"] = {
        "expected_count": len(snapshot.get("processed_files", [])),
        "current_count": len(current_processed),
        "match": len(snapshot.get("processed_files", [])) == len(current_processed)
    }
    
    return verification

if __name__ == '__main__':
    # Command line interface
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'update':
            step_name = sys.argv[2] if len(sys.argv) > 2 else 'manual'
            success = update_provenance_snapshot(step_name)
            print(f"Provenance update {'successful' if success else 'failed'}")
        elif sys.argv[1] == 'verify':
            result = verify_data_integrity()
            print(json.dumps(result, indent=2))
        else:
            print("Usage: python provenance.py [update <step_name>|verify]")
    else:
        # Default: update with manual trigger
        success = update_provenance_snapshot('manual')
        print(f"Provenance snapshot {'updated' if success else 'failed'}")
