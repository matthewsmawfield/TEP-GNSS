#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 0: Provenance Documentation
===================================================

Establishes complete computational provenance for reproducible research.
Documents analysis environment, data sources, and processing state to ensure
full transparency and scientific reproducibility.

Outputs: results/outputs/provenance_snapshot.json

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha256sum(p: Path, max_bytes: int | None = None) -> str:
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
    items = []
    for p in sorted(d.glob(glob)):
        try:
            items.append({
                "path": str(p.relative_to(ROOT)),
                "size_bytes": p.stat().st_size,
                "sha256_1mb": sha256sum(p, max_bytes=1024*1024)
            })
        except FileNotFoundError:
            continue
    return items

def csv_count(p: Path) -> int:
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return -1

def main():
    print("="*80)
    print("TEP GNSS Analysis Package v0.9")
    print("STEP 0: Provenance Documentation")
    print("="*80)
    
    out_dir = ROOT / 'results' / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    env_keys = [
        'TEP_MIN_STATIONS','TEP_SKIP_COORDS',
        'TEP_FILES_PER_CENTER','TEP_FILES_PER_CENTER_IGS','TEP_FILES_PER_CENTER_CODE','TEP_FILES_PER_CENTER_ESA',
        'TEP_INCLUDE_LOGS','TEP_LOGS_MAX','TEP_LOGS_CONCURRENCY'
    ]
    env = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}

    snapshot = {
        "env": env,
        "raw_files": list_dir(ROOT / 'data' / 'raw' / 'igs_combined', '*.gz')
                     + list_dir(ROOT / 'data' / 'raw' / 'code', '*.gz')
                     + list_dir(ROOT / 'data' / 'raw' / 'esa_final', '*.gz'),
        "processed_files": [],  # Step 3 removed - processing done directly in Step 4
        "results_files": list_dir(ROOT / 'results' / 'outputs', '*.json') + list_dir(ROOT / 'results' / 'outputs', '*.csv'),
        "counts": {
            "coords_stations": csv_count(ROOT / 'data' / 'coordinates' / 'station_coords_global.csv')
        }
    }

    with open(out_dir / 'provenance_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"wrote {out_dir / 'provenance_snapshot.json'}")

if __name__ == '__main__':
    main()


