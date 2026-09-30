#!/usr/bin/env python3
"""Run TEP-GNSS pipeline major step(s) 2 via the clean-run orchestrator."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.exit(subprocess.call([sys.executable, str(ROOT / "scripts" / "clean_run_full_pipeline.py"), "--steps", "2"] + sys.argv[1:]))
