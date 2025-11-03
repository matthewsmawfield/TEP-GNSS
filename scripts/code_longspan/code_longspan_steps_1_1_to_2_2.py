#!/usr/bin/env python3
"""
Exploratory driver: CODE-only long-span run (Steps 1.1 -> 2.2) with isolated namespace.
This script does not modify the main pipeline or its outputs/logs.

Usage:
  python scripts/code_longspan/code_longspan_steps_1_1_to_2_2.py \
    --namespace code_longspan_2000_2025 \
    --date-start 2000-01-01 \
    --date-end 2025-06-30

Notes:
- All outputs and logs are written under results/outputs/<namespace>/ and logs/<namespace>/
- Coordinates are saved under data/coordinates/<namespace>/
- Steps are run sequentially: 1.1, 1.2, 2.0, 2.1, 2.2 (CODE-only)
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_step(pyfile: Path, args_list=None, env=None) -> None:
    if not pyfile.exists():
        raise FileNotFoundError(f"Step script not found: {pyfile}")
    cmd = [sys.executable, str(pyfile)]
    if args_list:
        cmd.extend(args_list)
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    # Ensure PYTHONPATH includes project root
    current_env["PYTHONPATH"] = f"{ROOT}:{current_env.get('PYTHONPATH','')}" if current_env.get('PYTHONPATH') else str(ROOT)
    current_env["PYTHONUNBUFFERED"] = "1"
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=ROOT, env=current_env)
    if res.returncode != 0:
        raise RuntimeError(f"Step failed: {pyfile.name} (exit {res.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Exploratory CODE-only long-span pipeline driver (1.1 -> 2.2)")
    parser.add_argument("--namespace", default="code_longspan_2000_2025", help="Namespace for logs/outputs")
    parser.add_argument("--date-start", default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--date-end", default="2025-06-30", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    ns = args.namespace
    env = {
        "TEP_DATE_START": args.date_start,
        "TEP_DATE_END": args.date_end,
        "TEP_OUTPUT_NAMESPACE": ns,
        "TEP_LOG_NAMESPACE": ns,
    }

    # Step 1.1 (exploratory copy, CODE-only)
    run_step(ROOT / "scripts/code_longspan/step_1_1_code_longspan.py", env=env)

    # Step 1.2 (exploratory copy)
    run_step(ROOT / "scripts/code_longspan/step_1_2_code_longspan.py", env=env)

    # Step 2.0 (exploratory copy, force center=code)
    run_step(ROOT / "scripts/code_longspan/step_2_0_code_longspan.py", args_list=["--center", "code"], env=env)

    # Step 2.1 (exploratory copy)
    run_step(ROOT / "scripts/code_longspan/step_2_1_code_longspan.py", env=env)

    # Step 2.2 (exploratory copy)
    run_step(ROOT / "scripts/code_longspan/step_2_2_code_longspan.py", args_list=["--center", "code"], env=env)

    print("\nAll exploratory steps completed successfully (CODE-only long span).\n")


if __name__ == "__main__":
    main()
