#!/usr/bin/env python3
"""
TEP GNSS Analysis - Clean Full-Pipeline Orchestrator
====================================================

Runs the complete documented pipeline (Steps 1.0 onward) end-to-end:
every ``step_*.py`` under ``scripts/steps/`` in lexicographic order,
streaming stdout/stderr to the console and to per-step log files under
``logs/``, and recording per-step status in
``results/outputs/pipeline_run_summary.json``.

Cleaning (matches the documented behaviour "cleans all previous data,
outputs, logs, and figures for a fresh start", with one deliberate
deviation):

- ``--clean`` removes ``results/``, ``logs/`` and ``figures/`` contents.
- The raw IGS archive under ``data/`` is preserved by default because a
  fresh acquisition re-downloads ~912 days of products; pass
  ``--clean-data`` to remove it too.

Usage:
    python scripts/clean_run_full_pipeline.py [--clean] [--clean-data]
                                              [--steps 1,2] [--skip-clean]

Author: Matthew Lukin Smawfield
Theory: Temporal Equivalence Principle (TEP)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = ROOT / "scripts" / "steps"
LOGS_DIR = ROOT / "logs"


def discover_steps() -> list[Path]:
    return sorted(STEPS_DIR.rglob("step_*.py"))


def step_major(path: Path) -> int:
    # step_3_8_spatial_gls_refit.py -> 3
    digits = "".join(ch for ch in path.stem.split("_", 2)[1] if ch.isdigit() or ch == "_")
    return int(digits.split("_")[0])


def clean_outputs(clean_data: bool) -> None:
    for name in ("results", "logs", "figures"):
        d = ROOT / name
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    if clean_data:
        d = ROOT / "data"
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean", action="store_true",
                    help="remove results/, logs/ and figures/ before running")
    ap.add_argument("--clean-data", action="store_true",
                    help="also remove the raw data/ archive (implies --clean)")
    ap.add_argument("--steps", default=None,
                    help="comma-separated major step numbers to run, e.g. '3,4'")
    ap.add_argument("--skip-clean", action="store_true",
                    help="run steps without cleaning (overrides --clean)")
    args = ap.parse_args()

    if args.clean_data and not args.skip_clean:
        args.clean = True
    if (args.clean or args.clean_data) and not args.skip_clean:
        print("[pipeline] cleaning results/, logs/, figures/"
              + (" and data/" if args.clean_data else ""), flush=True)
        clean_outputs(args.clean_data)

    majors = None
    if args.steps:
        majors = {int(x) for x in args.steps.split(",")}

    steps = discover_steps()
    if majors is not None:
        steps = [s for s in steps if step_major(s) in majors]
    if not steps:
        print("[pipeline] no step scripts matched", flush=True)
        return 1

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "outputs").mkdir(parents=True, exist_ok=True)

    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }
    t0 = time.time()
    failures = 0

    for script in steps:
        rel = script.relative_to(ROOT)
        log_path = LOGS_DIR / f"{script.stem}.log"
        print(f"[pipeline] === {rel} ===", flush=True)
        t_start = time.time()
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
            logf.write(proc.stdout or "")
        tail = (proc.stdout or "").splitlines()
        for line in tail[-20:]:
            print(f"    {line}", flush=True)
        ok = proc.returncode == 0
        failures += 0 if ok else 1
        summary["steps"].append({
            "script": str(rel),
            "returncode": proc.returncode,
            "status": "ok" if ok else "failed",
            "runtime_s": round(time.time() - t_start, 1),
            "log": str(log_path.relative_to(ROOT)),
        })
        print(f"[pipeline] {'OK' if ok else 'FAILED'} "
              f"({summary['steps'][-1]['runtime_s']} s)", flush=True)

    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["runtime_s"] = round(time.time() - t0, 1)
    summary["n_steps"] = len(steps)
    summary["n_failed"] = failures
    summary["status"] = "ok" if failures == 0 else "failed"
    out = ROOT / "results" / "outputs" / "pipeline_run_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[pipeline] done: {len(steps)} steps, {failures} failed "
          f"({summary['runtime_s']} s) -> {out.relative_to(ROOT)}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
