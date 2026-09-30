#!/usr/bin/env python3
"""
TEP GNSS Analysis - STEP 4.4b (audit diagnostic): Autocorrelation-corrected
significance audit for the gravitational-temporal field correlations.

Replicates the Step 4.4 computation exactly on the pipeline's own exported
daily series (site/data/step_4_4/step_4_4_gravitational_temporal_daily.json),
then reports, for every smoothing window and every individual planetary
influence:

  - raw Pearson r and p
  - Bretherton et al. (1999) effective sample size N_eff
  - autocorrelation-corrected p-value (identical formula to Step 4.4)

It also reports a selection-adjusted significance for the best window,
since Step 4.4 selects the window with maximum |r| from a fixed scan.

This is an audit diagnostic (not part of the canonical pipeline runner); it
exists so that every p-value quoted in the manuscript traces to a computed,
autocorrelation-aware statistic.

Inputs:
  - site/data/step_4_4/step_4_4_gravitational_temporal_daily.json

Outputs:
  - results/outputs/step_4_4b_autocorrelation_audit.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def autocorr_robust_correlation(x, y, max_lags=None):
    """Identical to the implementation in step_4_4."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if max_lags is None:
        max_lags = min(20, n // 5)
    r_raw, p_raw = stats.pearsonr(x, y)
    acf_x = acf(x, nlags=max_lags, alpha=0.05)
    acf_y = acf(y, nlags=max_lags, alpha=0.05)
    r1_x = acf_x[0][1] if len(acf_x[0]) > 1 else 0
    r1_y = acf_y[0][1] if len(acf_y[0]) > 1 else 0
    r1_x = float(np.clip(r1_x, -0.95, 0.95))
    r1_y = float(np.clip(r1_y, -0.95, 0.95))
    n_eff = n * (1 - r1_x * r1_y) / (1 + r1_x * r1_y)
    n_eff = max(10, n_eff)
    se = np.sqrt((1 - r_raw ** 2) / (n_eff - 2))
    t_stat = r_raw / se
    df = n_eff - 2
    p_corr = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    return {
        "correlation": float(r_raw),
        "p_value_raw": float(p_raw),
        "p_value_autocorr_corrected": float(p_corr),
        "n_effective": float(n_eff),
        "n_original": int(n),
        "autocorr_x": r1_x,
        "autocorr_y": r1_y,
    }


def main():
    daily_path = PACKAGE_ROOT / "site/data/step_4_4/step_4_4_gravitational_temporal_daily.json"
    daily = json.loads(daily_path.read_text())

    n = len(daily["dates"])
    total_planetary = np.asarray(daily["total_planetary_influence"], dtype=float)
    coherence_std = np.asarray(daily["coherence_std"], dtype=float)

    out = {
        "audit_type": "autocorrelation_corrected_significance",
        "source_data": str(daily_path.relative_to(PACKAGE_ROOT)),
        "n_days": n,
        "note": (
            "Replicates step_4_4 smoothed-window scan and per-planet "
            "correlations on the pipeline-exported daily series; all p-values "
            "are Bretherton-corrected effective-sample-size statistics."
        ),
    }

    # ---- Multi-window scan (identical adjusted_window logic to step_4_4) ----
    test_windows = [30, 60, 91, 120, 180, 240, 365]
    scan = []
    for w in test_windows:
        aw = min(w, n // 4)
        if aw % 2 == 0:
            aw -= 1
        if aw < 31:
            aw = 31
        poly = min(3, aw - 2)
        poly = max(1, poly)
        if aw > poly and aw >= 31 and n > aw:
            sx = savgol_filter(total_planetary, aw, poly)
            sy = savgol_filter(coherence_std, aw, poly)
            res = autocorr_robust_correlation(sx, sy)
            res["smoothing_window"] = int(aw)
            scan.append(res)

    best = max(scan, key=lambda d: abs(d["correlation"]))
    best["p_value_selection_adjusted"] = float(
        min(1.0, best["p_value_autocorr_corrected"] * len(scan))
    )
    out["window_scan"] = scan
    out["best_window"] = best
    out["n_windows_tested"] = len(scan)

    # Raw (unsmoothed) total-planet vs coherence_std — the 'raw correlation'
    out["raw_total_vs_coherence_std"] = autocorr_robust_correlation(
        total_planetary, coherence_std
    )

    # ---- Individual planetary influences (unsmoothed daily series) ----
    planets = {}
    for name, series in daily["individual_influences"].items():
        x = np.asarray(series, dtype=float)
        planets[name] = {
            "coherence_std": autocorr_robust_correlation(x, coherence_std),
            "coherence_mean": autocorr_robust_correlation(
                x, np.asarray(daily["coherence_mean"], dtype=float)
            ),
        }
    out["planetary_influences"] = planets

    # Total influence (incl. Sun-dominated aggregate)
    out["total_influence_vs_coherence_std"] = autocorr_robust_correlation(
        np.asarray(daily["total_influence"], dtype=float), coherence_std
    )

    out_path = PACKAGE_ROOT / "results/outputs/step_4_4b_autocorrelation_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    # Console summary
    print("\nwindow  r        p_raw      p_corr     N_eff")
    for s in scan:
        print(
            f"{s['smoothing_window']:>5}  {s['correlation']:+.3f}  "
            f"{s['p_value_raw']:.2e}  {s['p_value_autocorr_corrected']:.2e}  "
            f"{s['n_effective']:.1f}"
        )
    print(
        f"\nBest window {best['smoothing_window']}d: r={best['correlation']:+.3f}, "
        f"p_corr={best['p_value_autocorr_corrected']:.2e}, "
        f"p_sel_adj={best['p_value_selection_adjusted']:.2e}"
    )
    print("\nplanet   r(coh_std)  p_raw      p_corr     N_eff")
    for name, d in planets.items():
        c = d["coherence_std"]
        print(
            f"{name:<8} {c['correlation']:+.3f}     {c['p_value_raw']:.2e}  "
            f"{c['p_value_autocorr_corrected']:.2e}  {c['n_effective']:.1f}"
        )


if __name__ == "__main__":
    sys.exit(main())
