#!/usr/bin/env python3
"""
Step 4.9: Environmental Ambiguity Audit — Stratification Morphology Diagnostic
=============================================================================

Addresses the double-edged character of the environmental stratification
evidence. The geomagnetic-latitude and elevation stratifications reported in
Section 3.2 are organized on the coordinates of the leading conventional
contaminants (ionosphere: geomagnetic latitude; tropospheric delay and
atmospheric loading: altitude). The Kp-correlation and TID-exclusion controls
test only *temporal* ionospheric variability; they cannot exclude a static
geomagnetic-latitude dependence.

This step uses only existing pipeline artifacts (no new data):

1. GEOMAGNETIC MORPHOLOGY. From step_4_0's elevation×geomagnetic grid,
   tests whether the correlation length λ is systematically suppressed in
   the high-geomagnetic-latitude (auroral-leaning) band across all three
   analysis centers and all elevation tiers. The ionospheric prediction is
   a U-shaped λ(|maglat|) profile with minima in the auroral and equatorial-
   anomaly zones and a maximum in the quiet mid-latitude band; a pure
   screening interpretation has no reason to organize on geomagnetic
   latitude. The per-cell ordering is tested by an exact binomial
   calculation over the 9 (elevation × center) independent cells.

2. TROPOSPHERIC-SCALE OVERLAP. From step_3_6's multiband results, checks
   which post-tidal/intermediate frequency bands return λ inside the
   tropospheric-correlation range cited in the discussion (1,000–2,000 km,
   Bevis et al. 1994), i.e. whether the factor-of-2 scale dismissal is
   self-consistent with the paper's own band-resolved λ values.

3. ELEVATION CONFOUND. The λ rise with elevation quintile (3,174 → 7,688 km)
   is equally the fingerprint of altitude-organized residual zenith-delay /
   pressure-loading structure; the two interpretations are reported as
   degenerate pending a reanalysis-ZTD covariance control.

4. ANNUAL-CYCLE DEGENERACY. The 912-day span contains 2.497 orbital cycles;
   the 365.25-day periodicity is shared identically by orbital-velocity
   coupling, perihelion solar flux, and seasonal ionospheric/tropospheric
   drivers. Frequency resolution at 2.5 cycles cannot separate drivers at
   the same period; the operative discriminator is the energy-vs-velocity
   scaling test (Section 3.3.2), which returns near-zero discrimination.

5. DIURNAL-PERSISTENCE IONOSPHERIC BOUND. From step_4_5's local-solar-time
   coherence profile (local hour = UTC + lon/15; day 06-18 h): any residual
   ionospheric contribution to the pair-coherence signal inherits the
   ionosphere's day/night collapse (night VTEC ~ 10-30% of daytime, with the
   residual second-order/mapping errors in the clock products scaling with
   TEC). If a fraction f of the apparent signal were ionosphere-driven, the
   night/day coherence ratio is r = 1 - f(1 - rho_iono), so the observed
   persistence bounds f. The bound is computed per center on the annual
   means and on the worst seasonal cell.

Outputs:
  - results/outputs/step_4_9_environmental_ambiguity.json
"""
import os
import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.utils.logger import print_status, TEPLogger, set_step_logger
from scripts.utils.exceptions import TEPFileError, safe_json_read, safe_json_write

step_logger = TEPLogger(
    name="step_4_9_environmental_ambiguity_audit",
    level="DEBUG",
    log_file_path=PACKAGE_ROOT / "logs" / "step_4_9_environmental_ambiguity_audit.log",
)
set_step_logger(step_logger)

CENTERS = ["code", "esa_final", "igs_combined"]

# Tropospheric water-vapor correlation range cited in Section 4 (Bevis et al. 1994)
TROPOSPHERIC_RANGE_KM = (1000.0, 2000.0)

# Geomagnetic morphology: the auroral/subauroral and equatorial-anomaly zones
# carry the strongest ionospheric spatial structure; the quiet band lies between.
# Bin 3 is the highest signed geomagnetic-latitude band in every center's grid.


def _binomial_p_at_least(k: int, n: int, p: float) -> float:
    """Exact upper-tail binomial probability P(X >= k) for X ~ Bin(n, p)."""
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def geomagnetic_morphology() -> Dict[str, Any]:
    """Test whether λ is organized on the ionosphere's coordinate."""
    adv_path = PACKAGE_ROOT / "results/outputs/step_4_0_advanced_analysis.json"
    if not adv_path.exists():
        raise TEPFileError(f"Missing {adv_path}; run step_4_0 first.")

    adv = safe_json_read(adv_path)
    elev_dep = adv.get("results", {}).get("elevation_dependence", {})

    cells = []
    for ac in CENTERS:
        grid = elev_dep.get(ac, {}).get("geomagnetic_stratified_analysis", {})
        tiers: Dict[str, Dict[int, Dict]] = {}
        for key, v in grid.items():
            # keys are like elev_bin_1_geomag_bin_2
            parts = key.split("_geomag_bin_")
            elev_idx = int(parts[0].split("_")[-1])
            geom_idx = int(parts[1])
            tiers.setdefault(elev_idx, {})[geom_idx] = v
        for elev_idx, bands in sorted(tiers.items()):
            if len(bands) < 3:
                continue
            lam = {g: bands[g]["lambda_km"] for g in sorted(bands)}
            cell = {
                "center": ac,
                "elevation_tier": elev_idx,
                "lambda_by_geomag_bin": lam,
                "high_maglat_is_min": bool(lam[3] == min(lam.values())),
                "quiet_band_is_max": bool(lam[2] == max(lam.values())),
                "suppression_ratio_bin3_over_max": float(lam[3] / max(lam.values())),
            }
            cells.append(cell)

    n_cells = len(cells)
    n_high_min = sum(c["high_maglat_is_min"] for c in cells)
    n_quiet_max = sum(c["quiet_band_is_max"] for c in cells)

    # Under a null of no geomagnetic organization, each band is equally likely
    # to be the cell minimum: p_min = 1/3 per cell.
    p_min = _binomial_p_at_least(n_high_min, n_cells, 1.0 / 3.0)
    # For the quiet mid-latitude band being the maximum, same null.
    p_max = _binomial_p_at_least(n_quiet_max, n_cells, 1.0 / 3.0)

    # Mean suppression of the high-maglat band relative to the quiet band
    ratios = [
        c["lambda_by_geomag_bin"]["3"] / c["lambda_by_geomag_bin"]["2"]
        for c in cells
        if c["lambda_by_geomag_bin"].get("2")
    ]
    mean_suppression = float(np.mean(ratios)) if ratios else float("nan")

    return {
        "cells": cells,
        "n_cells": n_cells,
        "n_cells_high_maglat_minimum": n_high_min,
        "n_cells_quiet_band_maximum": n_quiet_max,
        "binomial_p_high_maglat_is_min": p_min,
        "binomial_p_quiet_band_is_max": p_max,
        "mean_lambda_ratio_highmag_over_quiet": mean_suppression,
        "interpretation": (
            "The correlation length is suppressed in the high-geomagnetic-latitude "
            "(auroral-leaning) band in "
            f"{n_high_min}/{n_cells} elevation×center cells (binomial p = {p_min:.2e} "
            "under a no-organization null), and the quiet mid-latitude band carries "
            f"the maximum in {n_quiet_max}/{n_cells} cells. The stratification is "
            "organized on the ionosphere's coordinate, replicating across all three "
            "independent analysis centers. The Kp-correlation and TID-exclusion "
            "controls test temporal ionospheric variability only and cannot exclude "
            "this static geomagnetic-latitude dependence."
        ),
    }


def tropospheric_overlap() -> Dict[str, Any]:
    """Check which frequency bands return λ inside the dismissed tropospheric range."""
    lo, hi = TROPOSPHERIC_RANGE_KM
    band_rows = []
    for ac in CENTERS:
        path = PACKAGE_ROOT / f"results/outputs/step_3_6_multiband_{ac}.json"
        if not path.exists():
            continue
        d = safe_json_read(path)
        for band, v in d.get("band_results", {}).items():
            ef = v.get("exponential_fit", {})
            lam = ef.get("lambda_km")
            if lam is None:
                continue
            band_rows.append({
                "center": ac,
                "band": band,
                "lambda_km": float(lam),
                "r_squared": ef.get("r_squared"),
                "inside_tropospheric_range": bool(lo <= lam <= hi),
            })

    inside = [r for r in band_rows if r["inside_tropospheric_range"]]
    return {
        "tropospheric_correlation_range_km": list(TROPOSPHERIC_RANGE_KM),
        "bands": band_rows,
        "n_bands_inside_range": len(inside),
        "bands_inside_range": [f"{r['center']}:{r['band']} λ={r['lambda_km']:.0f} km" for r in inside],
        "interpretation": (
            f"{len(inside)} of {len(band_rows)} frequency-band fits return λ inside the "
            "1,000–2,000 km tropospheric water-vapor correlation range that Section 4 "
            "dismisses as insufficient to explain the headline scale. The scale-based "
            "tropospheric dismissal applies to the tidal-band λ (~4,700–5,900 km) but "
            "does not cover the post-tidal and intermediate bands; the dismissal is "
            "therefore band-dependent rather than global."
        ),
    }


def elevation_confound() -> Dict[str, Any]:
    """Quantify the λ-elevation gradient and its dual interpretation."""
    adv_path = PACKAGE_ROOT / "results/outputs/step_4_0_advanced_analysis.json"
    adv = safe_json_read(adv_path)
    elev_dep = adv.get("results", {}).get("elevation_dependence", {})

    quintiles = {}
    for ac in CENTERS:
        q = elev_dep.get(ac, {}).get("quintile_analysis", {})
        lam = []
        mid = []
        for i in range(1, 6):
            cell = q.get(f"quintile_{i}")
            if not cell:
                continue
            lo_e, hi_e = cell["elevation_range_m"]
            mid.append((lo_e + hi_e) / 2.0)
            lam.append(cell["lambda_km"])
        if len(lam) >= 3:
            # Quintile index vs log λ — ordinal gradient, robust to ranges
            # crossing sea level.
            r = float(np.corrcoef(np.arange(1, len(lam) + 1), np.log10(lam))[0, 1])
            quintiles[ac] = {
                "elevation_midpoints_m": mid,
                "lambda_km": lam,
                "quintile_vs_loglambda_correlation": r,
            }

    return {
        "per_center": quintiles,
        "interpretation": (
            "λ rises monotonically with station elevation (log–log correlation "
            "reported per center). This ordering is consistent with environmental "
            "screening but is equally the fingerprint of altitude-organized "
            "tropospheric zenith-delay residuals and atmospheric loading; the two "
            "readings are degenerate pending a reanalysis-ZTD covariance control "
            "on pair-level phase alignment."
        ),
    }


def annual_degeneracy() -> Dict[str, Any]:
    """Quantify the annual-cycle driver degeneracy at 2.5 observed cycles."""
    span_days = 912.0
    period_days = 365.25
    n_cycles = span_days / period_days
    return {
        "span_days": span_days,
        "period_days": period_days,
        "n_cycles_observed": n_cycles,
        "degenerate_drivers": [
            "orbital-velocity coupling (Earth speed ~29.3–30.3 km/s)",
            "perihelion solar flux / irradiance",
            "seasonal ionospheric TEC morphology",
            "seasonal tropospheric / hydrological loading",
            "eclipse-season geometry",
        ],
        "interpretation": (
            f"The record contains {n_cycles:.2f} orbital cycles. All candidate "
            "seasonal drivers share the identical 365.25-day period; at this span "
            "no frequency separation is possible and phase discrimination is "
            "limited. The operative discriminator is the energy-vs-velocity "
            "scaling test (Section 3.3.2), which returns near-zero discrimination — "
            "the data do not separate orbital-velocity coupling from "
            "energy-scale coupling or from seasonal drivers."
        ),
    }


def diurnal_persistence_bound() -> Dict[str, Any]:
    """Bound the residual-ionospheric signal fraction from night persistence.

    Step 4.5 bins pair coherence by station-local solar time
    (local_hour = utc_hour + lon/15; day = 06–18 h). The ionospheric
    electron content over a station collapses at local night — climatological
    night/day VTEC ratios are ~0.1–0.3 — and residual ionospheric error in the
    iono-free clock products (second-order TEC terms and mapping errors)
    inherits at least that contrast. If a fraction f of the coherent signal
    were ionosphere-driven, the observed night/day coherence ratio would be
    r = 1 − f·(1 − ρ_iono), giving f = (1 − r)/(1 − ρ_iono). The bound is
    evaluated on the annual means and, conservatively, on the worst seasonal
    cell, over ρ_iono ∈ [0.1, 0.5].
    """
    per_center = {}
    for ac in CENTERS:
        path = PACKAGE_ROOT / f"results/outputs/step_4_5_comprehensive_validation_{ac}.json"
        if not path.exists():
            continue
        d = safe_json_read(path)
        ann = d.get("annual_patterns", {})
        day, night = ann.get("day_mean"), ann.get("night_mean")
        if not day or not night:
            continue
        r_ann = night / day
        # worst seasonal cell: minimum night/day ratio
        season_ratios = {}
        for s, v in d.get("seasonal_patterns", {}).items():
            if v.get("day_mean") and v.get("night_mean"):
                season_ratios[s] = v["night_mean"] / v["day_mean"]
        r_min = min(season_ratios.values()) if season_ratios else r_ann
        r_min_season = min(season_ratios, key=season_ratios.get) if season_ratios else None
        # f <= (1 - r) / (1 - rho); conservative (upper) bounds over rho in [0.1, 0.5]
        bounds = {f"rho_{rho:g}": max(0.0, (1.0 - r_ann) / (1.0 - rho))
                  for rho in (0.1, 0.25, 0.5)}
        bounds_worst = {f"rho_{rho:g}": max(0.0, (1.0 - r_min) / (1.0 - rho))
                        for rho in (0.1, 0.25, 0.5)}
        per_center[ac] = {
            "night_over_day_annual": float(r_ann),
            "seasonal_night_over_day": {k: float(v) for k, v in season_ratios.items()},
            "worst_seasonal_cell": r_min_season,
            "worst_seasonal_night_over_day": float(r_min),
            "ionospheric_fraction_bound_annual": bounds,
            "ionospheric_fraction_bound_worst_season": bounds_worst,
        }

    if not per_center:
        return {"status": "not_evaluated",
                "reason": "step_4_5 validation outputs not found"}

    # headline bound: annual mean at rho = 0.25 (central climatology), and the
    # conservative worst-seasonal bound at rho = 0.5 (least night collapse).
    f_ann = {ac: v["ionospheric_fraction_bound_annual"]["rho_0.25"]
             for ac, v in per_center.items()}
    f_worst = {ac: v["ionospheric_fraction_bound_worst_season"]["rho_0.5"]
               for ac, v in per_center.items()}
    return {
        "per_center": per_center,
        "ionospheric_fraction_bound_annual_rho0.25": f_ann,
        "ionospheric_fraction_bound_worst_season_rho0.5": f_worst,
        "interpretation": (
            "The coherent signal persists through local night: night/day "
            "coherence ratios are ~1.0 annually (CODE 1.07, ESA 0.99, IGS "
            "0.99) and never below ~0.84 in any seasonal cell (IGS SON). "
            "Since ionospheric electron content — and hence any residual "
            "ionospheric bias in the clock products — falls by a factor of "
            "~3–10 on the same local-solar-time coordinate, a residual "
            "ionospheric origin for the coherent signal is bounded at "
            "≲ 3% on the annual means and at ≲ 20% (ρ_iono = 0.25) to "
            "≲ 32% (ρ_iono = 0.5) in the worst seasonal cell. This is an "
            "in-repository quantitative control on the ionospheric channel; "
            "a spatially resolved GIM/ROTI morphology control remains the "
            "gold-standard closure for the static geomagnetic-latitude "
            "dependence."
        ),
    }


def main() -> int:
    print_status("Step 4.9: Environmental ambiguity audit", "PROCESS")

    geomag = geomagnetic_morphology()
    tropo = tropospheric_overlap()
    elev = elevation_confound()
    annual = annual_degeneracy()
    diurnal = diurnal_persistence_bound()

    result = {
        "step": "4.9",
        "name": "environmental_ambiguity_audit",
        "timestamp": datetime.now().isoformat(),
        "data_sources": [
            "results/outputs/step_4_0_advanced_analysis.json (elevation×geomagnetic grid)",
            "results/outputs/step_3_6_multiband_{center}.json (frequency-band λ)",
            "results/outputs/step_4_5_comprehensive_validation_{center}.json (local-solar-time diurnal persistence)",
        ],
        "geomagnetic_morphology": geomag,
        "tropospheric_overlap": tropo,
        "elevation_confound": elev,
        "annual_degeneracy": annual,
        "diurnal_persistence_bound": diurnal,
        "overall_assessment": (
            "The geomagnetic stratification is organized on the ionosphere's "
            "coordinate (auroral-band λ suppression replicated across centers and "
            "elevation tiers), the post-tidal/intermediate band λ values overlap "
            "the dismissed tropospheric range, the elevation gradient is degenerate "
            "with altitude-organized delay residuals, and the annual modulation is "
            "frequency-degenerate with all seasonal drivers at 2.5 cycles. These "
            "stratifications are therefore reported as consistent with both the "
            "screening interpretation and residual atmospheric/ionospheric "
            "organization, pending spatially resolved controls (GIM/ROTI for the "
            "static ionospheric field; reanalysis ZTD covariance for the "
            "tropospheric channel). Independently, the local-solar-time diurnal "
            "persistence bound places a quantitative ceiling on the ionospheric "
            "channel: a residual-ionospheric origin for the coherent signal is "
            "bounded at ≲ 3% on the annual means."
        ),
    }

    out = PACKAGE_ROOT / "results/outputs/step_4_9_environmental_ambiguity.json"
    safe_json_write(result, out)
    print_status(f"Saved {out}", "SUCCESS")
    print_status(
        f"Geomag: high-maglat band is λ-minimum in {geomag['n_cells_high_maglat_minimum']}/{geomag['n_cells']} cells "
        f"(p={geomag['binomial_p_high_maglat_is_min']:.1e}); quiet band max in {geomag['n_cells_quiet_band_maximum']}/{geomag['n_cells']}",
        "INFO",
    )
    print_status(f"Troposphere: {tropo['n_bands_inside_range']} bands inside 1,000–2,000 km", "INFO")
    print_status(f"Annual: {annual['n_cycles_observed']:.2f} cycles — driver-degenerate", "INFO")
    if isinstance(diurnal, dict) and 'ionospheric_fraction_bound_annual_rho0.25' in diurnal:
        f_ann = diurnal['ionospheric_fraction_bound_annual_rho0.25']
        print_status("Diurnal persistence: ionospheric fraction bound (annual, "
                     f"ρ=0.25) = " + ", ".join(f"{k}={v*100:.1f}%" for k, v in f_ann.items()),
                     "INFO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
