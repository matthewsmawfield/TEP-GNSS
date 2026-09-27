#!/usr/bin/env python3
"""
TEP-GNSS Product-Conditioned Co-Visibility Kernel — STEP 5.2
==========================================================

Residual of issue 1-1 (corpus register `gnss-covisibility-kernel`): step 5.0
bounded the co-visibility/datum channel with a nominal Walker 24/6
constellation and synthetic satellite-clock noise.  The stated open item was
conditioning on product-specific error kernels — real ephemerides and the
actual satellite error content of the analysis-centre product.  This step
repeats the step-5.0 forward model with both substitutions made:

1. Real ephemerides: CODE MGEX precise orbits (SP3, IGS20 ECEF, 300 s
   cadence) interpolated to the 30 s clock grid by per-coordinate cubic
   spline — the same product family whose elevation-weighted kernel was
   computed in Paper 14 (step_3_2) and reproduced by step_5_0's nominal
   constellation.
2. Real satellite-clock content: the AS records of the same day's CODE MGEX
   clock product (30 s cadence) used directly as eps_s(t).  These series
   carry the realised orbit, ambiguity and datum structure of the centre's
   own solution, so the projection c_i(t) = sum_s w_is eps_s / sum_s w_is
   conditions on the product's actual error content rather than a noise
   model.  Because the estimated satellite clocks include the true
   stochastic clock signal as well as estimation error, this conditioning
   overstates rather than understates the artifact channel.

The station network, visibility weighting, zero-sum datum option, 10-500 uHz
bandpass estimator, log-bin grid and exponential fit are identical to
step_5_0, so the fitted decay scales are directly comparable.

Inputs
------
- data/raw/cod_mgex/<YYYYDDD>_COD.sp3    CODE MGEX precise orbits
- data/raw/cod_mgex/<YYYYDDD>_COD.clk.gz CODE MGEX clocks (AS, 30 s)
  (public CODE products, DOI 10.48350/197028; staged copies of files
  analysed in TEP-GNSS-MGEX step_3_2)
- data/coordinates/code_longspan/step_1_1_station_coords_global.csv
- data/processed/step_2_1_station_distances.csv

Outputs
-------
- results/outputs/step_5_2_product_conditioned_kernel.json

Author: TEP pipeline (issue 1-1 residual)
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COORDS_CSV = (PROJECT_ROOT / "data" / "coordinates" / "code_longspan"
              / "step_1_1_station_coords_global.csv")
DISTANCES_CSV = (PROJECT_ROOT / "data" / "processed"
                 / "step_2_1_station_distances.csv")
PRODUCT_DIR = PROJECT_ROOT / "data" / "raw" / "cod_mgex"

DAYS = ["2025001", "2025056", "2026102", "2026108"]

# --- estimator band (step_2_0 convention, identical to step_5_0) ---
DT = 30.0
F1, F2 = 1e-5, 5e-4
N_EPOCH = 2880                    # one solar day at 30 s
STATION_NOISE_REL = 0.5           # step_5_0 ratio: local noise / signal std

MEASURED_LAMBDAS = {"code": 4548.6, "igs_combined": 3763.6,
                    "esa_final": 3329.8}

RNG = np.random.default_rng(20260927)


# ---------------------------------------------------------------------------
# Product parsers
# ---------------------------------------------------------------------------

def load_sp3(path):
    """Parse SP3: returns (t_sod (n_ep,), {sat: (n_ep,3) km, NaN-filled})."""
    epochs = []
    recs = {}                       # sat -> {ep_i: [x,y,z]}
    with open(path) as f:
        ep_i = -1
        for ln in f:
            if ln.startswith("*"):
                tok = ln.split()
                epochs.append(int(tok[4]) * 3600 + int(tok[5]) * 60
                              + float(tok[6]))
                ep_i += 1
            elif ln.startswith("P"):
                tok = ln.split()
                if len(tok) >= 4:
                    recs.setdefault(tok[0][1:], {})[ep_i] = [
                        float(tok[1]), float(tok[2]), float(tok[3])]
    epochs = np.array(epochs)
    wrap = np.diff(epochs) < 0                 # midnight rollover (24:00 epoch)
    epochs = epochs + np.concatenate(([0], np.cumsum(wrap))) * 86400.0
    n_ep = len(epochs)
    pos = {}
    for sat, d in recs.items():
        p = np.full((n_ep, 3), np.nan)
        for i, xyz in d.items():
            p[i] = xyz
        pos[sat] = p
    return np.array(epochs), pos


def load_clk_sat(path, sats):
    """Parse AS clock records; returns {sat: (n,2) [sod, bias_s]}."""
    want = set(sats)
    out = {}
    in_data = False
    with gzip.open(path, "rt", errors="replace") as f:
        for ln in f:
            if not in_data:
                if "END OF HEADER" in ln:
                    in_data = True
                continue
            if not ln.startswith("AS"):
                continue
            tok = ln.split()
            if len(tok) < 10 or tok[1] not in want:
                continue
            # RINEX clock: AS sat Y M D h m s nvals bias [sigma]
            tod = (int(tok[5]) * 3600 + int(tok[6]) * 60
                   + float(tok[7]))
            out.setdefault(tok[1], []).append((tod, float(tok[9])))
    return {s: np.asarray(v) for s, v in out.items()}


# ---------------------------------------------------------------------------
# Geometry (chunked over epochs; identical weighting conventions to step_5_0)
# ---------------------------------------------------------------------------

def interp_orbits(t_sp3, pos_by_sat, t_grid, prefix=None, min_frac=0.9):
    """Spline each satellite to t_grid; returns (sat_list, (T,S,3)).

    Satellites covering < min_frac of the SP3 epochs are dropped; spline
    values outside a satellite's own coverage range are set to NaN so the
    visibility test treats them as absent rather than extrapolated.
    """
    min_ep = int(min_frac * len(t_sp3))
    keep = [s for s, p in pos_by_sat.items()
            if np.isfinite(p).all(1).sum() >= min_ep
            and (prefix is None or s.startswith(prefix))]
    keep.sort()
    sat30 = np.full((len(t_grid), len(keep), 3), np.nan, dtype=np.float32)
    for k, s in enumerate(keep):
        p = pos_by_sat[s]
        ok = np.isfinite(p).all(1)
        t_ok = t_sp3[ok]
        for c in range(3):
            sat30[:, k, c] = CubicSpline(t_ok, p[ok, c])(t_grid)
        out = (t_grid < t_ok[0]) | (t_grid > t_ok[-1])
        sat30[out, k, :] = np.nan
    return keep, sat30


def weights_chunked(sta_xyz, sat_xyz, mask_deg, weighting):
    """Yield (Tb,N,S) float32 weight blocks; block sized for ~0.6 GB."""
    S = sat_xyz.shape[1]
    block = max(1, int(240 * 32 / max(S, 1)))
    up = sta_xyz / np.linalg.norm(sta_xyz, axis=1, keepdims=True)
    sin_mask = np.sin(np.deg2rad(mask_deg))
    T = sat_xyz.shape[0]
    for a in range(0, T, block):
        sb = sat_xyz[a:a + block]                                # (Tb,S,3)
        los = sb[:, None, :, :] - sta_xyz[None, :, None, :]      # (Tb,N,S,3)
        rng = np.linalg.norm(los, axis=-1)
        rng[rng == 0] = np.nan
        with np.errstate(all="ignore"):
            sin_el = np.einsum("tnmk,nk->tnm", los, up) / rng
        vis = np.isfinite(sin_el) & (sin_el > sin_mask)
        if weighting == "sine":
            w = np.where(vis, np.clip(sin_el, 0.0, 1.0), 0.0)
        elif weighting == "sine2":
            w = np.where(vis, np.clip(
                sin_el / np.sin(np.deg2rad(30.0)), 0.0, 1.0) ** 2, 0.0)
        else:
            w = vis.astype(np.float32)
        yield np.nan_to_num(w.astype(np.float32))


def kernel_from_weights(sta_xyz, sat_xyz, mask_deg, weighting):
    """K(i,j) = <sum_s w_is w_js>/sqrt(<sum_s w_i^2><sum_s w_j^2>) (N,N)."""
    num = None
    n1 = None
    for w in weights_chunked(sta_xyz, sat_xyz, mask_deg, weighting):
        blk = np.einsum("tns,tms->nm", w.astype(np.float64),
                        w.astype(np.float64))
        num = blk if num is None else num + blk
        s1 = (w.astype(np.float64) ** 2).sum(axis=(0, 2))
        n1 = s1 if n1 is None else n1 + s1
    return num / np.maximum(np.sqrt(np.outer(n1, n1)), 1e-9)


def project_clocks(sta_xyz, sat_xyz, eps, mask_deg, weighting, rng,
                   datum_zero_mean=True):
    """c_i(t) = sum_s w_is eps_s / sum_s w_is (+ optional datum re-centre).

    Station-local white noise is added at STATION_NOISE_REL x the median
    satellite series std — the step_5_0 noise ratio carried over to real
    amplitudes.
    """
    T, _, S = sat_xyz.shape
    N = sta_xyz.shape[0]
    c = np.empty((T, N), dtype=np.float64)
    eps64 = np.asarray(eps, dtype=np.float64)
    a = 0
    for w in weights_chunked(sta_xyz, sat_xyz, mask_deg, weighting):
        Tb = w.shape[0]
        e = eps64[a:a + Tb]                                      # (Tb,S)
        if datum_zero_mean:
            W = w.sum(axis=1)                                    # (Tb,S)
            m = np.einsum("ts,ts->t", W, e) / np.maximum(
                W.sum(axis=1), 1e-30)
            e = e - m[:, None]
        num = np.einsum("tns,ts->tn", w, e)
        den = np.maximum(w.sum(axis=2), 1e-30)
        c[a:a + Tb] = num / den
        a += Tb
    sig = np.nanmedian(np.nanstd(eps64, axis=0))
    c += rng.standard_normal((T, N)) * STATION_NOISE_REL * sig
    return c


# ---------------------------------------------------------------------------
# Estimator — identical chain to step_5_0
# ---------------------------------------------------------------------------

def bandpass(x, dt, f1, f2):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], d=dt)
    X[~((f >= f1) & (f <= f2))] = 0.0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def band_corr_matrix(x_filt):
    x = x_filt - x_filt.mean(axis=0)
    den = np.sqrt((x ** 2).sum(axis=0))
    x[:, den <= 0.0] = 0.0
    x = x / np.maximum(den, 1e-300)
    with np.errstate(all="ignore"):
        return x.T @ x


def fit_exp(r, c, wgt):
    try:
        p0 = [max(c) - min(c), 4000.0, min(c)]
        p, _ = curve_fit(lambda d, A, l, C: A * np.exp(-d / l) + C,
                         r, c, p0=p0,
                         sigma=1.0 / np.sqrt(np.maximum(wgt, 1)),
                         bounds=([0, 100, -1], [2, 50000, 1]),
                         maxfev=20000)
        return dict(A=float(p[0]), lam=float(p[1]), C0=float(p[2]))
    except Exception as e:
        return dict(A=np.nan, lam=np.nan, C0=np.nan, err=str(e))


def bin_pairs(dist, val, edges):
    idx = np.digitize(dist, edges) - 1
    rb, cb, nb = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 50:
            rb.append(float(np.mean(dist[m])))
            cb.append(float(np.mean(val[m])))
            nb.append(int(m.sum()))
    return np.array(rb), np.array(cb), np.array(nb)


def band_table(dist, val, edges_m=(0, 1000, 2000, 3000, 5000, 8000,
                                 12000, 20000)):
    return {f"{int(lo)}-{int(hi)}": float(np.nanmean(
        val[(dist >= lo) & (dist < hi)]))
        for lo, hi in zip(edges_m[:-1], edges_m[1:])
        if ((dist >= lo) & (dist < hi)).sum()}


def build_eps(clk_path, sat_list, t_grid, mode):
    """(T,S) real satellite series on the 30 s grid.

    mode 'raw': per-satellite mean removed only — keeps the realised drift
    and stochastic structure of the estimated clocks.
    mode 'detrended': per-satellite linear detrend, absorbing the bulk
    frequency offset/drift carried by each clock's deterministic model.
    """
    ser = load_clk_sat(clk_path, sat_list)
    eps = np.full((len(t_grid), len(sat_list)), np.nan)
    good = []
    for k, s in enumerate(sat_list):
        v = ser.get(s)
        if v is None or len(v) < 0.5 * len(t_grid):
            continue
        eps[:, k] = np.interp(t_grid, v[:, 0], v[:, 1])
        good.append(k)
    eps = eps[:, good]
    used = [sat_list[k] for k in good]
    if mode == "detrended":
        tt = (t_grid - t_grid.mean()) / np.ptp(t_grid)
        for k in range(eps.shape[1]):
            a_, b_ = np.polyfit(tt, eps[:, k], 1)
            eps[:, k] -= a_ * tt + b_
    eps -= np.nanmean(eps, axis=0, keepdims=True)
    return eps, used


# ---------------------------------------------------------------------------

def main():
    print("[STEP 5.2] Product-conditioned co-visibility kernel")

    sta = pd.read_csv(COORDS_CSV)
    sta_xyz = sta[["X", "Y", "Z"]].to_numpy() / 1000.0   # km
    names = sta["code"].tolist()
    idx_of = {n: i for i, n in enumerate(names)}
    print(f"  {len(names)} stations")

    pairs = pd.read_csv(DISTANCES_CSV)
    pairs = pairs[pairs.station1.isin(idx_of) & pairs.station2.isin(idx_of)]
    ii = pairs.station1.map(idx_of).to_numpy()
    jj = pairs.station2.map(idx_of).to_numpy()
    dist = pairs.dist_km.to_numpy()
    print(f"  {len(dist)} pairs")

    t_grid = np.arange(N_EPOCH) * DT
    edges = np.logspace(np.log10(50), np.log10(13000), 29)

    out = {"step": "step_5_2_product_conditioned_kernel",
           "issue": "1-1 residual",
           "products": "CODE MGEX SP3 + 30 s AS clocks",
           "station_count": len(names), "pair_count": len(dist),
           "band_hz": [F1, F2], "cadence_s": DT,
           "measured_lambdas_km": MEASURED_LAMBDAS,
           "days": {}}

    for day in DAYS:
        sp3_path = PRODUCT_DIR / f"{day}_COD.sp3"
        clk_path = PRODUCT_DIR / f"{day}_COD.clk.gz"
        t_sp3, pos_by_sat = load_sp3(sp3_path)
        day_res = {"sp3": sp3_path.name, "clk": clk_path.name,
                   "satsets": {}}
        for tag, prefix in (("gps", "G"), ("all", None)):
            sats, sat30 = interp_orbits(t_sp3, pos_by_sat, t_grid, prefix)
            n_ok = int(np.isfinite(sat30).all(axis=(0, 2)).sum())
            print(f"  {day} {tag}: {n_ok} sats")
            set_res = {"n_sats": n_ok, "sats": sats, "kernels": {},
                       "sims": {}}
            kcfgs = ([(m, w) for m in (5.0, 10.0, 15.0)
                      for w in ("sine", "sine2", "uniform")]
                     if tag == "gps" else [(10.0, "sine2")])
            for mask_deg, weighting in kcfgs:
                K = kernel_from_weights(sta_xyz, sat30, mask_deg, weighting)
                kv = K[ii, jj]
                rb, cb, nb = bin_pairs(dist, kv, edges)
                fit = fit_exp(rb, cb, nb)
                key = f"mask{int(mask_deg)}_{weighting}"
                set_res["kernels"][key] = {
                    "fit": fit,
                    "bands_km": band_table(dist, kv),
                    "corr_with_exp4200": float(np.corrcoef(
                        cb, np.exp(-rb / 4200.0))[0, 1])}
                print(f"    kernel {key}: lam={fit['lam']:.0f} km")
            scfgs = ([(m, d, e) for m in (10.0, 5.0)
                      for d in (True, False)
                      for e in ("raw", "detrended")]
                     if tag == "gps"
                     else [(10.0, True, "raw"), (10.0, False, "raw")])
            eps_cache = {}
            for mask_deg, datum, emode in scfgs:
                if emode not in eps_cache:
                    eps_cache[emode] = build_eps(
                        clk_path, sats, t_grid, emode)
                eps, used = eps_cache[emode]
                cidx = [sats.index(s) for s in used]
                c = project_clocks(sta_xyz, sat30[:, cidx, :], eps,
                                   mask_deg, "sine", RNG, datum)
                cb_full = band_corr_matrix(bandpass(c, DT, F1, F2))
                kv = cb_full[ii, jj]
                rb, cb_, nb = bin_pairs(dist, kv, edges)
                fit = fit_exp(rb, cb_, nb)
                key = f"mask{int(mask_deg)}_datum{int(datum)}_{emode}"
                set_res["sims"][key] = {
                    "fit": fit,
                    "bands_km": band_table(dist, kv),
                    "corr_with_exp4200": float(np.corrcoef(
                        cb_, np.exp(-rb / 4200.0))[0, 1]),
                    "n_sats_with_clocks": len(used)}
                print(f"    sim {key}: lam={fit['lam']:.0f} km")
            day_res["satsets"][tag] = set_res
        out["days"][day] = day_res

    out_path = RESULTS_DIR / "step_5_2_product_conditioned_kernel.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
