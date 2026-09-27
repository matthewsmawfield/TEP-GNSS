#!/usr/bin/env python3
"""
TEP-GNSS Co-Visibility Kernel Simulation — STEP 5.0
===================================================

Issue 1-1: the co-visibility / clock-datum systematic is the one
conventional mechanism that produces distance-structured phase-aligned
clock residuals on exactly the fitted lambda_T range, and it was never
tested.  This step implements the audit's recommendation (b): synthetic
station-clock products containing ONLY the satellite-visibility/datum
structure — no TEP field, no atmosphere — are generated on the real IGS
station network and passed through the same band-averaged coherency
estimator and exponential fit used by step_2_0.  If geometry alone
recovers lambda ~ 3,000-5,500 km, the fitted correlation length is
consistent with a datum-level artefact; if not, the co-visibility
kernel cannot supply the observed scale.

Physical model
--------------
1. Real station network: ECEF coordinates for the 768-station
   longspan network (data/coordinates/.../step_1_1_station_coords_global.csv)
   and the persisted pair-distance table (data/processed/
   step_2_1_station_distances.csv).
2. Nominal GPS constellation: Walker 24/6 (a = 26,560 km, i = 55 deg,
   6 planes, semi-synchronous ~11.97 h period) propagated over one
   sidereal day at the 30 s cadence of the clock products, in ECEF
   (Earth rotation applied).
3. Satellite clock noise: per-satellite random-walk + white phase
   noise (flicker-dominated in the 10-500 uHz TEP band).
4. Network-solution projection: each station's estimated clock absorbs
   the elevation-weighted mean of its visible satellites' errors —
   c_i(t) = sum_s w_is eps_s(t) / sum_s w_is(t) + eta_i(t) — with the
   zero-sum satellite-clock datum constraint applied per epoch
   (the datum mode that redistributes the common satellite mode into
   the station clocks).
5. Estimator: FFT bandpass to the step-2_0 TEP band (10-500 uHz)
   followed by the pairwise correlation coefficient — the exact
   band-averaged Re(coherency) is validated against
   compute_band_averaged_coherency() on a random pair subsample.
6. Exponential fit C(r) = A exp(-r/lambda) + C0 on the same log-bin
   grid as the real analysis; the pure geometric overlap kernel
   K(r) = <sum_s w_is w_js>/sqrt(<sum_s w_is^2><sum_s w_js^2>) is fit
   the same way as a shape diagnostic.

Outputs
-------
- results/outputs/step_5_0_covisibility_kernel.json

Author: TEP pipeline (issue 1-1 implementation)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COORDS_CSV = (PROJECT_ROOT / "data" / "coordinates" / "code_longspan"
              / "step_1_1_station_coords_global.csv")
DISTANCES_CSV = (PROJECT_ROOT / "data" / "processed"
                 / "step_2_1_station_distances.csv")

# --- physical constants ---
R_E = 6371.0          # km, Earth radius
MU_E = 3.986005e5     # km^3/s^2
OMEGA_E = 7.2921159e-5  # rad/s, Earth rotation
T_SIDEREAL = 86164.0905  # s
A_GPS = 26560.0       # km, semi-synchronous semi-major axis
I_GPS = np.deg2rad(55.0)

# --- estimator band (step_2_0 convention) ---
DT = 30.0             # s, CODE clock cadence
F1, F2 = 1e-5, 5e-4   # Hz, TEP band 10-500 uHz
N_EPOCH = int(round(T_SIDEREAL / DT))   # 2872 epochs

RNG = np.random.default_rng(20260926)


# ---------------------------------------------------------------------------
# Constellation + visibility
# ---------------------------------------------------------------------------

def gps_constellation_ecef(t_s):
    """Nominal Walker 24/6/1 GPS constellation in ECEF.

    Returns (n_epochs, n_sat, 3) ECEF positions in km.
    """
    n_sat = 24
    n_planes = 6
    per_plane = 4
    n_orb = np.sqrt(MU_E / A_GPS**3)
    pos = np.zeros((len(t_s), n_sat, 3))
    k = 0
    for j in range(n_planes):
        raan = j * 2 * np.pi / n_planes
        for m in range(per_plane):
            # evenly spaced in argument of latitude with Walker phasing
            u0 = m * 2 * np.pi / per_plane + j * np.pi / 12.0
            u = u0 + n_orb * t_s
            x_o = A_GPS * np.cos(u)
            y_o = A_GPS * np.sin(u)
            # orbital plane -> ECI
            xi = x_o * np.cos(raan) - y_o * np.cos(I_GPS) * np.sin(raan)
            yi = x_o * np.sin(raan) + y_o * np.cos(I_GPS) * np.cos(raan)
            zi = y_o * np.sin(I_GPS)
            pos[:, k, :] = np.stack([xi, yi, zi], axis=1)
            k += 1
    # rotate ECI -> ECEF by Earth rotation angle
    th = OMEGA_E * t_s
    ct, st = np.cos(th), np.sin(th)
    x = pos[..., 0] * ct[:, None] + pos[..., 1] * st[:, None]
    y = -pos[..., 0] * st[:, None] + pos[..., 1] * ct[:, None]
    return np.stack([x, y, pos[..., 2]], axis=2)


def elevation_weights(sta_xyz, sat_xyz, mask_deg, weighting):
    """Per-epoch visibility weight matrix (n_epoch, n_sta, n_sat).

    w = sin(el) or 1 for elevation above mask, else 0.
    """
    # line of sight
    los = sat_xyz[:, None, :, :] - sta_xyz[None, :, None, :]
    up = sta_xyz / np.linalg.norm(sta_xyz, axis=1, keepdims=True)
    rng = np.linalg.norm(los, axis=-1)
    sin_el = np.einsum("tnmk,nk->tnm", los, up) / rng
    vis = sin_el > np.sin(np.deg2rad(mask_deg))
    if weighting == "sine":
        w = np.where(vis, np.clip(sin_el, 0.0, 1.0), 0.0)
    elif weighting == "sine2":
        # MGEX step_3_2 convention: sin^2(el)/sin^2(30 deg), capped at 1
        w = np.where(
            vis,
            np.clip(sin_el / np.sin(np.deg2rad(30.0)), 0.0, 1.0) ** 2,
            0.0)
    else:
        w = vis.astype(float)
    return w


# ---------------------------------------------------------------------------
# Synthetic clock products
# ---------------------------------------------------------------------------

def satellite_clock_noise(n_sat, n_ep, rng):
    """Random-walk + white phase noise, normalized to unit variance."""
    rw = np.cumsum(rng.standard_normal((n_sat, n_ep)), axis=1)
    rw -= rw[:, :1]
    white = rng.standard_normal((n_sat, n_ep)) * 0.15
    eps = rw + white
    eps -= eps.mean(axis=1, keepdims=True)
    eps /= eps.std(axis=1, keepdims=True)
    return eps


def synthesize_station_clocks(w, eps, rng, datum_zero_mean=True):
    """Station clock = elevation-weighted mean of visible satellite errors.

    With datum_zero_mean, the satellite errors are re-centred per epoch
    on the visibility-weighted network mean — the zero-sum datum
    constraint the network clock solution applies — so the common mode
    is carried by the stations.
    """
    T, N, S = w.shape
    if datum_zero_mean:
        # visibility-weighted network mean per epoch — the zero-sum
        # datum constraint the network clock solution applies
        W = w.sum(axis=1)                               # (T, S)
        m = np.einsum("ts,st->t", W, eps) / np.maximum(
            W.sum(axis=1), 1e-9)                        # (T,)
        eps = eps - m[None, :]
    num = np.einsum("tns,st->tn", w, eps)
    den = np.maximum(w.sum(axis=2), 1e-9)
    c = num / den
    c += rng.standard_normal((T, N)) * 0.5            # station-local noise
    return c


def bandpass(x, dt, f1, f2):
    """FFT brick-wall bandpass along axis 0."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], d=dt)
    X[~((f >= f1) & (f <= f2))] = 0.0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def band_corr_matrix(x_filt):
    """Pairwise correlation of band-limited series (band-coherency proxy)."""
    x = x_filt - x_filt.mean(axis=0)
    den = np.sqrt((x**2).sum(axis=0))
    x[:, den < 1e-12] = 0.0
    x /= np.maximum(den, 1e-30)
    with np.errstate(all="ignore"):      # spurious Accelerate/BLAS warnings on macOS
        return x.T @ x


def band_coherency_exact(x, y, fs):
    """Reference: step_2_0's band-averaged real coherency (Welch)."""
    nseg = min(256, len(x) // 4)
    f, Pxy = signal.csd(x, y, fs=fs, nperseg=nseg)
    _, Pxx = signal.welch(x, fs=fs, nperseg=nseg)
    _, Pyy = signal.welch(y, fs=fs, nperseg=nseg)
    den = np.sqrt(Pxx * Pyy)
    mask = (den > 1e-10)
    coh = np.zeros_like(Pxy, dtype=complex)
    coh[mask] = Pxy[mask] / den[mask]
    band = mask & (f >= F1) & (f <= F2)
    if not np.any(band):
        return np.nan
    return float(np.mean(np.real(coh[band])))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_exp(r, c, wgt):
    try:
        p0 = [max(c) - min(c), 4000.0, min(c)]
        p, _ = curve_fit(lambda d, A, l, C: A * np.exp(-d / l) + C,
                         r, c, p0=p0, sigma=1.0 / np.sqrt(np.maximum(wgt, 1)),
                         bounds=([0, 100, -1], [2, 50000, 1]), maxfev=20000)
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


# ---------------------------------------------------------------------------

def main():
    print("[STEP 5.0] Co-visibility kernel simulation (issue 1-1)")

    sta = pd.read_csv(COORDS_CSV)
    sta_xyz = sta[["X", "Y", "Z"]].to_numpy() / 1000.0   # -> km
    names = sta["code"].tolist()
    idx_of = {n: i for i, n in enumerate(names)}
    print(f"  {len(names)} stations")

    pairs = pd.read_csv(DISTANCES_CSV)
    pairs = pairs[pairs.station1.isin(idx_of) & pairs.station2.isin(idx_of)]
    ii = pairs.station1.map(idx_of).to_numpy()
    jj = pairs.station2.map(idx_of).to_numpy()
    dist = pairs.dist_km.to_numpy()
    print(f"  {len(dist)} pairs")

    t = np.arange(N_EPOCH) * DT
    sat = gps_constellation_ecef(t)
    print(f"  constellation: {sat.shape[1]} sats x {len(t)} epochs")

    edges = np.logspace(np.log10(50), np.log10(13000), 29)

    results = {}
    # --- pure geometric overlap kernel (no noise): the datum structure itself
    for mask_deg in (5.0, 10.0, 15.0):
        for weighting in ("sine", "sine2", "uniform"):
            w = elevation_weights(sta_xyz, sat, mask_deg, weighting)
            num = np.einsum("tns,tms->nm", w, w) / w.shape[0]   # <sum_s w_is w_js>
            n1 = (w**2).sum(axis=(0, 2)) / w.shape[0]           # <sum_s w_is^2>
            K = num / np.maximum(
                np.sqrt(np.outer(n1, n1)), 1e-9)                # (N, N)
            kv = K[ii, jj]
            rb, cb, nb = bin_pairs(dist, kv, edges)
            fit = fit_exp(rb, cb, nb)
            results[f"kernel_mask{int(mask_deg)}_{weighting}"] = {
                "fit": fit,
                "corr_with_exp4200": float(np.corrcoef(
                    cb, np.exp(-rb / 4200.0))[0, 1]),
            }
            print(f"  kernel mask={mask_deg} w={weighting}: "
                  f"lam={fit['lam']:.0f} km, r(K,exp4200)="
                  f"{results[f'kernel_mask{int(mask_deg)}_{weighting}']['corr_with_exp4200']:.3f}")

    # --- full synthetic-clock simulation ---
    configs = []
    for mask_deg in (10.0, 5.0):
        for datum in (True, False):
            configs.append((mask_deg, datum))
    for mask_deg, datum in configs:
        w = elevation_weights(sta_xyz, sat, mask_deg, "sine")
        lams, cors = [], []
        for rep in range(4):
            eps = satellite_clock_noise(sat.shape[1], len(t), RNG)
            c = synthesize_station_clocks(w, eps, RNG, datum)
            cb_full = band_corr_matrix(bandpass(c, DT, F1, F2))
            kv = cb_full[ii, jj]
            rb, cb_, nb = bin_pairs(dist, kv, edges)
            fit = fit_exp(rb, cb_, nb)
            lams.append(fit["lam"])
            cors.append(float(np.corrcoef(
                cb_, np.exp(-rb / 4200.0))[0, 1]))
        # validate proxy vs exact coherency on a subsample
        eps = satellite_clock_noise(sat.shape[1], len(t), RNG)
        c = synthesize_station_clocks(w, eps, RNG, datum)
        cb_fast = band_corr_matrix(bandpass(c, DT, F1, F2))
        sub = RNG.choice(len(dist), 400, replace=False)
        ex = np.array([band_coherency_exact(
            c[:, ii[s]], c[:, jj[s]], 1.0 / DT) for s in sub])
        fa = np.array([cb_fast[ii[s], jj[s]] for s in sub])
        proxy_r = float(np.corrcoef(np.nan_to_num(ex), fa)[0, 1])
        results[f"sim_mask{int(mask_deg)}_datum{int(datum)}"] = {
            "lambda_km": lams,
            "lambda_median": float(np.nanmedian(lams)),
            "corr_with_exp4200": cors,
            "proxy_vs_exact_r": proxy_r,
        }
        print(f"  sim mask={mask_deg} datum={datum}: "
              f"lam={np.nanmedian(lams):.0f} km (n={len(lams)}), "
              f"r(shape,exp4200)={np.nanmedian(cors):.3f}, "
              f"proxy-vs-exact r={proxy_r:.3f}")

    out = {
        "step": "step_5_0_covisibility_kernel",
        "issue": "1-1",
        "station_count": len(names),
        "pair_count": len(dist),
        "constellation": "GPS Walker 24/6, a=26560 km, i=55 deg",
        "epochs": int(N_EPOCH), "cadence_s": DT,
        "band_hz": [F1, F2],
        "configs": results,
    }
    out_path = RESULTS_DIR / "step_5_0_covisibility_kernel.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
