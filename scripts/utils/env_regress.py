"""Environmental covariate regression utilities.

This module provides a lightweight, configurable way to regress out known
environmental drivers (e.g., geomagnetic Kp index, solar F10.7 flux,
surface temperature, humidity, tropospheric delay) from the raw pairwise
GPS coherence before hypothesis-testing.  The goal is to make Step 2·2
(long-span) consistent with the stricter environmental controls used in
Paper 1.

Data sources are expected to be pre-downloaded into
`data/external/env_covariates/` as CSV files with a `date` column in
ISO-format.  If a file is missing the function logs a warning and skips
that covariate.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENV_DATA_DIR = PACKAGE_ROOT / "data/external/env_covariates"

# Default covariate -> filename mapping
_COVAR_FILES = {
    "kp": "kp_index_daily.csv",          # Columns: date, kp
    "f10_7": "f10_7_daily.csv",         # Columns: date, f10_7
    "temp": "surface_temp_daily.csv",    # Columns: date, temp_c
    "humidity": "surface_humidity_daily.csv",  # Columns: date, rh
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_env_data(covariates: List[str]) -> pd.DataFrame:
    """Load requested environmental covariates as a single daily dataframe."""
    frames: List[pd.DataFrame] = []
    for cov in covariates:
        fname = _COVAR_FILES.get(cov)
        if fname is None:
            warnings.warn(f"Unknown covariate '{cov}' – skipping.")
            continue
        fpath = ENV_DATA_DIR / fname
        if not fpath.exists():
            warnings.warn(f"Environmental covariate file missing: {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=["date"])
        frames.append(df.set_index("date"))
    if not frames:
        return pd.DataFrame()
    env_df = pd.concat(frames, axis=1).sort_index()
    return env_df.reset_index()


def apply_env_regression(pair_df: pd.DataFrame,
                         covariates: List[str] | None = None) -> pd.DataFrame:
    """Regress environmental covariates out of `coherence`.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Pair-level dataframe containing at least `date` and `coherence`.
    covariates : list[str] | None
        Covariate names to include.  If None, uses default list from
        TEPConfig (`TEP_ENV_COVARS`) or falls back to ["kp", "f10_7"].

    Returns
    -------
    pd.DataFrame
        Same as `pair_df` but with new column `coherence_resid`.  The
        original `coherence` is retained.
    """
    try:
        from scripts.utils.config import TEPConfig
        if covariates is None:
            covariates = TEPConfig.get_list("TEP_ENV_COVARS", ["kp", "f10_7"])
    except Exception:
        if covariates is None:
            covariates = ["kp", "f10_7"]

    if "date" not in pair_df.columns or "coherence" not in pair_df.columns:
        warnings.warn("apply_env_regression: required columns missing – skipping regression")
        pair_df["coherence_resid"] = pair_df.get("coherence", np.nan)
        return pair_df

    env_df = load_env_data(covariates)
    if env_df.empty:
        warnings.warn("No environmental data found – skipping regression")
        pair_df["coherence_resid"] = pair_df["coherence"]
        return pair_df

    # Merge on date (inner join keeps only dates with env data)
    merged = pair_df.merge(env_df, on="date", how="left")

    # Statsmodels OLS with intercept; drop rows with missing covariates
    reg_cols = [c for c in covariates if c in merged.columns]
    mask = merged[reg_cols].notna().all(axis=1)
    if mask.sum() < 100:
        warnings.warn("apply_env_regression: insufficient overlap with env data – using original coherence")
        pair_df["coherence_resid"] = pair_df["coherence"]
        return pair_df

    X = sm.add_constant(merged.loc[mask, reg_cols])
    y = merged.loc[mask, "coherence"]
    try:
        model = sm.OLS(y, X).fit()
        resid = y - model.predict(X)
        merged.loc[mask, "coherence_resid"] = resid
        merged.loc[~mask, "coherence_resid"] = np.nan
    except Exception as e:  # fallback to original coherence
        warnings.warn(f"Env regression failed: {e}")
        merged["coherence_resid"] = merged["coherence"]

    # Forward-fill any missing residuals with original coherence
    merged["coherence_resid"].fillna(merged["coherence"], inplace=True)

    return merged
