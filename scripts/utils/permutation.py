"""Permutation-test utilities for Step 2·2 long-span analyses."""
from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy import stats


def permuted_pearson(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    """Return (r_obs, empirical_p) for correlation via permutation.

    Parameters
    ----------
    x, y : 1-D numeric arrays of equal length.
    n_perm : int
        Number of label permutations.
    rng : np.random.Generator | None
        Random generator (default `np.random.default_rng(42)`).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    x = np.asarray(x)
    y = np.asarray(y)
    r_obs, _ = stats.pearsonr(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r_perm, _ = stats.pearsonr(x, y_perm)
        if abs(r_perm) >= abs(r_obs) - 1e-12:
            count += 1
    p_emp = (count + 1) / (n_perm + 1)
    return float(r_obs), float(p_emp)
