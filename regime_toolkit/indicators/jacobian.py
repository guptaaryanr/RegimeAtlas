from __future__ import annotations

from typing import Callable, Dict
import numpy as np

Params = Dict[str, float]


def jacobian_finite_difference(
    rhs: Callable[[float, np.ndarray, Params], np.ndarray],
    t: float,
    x: np.ndarray,
    params: Params,
    rel_step: float = 1e-6,
) -> np.ndarray:
    """
    Central finite-difference Jacobian.

    Why this shape:
    - Works for any RHS callable.
    - Stable enough for small dimensions; you can replace with analytic jacobians for speed/accuracy.

    Failure modes:
    - Too small rel_step => cancellation / noise.
    - Too large rel_step => truncation error.
    - For stiff systems, FD Jacobians can be noisy along fast directions; prefer analytic jacobian if available.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(rhs(t, x, params), dtype=float)
    n = x.size
    J = np.empty((n, n), dtype=float)

    for i in range(n):
        h = rel_step * (1.0 + abs(x[i]))
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        fp = np.asarray(rhs(t, xp, params), dtype=float)
        fm = np.asarray(rhs(t, xm, params), dtype=float)
        J[:, i] = (fp - fm) / (2.0 * h)

    # Basic sanity: if rhs returns wrong shape, this will likely surface as a broadcast error earlier.
    if f0.shape != (n,):
        raise ValueError(f"rhs returned shape {f0.shape}, expected ({n},)")
    return J
