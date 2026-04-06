from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import numpy as np


@dataclass(frozen=True)
class EffectiveDimensionResult:
    """
    Output container for effective dimension estimation.
    """
    dimension: float
    eigenvalues: np.ndarray
    method: str
    details: Dict[str, Any]


def _participation_ratio(evals: np.ndarray) -> float:
    evals = np.asarray(evals, dtype=float)
    evals = np.maximum(evals, 0.0)
    s1 = float(np.sum(evals))
    s2 = float(np.sum(evals**2))
    if s2 == 0.0:
        return 0.0
    return (s1 * s1) / s2


def effective_dimension_velocity_pca(
    t: np.ndarray,
    x: np.ndarray,
    stride: int = 1,
    metric: Literal["participation_ratio", "var_threshold"] = "participation_ratio",
    var_threshold: float = 0.95,
    demean: bool = True,
) -> EffectiveDimensionResult:
    """
    Estimate effective dimension using PCA on finite-difference velocities (a tangent estimate).

    Intuition:
    - If motion is mostly along a 1D slow manifold, velocity vectors cluster in ~1 direction => dim ~ 1.
    - If motion explores a plane more isotropically, velocities span ~2 directions => dim ~ 2 (for 2D systems).

    Structural assumptions probed:
    - Low effective dimension / dominant directionality in tangent space.
    - Often correlates with time-scale separation regimes in relaxation oscillators.

    Limitations / failure modes:
    - For nearly periodic limit cycles, velocities can be strongly directional yet still be 2D in state space.
    - If sampling dt is too large, finite differences smear fast jumps and bias the PCA.
    - Noise-free deterministic systems can yield very sharp covariance spectra; interpret thresholds carefully.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or t.shape[0] != x.shape[0]:
        raise ValueError("expected t shape (N,), x shape (N,d) with matching N")

    if stride < 1:
        raise ValueError("stride must be >= 1")

    # Finite difference velocities (use actual dt to handle nonuniform sampling).
    dt = t[stride:] - t[:-stride]
    if np.any(dt <= 0):
        raise ValueError("time array must be strictly increasing")
    v = (x[stride:] - x[:-stride]) / dt[:, None]

    if demean:
        v = v - np.mean(v, axis=0, keepdims=True)

    # Covariance and eigenvalues
    C = (v.T @ v) / max(1, (v.shape[0] - 1))
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(evals)[::-1]  # descending

    if metric == "participation_ratio":
        dim = _participation_ratio(evals)
        details = {"metric": metric}
        return EffectiveDimensionResult(dimension=float(dim), eigenvalues=evals, method="velocity_pca", details=details)

    if metric == "var_threshold":
        total = float(np.sum(np.maximum(evals, 0.0)))
        if total == 0.0:
            dim = 0.0
        else:
            frac = np.cumsum(np.maximum(evals, 0.0)) / total
            dim = float(np.searchsorted(frac, var_threshold) + 1)
        details = {"metric": metric, "var_threshold": float(var_threshold)}
        return EffectiveDimensionResult(dimension=float(dim), eigenvalues=evals, method="velocity_pca", details=details)

    raise ValueError(f"Unknown metric: {metric}")
