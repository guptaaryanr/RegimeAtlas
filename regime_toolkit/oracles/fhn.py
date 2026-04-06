from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from scipy.signal import find_peaks

from ..systems.base import Params


@dataclass(frozen=True)
class FHNEquilibriaResult:
    states: np.ndarray
    v_roots: np.ndarray
    details: Dict[str, Any]


@dataclass(frozen=True)
class FHNStabilityResult:
    state: np.ndarray
    eigenvalues: np.ndarray
    trace: float
    determinant: float
    stable: bool
    details: Dict[str, Any]


@dataclass(frozen=True)
class TailOscillationResult:
    amplitude: float
    mean_value: float
    std_value: float
    period: Optional[float]
    n_peaks: int
    regime_label: str
    details: Dict[str, Any]



def fhn_equilibria(params: Params, *, tol: float = 1e-9) -> FHNEquilibriaResult:
    """
    Compute real equilibria of the FitzHugh–Nagumo system.

    The equilibrium equation is independent of epsilon, so these states can be reused
    across epsilon sweeps with fixed a, b, I.
    """
    a = float(params["a"])
    b = float(params["b"])
    I = float(params["I"])

    coeffs = np.array([b, 0.0, -3.0 * (b - 1.0), 3.0 * a - 3.0 * b * I], dtype=float)
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(np.imag(roots)) <= tol].real
    if real_roots.size == 0:
        raise RuntimeError("no real equilibria found for FitzHugh–Nagumo parameters")

    # Merge numerically duplicated real roots.
    unique_roots = []
    for val in np.sort(real_roots):
        if not unique_roots or abs(val - unique_roots[-1]) > 10.0 * tol:
            unique_roots.append(float(val))
    v = np.asarray(unique_roots, dtype=float)
    w = (v + a) / b
    states = np.column_stack([v, w])
    return FHNEquilibriaResult(
        states=states,
        v_roots=v,
        details={
            "a": a,
            "b": b,
            "I": I,
            "n_equilibria": int(states.shape[0]),
        },
    )



def fhn_linear_stability(
    state: np.ndarray,
    params: Params,
    *,
    epsilon: float,
) -> FHNStabilityResult:
    """
    Linear stability of the FHN equilibrium at a given epsilon.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (2,):
        raise ValueError("state must have shape (2,)")
    v, w = state
    b = float(params["b"])
    eps = float(epsilon)

    J = np.array(
        [
            [(1.0 - v**2) / eps, -1.0 / eps],
            [1.0, -b],
        ],
        dtype=float,
    )
    eigvals = np.linalg.eigvals(J)
    trace = float(np.trace(J))
    determinant = float(np.linalg.det(J))
    stable = bool(np.all(np.real(eigvals) < 0.0))
    return FHNStabilityResult(
        state=state.copy(),
        eigenvalues=np.asarray(eigvals, dtype=complex),
        trace=trace,
        determinant=determinant,
        stable=stable,
        details={"epsilon": eps},
    )



def predict_fhn_hopf_epsilon(params: Params) -> Optional[float]:
    """
    Predict the first positive epsilon at which an equilibrium satisfies trace = 0 and det > 0.

    For the common single-equilibrium regime this returns a single scalar. If multiple
    equilibria admit Hopf candidates, the smallest positive epsilon is returned.
    """
    eq = fhn_equilibria(params)
    b = float(params["b"])
    candidates = []
    for state in eq.states:
        v = float(state[0])
        eps_h = (1.0 - v**2) / b
        if eps_h <= 0.0:
            continue
        stability = fhn_linear_stability(state, params, epsilon=eps_h)
        if stability.determinant > 0.0:
            candidates.append(float(eps_h))
    if not candidates:
        return None
    return float(min(candidates))



def tail_oscillation_metrics(
    t: np.ndarray,
    x: np.ndarray,
    *,
    state_index: int = 0,
    tail_fraction: float = 0.25,
    amplitude_threshold: float = 0.2,
) -> TailOscillationResult:
    """
    Simple oracle-friendly oscillation summary from the tail of a trajectory.

    Why this is enough for now:
    - The current paper boundary is oscillatory motion versus convergence to a fixed point.
    - Tail amplitude is the cleanest scalar for that distinction in FHN.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or t.shape[0] != x.shape[0]:
        raise ValueError("expected matching t and x arrays")
    if not (0 <= state_index < x.shape[1]):
        raise ValueError("state_index is out of range")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must lie in (0, 1]")

    start = max(0, int(np.floor((1.0 - tail_fraction) * t.shape[0])))
    y = x[start:, state_index]
    tt = t[start:]
    amplitude = float(np.max(y) - np.min(y)) if y.size else 0.0
    mean_value = float(np.mean(y)) if y.size else 0.0
    std_value = float(np.std(y)) if y.size else 0.0

    # Peak detection with a conservative prominence tied to the tail amplitude.
    prominence = max(0.05 * amplitude, 1e-6)
    peaks, _ = find_peaks(y, prominence=prominence)
    period = None
    if peaks.size >= 2:
        peak_times = tt[peaks]
        diffs = np.diff(peak_times)
        if diffs.size > 0:
            period = float(np.mean(diffs))

    regime_label = "oscillation" if amplitude >= float(amplitude_threshold) else "fixed_point"
    return TailOscillationResult(
        amplitude=amplitude,
        mean_value=mean_value,
        std_value=std_value,
        period=period,
        n_peaks=int(peaks.size),
        regime_label=regime_label,
        details={
            "state_index": int(state_index),
            "tail_fraction": float(tail_fraction),
            "amplitude_threshold": float(amplitude_threshold),
        },
    )
