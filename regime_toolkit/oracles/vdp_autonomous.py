from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class VDPStroboscopicResult:
    points: np.ndarray
    spread: float
    n_points: int
    details: Dict[str, Any]


@dataclass(frozen=True)
class VDPTailPeakResult:
    amplitude: float
    period_mean: Optional[float]
    period_cv: Optional[float]
    n_peaks: int
    details: Dict[str, Any]


@dataclass(frozen=True)
class VDPComplexityOracleResult:
    points: np.ndarray
    spread: float
    norm_spread: float
    state_scale: float
    cluster_count: int
    cluster_ratio: float
    cluster_tolerance: float
    period_cv: Optional[float]
    amplitude: float
    is_complex: bool
    details: Dict[str, Any]



def stroboscopic_section(
    t: np.ndarray,
    x: np.ndarray,
    *,
    phase_index: int = 2,
    state_dims: tuple[int, int] = (0, 1),
    tail_fraction: float = 0.5,
) -> np.ndarray:
    """
    Sample the autonomous forced VdP flow once per forcing cycle using the lifted phase.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or t.shape[0] != x.shape[0]:
        raise ValueError("expected matching t and x arrays")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must lie in (0, 1]")

    start = max(0, int(np.floor((1.0 - tail_fraction) * t.shape[0])))
    xx = x[start:, :]
    phi = np.asarray(xx[:, phase_index], dtype=float)
    wraps = np.floor(phi / (2.0 * np.pi)).astype(int)

    d0, d1 = state_dims
    points = []
    for i in range(len(wraps) - 1):
        if wraps[i + 1] > wraps[i]:
            target = 2.0 * np.pi * wraps[i + 1]
            phi0 = float(phi[i])
            phi1 = float(phi[i + 1])
            alpha = 1.0 if phi1 == phi0 else (target - phi0) / (phi1 - phi0)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            p0 = np.asarray(xx[i, [d0, d1]], dtype=float)
            p1 = np.asarray(xx[i + 1, [d0, d1]], dtype=float)
            points.append((1.0 - alpha) * p0 + alpha * p1)

    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)



def _tail_phase_plane_scale(
    x: np.ndarray,
    *,
    tail_fraction: float,
    state_dims: tuple[int, int] = (0, 1),
) -> float:
    xx = np.asarray(x, dtype=float)
    start = max(0, int(np.floor((1.0 - tail_fraction) * xx.shape[0])))
    tail = xx[start:, list(state_dims)]
    if tail.shape[0] == 0:
        return 0.0
    center = np.mean(tail, axis=0)
    radii = np.sqrt(np.sum((tail - center) ** 2, axis=1))
    return float(np.sqrt(np.mean(radii ** 2))) if radii.size > 0 else 0.0



def _cluster_count(points: np.ndarray, tolerance: float) -> int:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    n = pts.shape[0]
    if n == 0:
        return 0
    if n == 1:
        return 1
    adjacency = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2) <= float(tolerance)
    visited = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if visited[i]:
            continue
        count += 1
        stack = [i]
        visited[i] = True
        while stack:
            j = stack.pop()
            neighbors = np.where(adjacency[j])[0]
            for k in neighbors:
                if not visited[k]:
                    visited[k] = True
                    stack.append(int(k))
    return int(count)



def stroboscopic_metrics(
    t: np.ndarray,
    x: np.ndarray,
    *,
    tail_fraction: float = 0.5,
    phase_index: int = 2,
) -> VDPStroboscopicResult:
    points = stroboscopic_section(
        t,
        x,
        phase_index=phase_index,
        tail_fraction=tail_fraction,
    )
    if points.shape[0] == 0:
        spread = np.nan
    elif points.shape[0] == 1:
        spread = 0.0
    else:
        center = np.mean(points, axis=0)
        spread = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))
    return VDPStroboscopicResult(
        points=points,
        spread=float(spread),
        n_points=int(points.shape[0]),
        details={
            "tail_fraction": float(tail_fraction),
            "phase_index": int(phase_index),
        },
    )



def tail_peak_metrics(
    t: np.ndarray,
    x: np.ndarray,
    *,
    state_index: int = 0,
    tail_fraction: float = 0.5,
) -> VDPTailPeakResult:
    """
    Peak-based oracle summary for forced VdP.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    start = max(0, int(np.floor((1.0 - tail_fraction) * t.shape[0])))
    tt = t[start:]
    y = x[start:, state_index]
    amplitude = float(np.max(y) - np.min(y)) if y.size else 0.0
    prominence = max(0.05 * amplitude, 1e-6)
    peaks, _ = find_peaks(y, prominence=prominence)

    period_mean: Optional[float] = None
    period_cv: Optional[float] = None
    if peaks.size >= 2:
        peak_times = tt[peaks]
        diffs = np.diff(peak_times)
        if diffs.size > 0:
            period_mean = float(np.mean(diffs))
            denom = max(abs(period_mean), 1e-12)
            period_cv = float(np.std(diffs) / denom)

    return VDPTailPeakResult(
        amplitude=amplitude,
        period_mean=period_mean,
        period_cv=period_cv,
        n_peaks=int(peaks.size),
        details={
            "state_index": int(state_index),
            "tail_fraction": float(tail_fraction),
        },
    )



def vdp_complexity_oracle(
    t: np.ndarray,
    x: np.ndarray,
    *,
    tail_fraction: float = 0.7,
    phase_index: int = 2,
    state_dims: tuple[int, int] = (0, 1),
    cluster_tol_fraction: float = 0.05,
    cluster_threshold: int = 4,
    period_cv_threshold: float = 0.03,
    norm_spread_threshold: float = 0.75,
) -> VDPComplexityOracleResult:
    """
    Oracle for simple vs complex forced-response structure.

    Simple entrained responses typically collapse to a small-cardinality stroboscopic set
    with low inter-peak period variability. More complex responses fill a larger stroboscopic
    set or show stronger period variability.
    """
    points = stroboscopic_section(
        t,
        x,
        phase_index=phase_index,
        state_dims=state_dims,
        tail_fraction=tail_fraction,
    )
    strobo = stroboscopic_metrics(t, x, tail_fraction=tail_fraction, phase_index=phase_index)
    peaks = tail_peak_metrics(t, x, state_index=state_dims[0], tail_fraction=tail_fraction)

    state_scale = _tail_phase_plane_scale(x, tail_fraction=tail_fraction, state_dims=state_dims)
    norm_spread = float(strobo.spread / state_scale) if np.isfinite(strobo.spread) and state_scale > 0.0 else np.nan
    cluster_tolerance = max(float(cluster_tol_fraction) * max(state_scale, 1e-12), 1e-3)
    cluster_count = _cluster_count(points, cluster_tolerance) if points.shape[0] > 0 else 0
    cluster_ratio = float(cluster_count / max(strobo.n_points, 1))

    period_cv = peaks.period_cv
    is_complex = bool(
        (cluster_count > int(cluster_threshold))
        or (period_cv is not None and np.isfinite(period_cv) and float(period_cv) > float(period_cv_threshold))
        or (np.isfinite(norm_spread) and float(norm_spread) > float(norm_spread_threshold))
    )

    return VDPComplexityOracleResult(
        points=points,
        spread=float(strobo.spread),
        norm_spread=float(norm_spread),
        state_scale=float(state_scale),
        cluster_count=int(cluster_count),
        cluster_ratio=float(cluster_ratio),
        cluster_tolerance=float(cluster_tolerance),
        period_cv=None if period_cv is None else float(period_cv),
        amplitude=float(peaks.amplitude),
        is_complex=is_complex,
        details={
            "tail_fraction": float(tail_fraction),
            "phase_index": int(phase_index),
            "state_dims": tuple(int(v) for v in state_dims),
            "cluster_tol_fraction": float(cluster_tol_fraction),
            "cluster_threshold": int(cluster_threshold),
            "period_cv_threshold": float(period_cv_threshold),
            "norm_spread_threshold": float(norm_spread_threshold),
        },
    )
