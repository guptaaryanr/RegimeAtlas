from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class WeightedParticipationRatioResult:
    value: float
    eigenvalues: np.ndarray
    weights: np.ndarray
    details: Dict[str, Any]


@dataclass(frozen=True)
class SpeedHeterogeneityResult:
    value: float
    mean_speed: float
    std_speed: float
    q10_speed: float
    q90_speed: float
    metric: str



def _participation_ratio(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    s1 = float(np.sum(eigenvalues))
    s2 = float(np.sum(eigenvalues**2))
    if s2 == 0.0:
        return 0.0
    return (s1 * s1) / s2



def _segment_data(
    t: np.ndarray,
    x: np.ndarray,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or t.shape[0] != x.shape[0]:
        raise ValueError("expected t shape (N,), x shape (N,d) with matching N")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if t.shape[0] <= stride:
        raise ValueError("trajectory is too short for the requested stride")

    dt = t[stride:] - t[:-stride]
    if np.any(dt <= 0.0):
        raise ValueError("time array must be strictly increasing")
    dx = x[stride:] - x[:-stride]
    ds = np.linalg.norm(dx, axis=1)
    speed = ds / dt
    return dt, dx, ds, speed



def _weighted_pr_from_segments(
    dx: np.ndarray,
    weights: np.ndarray,
    *,
    normalize_segments: bool = True,
    demean: bool = False,
) -> WeightedParticipationRatioResult:
    dx = np.asarray(dx, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if dx.ndim != 2:
        raise ValueError(f"expected dx shape (M, d), got {dx.shape}")
    if weights.ndim != 1 or weights.shape[0] != dx.shape[0]:
        raise ValueError("weights must be 1D with length matching dx.shape[0]")

    valid = np.isfinite(weights) & (weights > 0.0)
    if normalize_segments:
        seg_norms = np.linalg.norm(dx, axis=1)
        valid &= seg_norms > np.finfo(float).eps
        vectors = dx[valid] / seg_norms[valid, None]
    else:
        vectors = dx[valid]
    w = weights[valid]

    if vectors.shape[0] == 0:
        d = dx.shape[1]
        return WeightedParticipationRatioResult(
            value=0.0,
            eigenvalues=np.zeros(d, dtype=float),
            weights=np.array([], dtype=float),
            details={"normalize_segments": normalize_segments, "demean": demean},
        )

    if demean:
        mean = np.average(vectors, axis=0, weights=w)
        vectors = vectors - mean[None, :]

    C = (vectors.T * w) @ vectors / np.sum(w)
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(evals)[::-1]
    value = _participation_ratio(evals)
    return WeightedParticipationRatioResult(
        value=float(value),
        eigenvalues=evals,
        weights=w,
        details={
            "normalize_segments": normalize_segments,
            "demean": demean,
            "n_segments_used": int(vectors.shape[0]),
        },
    )



def velocity_participation_ratio_time_weighted(
    t: np.ndarray,
    x: np.ndarray,
    *,
    stride: int = 1,
    normalize_segments: bool = True,
    demean: bool = False,
) -> WeightedParticipationRatioResult:
    """
    Time-weighted participation ratio of local secant/tangent directions.

    This measures how many directions dominate the motion when occupancy is weighted by
    physical time. Strong slow-fast dwell tends to push this closer to 1.
    """
    dt, dx, ds, speed = _segment_data(t, x, stride=stride)
    result = _weighted_pr_from_segments(
        dx,
        dt,
        normalize_segments=normalize_segments,
        demean=demean,
    )
    return WeightedParticipationRatioResult(
        value=result.value,
        eigenvalues=result.eigenvalues,
        weights=result.weights,
        details={**result.details, "weighting": "time"},
    )



def velocity_participation_ratio_arclength_weighted(
    t: np.ndarray,
    x: np.ndarray,
    *,
    stride: int = 1,
    normalize_segments: bool = True,
    demean: bool = False,
) -> WeightedParticipationRatioResult:
    """
    Arc-length-weighted participation ratio of local secant/tangent directions.

    This weights by geometric travel rather than dwell time. Comparing this to the
    time-weighted version exposes occupancy imbalance.
    """
    dt, dx, ds, speed = _segment_data(t, x, stride=stride)
    result = _weighted_pr_from_segments(
        dx,
        ds,
        normalize_segments=normalize_segments,
        demean=demean,
    )
    return WeightedParticipationRatioResult(
        value=result.value,
        eigenvalues=result.eigenvalues,
        weights=result.weights,
        details={**result.details, "weighting": "arclength"},
    )



def occupancy_gap(
    t: np.ndarray,
    x: np.ndarray,
    *,
    stride: int = 1,
    normalize_segments: bool = True,
    demean: bool = False,
) -> float:
    """
    Arc-length-weighted PR minus time-weighted PR.

    Positive values mean the trajectory covers more geometric directional variety than
    its time occupancy alone would suggest.
    """
    time_pr = velocity_participation_ratio_time_weighted(
        t,
        x,
        stride=stride,
        normalize_segments=normalize_segments,
        demean=demean,
    )
    arc_pr = velocity_participation_ratio_arclength_weighted(
        t,
        x,
        stride=stride,
        normalize_segments=normalize_segments,
        demean=demean,
    )
    return float(arc_pr.value - time_pr.value)



def speed_heterogeneity(
    t: np.ndarray,
    x: np.ndarray,
    *,
    stride: int = 1,
    metric: str = "cv",
) -> SpeedHeterogeneityResult:
    """
    Measure nonuniformity of trajectory speed.

    Supported metrics:
    - "cv": std(speed) / mean(speed)
    - "log_q90_q10": log(q90 / q10)
    """
    dt, dx, ds, speed = _segment_data(t, x, stride=stride)
    valid = np.isfinite(speed) & (speed > np.finfo(float).eps)
    speed = speed[valid]
    if speed.shape[0] == 0:
        return SpeedHeterogeneityResult(
            value=0.0,
            mean_speed=0.0,
            std_speed=0.0,
            q10_speed=0.0,
            q90_speed=0.0,
            metric=metric,
        )

    mean_speed = float(np.mean(speed))
    std_speed = float(np.std(speed))
    q10 = float(np.quantile(speed, 0.10))
    q90 = float(np.quantile(speed, 0.90))

    if metric == "cv":
        value = 0.0 if mean_speed == 0.0 else std_speed / mean_speed
    elif metric == "log_q90_q10":
        tiny = np.finfo(float).tiny
        value = float(np.log(max(q90, tiny) / max(q10, tiny)))
    else:
        raise ValueError(f"unknown metric: {metric}")

    return SpeedHeterogeneityResult(
        value=float(value),
        mean_speed=mean_speed,
        std_speed=std_speed,
        q10_speed=q10,
        q90_speed=q90,
        metric=metric,
    )
