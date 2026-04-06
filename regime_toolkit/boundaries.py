from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class ChangePointResult:
    """
    Summary of a 1D parameter-space change point.

    The chosen index corresponds to the last sample in the left segment.
    The returned parameter value is taken from that same index so the result stays
    tied to a real sweep sample.
    """
    param_value: float
    index: int
    score: float
    left_slope: float
    right_slope: float
    sse: float
    smoothed_values: np.ndarray
    details: Dict[str, Any]


@dataclass(frozen=True)
class BootstrapChangePointResult:
    estimate: ChangePointResult
    bootstrap_param_values: np.ndarray
    ci_low: Optional[float]
    ci_high: Optional[float]
    details: Dict[str, Any]


@dataclass(frozen=True)
class SensitivityChangePointResult:
    estimate: ChangePointResult
    candidate_param_values: np.ndarray
    ci_low: Optional[float]
    ci_high: Optional[float]
    median_param: Optional[float]
    details: Dict[str, Any]


@dataclass(frozen=True)
class LeadDistanceResult:
    structural_param: float
    qualitative_param: float
    lead_distance: float
    structural_ci: Optional[tuple[float, float]]
    qualitative_ci: Optional[tuple[float, float]]
    details: Dict[str, Any]



def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Edge-padded centered moving average.

    Why this instead of something fancier:
    - It is deterministic, dependency-light, and adequate for low-noise sweep curves.
    - Boundary extraction should stay interpretable and avoid heavy smoothing choices.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1 or values.shape[0] == 0:
        return values.copy()

    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")



def robust_standardize(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Robust z-scoring based on median and MAD, with std fallback when needed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be 1D")

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(values)
    return (values - center) / scale



def orient_metric(values: np.ndarray, direction: int | str) -> np.ndarray:
    """
    Orient a curve so that larger values indicate stronger structural degradation.

    Supported directions:
    - +1 or "increasing": use as-is
    - -1 or "decreasing": flip sign
    """
    values = np.asarray(values, dtype=float)
    if direction in (+1, "increasing"):
        return values.copy()
    if direction in (-1, "decreasing"):
        return -values
    raise ValueError("direction must be +1/'increasing' or -1/'decreasing'")



def composite_structural_score(
    param_values: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    metric_directions: Mapping[str, int | str],
    *,
    weights: Optional[Mapping[str, float]] = None,
    smooth_window: int = 1,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a single structural score from multiple oriented and standardized metrics.

    The score is intentionally simple: weighted average of robust-z-scored components.
    This keeps the paper claim away from learned or opaque fusion layers.
    """
    param_values = np.asarray(param_values, dtype=float)
    if param_values.ndim != 1:
        raise ValueError("param_values must be 1D")

    components: Dict[str, np.ndarray] = {}
    total = np.zeros_like(param_values, dtype=float)
    total_weight = 0.0

    for key, direction in metric_directions.items():
        if key not in metrics:
            raise KeyError(f"metric '{key}' is missing from metrics")
        oriented = orient_metric(np.asarray(metrics[key], dtype=float), direction)
        standardized = robust_standardize(oriented)
        if smooth_window > 1:
            standardized = centered_moving_average(standardized, smooth_window)
        components[key] = standardized
        w = 1.0 if weights is None else float(weights.get(key, 1.0))
        total += w * standardized
        total_weight += w

    if total_weight <= 0.0:
        raise ValueError("total metric weight must be positive")

    score = total / total_weight
    return score, components



def _line_fit_sse(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y = a*x + b and return slope, intercept, and SSE.

    A closed-form least-squares fit is more stable here than polyfit on very short or nearly
    collinear segments, and it avoids noisy LAPACK warnings during large study runs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    denom = float(np.dot(dx, dx))
    if not np.isfinite(denom) or denom <= 1e-18:
        slope = 0.0
        intercept = y_mean
    else:
        slope = float(np.dot(dx, y - y_mean) / denom)
        intercept = float(y_mean - slope * x_mean)
    pred = slope * x + intercept
    sse = float(np.sum((y - pred) ** 2))
    return slope, intercept, sse



def piecewise_linear_change_point(
    param_values: np.ndarray,
    values: np.ndarray,
    *,
    min_segment_size: int = 3,
    smooth_window: int = 1,
) -> ChangePointResult:
    """
    Detect a single change point by minimizing two-segment linear SSE.

    This is a good fit for the current paper need because most indicator curves are
    smooth and monotone-ish rather than step-like.
    """
    x = np.asarray(param_values, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("param_values and values must be matching 1D arrays")
    if x.shape[0] < 2 * min_segment_size:
        raise ValueError("not enough samples for the requested min_segment_size")

    ys = centered_moving_average(y, smooth_window) if smooth_window > 1 else y.copy()

    best: Optional[tuple[int, float, float, float, float]] = None
    best_sse = np.inf

    for split in range(min_segment_size - 1, x.shape[0] - min_segment_size):
        x_left = x[: split + 1]
        y_left = ys[: split + 1]
        x_right = x[split + 1 :]
        y_right = ys[split + 1 :]

        left_slope, _, left_sse = _line_fit_sse(x_left, y_left)
        right_slope, _, right_sse = _line_fit_sse(x_right, y_right)
        total_sse = left_sse + right_sse

        if total_sse < best_sse:
            best_sse = total_sse
            best = (split, left_slope, right_slope, left_sse, right_sse)

    if best is None:
        raise RuntimeError("failed to identify a change point")

    split, left_slope, right_slope, left_sse, right_sse = best
    score = abs(right_slope - left_slope)
    return ChangePointResult(
        param_value=float(x[split]),
        index=int(split),
        score=float(score),
        left_slope=float(left_slope),
        right_slope=float(right_slope),
        sse=float(best_sse),
        smoothed_values=ys,
        details={
            "min_segment_size": int(min_segment_size),
            "smooth_window": int(smooth_window),
            "left_sse": float(left_sse),
            "right_sse": float(right_sse),
        },
    )



def bootstrap_change_point_from_replicates(
    param_values: np.ndarray,
    replicate_values: np.ndarray,
    *,
    n_bootstrap: int = 200,
    min_segment_size: int = 3,
    smooth_window: int = 1,
    seed: int = 0,
) -> BootstrapChangePointResult:
    """
    Bootstrap change-point uncertainty from replicate curves.

    replicate_values shape must be (n_replicates, n_params). The bootstrap resamples
    replicate curves with replacement, averages them, then re-detects the change point.

    This is designed for robustness studies where each case or run produces a full curve.
    """
    x = np.asarray(param_values, dtype=float)
    curves = np.asarray(replicate_values, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != x.shape[0]:
        raise ValueError("replicate_values must have shape (n_replicates, n_params)")

    estimate = piecewise_linear_change_point(
        x,
        np.mean(curves, axis=0),
        min_segment_size=min_segment_size,
        smooth_window=smooth_window,
    )

    if curves.shape[0] < 2:
        return BootstrapChangePointResult(
            estimate=estimate,
            bootstrap_param_values=np.array([], dtype=float),
            ci_low=None,
            ci_high=None,
            details={
                "n_replicates": int(curves.shape[0]),
                "n_bootstrap": 0,
                "note": "at least two replicate curves are required for a bootstrap CI",
            },
        )

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_bootstrap, dtype=float)
    n_reps = curves.shape[0]
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_reps, size=n_reps)
        mean_curve = np.mean(curves[idx], axis=0)
        cp = piecewise_linear_change_point(
            x,
            mean_curve,
            min_segment_size=min_segment_size,
            smooth_window=smooth_window,
        )
        boot_values[b] = cp.param_value

    ci_low = float(np.quantile(boot_values, 0.025))
    ci_high = float(np.quantile(boot_values, 0.975))
    return BootstrapChangePointResult(
        estimate=estimate,
        bootstrap_param_values=boot_values,
        ci_low=ci_low,
        ci_high=ci_high,
        details={
            "n_replicates": int(curves.shape[0]),
            "n_bootstrap": int(n_bootstrap),
            "min_segment_size": int(min_segment_size),
            "smooth_window": int(smooth_window),
            "seed": int(seed),
        },
    )



def change_point_sensitivity_scan(
    param_values: np.ndarray,
    values: np.ndarray,
    *,
    smooth_windows: Sequence[int] = (1, 3, 5),
    min_segment_sizes: Sequence[int] = (2, 3, 4),
) -> SensitivityChangePointResult:
    """
    Scan change-point settings and return an algorithmic sensitivity interval.

    This is not a statistical CI. It is a stability band over reasonable detector settings,
    which is often more informative for small sweep grids than pretending we have many
    independent replicates.
    """
    x = np.asarray(param_values, dtype=float)
    y = np.asarray(values, dtype=float)
    candidates = []
    last_valid: Optional[ChangePointResult] = None

    for smooth in smooth_windows:
        for seg in min_segment_sizes:
            if x.shape[0] < 2 * seg:
                continue
            cp = piecewise_linear_change_point(
                x,
                y,
                min_segment_size=int(seg),
                smooth_window=int(smooth),
            )
            last_valid = cp
            candidates.append(float(cp.param_value))

    if last_valid is None:
        raise ValueError("no valid change-point settings were available for the scan")

    candidate_array = np.asarray(candidates, dtype=float)
    if candidate_array.size == 0:
        ci_low = None
        ci_high = None
        median_param = None
    else:
        ci_low = float(np.min(candidate_array))
        ci_high = float(np.max(candidate_array))
        median_param = float(np.median(candidate_array))

    return SensitivityChangePointResult(
        estimate=last_valid,
        candidate_param_values=candidate_array,
        ci_low=ci_low,
        ci_high=ci_high,
        median_param=median_param,
        details={
            "smooth_windows": [int(v) for v in smooth_windows],
            "min_segment_sizes": [int(v) for v in min_segment_sizes],
            "n_candidates": int(candidate_array.size),
        },
    )



def first_threshold_crossing(
    param_values: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    direction: str = "decreasing",
) -> Optional[float]:
    """
    Return the interpolated first threshold crossing parameter value.

    This is useful for qualitative regime labels such as oscillation amplitude collapse.
    """
    x = np.asarray(param_values, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("param_values and values must be matching 1D arrays")

    if direction == "decreasing":
        predicate = lambda a, b: (a > threshold) and (b <= threshold)
    elif direction == "increasing":
        predicate = lambda a, b: (a < threshold) and (b >= threshold)
    else:
        raise ValueError("direction must be 'decreasing' or 'increasing'")

    for i in range(x.shape[0] - 1):
        if predicate(float(y[i]), float(y[i + 1])):
            x0, x1 = float(x[i]), float(x[i + 1])
            y0, y1 = float(y[i]), float(y[i + 1])
            if y1 == y0:
                return x1
            alpha = (threshold - y0) / (y1 - y0)
            return float(x0 + alpha * (x1 - x0))
    return None



def lead_distance(
    structural_param: float,
    qualitative_param: float,
    *,
    structural_ci: Optional[tuple[float, float]] = None,
    qualitative_ci: Optional[tuple[float, float]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> LeadDistanceResult:
    """
    Positive lead_distance means the structural boundary appears before the qualitative one.
    """
    return LeadDistanceResult(
        structural_param=float(structural_param),
        qualitative_param=float(qualitative_param),
        lead_distance=float(qualitative_param - structural_param),
        structural_ci=structural_ci,
        qualitative_ci=qualitative_ci,
        details={} if details is None else dict(details),
    )
