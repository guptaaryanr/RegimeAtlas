from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import linregress

from ..observations import delay_embed


@dataclass(frozen=True)
class DelayDivergenceResult:
    rate: float
    intercept: float
    rvalue: float
    lag_times: np.ndarray
    mean_log_distance: np.ndarray
    counts: np.ndarray
    details: Dict[str, Any]



def _select_neighbors(
    embedded: np.ndarray,
    anchor_indices: np.ndarray,
    *,
    theiler_window: int,
    max_neighbors: int,
) -> list[tuple[int, int]]:
    tree = cKDTree(embedded)
    k = min(max_neighbors, embedded.shape[0])
    distances, neighbor_ids = tree.query(embedded, k=k)
    if k == 1:
        neighbor_ids = neighbor_ids[:, None]

    pairs: list[tuple[int, int]] = []
    for i in range(embedded.shape[0]):
        chosen_j: Optional[int] = None
        for j in neighbor_ids[i, 1:]:
            if abs(int(anchor_indices[i]) - int(anchor_indices[j])) > theiler_window:
                chosen_j = int(j)
                break
        if chosen_j is not None:
            pairs.append((i, chosen_j))
    return pairs



def rosenstein_style_divergence_rate(
    observation: np.ndarray,
    *,
    dt: float,
    embedding_dim: Optional[int] = None,
    delay: Optional[int] = None,
    theiler_window: Optional[int] = None,
    max_horizon_steps: int = 30,
    fit_start_step: int = 1,
    fit_stop_step: int = 10,
    max_neighbors: int = 32,
    min_pairs_per_step: int = 8,
    min_distance: float = 1e-12,
) -> DelayDivergenceResult:
    """
    Estimate a short-horizon divergence rate from a scalar observation or embedded data.

    This is deliberately a finite-time, trajectory-only diagnostic. It is useful as one
    atlas component, not as a replacement for an oracle Lyapunov-spectrum calculation.
    """
    obs = np.asarray(observation, dtype=float)
    if obs.ndim == 1:
        if embedding_dim is None or delay is None:
            raise ValueError(
                "embedding_dim and delay are required when observation is 1D"
            )
        embedded, anchor_indices = delay_embed(
            obs,
            embedding_dim=embedding_dim,
            delay=delay,
            return_time_indices=True,
        )
    elif obs.ndim == 2:
        embedded = obs
        anchor_indices = np.arange(obs.shape[0], dtype=int)
        if embedding_dim is None:
            embedding_dim = obs.shape[1]
        if delay is None:
            delay = 1
    else:
        raise ValueError("observation must be a 1D scalar series or a 2D embedded array")

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if max_horizon_steps < 2:
        raise ValueError("max_horizon_steps must be at least 2")
    if fit_start_step < 0 or fit_stop_step <= fit_start_step:
        raise ValueError("fit range must satisfy 0 <= fit_start_step < fit_stop_step")

    if theiler_window is None:
        theiler_window = int(max(embedding_dim * delay, delay))
    if theiler_window < 0:
        raise ValueError("theiler_window must be nonnegative")

    pairs = _select_neighbors(
        embedded,
        anchor_indices,
        theiler_window=theiler_window,
        max_neighbors=max_neighbors,
    )
    if len(pairs) == 0:
        raise RuntimeError("no admissible neighbor pairs found for divergence estimate")

    horizon = min(max_horizon_steps, embedded.shape[0] - 1)
    sum_logs = np.zeros(horizon, dtype=float)
    counts = np.zeros(horizon, dtype=int)

    for i, j in pairs:
        usable = min(horizon, embedded.shape[0] - max(i, j))
        if usable <= 0:
            continue
        diffs = embedded[i : i + usable] - embedded[j : j + usable]
        distances = np.linalg.norm(diffs, axis=1)
        valid = np.isfinite(distances) & (distances > float(min_distance))
        idx = np.nonzero(valid)[0]
        sum_logs[idx] += np.log(distances[idx])
        counts[idx] += 1

    lag_steps = np.arange(horizon, dtype=float)
    lag_times = lag_steps * float(dt)
    mean_log_distance = np.full(horizon, np.nan, dtype=float)
    valid_counts = counts >= int(min_pairs_per_step)
    mean_log_distance[valid_counts] = sum_logs[valid_counts] / counts[valid_counts]

    fit_mask = (
        (lag_steps >= float(fit_start_step))
        & (lag_steps <= float(fit_stop_step))
        & np.isfinite(mean_log_distance)
    )
    if np.count_nonzero(fit_mask) < 2:
        raise RuntimeError(
            "insufficient valid lag steps for divergence-rate regression; increase trajectory length or relax fit settings"
        )

    regression = linregress(lag_times[fit_mask], mean_log_distance[fit_mask])
    return DelayDivergenceResult(
        rate=float(regression.slope),
        intercept=float(regression.intercept),
        rvalue=float(regression.rvalue),
        lag_times=lag_times,
        mean_log_distance=mean_log_distance,
        counts=counts,
        details={
            "embedding_dim": int(embedding_dim),
            "delay": int(delay),
            "theiler_window": int(theiler_window),
            "max_horizon_steps": int(horizon),
            "fit_start_step": int(fit_start_step),
            "fit_stop_step": int(fit_stop_step),
            "neighbor_pairs": int(len(pairs)),
            "min_pairs_per_step": int(min_pairs_per_step),
        },
    )
