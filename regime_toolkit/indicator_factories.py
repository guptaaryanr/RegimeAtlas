from __future__ import annotations

from typing import Callable, Optional, Sequence
import numpy as np

from .observations import (
    observed_trajectory,
    state_to_scalar_observation,
    add_observation_noise,
)
from .indicators import (
    velocity_participation_ratio_time_weighted,
    velocity_participation_ratio_arclength_weighted,
    speed_heterogeneity,
    rosenstein_style_divergence_rate,
)
from .sweep import IndicatorSpec, TRAJECTORY_ONLY


StateSelector = Optional[Callable[[np.ndarray], np.ndarray]]


def _safe_delay_divergence_result(
    y: np.ndarray,
    *,
    dt: float,
    embedding_dim: int,
    delay: int,
    max_horizon_steps: int,
    fit_start_step: int,
    fit_stop_step: int,
    theiler_window: int,
    min_pairs_per_step: int,
) -> dict[str, float]:
    """
    Compute delay-space divergence, but degrade gracefully on known finite-data
    failures instead of aborting an entire sweep.

    This is safe because delay_div_rate is not one of the fused boundary metrics.
    """
    try:
        result = rosenstein_style_divergence_rate(
            y,
            dt=dt,
            embedding_dim=embedding_dim,
            delay=delay,
            max_horizon_steps=max_horizon_steps,
            fit_start_step=fit_start_step,
            fit_stop_step=fit_stop_step,
            theiler_window=theiler_window,
            min_pairs_per_step=min_pairs_per_step,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        nonfatal = (
            "insufficient valid lag steps" in message
            or "insufficient valid lag-step counts" in message
            or "not enough valid lag steps" in message
            or "not enough valid pairs" in message
            or "no valid lag steps" in message
        )
        if not nonfatal:
            raise

        return {
            "delay_div_rate": float("nan"),
            "delay_div_rvalue": float("nan"),
            "delay_div_quality": float("nan"),
        }

    quality = float(abs(result.rvalue))
    return {
        "delay_div_rate": float(result.rate),
        "delay_div_rvalue": float(result.rvalue),
        "delay_div_quality": quality,
    }


def make_trajectory_only_indicator_specs(
    *,
    representation: str,
    observation_index: int,
    noise_sigma: float,
    noise_relative: bool,
    embedding_dim: int,
    delay: int,
    stride: int,
    seed: int,
    state_selector: StateSelector = None,
    delay_div_max_horizon_steps: int = 30,
    delay_div_fit_start_step: int = 1,
    delay_div_fit_stop_step: int = 10,
    delay_div_min_pairs_per_step: int = 8,
) -> list[IndicatorSpec]:
    """
    Build the common trajectory-only atlas stack used across benchmarks.

    The first spec returns the occupancy/anisotropy family.
    The second spec returns the delay-space divergence family.
    """

    def select_state(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        if state_selector is None:
            return xx
        return np.asarray(state_selector(xx), dtype=float)

    def occupancy_stack(t, x, system, params):
        x_sel = select_state(x)
        obs = observed_trajectory(
            x_sel,
            representation=representation,
            scalar_mode="coordinate",
            index=observation_index,
            noise_sigma=noise_sigma,
            noise_relative=noise_relative,
            embedding_dim=embedding_dim,
            delay=delay,
            stride=stride,
            seed=seed,
        )
        t_obs = np.asarray(t[obs.time_indices], dtype=float)
        z = np.asarray(obs.data, dtype=float)

        time_pr = velocity_participation_ratio_time_weighted(t_obs, z).value
        arc_pr = velocity_participation_ratio_arclength_weighted(t_obs, z).value
        speed_cv = speed_heterogeneity(t_obs, z, metric="cv").value
        speed_log_spread = speed_heterogeneity(t_obs, z, metric="log_q90_q10").value
        return {
            "time_pr": float(time_pr),
            "arc_pr": float(arc_pr),
            "occupancy_gap": float(arc_pr - time_pr),
            "speed_cv": float(speed_cv),
            "speed_log_spread": float(speed_log_spread),
        }

    def delay_divergence(t, x, system, params):
        x_sel = select_state(x)
        y = state_to_scalar_observation(
            x_sel, mode="coordinate", index=observation_index
        )
        y = add_observation_noise(
            y,
            noise_sigma,
            relative=noise_relative,
            seed=seed,
        )
        return _safe_delay_divergence_result(
            y,
            dt=float(t[1] - t[0]),
            embedding_dim=embedding_dim,
            delay=delay,
            max_horizon_steps=delay_div_max_horizon_steps,
            fit_start_step=delay_div_fit_start_step,
            fit_stop_step=delay_div_fit_stop_step,
            theiler_window=max(delay * embedding_dim, 10),
            min_pairs_per_step=delay_div_min_pairs_per_step,
        )

    return [
        IndicatorSpec(
            name="occupancy_stack",
            fn=occupancy_stack,
            source_class=TRAJECTORY_ONLY,
            description="time/arc tangent participation ratio, occupancy gap, and speed heterogeneity",
        ),
        IndicatorSpec(
            name="delay_divergence",
            fn=delay_divergence,
            source_class=TRAJECTORY_ONLY,
            description="Rosenstein-style finite-time delay-space divergence estimate",
        ),
    ]
