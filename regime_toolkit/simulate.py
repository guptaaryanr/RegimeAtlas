from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp

from .systems.base import ODESystem, Params


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration for trajectory generation.

    dt is the sampling interval for returned trajectories.
    solve_ivp will internally use adaptive steps; dt only controls output sampling.
    """
    t_final: float
    dt: float
    transient: float = 0.0
    t0: float = 0.0

    method: str = "RK45"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: Optional[float] = None

    reset_time_after_transient: bool = False


def simulate(
    system: ODESystem,
    params: Params,
    config: SimulationConfig,
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate an ODE system and return a uniformly sampled trajectory.

    Returns:
        t: (N,) time array
        x: (N, d) state trajectory
        meta: dict with solver info and initial condition used
    """
    rng = np.random.default_rng(seed)

    if x0 is None:
        x0 = system.default_initial_condition(params, rng)
    x0 = np.asarray(x0, dtype=float)
    system.validate_state(x0)

    t_start = float(config.t0)
    t_end = t_start + float(config.transient) + float(config.t_final)

    # Uniform output sampling.
    # Include endpoint (within floating tolerance) to make plotting and differencing simpler.
    n_steps = int(np.floor((t_end - t_start) / config.dt)) + 1
    t_eval = t_start + config.dt * np.arange(n_steps, dtype=float)

    def f(t: float, x: np.ndarray) -> np.ndarray:
        return system.rhs(t, x, params)

    solve_kwargs = dict(
        fun=f,
        t_span=(t_start, t_end),
        y0=x0,
        t_eval=t_eval,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
    )
    if config.max_step is not None:
        solve_kwargs["max_step"] = float(config.max_step)

    sol = solve_ivp(**solve_kwargs)
    if not sol.success:
        raise RuntimeError(f"Integration failed for {system.name}: {sol.message}")

    t = np.asarray(sol.t, dtype=float)
    x = np.asarray(sol.y.T, dtype=float)  # (N, d)

    # Discard transient portion if requested.
    if config.transient > 0.0:
        t_cut = t_start + float(config.transient)
        keep = t >= t_cut
        t = t[keep]
        x = x[keep]

        if config.reset_time_after_transient and t.size > 0:
            t = t - t[0]

    meta = {
        "x0": x0,
        "nfev": sol.nfev,
        "njev": sol.njev,
        "nlu": sol.nlu,
        "status": sol.status,
        "message": sol.message,
        "t_start": t_start,
        "t_end": t_end,
    }
    return t, x, meta
