from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Sequence, Union
import numpy as np

from .systems.base import ODESystem, Params
from .simulate import simulate, SimulationConfig
from .io import save_sweep_bundle


IndicatorOutput = Union[float, Dict[str, float]]

TRAJECTORY_ONLY = "trajectory_only"
ORACLE = "oracle"


@dataclass(frozen=True)
class IndicatorSpec:
    """
    An indicator is a pure function that consumes (t, x, system, params) and returns:
      - a scalar float, or
      - a dict of scalar metrics

    The source_class field is important for paper correctness.
    It lets the code enforce that the structural score only uses trajectory-only metrics,
    while oracle metrics remain in the validation layer.
    """
    name: str
    fn: Callable[[np.ndarray, np.ndarray, ODESystem, Params], IndicatorOutput]
    source_class: str = TRAJECTORY_ONLY
    description: str = ""



def _register_metric_metadata(
    metric_metadata: Dict[str, Dict[str, Any]],
    metric_name: str,
    spec: IndicatorSpec,
) -> None:
    metric_metadata.setdefault(
        metric_name,
        {
            "indicator_name": spec.name,
            "source_class": spec.source_class,
            "description": spec.description,
        },
    )



def parameter_sweep(
    system: ODESystem,
    base_params: Params,
    control_param: str,
    values: Sequence[float],
    sim_config: SimulationConfig,
    indicators: Sequence[IndicatorSpec],
    *,
    seed: Optional[int] = 0,
    store_trajectories_at: Optional[Sequence[int]] = None,
    verbose: bool = True,
    save_dir: Optional[str | Path] = None,
    save_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generic 1D parameter sweep.

    Returns a results dict with:
      - param_values
      - metrics: dict[str, np.ndarray]
      - metric_metadata: dict[str, dict] describing provenance of each metric
      - trajectories: optional stored trajectories
      - metadata about sweep and simulation

    Notes:
    - This is intentionally sequential and simple. Parallelization is easy to add later.
    - If save_dir is provided, a stable artifact bundle is written automatically:
      results.json, trajectories.npz, config.json, manifest.json.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D sequence")

    if control_param not in base_params:
        raise KeyError(f"control_param '{control_param}' not found in base_params keys {list(base_params.keys())}")

    metrics: Dict[str, List[float]] = {}
    metric_metadata: Dict[str, Dict[str, Any]] = {}
    trajectories: Dict[float, Dict[str, np.ndarray]] = {}

    store_set = set(store_trajectories_at or [])
    rng = np.random.default_rng(seed)

    for i, val in enumerate(values):
        params = dict(base_params)
        params[control_param] = float(val)

        sim_seed = int(rng.integers(0, 2**31 - 1))

        if verbose:
            print(f"[{i+1:>3}/{len(values)}] {control_param}={val:g}")

        t, x, meta = simulate(system, params, sim_config, x0=None, seed=sim_seed)

        for spec in indicators:
            out = spec.fn(t, x, system, params)
            if isinstance(out, (float, np.floating, int)):
                metrics.setdefault(spec.name, []).append(float(out))
                _register_metric_metadata(metric_metadata, spec.name, spec)
            elif isinstance(out, dict):
                for k, v in out.items():
                    metrics.setdefault(k, []).append(float(v))
                    _register_metric_metadata(metric_metadata, k, spec)
            else:
                raise TypeError(f"Indicator '{spec.name}' returned unsupported type: {type(out)}")

        if i in store_set:
            trajectories[float(val)] = {
                "t": t.copy(),
                "x": x.copy(),
                "meta": {
                    **meta,
                    "params": dict(params),
                    "sim_seed": sim_seed,
                },
            }

    metrics_array = {k: np.asarray(v, dtype=float) for k, v in metrics.items()}
    results: Dict[str, Any] = {
        "system": system.name,
        "control_param": control_param,
        "param_values": values,
        "metrics": metrics_array,
        "metric_metadata": metric_metadata,
        "trajectories": trajectories,
        "simulation_config": sim_config,
        "base_params": dict(base_params),
        "indicator_names": [spec.name for spec in indicators],
        "seed": seed,
    }

    if save_dir is not None:
        config_payload = {
            "system": system.name,
            "control_param": control_param,
            "base_params": dict(base_params),
            "param_values": values,
            "simulation_config": sim_config,
            "indicator_names": [spec.name for spec in indicators],
            "indicator_specs": {
                spec.name: {
                    "source_class": spec.source_class,
                    "description": spec.description,
                }
                for spec in indicators
            },
            "seed": seed,
        }
        if save_config is not None:
            config_payload.update(dict(save_config))
        save_sweep_bundle(results, save_dir, config=config_payload)

    return results
