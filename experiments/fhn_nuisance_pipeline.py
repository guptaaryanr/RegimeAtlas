from __future__ import annotations

import sys
from pathlib import Path as _PathBootstrap

REPO_ROOT = _PathBootstrap(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path
import numpy as np

from regime_toolkit.systems import FitzHughNagumo
from regime_toolkit.simulate import SimulationConfig
from regime_toolkit.sweep import IndicatorSpec, ORACLE, parameter_sweep
from regime_toolkit.indicator_factories import make_trajectory_only_indicator_specs
from regime_toolkit.boundaries import (
    composite_structural_score,
    piecewise_linear_change_point,
    first_threshold_crossing,
)
from regime_toolkit.contracts import (
    default_paper_study_contract,
    validate_metric_sources,
)
from regime_toolkit.oracles import tail_oscillation_metrics
from regime_toolkit.io import save_json
from regime_toolkit.calibration import (
    resolve_sweep_values,
    raw_metric_spans,
    default_min_segment_size,
    admissible_min_segment_sizes,
)
from regime_toolkit.plots import (
    set_publication_style,
    plot_metric_vs_param,
    plot_curve_with_boundaries,
)
from experiments.common import (
    BOUNDARY_METRIC_DIRECTIONS,
    BOUNDARY_METRIC_WEIGHTS,
)


def make_oracle_indicators(amplitude_threshold: float) -> list[IndicatorSpec]:
    def oracle_tail_metrics(t, x, system, params):
        metrics = tail_oscillation_metrics(
            t,
            x,
            state_index=0,
            tail_fraction=0.25,
            amplitude_threshold=amplitude_threshold,
        )
        return {
            "tail_amplitude_v": float(metrics.amplitude),
            "is_oscillation": 1.0 if metrics.regime_label == "oscillation" else 0.0,
        }

    return [
        IndicatorSpec(
            name="oracle_tail_metrics",
            fn=oracle_tail_metrics,
            source_class=ORACLE,
            description="tail amplitude oracle for nuisance sweep context",
        )
    ]


def run_pipeline(
    *,
    outdir: Path,
    n: int | None = 9,
    I_min: float | None = 0.495,
    I_max: float | None = 0.505,
    I_values: list[float] | None = None,
    epsilon_fixed: float = 0.08,
    seed: int = 0,
    amplitude_threshold: float = 0.2,
    sim_config: SimulationConfig | None = None,
) -> dict:
    set_publication_style()
    contract = default_paper_study_contract()

    system = FitzHughNagumo()
    base_params = system.default_params()
    base_params["epsilon"] = float(epsilon_fixed)
    values = resolve_sweep_values(
        param_values=I_values,
        n=n,
        min_value=I_min,
        max_value=I_max,
        scale="linear",
    )

    sim_cfg = sim_config or SimulationConfig(
        t_final=220.0,
        dt=0.03,
        transient=120.0,
        method="Radau",
        rtol=1e-7,
        atol=1e-9,
        reset_time_after_transient=True,
    )

    structural_results = parameter_sweep(
        system=system,
        base_params=base_params,
        control_param="I",
        values=values,
        sim_config=sim_cfg,
        indicators=make_trajectory_only_indicator_specs(
            representation="full_state",
            observation_index=0,
            noise_sigma=0.0,
            noise_relative=False,
            embedding_dim=3,
            delay=10,
            stride=1,
            seed=seed,
        ),
        seed=seed,
        store_trajectories_at=[0, len(values) // 2, len(values) - 1],
        verbose=True,
        save_dir=outdir / "structural_baseline",
        save_config={
            "experiment": "fhn_nuisance_pipeline.structural_baseline",
            "benchmark_family": "FitzHugh–Nagumo",
            "sweep_role": "nuisance",
            "fixed_epsilon": float(epsilon_fixed),
            "claim_contract": contract.name,
        },
    )
    validate_metric_sources(
        structural_results["metric_metadata"],
        list(BOUNDARY_METRIC_DIRECTIONS.keys()),
        allowed_sources=contract.allowed_structural_sources,
    )

    oracle_results = parameter_sweep(
        system=system,
        base_params=base_params,
        control_param="I",
        values=values,
        sim_config=sim_cfg,
        indicators=make_oracle_indicators(amplitude_threshold),
        seed=seed,
        store_trajectories_at=None,
        verbose=True,
        save_dir=outdir / "oracle_baseline",
        save_config={
            "experiment": "fhn_nuisance_pipeline.oracle_baseline",
            "benchmark_family": "FitzHugh–Nagumo",
            "sweep_role": "validation",
            "fixed_epsilon": float(epsilon_fixed),
            "claim_contract": contract.name,
        },
    )

    structural_score, structural_components = composite_structural_score(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_results["metrics"],
        BOUNDARY_METRIC_DIRECTIONS,
        weights=BOUNDARY_METRIC_WEIGHTS,
        smooth_window=3,
    )
    structural_cp = piecewise_linear_change_point(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        min_segment_size=default_min_segment_size(len(values), preferred=2),
        smooth_window=1,
    )
    amp_boundary = first_threshold_crossing(
        np.asarray(oracle_results["param_values"], dtype=float),
        np.asarray(oracle_results["metrics"]["tail_amplitude_v"], dtype=float),
        float(amplitude_threshold),
        direction="decreasing",
    )
    unique_labels = int(
        np.unique(
            np.asarray(oracle_results["metrics"]["is_oscillation"], dtype=float)
        ).size
    )
    raw_spans = raw_metric_spans(
        structural_results["metrics"], contract.default_structural_metrics
    )

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_param(
        structural_results,
        metric="time_pr",
        xlabel="I",
        ylabel="Time-weighted tangent PR",
        title=rf"FHN final nuisance sweep at $\epsilon={epsilon_fixed:g}$",
        savepath=figures_dir / "time_pr_vs_I.png",
    )
    plot_metric_vs_param(
        structural_results,
        metric="occupancy_gap",
        xlabel="I",
        ylabel="Occupancy gap",
        title=rf"FHN final nuisance occupancy gap at $\epsilon={epsilon_fixed:g}$",
        savepath=figures_dir / "occupancy_gap_vs_I.png",
    )
    boundaries = [(structural_cp.param_value, "structural boundary")]
    if amp_boundary is not None:
        boundaries.append((amp_boundary, "amplitude boundary"))
    plot_curve_with_boundaries(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        xlabel="I",
        ylabel="Composite structural score",
        title=rf"FHN final nuisance structural score at $\epsilon={epsilon_fixed:g}$",
        boundaries=boundaries,
        savepath=figures_dir / "structural_score_vs_I.png",
    )

    summary = {
        "system": system.name,
        "summary_version": "v1",
        "benchmark_family": "FitzHugh–Nagumo",
        "sweep_role": "nuisance",
        "control_param": "I",
        "param_values": values,
        "fixed_epsilon": float(epsilon_fixed),
        "base_params": base_params,
        "amplitude_threshold": float(amplitude_threshold),
        "structural_components": structural_components,
        "structural_score": structural_score,
        "structural_boundary": {
            "param_value": float(structural_cp.param_value),
            "index": int(structural_cp.index),
            "score": float(structural_cp.score),
        },
        "primary_qualitative_boundary_kind": "amplitude_boundary",
        "primary_qualitative_boundary": (
            None
            if amp_boundary is None
            else {
                "param_value": float(amp_boundary),
                "kind": "amplitude_boundary",
            }
        ),
        "primary_qualitative_label_unique_count": unique_labels,
        "oracle": {
            "amplitude_boundary": amp_boundary,
            "tail_amplitude_v": oracle_results["metrics"]["tail_amplitude_v"],
            "is_oscillation": oracle_results["metrics"]["is_oscillation"],
        },
        "raw_metric_spans": raw_spans,
        "claim_compliance": {
            "contract_name": contract.name,
            "contract_version": contract.version,
            "core_is_trajectory_only": True,
            "oracle_saved_separately": True,
        },
    }
    save_json(summary, outdir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FHN final nuisance sweep over I at fixed epsilon."
    )
    parser.add_argument("--out", type=str, default="outputs/fhn_nuisance")
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--I_min", type=float, default=0.495)
    parser.add_argument("--I_max", type=float, default=0.505)
    parser.add_argument("--epsilon_fixed", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amplitude_threshold", type=float, default=0.2)
    args = parser.parse_args()

    run_pipeline(
        outdir=Path(args.out),
        n=args.n,
        I_min=args.I_min,
        I_max=args.I_max,
        epsilon_fixed=args.epsilon_fixed,
        seed=args.seed,
        amplitude_threshold=args.amplitude_threshold,
    )


if __name__ == "__main__":
    main()
