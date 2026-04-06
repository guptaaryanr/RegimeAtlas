from __future__ import annotations

import sys
from pathlib import Path as _PathBootstrap

REPO_ROOT = _PathBootstrap(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path
import numpy as np

from regime_toolkit.systems import AutonomousForcedVanDerPol
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
from regime_toolkit.oracles import vdp_complexity_oracle
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
from experiments.common import BOUNDARY_METRIC_DIRECTIONS, BOUNDARY_METRIC_WEIGHTS


def _physical_state(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[:, :2], dtype=float)


def make_oracle_indicators(
    *,
    cluster_threshold: int,
    period_cv_threshold: float,
    norm_spread_threshold: float,
) -> list[IndicatorSpec]:
    def oracle_metrics(t, x, system, params):
        result = vdp_complexity_oracle(
            t,
            x,
            tail_fraction=0.7,
            phase_index=2,
            state_dims=(0, 1),
            cluster_tol_fraction=0.05,
            cluster_threshold=cluster_threshold,
            period_cv_threshold=period_cv_threshold,
            norm_spread_threshold=norm_spread_threshold,
        )
        return {
            "strobe_cluster_count": float(result.cluster_count),
            "strobe_cluster_ratio": float(result.cluster_ratio),
            "strobe_norm_spread": float(result.norm_spread),
            "tail_peak_period_cv": (
                np.nan if result.period_cv is None else float(result.period_cv)
            ),
            "tail_peak_amplitude": float(result.amplitude),
            "is_complex_response": 1.0 if result.is_complex else 0.0,
        }

    return [
        IndicatorSpec(
            name="oracle_metrics",
            fn=oracle_metrics,
            source_class=ORACLE,
            description="stroboscopic cluster-count oracle for nuisance sweep context",
        )
    ]


def run_pipeline(
    *,
    outdir: Path,
    n: int | None = 9,
    omega_min: float | None = 0.86,
    omega_max: float | None = 0.94,
    omega_values: list[float] | None = None,
    A_fixed: float = 0.8,
    mu_fixed: float = 8.0,
    seed: int = 0,
    cluster_threshold: int = 4,
    period_cv_threshold: float = 0.03,
    norm_spread_threshold: float = 0.75,
    sim_config: SimulationConfig | None = None,
) -> dict:
    set_publication_style()
    contract = default_paper_study_contract()

    system = AutonomousForcedVanDerPol()
    base_params = system.default_params()
    base_params["A"] = float(A_fixed)
    base_params["mu"] = float(mu_fixed)
    values = resolve_sweep_values(
        param_values=omega_values,
        n=n,
        min_value=omega_min,
        max_value=omega_max,
        scale="linear",
    )

    sim_cfg = sim_config or SimulationConfig(
        t_final=160.0,
        dt=0.05,
        transient=80.0,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        max_step=0.05,
        reset_time_after_transient=True,
    )

    structural_results = parameter_sweep(
        system=system,
        base_params=base_params,
        control_param="omega",
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
            state_selector=_physical_state,
        ),
        seed=seed,
        store_trajectories_at=[0, len(values) // 2, len(values) - 1],
        verbose=True,
        save_dir=outdir / "structural_baseline",
        save_config={
            "experiment": "vdp_nuisance_pipeline.structural_baseline",
            "benchmark_family": "Autonomous forced van der Pol",
            "sweep_role": "nuisance",
            "fixed_A": float(A_fixed),
            "fixed_mu": float(mu_fixed),
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
        control_param="omega",
        values=values,
        sim_config=sim_cfg,
        indicators=make_oracle_indicators(
            cluster_threshold=cluster_threshold,
            period_cv_threshold=period_cv_threshold,
            norm_spread_threshold=norm_spread_threshold,
        ),
        seed=seed,
        store_trajectories_at=None,
        verbose=True,
        save_dir=outdir / "oracle_baseline",
        save_config={
            "experiment": "vdp_nuisance_pipeline.oracle_baseline",
            "benchmark_family": "Autonomous forced van der Pol",
            "sweep_role": "validation",
            "fixed_A": float(A_fixed),
            "fixed_mu": float(mu_fixed),
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
    complexity_boundary = first_threshold_crossing(
        np.asarray(oracle_results["param_values"], dtype=float),
        np.asarray(oracle_results["metrics"]["strobe_cluster_count"], dtype=float),
        float(cluster_threshold),
        direction="increasing",
    )
    unique_labels = int(
        np.unique(
            np.asarray(oracle_results["metrics"]["is_complex_response"], dtype=float)
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
        xlabel="omega",
        ylabel="Time-weighted tangent PR",
        title=rf"Autonomous forced VdP final nuisance sweep at $A={A_fixed:g}$",
        savepath=figures_dir / "time_pr_vs_omega.png",
    )
    plot_metric_vs_param(
        structural_results,
        metric="occupancy_gap",
        xlabel="omega",
        ylabel="Occupancy gap",
        title=rf"Autonomous forced VdP final nuisance occupancy gap at $A={A_fixed:g}$",
        savepath=figures_dir / "occupancy_gap_vs_omega.png",
    )
    boundaries = [(structural_cp.param_value, "structural boundary")]
    if complexity_boundary is not None:
        boundaries.append((complexity_boundary, "complexity boundary"))
    plot_curve_with_boundaries(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        xlabel="omega",
        ylabel="Composite structural score",
        title=rf"Autonomous forced VdP final nuisance structural score at $A={A_fixed:g}$",
        boundaries=boundaries,
        savepath=figures_dir / "structural_score_vs_omega.png",
    )

    summary = {
        "system": system.name,
        "summary_version": "v1",
        "benchmark_family": "Autonomous forced van der Pol",
        "sweep_role": "nuisance",
        "control_param": "omega",
        "param_values": values,
        "fixed_A": float(A_fixed),
        "fixed_mu": float(mu_fixed),
        "base_params": base_params,
        "cluster_threshold": int(cluster_threshold),
        "period_cv_threshold": float(period_cv_threshold),
        "norm_spread_threshold": float(norm_spread_threshold),
        "structural_components": structural_components,
        "structural_score": structural_score,
        "structural_boundary": {
            "param_value": float(structural_cp.param_value),
            "index": int(structural_cp.index),
            "score": float(structural_cp.score),
        },
        "primary_qualitative_boundary_kind": "complexity_boundary",
        "primary_qualitative_boundary": (
            None
            if complexity_boundary is None
            else {
                "param_value": float(complexity_boundary),
                "kind": "complexity_boundary",
            }
        ),
        "primary_qualitative_label_unique_count": unique_labels,
        "oracle": {
            "complexity_boundary": complexity_boundary,
            "strobe_cluster_count": oracle_results["metrics"]["strobe_cluster_count"],
            "strobe_cluster_ratio": oracle_results["metrics"]["strobe_cluster_ratio"],
            "strobe_norm_spread": oracle_results["metrics"]["strobe_norm_spread"],
            "tail_peak_period_cv": oracle_results["metrics"]["tail_peak_period_cv"],
            "is_complex_response": oracle_results["metrics"]["is_complex_response"],
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
        description="Autonomous forced VdP final nuisance sweep over omega at fixed A."
    )
    parser.add_argument("--out", type=str, default="outputs/vdp_nuisance")
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--omega_min", type=float, default=0.86)
    parser.add_argument("--omega_max", type=float, default=0.94)
    parser.add_argument("--A_fixed", type=float, default=0.8)
    parser.add_argument("--mu_fixed", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_pipeline(
        outdir=Path(args.out),
        n=args.n,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        A_fixed=args.A_fixed,
        mu_fixed=args.mu_fixed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
