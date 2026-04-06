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
    change_point_sensitivity_scan,
    first_threshold_crossing,
    lead_distance,
)
from regime_toolkit.contracts import (
    default_paper_study_contract,
    validate_metric_sources,
)
from regime_toolkit.oracles import vdp_complexity_oracle
from regime_toolkit.robustness import RobustnessCase, run_robustness_cases
from regime_toolkit.ablation import evaluate_structural_ablation_suite
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
    plot_boundary_overlay,
    plot_robustness_boundary_summary,
)
from experiments.common import (
    BOUNDARY_METRIC_DIRECTIONS,
    BOUNDARY_METRIC_WEIGHTS,
    build_primary_robustness_cases,
    build_supplemental_delay_cases,
)


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
            description="stroboscopic cluster-count oracle for simple-vs-complex forced response",
        )
    ]


def _make_indicator_factory(state_selector):
    def factory(case: RobustnessCase, seed: int):
        return make_trajectory_only_indicator_specs(
            representation=case.representation,
            observation_index=case.observation_index,
            noise_sigma=case.observation_noise_sigma,
            noise_relative=case.observation_noise_relative,
            embedding_dim=case.embedding_dim,
            delay=case.delay,
            stride=case.stride,
            seed=seed,
            state_selector=state_selector,
        )

    return factory


def run_pipeline(
    *,
    outdir: Path,
    n: int | None = None,
    A_min: float | None = 0.2,
    A_max: float | None = 1.6,
    A_values: list[float] | None = None,
    mu_fixed: float = 8.0,
    omega_fixed: float = 0.9,
    seed: int = 0,
    cluster_threshold: int = 4,
    period_cv_threshold: float = 0.03,
    norm_spread_threshold: float = 0.75,
    run_robustness: bool = True,
    sim_config: SimulationConfig | None = None,
) -> dict:
    set_publication_style()
    contract = default_paper_study_contract()

    system = AutonomousForcedVanDerPol()
    base_params = system.default_params()
    base_params["mu"] = float(mu_fixed)
    base_params["omega"] = float(omega_fixed)
    if A_values is None and n is None:
        A_values = [
            0.2,
            0.35,
            0.5,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
            1.025,
            1.05,
            1.075,
            1.1,
            1.15,
            1.2,
            1.25,
            1.4,
            1.6,
        ]
    values = resolve_sweep_values(
        param_values=A_values,
        n=n,
        min_value=A_min,
        max_value=A_max,
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
        control_param="A",
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
            "experiment": "vdp_pipeline.structural_baseline",
            "benchmark_family": "Autonomous forced van der Pol",
            "sweep_role": "main",
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
        control_param="A",
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
            "experiment": "vdp_pipeline.oracle_baseline",
            "benchmark_family": "Autonomous forced van der Pol",
            "sweep_role": "validation",
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
    sensitivity = change_point_sensitivity_scan(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        smooth_windows=(1, 3, 5),
        min_segment_sizes=admissible_min_segment_sizes(
            len(values), (1, 2, 3, max(1, len(values) // 6))
        ),
    )

    complexity_boundary = first_threshold_crossing(
        np.asarray(oracle_results["param_values"], dtype=float),
        np.asarray(oracle_results["metrics"]["strobe_cluster_count"], dtype=float),
        float(cluster_threshold),
        direction="decreasing",
    )
    unique_labels = int(
        np.unique(
            np.asarray(oracle_results["metrics"]["is_complex_response"], dtype=float)
        ).size
    )

    structural_ci = None
    if sensitivity.ci_low is not None and sensitivity.ci_high is not None:
        structural_ci = (float(sensitivity.ci_low), float(sensitivity.ci_high))

    lead_vs_primary = None
    if complexity_boundary is not None:
        lead_vs_primary = lead_distance(
            structural_cp.param_value,
            complexity_boundary,
            structural_ci=structural_ci,
            qualitative_ci=None,
            details={"comparison": "complexity_boundary"},
        )

    raw_spans = raw_metric_spans(
        structural_results["metrics"], contract.default_structural_metrics
    )

    ablation = evaluate_structural_ablation_suite(
        param_values=np.asarray(structural_results["param_values"], dtype=float),
        metrics=structural_results["metrics"],
        metric_directions=BOUNDARY_METRIC_DIRECTIONS,
        weights=BOUNDARY_METRIC_WEIGHTS,
        qualitative_boundary=complexity_boundary,
    )

    robustness = None
    if run_robustness:
        forcing_period = 2.0 * np.pi / float(omega_fixed)
        baseline_post_window = float(sim_cfg.t_final - sim_cfg.transient)
        shorter_post_window = max(8.0 * forcing_period, 0.85 * baseline_post_window)

        cases = build_primary_robustness_cases(
            base_dt=sim_cfg.dt,
            base_t_final=sim_cfg.t_final,
            base_transient=sim_cfg.transient,
            base_method=sim_cfg.method,
            full_state_noise_sigma=0.0010,
            coarse_dt_factor=1.10,
            shorter_window_t_final=float(sim_cfg.transient + shorter_post_window),
            shorter_window_transient=float(sim_cfg.transient),
            baseline_replicates=3,
            noisy_replicates=6,
            secondary_replicates=4,
            solver_crosscheck_method="Radau",
        ) + build_supplemental_delay_cases(
            delay=10,
            embedding_dim=3,
            clean_delay=24,
            clean_embedding_dim=3,
            noisy_delay=24,
            noisy_embedding_dim=3,
            noisy_sigma=0.0003,
            stress_sigma=0.002,
            clean_replicates=4,
            noisy_replicates=12,
            stress_replicates=6,
        )

        robustness = run_robustness_cases(
            system=system,
            base_params=base_params,
            control_param="A",
            values=values,
            base_sim_config=sim_cfg,
            cases=cases,
            indicator_factory=_make_indicator_factory(_physical_state),
            boundary_metric_directions=BOUNDARY_METRIC_DIRECTIONS,
            boundary_weights=BOUNDARY_METRIC_WEIGHTS,
            seed=seed,
            precomputed_baseline=structural_results,
            verbose=True,
            save_dir=outdir / "robustness",
        )

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [(structural_cp.param_value, "structural boundary")]
    if complexity_boundary is not None:
        boundaries.append((complexity_boundary, "complexity boundary"))

    plot_metric_vs_param(
        structural_results,
        metric="time_pr",
        xlabel="A",
        ylabel="Time-weighted tangent PR",
        title=rf"Autonomous forced VdP final main sweep at $\omega={omega_fixed:g}$",
        savepath=figures_dir / "time_pr_vs_A.png",
    )
    plot_metric_vs_param(
        structural_results,
        metric="occupancy_gap",
        xlabel="A",
        ylabel="Occupancy gap",
        title=rf"Autonomous forced VdP final occupancy gap at $\omega={omega_fixed:g}$",
        savepath=figures_dir / "occupancy_gap_vs_A.png",
    )
    plot_curve_with_boundaries(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        xlabel="A",
        ylabel="Composite structural score",
        title=rf"Autonomous forced VdP final structural score at $\omega={omega_fixed:g}$",
        boundaries=boundaries,
        savepath=figures_dir / "structural_score_vs_A.png",
    )
    plot_boundary_overlay(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        np.asarray(oracle_results["metrics"]["strobe_cluster_count"], dtype=float),
        xlabel="A",
        structural_ylabel="Structural score",
        oracle_ylabel="Strobe cluster count",
        structural_boundary=float(structural_cp.param_value),
        qualitative_boundary=(
            None if complexity_boundary is None else float(complexity_boundary)
        ),
        qualitative_label="complexity boundary",
        savepath=figures_dir / "boundary_overlay.png",
    )
    if robustness is not None:
        plot_robustness_boundary_summary(
            robustness["summary_rows"],
            savepath=figures_dir / "robustness_boundary_shifts.png",
        )

    claim_compliance = {
        "contract_name": contract.name,
        "contract_version": contract.version,
        "core_is_trajectory_only": True,
        "primary_observation_scope": contract.primary_observation_scope,
        "supplemental_scope": contract.supplemental_scope,
        "oracle_saved_separately": True,
        "structural_boundary_metrics": list(contract.default_structural_metrics),
        "structural_metrics_validated": True,
        "structural_sources_allowed": list(contract.allowed_structural_sources),
    }

    summary = {
        "system": system.name,
        "summary_version": "v1",
        "benchmark_family": "Autonomous forced van der Pol",
        "sweep_role": "main",
        "control_param": "A",
        "param_values": values,
        "base_params": base_params,
        "cluster_threshold": int(cluster_threshold),
        "period_cv_threshold": float(period_cv_threshold),
        "norm_spread_threshold": float(norm_spread_threshold),
        "boundary_metric_directions": BOUNDARY_METRIC_DIRECTIONS,
        "boundary_metric_weights": BOUNDARY_METRIC_WEIGHTS,
        "structural_components": structural_components,
        "structural_score": structural_score,
        "structural_boundary": {
            "param_value": float(structural_cp.param_value),
            "index": int(structural_cp.index),
            "score": float(structural_cp.score),
            "left_slope": float(structural_cp.left_slope),
            "right_slope": float(structural_cp.right_slope),
        },
        "structural_boundary_sensitivity": {
            "candidate_param_values": sensitivity.candidate_param_values,
            "ci_low": sensitivity.ci_low,
            "ci_high": sensitivity.ci_high,
            "median_param": sensitivity.median_param,
            "details": sensitivity.details,
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
        "lead_distance_primary": (
            None
            if lead_vs_primary is None
            else {
                "structural_param": lead_vs_primary.structural_param,
                "qualitative_param": lead_vs_primary.qualitative_param,
                "lead_distance": lead_vs_primary.lead_distance,
                "structural_ci": lead_vs_primary.structural_ci,
                "qualitative_ci": lead_vs_primary.qualitative_ci,
                "details": lead_vs_primary.details,
            }
        ),
        "oracle": {
            "complexity_boundary": complexity_boundary,
            "strobe_cluster_count": oracle_results["metrics"]["strobe_cluster_count"],
            "strobe_cluster_ratio": oracle_results["metrics"]["strobe_cluster_ratio"],
            "strobe_norm_spread": oracle_results["metrics"]["strobe_norm_spread"],
            "tail_peak_period_cv": oracle_results["metrics"]["tail_peak_period_cv"],
            "is_complex_response": oracle_results["metrics"]["is_complex_response"],
        },
        "raw_metric_spans": raw_spans,
        "ablation": {
            "passed": bool(ablation.passed),
            "details": ablation.details,
            "variants": {
                name: {
                    "metric_names": list(variant.metric_names),
                    "weights": variant.weights,
                    "boundary_param": variant.boundary_param,
                    "ci_low": variant.ci_low,
                    "ci_high": variant.ci_high,
                    "lead_distance": variant.lead_distance,
                    "details": variant.details,
                }
                for name, variant in ablation.variants.items()
            },
        },
        "robustness_summary_rows": (
            None if robustness is None else robustness["summary_rows"]
        ),
        "robustness_acceptance_primary": (
            None if robustness is None else robustness["acceptance_core"]
        ),
        "robustness_acceptance_supplemental": (
            None if robustness is None else robustness["acceptance_supplemental"]
        ),
        "claim_compliance": claim_compliance,
        "acceptance": {
            "lead_distance_positive_vs_primary": (
                None
                if lead_vs_primary is None
                else bool(lead_vs_primary.lead_distance > 0.0)
            ),
            "primary_robustness_passed": (
                None
                if robustness is None or robustness["acceptance_core"] is None
                else bool(robustness["acceptance_core"]["core_passed"])
            ),
            "supplemental_robustness_passed": (
                None
                if robustness is None or robustness["acceptance_supplemental"] is None
                else bool(robustness["acceptance_supplemental"]["supplemental_passed"])
            ),
            "stress_robustness_passed": (
                None
                if robustness is None or robustness.get("acceptance_stress") is None
                else bool(robustness["acceptance_stress"]["stress_passed"])
            ),
        },
    }
    save_json(summary, outdir / "summary.json")
    save_json(claim_compliance, outdir / "claim_compliance.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous forced VdP final main sweep with calibrated complexity oracle."
    )
    parser.add_argument("--out", type=str, default="outputs/vdp_main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omega_fixed", type=float, default=0.9)
    parser.add_argument("--cluster_threshold", type=int, default=4)
    parser.add_argument("--period_cv_threshold", type=float, default=0.03)
    parser.add_argument("--norm_spread_threshold", type=float, default=0.75)
    parser.add_argument("--run_robustness", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        outdir=Path(args.out),
        seed=args.seed,
        omega_fixed=args.omega_fixed,
        cluster_threshold=args.cluster_threshold,
        period_cv_threshold=args.period_cv_threshold,
        norm_spread_threshold=args.norm_spread_threshold,
        run_robustness=args.run_robustness,
    )


if __name__ == "__main__":
    main()
