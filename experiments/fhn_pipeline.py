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
    change_point_sensitivity_scan,
    first_threshold_crossing,
    lead_distance,
)
from regime_toolkit.contracts import (
    default_paper_study_contract,
    validate_metric_sources,
)
from regime_toolkit.oracles import predict_fhn_hopf_epsilon, tail_oscillation_metrics
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
            "tail_std_v": float(metrics.std_value),
            "tail_period_v": (
                np.nan if metrics.period is None else float(metrics.period)
            ),
            "is_oscillation": 1.0 if metrics.regime_label == "oscillation" else 0.0,
        }

    return [
        IndicatorSpec(
            name="oracle_tail_metrics",
            fn=oracle_tail_metrics,
            source_class=ORACLE,
            description="tail amplitude and period oracle for oscillation-vs-fixed-point classification",
        )
    ]


def _make_indicator_factory(seed_base: int):
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
        )

    return factory


def run_pipeline(
    *,
    outdir: Path,
    n: int | None = 12,
    eps_min: float | None = 0.02,
    eps_max: float | None = 0.5,
    eps_values: list[float] | None = None,
    seed: int = 0,
    amplitude_threshold: float = 0.2,
    run_robustness: bool = True,
    sim_config: SimulationConfig | None = None,
) -> dict:
    set_publication_style()
    contract = default_paper_study_contract()

    system = FitzHughNagumo()
    base_params = system.default_params()
    values = resolve_sweep_values(
        param_values=eps_values,
        n=n,
        min_value=eps_min,
        max_value=eps_max,
        scale="geom",
    )

    sim_cfg = sim_config or SimulationConfig(
        t_final=260.0,
        dt=0.03,
        transient=180.0,
        method="Radau",
        rtol=1e-7,
        atol=1e-9,
        reset_time_after_transient=True,
    )

    structural_results = parameter_sweep(
        system=system,
        base_params=base_params,
        control_param="epsilon",
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
            "experiment": "fhn_pipeline.structural_baseline",
            "benchmark_family": "FitzHugh–Nagumo",
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
        control_param="epsilon",
        values=values,
        sim_config=sim_cfg,
        indicators=make_oracle_indicators(amplitude_threshold),
        seed=seed,
        store_trajectories_at=None,
        verbose=True,
        save_dir=outdir / "oracle_baseline",
        save_config={
            "experiment": "fhn_pipeline.oracle_baseline",
            "benchmark_family": "FitzHugh–Nagumo",
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

    predicted_hopf = predict_fhn_hopf_epsilon(base_params)
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

    structural_ci = None
    if sensitivity.ci_low is not None and sensitivity.ci_high is not None:
        structural_ci = (float(sensitivity.ci_low), float(sensitivity.ci_high))

    lead_vs_primary = None
    if amp_boundary is not None:
        lead_vs_primary = lead_distance(
            structural_cp.param_value,
            amp_boundary,
            structural_ci=structural_ci,
            qualitative_ci=None,
            details={"comparison": "amplitude_threshold_crossing"},
        )

    lead_vs_hopf = None
    if predicted_hopf is not None:
        lead_vs_hopf = lead_distance(
            structural_cp.param_value,
            predicted_hopf,
            structural_ci=structural_ci,
            qualitative_ci=None,
            details={"comparison": "predicted_hopf"},
        )

    raw_spans = raw_metric_spans(
        structural_results["metrics"], contract.default_structural_metrics
    )

    ablation = evaluate_structural_ablation_suite(
        param_values=np.asarray(structural_results["param_values"], dtype=float),
        metrics=structural_results["metrics"],
        metric_directions=BOUNDARY_METRIC_DIRECTIONS,
        weights=BOUNDARY_METRIC_WEIGHTS,
        qualitative_boundary=amp_boundary,
    )

    robustness = None
    if run_robustness:
        cases = build_primary_robustness_cases(
            base_dt=sim_cfg.dt,
            base_t_final=sim_cfg.t_final,
            base_transient=sim_cfg.transient,
            base_method=sim_cfg.method,
            full_state_noise_sigma=0.005,
            coarse_dt_factor=2.0,
            shorter_window_fraction=0.70,
            baseline_replicates=3,
            noisy_replicates=5,
            secondary_replicates=3,
        ) + build_supplemental_delay_cases(
            delay=10,
            embedding_dim=3,
            clean_delay=3,
            clean_embedding_dim=3,
            noisy_delay=3,
            noisy_embedding_dim=3,
            noisy_sigma=0.0002,
            stress_sigma=0.005,
            clean_replicates=4,
            noisy_replicates=14,
            stress_replicates=4,
        )

        robustness = run_robustness_cases(
            system=system,
            base_params=base_params,
            control_param="epsilon",
            values=values,
            base_sim_config=sim_cfg,
            cases=cases,
            indicator_factory=_make_indicator_factory(seed),
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
    if amp_boundary is not None:
        boundaries.append((amp_boundary, "amplitude boundary"))
    if predicted_hopf is not None:
        boundaries.append((predicted_hopf, "predicted Hopf"))

    plot_metric_vs_param(
        structural_results,
        metric="time_pr",
        xlabel=r"$\epsilon$",
        ylabel="Time-weighted tangent PR",
        title="FHN final: time-weighted tangent PR",
        savepath=figures_dir / "time_pr_vs_eps.png",
    )
    plot_metric_vs_param(
        structural_results,
        metric="occupancy_gap",
        xlabel=r"$\epsilon$",
        ylabel="Occupancy gap",
        title="FHN final: occupancy gap",
        savepath=figures_dir / "occupancy_gap_vs_eps.png",
    )
    plot_curve_with_boundaries(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        xlabel=r"$\epsilon$",
        ylabel="Composite structural score",
        title="FHN final: structural score with boundaries",
        boundaries=boundaries,
        savepath=figures_dir / "structural_score_vs_eps.png",
    )
    plot_boundary_overlay(
        np.asarray(structural_results["param_values"], dtype=float),
        structural_score,
        np.asarray(oracle_results["metrics"]["tail_amplitude_v"], dtype=float),
        xlabel=r"$\epsilon$",
        structural_ylabel="Structural score",
        oracle_ylabel="Tail amplitude",
        structural_boundary=float(structural_cp.param_value),
        qualitative_boundary=None if amp_boundary is None else float(amp_boundary),
        qualitative_label="amplitude boundary",
        savepath=figures_dir / "boundary_overlay.png",
    )
    if robustness is not None:
        plot_robustness_boundary_summary(
            robustness["summary_rows"],
            savepath=figures_dir / "robustness_boundary_shifts.png",
            show=False,
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
        "forbidden_core_labels": list(contract.forbidden_as_core),
    }

    summary = {
        "system": system.name,
        "summary_version": "v1",
        "benchmark_family": "FitzHugh–Nagumo",
        "sweep_role": "main",
        "control_param": "epsilon",
        "param_values": values,
        "base_params": base_params,
        "amplitude_threshold": float(amplitude_threshold),
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
        "lead_distance_vs_hopf": (
            None
            if lead_vs_hopf is None
            else {
                "structural_param": lead_vs_hopf.structural_param,
                "qualitative_param": lead_vs_hopf.qualitative_param,
                "lead_distance": lead_vs_hopf.lead_distance,
                "structural_ci": lead_vs_hopf.structural_ci,
                "qualitative_ci": lead_vs_hopf.qualitative_ci,
                "details": lead_vs_hopf.details,
            }
        ),
        "oracle": {
            "amplitude_boundary": amp_boundary,
            "predicted_hopf_epsilon": predicted_hopf,
            "tail_amplitude_v": oracle_results["metrics"]["tail_amplitude_v"],
            "tail_period_v": oracle_results["metrics"]["tail_period_v"],
            "is_oscillation": oracle_results["metrics"]["is_oscillation"],
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
            "lead_distance_positive_vs_hopf": (
                None if lead_vs_hopf is None else bool(lead_vs_hopf.lead_distance > 0.0)
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
        description="FHN final pipeline with calibrated main sweep."
    )
    parser.add_argument("--out", type=str, default="outputs/fhn_main")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--eps_min", type=float, default=0.02)
    parser.add_argument("--eps_max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amplitude_threshold", type=float, default=0.2)
    parser.add_argument("--run_robustness", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        outdir=Path(args.out),
        n=args.n,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        seed=args.seed,
        amplitude_threshold=args.amplitude_threshold,
        run_robustness=args.run_robustness,
    )


if __name__ == "__main__":
    main()
