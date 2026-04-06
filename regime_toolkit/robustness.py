from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import numpy as np
from scipy.stats import spearmanr

from .simulate import SimulationConfig
from .sweep import IndicatorSpec, parameter_sweep
from .systems.base import ODESystem, Params
from .io import save_results_json
from .boundaries import (
    composite_structural_score,
    piecewise_linear_change_point,
    bootstrap_change_point_from_replicates,
)
from .calibration import default_min_segment_size


CORE_TIER = "core"
SUPPLEMENTAL_TIER = "supplemental"
STRESS_TIER = "stress"


@dataclass(frozen=True)
class RobustnessCase:
    """
    Perturbation spec for repeatable robustness runs.

    Final adds solver and replicate controls so robustness can address the most common
    criticism routes: finite-data instability, observation perturbation, and solver dependence.
    """

    name: str
    description: str = ""
    representation: str = "full_state"
    observation_index: int = 0
    observation_noise_sigma: float = 0.0
    observation_noise_relative: bool = False
    embedding_dim: int = 3
    delay: int = 10
    stride: int = 1
    dt: Optional[float] = None
    t_final: Optional[float] = None
    transient: Optional[float] = None
    method: Optional[str] = None
    max_step: Optional[float] = None
    n_replicates: int = 1
    seed_stride: int = 1000
    tier: str = CORE_TIER


def apply_robustness_case(
    base_config: SimulationConfig,
    case: RobustnessCase,
) -> SimulationConfig:
    return replace(
        base_config,
        dt=base_config.dt if case.dt is None else float(case.dt),
        t_final=base_config.t_final if case.t_final is None else float(case.t_final),
        transient=(
            base_config.transient if case.transient is None else float(case.transient)
        ),
        method=base_config.method if case.method is None else str(case.method),
        max_step=(
            base_config.max_step if case.max_step is None else float(case.max_step)
        ),
    )


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    corr = spearmanr(x, y).correlation
    return float(corr) if np.isfinite(corr) else np.nan


def summarize_metric_rank_correlation(
    base_results: Dict[str, Any],
    case_results: Dict[str, Any],
    metrics: Sequence[str],
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key in metrics:
        x = np.asarray(base_results["metrics"][key], dtype=float)
        y = np.asarray(case_results["metrics"][key], dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"metric '{key}' has mismatched shapes {x.shape} and {y.shape}"
            )
        summary[key] = _finite_spearman(x, y)
    return summary


def summarize_metric_rank_correlation_aligned(
    base_results: Dict[str, Any],
    case_results: Dict[str, Any],
    metrics: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key in metrics:
        x = np.asarray(base_results["metrics"][key], dtype=float)
        y = np.asarray(case_results["metrics"][key], dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"metric '{key}' has mismatched shapes {x.shape} and {y.shape}"
            )
        corr_same = _finite_spearman(x, y)
        corr_flip = _finite_spearman(x, -y)
        if np.isnan(corr_same) and np.isnan(corr_flip):
            best_corr = np.nan
            best_sign = 1.0
        elif np.isnan(corr_flip) or (
            not np.isnan(corr_same) and corr_same >= corr_flip
        ):
            best_corr = corr_same
            best_sign = 1.0
        else:
            best_corr = corr_flip
            best_sign = -1.0
        summary[key] = {
            "aligned_spearman": float(best_corr),
            "applied_sign": float(best_sign),
            "raw_spearman": float(corr_same),
        }
    return summary


def _aligned_results_for_boundary(
    *,
    baseline_results: Dict[str, Any],
    case_results: Dict[str, Any],
    boundary_metric_directions: Mapping[str, int | str],
) -> Dict[str, Any]:
    """
    Return a shallow copy of case_results where boundary-driving metrics are sign-aligned
    to the baseline using the same logic as the aligned Spearman report.

    Why this exists:
    the supplemental scalar-delay cases can preserve the same structural story with an
    overall sign flip in one metric. The robustness report already detects that via
    alignment_sign_*, but boundary extraction must use the same aligned orientation.
    """
    metric_keys = list(boundary_metric_directions.keys())
    aligned = summarize_metric_rank_correlation_aligned(
        baseline_results,
        case_results,
        metric_keys,
    )

    metrics_copy = {
        key: np.asarray(value, dtype=float).copy()
        for key, value in case_results["metrics"].items()
    }

    for key in metric_keys:
        sign = float(aligned[key]["applied_sign"])
        metrics_copy[key] = sign * np.asarray(metrics_copy[key], dtype=float)

    out = dict(case_results)
    out["metrics"] = metrics_copy
    return out


def _aligned_spearman_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Return:
        aligned_rho, applied_sign, raw_rho

    aligned_rho is the better of rho(x, y) and rho(x, -y).
    """
    raw = _finite_spearman(x, y)
    flipped = _finite_spearman(x, -y)

    if np.isnan(flipped) or (not np.isnan(raw) and raw >= flipped):
        return float(raw), 1.0, float(raw)
    return float(flipped), -1.0, float(raw)


def _mean_results(results_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(results_list) == 0:
        raise ValueError("at least one result is required")
    metric_keys = list(results_list[0]["metrics"].keys())
    mean_metrics: Dict[str, np.ndarray] = {}
    for key in metric_keys:
        curves = [np.asarray(res["metrics"][key], dtype=float) for res in results_list]
        mean_metrics[key] = np.mean(np.vstack(curves), axis=0)
    return {
        "param_values": np.asarray(results_list[0]["param_values"], dtype=float),
        "metrics": mean_metrics,
    }


def _structural_score_from_results(
    results: Dict[str, Any],
    *,
    boundary_metric_directions: Mapping[str, int | str],
    boundary_weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    score, _ = composite_structural_score(
        np.asarray(results["param_values"], dtype=float),
        results["metrics"],
        boundary_metric_directions,
        weights=boundary_weights,
        smooth_window=3,
    )
    return np.asarray(score, dtype=float)


def evaluate_robustness_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    boundary_shift_tolerance: float = 0.08,
    min_aligned_spearman: float = 0.75,
    max_boundary_ci_width: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Turn robustness rows into an explicit pass/fail report.

    Core (or primary) cases are judged on:
      - boundary shift
      - CI width
      - aligned Spearman of the composite structural score

    Supplemental cases still report per-metric alignment, but they do not gate the
    core contract.
    """
    per_case = []
    core_passed = True
    supplemental_passed = True
    stress_passed = True
    all_cases_passed = True
    has_core = False
    has_supplemental = False
    has_stress = False

    for row in summary_rows:
        boundary_ok = abs(float(row["boundary_shift"])) <= float(
            boundary_shift_tolerance
        )

        if (
            max_boundary_ci_width is not None
            and row.get("boundary_ci_width") is not None
        ):
            boundary_ok = boundary_ok and float(row["boundary_ci_width"]) <= float(
                max_boundary_ci_width
            )

        tier = str(row.get("tier", "core"))

        if tier in {"core", "primary"}:
            has_core = True
            rho = float(row.get("aligned_spearman_structural_score", np.nan))
            alignment_ok = np.isfinite(rho) and rho >= float(min_aligned_spearman)

        elif tier == SUPPLEMENTAL_TIER:
            has_supplemental = True
            rho = float(row.get("aligned_spearman_time_pr", np.nan))
            alignment_ok = np.isfinite(rho) and rho >= float(min_aligned_spearman)

        elif tier == STRESS_TIER:
            has_stress = True
            rho = float(row.get("aligned_spearman_time_pr", np.nan))
            alignment_ok = np.isfinite(rho) and rho >= float(min_aligned_spearman)

        else:
            raise ValueError(f"unknown robustness tier '{tier}'")

        case_passed = boundary_ok and alignment_ok
        all_cases_passed = all_cases_passed and case_passed

        if tier in {"core", "primary"}:
            core_passed = core_passed and case_passed
        elif tier == SUPPLEMENTAL_TIER:
            supplemental_passed = supplemental_passed and case_passed
        else:
            stress_passed = stress_passed and case_passed

        per_case.append(
            {
                "case": row["case"],
                "tier": tier,
                "boundary_ok": boundary_ok,
                "alignment_ok": alignment_ok,
                "passed": case_passed,
            }
        )

    return {
        "boundary_shift_tolerance": float(boundary_shift_tolerance),
        "min_aligned_spearman": float(min_aligned_spearman),
        "max_boundary_ci_width": (
            None if max_boundary_ci_width is None else float(max_boundary_ci_width)
        ),
        "per_case": per_case,
        # keep both names so older downstream code does not break
        "core_passed": bool(core_passed),
        "primary_passed": bool(core_passed),
        "supplemental_passed": bool(supplemental_passed),
        "all_cases_passed": bool(all_cases_passed),
        "stress_passed": bool(stress_passed),
        "passed": bool(
            core_passed
            if has_core and not has_supplemental and not has_stress
            else (
                supplemental_passed
                if has_supplemental and not has_core and not has_stress
                else (
                    stress_passed
                    if has_stress and not has_core and not has_supplemental
                    else (core_passed and supplemental_passed and stress_passed)
                )
            )
        ),
    }


def run_robustness_cases(
    *,
    system: ODESystem,
    base_params: Params,
    control_param: str,
    values: Sequence[float],
    base_sim_config: SimulationConfig,
    cases: Sequence[RobustnessCase],
    indicator_factory: Callable[[RobustnessCase, int], Sequence[IndicatorSpec]],
    boundary_metric_directions: Mapping[str, int | str],
    boundary_weights: Optional[Mapping[str, float]] = None,
    seed: int = 0,
    precomputed_baseline: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    save_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if len(cases) == 0:
        raise ValueError("at least one robustness case is required")

    per_case_results: Dict[str, Dict[str, Any]] = {}
    per_case_replicates: Dict[str, list[Dict[str, Any]]] = {}
    per_case_boundary: Dict[str, Dict[str, Any]] = {}

    for idx, case in enumerate(cases):
        sim_config = apply_robustness_case(base_sim_config, case)
        case_outdir = None if save_dir is None else Path(save_dir) / case.name
        if verbose:
            print(f"[robustness] case={case.name}")

        replicate_results = []
        replicate_scores = []

        use_precomputed_baseline = (
            case.name == "baseline"
            and precomputed_baseline is not None
            and int(case.n_replicates) == 1
        )

        if use_precomputed_baseline:
            replicate_results = [precomputed_baseline]
            score, _ = composite_structural_score(
                np.asarray(precomputed_baseline["param_values"], dtype=float),
                precomputed_baseline["metrics"],
                boundary_metric_directions,
                weights=boundary_weights,
                smooth_window=3,
            )
            replicate_scores = [score]
        else:
            for rep in range(max(1, int(case.n_replicates))):
                rep_seed = int(seed + idx + rep * max(1, int(case.seed_stride)))
                indicators = list(indicator_factory(case, rep_seed))
                rep_outdir = (
                    None
                    if case_outdir is None
                    else case_outdir / f"replicate_{rep:02d}"
                )
                results = parameter_sweep(
                    system=system,
                    base_params=base_params,
                    control_param=control_param,
                    values=values,
                    sim_config=sim_config,
                    indicators=indicators,
                    seed=rep_seed,
                    store_trajectories_at=None,
                    verbose=verbose,
                    save_dir=rep_outdir,
                    save_config=(
                        {
                            "experiment": "robustness_case",
                            "case": case.__dict__,
                            "replicate_index": rep,
                        }
                        if rep_outdir is not None
                        else None
                    ),
                )
                replicate_results.append(results)
                score, _ = composite_structural_score(
                    np.asarray(results["param_values"], dtype=float),
                    results["metrics"],
                    boundary_metric_directions,
                    weights=boundary_weights,
                    smooth_window=3,
                )
                replicate_scores.append(score)

        aggregate_results = _mean_results(replicate_results)
        per_case_results[case.name] = aggregate_results
        per_case_replicates[case.name] = replicate_results

        score_curves = np.vstack(replicate_scores)
        boot = bootstrap_change_point_from_replicates(
            np.asarray(values, dtype=float),
            score_curves,
            n_bootstrap=64,
            min_segment_size=default_min_segment_size(len(values), preferred=2),
            smooth_window=1,
            seed=seed + idx,
        )
        per_case_boundary[case.name] = {
            "param_value": float(boot.estimate.param_value),
            "index": int(boot.estimate.index),
            "score": float(boot.estimate.score),
            "left_slope": float(boot.estimate.left_slope),
            "right_slope": float(boot.estimate.right_slope),
            "ci_low": None if boot.ci_low is None else float(boot.ci_low),
            "ci_high": None if boot.ci_high is None else float(boot.ci_high),
            "replicate_param_values": [
                float(v)
                for v in np.asarray(
                    [
                        piecewise_linear_change_point(
                            np.asarray(values, dtype=float),
                            s,
                            min_segment_size=default_min_segment_size(
                                len(values), preferred=2
                            ),
                            smooth_window=1,
                        ).param_value
                        for s in score_curves
                    ],
                    dtype=float,
                )
            ],
            "n_replicates": int(len(replicate_results)),
        }

        if case_outdir is not None:
            case_outdir.mkdir(parents=True, exist_ok=True)
            save_results_json(
                {
                    "case": case.__dict__,
                    "aggregate_boundary": per_case_boundary[case.name],
                },
                case_outdir / "case_summary.json",
            )

    baseline_name = (
        "baseline"
        if "baseline" in per_case_results
        else next(iter(per_case_results.keys()))
    )
    baseline = per_case_results[baseline_name]
    boundary_baseline = per_case_boundary[baseline_name]["param_value"]

    # Recompute supplemental/stress boundaries using sign-aligned metrics.
    # This keeps the boundary extraction consistent with the aligned Spearman logic
    # already used in the robustness report.
    for case in cases:
        if case.name == baseline_name:
            continue
        if case.tier not in {SUPPLEMENTAL_TIER, STRESS_TIER, "supplemental", "stress"}:
            continue

        aligned_results = _aligned_results_for_boundary(
            baseline_results=baseline,
            case_results=per_case_results[case.name],
            boundary_metric_directions=boundary_metric_directions,
        )

        score = _structural_score_from_results(
            aligned_results,
            boundary_metric_directions=boundary_metric_directions,
            boundary_weights=boundary_weights,
        )
        cp = piecewise_linear_change_point(
            np.asarray(aligned_results["param_values"], dtype=float),
            score,
            min_segment_size=max(2, len(aligned_results["param_values"]) // 6),
            smooth_window=1,
        )

        replicate_scores = []
        for rep in per_case_replicates[case.name]:
            aligned_rep = _aligned_results_for_boundary(
                baseline_results=baseline,
                case_results=rep,
                boundary_metric_directions=boundary_metric_directions,
            )
            replicate_scores.append(
                _structural_score_from_results(
                    aligned_rep,
                    boundary_metric_directions=boundary_metric_directions,
                    boundary_weights=boundary_weights,
                )
            )

        replicate_scores = np.stack(replicate_scores, axis=0)
        cp_boot = bootstrap_change_point_from_replicates(
            np.asarray(aligned_results["param_values"], dtype=float),
            replicate_scores,
            n_bootstrap=100,
            min_segment_size=max(2, len(aligned_results["param_values"]) // 6),
            smooth_window=1,
            seed=seed + hash(case.name) % 997,
        )

        per_case_boundary[case.name] = {
            "param_value": float(cp.param_value),
            "index": int(cp.index),
            "score": float(cp.score),
            "left_slope": float(cp.left_slope),
            "right_slope": float(cp.right_slope),
            "ci_low": None if cp_boot.ci_low is None else float(cp_boot.ci_low),
            "ci_high": None if cp_boot.ci_high is None else float(cp_boot.ci_high),
            "n_replicates": int(len(per_case_replicates[case.name])),
        }

    baseline_structural_score, _ = composite_structural_score(
        np.asarray(baseline["param_values"], dtype=float),
        baseline["metrics"],
        boundary_metric_directions,
        weights=boundary_weights,
        smooth_window=3,
    )

    metric_keys = list(boundary_metric_directions.keys())
    summary_rows = []
    for case in cases:
        results = per_case_results[case.name]
        case_structural_score, _ = composite_structural_score(
            np.asarray(results["param_values"], dtype=float),
            results["metrics"],
            boundary_metric_directions,
            weights=boundary_weights,
            smooth_window=3,
        )
        aligned_score_rho, aligned_score_sign, raw_score_rho = _aligned_spearman_pair(
            baseline_structural_score,
            case_structural_score,
        )
        correlations = summarize_metric_rank_correlation(baseline, results, metric_keys)
        aligned = summarize_metric_rank_correlation_aligned(
            baseline, results, metric_keys
        )
        boundary = per_case_boundary[case.name]["param_value"]
        ci_low = per_case_boundary[case.name].get("ci_low")
        ci_high = per_case_boundary[case.name].get("ci_high")
        ci_width = None
        if ci_low is not None and ci_high is not None:
            ci_width = float(ci_high) - float(ci_low)
        summary_rows.append(
            {
                "case": case.name,
                "tier": case.tier,
                "description": case.description,
                "boundary_param": float(boundary),
                "boundary_shift": float(boundary - boundary_baseline),
                "boundary_ci_low": ci_low,
                "boundary_ci_high": ci_high,
                "boundary_ci_width": ci_width,
                "n_replicates": int(case.n_replicates),
                "spearman_structural_score": float(raw_score_rho),
                "aligned_spearman_structural_score": float(aligned_score_rho),
                "alignment_sign_structural_score": float(aligned_score_sign),
                **{f"spearman_{k}": float(v) for k, v in correlations.items()},
                **{
                    f"aligned_spearman_{k}": float(v["aligned_spearman"])
                    for k, v in aligned.items()
                },
                **{
                    f"alignment_sign_{k}": float(v["applied_sign"])
                    for k, v in aligned.items()
                },
            }
        )

    core_rows = [row for row in summary_rows if row.get("tier", CORE_TIER) == CORE_TIER]
    supplemental_rows = [
        row for row in summary_rows if row.get("tier") == SUPPLEMENTAL_TIER
    ]
    stress_rows = [row for row in summary_rows if row.get("tier") == STRESS_TIER]

    acceptance_core = (
        evaluate_robustness_summary(
            core_rows,
            boundary_shift_tolerance=0.08,
            min_aligned_spearman=0.75,
            max_boundary_ci_width=0.25,
        )
        if core_rows
        else None
    )

    acceptance_supplemental = (
        evaluate_robustness_summary(
            supplemental_rows,
            boundary_shift_tolerance=0.12,
            min_aligned_spearman=0.60,
            max_boundary_ci_width=0.25,
        )
        if supplemental_rows
        else None
    )

    acceptance_stress = (
        evaluate_robustness_summary(
            stress_rows,
            boundary_shift_tolerance=0.20,
            min_aligned_spearman=0.35,
            max_boundary_ci_width=0.45,
        )
        if stress_rows
        else None
    )

    acceptance_all = {
        "core_passed": (
            None if acceptance_core is None else acceptance_core["core_passed"]
        ),
        "supplemental_passed": (
            None
            if acceptance_supplemental is None
            else acceptance_supplemental["supplemental_passed"]
        ),
        "stress_passed": (
            None if acceptance_stress is None else acceptance_stress["stress_passed"]
        ),
        "passed": bool(
            (True if acceptance_core is None else acceptance_core["core_passed"])
            and (
                True
                if acceptance_supplemental is None
                else acceptance_supplemental["supplemental_passed"]
            )
            and (
                True
                if acceptance_stress is None
                else acceptance_stress["stress_passed"]
            )
        ),
    }

    payload = {
        "baseline_case": baseline_name,
        "cases": [case.__dict__ for case in cases],
        "case_boundaries": per_case_boundary,
        "summary_rows": summary_rows,
        "acceptance": acceptance_all,
        "acceptance_core": acceptance_core,
        "acceptance_supplemental": acceptance_supplemental,
        "acceptance_stress": acceptance_stress,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_results_json(payload, save_dir / "robustness_summary.json")

    return payload
