from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np

from .boundaries import (
    composite_structural_score,
    piecewise_linear_change_point,
    change_point_sensitivity_scan,
    lead_distance,
)
from .calibration import admissible_min_segment_sizes, default_min_segment_size


@dataclass(frozen=True)
class AblationVariantResult:
    name: str
    metric_names: tuple[str, ...]
    weights: Dict[str, float]
    boundary_param: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    lead_distance: Optional[float]
    details: Dict[str, Any]


@dataclass(frozen=True)
class AblationSuiteResult:
    variants: Dict[str, AblationVariantResult]
    passed: bool
    details: Dict[str, Any]



def _variant_from_metric_subset(
    *,
    name: str,
    param_values: np.ndarray,
    metrics: Mapping[str, np.ndarray],
    metric_directions: Mapping[str, int | str],
    subset: Sequence[str],
    weights: Optional[Mapping[str, float]],
    qualitative_boundary: Optional[float],
) -> AblationVariantResult:
    directions = {key: metric_directions[key] for key in subset}
    local_weights = {key: 1.0 for key in subset}
    if weights is not None:
        for key in subset:
            local_weights[key] = float(weights.get(key, 1.0))

    score, _ = composite_structural_score(
        np.asarray(param_values, dtype=float),
        metrics,
        directions,
        weights=local_weights,
        smooth_window=3,
    )
    cp = piecewise_linear_change_point(
        np.asarray(param_values, dtype=float),
        score,
        min_segment_size=default_min_segment_size(len(param_values), preferred=2),
        smooth_window=1,
    )
    sensitivity = change_point_sensitivity_scan(
        np.asarray(param_values, dtype=float),
        score,
        smooth_windows=(1, 3, 5),
        min_segment_sizes=admissible_min_segment_sizes(len(param_values), (1, 2, 3, max(1, len(param_values) // 6))),
    )
    ld = None
    if qualitative_boundary is not None:
        ld = float(lead_distance(cp.param_value, qualitative_boundary).lead_distance)
    return AblationVariantResult(
        name=name,
        metric_names=tuple(subset),
        weights=local_weights,
        boundary_param=float(cp.param_value),
        ci_low=None if sensitivity.ci_low is None else float(sensitivity.ci_low),
        ci_high=None if sensitivity.ci_high is None else float(sensitivity.ci_high),
        lead_distance=ld,
        details={
            "boundary_score": float(cp.score),
            "median_param": sensitivity.median_param,
            "candidate_param_values": sensitivity.candidate_param_values.tolist(),
        },
    )



def evaluate_structural_ablation_suite(
    *,
    param_values: Sequence[float],
    metrics: Mapping[str, np.ndarray],
    metric_directions: Mapping[str, int | str],
    weights: Optional[Mapping[str, float]] = None,
    qualitative_boundary: Optional[float] = None,
    tolerance: float = 0.05,
) -> AblationSuiteResult:
    """
    Compare the full atlas against simpler single-metric variants.

    This is not meant to prove universal dominance. It is a criticism-hardening check that
    the paper's full atlas is not materially worse than its constituent trajectory-only
    metrics while preserving a positive lead distance when a qualitative boundary exists.
    """
    x = np.asarray(param_values, dtype=float)
    metric_names = tuple(metric_directions.keys())

    variants: Dict[str, AblationVariantResult] = {}
    variants["atlas_full"] = _variant_from_metric_subset(
        name="atlas_full",
        param_values=x,
        metrics=metrics,
        metric_directions=metric_directions,
        subset=metric_names,
        weights=weights,
        qualitative_boundary=qualitative_boundary,
    )
    for key in metric_names:
        variants[f"single_{key}"] = _variant_from_metric_subset(
            name=f"single_{key}",
            param_values=x,
            metrics=metrics,
            metric_directions=metric_directions,
            subset=(key,),
            weights={key: 1.0},
            qualitative_boundary=qualitative_boundary,
        )

    full = variants["atlas_full"]
    single_leads = [v.lead_distance for k, v in variants.items() if k.startswith("single_") and v.lead_distance is not None]
    single_boundaries = [v.boundary_param for k, v in variants.items() if k.startswith("single_")]

    lead_positive = True if qualitative_boundary is None else (full.lead_distance is not None and float(full.lead_distance) > 0.0)
    competitive_with_singles = True
    if single_leads and full.lead_distance is not None:
        competitive_with_singles = float(full.lead_distance) >= (float(np.median(single_leads)) - float(tolerance))

    boundary_band_ok = True
    if single_boundaries:
        lo = min(single_boundaries) - float(tolerance)
        hi = max(single_boundaries) + float(tolerance)
        boundary_band_ok = lo <= float(full.boundary_param) <= hi

    passed = bool(lead_positive and competitive_with_singles and boundary_band_ok)
    return AblationSuiteResult(
        variants=variants,
        passed=passed,
        details={
            "qualitative_boundary": qualitative_boundary,
            "lead_positive": bool(lead_positive),
            "competitive_with_singles": bool(competitive_with_singles),
            "boundary_band_ok": bool(boundary_band_ok),
            "tolerance": float(tolerance),
        },
    )
