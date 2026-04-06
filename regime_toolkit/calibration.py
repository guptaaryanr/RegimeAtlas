from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class SpecificityComparisonResult:
    benchmark_family: str
    metric_ratios: Dict[str, float]
    oracle_constancy_ok: bool
    max_ratio: float
    ratio_tolerance: float
    passed: bool
    details: Dict[str, Any]



def resolve_sweep_values(
    *,
    param_values: Optional[Sequence[float]] = None,
    n: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    scale: str = "linear",
) -> np.ndarray:
    """
    Resolve a 1D sweep grid from either an explicit list or (min, max, n).

    Explicit values are preferred for final paper sweeps because they let us densify near
    structural and qualitative boundaries without complicating the sweep engine.
    """
    if param_values is not None:
        values = np.asarray(list(param_values), dtype=float)
    else:
        if n is None or min_value is None or max_value is None:
            raise ValueError("either param_values or (n, min_value, max_value) must be provided")
        if int(n) < 2:
            raise ValueError("n must be >= 2")
        if scale == "linear":
            values = np.linspace(float(min_value), float(max_value), int(n), dtype=float)
        elif scale in {"geom", "geometric", "log"}:
            if float(min_value) <= 0.0 or float(max_value) <= 0.0:
                raise ValueError("geometric sweeps require positive min_value and max_value")
            values = np.geomspace(float(min_value), float(max_value), int(n), dtype=float)
        else:
            raise ValueError(f"unknown scale '{scale}'")

    if values.ndim != 1 or values.size < 2:
        raise ValueError("resolved sweep values must be a 1D array with at least two entries")
    diffs = np.diff(values)
    if not np.all(diffs > 0.0):
        raise ValueError("sweep values must be strictly increasing")
    return values



def raw_metric_spans(metrics: Mapping[str, np.ndarray], metric_names: Sequence[str]) -> Dict[str, float]:
    """
    Compute raw peak-to-peak spans for a set of scalar sweep curves.

    Raw spans are what we want for specificity checks. Standardized/fused scores are great
    for boundary finding, but they hide the difference between a tiny nuisance drift and a
    large structural shift.
    """
    spans: Dict[str, float] = {}
    for key in metric_names:
        values = np.asarray(metrics[key], dtype=float)
        finite = values[np.isfinite(values)]
        spans[key] = float(np.ptp(finite)) if finite.size > 0 else float("nan")
    return spans



def flatten_lead_distance(record: Any) -> Optional[float]:
    if record is None:
        return None
    if isinstance(record, Mapping):
        value = record.get("lead_distance")
        return None if value is None else float(value)
    try:
        return float(record)
    except Exception:
        return None



def qualitative_boundary_param(summary: Mapping[str, Any]) -> Optional[float]:
    record = summary.get("primary_qualitative_boundary")
    if record is None:
        return None
    if isinstance(record, Mapping):
        value = record.get("param_value")
        return None if value is None else float(value)
    try:
        return float(record)
    except Exception:
        return None



def qualitative_label_unique_count(summary: Mapping[str, Any]) -> Optional[int]:
    value = summary.get("primary_qualitative_label_unique_count")
    if value is None:
        return None
    return int(value)



def compare_main_vs_nuisance_specificity(
    *,
    benchmark_family: str,
    main_summary: Mapping[str, Any],
    nuisance_summary: Mapping[str, Any],
    metric_names: Sequence[str],
    ratio_tolerance: float = 0.35,
) -> SpecificityComparisonResult:
    """
    Compare nuisance and main sweeps using raw trajectory-only metric spans.

    A nuisance sweep passes specificity if:
    - its raw span stays materially smaller than the main sweep for each core metric
    - it does not trigger a qualitative boundary crossing in the oracle layer
    """
    main_spans = dict(main_summary.get("raw_metric_spans", {}))
    nuisance_spans = dict(nuisance_summary.get("raw_metric_spans", {}))

    ratios: Dict[str, float] = {}
    for key in metric_names:
        main_span = float(main_spans.get(key, np.nan))
        nuisance_span = float(nuisance_spans.get(key, np.nan))
        denom = max(abs(main_span), 1e-12)
        ratios[key] = float(nuisance_span / denom)

    nuisance_boundary = qualitative_boundary_param(nuisance_summary)
    label_unique = qualitative_label_unique_count(nuisance_summary)
    oracle_constancy_ok = (nuisance_boundary is None) and (label_unique is None or label_unique <= 1)

    finite_ratios = [abs(v) for v in ratios.values() if np.isfinite(v)]
    max_ratio = float(max(finite_ratios)) if finite_ratios else float("nan")
    passed = oracle_constancy_ok and (not np.isfinite(max_ratio) or max_ratio <= float(ratio_tolerance))

    return SpecificityComparisonResult(
        benchmark_family=str(benchmark_family),
        metric_ratios=ratios,
        oracle_constancy_ok=bool(oracle_constancy_ok),
        max_ratio=max_ratio,
        ratio_tolerance=float(ratio_tolerance),
        passed=bool(passed),
        details={
            "main_control_param": main_summary.get("control_param"),
            "nuisance_control_param": nuisance_summary.get("control_param"),
            "nuisance_primary_qualitative_boundary": nuisance_boundary,
            "nuisance_label_unique_count": label_unique,
        },
    )


def admissible_min_segment_sizes(n_samples: int, candidates: Sequence[int]) -> tuple[int, ...]:
    """Filter segment sizes so piecewise change-point fits stay admissible."""
    n = int(n_samples)
    valid = sorted({int(s) for s in candidates if int(s) >= 1 and 2 * int(s) <= n})
    if not valid:
        return (1,)
    return tuple(valid)


def default_min_segment_size(n_samples: int, preferred: int = 2) -> int:
    valid = admissible_min_segment_sizes(int(n_samples), (1, int(preferred), max(1, int(n_samples) // 6)))
    return int(max(valid))
