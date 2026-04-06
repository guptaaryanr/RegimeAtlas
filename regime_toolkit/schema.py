from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


_RUN_SPECS = {
    "fhn_main": {
        "experiment": "fhn_pipeline",
        "benchmark_family": "FitzHugh–Nagumo",
        "sweep_role": "main",
    },
    "fhn_nuisance": {
        "experiment": "fhn_nuisance_pipeline",
        "benchmark_family": "FitzHugh–Nagumo",
        "sweep_role": "nuisance",
    },
    "vdp_main": {
        "experiment": "vdp_pipeline",
        "benchmark_family": "Autonomous forced van der Pol",
        "sweep_role": "main",
    },
    "vdp_nuisance": {
        "experiment": "vdp_nuisance_pipeline",
        "benchmark_family": "Autonomous forced van der Pol",
        "sweep_role": "nuisance",
    },
}

_REQUIRED_RUN_KEYS = {"run_name", "experiment", "benchmark_family", "sweep_role"}
_ALLOWED_SWEEP_ROLES = {"main", "nuisance"}



def _validate_runs(runs: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    normalized = []
    seen_names = set()
    seen_family_role = set()
    for idx, run in enumerate(runs):
        record = dict(run)
        missing = sorted(_REQUIRED_RUN_KEYS - set(record.keys()))
        if missing:
            raise ValueError(f"run index {idx} is missing required keys: {missing}")

        run_name = str(record["run_name"])
        sweep_role = str(record["sweep_role"])
        benchmark_family = str(record["benchmark_family"])

        if not run_name:
            raise ValueError(f"run index {idx} has empty run_name")
        if run_name in seen_names:
            raise ValueError(f"duplicate run_name '{run_name}' in study config")
        if sweep_role not in _ALLOWED_SWEEP_ROLES:
            raise ValueError(
                f"run '{run_name}' has invalid sweep_role '{sweep_role}'. Expected one of {sorted(_ALLOWED_SWEEP_ROLES)}"
            )
        family_role = (benchmark_family, sweep_role)
        if family_role in seen_family_role:
            raise ValueError(
                f"duplicate benchmark_family/sweep_role combination {family_role} in study config"
            )

        seen_names.add(run_name)
        seen_family_role.add(family_role)
        normalized.append(record)

    if not normalized:
        raise ValueError("study config contains no runs")
    return normalized



def _normalize_benchmark_suite(config: Mapping[str, Any]) -> Dict[str, Any]:
    suite_cfg = dict(config.get("config", {}))
    runs = []
    for key, spec in _RUN_SPECS.items():
        if key not in suite_cfg:
            raise ValueError(f"benchmark suite config is missing '{key}'")
        payload = dict(spec)
        payload["run_name"] = key
        payload.update(dict(suite_cfg[key]))
        runs.append(payload)

    study_name = str(config.get("study_name", config.get("benchmark_suite_name", config.get("experiment", "benchmark_suite"))))
    outdir = str(config.get("outdir", "outputs/benchmark_suite"))
    return {
        "study_name": study_name,
        "outdir": outdir,
        "runs": _validate_runs(runs),
        "source_format": "benchmark_suite",
    }



def normalize_study_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize supported study config formats into the canonical shape.

    Supported inputs:
    - Canonical study config with keys: study_name, outdir, runs
    - Legacy benchmark-suite config with keys: experiment, outdir, config
    """
    if "runs" in config:
        runs = _validate_runs(config.get("runs", []))
        return {
            "study_name": str(config.get("study_name", "paper_suite")),
            "outdir": str(config.get("outdir", "outputs/paper_suite")),
            "runs": runs,
            "source_format": "study_suite",
        }

    experiment_name = str(config.get("experiment", ""))
    if experiment_name.endswith("benchmark_suite") and "config" in config:
        return _normalize_benchmark_suite(config)

    raise ValueError(
        "Unsupported study config format. Expected canonical {study_name, outdir, runs} or legacy benchmark-suite {experiment, outdir, config}."
    )
