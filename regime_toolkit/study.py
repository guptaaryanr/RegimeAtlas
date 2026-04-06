from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
import csv

from .contracts import (
    PaperStudyContract,
    default_paper_study_contract,
    evaluate_study_contract,
)
from .io import save_json
from .provenance import capture_provenance
from .simulate import SimulationConfig
from .calibration import (
    flatten_lead_distance,
    qualitative_boundary_param,
    compare_main_vs_nuisance_specificity,
)
from .integrity import validate_manifest_file, write_manifest


def _acceptance_value(
    summary: Mapping[str, Any], direct_key: str, nested_key: str, inside_key: str
):
    if summary.get(direct_key) is not None:
        return summary.get(direct_key)
    block = summary.get(nested_key)
    if isinstance(block, Mapping):
        if inside_key in block:
            return block.get(inside_key)
        if "passed" in block:
            return block.get("passed")
    return None


def extract_summary_record(
    *,
    run_name: str,
    experiment: str,
    benchmark_family: str,
    sweep_role: str,
    output_dir: str | Path,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    structural_boundary = summary.get("structural_boundary")
    if isinstance(structural_boundary, Mapping):
        structural_boundary_param = structural_boundary.get("param_value")
    else:
        structural_boundary_param = structural_boundary

    raw_spans = dict(summary.get("raw_metric_spans", {}))
    record = {
        "run_name": run_name,
        "experiment": experiment,
        "benchmark_family": benchmark_family,
        "sweep_role": sweep_role,
        "system": summary.get("system"),
        "summary_version": summary.get("summary_version"),
        "control_param": summary.get("control_param"),
        "output_dir": str(output_dir),
        "structural_boundary_param": structural_boundary_param,
        "primary_qualitative_boundary_param": qualitative_boundary_param(summary),
        "primary_qualitative_boundary_kind": summary.get(
            "primary_qualitative_boundary_kind"
        ),
        "lead_distance_primary": flatten_lead_distance(
            summary.get("lead_distance_primary")
        ),
        "lead_distance_vs_hopf": flatten_lead_distance(
            summary.get("lead_distance_vs_hopf")
        ),
        "robustness_core_passed": _acceptance_value(
            summary,
            "robustness_core_passed",
            "robustness_acceptance_primary",
            "core_passed",
        ),
        "robustness_supplemental_passed": _acceptance_value(
            summary,
            "robustness_supplemental_passed",
            "robustness_acceptance_supplemental",
            "supplemental_passed",
        ),
        "robustness_stress_passed": _acceptance_value(
            summary,
            "robustness_stress_passed",
            "robustness_acceptance_stress",
            "stress_passed",
        ),
        "ablation_passed": (
            None
            if not isinstance(
                summary.get("ablation_report", summary.get("ablation")), Mapping
            )
            else bool(
                summary.get("ablation_report", summary.get("ablation")).get("passed")
            )
        ),
        "core_is_trajectory_only": (
            None
            if not isinstance(summary.get("claim_compliance", {}), Mapping)
            else summary.get("claim_compliance", {}).get("core_is_trajectory_only")
        ),
        "oracle_saved_separately": (
            None
            if not isinstance(summary.get("claim_compliance", {}), Mapping)
            else summary.get("claim_compliance", {}).get("oracle_saved_separately")
        ),
        "primary_qualitative_label_unique_count": summary.get(
            "primary_qualitative_label_unique_count"
        ),
        "n_param_values": len(summary.get("param_values", [])),
        "core_is_trajectory_only": (
            None
            if not isinstance(summary.get("claim_compliance"), Mapping)
            else summary["claim_compliance"].get("core_is_trajectory_only")
        ),
    }
    for key, value in raw_spans.items():
        record[f"raw_span_{key}"] = value
    return record


def save_summary_csv(
    records: Sequence[Mapping[str, Any]], filepath: str | Path
) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        filepath.write_text("", encoding="utf-8")
        return filepath

    fieldnames = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with filepath.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return filepath


def build_specificity_report(
    *,
    summaries_by_run: Mapping[str, Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    contract: PaperStudyContract,
) -> Dict[str, Any]:
    by_family_role: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for run_cfg in runs:
        run_name = str(run_cfg["run_name"])
        family = str(run_cfg["benchmark_family"])
        role = str(run_cfg["sweep_role"])
        if run_name in summaries_by_run:
            by_family_role[(family, role)] = summaries_by_run[run_name]

    comparisons = {}
    passed = True
    for family in contract.required_benchmark_families:
        main_summary = by_family_role.get((family, "main"))
        nuisance_summary = by_family_role.get((family, "nuisance"))
        if main_summary is None or nuisance_summary is None:
            comparisons[family] = {
                "passed": False,
                "reason": "missing main or nuisance summary",
            }
            passed = False
            continue

        result = compare_main_vs_nuisance_specificity(
            benchmark_family=family,
            main_summary=main_summary,
            nuisance_summary=nuisance_summary,
            metric_names=contract.default_structural_metrics,
        )
        comparisons[family] = {
            "benchmark_family": result.benchmark_family,
            "metric_ratios": result.metric_ratios,
            "oracle_constancy_ok": result.oracle_constancy_ok,
            "max_ratio": result.max_ratio,
            "ratio_tolerance": result.ratio_tolerance,
            "passed": result.passed,
            "details": result.details,
        }
        passed = passed and result.passed

    return {
        "comparisons": comparisons,
        "passed": bool(passed),
    }


def build_ablation_report(
    *,
    summaries_by_run: Mapping[str, Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    runs_report: Dict[str, Any] = {}
    passed = True
    for run_cfg in runs:
        if str(run_cfg.get("sweep_role")) != "main":
            continue
        run_name = str(run_cfg["run_name"])
        summary = summaries_by_run.get(run_name, {})
        ablation = summary.get("ablation")
        ablation_passed = (
            bool(ablation.get("passed")) if isinstance(ablation, Mapping) else False
        )
        if not isinstance(ablation, Mapping):
            passed = False
            runs_report[run_name] = {
                "passed": False,
                "reason": "missing ablation summary",
            }
            continue
        runs_report[run_name] = {
            "passed": ablation_passed,
            "details": dict(ablation.get("details", {})),
            "variant_names": sorted(list(ablation.get("variants", {}).keys())),
        }
        passed = passed and ablation_passed
    return {
        "runs": runs_report,
        "passed": bool(passed),
    }


def build_integrity_report(
    *,
    outdir: str | Path,
    executed_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    outdir = Path(outdir)
    run_reports: Dict[str, Any] = {}
    passed = True
    for run in executed_runs:
        run_name = str(run["run_name"])
        run_dir = Path(run["output_dir"])
        manifest_paths = sorted(run_dir.rglob("manifest.json"))
        manifests = []
        run_ok = True
        for path in manifest_paths:
            report = validate_manifest_file(path)
            manifests.append(
                {
                    "manifest": str(path.relative_to(run_dir)),
                    "passed": report["passed"],
                    "n_files": report["n_files"],
                }
            )
            run_ok = run_ok and bool(report["passed"])
        run_reports[run_name] = {
            "n_manifests": len(manifest_paths),
            "manifests": manifests,
            "passed": bool(run_ok),
        }
        passed = passed and bool(run_ok) and len(manifest_paths) > 0

    study_manifest_path = outdir / "study_manifest.json"
    study_manifest_report = None
    if study_manifest_path.exists():
        study_manifest_report = validate_manifest_file(study_manifest_path)
        passed = passed and bool(study_manifest_report["passed"])

    study_manifest_paths = sorted(outdir.glob("*.json"))
    return {
        "runs": run_reports,
        "n_study_json_files": len(study_manifest_paths),
        "study_manifest": {
            "present": study_manifest_path.exists(),
            "report": study_manifest_report,
        },
        "passed": bool(passed),
    }


def run_study_suite(
    *,
    study_name: str,
    outdir: str | Path,
    runs: Sequence[Mapping[str, Any]],
    runner_map: Mapping[str, Callable[..., Mapping[str, Any]]],
    contract: Optional[PaperStudyContract] = None,
) -> Dict[str, Any]:
    contract = default_paper_study_contract() if contract is None else contract
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_records = []
    executed_runs = []
    summaries_by_run: Dict[str, Mapping[str, Any]] = {}

    for run_cfg in runs:
        experiment = str(run_cfg["experiment"])
        run_name = str(run_cfg["run_name"])
        benchmark_family = str(run_cfg["benchmark_family"])
        sweep_role = str(run_cfg["sweep_role"])
        if experiment not in runner_map:
            raise ValueError(f"unsupported experiment '{experiment}' in study suite")

        runner = runner_map[experiment]
        run_outdir = outdir / run_name
        kwargs = dict(run_cfg)
        kwargs.pop("experiment", None)
        kwargs.pop("run_name", None)
        kwargs.pop("benchmark_family", None)
        kwargs.pop("sweep_role", None)
        kwargs["outdir"] = run_outdir
        if isinstance(kwargs.get("sim_config"), Mapping):
            kwargs["sim_config"] = SimulationConfig(**dict(kwargs["sim_config"]))

        summary = runner(**kwargs)
        summaries_by_run[run_name] = summary
        record = extract_summary_record(
            run_name=run_name,
            experiment=experiment,
            benchmark_family=benchmark_family,
            sweep_role=sweep_role,
            output_dir=run_outdir,
            summary=summary,
        )
        run_records.append(record)
        summary_file = None
        for candidate in (
            "summary.json",
        ):
            if (run_outdir / candidate).exists():
                summary_file = candidate
                break
        executed_runs.append(
            {
                **record,
                "summary_file": summary_file,
            }
        )

    specificity_report = build_specificity_report(
        summaries_by_run=summaries_by_run,
        runs=runs,
        contract=contract,
    )
    ablation_report = build_ablation_report(
        summaries_by_run=summaries_by_run,
        runs=runs,
    )

    study_artifacts = [
        "study_index.json",
        "study_summary.csv",
        "specificity_report.json",
        "ablation_report.json",
        "integrity_report.json",
        "contract_report.json",
        "study_manifest.json",
    ]
    study_index = {
        "study_name": study_name,
        "runs": executed_runs,
        "study_artifacts": study_artifacts,
        "provenance": capture_provenance(
            extra={"study_name": study_name, "n_runs": len(executed_runs)}
        ),
    }
    study_index_path = save_json(study_index, outdir / "study_index.json")
    study_summary_path = save_summary_csv(run_records, outdir / "study_summary.csv")
    specificity_path = save_json(specificity_report, outdir / "specificity_report.json")
    ablation_path = save_json(ablation_report, outdir / "ablation_report.json")

    write_manifest(
        files={
            "study_index": study_index_path,
            "study_summary": study_summary_path,
            "specificity_report": specificity_path,
            "ablation_report": ablation_path,
        },
        base_dir=outdir,
        manifest_name="study_manifest.json",
        format_version="v1",
    )

    integrity_report = build_integrity_report(
        outdir=outdir, executed_runs=executed_runs
    )
    save_json(integrity_report, outdir / "integrity_report.json")

    contract_report = evaluate_study_contract(
        study_index,
        records=run_records,
        specificity_report=specificity_report,
        ablation_report=ablation_report,
        integrity_report=integrity_report,
        contract=contract,
    )
    save_json(contract_report, outdir / "contract_report.json")

    return {
        "study_index": study_index,
        "records": run_records,
        "specificity_report": specificity_report,
        "ablation_report": ablation_report,
        "integrity_report": integrity_report,
        "contract_report": contract_report,
    }
