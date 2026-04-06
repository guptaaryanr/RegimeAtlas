from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Sequence

TRAJECTORY_ONLY = "trajectory_only"
ORACLE = "oracle"


@dataclass(frozen=True)
class PaperStudyContract:
    """
    Executable guardrails for the v1 paper claim.

    This contract hardens three things in code, not prose:
    - robustness is required for the primary full-state claim,
    - reproducibility artifacts must include integrity and schema-normalized study outputs,
    - novelty must be backed by specificity and ablation evidence.
    """

    name: str
    version: str
    core_claim: str
    novelty_lane: str
    primary_observation_scope: str
    supplemental_scope: str
    allowed_structural_sources: tuple[str, ...]
    required_benchmark_families: tuple[str, ...]
    required_sweep_roles: tuple[str, ...]
    required_study_artifacts: tuple[str, ...]
    default_structural_metrics: tuple[str, ...]
    forbidden_as_core: tuple[str, ...]
    required_specificity_pass: bool
    required_ablation_pass: bool
    required_integrity_pass: bool
    require_main_qualitative_boundary: bool
    require_main_positive_lead_distance: bool
    require_main_primary_robustness: bool
    require_main_supplemental_robustness: bool


def default_paper_study_contract() -> PaperStudyContract:
    return PaperStudyContract(
        name="structural_regimes_paper_contract",
        version="v1",
        core_claim=(
            "Trajectory-only structural-regime atlases from multivariate trajectories detect "
            "degradation of reduced-model assumptions before final qualitative regime change."
        ),
        novelty_lane=(
            "Non-ML, trajectory-only structural-validity atlas validated by but not dependent on model-aware oracles, "
            "with specificity and ablation evidence against simpler alternatives."
        ),
        primary_observation_scope=(
            "Core acceptance is based on multivariate/full-state trajectories under finite-data, perturbation, "
            "replicate, and solver-crosscheck robustness."
        ),
        supplemental_scope=(
            "Scalar delay-embedding robustness is reported separately as a harder supplemental test, not the primary claim."
        ),
        allowed_structural_sources=(TRAJECTORY_ONLY,),
        required_benchmark_families=(
            "FitzHugh–Nagumo",
            "Autonomous forced van der Pol",
        ),
        required_sweep_roles=("main", "nuisance"),
        required_study_artifacts=(
            "study_index.json",
            "study_summary.csv",
            "contract_report.json",
            "specificity_report.json",
            "ablation_report.json",
            "integrity_report.json",
            "study_manifest.json",
        ),
        default_structural_metrics=("time_pr", "occupancy_gap"),
        forbidden_as_core=(
            "oracle_tail_amplitude",
            "lyapunov_qr",
            "equilibrium_stability",
        ),
        required_specificity_pass=True,
        required_ablation_pass=True,
        required_integrity_pass=True,
        require_main_qualitative_boundary=True,
        require_main_positive_lead_distance=True,
        require_main_primary_robustness=True,
        require_main_supplemental_robustness=True,
    )


def validate_metric_sources(
    metric_metadata: Mapping[str, Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    allowed_sources: Sequence[str] = (TRAJECTORY_ONLY,),
) -> None:
    allowed = set(allowed_sources)
    for name in metric_names:
        if name not in metric_metadata:
            raise KeyError(f"metric '{name}' is missing from metric_metadata")
        source = metric_metadata[name].get("source_class")
        if source not in allowed:
            raise ValueError(
                f"metric '{name}' has source_class='{source}', expected one of {sorted(allowed)}"
            )


def evaluate_study_contract(
    study_index: Mapping[str, Any],
    *,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    specificity_report: Optional[Mapping[str, Any]] = None,
    ablation_report: Optional[Mapping[str, Any]] = None,
    integrity_report: Optional[Mapping[str, Any]] = None,
    contract: Optional[PaperStudyContract] = None,
) -> Dict[str, Any]:
    contract = default_paper_study_contract() if contract is None else contract
    runs = list(study_index.get("runs", []))
    records = list(records or [])

    families_present = set()
    roles_by_family: Dict[str, set[str]] = {}
    for run in runs:
        family = run.get("benchmark_family")
        role = run.get("sweep_role")
        if family is None or role is None:
            continue
        families_present.add(str(family))
        roles_by_family.setdefault(str(family), set()).add(str(role))

    family_checks = {
        family: {
            "present": family in families_present,
            "roles_present": sorted(roles_by_family.get(family, set())),
            "missing_roles": sorted(
                set(contract.required_sweep_roles) - roles_by_family.get(family, set())
            ),
        }
        for family in contract.required_benchmark_families
    }

    study_artifacts = set(study_index.get("study_artifacts", []))
    artifact_checks = {
        artifact: (artifact in study_artifacts)
        for artifact in contract.required_study_artifacts
    }

    main_records = [r for r in records if r.get("sweep_role") == "main"]
    main_checks = []
    for record in main_records:
        qualitative_boundary = record.get("primary_qualitative_boundary_param")
        lead_distance = record.get("lead_distance_primary")
        robustness_core = record.get("robustness_core_passed")
        ablation_passed = record.get("ablation_passed")
        check = {
            "run_name": record.get("run_name"),
            "benchmark_family": record.get("benchmark_family"),
            "has_structural_boundary": record.get("structural_boundary_param")
            is not None,
            "has_qualitative_boundary": qualitative_boundary is not None,
            "lead_distance_primary": lead_distance,
            "lead_distance_positive": (
                None if lead_distance is None else float(lead_distance) > 0.0
            ),
            "robustness_core_passed": robustness_core,
            "ablation_passed": ablation_passed,
            "robustness_supplemental_passed": record.get(
                "robustness_supplemental_passed"
            ),
        }
        main_checks.append(check)

    require_main_supplemental_robustness = True
    specificity_passed = (
        None if specificity_report is None else bool(specificity_report.get("passed"))
    )
    ablation_passed = (
        None if ablation_report is None else bool(ablation_report.get("passed"))
    )
    integrity_passed = (
        None if integrity_report is None else bool(integrity_report.get("passed"))
    )

    passed = bool(
        all(
            item["present"] and len(item["missing_roles"]) == 0
            for item in family_checks.values()
        )
        and all(artifact_checks.values())
        and (
            not contract.require_main_qualitative_boundary
            or all(item["has_qualitative_boundary"] for item in main_checks)
        )
        and (
            not contract.require_main_positive_lead_distance
            or all(item.get("lead_distance_positive") for item in main_checks)
        )
        and (
            not contract.require_main_primary_robustness
            or all(bool(item.get("robustness_core_passed")) for item in main_checks)
        )
        and (not contract.required_specificity_pass or bool(specificity_passed))
        and (not contract.required_ablation_pass or bool(ablation_passed))
        and (not contract.required_integrity_pass or bool(integrity_passed))
        and (
            not contract.require_main_supplemental_robustness
            or all(
                bool(item.get("robustness_supplemental_passed")) for item in main_checks
            )
        )
    )

    return {
        "contract": asdict(contract),
        "n_runs": len(runs),
        "family_checks": family_checks,
        "artifact_checks": artifact_checks,
        "main_run_checks": main_checks,
        "specificity_passed": specificity_passed,
        "ablation_passed": ablation_passed,
        "integrity_passed": integrity_passed,
        "passed": passed,
    }
