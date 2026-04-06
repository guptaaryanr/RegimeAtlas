from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from regime_toolkit.contracts import (
    default_paper_study_contract,
    validate_metric_sources,
    TRAJECTORY_ONLY,
    ORACLE,
)
from regime_toolkit.io import save_json
from regime_toolkit.integrity import describe_file
from regime_toolkit.study import run_study_suite


def _write_minimal_manifest(run_dir: Path, summary_name: str) -> None:
    summary_path = run_dir / summary_name
    payload = {
        "format_version": "v1-test",
        "files": {
            "summary_json": describe_file(summary_path),
        },
    }
    save_json(payload, run_dir / "manifest.json")


class TestContractsAndStudy(unittest.TestCase):
    def test_validate_metric_sources(self) -> None:
        good = {
            "time_pr": {"source_class": TRAJECTORY_ONLY},
            "occupancy_gap": {"source_class": TRAJECTORY_ONLY},
        }
        validate_metric_sources(good, ["time_pr", "occupancy_gap"])

        bad = {
            "time_pr": {"source_class": ORACLE},
        }
        with self.assertRaises(ValueError):
            validate_metric_sources(bad, ["time_pr"])

    def test_run_study_suite_and_contract_report(self) -> None:
        contract = default_paper_study_contract()

        def dummy_runner(*, outdir: Path, **kwargs):
            outdir.mkdir(parents=True, exist_ok=True)
            benchmark_family = str(kwargs.get("family_hint", "unknown"))
            sweep_role = str(kwargs.get("role_hint", "main"))
            if sweep_role == "main":
                ablation_payload = {
                    "passed": True,
                    "details": {"atlas_lead_distance": 0.3},
                    "variants": {
                        "atlas_full": {"metric_names": ["time_pr", "occupancy_gap"]}
                    },
                }

                summary = {
                    "summary_version": "v1",
                    "system": benchmark_family,
                    "control_param": "p",
                    "param_values": [0.0, 1.0],
                    "structural_boundary": {"param_value": 0.5},
                    "primary_qualitative_boundary": {"param_value": 0.8},
                    "primary_qualitative_boundary_kind": "qualitative_boundary",
                    "lead_distance_primary": {"lead_distance": 0.3},
                    # new-style robustness fields
                    "robustness_acceptance_primary": {
                        "core_passed": True,
                        "passed": True,
                    },
                    "robustness_acceptance_supplemental": {
                        "supplemental_passed": True,
                        "passed": True,
                    },
                    "robustness_acceptance_stress": {
                        "stress_passed": True,
                        "passed": True,
                    },
                    # direct fallback fields for backward compatibility
                    "robustness_core_passed": True,
                    "robustness_supplemental_passed": True,
                    "robustness_stress_passed": True,
                    "raw_metric_spans": {"time_pr": 0.4, "occupancy_gap": 0.3},
                    "primary_qualitative_label_unique_count": 2,
                    "claim_compliance": {
                        "core_is_trajectory_only": True,
                        "oracle_saved_separately": True,
                    },
                    # support both old and new field names
                    "ablation": ablation_payload,
                    "ablation_report": ablation_payload,
                }
            else:
                summary = {
                    "summary_version": "v1",
                    "system": benchmark_family,
                    "control_param": "p2",
                    "param_values": [0.0, 1.0],
                    "structural_boundary": {"param_value": 0.5},
                    "primary_qualitative_boundary": None,
                    "primary_qualitative_boundary_kind": "qualitative_boundary",
                    "lead_distance_primary": None,
                    "raw_metric_spans": {"time_pr": 0.01, "occupancy_gap": 0.01},
                    "primary_qualitative_label_unique_count": 1,
                    "claim_compliance": {
                        "core_is_trajectory_only": True,
                        "oracle_saved_separately": True,
                    },
                }
            save_json(summary, outdir / "summary.json")
            _write_minimal_manifest(outdir, "summary.json")
            return summary

        runs = [
            {
                "run_name": "fhn_main",
                "experiment": "dummy",
                "benchmark_family": "FitzHugh–Nagumo",
                "sweep_role": "main",
                "family_hint": "FitzHugh–Nagumo",
                "role_hint": "main",
            },
            {
                "run_name": "fhn_nuisance",
                "experiment": "dummy",
                "benchmark_family": "FitzHugh–Nagumo",
                "sweep_role": "nuisance",
                "family_hint": "FitzHugh–Nagumo",
                "role_hint": "nuisance",
            },
            {
                "run_name": "vdp_main",
                "experiment": "dummy",
                "benchmark_family": "Autonomous forced van der Pol",
                "sweep_role": "main",
                "family_hint": "Autonomous forced van der Pol",
                "role_hint": "main",
            },
            {
                "run_name": "vdp_nuisance",
                "experiment": "dummy",
                "benchmark_family": "Autonomous forced van der Pol",
                "sweep_role": "nuisance",
                "family_hint": "Autonomous forced van der Pol",
                "role_hint": "nuisance",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_study_suite(
                study_name="unit_study",
                outdir=Path(tmpdir),
                runs=runs,
                runner_map={"dummy": dummy_runner},
                contract=contract,
            )
            self.assertTrue(result["specificity_report"]["passed"])
            self.assertTrue(result["ablation_report"]["passed"])
            self.assertTrue(result["integrity_report"]["passed"])
            self.assertTrue(result["contract_report"]["passed"])
            self.assertTrue((Path(tmpdir) / "study_index.json").exists())
            self.assertTrue((Path(tmpdir) / "study_summary.csv").exists())
            self.assertTrue((Path(tmpdir) / "specificity_report.json").exists())
            self.assertTrue((Path(tmpdir) / "ablation_report.json").exists())
            self.assertTrue((Path(tmpdir) / "integrity_report.json").exists())
            self.assertTrue((Path(tmpdir) / "contract_report.json").exists())


if __name__ == "__main__":
    unittest.main()
