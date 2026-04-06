from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from regime_toolkit.cli import main as cli_main
from regime_toolkit.io import save_json
from regime_toolkit.integrity import describe_file, validate_manifest_file
from regime_toolkit.schema import normalize_study_config
from regime_toolkit.study import run_study_suite


class TestReleaseAndCli(unittest.TestCase):
    def test_normalize_rejects_duplicate_run_names(self) -> None:
        raw = {
            "study_name": "bad",
            "outdir": "outputs/bad",
            "runs": [
                {
                    "run_name": "dup",
                    "experiment": "x",
                    "benchmark_family": "FitzHugh–Nagumo",
                    "sweep_role": "main",
                },
                {
                    "run_name": "dup",
                    "experiment": "y",
                    "benchmark_family": "Autonomous forced van der Pol",
                    "sweep_role": "nuisance",
                },
            ],
        }
        with self.assertRaises(ValueError):
            normalize_study_config(raw)

    def test_cli_validate_config_on_release_smoke(self) -> None:
        rc = cli_main(["validate-config", "configs/paper_suite_smoke.json"])
        self.assertEqual(rc, 0)

    def test_cli_validate_manifest_on_saved_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            payload_path = save_json({"hello": "world"}, out / "payload.json")
            manifest = {
                "format_version": "v1-test",
                "files": {"payload": describe_file(payload_path, relative_to=out)},
            }
            save_json(manifest, out / "manifest.json")
            rc = cli_main(["validate-manifest", str(out / "manifest.json")])
            self.assertEqual(rc, 0)
            report = validate_manifest_file(out / "manifest.json")
            self.assertTrue(report["passed"])

    def test_study_suite_writes_study_manifest(self) -> None:
        def dummy_runner(*, outdir: Path, **kwargs):
            outdir.mkdir(parents=True, exist_ok=True)
            summary = {
                "summary_version": "v1",
                "system": kwargs.get("family_hint", "dummy"),
                "control_param": "p",
                "param_values": [0.0, 1.0],
                "structural_boundary": {"param_value": 0.4},
                "primary_qualitative_boundary": {"param_value": 0.8} if kwargs.get("role_hint") == "main" else None,
                "primary_qualitative_boundary_kind": "qualitative_boundary",
                "lead_distance_primary": {"lead_distance": 0.4} if kwargs.get("role_hint") == "main" else None,
                "robustness_acceptance_primary": {"core_passed": True, "passed": True} if kwargs.get("role_hint") == "main" else None,
                "robustness_acceptance_supplemental": {"supplemental_passed": True, "passed": True} if kwargs.get("role_hint") == "main" else None,
                "robustness_acceptance_stress": {"stress_passed": True, "passed": True} if kwargs.get("role_hint") == "main" else None,
                "robustness_core_passed": True if kwargs.get("role_hint") == "main" else None,
                "robustness_supplemental_passed": True if kwargs.get("role_hint") == "main" else None,
                "robustness_stress_passed": True if kwargs.get("role_hint") == "main" else None,
                "raw_metric_spans": {"time_pr": 0.4, "occupancy_gap": 0.3} if kwargs.get("role_hint") == "main" else {"time_pr": 0.01, "occupancy_gap": 0.01},
                "primary_qualitative_label_unique_count": 2 if kwargs.get("role_hint") == "main" else 1,
                "claim_compliance": {"core_is_trajectory_only": True, "oracle_saved_separately": True},
                "ablation": {"passed": True, "details": {}, "variants": {"atlas_full": {}}} if kwargs.get("role_hint") == "main" else None,
                "ablation_report": {"passed": True, "details": {}, "variants": {"atlas_full": {}}} if kwargs.get("role_hint") == "main" else None,
            }
            save_json(summary, outdir / "summary.json")
            save_json(
                {
                    "format_version": "v1-test",
                    "files": {"summary_json": describe_file(outdir / "summary.json", relative_to=outdir)},
                },
                outdir / "manifest.json",
            )
            return summary

        runs = [
            {"run_name": "fhn_main", "experiment": "dummy", "benchmark_family": "FitzHugh–Nagumo", "sweep_role": "main", "family_hint": "FitzHugh–Nagumo", "role_hint": "main"},
            {"run_name": "fhn_nuisance", "experiment": "dummy", "benchmark_family": "FitzHugh–Nagumo", "sweep_role": "nuisance", "family_hint": "FitzHugh–Nagumo", "role_hint": "nuisance"},
            {"run_name": "vdp_main", "experiment": "dummy", "benchmark_family": "Autonomous forced van der Pol", "sweep_role": "main", "family_hint": "Autonomous forced van der Pol", "role_hint": "main"},
            {"run_name": "vdp_nuisance", "experiment": "dummy", "benchmark_family": "Autonomous forced van der Pol", "sweep_role": "nuisance", "family_hint": "Autonomous forced van der Pol", "role_hint": "nuisance"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_study_suite(
                study_name="release_test",
                outdir=Path(tmpdir),
                runs=runs,
                runner_map={"dummy": dummy_runner},
            )
            self.assertTrue((Path(tmpdir) / "study_manifest.json").exists())
            self.assertTrue(result["integrity_report"]["study_manifest"]["present"])
            self.assertTrue(result["integrity_report"]["study_manifest"]["report"]["passed"])


if __name__ == "__main__":
    unittest.main()
