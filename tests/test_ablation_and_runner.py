from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from regime_toolkit.ablation import evaluate_structural_ablation_suite
from regime_toolkit.boundaries import composite_structural_score
from regime_toolkit.calibration import compare_main_vs_nuisance_specificity
from regime_toolkit.schema import normalize_study_config
from regime_toolkit.study import run_study_suite
from regime_toolkit.io import save_json
from regime_toolkit.integrity import describe_file


METRIC_DIRECTIONS = {"time_pr": "increasing", "occupancy_gap": "decreasing"}
WEIGHTS = {"time_pr": 1.0, "occupancy_gap": 1.0}


def _write_minimal_manifest(run_dir: Path, summary_name: str) -> None:
    save_json(
        {
            "format_version": "v1-test",
            "files": {"summary_json": describe_file(run_dir / summary_name)},
        },
        run_dir / "manifest.json",
    )


class TestV1AblationAndRunner(unittest.TestCase):
    def test_ablation_prefers_full_atlas_over_single_metrics(self) -> None:
        p = np.linspace(0.0, 1.0, 9)
        metrics = {
            "time_pr": np.array([1.00, 1.02, 1.04, 1.08, 1.20, 1.32, 1.42, 1.48, 1.50]),
            "occupancy_gap": np.array(
                [0.60, 0.58, 0.56, 0.50, 0.38, 0.26, 0.18, 0.14, 0.12]
            ),
        }
        full_score, _ = composite_structural_score(
            p, metrics, METRIC_DIRECTIONS, weights=WEIGHTS, smooth_window=1
        )
        qualitative_boundary = 0.75
        result = evaluate_structural_ablation_suite(
            param_values=p,
            metrics=metrics,
            metric_directions=METRIC_DIRECTIONS,
            weights=WEIGHTS,
            qualitative_boundary=qualitative_boundary,
        )
        self.assertIn("atlas_full", result.variants)
        self.assertTrue(result.passed)
        self.assertGreater(result.variants["atlas_full"].lead_distance, 0.0)

    def test_specificity_can_distinguish_main_from_small_nuisance(self) -> None:
        main_summary = {
            "raw_metric_spans": {"time_pr": 0.5, "occupancy_gap": 0.4},
            "primary_qualitative_boundary": {"param_value": 1.2},
            "primary_qualitative_label_unique_count": 2,
        }
        nuisance_summary = {
            "raw_metric_spans": {"time_pr": 0.03, "occupancy_gap": 0.02},
            "primary_qualitative_boundary": None,
            "primary_qualitative_label_unique_count": 1,
        }
        result = compare_main_vs_nuisance_specificity(
            benchmark_family="Autonomous forced van der Pol",
            main_summary=main_summary,
            nuisance_summary=nuisance_summary,
            metric_names=["time_pr", "occupancy_gap"],
        )
        self.assertTrue(result.passed)
        self.assertLess(result.max_ratio, 0.35)

    def test_normalized_benchmark_suite_can_run_dummy_study(self) -> None:
        raw = {
            "experiment": "benchmark_suite",
            "outdir": "unused",
            "config": {
                "fhn_main": {},
                "fhn_nuisance": {},
                "vdp_main": {"omega_fixed": 0.9},
                "vdp_nuisance": {"A_fixed": 0.8},
            },
        }
        cfg = normalize_study_config(raw)

        def dummy_runner(*, outdir: Path, **kwargs):
            outdir.mkdir(parents=True, exist_ok=True)
            is_main = kwargs.get("run_name_hint", "main").endswith("main")
            ablation_payload = {
                "passed": True,
                "details": {},
                "variants": {"atlas_full": {}},
            }

            summary = {
                "summary_version": "v1",
                "system": kwargs.get("system_hint", "dummy"),
                "control_param": "p",
                "param_values": [0.0, 1.0],
                "structural_boundary": {"param_value": 0.4},
                "primary_qualitative_boundary": (
                    {"param_value": 0.8} if is_main else None
                ),
                "primary_qualitative_boundary_kind": "qualitative_boundary",
                "lead_distance_primary": {"lead_distance": 0.4} if is_main else None,
                "robustness_acceptance_primary": (
                    {"core_passed": True, "passed": True} if is_main else None
                ),
                "robustness_acceptance_supplemental": (
                    {"supplemental_passed": True, "passed": True} if is_main else None
                ),
                "robustness_acceptance_stress": (
                    {"stress_passed": True, "passed": True} if is_main else None
                ),
                "robustness_core_passed": True if is_main else None,
                "robustness_supplemental_passed": True if is_main else None,
                "robustness_stress_passed": True if is_main else None,
                "raw_metric_spans": (
                    {"time_pr": 0.4, "occupancy_gap": 0.35}
                    if is_main
                    else {"time_pr": 0.01, "occupancy_gap": 0.01}
                ),
                "primary_qualitative_label_unique_count": 2 if is_main else 1,
                "claim_compliance": {
                    "core_is_trajectory_only": True,
                    "oracle_saved_separately": True,
                },
                "ablation": ablation_payload if is_main else None,
                "ablation_report": ablation_payload if is_main else None,
            }
            save_json(summary, outdir / "summary.json")
            _write_minimal_manifest(outdir, "summary.json")
            return summary

        runs = []
        for run in cfg["runs"]:
            enriched = dict(run)
            enriched["run_name_hint"] = run["run_name"]
            enriched["system_hint"] = run["benchmark_family"]
            runs.append(enriched)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_study_suite(
                study_name="bench_suite_dummy",
                outdir=Path(tmpdir),
                runs=runs,
                runner_map={
                    "fhn_pipeline": dummy_runner,
                    "fhn_nuisance_pipeline": dummy_runner,
                    "vdp_pipeline": dummy_runner,
                    "vdp_nuisance_pipeline": dummy_runner,
                },
            )
            self.assertTrue(result["contract_report"]["passed"])
            self.assertTrue(result["integrity_report"]["passed"])
            self.assertTrue(result["ablation_report"]["passed"])
            self.assertTrue(result["specificity_report"]["passed"])


if __name__ == "__main__":
    unittest.main()
