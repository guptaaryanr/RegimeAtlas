from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from regime_toolkit.integrity import describe_file, validate_manifest_file
from regime_toolkit.io import save_json, save_sweep_bundle
from regime_toolkit.schema import normalize_study_config


class TestV1SchemaAndIntegrity(unittest.TestCase):
    def test_normalize_legacy_benchmark_suite_config(self) -> None:
        raw = {
            "experiment": "benchmark_suite",
            "outdir": "outputs/bench",
            "config": {
                "fhn_main": {"eps_values": [0.1, 0.2]},
                "fhn_nuisance": {"I_values": [0.49, 0.5]},
                "vdp_main": {"A_values": [0.5, 1.0], "omega_fixed": 0.9},
                "vdp_nuisance": {"omega_values": [0.88, 0.9, 0.92], "A_fixed": 0.8},
            },
        }
        cfg = normalize_study_config(raw)
        self.assertEqual(cfg["source_format"], "benchmark_suite")
        self.assertEqual(len(cfg["runs"]), 4)
        names = {run["run_name"] for run in cfg["runs"]}
        self.assertEqual(names, {"fhn_main", "fhn_nuisance", "vdp_main", "vdp_nuisance"})
        exp_map = {run["run_name"]: run["experiment"] for run in cfg["runs"]}
        self.assertEqual(exp_map["vdp_main"], "vdp_pipeline")
        self.assertEqual(exp_map["vdp_nuisance"], "vdp_nuisance_pipeline")

    def test_manifest_validation_passes_for_saved_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            results = {
                "system": "dummy",
                "param_values": np.array([0.0, 1.0]),
                "metrics": {"time_pr": np.array([0.1, 0.2])},
                "trajectories": {},
            }
            save_sweep_bundle(
                results=results,
                outdir=out,
                config={"experiment": "dummy"},
            )
            report = validate_manifest_file(out / "manifest.json")
            self.assertTrue(report["passed"])
            self.assertGreaterEqual(report["n_files"], 3)

    def test_manifest_validation_fails_on_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            payload = {"hello": "world"}
            save_json(payload, out / "sample.json")
            manifest = {
                "format_version": "v1-test",
                "files": {
                    "sample": describe_file(out / "sample.json"),
                },
            }
            save_json(manifest, out / "manifest.json")
            save_json({"hello": "tampered"}, out / "sample.json")
            report = validate_manifest_file(out / "manifest.json")
            self.assertFalse(report["passed"])
            self.assertFalse(report["per_file"]["sample"]["sha256_ok"])


if __name__ == "__main__":
    unittest.main()
