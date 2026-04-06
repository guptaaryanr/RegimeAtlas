from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from regime_toolkit.simulate import SimulationConfig
from experiments.fhn_pipeline import run_pipeline as run_fhn
from experiments.vdp_pipeline import run_pipeline as run_vdp


class TestPipelinesSmoke(unittest.TestCase):
    def test_fhn_pipeline_positive_lead_and_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_fhn(
                outdir=Path(tmpdir) / "fhn_main",
                eps_values=[0.05, 0.08, 0.13, 0.2, 0.47, 0.5],
                amplitude_threshold=0.2,
                run_robustness=False,
                sim_config=SimulationConfig(
                    t_final=50.0,
                    dt=0.05,
                    transient=30.0,
                    method="Radau",
                    rtol=1e-7,
                    atol=1e-9,
                    reset_time_after_transient=True,
                ),
            )
            self.assertEqual(summary["summary_version"], "v1")
            self.assertTrue(summary["ablation"]["passed"])
            self.assertIsNotNone(summary["lead_distance_primary"])
            self.assertGreater(summary["lead_distance_primary"]["lead_distance"], 0.0)

    def test_vdp_pipeline_strengthened_cut_positive_lead_and_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_vdp(
                outdir=Path(tmpdir) / "vdp_main",
                A_values=[0.35, 0.5, 0.75, 1.0, 1.25, 1.6],
                omega_fixed=0.9,
                run_robustness=False,
                sim_config=SimulationConfig(
                    t_final=60.0,
                    dt=0.05,
                    transient=40.0,
                    method="RK45",
                    rtol=1e-6,
                    atol=1e-8,
                    max_step=0.05,
                    reset_time_after_transient=True,
                ),
            )
            self.assertEqual(summary["summary_version"], "v1")
            self.assertTrue(summary["ablation"]["passed"])
            self.assertIsNotNone(summary["primary_qualitative_boundary"])
            self.assertIsNotNone(summary["lead_distance_primary"])
            self.assertGreater(summary["lead_distance_primary"]["lead_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
