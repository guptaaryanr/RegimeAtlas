from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from regime_toolkit.systems import FitzHughNagumo
from regime_toolkit.simulate import SimulationConfig
from regime_toolkit.sweep import IndicatorSpec, parameter_sweep
from regime_toolkit.plots import (
    plot_metric_vs_param,
    plot_regime_atlas,
    plot_attractor_projection,
)


class TestSweepIoAndPlots(unittest.TestCase):
    def test_parameter_sweep_saves_bundle_and_plots(self) -> None:
        system = FitzHughNagumo()
        base_params = system.default_params()
        sim_cfg = SimulationConfig(
            t_final=8.0,
            dt=0.05,
            transient=4.0,
            method="Radau",
            reset_time_after_transient=True,
        )

        def mean_abs_v_indicator(t, x, system, params):
            return {"mean_abs_v": float(np.mean(np.abs(x[:, 0])))}

        indicators = [IndicatorSpec(name="mean_abs_v", fn=mean_abs_v_indicator)]

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            results = parameter_sweep(
                system=system,
                base_params=base_params,
                control_param="epsilon",
                values=[0.05, 0.1],
                sim_config=sim_cfg,
                indicators=indicators,
                seed=0,
                store_trajectories_at=[0],
                verbose=False,
                save_dir=outdir,
                save_config={"experiment": "unit_test"},
            )

            self.assertIn("mean_abs_v", results["metrics"])
            self.assertIn("mean_abs_v", results["metric_metadata"])
            self.assertTrue((outdir / "results.json").exists())
            self.assertTrue((outdir / "config.json").exists())
            self.assertTrue((outdir / "manifest.json").exists())
            self.assertTrue((outdir / "provenance.json").exists())
            self.assertTrue((outdir / "trajectories.npz").exists())

            with (outdir / "results.json").open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["control_param"], "epsilon")
            self.assertIn("metrics", payload)

            fig1 = plot_metric_vs_param(
                results,
                metric="mean_abs_v",
                savepath=outdir / "mean_abs_v.png",
            )
            fig2 = plot_regime_atlas(
                results,
                metric_keys=["mean_abs_v"],
                savepath=outdir / "regime_atlas.png",
            )
            first_traj = next(iter(results["trajectories"].values()))
            fig3 = plot_attractor_projection(
                first_traj["t"],
                first_traj["x"],
                dims=(0, 1),
                savepath=outdir / "phase.png",
            )
            self.assertTrue((outdir / "mean_abs_v.png").exists())
            self.assertTrue((outdir / "regime_atlas.png").exists())
            self.assertTrue((outdir / "phase.png").exists())
            fig1.clf(); fig2.clf(); fig3.clf()


if __name__ == "__main__":
    unittest.main()
