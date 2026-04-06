from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from regime_toolkit.boundaries import (
    composite_structural_score,
    piecewise_linear_change_point,
    change_point_sensitivity_scan,
    first_threshold_crossing,
    lead_distance,
)
from regime_toolkit.oracles import (
    fhn_equilibria,
    predict_fhn_hopf_epsilon,
    tail_oscillation_metrics,
)
from regime_toolkit.robustness import RobustnessCase, run_robustness_cases
from regime_toolkit.systems import FitzHughNagumo
from regime_toolkit.simulate import SimulationConfig
from regime_toolkit.sweep import IndicatorSpec
from regime_toolkit.observations import observed_trajectory
from regime_toolkit.indicators import (
    velocity_participation_ratio_time_weighted,
    velocity_participation_ratio_arclength_weighted,
)


class TestBoundariesAndOracles(unittest.TestCase):
    def test_piecewise_linear_change_point_on_synthetic_curve(self) -> None:
        x = np.linspace(0.0, 1.0, 11)
        y = np.where(x <= 0.5, 0.5 * x, 0.25 + 2.0 * (x - 0.5))
        cp = piecewise_linear_change_point(x, y, min_segment_size=3)
        self.assertIn(cp.index, {4, 5})
        sensitivity = change_point_sensitivity_scan(x, y, smooth_windows=(1, 3), min_segment_sizes=(2, 3))
        self.assertIsNotNone(sensitivity.ci_low)
        self.assertIsNotNone(sensitivity.ci_high)
        self.assertAlmostEqual(cp.param_value, x[cp.index])
        self.assertGreater(cp.score, 1.0)

    def test_composite_score_and_threshold_crossing(self) -> None:
        x = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = {
            "a": np.array([1.0, 2.0, 3.0, 4.0]),
            "b": np.array([4.0, 3.0, 2.0, 1.0]),
        }
        score, components = composite_structural_score(
            x,
            metrics,
            {"a": "increasing", "b": "decreasing"},
            smooth_window=1,
        )
        self.assertEqual(score.shape, x.shape)
        self.assertIn("a", components)
        crossing = first_threshold_crossing(x, np.array([1.0, 0.8, 0.4, 0.1]), 0.5)
        self.assertAlmostEqual(crossing, 0.275)

        ld = lead_distance(0.2, 0.35)
        self.assertAlmostEqual(ld.lead_distance, 0.15)

    def test_fhn_oracle_utilities(self) -> None:
        params = FitzHughNagumo().default_params()
        eq = fhn_equilibria(params)
        self.assertGreaterEqual(eq.states.shape[0], 1)
        hopf = predict_fhn_hopf_epsilon(params)
        self.assertIsNotNone(hopf)
        self.assertAlmostEqual(hopf, 0.439, places=2)

        t = np.linspace(0.0, 10.0, 200)
        x = np.zeros((200, 2), dtype=float)
        metrics = tail_oscillation_metrics(t, x, amplitude_threshold=0.2)
        self.assertEqual(metrics.regime_label, "fixed_point")
        self.assertAlmostEqual(metrics.amplitude, 0.0)

    def test_small_robustness_runner(self) -> None:
        system = FitzHughNagumo()
        base_params = system.default_params()
        values = np.geomspace(0.03, 0.08, 4)
        sim_cfg = SimulationConfig(
            t_final=30.0,
            dt=0.05,
            transient=20.0,
            method="Radau",
            reset_time_after_transient=True,
        )

        def indicator_factory(case: RobustnessCase, seed: int):
            def occ(t, x, system, params):
                obs = observed_trajectory(
                    x,
                    representation=case.representation,
                    scalar_mode="coordinate",
                    index=0,
                    embedding_dim=3,
                    delay=5,
                    noise_sigma=case.observation_noise_sigma,
                    noise_relative=case.observation_noise_relative,
                    seed=seed,
                )
                t_obs = t[obs.time_indices]
                z = obs.data
                time_pr = velocity_participation_ratio_time_weighted(t_obs, z).value
                arc_pr = velocity_participation_ratio_arclength_weighted(t_obs, z).value
                return {
                    "time_pr": float(time_pr),
                    "occupancy_gap": float(arc_pr - time_pr),
                }
            return [IndicatorSpec(name="occ", fn=occ)]

        cases = [
            RobustnessCase(name="baseline", representation="full_state"),
            RobustnessCase(name="delay_scalar", representation="delay_scalar", embedding_dim=3, delay=5),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = run_robustness_cases(
                system=system,
                base_params=base_params,
                control_param="epsilon",
                values=values,
                base_sim_config=sim_cfg,
                cases=cases,
                indicator_factory=indicator_factory,
                boundary_metric_directions={"time_pr": "increasing", "occupancy_gap": "decreasing"},
                seed=0,
                verbose=False,
                save_dir=Path(tmpdir),
            )
            self.assertEqual(len(out["summary_rows"]), 2)
            self.assertIn("acceptance", out)
            self.assertTrue((Path(tmpdir) / "robustness_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
