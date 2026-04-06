from __future__ import annotations

import unittest
import numpy as np

from regime_toolkit.systems import FitzHughNagumo, AutonomousForcedVanDerPol
from regime_toolkit.simulate import SimulationConfig, simulate


class TestSystemsAndSimulation(unittest.TestCase):
    def test_fhn_simulation_returns_finite_trajectory(self) -> None:
        system = FitzHughNagumo()
        params = system.default_params()
        params["epsilon"] = 0.08
        cfg = SimulationConfig(
            t_final=20.0,
            dt=0.05,
            transient=10.0,
            method="Radau",
            reset_time_after_transient=True,
        )
        t, x, meta = simulate(system, params, cfg, seed=0)
        self.assertEqual(x.shape[1], 2)
        self.assertEqual(t.shape[0], x.shape[0])
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertIn("x0", meta)

    def test_autonomous_forced_vdp_simulation_returns_3d_state(self) -> None:
        system = AutonomousForcedVanDerPol()
        params = system.default_params()
        cfg = SimulationConfig(
            t_final=10.0,
            dt=0.05,
            transient=5.0,
            method="RK45",
            reset_time_after_transient=True,
        )
        t, x, _ = simulate(system, params, cfg, seed=0)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(t.shape[0], x.shape[0])
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertGreater(np.mean(np.diff(x[:, 2])), 0.0)


if __name__ == "__main__":
    unittest.main()
