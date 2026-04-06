from __future__ import annotations

import unittest
import numpy as np

from regime_toolkit.systems.base import ODESystem
from regime_toolkit.indicators import lyapunov_spectrum_qr


def make_linear_stable_system() -> ODESystem:
    def default_params():
        return {"a": -1.0, "b": -2.0}

    def rhs(t: float, x: np.ndarray, p: dict[str, float]) -> np.ndarray:
        return np.array([p["a"] * x[0], p["b"] * x[1]], dtype=float)

    def jacobian(t: float, x: np.ndarray, p: dict[str, float]) -> np.ndarray:
        return np.array([[p["a"], 0.0], [0.0, p["b"]]], dtype=float)

    def default_initial_condition(p: dict[str, float], rng: np.random.Generator) -> np.ndarray:
        return np.array([1.0, 1.0], dtype=float)

    return ODESystem(
        name="LinearStable",
        dimension=2,
        rhs=rhs,
        jacobian=jacobian,
        default_params=default_params,
        default_initial_condition=default_initial_condition,
    )


class TestLyapunov(unittest.TestCase):
    def test_linear_stable_system_has_negative_exponents(self) -> None:
        system = make_linear_stable_system()
        params = system.default_params()
        result = lyapunov_spectrum_qr(
            system,
            params,
            x0=np.array([1.0, 1.0], dtype=float),
            t_max=40.0,
            dt_orth=0.1,
            k=2,
            method="RK45",
            rtol=1e-9,
            atol=1e-11,
            seed=0,
        )
        exps = np.sort(result.exponents)[::-1]
        self.assertTrue(np.all(np.isfinite(exps)))
        self.assertLess(exps[0], 0.0)
        self.assertLess(exps[1], 0.0)
        self.assertAlmostEqual(float(np.sum(exps)), -3.0, places=6)
        self.assertTrue(np.allclose(exps, np.array([-1.0, -2.0]), atol=0.1))


if __name__ == "__main__":
    unittest.main()
