from __future__ import annotations

import numpy as np
from .base import ODESystem, Params



def AutonomousForcedVanDerPol() -> ODESystem:
    """
    Autonomous 3D lift of the periodically forced van der Pol oscillator.

    Equations:
        x'   = y
        y'   = mu*(1 - x^2)*y - x + A*sin(phi)
        phi' = omega
    """

    def default_params() -> Params:
        return {
            "mu": 8.0,
            "A": 1.2,
            "omega": 1.0,
        }

    def rhs(t: float, x: np.ndarray, p: Params) -> np.ndarray:
        x1, x2, phi = x
        mu = float(p["mu"])
        A = float(p["A"])
        omega = float(p["omega"])
        dx1 = x2
        dx2 = mu * (1.0 - x1**2) * x2 - x1 + A * np.sin(phi)
        dphi = omega
        return np.array([dx1, dx2, dphi], dtype=float)

    def jacobian(t: float, x: np.ndarray, p: Params) -> np.ndarray:
        x1, x2, phi = x
        mu = float(p["mu"])
        A = float(p["A"])
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0 - 2.0 * mu * x1 * x2, mu * (1.0 - x1**2), A * np.cos(phi)],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def default_initial_condition(p: Params, rng: np.random.Generator) -> np.ndarray:
        return np.array([2.0, 0.0, 0.0], dtype=float) + 0.05 * rng.standard_normal(
            size=3
        )

    return ODESystem(
        name="Autonomous forced van der Pol",
        dimension=3,
        rhs=rhs,
        jacobian=jacobian,
        default_params=default_params,
        default_initial_condition=default_initial_condition,
    )
