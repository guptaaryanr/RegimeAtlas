from __future__ import annotations

import numpy as np
from .base import ODESystem, Params


def ForcedVanDerPol() -> ODESystem:
    """
    Forced van der Pol oscillator.

    Equations:
        x' = y
        y' = mu*(1 - x^2)*y - x + A*sin(omega*t)

    Structural control parameters:
    - A: forcing amplitude (can induce entrainment breakdown and complex dynamics)
    - mu: nonlinearity/relaxation strength (controls how sharp slow-fast switching is)

    This system is useful when you want a canonical relaxation oscillator plus a knob that can
    destabilize simple limit cycles (mixed-mode / complex responses in certain regimes).
    """
    def default_params() -> Params:
        return {
            "mu": 8.0,      # relaxation strength (larger -> more slow-fast)
            "A": 1.2,       # forcing amplitude
            "omega": 1.0,   # forcing frequency
        }

    def rhs(t: float, x: np.ndarray, p: Params) -> np.ndarray:
        x1, x2 = x
        mu = float(p["mu"])
        A = float(p["A"])
        omega = float(p["omega"])
        dx1 = x2
        dx2 = mu * (1.0 - x1**2) * x2 - x1 + A * np.sin(omega * t)
        return np.array([dx1, dx2], dtype=float)

    def jacobian(t: float, x: np.ndarray, p: Params) -> np.ndarray:
        x1, x2 = x
        mu = float(p["mu"])
        # Partial derivatives:
        # dx1/dx1 = 0, dx1/dx2 = 1
        # dx2/dx1 = mu*(-2*x1)*x2 - 1
        # dx2/dx2 = mu*(1 - x1^2)
        j11 = 0.0
        j12 = 1.0
        j21 = -1.0 - 2.0 * mu * x1 * x2
        j22 = mu * (1.0 - x1**2)
        return np.array([[j11, j12], [j21, j22]], dtype=float)

    def default_initial_condition(p: Params, rng: np.random.Generator) -> np.ndarray:
        return np.array([2.0, 0.0], dtype=float) + 0.05 * rng.standard_normal(size=2)

    return ODESystem(
        name="Forced van der Pol",
        dimension=2,
        rhs=rhs,
        jacobian=jacobian,
        default_params=default_params,
        default_initial_condition=default_initial_condition,
    )
