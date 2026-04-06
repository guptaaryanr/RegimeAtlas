from __future__ import annotations

import numpy as np
from .base import ODESystem, Params


def FitzHughNagumo() -> ODESystem:
    """
    FitzHugh–Nagumo fast–slow excitable system.

    Structural control parameter:
    - epsilon: time-scale separation. As epsilon increases, fast/slow separation collapses.

    Notes:
    - With certain parameters, system transitions between fixed point (quiescent) and limit cycle.
    - Small epsilon can make the system stiff; consider Radau/BDF in simulations.
    """

    def default_params() -> Params:
        # Common textbook-ish parameters; you will likely tune I for excitability vs oscillation regimes.
        return {
            "epsilon": 0.08,  # fast/slow separation
            "a": 0.7,
            "b": 0.8,
            "I": 0.5,  # input/current
        }

    def rhs(t: float, x: np.ndarray, p: Params) -> np.ndarray:
        v, w = x
        eps = float(p["epsilon"])
        dv = (v - (v**3) / 3.0 - w + float(p["I"])) / eps
        dw = v + float(p["a"]) - float(p["b"]) * w
        return np.array([dv, dw], dtype=float)

    def jacobian(t: float, x: np.ndarray, p: Params) -> np.ndarray:
        v, w = x
        eps = float(p["epsilon"])
        # dv/dv = (1 - v^2)/eps, dv/dw = -1/eps
        # dw/dv = 1, dw/dw = -b
        j11 = (1.0 - v**2) / eps
        j12 = -1.0 / eps
        j21 = 1.0
        j22 = -float(p["b"])
        return np.array([[j11, j12], [j21, j22]], dtype=float)

    def default_initial_condition(p: Params, rng: np.random.Generator) -> np.ndarray:
        # Slight randomness helps avoid landing exactly on unstable manifolds in some parameterizations.
        v0 = -1.0 + 0.05 * rng.standard_normal()
        w0 = 1.0 + 0.05 * rng.standard_normal()
        return np.array([v0, w0], dtype=float)

    return ODESystem(
        name="FitzHugh–Nagumo",
        dimension=2,
        rhs=rhs,
        jacobian=jacobian,
        default_params=default_params,
        default_initial_condition=default_initial_condition,
    )


def fhn_equilibrium(params: Params) -> tuple[float, float]:
    """
    Compute the equilibrium (v*, w*) for the FHN system.

    Note: independent of epsilon for this formulation (epsilon rescales dv/dt).
    """
    a = float(params["a"])
    b = float(params["b"])
    I = float(params["I"])

    # Solve for v: v - v^3/3 - (v+a)/b + I = 0
    # Multiply by 3b to avoid fractions:
    # 3b v - b v^3 - 3(v+a) + 3b I = 0
    # => -b v^3 + (3b - 3) v + (3b I - 3a) = 0
    # Multiply by -1:
    # b v^3 + (3 - 3b) v + (3a - 3b I) = 0
    c3 = b
    c2 = 0.0
    c1 = 3.0 - 3.0 * b
    c0 = 3.0 * a - 3.0 * b * I

    roots = np.roots([c3, c2, c1, c0])
    roots_real = roots[np.isclose(roots.imag, 0.0, atol=1e-10)].real

    if roots_real.size == 0:
        raise RuntimeError("No real equilibrium root found (unexpected).")

    # If multiple real equilibria exist, pick the one the trajectory actually approaches by default.
    # For typical parameters here, it's unique anyway.
    v_star = float(roots_real[0])
    w_star = float((v_star + a) / b)
    return v_star, w_star


def fhn_equilibrium_jacobian_eigs(params: Params, epsilon: float) -> np.ndarray:
    """
    Eigenvalues of Jacobian evaluated at the equilibrium.
    For a stable fixed point, Lyapunov exponents should match Re(eigs) (autonomous system).
    """
    p = dict(params)
    p["epsilon"] = float(epsilon)
    v_star, w_star = fhn_equilibrium(p)

    # reuse the system jacobian formula
    eps = float(p["epsilon"])
    b = float(p["b"])
    j11 = (1.0 - v_star**2) / eps
    j12 = -1.0 / eps
    j21 = 1.0
    j22 = -b
    J = np.array([[j11, j12], [j21, j22]], dtype=float)
    return np.linalg.eigvals(J)


def fhn_jacobian_eigs_at_state(
    params: Params, epsilon: float, x: np.ndarray
) -> np.ndarray:
    p = dict(params)
    p["epsilon"] = float(epsilon)
    v, w = float(x[0]), float(x[1])
    eps = float(p["epsilon"])
    b = float(p["b"])
    j11 = (1.0 - v**2) / eps
    j12 = -1.0 / eps
    j21 = 1.0
    j22 = -b
    J = np.array([[j11, j12], [j21, j22]], dtype=float)
    return np.linalg.eigvals(J)
