from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from scipy.integrate import solve_ivp

from ..systems.base import ODESystem, Params
from .jacobian import jacobian_finite_difference


@dataclass(frozen=True)
class LyapunovResult:
    """
    Output container for Lyapunov spectrum estimation.
    """

    exponents: np.ndarray
    gap: Optional[float]
    stderr: Optional[np.ndarray]  # standard error estimates per exponents, if computed
    details: Dict[str, Any]


def _orthonormal_matrix(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random n x k orthonormal matrix via QR.
    """
    A = rng.standard_normal(size=(n, k))
    Q, R = np.linalg.qr(A)
    # Make diagonal of R positive for determinism of the basis orientation.
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs
    return Q


def lyapunov_spectrum_qr(
    system: ODESystem,
    params: Params,
    x0: np.ndarray,
    t_max: float,
    dt_orth: float,
    k: Optional[int] = None,
    t0: float = 0.0,
    transient: float = 0.0,
    method: str = "RK45",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    jacobian_rel_step: float = 1e-6,
    seed: Optional[int] = 0,
    block_size_steps: int = 20,
) -> LyapunovResult:
    """
    Estimate the first k Lyapunov exponents using the classic Benettin method with QR re-orthonormalization.

    Implementation sketch:
    - Integrate the state x(t) and k tangent vectors P(t) under P' = J(x,t) P.
    - Every dt_orth, QR-decompose P to maintain numerical stability.
    - Accumulate log of R diagonal to get exponents.

    Structural assumptions probed:
    - Sensitivity to perturbations (stability vs chaos).
    - A "spectral gap" (lambda1 - lambda2) can indicate dimensional reduction / dominant mode separation.

    Limitations / failure modes:
    - Convergence can be slow near bifurcations or for weak chaos (small positive exponents).
    - For limit cycles in autonomous systems, the largest exponent should approach 0;
      finite-time estimates often show small negative bias depending on dt_orth and tolerances.
    - For stiff systems, use a stiff integrator (Radau/BDF) and/or smaller dt_orth.
    """
    x0 = np.asarray(x0, dtype=float)
    system.validate_state(x0)

    n = system.dimension
    if k is None:
        k = n
    if not (1 <= k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n (n={n}), got k={k}")

    if t_max <= 0 or dt_orth <= 0:
        raise ValueError("t_max and dt_orth must be positive")

    rng = np.random.default_rng(seed)

    # Optional transient run to move onto attractor
    t = float(t0)
    x = x0.copy()

    def f_state(tt: float, xx: np.ndarray) -> np.ndarray:
        return system.rhs(tt, xx, params)

    if transient > 0.0:
        sol_tr = solve_ivp(
            f_state,
            (t, t + transient),
            x,
            t_eval=[t + transient],
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not sol_tr.success:
            raise RuntimeError(
                f"Transient integration failed for {system.name}: {sol_tr.message}"
            )
        x = np.asarray(sol_tr.y[:, -1], dtype=float)
        t = t + transient

    # Initial orthonormal tangent frame
    Q = _orthonormal_matrix(n, k, rng)

    # Logs accumulator for diagonal of R
    sum_log_diag = np.zeros(k, dtype=float)
    log_diag_history = []

    # Number of orthonormalization steps
    n_steps = int(np.floor(t_max / dt_orth))
    if n_steps < 1:
        raise ValueError("t_max is too small relative to dt_orth; need at least 1 step")

    # Choose Jacobian function
    if system.jacobian is not None:

        def J(tt: float, xx: np.ndarray) -> np.ndarray:
            return system.jacobian(tt, xx, params)

    else:

        def J(tt: float, xx: np.ndarray) -> np.ndarray:
            return jacobian_finite_difference(
                system.rhs, tt, xx, params, rel_step=jacobian_rel_step
            )

    def f_extended(tt: float, y: np.ndarray) -> np.ndarray:
        xx = y[:n]
        P = y[n:].reshape(n, k)
        dxx = system.rhs(tt, xx, params)
        dP = J(tt, xx) @ P
        return np.concatenate([dxx, dP.reshape(-1)])

    for _ in range(n_steps):
        y0 = np.concatenate([x, Q.reshape(-1)])

        sol = solve_ivp(
            f_extended,
            (t, t + dt_orth),
            y0,
            t_eval=[t + dt_orth],
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(
                f"Lyapunov integration failed for {system.name}: {sol.message}"
            )

        y1 = np.asarray(sol.y[:, -1], dtype=float)
        x = y1[:n]
        P = y1[n:].reshape(n, k)

        # Re-orthonormalize tangent vectors
        Q, R = np.linalg.qr(P)

        # Fix sign ambiguity so logs are stable and deterministic
        diag = np.diag(R).copy()
        signs = np.sign(diag)
        signs[signs == 0.0] = 1.0
        Q = Q * signs
        diag = diag * signs

        tiny = np.finfo(float).tiny
        # sum_log_diag += np.log(np.maximum(np.abs(diag), tiny))
        log_diag = np.log(np.maximum(np.abs(diag), tiny))
        log_diag_history.append(log_diag)
        sum_log_diag += log_diag
        t += dt_orth

    exponents = sum_log_diag / (n_steps * dt_orth)
    stderr = None
    if block_size_steps is not None and block_size_steps > 1:
        logs = np.vstack(log_diag_history)  # (n_steps, k)
        B = int(block_size_steps)
        n_blocks = logs.shape[0] // B
        if n_blocks >= 2:
            logs = logs[: n_blocks * B]
            block_sums = logs.reshape(n_blocks, B, k).sum(axis=1)  # (n_blocks, k)
            block_exps = block_sums / (B * dt_orth)  # (n_blocks, k)
            # standard error of the mean
            stderr = block_exps.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    gap = None
    if k >= 2:
        gap = float(exponents[0] - exponents[1])

    details: Dict[str, Any] = {
        "t0": float(t0),
        "transient": float(transient),
        "t_max": float(t_max),
        "dt_orth": float(dt_orth),
        "n_steps": int(n_steps),
        "k": int(k),
        "method": str(method),
        "rtol": float(rtol),
        "atol": float(atol),
        "used_analytic_jacobian": system.jacobian is not None,
        "seed": seed,
    }
    return LyapunovResult(exponents=exponents, gap=gap, stderr=stderr, details=details)
