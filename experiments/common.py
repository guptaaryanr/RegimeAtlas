from __future__ import annotations

from regime_toolkit.robustness import (
    RobustnessCase,
    CORE_TIER,
    SUPPLEMENTAL_TIER,
    STRESS_TIER,
)

BOUNDARY_METRIC_DIRECTIONS = {
    "time_pr": "increasing",
    "occupancy_gap": "decreasing",
}
BOUNDARY_METRIC_WEIGHTS = {
    "time_pr": 1.0,
    "occupancy_gap": 1.0,
}


def build_primary_robustness_cases(
    *,
    base_dt: float,
    base_t_final: float,
    base_transient: float,
    base_method: str,
    full_state_noise_sigma: float = 0.01,
    coarse_dt_factor: float = 2.0,
    shorter_window_fraction: float = 0.70,
    shorter_window_t_final: float | None = None,
    shorter_window_transient: float | None = None,
    baseline_replicates: int = 1,
    noisy_replicates: int = 3,
    secondary_replicates: int = 2,
    solver_crosscheck_method: str | None = None,
) -> list[RobustnessCase]:
    alt_method = (
        str(solver_crosscheck_method)
        if solver_crosscheck_method is not None
        else ("BDF" if str(base_method) != "BDF" else "RK45")
    )
    return [
        RobustnessCase(
            name="baseline",
            description="full-state baseline with replicate ICs",
            representation="full_state",
            n_replicates=baseline_replicates,
            tier=CORE_TIER,
        ),
        RobustnessCase(
            name="full_state_noisy",
            description="full-state trajectory with relative observation noise",
            representation="full_state",
            observation_noise_sigma=full_state_noise_sigma,
            observation_noise_relative=True,
            n_replicates=noisy_replicates,
            tier=CORE_TIER,
        ),
        RobustnessCase(
            name="coarse_sampling",
            description="coarser temporal sampling",
            representation="full_state",
            dt=max(coarse_dt_factor * float(base_dt), float(base_dt) + 1e-12),
            n_replicates=secondary_replicates,
            tier=CORE_TIER,
        ),
        RobustnessCase(
            name="shorter_window",
            description="shorter post-transient window",
            representation="full_state",
            t_final=(
                float(shorter_window_t_final)
                if shorter_window_t_final is not None
                else max(shorter_window_fraction * float(base_t_final), 10.0)
            ),
            transient=(
                float(shorter_window_transient)
                if shorter_window_transient is not None
                else max(shorter_window_fraction * float(base_transient), 5.0)
            ),
            n_replicates=secondary_replicates,
            tier=CORE_TIER,
        ),
        RobustnessCase(
            name="solver_crosscheck",
            description=f"solver cross-check using {alt_method}",
            representation="full_state",
            method=alt_method,
            n_replicates=secondary_replicates,
            tier=CORE_TIER,
        ),
    ]


def build_supplemental_delay_cases(
    *,
    delay: int,
    embedding_dim: int = 3,
    clean_delay: int | None = None,
    clean_embedding_dim: int | None = None,
    noisy_delay: int | None = None,
    noisy_embedding_dim: int | None = None,
    noisy_sigma: float = 0.02,
    stress_sigma: float = 0.03,
    clean_replicates: int = 2,
    noisy_replicates: int = 2,
    stress_replicates: int = 2,
) -> list[RobustnessCase]:
    clean_delay = delay if clean_delay is None else int(clean_delay)
    clean_embedding_dim = (
        embedding_dim if clean_embedding_dim is None else int(clean_embedding_dim)
    )
    noisy_delay = delay if noisy_delay is None else int(noisy_delay)
    noisy_embedding_dim = (
        embedding_dim if noisy_embedding_dim is None else int(noisy_embedding_dim)
    )

    return [
        RobustnessCase(
            name="delay_scalar_clean",
            description="scalar observation with delay embedding",
            representation="delay_scalar",
            observation_index=0,
            embedding_dim=clean_embedding_dim,
            delay=clean_delay,
            n_replicates=clean_replicates,
            tier=SUPPLEMENTAL_TIER,
        ),
        RobustnessCase(
            name="delay_scalar_noisy",
            description="scalar delay embedding with relative noise",
            representation="delay_scalar",
            observation_index=0,
            observation_noise_sigma=float(noisy_sigma),
            observation_noise_relative=True,
            embedding_dim=noisy_embedding_dim,
            delay=noisy_delay,
            n_replicates=noisy_replicates,
            tier=SUPPLEMENTAL_TIER,
        ),
        RobustnessCase(
            name="full_state_noisy_stress",
            description="full-state relative-noise stress test",
            representation="full_state",
            observation_noise_sigma=float(stress_sigma),
            observation_noise_relative=True,
            n_replicates=stress_replicates,
            tier=STRESS_TIER,
        ),
    ]
