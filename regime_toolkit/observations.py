from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ObservedTrajectory:
    """
    Representation of a trajectory after an observation model is applied.

    data:
        Observed trajectory of shape (N_obs, d_obs). For a scalar channel without
        delay embedding this is returned as shape (N_obs, 1).
    time_indices:
        Indices into the original state trajectory that align with each observed row.
        This matters when delay embedding shortens the usable trajectory.
    details:
        Small metadata payload describing how the observation was produced.
    """
    data: np.ndarray
    time_indices: np.ndarray
    details: dict



def state_to_scalar_observation(
    x: np.ndarray,
    *,
    mode: str = "coordinate",
    index: int = 0,
    weights: Optional[np.ndarray] = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Map a state-space trajectory to a scalar observation.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected x shape (N, d), got {x.shape}")

    if mode == "coordinate":
        if not (0 <= index < x.shape[1]):
            raise ValueError(f"index must be in [0, {x.shape[1] - 1}], got {index}")
        y = x[:, index]
    elif mode == "linear":
        if weights is None:
            raise ValueError("weights are required when mode='linear'")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (x.shape[1],):
            raise ValueError(f"weights must have shape ({x.shape[1]},), got {weights.shape}")
        y = x @ weights
    elif mode == "norm":
        y = np.linalg.norm(x, axis=1)
    else:
        raise ValueError(f"unknown observation mode: {mode}")

    return np.asarray(y + float(offset), dtype=float)



def add_observation_noise(
    y: np.ndarray,
    sigma: float,
    *,
    relative: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add zero-mean Gaussian measurement noise to either a scalar or multivariate observation.

    If relative=True, sigma is scaled by the per-channel standard deviation.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim not in {1, 2}:
        raise ValueError(f"expected y shape (N,) or (N,d), got {y.shape}")
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")

    scale = float(sigma)
    if scale == 0.0:
        return y.copy()

    rng = np.random.default_rng(seed)
    if y.ndim == 1:
        if relative:
            signal_scale = float(np.std(y))
            scale *= signal_scale if signal_scale > 0.0 else 1.0
        return y + scale * rng.standard_normal(size=y.shape[0])

    noise_scale = np.full((y.shape[1],), scale, dtype=float)
    if relative:
        channel_scale = np.std(y, axis=0)
        channel_scale = np.where(channel_scale > 0.0, channel_scale, 1.0)
        noise_scale = scale * channel_scale
    return y + rng.standard_normal(size=y.shape) * noise_scale[None, :]



def delay_embed(
    y: np.ndarray,
    embedding_dim: int,
    delay: int,
    *,
    stride: int = 1,
    return_time_indices: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Construct a delay embedding from a scalar observation.

    Convention:
        Z[i] = [y[i], y[i + delay], ..., y[i + (embedding_dim - 1) * delay]]

    The returned anchor index corresponds to the last sample in each row.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"expected y shape (N,), got {y.shape}")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    last_offset = (embedding_dim - 1) * delay
    n_vectors = y.shape[0] - last_offset
    if n_vectors <= 0:
        raise ValueError("time series is too short for the requested embedding_dim and delay")

    starts = np.arange(0, n_vectors, stride, dtype=int)
    embedded = np.empty((starts.shape[0], embedding_dim), dtype=float)
    for j in range(embedding_dim):
        embedded[:, j] = y[starts + j * delay]

    if return_time_indices:
        anchor_indices = starts + last_offset
        return embedded, anchor_indices
    return embedded



def observed_trajectory(
    x: np.ndarray,
    *,
    representation: str = "full_state",
    scalar_mode: str = "coordinate",
    index: int = 0,
    weights: Optional[np.ndarray] = None,
    offset: float = 0.0,
    noise_sigma: float = 0.0,
    noise_relative: bool = False,
    embedding_dim: Optional[int] = None,
    delay: Optional[int] = None,
    stride: int = 1,
    seed: Optional[int] = None,
) -> ObservedTrajectory:
    """
    Build the trajectory representation consumed by trajectory-only indicators.

    Supported representations:
    - "full_state": use the state trajectory as-is (optionally with additive observation noise)
    - "scalar": use a single observed channel as a 1D trajectory (returned as Nx1)
    - "delay_scalar": use a scalar channel with delay embedding
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected x shape (N, d), got {x.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    if representation == "full_state":
        indices = np.arange(0, x.shape[0], stride, dtype=int)
        data = x[indices].copy()
        if noise_sigma > 0.0:
            data = add_observation_noise(
                data,
                noise_sigma,
                relative=noise_relative,
                seed=seed,
            )
        return ObservedTrajectory(
            data=data,
            time_indices=indices,
            details={
                "representation": representation,
                "stride": stride,
                "noise_sigma": float(noise_sigma),
                "noise_relative": bool(noise_relative),
            },
        )

    y = state_to_scalar_observation(
        x,
        mode=scalar_mode,
        index=index,
        weights=weights,
        offset=offset,
    )
    y = add_observation_noise(
        y,
        noise_sigma,
        relative=noise_relative,
        seed=seed,
    )

    if representation == "scalar":
        indices = np.arange(0, y.shape[0], stride, dtype=int)
        return ObservedTrajectory(
            data=y[indices, None],
            time_indices=indices,
            details={
                "representation": representation,
                "scalar_mode": scalar_mode,
                "index": index,
                "noise_sigma": float(noise_sigma),
                "noise_relative": bool(noise_relative),
                "stride": stride,
            },
        )

    if representation == "delay_scalar":
        if embedding_dim is None or delay is None:
            raise ValueError("embedding_dim and delay are required when representation='delay_scalar'")
        embedded, anchor_indices = delay_embed(
            y,
            embedding_dim=embedding_dim,
            delay=delay,
            stride=stride,
            return_time_indices=True,
        )
        return ObservedTrajectory(
            data=embedded,
            time_indices=anchor_indices,
            details={
                "representation": representation,
                "scalar_mode": scalar_mode,
                "index": index,
                "noise_sigma": float(noise_sigma),
                "noise_relative": bool(noise_relative),
                "embedding_dim": int(embedding_dim),
                "delay": int(delay),
                "stride": stride,
            },
        )

    raise ValueError(f"unknown representation: {representation}")
