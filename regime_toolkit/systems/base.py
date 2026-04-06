from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np

Params = Dict[str, float]


@dataclass(frozen=True)
class ODESystem:
    """
    Minimal container for an ODE system.

    Design goals:
    - Keep system definition (physics) separate from integration, indicators, and plotting.
    - Provide an optional analytic Jacobian to support tangent-space diagnostics and stiff solvers.
    """
    name: str
    dimension: int
    rhs: Callable[[float, np.ndarray, Params], np.ndarray]
    default_params: Callable[[], Params]
    default_initial_condition: Callable[[Params, np.random.Generator], np.ndarray]
    jacobian: Optional[Callable[[float, np.ndarray, Params], np.ndarray]] = None

    def validate_state(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.shape[0] != self.dimension:
            raise ValueError(f"{self.name}: expected state shape ({self.dimension},), got {x.shape}")
