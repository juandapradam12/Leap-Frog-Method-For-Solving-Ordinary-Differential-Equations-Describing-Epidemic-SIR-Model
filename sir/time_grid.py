"""Uniform time discretizations shared by all integrators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IntegrationGrid:
    """Uniform time grid for a fixed-step numerical scheme."""

    t_min: float
    t_max: float
    step: float

    def __post_init__(self) -> None:
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min")
        if self.step <= 0:
            raise ValueError("step must be positive")

    @property
    def num_points(self) -> int:
        return int(np.floor((self.t_max - self.t_min) / self.step)) + 1

    def build(self) -> NDArray[np.floating]:
        return self.t_min + self.step * np.arange(self.num_points, dtype=float)
