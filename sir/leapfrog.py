"""Two-step leap-frog (centered difference) integrator for autonomous ODEs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sir.model import SIRModel, SIRParameters, SIRState, SimulationResult

DerivativeFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


@dataclass(frozen=True)
class IntegrationGrid:
    """Uniform time grid for the numerical scheme."""

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


class LeapFrogIntegrator:
    """Integrate y' = f(y) with the explicit leap-frog method.

    After an Euler starter step,

    $$
    y_{n+1} = y_{n-1} + 2 h\\, f(y_n).
    $$

    The scheme is second-order accurate and symplectic for separable
    Hamiltonian systems; here it is applied as a centered two-step method
    on the first-order SIR vector field.
    """

    def __init__(self, grid: IntegrationGrid) -> None:
        self.grid = grid

    def integrate(
        self,
        derivative: DerivativeFn,
        y0: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return (time, trajectory) with shape (n,) and (n, d)."""
        h = self.grid.step
        time = self.grid.build()
        n = len(time)
        y = np.empty((n, y0.size), dtype=float)
        y[0] = np.asarray(y0, dtype=float)

        if n == 1:
            return time, y

        y[1] = y[0] + h * derivative(y[0])

        for i in range(2, n):
            y[i] = y[i - 2] + 2.0 * h * derivative(y[i - 1])

        return time, y

    def integrate_sir(
        self,
        model: SIRModel,
        initial: SIRState,
    ) -> SimulationResult:
        """Integrate an SIR model and return a structured result."""
        params = model.parameters
        initial.validate(params.population)
        time, trajectory = self.integrate(model.derivative, initial.as_array())
        return SimulationResult(
            time=time,
            susceptible=trajectory[:, 0],
            infected=trajectory[:, 1],
            recovered=trajectory[:, 2],
            parameters=params,
        )


def simulate_sir(
    parameters: SIRParameters,
    initial: SIRState,
    *,
    t_min: float = 0.0,
    t_max: float = 17.0,
    step: float = 1e-3,
) -> SimulationResult:
    """Convenience wrapper: build model + leap-frog and integrate."""
    grid = IntegrationGrid(t_min=t_min, t_max=t_max, step=step)
    integrator = LeapFrogIntegrator(grid)
    return integrator.integrate_sir(SIRModel(parameters), initial)
